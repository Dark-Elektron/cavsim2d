#!/usr/bin/env python
"""
Benchmark: Coaxial Resonator Cavity
===================================
Compares cavsim2d's NGSolve eigenmode solver against closed-form analytical
solutions for a fully-closed coaxial PEC cavity.

Geometry
--------
Inner radius  a = 50 mm,  Outer radius b = 150 mm,  Length d = 200 mm.
All walls PEC.

Modes solved
------------
- m = 0 (monopole):  TEM and TM modes  (HCurl 2-D solver)
- m = 1 … 5:         TM + TE modes  (HCurl × H1 product-space solver)

Outputs  (saved to  benchmark/coaxial_cavity/)
-------
- Convergence:  relative error vs DOF and vs wall-time for the first
  6 modes at each m, sweeping h and p.
- Mode comparison:  analytical vs numerical frequencies for 50 modes
  at each m on a single fine mesh.
- CSV with all raw data.

Usage
-----
    python benchmark/benchmark_coaxial.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import warnings

import numpy as np
import pandas as pd
from scipy.special import jv, yv, jvp, yvp
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Matplotlib – headless backend so plots are saved, never shown
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Make the in-repo cavsim2d importable
# ---------------------------------------------------------------------------
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ---------------------------------------------------------------------------
# NGSolve / Netgen / Gmsh
# ---------------------------------------------------------------------------
from ngsolve import (
    Mesh, HCurl, H1, BilinearForm, Preconditioner, GridFunction,
    Integrate, InnerProduct, Conj, Norm, TaskManager,
    IdentityMatrix, solvers, curl, grad, exp, BND,
)
from ngsolve import x as ng_x, y as ng_y, dx as ng_dx, pi as NG_PI  # type: ignore
from ngsolve.comp import VorB  # type: ignore
from netgen.occ import OCCGeometry
import gmsh

# Re-use the direct-solver selector from the project
from cavsim2d.solvers.NGSolve.eigen_ngsolve import (
    default_direct_solver,
    surface_resistance,
    get_boundary_nodes,
    suppress_c_stdout_stderr,
    SIGMA_COPPER,
)

# ═══════════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════════
c0   = 299_792_458.0            # speed of light  [m/s]
mu0  = 4.0 * np.pi * 1e-7      # permeability     [H/m]
eps0 = 8.854_187_82e-12         # permittivity     [F/m]
eta0 = np.sqrt(mu0 / eps0)     # free-space impedance  [Ω]

# ═══════════════════════════════════════════════════════════════════════════
# Cavity parameters  (Coaxial resonator)
# ═══════════════════════════════════════════════════════════════════════════
R_INNER = 0.050     # inner conductor radius  [m]
R_OUTER = 0.150     # outer conductor radius  [m]
D_CAV   = 0.200     # cavity length  [m]

# ═══════════════════════════════════════════════════════════════════════════
# Benchmark settings
# ═══════════════════════════════════════════════════════════════════════════
M_RANGE          = range(0, 6)           # azimuthal orders m = 0 … 5
N_CONV_MODES     = 6                     # modes per m for convergence
N_COMP_MODES     = 50                    # modes per m for mode comparison
P_ORDERS         = [2, 3, 4, 5, 6]      # polynomial orders
MAXH_LIST_MM     = [20, 15, 10, 7, 5, 3] # mesh sizes in mm
COMP_P           = 5                     # p-order for mode comparison
COMP_MAXH_MM     = 5                     # mesh size for mode comparison [mm]

# ═══════════════════════════════════════════════════════════════════════════
# ANALYTICAL SOLUTIONS FOR COAXIAL CAVITY
# ═══════════════════════════════════════════════════════════════════════════

def find_bessel_cross_roots(m: int, a: float, b: float, kind: str, N: int) -> list[float]:
    """Find the first *N* roots of the cross-product Bessel equation.

    For kind="TM":  J_m(k_c a) Y_m(k_c b) - J_m(k_c b) Y_m(k_c a) = 0
    For kind="TE":  J'_m(k_c a) Y'_m(k_c b) - J'_m(k_c b) Y'_m(k_c a) = 0
    """
    roots = []

    # Grid search parameters
    # The roots are spaced approximately by pi / (b - a)
    step = 0.8 * np.pi / (b - a)
    k_c = 1e-3

    def target_tm(k):
        return jv(m, k * a) * yv(m, k * b) - jv(m, k * b) * yv(m, k * a)

    def target_te(k):
        return jvp(m, k * a) * yvp(m, k * b) - jvp(m, k * b) * yvp(m, k * a)

    target = target_tm if kind == "TM" else target_te

    # Find sign changes
    val_prev = target(k_c)
    while len(roots) < N and k_c < 5000.0:
        k_next = k_c + step
        val_next = target(k_next)
        if val_prev * val_next < 0:
            # Sign change detected, bracket the root
            root = brentq(target, k_c, k_next)
            roots.append(root)
        k_c = k_next
        val_prev = val_next

    return roots


def analytical_modes(m: int, a: float, b: float, d: float, N_modes: int) -> list[dict]:
    """Return the *N_modes* lowest-frequency analytical eigenmodes for
    azimuthal order *m* in a closed coaxial cavity.
    """
    modes: list[dict] = []
    N_r, N_z = 35, 35

    # --- TEM_00p  (only for m = 0) ---------------------------------------
    if m == 0:
        # TEM modes have k_c = 0. Only exist for p >= 1 (p=0 is DC/trivial)
        for p in range(1, N_z):
            f = c0 / (2 * np.pi) * (p * np.pi / d)
            modes.append(dict(type="TEM", m=0, n=0, p=p, freq=f, k_c=0.0))

    # --- TM_mnp ---------------------------------------------------------
    # Roots of cross-product J_m(k_c a)Y_m(k_c b) - J_m(k_c b)Y_m(k_c a) = 0
    tm_roots = find_bessel_cross_roots(m, a, b, "TM", N_r)
    for n_idx, k_c in enumerate(tm_roots):
        n = n_idx + 1
        for p in range(N_z):
            k_z = p * np.pi / d
            f   = c0 / (2 * np.pi) * np.sqrt(k_c**2 + k_z**2)
            modes.append(dict(type="TM", m=m, n=n, p=p, freq=f, k_c=k_c))

    # --- TE_mnp  (for m >= 1; m=0 TE is also present in coaxial but
    # our 2-D monopole solver cannot see m=0 TE because it is purely azimuthal)
    te_roots = find_bessel_cross_roots(m, a, b, "TE", N_r)
    for n_idx, k_c in enumerate(te_roots):
        n = n_idx + 1
        # p = 0 is trivial/zero-frequency for TE modes in coaxial
        for p in range(1, N_z + 1):
            k_z = p * np.pi / d
            f   = c0 / (2 * np.pi) * np.sqrt(k_c**2 + k_z**2)
            modes.append(dict(type="TE", m=m, n=n, p=p, freq=f, k_c=k_c))

    modes.sort(key=lambda x: x["freq"])
    return modes[:N_modes]

# ═══════════════════════════════════════════════════════════════════════════
# GEOMETRY  &  MESH
# ═══════════════════════════════════════════════════════════════════════════

def write_coaxial_geo(filepath: str, a: float, b: float, d: float) -> str:
    """Write a Gmsh .geo file for a closed coaxial cavity.

    Coordinates: x = z (beam axis, 0 … d), y = r (radial, a … b).
    Boundary labels:
        PEC — left wall, top wall, right wall, inner conductor (bottom wall)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write('SetFactory("OpenCASCADE");\n')
        f.write(f"Point(1) = {{0, {a:.16e}, 0}};\n")
        f.write(f"Point(2) = {{0, {b:.16e}, 0}};\n")
        f.write(f"Point(3) = {{{d:.16e}, {b:.16e}, 0}};\n")
        f.write(f"Point(4) = {{{d:.16e}, {a:.16e}, 0}};\n")
        f.write("Line(1) = {1, 2};\n")   # left wall
        f.write("Line(2) = {2, 3};\n")   # outer wall
        f.write("Line(3) = {3, 4};\n")   # right wall
        f.write("Line(4) = {4, 1};\n")   # inner wall
        f.write('\nPhysical Line("PEC") = {1, 2, 3, 4};\n')
        f.write("\nCurve Loop(1) = {1, 2, 3, 4};\n")
        f.write("Plane Surface(1) = {1};\n")
        f.write("Reverse Surface 1;\n")
        f.write('Physical Surface("Domain") = {1};\n')
    return filepath


def make_mesh(geo_path: str, maxh: float):
    """Generate an NGSolve Mesh from a .geo file."""
    out_dir = os.path.dirname(geo_path)
    step_path = os.path.join(out_dir, "mesh.step")

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.open(geo_path)
    gmsh.model.mesh.generate(2)
    with suppress_c_stdout_stderr():
        gmsh.write(step_path)

    bcs: dict[int, str] = {}
    for dim, phys_tag in gmsh.model.getPhysicalGroups():
        if dim != 1:
            continue
        name = gmsh.model.getPhysicalName(dim, phys_tag)
        for line_id in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
            if isinstance(line_id, tuple):
                _, line_id = line_id
            bcs[line_id] = name
    gmsh.finalize()

    geo = OCCGeometry(step_path, dim=2)
    ngmesh = geo.GenerateMesh(maxh=maxh)
    for key, bc in bcs.items():
        ngmesh.SetBCName(key - 1, bc)

    mesh = Mesh(ngmesh)
    n_el = mesh.GetNE(VorB.VOL)
    return mesh, n_el


# ═══════════════════════════════════════════════════════════════════════════
# EIGENSOLVERS
# ═══════════════════════════════════════════════════════════════════════════

def solve_monopole(mesh, p_order: int, n_modes: int):
    """Solve the monopole (m = 0) Maxwell eigenvalue problem."""
    t0 = time.perf_counter()
    mesh.Curve(p_order)
    fes = HCurl(mesh, order=p_order, dirichlet="PEC")
    ndof = fes.ndof
    u, v = fes.TnT()

    direct = 'sparsecholesky'

    a    = BilinearForm(ng_y * curl(u) * curl(v) * ng_dx)
    m_bf = BilinearForm(ng_y * u * v * ng_dx)
    apre = BilinearForm(ng_y * curl(u) * curl(v) * ng_dx + ng_y * u * v * ng_dx)
    pre  = Preconditioner(apre, "direct", inverse=direct)

    with TaskManager():
        a.Assemble();  m_bf.Assemble();  apre.Assemble()

        gradmat, fesh1 = fes.CreateGradient()
        gradmattrans   = gradmat.CreateTranspose()
        math1          = gradmattrans @ m_bf.mat @ gradmat
        math1[0, 0]   += 1
        invh1 = math1.Inverse(inverse=direct, freedofs=fesh1.FreeDofs())

        proj    = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m_bf.mat
        projpre = proj @ pre.mat

        evals, evecs = solvers.PINVIT(
            a.mat, m_bf.mat, pre=projpre,
            num=n_modes + 4, maxit=30, printrates=False,
        )

    freqs, gfu_E, gfu_H = [], [], []
    for i, lam in enumerate(evals):
        f = c0 * np.sqrt(abs(lam)) / (2 * np.pi) * 1e-6   # Hz → MHz
        w = 2 * np.pi * f * 1e6
        gfu = GridFunction(fes)
        gfu.vec.data = evecs[i]
        freqs.append(f)
        gfu_E.append(gfu)
        gfu_H.append(1j / (mu0 * w) * curl(gfu) if w > 0 else None)

    elapsed = time.perf_counter() - t0
    return freqs, gfu_E, gfu_H, ndof, elapsed


def solve_mpole(mesh, p_order: int, m: int, n_modes: int):
    """Solve the m-pole (m ≥ 1) Maxwell eigenvalue problem."""
    t0 = time.perf_counter()
    mesh.Curve(p_order)
    r = ng_y

    fes_rz = HCurl(mesh, order=p_order, dirichlet="PEC")
    Grz, fes_phi = fes_rz.CreateGradient()
    fes = fes_rz * fes_phi
    ndof = fes.ndof
    (u, u_phi), (v, v_phi) = fes.TnT()

    with TaskManager():
        a = BilinearForm(
            (r * curl(u) * curl(v)
             + 1 / r * (m**2 * u * v
                        + m * u * grad(v_phi)
                        + m * grad(u_phi) * v
                        + grad(u_phi) * grad(v_phi))) * ng_dx
        ).Assemble()

        b = BilinearForm(
            (r * u * v + 1 / r * u_phi * v_phi) * ng_dx
        ).Assemble()

        sum_mat = (a.mat + b.mat).CreateSparseMatrix()
        pre = sum_mat.Inverse(fes.FreeDofs())

        embU, embPhi = fes.embeddings
        G  = (embU @ Grz - m * embPhi).CreateSparseMatrix()
        GT = G.CreateTranspose()
        math1  = GT @ b.mat @ G
        invh1  = math1.Inverse(inverse='sparsecholesky', freedofs=fes_phi.FreeDofs())

        proj    = IdentityMatrix(fes.ndof) - G @ invh1 @ GT @ b.mat
        projpre = proj @ pre

        evals_, evecs_ = solvers.PINVIT(
            a.mat, b.mat, pre=projpre,
            num=n_modes + 6, maxit=20, printrates=False,
        )

        mask = np.array(evals_) > 1
        evals = np.array(evals_)[mask]
        evecs = np.array(evecs_)[mask]

    freqs, gfu_E, gfu_H = [], [], []
    for i in range(len(evals)):
        f = c0 * np.sqrt(abs(evals[i])) / (2 * np.pi) * 1e-6
        w = 2 * np.pi * f * 1e6
        gfu = GridFunction(fes)
        gfu.vec.data = evecs[i]
        freqs.append(f)
        gfu_E.append(gfu)

        u_gf, uphi_gf = gfu.components
        H_inplane = 1 / (mu0 * w * r) * (m * u_gf + grad(uphi_gf))
        H_phi     = 1 / (mu0 * w) * curl(u_gf)
        gfu_H.append((H_inplane, H_phi))

    elapsed = time.perf_counter() - t0
    return freqs, gfu_E, gfu_H, ndof, elapsed

# ═══════════════════════════════════════════════════════════════════════════
# MATCHING
# ═══════════════════════════════════════════════════════════════════════════

def match_modes(num_freqs, ana_modes):
    ana_freqs = np.array([m["freq"] * 1e-6 for m in ana_modes])  # Hz → MHz
    matches = []
    used = set()
    for ni, nf in enumerate(num_freqs):
        diffs = np.abs(ana_freqs - nf)
        for ai in np.argsort(diffs):
            if ai not in used:
                matches.append((ni, int(ai), ana_modes[ai]))
                used.add(ai)
                break
    return matches

# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

MODE_COLORS = [
    "#2ca02c", "#d62728", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
]
P_MARKERS = {2: "o", 3: "s", 4: "^", 5: "D", 6: "v", 7: "P"}


def _setup_ax(ax, xlabel, ylabel, title, logx=True, logy=True):
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.7)
    ax.tick_params(labelsize=10)


def plot_convergence(df, m, quantity, vs, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    x_col = "ndof" if vs == "ndof" else "time"
    y_col = f"rel_err_{quantity}"
    x_lab = "Number of DOFs" if vs == "ndof" else "Time required — $t$ / s"
    y_lab = {
        "freq": r"Relative frequency deviation — $\sigma_\nu$",
    }.get(quantity, f"Relative {quantity} error")

    for mode_n in sorted(df["mode_n"].unique()):
        sub = df[df["mode_n"] == mode_n]
        color = MODE_COLORS[mode_n % len(MODE_COLORS)]
        for p_order in sorted(sub["p"].unique()):
            ss = sub[sub["p"] == p_order].sort_values(x_col)
            valid = ss[y_col] > 0
            if valid.sum() == 0:
                continue
            marker = P_MARKERS.get(p_order, "o")
            label = f"Mode {mode_n+1}" if p_order == sorted(sub["p"].unique())[0] else None
            ax.plot(
                ss.loc[valid, x_col], ss.loc[valid, y_col],
                color=color, marker=marker, ms=5, lw=1.2,
                label=label,
            )

    for p_order in sorted(df["p"].unique()):
        marker = P_MARKERS.get(p_order, "o")
        ax.plot([], [], "k", marker=marker, lw=0, ms=5, label=f"$p = {p_order}$")

    pol_name = {0: "Monopole", 1: "Dipole", 2: "Quadrupole",
                3: "Sextupole", 4: "Octupole", 5: "Decapole"}.get(m, f"m={m}")
    title = f"Coaxial Resonator — {pol_name} (m={m})"
    _setup_ax(ax, x_lab, y_lab, title)
    ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)
    fig.tight_layout()

    tag = "dof" if vs == "ndof" else "time"
    fname = f"convergence_{quantity}_vs_{tag}_m{m}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {fname}")


def plot_mode_comparison(ana_freqs_ghz, num_freqs_ghz, m, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    n = np.arange(1, len(ana_freqs_ghz) + 1)

    ax.plot(n, ana_freqs_ghz, "o", mfc="none", mec="#2ca02c", ms=7,
            label="Analytic", zorder=3)
    n_num = np.arange(1, len(num_freqs_ghz) + 1)
    ax.plot(n_num, num_freqs_ghz, "x", color="#d62728", ms=6,
            label="cavsim2d", zorder=4)

    pol_name = {0: "Monopole (m=0)", 1: "Dipole (m=1)", 2: "Quadrupole (m=2)",
                3: "Sextupole (m=3)", 4: "Octupole (m=4)",
                5: "Decapole (m=5)"}.get(m, f"m={m}")
    _setup_ax(ax, r"Mode number — $n$", r"Frequency — $\nu$ / GHz",
              f"Coaxial Resonator — First {len(ana_freqs_ghz)} Modes ({pol_name})",
              logx=False, logy=False)
    ax.legend(fontsize=11, loc="upper left")
    fig.tight_layout()

    fname = f"freq_vs_mode_m{m}.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {fname}")


def plot_mode_comparison_all(all_data, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    pol_labels = {0: "m=0", 1: "m=1", 2: "m=2", 3: "m=3", 4: "m=4", 5: "m=5"}
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (m_val, data) in enumerate(sorted(all_data.items())):
        c = colors[i % len(colors)]
        n = np.arange(1, len(data["ana"]) + 1)
        ax.plot(n, data["ana"], "o", mfc="none", mec=c, ms=5)
        nn = np.arange(1, len(data["num"]) + 1)
        ax.plot(nn, data["num"], "x", color=c, ms=4,
                label=pol_labels.get(m_val, f"m={m_val}"))

    _setup_ax(ax, r"Mode number — $n$", r"Frequency — $\nu$ / GHz",
              "Coaxial Resonator — All Polarisations",
              logx=False, logy=False)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="k", mfc="none", lw=0, ms=7, label="Analytic"),
        Line2D([0], [0], marker="x", color="k", lw=0, ms=6, label="cavsim2d"),
    ]
    for i, m_val in enumerate(sorted(all_data)):
        c = colors[i % len(colors)]
        handles.append(Line2D([0], [0], marker="s", color=c, lw=0, ms=6,
                              label=pol_labels.get(m_val, f"m={m_val}")))
    ax.legend(handles=handles, fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()

    fname = "freq_vs_mode_all.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN  BENCHMARK  ROUTINES
# ═══════════════════════════════════════════════════════════════════════════

def run_convergence_for_m(m, a_rad, b_rad, d, geo_path, output_dir):
    print(f"\n{'='*60}")
    print(f"  Convergence study  m = {m}")
    print(f"{'='*60}")

    ana_modes = analytical_modes(m, a_rad, b_rad, d, N_CONV_MODES * 3)  # over-generate
    ana_modes = ana_modes[:N_CONV_MODES]

    if m == 0:
        p_orders = P_ORDERS
        maxh_list = MAXH_LIST_MM
    else:
        p_orders = [2, 3, 4, 5]
        maxh_list = [20, 15, 10, 7]

    rows = []

    for p_order in p_orders:
        for maxh_mm in maxh_list:
            maxh = maxh_mm * 1e-3
            print(f"  m={m}  p={p_order}  maxh={maxh_mm} mm ... ", end="", flush=True)

            mesh, n_el = make_mesh(geo_path, maxh)

            if m == 0:
                freqs, gfu_E, gfu_H, ndof, elapsed = solve_monopole(
                    mesh, p_order, N_CONV_MODES)
            else:
                freqs, gfu_E, gfu_H, ndof, elapsed = solve_mpole(
                    mesh, p_order, m, N_CONV_MODES)

            n_avail = min(len(freqs), N_CONV_MODES)
            num_freqs = freqs[:n_avail]

            matches = match_modes(num_freqs, ana_modes)

            for ni, ai, amode in matches:
                if ni >= n_avail:
                    continue
                f_num = num_freqs[ni]
                f_ana = amode["freq"] * 1e-6  # MHz

                rel_err_f = abs(f_num - f_ana) / abs(f_ana) if f_ana != 0 else 0

                row = dict(
                    m=m, p=p_order, maxh_mm=maxh_mm, n_elements=n_el,
                    ndof=ndof, time=elapsed,
                    mode_n=ai,
                    mode_type=amode["type"], mode_nn=amode["n"], mode_p=amode["p"],
                    freq_num=f_num, freq_ana=f_ana,
                    rel_err_freq=rel_err_f,
                )
                rows.append(row)

            print(f"ndof={ndof:>6d}  t={elapsed:.2f}s  "
                  f"df1={rows[-n_avail]['rel_err_freq']:.2e}" if rows else "")

    df = pd.DataFrame(rows)
    return df


def run_mode_comparison_for_m(m, a_rad, b_rad, d, geo_path, output_dir):
    print(f"\n{'='*60}")
    print(f"  Mode comparison  m = {m}  ({N_COMP_MODES} modes)")
    print(f"{'='*60}")

    ana_modes = analytical_modes(m, a_rad, b_rad, d, N_COMP_MODES + 20)
    ana_modes = ana_modes[:N_COMP_MODES]

    maxh = COMP_MAXH_MM * 1e-3
    mesh, n_el = make_mesh(geo_path, maxh)

    if m == 0:
        freqs, gfu_E, gfu_H, ndof, elapsed = solve_monopole(
            mesh, COMP_P, N_COMP_MODES)
    else:
        freqs, gfu_E, gfu_H, ndof, elapsed = solve_mpole(
            mesh, COMP_P, m, N_COMP_MODES)

    n_avail = min(len(freqs), N_COMP_MODES)

    # Match modes
    matches = match_modes(freqs[:n_avail], ana_modes)
    matched_ana = np.array([ana_modes[ai]["freq"] * 1e-9 for _, ai, _ in matches])
    matched_num = np.array([freqs[ni] * 1e-3 for ni, _, _ in matches])

    plot_mode_comparison(matched_ana, matched_num, m, output_dir)

    return dict(ana=matched_ana.tolist(), num=matched_num.tolist())


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  BENCHMARK: Coaxial Resonator Cavity")
    print(f"  a = {R_INNER*1e3:.0f} mm   b = {R_OUTER*1e3:.0f} mm   d = {D_CAV*1e3:.0f} mm")
    print("=" * 70)

    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "coaxial_cavity")
    os.makedirs(output_dir, exist_ok=True)

    # Write geometry
    geo_path = write_coaxial_geo(
        os.path.join(output_dir, "_work", "coaxial.geo"), R_INNER, R_OUTER, D_CAV)

    # ── Phase A: h-p convergence ──────────────────────────────────────────
    all_conv_dfs = []
    for m in M_RANGE:
        df_m = run_convergence_for_m(m, R_INNER, R_OUTER, D_CAV, geo_path, output_dir)
        all_conv_dfs.append(df_m)

        plot_convergence(df_m, m, "freq", "ndof", output_dir)
        plot_convergence(df_m, m, "freq", "time", output_dir)

    df_all = pd.concat(all_conv_dfs, ignore_index=True)
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n  Convergence data saved to: {csv_path}")

    # ── Phase B: mode comparison ──────────────────────────────────────────
    all_comp = {}
    for m in M_RANGE:
        comp = run_mode_comparison_for_m(m, R_INNER, R_OUTER, D_CAV, geo_path, output_dir)
        all_comp[m] = comp

    plot_mode_comparison_all(all_comp, output_dir)

    comp_rows = []
    for m_val, data in all_comp.items():
        for i, (af, nf) in enumerate(zip(data["ana"], data["num"])):
            comp_rows.append(dict(m=m_val, mode_index=i+1,
                                  freq_ana_ghz=af, freq_num_ghz=nf,
                                  rel_err=abs(nf - af) / abs(af) if af else 0))
    df_comp = pd.DataFrame(comp_rows)
    comp_csv = os.path.join(output_dir, "mode_comparison_results.csv")
    df_comp.to_csv(comp_csv, index=False)
    print(f"  Mode comparison data saved to: {comp_csv}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print(f"  All results in: {output_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
