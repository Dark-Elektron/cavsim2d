import json
import functools
import gc
import os.path
import pickle
import sys
import time
import warnings

from matplotlib import tri
from cavsim2d.utils.shared_functions import *
from ngsolve import *
from ngsolve import (x, y, dx, pi, Mesh, exp, BND, # type: ignore
                     GridFunction, BilinearForm, InnerProduct, curl, grad, Conj, # type: ignore
                     Integrate, TaskManager, HCurl, H1, Preconditioner, solvers, Norm, IdentityMatrix) # type: ignore
from ngsolve.la import Embedding # type: ignore
from ngsolve.webgui import Draw
from ngsolve.comp import VorB # type: ignore
from netgen.occ import *
from netgen.occ import OCCGeometry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from cavsim2d.utils.printing import *
from cavsim2d.solvers.eigenmode_result import pol_name, pol_number
import gmsh
import platform

mu0 = 4 * pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458
SIGMA_COPPER = 5.96e7  # electrical conductivity of copper [S/m]
DEFAULT_N_MODES = 10
# Radius [m] below which a point counts as "on the axis". The 1/r weights in the
# azimuthal field are singular there but the fields are defined by their limit,
# so the weight is clamped rather than evaluated (the axis is measure-zero in
# every integral that uses it).
AXIS_EPS = 1e-9


def mesh_h_metres(mesh_config, default=20):
    """``mesh_config['h']`` (mm) -> metres, trapping the classic units slip.

    ``h`` is in **millimetres**. Passing metres — e.g. ``h=25e-3`` intending
    "25 mm" — asks for 25-*micron* elements: the mesher then grinds on millions
    of elements and the run looks hung. Warn loudly instead of silently melting
    the machine.
    """
    h = (mesh_config or {}).get('h', default)
    if h < 0.5:
        # warnings.warn (not the verbosity-gated warning()): a silent units
        # slip just looks like a hung run.
        warnings.warn(
            f"mesh_config['h'] is in MILLIMETRES: h={h:g} requests {h:g} mm "
            f"({h * 1e-3:.1e} m) elements — meshing will be extremely slow. "
            f"If you meant {h * 1e3:g} mm, pass h={h * 1e3:g}.",
            UserWarning, stacklevel=3)
    return h * 1e-3


def surface_resistance(w, conductivity=SIGMA_COPPER, rs=None):
    """Surface resistance [Ohm] at angular frequency *w*.

    Normal conductor: Rs = sqrt(mu0 * w / (2 * conductivity)). For a
    superconductor (or any fixed-Rs material) pass ``rs`` to use that value
    directly instead."""
    if rs is not None:
        return rs
    return np.sqrt(mu0 * w / (2 * conductivity))


@functools.lru_cache(maxsize=None)
def direct_solver_available(name):
    """Whether this NGSolve build can actually factorise with backend *name*.

    Which sparse direct solvers are compiled in varies by platform and by how
    NGSolve was built (pip wheel, conda, source), so this probes a 2x2 problem
    rather than guessing. Cached: the probe runs at most once per backend.
    """
    try:
        face = WorkPlane().Rectangle(1, 1).Face()
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=1.0))
        fes = H1(mesh, order=1)
        u, v = fes.TnT()
        a = BilinearForm(grad(u) * grad(v) * dx + u * v * dx).Assemble()
        a.mat.Inverse(fes.FreeDofs(), inverse=name)
        return True
    except Exception:
        return False


def default_direct_solver():
    """Name of the sparse direct-solver backend for the monopole eigenproblem.

    Preference order by platform, filtered by what the build actually provides:

    - Windows: ``pardiso`` (Intel MKL, shipped with the Windows NGSolve build).
    - macOS / Linux: ``umfpack`` (SuiteSparse) first — it is the fast option on
      those platforms, where PARDISO usually is not compiled in.

    Falls back to ``sparsecholesky``, which is always built in. Override per run
    with ``eigenmode_config['direct_solver']``.
    """
    if platform.system() == 'Windows':
        preferred = ('pardiso', 'umfpack')
    else:
        preferred = ('umfpack', 'pardiso')
    for name in preferred:
        if direct_solver_available(name):
            return name
    return 'sparsecholesky'


def parse_polarisations(value):
    """Normalize an eigenmode_config 'polarisation' entry to a sorted list of
    azimuthal mode numbers. Accepts an int, a name ('dipole'), or a list of
    either; None defaults to the monopole."""
    if value is None:
        return [0]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return sorted({pol_number(v) for v in value})


class NGSolveMEVP:
    """NGSolve-based Maxwell Eigenvalue Problem solver for axisymmetric RF cavities."""

    def __init__(self):
        self.geo = None
        self.step = None
        self.mesh = None
        self.fields = None
        self._last_adaptive_history = None

    @staticmethod
    def requested_n_modes(cav=None, eigenmode_config=None, n_modes=None):
        """Number of user-requested eigenmodes.

        Explicit ``n_modes``/``nmodes`` in the config wins. If no explicit
        value is supplied and *cav* is a cavity-like object, default to
        ``cav.n_cells + 2``; otherwise use the general default of 10.
        """
        if n_modes is None and eigenmode_config:
            n_modes = eigenmode_config.get('n_modes', eigenmode_config.get('nmodes', None))

        if n_modes is None and hasattr(cav, 'n_cells'):
            n_modes = int(cav.n_cells) + 2

        if n_modes is None:
            n_modes = DEFAULT_N_MODES

        if isinstance(n_modes, bool) or int(n_modes) <= 0:
            raise ValueError("'n_modes' must be a positive integer.")

        return int(n_modes)

    @staticmethod
    def pinvit_n_modes(requested_n_modes):
        """PINVIT search size: always two more than requested."""
        return int(requested_n_modes) + 2

    @staticmethod
    def modes_of_interest(cav, m, eigenmode_config=None, n_modes=None):
        """0-based indices of the modes whose QOIs are reported for polarisation *m*.

        ``eigenmode_config['mode_of_interest']`` is **1-based** — mode 1 is the
        lowest of the passband. A polarisation may have **any number** of modes of
        interest, so each entry is an int or a list of ints of any length. The value
        may be a single entry applied to every polarisation, or a dict keyed by
        polarisation (name or azimuthal number)::

            'mode_of_interest': 9                                    # every polarisation
            'mode_of_interest': [1, 2, 3]                            # three, for every one
            'mode_of_interest': {'monopole': [1, 2, 3, 4], 'dipole': 1}

        Order is preserved and duplicates collapse.

        The **first** mode listed is the primary one: its QOIs become
        ``qois.json``. Every mode of interest is also written to ``qois_moi.json``,
        keyed by its 1-based index.

        Defaults, when the key is absent or a polarisation is missing from the
        dict:

        - monopole (m=0): ``n_cells`` — the pi-mode, the operating mode of an
          accelerating structure. For a 1-cell cavity, gun or pillbox this is
          mode 1, so the convention degrades gracefully to non-accelerator
          geometries.
        - m-pole (m>=1): mode 1, the lowest of the deflecting passband.
        """
        n_cells = int(getattr(cav, 'n_cells', 1) or 1)
        default = n_cells if int(m) == 0 else 1

        requested = (eigenmode_config or {}).get('mode_of_interest', None)
        if isinstance(requested, dict):
            requested = next((requested[k] for k in (pol_name(m), int(m), str(m))
                              if k in requested), None)
        if requested is None:
            requested = default
        if not isinstance(requested, (list, tuple)):
            requested = [requested]
        if len(requested) == 0:
            raise ValueError(
                f"'mode_of_interest' for polarisation {pol_name(m)!r} is empty; give at "
                f"least one 1-based mode index.")

        out = []
        for mode in requested:
            if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)):
                raise ValueError(
                    f"'mode_of_interest' must be a positive integer (1-based), a list of "
                    f"them, or a dict of either; got {mode!r} for polarisation "
                    f"{pol_name(m)!r}.")
            mode = int(mode)
            if mode < 1:
                raise ValueError(
                    f"'mode_of_interest' is 1-based: mode 1 is the lowest of the passband. "
                    f"Got {mode} for polarisation {pol_name(m)!r}.")
            if n_modes is not None and mode > n_modes:
                raise ValueError(
                    f"'mode_of_interest'={mode} for polarisation {pol_name(m)!r} exceeds the "
                    f"{n_modes} modes solved for. Raise eigenmode_config['n_modes'] to at "
                    f"least {mode}.")
            if mode - 1 not in out:
                out.append(mode - 1)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Geometry
    # ──────────────────────────────────────────────────────────────────────

    def write_geometry_multicell(self, folder, n_cells, multicell,
                                 beampipe='none', plot=False, cell_parameterisation='normal'):
        """Write a multicell cavity geometry file to *folder*."""
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except FileNotFoundError:
                error("Could not create multicell simulation directory. Check folder path.")
                exit()

        file_path = os.path.join(folder, 'geodata.n')
        if cell_parameterisation == 'normal':
            write_cavity_geometry_cli_multicell(n_cells, multicell, beampipe, write=file_path, plot=plot)
        else:
            write_cavity_geometry_cli_flattop(multicell, 'both', n_cell=n_cells, write=file_path)

    def load_geo(self, filepath, output_filepath=None, maxh=1):
        """Load a .geo file via Gmsh, mesh it, and return an NGSolve-ready mesh."""
        if output_filepath is None:
            output_filepath = os.path.dirname(filepath)

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.open(filepath)
        gmsh.model.mesh.generate(2)
        with suppress_c_stdout_stderr():
            gmsh.write(os.path.join(output_filepath, "mesh.step"))

        self.step_geo = OCCGeometry(os.path.join(output_filepath, "mesh.step"), dim=2)
        self.ngmesh = self.step_geo.GenerateMesh(maxh=maxh)
        self.bcs = self._get_boundaries_from_gmsh()
        gmsh.finalize()

        return self.step_geo, self.ngmesh, self.bcs

    def _build_mesh(self, cav, maxh, order):
        """Return a boundary-tagged, curved NGSolve mesh for *cav*.

        Two backends behind one call:
        - Native: if the cavity exposes a unified ``profile()`` (a geometry
          :class:`~cavsim2d.geometry.Profile`), mesh it directly with
          netgen.occ — exact edges, no gmsh, no ``.geo`` round-trip.
        - Import: otherwise mesh the cavity's ``.geo`` file via gmsh
          (elliptical, spline, imported CAD).
        """
        maker = getattr(cav, 'profile', None)
        profile = maker() if callable(maker) else None
        if profile is not None:
            return profile.mesh(maxh=maxh, order=order)

        if not cav.geo_filepath:
            raise RuntimeError(
                f"{type(cav).__name__} {cav.name!r} has no .geo file and its profile() "
                "returned None, so there is no geometry to mesh. A cavity with "
                "independently varying cells (set_half_cells) is native-only: the gmsh "
                "writer can express uniform mid-cells only, so falling back to it would "
                "silently solve a different cavity.")

        step_geo, ngmesh, bcs = self.load_geo(cav.geo_filepath, maxh=maxh)
        for key, bc in bcs.items():
            ngmesh.SetBCName(key - 1, bc)
        mesh = Mesh(ngmesh)
        mesh.Curve(order)
        return mesh

    def _get_boundaries_from_gmsh(self):
        """Extract boundary condition name map from the current Gmsh model."""
        line_bc_map = {}
        for dim, phys_tag in gmsh.model.getPhysicalGroups():
            if dim != 1:
                continue
            name = gmsh.model.getPhysicalName(dim, phys_tag)
            for line_id in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
                if isinstance(line_id, tuple):
                    _, line_id = line_id
                line_bc_map[line_id] = name
        return line_bc_map

    # ──────────────────────────────────────────────────────────────────────
    # Solve
    # ──────────────────────────────────────────────────────────────────────

    def solve(self, cav, eigenmode_config=None):
        """Run eigenmode analysis on a single cavity object.

        ``eigenmode_config['polarisation']`` selects the azimuthal mode
        number(s) to solve for: an int, a name ('monopole', 'dipole',
        'quadrupole', 'sextupole', ...), or a list of either. The monopole
        (m=0, default) keeps its flat ``eigenmode/`` layout; every m >= 1
        is solved with the HCurl x H1 product-space formulation and saved
        to ``eigenmode/<pol name>/``.
        """
        eigenmode_folder_structure = {
            'eigenmode': None
        }
        make_dirs_from_dict(eigenmode_folder_structure, cav.self_dir)

        # mesh_config['h'] is in mm, the solver works in metres. The default has to be
        # converted too: leaving it at 20 means a 20 m maxh, i.e. no size constraint at
        # all, so each meshing backend silently picks its own element size.
        mesh_config = (eigenmode_config or {}).get('mesh_config', {})
        mesh_h = mesh_h_metres(mesh_config)
        mesh_p = mesh_config.get('p', 3)
        # Opt-in adaptive (error-driven) h-refinement — applied to EVERY
        # requested polarisation (the recovery-error estimator is m-agnostic).
        adaptive = self._parse_adaptive(mesh_config)

        pols = parse_polarisations((eigenmode_config or {}).get('polarisation', 0))

        # One solve path for every polarisation, each on its OWN mesh: with
        # adaptive on, m >= 1 refines to resolve its own deflecting field just
        # as the monopole does, so the meshes (and per-mode convergence) are
        # independent per polarisation.
        for m in sorted(pols):
            self._solve_pol(cav, m, mesh_h, mesh_p, eigenmode_config, adaptive=adaptive)
        return True

    def _solve_pol(self, cav, m, mesh_h, mesh_p, eigenmode_config=None, adaptive=None):
        """Build a mesh for *cav* and solve azimuthal order *m* on it, writing
        all results to ``<cavity>/eigenmode/<pol name>/``.

        ONE path for every polarisation: the eigenproblem (:meth:`_build_system`),
        the recovery-error estimator (:meth:`_error_fields`) and the QOIs
        (:meth:`evaluate_qois`) are all m-agnostic, so this — including the
        adaptive h-refinement — runs identically for the monopole and every
        m-pole. Each polarisation gets its OWN mesh (so its refinement is driven
        by its own field error), built native (``profile()``) or from the
        ``.geo`` file.
        """
        pol_dir = os.path.join(cav.self_dir, 'eigenmode', pol_name(m))
        os.makedirs(pol_dir, exist_ok=True)

        n_modes = self.requested_n_modes(cav, eigenmode_config)
        conductivity = (eigenmode_config or {}).get('conductivity', SIGMA_COPPER)
        rs_ohm = (eigenmode_config or {}).get('surface_resistance', None)

        # Active-length normalisation. Elliptical cavities store the half-cell
        # length as 'L_m'; otherwise take an explicit 'normalization_length'
        # (monopole) and finally let evaluate_qois fall back to the on-axis
        # field extent (L=None).
        L_norm = cav.parameters.get('L_m', None)
        if L_norm is None:
            L_norm = (eigenmode_config or {}).get('normalization_length', None) if m == 0 else 1

        # Solve on this polarisation's own mesh; adaptive refines it in place to
        # resolve *this* polarisation's modes (adaptive=None -> single solve).
        mesh = self._build_mesh(cav, mesh_h, mesh_p)
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(cav, pol_dir, mesh, mesh_p,
                                                          n_modes, m=m, adaptive=adaptive)
        # Save after solving: adaptive refinement mutates *mesh* in place, so
        # this persists the finest mesh actually used for the QOIs.
        self.save_mesh(pol_dir, mesh)

        # Modes of interest: the pi-mode for the monopole passband, mode 1 (the
        # lowest of the deflecting passband) for m >= 1 — see
        # :meth:`modes_of_interest`. The first listed is primary (-> qois.json).
        moi = self.modes_of_interest(cav, m, eigenmode_config, len(freq_fes))
        if m == 0:
            n_dofs = HCurl(mesh, order=mesh_p, dirichlet="PEC").ndof
        else:
            fes_rz = HCurl(mesh, order=mesh_p, dirichlet="PEC")
            _, fes_phi = fes_rz.CreateGradient()
            n_dofs = fes_rz.ndof + fes_phi.ndof

        qois_moi = {}
        for idx in moi:
            q = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=idx,
                                   n_cells=cav.n_cells, L=L_norm, save_dir=pol_dir,
                                   conductivity=conductivity, surface_resistance_ohm=rs_ohm,
                                   write_axis=(idx == moi[0]))
            # String metadata (UQ coerces the qois table to numerics and drops
            # text columns; a bare int would be averaged into nonsense stats).
            q['mode_of_interest'] = str(idx + 1)
            q["No of DOFs"] = n_dofs
            qois_moi[str(idx + 1)] = q

        with open(os.path.join(pol_dir, 'qois_moi.json'), "w") as f:
            json.dump(qois_moi, f, indent=4, separators=(',', ': '))
        # qois.json is the primary (first) mode of interest.
        with open(os.path.join(pol_dir, 'qois.json'), "w") as f:
            json.dump(qois_moi[str(moi[0] + 1)], f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=ii,
                                                    n_cells=cav.n_cells, L=L_norm, save_dir=pol_dir,
                                                    conductivity=conductivity, surface_resistance_ohm=rs_ohm)

        with open(os.path.join(pol_dir, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    def cavity_multicell(self, no_of_cells=1, no_of_modules=1, multicell=None,
                         fid=None, bc=33, pol='monopole', f_shift='default', beta=1,
                         n_modes=None, beampipes='None', sim_folder='NGSolveMEVP',
                         parentDir=None, projectDir=None, subdir='',
                         expansion=None, expansion_r=None, mesh_args=None, opt=False,
                         deformation_params=None, eigenmode_config=None):
        """Write geometry and run eigenmode analysis for a multicell cavity."""
        if mesh_args is None:
            mesh_args = [20, 20]

        # Unified eigenmode folder — flat structure, no Cavities/ subfolder
        if opt:
            run_save_directory = projectDir / f'{fid}/eigenmode'
        elif subdir == '':
            run_save_directory = projectDir / f'{fid}/eigenmode'
        else:
            run_save_directory = projectDir / f'{subdir}/eigenmode/{fid}'

        self.write_geometry_multicell(run_save_directory, no_of_cells, multicell,
                                      beampipes, plot=False)

        if not os.path.exists(os.path.join(run_save_directory, 'geodata.geo')):
            error('Could not run eigenmode analysis due to error in geometry.')
            return False

        mesh_h, mesh_p = 20, 3
        A_m, B_m, a_m, b_m, Ri_m, L, Req = np.array(multicell[:7])
        maxh = L / mesh_h * 1e-3

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(os.path.join(run_save_directory, "geodata.geo"))
        gmsh.model.mesh.generate(2)

        with suppress_c_stdout_stderr():
            gmsh.write(os.path.join(run_save_directory, "mesh.step"))

        step_geo = OCCGeometry(os.path.join(run_save_directory, "mesh.step"), dim=2)
        ngmesh = step_geo.GenerateMesh(maxh=maxh)
        bcs = self._get_boundaries_from_gmsh()
        gmsh.finalize()

        for key, bc_name in bcs.items():
            ngmesh.SetBCName(key - 1, bc_name)

        mesh = Mesh(ngmesh)
        mesh.Curve(mesh_p)
        self.save_mesh(run_save_directory, mesh)

        n_modes = self.requested_n_modes(n_modes=n_modes if n_modes is not None else no_of_cells + 2)
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(run_save_directory, run_save_directory, mesh, mesh_p, n_modes)

        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=no_of_cells - 1, n_cells=no_of_cells,
                                  L=L, save_dir=run_save_directory, write_axis=True)

        with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii, n_cells=no_of_cells, L=L)

        with open(os.path.join(run_save_directory, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    @staticmethod
    def _parse_adaptive(mesh_config):
        """Normalise ``mesh_config['adaptive']`` to a settings dict, or None if off.

        Adaptive mesh refinement is opt-in: pass ``True`` to enable it with the
        defaults, or a dict to override them::

            'adaptive': True
            'adaptive': {'tol': 1e-12, 'max_refinements': 8, 'max_ndof': 100000,
                         'theta': 0.25}

        - ``tol``: error tolerance. Refinement stops once **every** solved
          mode's recovery ``max_err`` has fallen below this (default 1e-12).
        - ``theta``: Doerfler marking fraction — elements with error above
          ``theta * max_error`` are refined.
        - ``max_ndof`` / ``max_refinements``: hard caps; stop once either is
          reached even if the tolerance is not met.
        """
        a = (mesh_config or {}).get('adaptive', None)
        if a is None or a is False:
            return None
        cfg = dict(a) if isinstance(a, dict) else {}
        cfg.setdefault('tol', 1e-12)
        cfg.setdefault('theta', 0.25)
        cfg.setdefault('max_ndof', 100000)
        cfg.setdefault('max_refinements', 8)
        return cfg

    def _build_system(self, mesh, mesh_p, m_pol, f_shift=0, direct_solver=None):
        """Build the reusable space, forms and preconditioner for azimuthal order
        *m_pol* — the single formulation used for **every** polarisation.

        The field ansatz is E = (E_r, E_z) cos(m phi) + E_phi sin(m phi) e_phi
        with the scaled azimuthal unknown ``u_phi = r * E_phi``, discretised on
        the product space ``HCurl x H1``. Two details make it valid for all m:

        - ``fes_phi`` is an H1 space of order ``mesh_p + 1`` that is **zero on
          the axis** (``dirichlet="PEC|AXI"``), not the space returned by
          ``CreateGradient``. The CreateGradient space (same ndof, PEC-only
          dirichlet) is used solely for the gradient-kernel projector.
        - the kernel (u, u_phi) = (grad psi, m psi) is removed with a
          b-orthogonal projector built from ``G = embU @ Grz - m * embPhi``.

        At **m = 0** the two blocks decouple: the HCurl block gives the monopole
        TM modes and the H1 block the monopole **TE** modes — which an HCurl-only
        formulation cannot represent at all. Validated against the analytical
        pillbox spectrum (TM and TE) for m = 0..8.

        The space and forms are created **once** and reused across adaptive
        refinements (each pass calls ``fes.Update()`` and reassembles). Creating
        a fresh space per pass instead leaves stale spaces registered on the
        mesh, and netgen's ``Refine()`` then updates them onto freed memory —
        an access violation on the second refinement.
        """
        if direct_solver is None:
            direct_solver = default_direct_solver()
        r = y
        fes_rz = HCurl(mesh, order=mesh_p, dirichlet="PEC")
        fes_phi = H1(mesh, order=mesh_p + 1, dirichlet="PEC|AXI")
        fes = fes_rz * fes_phi
        (u, u_phi), (v, v_phi) = fes.TnT()

        stiff = (r * curl(u) * curl(v)
                 + 1 / r * (m_pol**2 * u * v
                            + m_pol * u * grad(v_phi)
                            + m_pol * grad(u_phi) * v
                            + grad(u_phi) * grad(v_phi))) * dx
        mass = (r * u * v + 1 / r * u_phi * v_phi) * dx

        # Search around a frequency if a shift is provided
        if f_shift and f_shift != 'default':
            shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
            a = BilinearForm(stiff - shift_lam * mass)
        else:
            a = BilinearForm(stiff)
        b = BilinearForm(mass)

        return {'fes': fes, 'fes_rz': fes_rz, 'fes_phi': fes_phi,
                'a': a, 'b': b, 'm': m_pol,
                'f_shift': f_shift, 'direct_solver': direct_solver}

    def _solve_system(self, system, n_modes, pinvit_maxit=20):
        """Update, assemble and solve the reusable *system* on its (possibly
        just-refined) mesh. Returns ``(freq_fes, gfu_E, gfu_H)`` where each
        ``gfu_E`` entry is a product-space GridFunction (components: in-plane
        (E_z, E_r), u_phi = r*E_phi) and each ``gfu_H`` entry is a pair
        ``(H_inplane_cf, H_phi_cf)`` of azimuthal envelope coefficient
        functions. The representation is the same for every m."""
        fes, fes_rz = system['fes'], system['fes_rz']
        a, b, m_pol = system['a'], system['b'], system['m']
        f_shift, direct_solver = system['f_shift'], system['direct_solver']
        r = y

        with TaskManager():
            fes.Update()
            a.Assemble()
            b.Assemble()

            pre = (a.mat + b.mat).CreateSparseMatrix().Inverse(fes.FreeDofs())

            # Remove the gradient kernel (u, u_phi) = (grad psi, m psi). The
            # potential space comes from CreateGradient (PEC-only dirichlet);
            # it has the same ndof as fes_phi, so the two embeddings combine.
            Grz, fes_pot = fes_rz.CreateGradient()
            embU, embPhi = fes.embeddings
            G = (embU @ Grz - m_pol * embPhi).CreateSparseMatrix()
            GT = G.CreateTranspose()
            invh1 = (GT @ b.mat @ G).Inverse(inverse=direct_solver,
                                             freedofs=fes_pot.FreeDofs())
            projpre = (IdentityMatrix(fes.ndof) - G @ invh1 @ GT @ b.mat) @ pre

            evals_, evecs_ = solvers.PINVIT(a.mat, b.mat, pre=projpre,
                                            num=self.pinvit_n_modes(n_modes),
                                            maxit=pinvit_maxit, printrates=False)
            mask_ = np.array(evals_) > 1
            evals = np.array(evals_)[mask_]
            evecs = np.array(evecs_)[mask_]

            if f_shift and f_shift != 'default':
                shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
                freq_fes = [c0 * np.sqrt(np.abs(lam + shift_lam)) / (2 * np.pi) * 1e-6 for lam in evals]
            else:
                freq_fes = [c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6 for lam in evals]

            # 1/r is singular on the axis; H_inplane there is defined by its
            # limit, so clamp the weight instead of dividing by zero (the axis
            # is a measure-zero set in every integral that uses it).
            inv_r = IfPos(r - AXIS_EPS, 1 / r, 0)

            gfu_E = []
            gfu_H = []
            for i in range(len(evecs)):
                w = 2 * pi * freq_fes[i] * 1e6
                gfu = GridFunction(fes)
                gfu.vec.data = evecs[i]
                gfu_E.append(gfu)

                u_gf, uphi_gf = gfu.components
                # Azimuthal envelopes of H = curl(E) / (-i mu0 w); the phase
                # factor is dropped since only magnitudes enter the QOIs.
                H_inplane = inv_r / (mu0 * w) * (m_pol * u_gf + grad(uphi_gf))
                H_phi = 1 / (mu0 * w) * curl(u_gf)
                gfu_H.append((H_inplane, H_phi))

        return freq_fes, gfu_E, gfu_H

    def _solve_modes(self, mesh, mesh_p, m_pol, n_modes, save_dir=None,
                     f_shift=0, direct_solver=None, pinvit_maxit=20):
        """Solve the Maxwell eigenproblem for a single azimuthal order *m_pol*.

        The one entry point for every polarisation (m = 0 monopole included);
        builds a fresh system and solves it. See :meth:`_build_system`.
        """
        n_modes = self.requested_n_modes(n_modes=n_modes)
        system = self._build_system(mesh, mesh_p, m_pol, f_shift, direct_solver)
        freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)
        if save_dir:
            self.save_fields(save_dir, gfu_E, gfu_H)
        return freq_fes, gfu_E, gfu_H

    @staticmethod
    def _mode_error_field(mesh, gfu, fes_rec, fes_rec_vec):
        """Per-element Zienkiewicz-Zhu recovery error field for a single mode.

        Both halves of the product-space solution are covered, so the estimator
        sees every mode:

        - the in-plane part through ``curl(u)`` (the azimuthal H) recovered into
          the continuous scalar space *fes_rec* — this is the whole error for a
          TM mode;
        - the azimuthal part through ``grad(u_phi)`` recovered into the vector
          space *fes_rec_vec* — this is the whole error for a **TE** mode, whose
          ``u`` (and hence ``curl(u)``) is identically zero.

        Each recovery gap is integrated element-wise as ``integral r*(f - proj)^2``
        and the two are summed. Returns a per-element numpy array (non-negative).
        """
        u, uphi = gfu.components
        h = curl(u)
        hstar = GridFunction(fes_rec)
        hstar.Set(h)
        err = np.abs(Integrate(y * (h - hstar) * (h - hstar), mesh, VOL,
                               element_wise=True).NumPy())

        g = grad(uphi)
        gstar = GridFunction(fes_rec_vec)
        gstar.Set(g)
        err = err + np.abs(Integrate(y * InnerProduct(g - gstar, g - gstar), mesh, VOL,
                                     element_wise=True).NumPy())
        return err

    def _error_fields(self, mesh, fes_rz, gfu_list):
        """Per-mode recovery error field for every mode in *gfu_list*.

        Returns a list of per-element numpy arrays (one per mode). Callers pass
        them to :meth:`_refinement_driver` for the marking field and take each
        one's max for the per-mode convergence tolerance.
        """
        _, fes_rec = fes_rz.CreateGradient()
        fes_rec_vec = VectorH1(mesh, order=fes_rz.globalorder)
        return [self._mode_error_field(mesh, g, fes_rec, fes_rec_vec) for g in gfu_list]

    @staticmethod
    def _refinement_driver(fields):
        """Marking field for a mesh that has to resolve several modes at once.

        Each mode's error field is normalised by its **own** peak before the
        modes are combined (element-wise max). Combining the *raw* fields
        instead lets the high modes dominate — their errors are orders of
        magnitude larger — so a low mode's worst element never clears
        ``theta * max``, is never refined, and its error freezes at exactly the
        same value level after level (a flat step in an error-vs-DOF plot).

        Normalised, every mode's worst element scores 1.0 and is always marked,
        so no mode can stall. Returns a per-element array in [0, 1] (peak 1),
        which makes ``theta`` a fraction of *each mode's own* peak. None if
        there is no error to act on.
        """
        driver = None
        for f in fields:
            fmax = float(f.max()) if len(f) else 0.0
            if fmax <= 0:
                continue
            fn = f / fmax
            driver = fn if driver is None else np.maximum(driver, fn)
        return driver

    def _adaptive_refine_hcurl(self, mesh, mesh_p, n_modes, pinvit_maxit,
                               system, adaptive, first, save_dir=None):
        """Error-driven h-refinement of *mesh* (refined in place).

        Reuses the *system* (space + forms) built by :meth:`_build_system`
        so only one space is ever registered on the mesh. Starting from the
        *first* solve, mark-and-refine until every mode's recovery ``max_err``
        is below ``tol`` (or a DOF/refinement cap is hit), then return the
        solution on the finest mesh. Every level is recorded — DOF count,
        element count, the frequencies and the per-mode ``max_err`` — so a
        convergence study can plot error vs DOFs along the refinement path.
        """
        theta = float(adaptive['theta'])
        max_ndof = int(adaptive['max_ndof'])
        max_ref = int(adaptive['max_refinements'])
        tol = float(adaptive['tol'])
        fes, fes_rz = system['fes'], system['fes_rz']

        freq_fes, gfu_E, gfu_H = first
        history = []
        for step in range(max_ref + 1):
            fields = self._error_fields(mesh, fes_rz, gfu_E)
            per_mode_max = [float(f.max()) if len(f) else 0.0 for f in fields]
            # Drive and gate on the requested physical modes only. PINVIT solves
            # n_modes + 2 for accuracy; the top padding modes are barely converged
            # (huge, noisy error) and would otherwise hijack the refinement and
            # make the tolerance unreachable.
            n_use = min(n_modes, len(fields))
            gate_max = max(per_mode_max[:n_use]) if n_use else 0.0

            history.append({
                'refinement': step,
                'No of DOFs': int(fes.ndof),
                'No of Mesh Elements': int(mesh.GetNE(VorB.VOL)),
                'freq [MHz]': [float(f) for f in freq_fes],
                'max_err': per_mode_max,
            })

            # Stop once every requested mode is converged to tol, or a cap is hit.
            if fes.ndof >= max_ndof or step >= max_ref or gate_max < tol:
                break

            driver = self._refinement_driver(fields[:n_use])
            if driver is None:
                break
            # driver is normalised to peak 1 per mode, so theta is a fraction of
            # each mode's own peak — no mode can be crowded out of the marking.
            mesh.ngmesh.Elements2D().NumPy()["refine"] = (driver > theta)
            del gfu_E, gfu_H, fields, driver
            gc.collect()
            mesh.Refine()
            mesh.Curve(mesh_p)
            freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)

        self._last_adaptive_history = history
        if save_dir:
            with open(os.path.join(save_dir, 'adaptive_history.json'), 'w') as f:
                json.dump(history, f, indent=4, separators=(',', ': '))
        return freq_fes, gfu_E, gfu_H

    def _solve_eigenproblem(self, cav, save_dir, mesh, mesh_p, n_modes=None, m=0, adaptive=None):
        """Assemble and solve the Maxwell eigenvalue problem for azimuthal order
        *m*. Returns (freqs, E_fields, H_fields).

        When *adaptive* is a settings dict (see :meth:`_parse_adaptive`) the
        mesh is refined in place to resolve the requested modes — the recovery
        error estimator is polarisation-agnostic, so this drives refinement for
        any *m* — and the returned fields are those of the finest refinement.
        """
        n_modes = self.requested_n_modes(cav, n_modes=n_modes)

        f_shift = 0
        direct_solver = default_direct_solver()
        pinvit_maxit = 20            # PINVIT iterations (P3-4: exposed via config)
        if hasattr(cav, 'eigenmode_config') and cav.eigenmode_config:
            f_shift = cav.eigenmode_config.get('f_shift', 0)
            direct_solver = cav.eigenmode_config.get('direct_solver', direct_solver)
            pinvit_maxit = int(cav.eigenmode_config.get('pinvit_maxit', pinvit_maxit))
        elif isinstance(cav, dict) and 'f_shift' in cav: # Fallback for some legacy calls
             f_shift = cav['f_shift']

        system = self._build_system(mesh, mesh_p, m, f_shift, direct_solver)
        freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)

        self._last_adaptive_history = None
        if adaptive:
            freq_fes, gfu_E, gfu_H = self._adaptive_refine_hcurl(
                mesh, mesh_p, n_modes, pinvit_maxit, system,
                adaptive, first=(freq_fes, gfu_E, gfu_H), save_dir=save_dir)

        self.save_fields(save_dir, gfu_E, gfu_H)
        return freq_fes, gfu_E, gfu_H

    def solve_convergence(self, cav, eigenmode_config=None):
        """Adaptive-h convergence data: the full per-mode QOIs at every level.

        The mesh is refined adaptively (monopole-error driven, see
        :meth:`_parse_adaptive`). At **each** refinement level every requested
        polarisation is solved on the current mesh and the full QOIs of all its
        modes are evaluated — the same rich per-mode records
        :meth:`evaluate_qois` produces for a normal
        run. Each record is tagged with the refinement level (``h_pass``), the
        monopole DOF count at that level (``No of DOFs``) and the level solve
        time (``time [s]``). Returns a flat list of QOI dicts; nothing is
        written to disk.

        Only the *mesh* refinement is adaptive/monopole-driven; the m-pole
        solves reuse that mesh. Their fresh product spaces are released before
        each ``Refine()`` — leaving them registered makes netgen update stale
        spaces onto freed memory (an access violation).
        """
        eigenmode_config = eigenmode_config or {}
        mesh_config = eigenmode_config.get('mesh_config', {})
        mesh_h = mesh_h_metres(mesh_config)
        mesh_p = mesh_config.get('p', 3)
        adaptive = self._parse_adaptive(mesh_config) or self._parse_adaptive({'adaptive': True})
        theta = float(adaptive['theta'])
        max_ndof = int(adaptive['max_ndof'])
        max_ref = int(adaptive['max_refinements'])
        tol = float(adaptive['tol'])

        pols = parse_polarisations(eigenmode_config.get('polarisation', 0))
        n_modes = self.requested_n_modes(cav, eigenmode_config)
        conductivity = eigenmode_config.get('conductivity', SIGMA_COPPER)
        rs_ohm = eigenmode_config.get('surface_resistance', None)
        L_mono = cav.parameters.get('L_m', None)
        if L_mono is None:
            L_mono = eigenmode_config.get('normalization_length', None)
        L_mpole = cav.parameters.get('L_m', 1)

        f_shift = eigenmode_config.get('f_shift', 0)
        direct_solver = eigenmode_config.get('direct_solver', default_direct_solver())
        pinvit_maxit = int(eigenmode_config.get('pinvit_maxit', 20))

        mesh = self._build_mesh(cav, mesh_h, mesh_p)
        system = self._build_system(mesh, mesh_p, 0, f_shift, direct_solver)

        rows = []
        for level in range(max_ref + 1):
            t0 = time.perf_counter()
            freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)
            ndof_level = int(system['fes'].ndof)

            # Per-mode recovery error fields: each mode gets its OWN 'max_err' so
            # its convergence can be plotted individually. The refinement is
            # driven by (and the tolerance gates on) the requested physical
            # monopole modes only — the top PINVIT padding modes are barely
            # converged and would otherwise hijack the refinement.
            mono_err = self._error_fields(mesh, system['fes_rz'], gfu_E)
            n_use = min(n_modes, len(mono_err))
            driver = self._refinement_driver(mono_err[:n_use])
            gate_max = max((float(f.max()) for f in mono_err[:n_use]), default=0.0)

            level_rows = []
            if 0 in pols:
                for ii in range(len(freq_fes)):
                    q = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii,
                                           n_cells=cav.n_cells, L=L_mono,
                                           conductivity=conductivity, surface_resistance_ohm=rs_ohm)
                    # composite key '<m>-<mode>' — polarisation then mode index,
                    # e.g. '0-0' monopole mode 0, '1-0' dipole mode 0.
                    q['mode_index'] = f"{q['m']}-{ii}"
                    q['max_err'] = float(mono_err[ii].max()) if len(mono_err[ii]) else 0.0
                    level_rows.append(q)

            mpole_spaces = []
            for m_pol in [pp for pp in pols if pp > 0]:
                fr_m, gE_m, gH_m = self._solve_modes(mesh, mesh_p, m_pol, n_modes)
                m_err = self._error_fields(mesh, gE_m[0].components[0].space, gE_m) if gE_m else []
                for ii in range(len(fr_m)):
                    q = self.evaluate_qois(mesh, gE_m, gH_m, fr_m, m_pol, mode_idx=ii,
                                           n_cells=cav.n_cells, L=L_mpole,
                                           conductivity=conductivity, surface_resistance_ohm=rs_ohm)
                    q['mode_index'] = f"{q['m']}-{ii}"
                    q['max_err'] = float(m_err[ii].max()) if len(m_err[ii]) else 0.0
                    level_rows.append(q)
                mpole_spaces.append((gE_m, gH_m))

            elapsed = time.perf_counter() - t0
            for q in level_rows:
                q['h_pass'] = level
                q['No of DOFs'] = ndof_level
                q['time [s]'] = elapsed
            rows.extend(level_rows)

            # stop once every requested monopole mode's max_err is below tol, or
            # a hard cap is hit (DOF count / refinement passes).
            if ndof_level >= max_ndof or level >= max_ref or gate_max < tol:
                break
            if driver is None:
                break
            # driver is normalised to peak 1 per mode, so theta is a fraction of
            # each mode's own peak — no mode can be crowded out of the marking.
            mesh.ngmesh.Elements2D().NumPy()["refine"] = (driver > theta)
            # Release this level's fresh (m-pole + monopole) grid functions so no
            # stale space is updated onto freed memory by Refine().
            del gfu_E, gfu_H, mpole_spaces, level_rows, mono_err, driver
            gc.collect()
            mesh.Refine()
            mesh.Curve(mesh_p)

        return rows

    # ──────────────────────────────────────────────────────────────────────
    # QOI evaluation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m=0, beta=1, save_dir=None, mode_idx=1,
                      n_cells=1, L=1, conductivity=SIGMA_COPPER, surface_resistance_ohm=None,
                      write_axis=False):
        """Cavity figures of merit for azimuthal order *m*, mode *mode_idx*.

        One evaluator for every polarisation — the eigenproblem is the unified
        product space (``u = (E_z, E_r)``, ``u_phi = r*E_phi``) for all m, and so
        are the QOIs here. Only the **voltage-derived** quantities branch on m:

        - m = 0 (monopole): the accelerating voltage is the on-axis integral of
          E_z and R/Q is longitudinal; the azimuthal integral gives 2*pi. TM
          modes (u_phi = 0) accelerate; TE modes (u = 0) get zero Vacc/R/Q.
        - m >= 1 (deflecting): E_z ~ r^m vanishes on axis, so the voltage is
          taken along an off-axis line r0 = aperture/2 and converted to the
          transverse kick by Panofsky-Wenzel, V_t = m V_z(r0)/(k r0); R/Q is
          transverse and 'R/Q_t' is normalised by r0^(2(m-1)) (r0-independent).
          The azimuthal integral gives pi.

        Everything else — stored energy, peak surface fields, the H1-projected
        surface-loss integral, Q, G, k_cc — is one expression with the
        polarisation's azimuthal factor. The wall material enters Q/Ploss/Rsh/G
        through the surface resistance (copper by default; pass
        *surface_resistance_ohm* for a fixed Rs, e.g. SRF niobium); G is
        material-independent.
        """
        w = 2 * pi * freq_fes[mode_idx] * 1e6
        k_wave = w / (beta * c0)
        u_gf, uphi_gf = gfu_E[mode_idx].components
        H_inplane, H_phi = gfu_H[mode_idx]
        az = 2 * pi if m == 0 else pi          # azimuthal integral factor

        # Active length for the Eacc normalisation. Elliptical cavities pass L
        # (half-cell length, so the active length is 2*L*n_cells); guns/pillboxes
        # pass L=None, so fall back to the on-axis field extent.
        axis_nodes = np.array(list(get_boundary_nodes(mesh, 'AXI')))
        minz, maxz = axis_nodes[:, 0].min(), axis_nodes[:, 0].max()
        if L is None:
            L = (maxz - minz) * 1e3 / (2 * n_cells)  # -> 2*L*n_cells*1e-3 == active length [m]

        # Stored energy. |E|^2 = |E_inplane|^2 + |E_phi|^2 with E_phi = u_phi/r,
        # so r*|E|^2 = r*|u|^2 + |u_phi|^2/r. The phi integral gives 2*pi (m=0)
        # or pi (m>=1, cos^2/sin^2 average).
        U = az * 0.5 * eps0 * Integrate(
            y * InnerProduct(u_gf, Conj(u_gf)) + 1 / y * uphi_gf * Conj(uphi_gf), mesh).real

        norm_u = Norm(u_gf)
        xpnts_surf = np.array(list(get_boundary_nodes(mesh, 'PEC')))
        r0 = 0.5 * xpnts_surf[:, 1][xpnts_surf[:, 1] > AXIS_EPS].min()   # aperture/2 (m>=1)
        n_ax_pts = int(5000 * (maxz - minz))
        xpnts_ax = np.linspace(minz, maxz, n_ax_pts)

        # --- Voltage / gradient / R-over-Q: the only physics that branches -----
        if m == 0:
            # On-axis E_z; a TE mode has no E_z, so its Vacc is zero.
            Vout = abs(Integrate(u_gf[0] * exp(1j * k_wave * x), mesh,
                                 definedon=mesh.Boundaries('AXI')))
            RoQ_norm = RoQ = Vout ** 2 / (w * U)
            Ez_axis = np.array([norm_u(mesh(xi, 0.0)) for xi in xpnts_ax])   # |E| on axis
        else:
            # Off-axis line r0 + Panofsky-Wenzel kick (E_z ~ r^m vanishes on axis).
            Ez_r0 = np.array([u_gf(mesh(xi, r0))[0] for xi in xpnts_ax])
            trapz = getattr(np, 'trapezoid', np.trapz)
            Vz = abs(trapz(Ez_r0 * np.exp(1j * k_wave * xpnts_ax), xpnts_ax))
            Vout = m * Vz / (k_wave * r0)
            RoQ = Vout ** 2 / (w * U)
            RoQ_norm = RoQ / r0 ** (2 * (m - 1))
            Ez_axis = np.abs(Ez_r0)
        Eout = Vout / (n_cells * L * 1e-3 * 2)   # active length = 2*L*n_cells

        # --- Peak surface fields (pointwise maxima over the azimuth) -----------
        norm_H_in, norm_H_phi = Norm(H_inplane), Norm(H_phi)
        Esurf, Hsurf = [], []
        for (xi, yi) in xpnts_surf:
            mip = mesh(xi, yi)
            e_phi = abs(uphi_gf(mip)) / yi if yi > AXIS_EPS else 0.0
            Esurf.append(max(norm_u(mip), e_phi))
            Hsurf.append(max(norm_H_in(mip), norm_H_phi(mip)))
        Epk = max(Esurf)
        Hpk = max(Hsurf)

        # --- Surface power loss: H1-projected boundary integral (all m) --------
        # H = curl(E)/(mu0 w) cannot be SIMD-evaluated as a GridFunction curl on
        # a boundary, so it is projected into continuous H1 fields whose traces
        # integrate cleanly — mesh-convergent and higher-order than sampling |H|
        # at the element endpoints. 0.5 = time averaging, az = azimuth.
        Rs = surface_resistance(w, conductivity, surface_resistance_ohm)
        order_ = u_gf.space.globalorder
        Hphi_gf = GridFunction(H1(mesh, order=order_, complex=True))
        Hphi_gf.Set(H_phi)
        Hin_gf = GridFunction(VectorH1(mesh, order=order_, complex=True))
        Hin_gf.Set(H_inplane)
        Ploss = az * 0.5 * Rs * Integrate(
            y * (Hphi_gf * Conj(Hphi_gf) + InnerProduct(Hin_gf, Conj(Hin_gf))), mesh,
            definedon=mesh.Boundaries('PEC')).real

        # Cell-to-cell coupling: mode vs the lowest passband mode (index 0).
        if len(freq_fes) > 1:
            kcc = 2 * (freq_fes[mode_idx] - freq_fes[0]) / (freq_fes[mode_idx] + freq_fes[0]) * 100
        else:
            kcc = 0

        Q = w * U / Ploss
        G = Q * Rs

        # Axis field flatness (min/max of the on-axis |E| peaks)
        peaks, _ = find_peaks(Ez_axis, distance=n_ax_pts // 100, width=100)
        try:
            ff = min(Ez_axis[peaks]) / max(Ez_axis[peaks]) * 100
        except ValueError:
            ff = 0

        qois = {
            "m": m,
            "polarisation": pol_name(m),
            "Normalization Length [mm]": 2 * L,
            "N Cells": n_cells,
            "freq [MHz]": freq_fes[mode_idx],
            "Q []": Q,
            "Vacc [MV]": Vout * 1e-6,
            "Eacc [MV/m]": Eout * 1e-6,
            "Epk [MV/m]": Epk * 1e-6,
            "Hpk [A/m]": Hpk,
            "Bpk [mT]": mu0 * Hpk * 1e3,
            "kcc [%]": kcc,
            "ff [%]": ff,
            "Rsh [MOhm]": RoQ * Q * 1e-6,
            "R/Q [Ohm]": RoQ,
            "Epk/Eacc []": Epk / Eout,
            "Bpk/Eacc [mT/MV/m]": mu0 * Hpk * 1e9 / Eout,
            "G [Ohm]": G,
            "GR/Q [Ohm^2]": G * RoQ,
            "No of Mesh Elements": mesh.GetNE(VorB.VOL),
        }
        if m >= 1:
            # Transverse-specific keys (the shared Vacc/Eacc/R-Q above already
            # hold the transverse analogues for m>=1).
            qois["r0 [mm]"] = r0 * 1e3
            qois["Vt [MV]"] = Vout * 1e-6
            qois["Et [MV/m]"] = Eout * 1e-6
            qois["R/Q_t [Ohm/m^(2(m-1))]"] = RoQ_norm

        # Only the PRIMARY mode of interest owns the axis-field CSV the plots
        # read; without the write_axis gate the all-modes loop clobbered it
        # (the LAST mode won), so a single-cell fundamental's plot showed a
        # higher mode (a node at the centre instead of the accelerating peak).
        if save_dir and write_axis:
            if m == 0:
                pd.DataFrame({'z(0, 0)': xpnts_ax, '|Ez(0, 0)|': Ez_axis}).to_csv(
                    os.path.join(save_dir, 'Ez_0_abs.csv'),
                    index=False, sep='\t', float_format='%.32f')
            else:
                pd.DataFrame({'z': xpnts_ax, '|Ez(r0)|': Ez_axis}).to_csv(
                    os.path.join(save_dir, f'Ez_r0_abs_mode_{mode_idx}.csv'),
                    index=False, sep='\t', float_format='%.32f')

        return qois

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def save_fields(project_folder, gfu_E, gfu_H):
        with open(os.path.join(project_folder, 'gfu_EH.pkl'), "wb") as f:
            pickle.dump([gfu_E, gfu_H], f)

    @staticmethod
    def save_mesh(folder, mesh):
        """Save the mesh into *folder* (a polarisation results folder)."""
        with open(os.path.join(folder, "mesh.pkl"), "wb") as f:
            pickle.dump(mesh, f)

    @staticmethod
    def load_fields(folder, mode):
        with open(os.path.join(folder, 'gfu_EH.pkl'), "rb") as f:
            [gfu_E, gfu_H] = pickle.load(f)
        return gfu_E, gfu_H

    @staticmethod
    def load_mesh(folder):
        with open(os.path.join(folder, 'mesh.pkl'), 'rb') as f:
            mesh = pickle.load(f)
        return mesh

    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _field_cf(gfu_e, gfu_h, which):
        """Magnitude coefficient function for plotting the product-space fields
        (every polarisation, monopole included).

        'E'/'H' give the total magnitude of the azimuthal envelope, combining
        the in-plane and azimuthal groups; 'Ephi'/'Hphi' isolate the azimuthal
        component. For a TM mode the azimuthal E vanishes and the in-plane H
        vanishes, so 'E' reduces to |E_inplane| and 'H' to |H_phi| — the same
        fields the HCurl-only monopole formulation used to plot.
        """
        u_gf, uphi_gf = gfu_e.components
        H_inplane, H_phi = gfu_h
        which = which.lower()
        e_phi = uphi_gf * IfPos(y - AXIS_EPS, 1 / y, 0)   # E_phi = u_phi / r
        if which in ('ephi', 'e_phi'):
            return Norm(e_phi)
        if which in ('hphi', 'h_phi'):
            return Norm(H_phi)
        if which == 'h':
            return sqrt(Norm(H_inplane) ** 2 + Norm(H_phi) ** 2)
        return sqrt(Norm(u_gf) ** 2 + Norm(e_phi) ** 2)

    def show_fields(self, folder, mode=1, which='E', plotter='ngsolve'):
        mesh = self.load_mesh(folder)
        gfu_E, gfu_H = self.load_fields(folder, mode)
        field_cf = self._field_cf(gfu_E[mode], gfu_H[mode], which)

        if plotter == 'matplotlib':
            self._plot_field_matplotlib(mesh, field_cf)
        else:
            return Draw(field_cf, mesh, order=2, settings={'Objects': {'Wireframe': False}})

    def show_mesh(self, folder, plotter='ngsolve'):
        mesh = self.load_mesh(folder)

        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            return Draw(mesh)

    def show_geometry(self, cav, maxh=20e-3, order=1, plotter='ngsolve'):
        """Draw the cavity's meshed geometry *without* needing a prior run — a
        coarse mesh is built on the fly just to preview the analysed domain."""
        mesh = self._build_mesh(cav, maxh, order)
        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            return Draw(mesh)

    @staticmethod
    def _mesh_points_and_triangles(mesh, subdivide=0):
        """Build (points, triangle-index-array) from the mesh's *actual*
        element connectivity.

        Using the real triangles avoids the spurious elements a point-cloud
        Delaunay triangulation creates across concave regions (the beampipe
        gap, the axis), which the old code tried — and failed — to remove with
        a hardcoded, scale-dependent edge-length mask. ``subdivide`` levels of
        uniform refinement add edge/'face' points so the high-order field and
        the curved boundary render more smoothly.
        """
        verts = list(mesh.vertices)
        vidx = {v.nr: i for i, v in enumerate(verts)}
        pts = [tuple(v.point) for v in verts]

        triangles = []
        for el in mesh.Elements(VorB.VOL):
            vs = [vidx[v.nr] for v in el.vertices]
            if len(vs) == 3:
                triangles.append(vs)
            elif len(vs) == 4:  # quad -> two triangles
                triangles.append([vs[0], vs[1], vs[2]])
                triangles.append([vs[0], vs[2], vs[3]])

        for _ in range(subdivide):
            pts, triangles = NGSolveMEVP._subdivide_triangles(pts, triangles)

        pts = np.array(pts)
        return pts, np.array(triangles)

    @staticmethod
    def _subdivide_triangles(pts, triangles):
        """One level of 1->4 midpoint refinement of a triangle list."""
        pts = list(pts)
        mids = {}

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in mids:
                mids[key] = len(pts)
                pts.append(((pts[a][0] + pts[b][0]) / 2.0,
                            (pts[a][1] + pts[b][1]) / 2.0))
            return mids[key]

        new_tris = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_tris += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        return pts, new_tris

    @staticmethod
    def _sample_field(mesh, pts, field_cf):
        """Evaluate *field_cf* at each (x, y) in *pts*, robust to boundary
        points that fall just outside curved elements and to 1/r envelopes
        that blow up on the axis."""
        vals = []
        for xv, yv in pts:
            try:
                val = field_cf(mesh(xv, yv))
            except Exception:
                try:
                    val = field_cf(mesh(xv, yv * (1 - 1e-9) if yv else 1e-9))
                except Exception:
                    val = np.nan
            vals.append(val if np.isfinite(val) else np.nan)
        return np.array(vals)

    @staticmethod
    def _plot_field_matplotlib(mesh, field_cf):
        """Render a field magnitude over the mesh using matplotlib tricontourf."""
        pts, triangles = NGSolveMEVP._mesh_points_and_triangles(mesh, subdivide=2)
        triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles=triangles)
        vals = NGSolveMEVP._sample_field(mesh, pts, field_cf)
        # Fill any residual NaNs (axis/boundary) so tricontourf spans the domain.
        if np.isnan(vals).any():
            vals = np.where(np.isnan(vals), np.nanmin(vals), vals)

        plt.tricontourf(triang, vals, levels=40, cmap='jet')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    @staticmethod
    def _plot_mesh_matplotlib(mesh):
        """Render the mesh using matplotlib triplot (true element edges)."""
        pts, triangles = NGSolveMEVP._mesh_points_and_triangles(mesh)
        triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles=triangles)
        plt.triplot(triang, lw=0.6, c='k')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Deformation utilities
    # ──────────────────────────────────────────────────────────────────────

    def gauss(self, n=11, sigma=1.0, shift=0.0):
        r = np.linspace(-int(n / 2) + 0.5, int(n / 2) - 0.5, n)
        g = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(r - shift) ** 2 / (2 * sigma ** 2))
        return g / max(g)

    def gaussian_deform(self, n_cells, surface, deformation_params):
        deform_vector = np.zeros((len(surface), 1))
        for n_cell in range(n_cells):
            n_cell_surf_indx = np.where(surface[:, 2] == n_cell + 1)[0]
            mn, mx = np.min(n_cell_surf_indx), np.max(n_cell_surf_indx)

            disp = deformation_params[n_cell * 3]
            sigma = (0.1 + (0.2 - 0.1) * (1 + deformation_params[n_cell * 3 + 1]) / 2) * len(n_cell_surf_indx)
            shift = deformation_params[n_cell * 3 + 2] * (mn + mx) / 2
            deform_vector += np.atleast_2d(disp * 2e-3 * self.gauss(len(surface), sigma, shift=shift)).T

        surface_def = deform_vector + surface
        surface_def[:, 2] = surface[:, 2]

        # enforce end planes
        surface_def[1, 0] = surface_def[0, 0]
        surface_def[-1, 0] = surface_def[-2, 0]
        surface_def[0, 1] = 0
        surface_def[-1, 1] = 0

        return pd.DataFrame(surface_def, columns=[1, 0, 2])


def get_boundary_nodes(mesh, boundary_name):
    """Extract unique boundary node coordinates for a named boundary."""
    boundary_nodes = set()
    for e in mesh.Elements(BND):
        if e.mat == boundary_name:
            for v in e.vertices:
                boundary_nodes.add(mesh[v].point)
    return boundary_nodes


class suppress_c_stdout_stderr:
    """Context manager to suppress C-level stdout/stderr (e.g. from Gmsh)."""

    def __enter__(self):
        self.stdout_fd = sys.__stdout__.fileno()
        self.stderr_fd = sys.__stderr__.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.null_fd, self.stdout_fd)
        os.dup2(self.null_fd, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.null_fd)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
