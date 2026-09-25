"""Damped eigenmodes from a waveguide-port transparent boundary condition.

What this computes
------------------
The complex eigenfrequency of a cavity whose beam pipes are terminated in the exact
modal impedance of the pipe, and from it

    Q_ext = |Re w| / (2 |Im w|)

directly, with no perturbation step and no absorber to trust.

No claim is made here about how any commercial code does this. What IS solver-
independent is that a modal port condition makes the eigenproblem nonlinear in ``w``
(the operator contains the eigenvalue through ``beta``), and that ``Q_ext`` needs a
complex eigenvalue, hence a radiating or lossy boundary.

Why it is a NONLINEAR eigenproblem
----------------------------------
The port contributes a boundary term ``sum_n c_n(w) P_n P_n^T`` with

    c_n(w) = i k0^2 / beta_n(w),      beta_n = sqrt(k0^2 - k_c,n^2)

so the operator depends on the eigenvalue through ``beta``. Written as a pencil,

    stiff . E = k0^2 [ mass - i sum_n P_n P_n^T / beta_n(w) ] . E
              = k0^2 B(w) . E

which is a plain generalised eigenproblem *except* that ``B`` moves with ``w``. Two
things follow. Below a mode's cutoff ``beta`` is negative-imaginary, ``-i/beta`` is
real, and that port mode adds a **reactive** term: it shifts the frequency and adds
no damping, which is the correct physics for an evanescent mode and is exactly what
an absorber gets wrong near cutoff. Above cutoff ``beta`` is real, the term is
imaginary, ``B`` is complex symmetric rather than Hermitian, and the eigenvalue
becomes complex -- that imaginary part *is* the radiation loss.

How it is solved
----------------
Fixed-point iteration on ``w``: freeze ``beta_n(w^k)``, solve the resulting **linear**
complex-symmetric pencil for the eigenvalue nearest the previous one, update, repeat.
Each pass reuses the assembled ``stiff``/``mass`` -- only the rank-1 port block is
rebuilt, which is a few outer products.

This converges quickly when the port term is a modest correction, which is the
regime of a beam pipe on a cavity. It is not guaranteed for strongly overlapping
poles; :meth:`solve` reports the residual movement per mode so a mode that has not
settled is visible rather than silently wrong. Contour integration (Beyn) is the
route if that becomes a limitation.

Why not the alternative you may have in mind
--------------------------------------------
"Solve the closed (PMC) cavity, then integrate the Poynting flux at the port" does
not work, and not by a small margin. A lossless closed cavity has a **real**
eigenmode; ``H`` is then 90 degrees out of phase with ``E`` and the time-averaged
flux ``1/2 Re(E x H*)`` vanishes *identically*, everywhere. There is no power to
integrate and every ``Q_ext`` comes out infinite. Replacing the flux with a modal
decomposition of the standing wave does give a number, but it is a number that
depends on where the port plane was put -- move it a quarter guide-wavelength and
the standing-wave amplitude changes. That position dependence is precisely why the
classical recipes (Kroll-Yu; Balleyguier) either sweep the pipe length or solve with
two different port-plane conditions and combine them.
"""
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ngsolve import (GridFunction, IfPos, TaskManager, curl,  # type: ignore
                     grad, y as _y)

from cavsim2d.solvers.NGSolve.eigen_ngsolve import (AXIS_EPS, NGSolveMEVP, SIGMA_COPPER,
                                                    mesh_h_metres,
                                                    parse_boundary_conditions,
                                                    parse_polarisations)
from cavsim2d.solvers.eigenmode_result import pol_name
from cavsim2d.solvers.NGSolve.ports import (C0, port_planes, projection_vectors,
                                            propagation_constant)
from cavsim2d.utils.printing import info, warning


def _to_scipy(mat):
    """NGSolve sparse matrix -> scipy CSR (complex)."""
    rows, cols, vals = mat.COO()
    n = mat.height
    return sp.csr_matrix((np.array(vals, dtype=complex),
                          (np.array(rows), np.array(cols))), shape=(n, n))


class PortEigenSolver:
    """Complex eigenmodes of a cavity terminated in waveguide ports.

    >>> res = PortEigenSolver(n_port_modes=3).solve(cav, n_modes=30,
    ...                                             beampipe_length=0.30)
    >>> res[['freq [MHz]', 'Q_ext []']]
    """

    #: Relative change in w below which a mode is called converged.
    TOL = 1e-8
    #: Fixed-point passes. **One is the default, and the recommended value.**
    #:
    #: One pass is the LINEARISED problem: freeze ``Y`` at the closed-cavity
    #: resonance and solve a single complex-symmetric eigenproblem. The error is
    #: second order in how far the mode moves, and it is robust.
    #:
    #: Iterating is not. Measured on a pillbox mode above cutoff: pass 1 gives
    #: f = 2946.1 MHz, Q = 29.7; by pass 250 it has drifted to f = 2834.6, Q = 3.87
    #: -- BELOW the port cutoff while still reporting finite damping, which is
    #: impossible -- with the eigenvector overlap between passes swinging 0.21-1.00,
    #: i.e. the shift-invert latching onto different modes as the shift moves. The
    #: iteration is drawn toward the cutoff, where ``beta -> 0`` and ``Y -> inf``.
    MAX_PASSES = 1

    def __init__(self, n_port_modes=3, m=0, kind='TM'):
        self.n_port_modes = int(n_port_modes)
        self.m = int(m)
        self.kind = str(kind).upper()

    # -- assembly ---------------------------------------------------------
    def _assemble(self, cav, mesh_h, mesh_p, beampipe_length):
        solver = NGSolveMEVP()
        cfg = {'mesh_config': {'h': mesh_h, 'p': mesh_p}}
        if beampipe_length is not None:
            cfg['beampipe_length'] = beampipe_length
        mesh = solver._build_mesh(cav, mesh_h_metres({'h': mesh_h}), int(mesh_p),
                                  boundary_conditions=33, eigenmode_config=cfg)

        # The eigen weak form, unchanged, in complex arithmetic. Assembled ONCE:
        # neither matrix depends on w, only the rank-1 port block does.
        system = solver._build_system(mesh, int(mesh_p), self.m, complex_fes=True)
        fes, a, b = system['fes'], system['a'], system['b']
        with TaskManager():
            fes.Update()
            a.Assemble()
            b.Assemble()

        K, M = _to_scipy(a.mat), _to_scipy(b.mat)
        free = np.array([bool(d) for d in fes.FreeDofs()])
        idx = np.where(free)[0]

        p_vecs, kcs = projection_vectors(mesh, fes, self.n_port_modes,
                                         m=self.m, kind=self.kind)
        P = np.column_stack([p[idx] for p in p_vecs])
        return mesh, fes, K[idx][:, idx], M[idx][:, idx], P, kcs, idx

    @staticmethod
    def _port_block(P, kcs, k0, n_free):
        """``- i sum_n P_n P_n^T / beta_n(k0)`` as a sparse matrix.

        This is the whole w-dependence. Below a mode's cutoff ``beta`` is
        negative-imaginary so the term is REAL -- a reactance, no damping. Above it,
        the term is imaginary and the eigenvalue picks up its imaginary part.
        """
        B = sp.lil_matrix((n_free, n_free), dtype=complex)
        for j, k_c in enumerate(kcs):
            beta = propagation_constant(k0, k_c)
            if beta == 0:
                continue
            pj = P[:, j]
            nz = np.where(np.abs(pj) > 0)[0]
            if nz.size == 0:
                continue
            B[np.ix_(nz, nz)] += (-1j / beta) * np.outer(pj[nz], pj[nz])
        return B.tocsr()

    def _mode_fields(self, fes, idx, vec_free, f_mhz):
        """``(gfu_E, (H_inplane, H_phi))`` for one port mode.

        The eigenvector comes back on the free dofs only; it is scattered into a
        product-space GridFunction and the H fields are rebuilt the way the lossy
        path does (H stored with the ``i`` dropped), so ``evaluate_qois`` sees
        exactly what it sees for every other solver.
        """
        MU0_ = 4e-7 * np.pi
        gfu = GridFunction(fes)
        full = np.zeros(fes.ndof, dtype=complex)
        full[idx] = vec_free
        gfu.vec.FV().NumPy()[:] = full
        u_gf, uphi_gf = gfu.components
        w = 2 * np.pi * f_mhz * 1e6
        inv_r = IfPos(_y - AXIS_EPS, 1 / _y, 0)
        gH = (inv_r / (MU0_ * w) * (self.m * u_gf + grad(uphi_gf)),
              1 / (MU0_ * w) * curl(u_gf))
        return gfu, gH

    def _mode_qois(self, mesh, fes, idx, vec_free, f_mhz, cav):
        """R/Q and friends for one port mode, via the project's own evaluator."""
        gfu, gH = self._mode_fields(fes, idx, vec_free, f_mhz)
        try:
            q = NGSolveMEVP.evaluate_qois(mesh, [gfu], [gH], [f_mhz], m=self.m,
                                          mode_idx=0, n_cells=getattr(cav, 'n_cells', 1),
                                          L=cav.parameters.get('L_m', None))
        except Exception as exc:                      # a QOI must never kill a solve
            warning(f'{f_mhz:.1f} MHz: evaluate_qois failed ({exc})')
            return {}
        return {k: v for k, v in q.items()
                if k in ('R/Q [Ohm]', 'U [J]', 'Vacc [MV]', 'Epk [MV/m]')}

    # -- the solve --------------------------------------------------------
    def solve(self, cav, n_modes=20, mesh_h=12.0, mesh_p=3, beampipe_length=None,
              f_min_mhz=1.0, passes=None, with_qois=False):
        """Complex eigenmodes with the ports attached.

        With the default ``passes = 1`` this is the **linearised** port problem: the
        modal admittance is evaluated at the closed-cavity resonance and one complex
        eigenproblem is solved. See :data:`MAX_PASSES` for why iterating further is
        not recommended.

        With *with_qois* each mode's eigenvector is pushed back through the
        project's own :meth:`NGSolveMEVP.evaluate_qois`, adding ``R/Q [Ohm]`` and the
        rest -- which an impedance reconstruction needs and the eigenvalue alone
        cannot give.

        Returns a DataFrame with ``freq [MHz]``, ``Q_ext []``, ``Im(f) [MHz]``,
        ``passes`` and ``converged``.
        """
        max_passes = int(self.MAX_PASSES if passes is None else passes)
        mesh, fes, idx, rows, vecs = self._solve_raw(cav, n_modes, mesh_h, mesh_p,
                                                     beampipe_length, f_min_mhz,
                                                     max_passes)
        if with_qois:
            for row, vec in zip(rows, vecs):
                if vec is not None:
                    row.update(self._mode_qois(mesh, fes, idx, vec, row['freq [MHz]'], cav))

        df = pd.DataFrame(rows)
        if max_passes > 1:
            bad = int((~df['converged']).sum()) if len(df) else 0
            if bad:
                warning(f'{bad} of {len(df)} modes did not reach the fixed point in '
                        f'{max_passes} passes. Iterating is not recommended — see '
                        f'PortEigenSolver.MAX_PASSES.')
        return df

    def _solve_raw(self, cav, n_modes, mesh_h, mesh_p, beampipe_length,
                   f_min_mhz=1.0, max_passes=1):
        """The port eigenproblem, keeping the eigenvectors.

        Returns ``(mesh, fes, idx, rows, vecs)``: one row of frequency, ``Q_ext`` and
        convergence data per mode, sorted by frequency, and its free-dof
        eigenvector alongside (``None`` if that mode's solve failed).
        """
        mesh, fes, K, M, P, kcs, idx = self._assemble(cav, mesh_h, mesh_p,
                                                      beampipe_length)
        n_free = K.shape[0]
        info(f'port eigen solve: {n_free} free dofs, {P.shape[1]} port modes '
             f'({self.n_port_modes} per end), cutoffs '
             f'{np.unique(np.round(kcs * C0 / (2 * np.pi) * 1e-6, 1))} MHz')

        # Seed from the CLOSED cavity, through the project's own eigensolver rather
        # than a bare shift-invert. 'stiff' carries an enormous gradient null space
        # (every grad(psi) sits at lambda = 0); _solve_modes removes it with the
        # b-orthogonal kernel projector and the relative filter that go with this
        # weak form. Handing the raw pencil to ARPACK instead simply fails to
        # converge -- the kernel swamps any shift placed near the physical modes.
        f_seed_mhz, _, _ = NGSolveMEVP()._solve_modes(mesh, int(mesh_p), self.m,
                                                      int(n_modes))
        rows, vecs = [], []
        for f_seed in np.sort(np.asarray(f_seed_mhz, dtype=float)):
            if f_seed < f_min_mhz:
                continue
            lam_seed = (2 * np.pi * f_seed * 1e6 / C0) ** 2

            # Fixed point, with Aitken acceleration. Plain iteration converges
            # LINEARLY here -- measured contraction ~0.85 for a Q ~ 10 mode, i.e.
            # ~100 passes for 1e-8, each one a shift-invert factorisation. It does
            # not oscillate, so under-relaxation only makes it slower (measured).
            # Aitken extrapolates the geometric tail from three iterates at no extra
            # solve, which is what a linearly convergent scalar map is made for.
            lam, converged, k = complex(lam_seed), False, 0
            vec = None
            hist = []
            for k in range(1, max_passes + 1):
                Bw = M + self._port_block(P, kcs, np.sqrt(lam), n_free)
                try:
                    vals, evecs = spla.eigs(K, k=1, M=Bw, sigma=lam, which='LM')
                except Exception as exc:
                    warning(f'{f_seed:.1f} MHz: port eigensolve failed ({exc})')
                    break
                lam_new, vec = complex(vals[0]), evecs[:, 0]
                rel = abs(lam_new - lam) / max(abs(lam), 1e-30)
                lam = lam_new
                if rel < self.TOL:
                    converged = True
                    break

                hist.append(lam)
                if len(hist) >= 3:
                    d1 = hist[-2] - hist[-3]
                    d2 = hist[-1] - hist[-2]
                    denom = d2 - d1
                    if abs(denom) > 1e-30 * max(abs(hist[-1]), 1e-30):
                        lam_ex = hist[-1] - d2 * d2 / denom
                        # only accept an extrapolation that stays put: a bad one
                        # would throw the shift somewhere with no eigenvalue
                        if abs(lam_ex - lam) < 0.25 * max(abs(lam), 1e-30):
                            lam = lam_ex
                    hist.clear()

            w = C0 * np.sqrt(lam)
            f = w.real / (2 * np.pi) * 1e-6
            q = abs(w.real) / (2 * abs(w.imag)) if w.imag else np.inf
            rows.append({'freq [MHz]': f, 'Im(f) [MHz]': w.imag / (2 * np.pi) * 1e-6,
                         'Q_ext []': q, 'passes': k, 'converged': converged,
                         'f_closed [MHz]': f_seed})
            vecs.append(vec)

        order = np.argsort([r['freq [MHz]'] for r in rows])
        return mesh, fes, idx, [rows[i] for i in order], [vecs[i] for i in order]

    # -- boundary_conditions='port' -----------------------------------------
    def run(self, cav, eigenmode_config=None):
        """Solve *cav* with a waveguide port on each pipe end and write the results
        where every eigenmode run writes them.

        This is what ``cav.eigenmode.run({'boundary_conditions': 'port'})`` calls, so
        the modes land in ``cav.eigenmode.qois_df`` like any other solve. Each mode
        carries ``'Q_ext []'`` (from the complex eigenvalue, also reported as
        ``'Q_eig []'``), ``'Q_wall []'`` and their combination ``'Q []'``.

        Monopole only: the port profiles are implemented for the TM_0n pipe modes.
        """
        cfg = eigenmode_config or {}
        pols = parse_polarisations(cfg.get('polarisation', 0))
        if pols != [0]:
            raise NotImplementedError(
                "boundary_conditions='port' solves the monopole only (the port "
                f"profiles are the TM_0n pipe modes); asked for polarisations {pols}. "
                "Use boundary_conditions='oo' (PML) for m >= 1.")
        ends = parse_boundary_conditions(cfg.get('boundary_conditions', 'port'))
        if ends != ('port', 'port'):
            raise ValueError(
                f"waveguide ports go on both pipe ends ('pp' or 'port'); got {ends}.")
        if getattr(cav, 'dielectrics', None):
            raise NotImplementedError(
                f"cavity {cav.name!r} has dielectric regions, which the port solve "
                "does not model yet. Use boundary_conditions='oo' (PML) or remove them.")

        mesh_config = cfg.get('mesh_config') or {}
        if mesh_config.get('adaptive'):
            warning("adaptive refinement is not applied with waveguide ports; solving "
                    "on the mesh as built. Use mesh_config['h'] to control resolution.")
        mesh_h, mesh_p = mesh_config.get('h', 20), int(mesh_config.get('p', 3))
        n_modes = NGSolveMEVP.requested_n_modes(cav, cfg)
        self.n_port_modes = int(cfg.get('n_port_modes') or self.n_port_modes)

        mesh, fes, idx, rows, vecs = self._solve_raw(
            cav, n_modes, mesh_h, mesh_p, cfg.get('beampipe_length'))
        # The requested modes only: the closed-cavity seeds include two padding modes.
        keep = [i for i, v in enumerate(vecs) if v is not None][:n_modes]
        rows, vecs = [rows[i] for i in keep], [vecs[i] for i in keep]
        freqs = [float(r['freq [MHz]']) for r in rows]
        q_ext = [float(r['Q_ext []']) for r in rows]
        fields = [self._mode_fields(fes, idx, v, f) for v, f in zip(vecs, freqs)]
        gfu_E = [f[0] for f in fields]
        gfu_H = [f[1] for f in fields]

        solver = NGSolveMEVP()
        pol_dir = os.path.join(cav.self_dir, 'eigenmode', pol_name(0))
        os.makedirs(pol_dir, exist_ok=True)
        solver.save_fields(pol_dir, gfu_E, gfu_H, mesh_p, 0, freqs)

        L_norm = cav.parameters.get('L_m', None)
        if L_norm is None:
            L_norm = cfg.get('normalization_length', None)

        def qois(i, write_axis=False):
            q = NGSolveMEVP.evaluate_qois(
                mesh, gfu_E, gfu_H, freqs, 0, mode_idx=i, n_cells=cav.n_cells,
                L=L_norm, save_dir=pol_dir, write_axis=write_axis,
                conductivity=cfg.get('conductivity', SIGMA_COPPER),
                surface_resistance_ohm=cfg.get('surface_resistance'), q_diel=q_ext)
            q['Q_ext []'] = q_ext[i]
            return q

        moi = NGSolveMEVP.modes_of_interest(cav, 0, cfg, len(freqs))
        qois_moi = {}
        for i in moi:
            q = qois(i, write_axis=(i == moi[0]))
            q['mode_of_interest'] = str(i + 1)
            q['No of DOFs'] = int(fes.ndof)
            qois_moi[str(i + 1)] = q
        with open(os.path.join(pol_dir, 'qois_moi.json'), 'w') as f:
            json.dump(qois_moi, f, indent=4, separators=(',', ': '))
        with open(os.path.join(pol_dir, 'qois.json'), 'w') as f:
            json.dump(qois_moi[str(moi[0] + 1)], f, indent=4, separators=(',', ': '))
        with open(os.path.join(pol_dir, 'qois_all_modes.json'), 'w') as f:
            json.dump({i: qois(i) for i in range(len(freqs))}, f, indent=4,
                      separators=(',', ': '))
        return True


# ---------------------------------------------------------------------------
# Beyn contour integration
# ---------------------------------------------------------------------------

def beyn(T, centre, radius, n_probe=20, n_quad=64, rank_tol=1e-8,
         min_cliff=10.0, rng=None):
    """Every eigenvalue of the nonlinear problem ``T(z) v = 0`` inside a circle.

    Beyn's method. ``T(z)^-1`` has a pole at each eigenvalue, so Cauchy's theorem
    turns "find the eigenvalues inside this contour" into two contour integrals and
    a small dense eigenproblem::

        A0 = (1/2 pi i) contour_integral T(z)^-1 V dz
        A1 = (1/2 pi i) contour_integral z T(z)^-1 V dz

    with ``V`` a random probe of ``n_probe`` columns. An SVD of ``A0`` reveals how
    many eigenvalues are enclosed -- its singular values fall off a cliff at that
    count -- and projecting ``A1`` onto the retained left/right subspaces gives a
    ``k x k`` matrix whose eigenvalues are the ones sought.

    The integral is a trapezoidal sum on the circle, which for an analytic integrand
    converges *geometrically* in ``n_quad``: a few dozen points is plenty, and each
    is an independent factorisation.

    Parameters
    ----------
    T : callable
        ``T(z)`` -> sparse matrix, for complex ``z``.
    centre, radius : complex, float
        The circle. It must enclose no singularity of ``T`` other than the
        eigenvalues themselves -- see :meth:`PortEigenSolver.solve_beyn` for why
        that matters here.
    n_probe : int
        Probe columns. Must exceed the number of eigenvalues inside, or they will
        not all be found; the returned count hitting this value is the signal to
        raise it.
    rank_tol : float
        Floor below which singular values are pure noise. The rank itself is taken
        from the largest *ratio* between consecutive singular values above that
        floor -- the cliff -- not from the floor, which would keep noise directions
        and turn each one into a spurious eigenvalue.
    min_cliff : float
        Smallest singular-value ratio that counts as a cliff. Below it the contour
        is taken to enclose nothing, which is what stops an empty contour from
        manufacturing eigenvalues out of quadrature noise.

    Returns
    -------
    (eigenvalues, n_probe_used, singular_values)
    """
    rng = np.random.default_rng(0) if rng is None else rng
    n = T(centre).shape[0]
    ell = int(min(n_probe, n))
    V = (rng.standard_normal((n, ell)) + 1j * rng.standard_normal((n, ell))) / np.sqrt(n)

    A0 = np.zeros((n, ell), dtype=complex)
    A1 = np.zeros((n, ell), dtype=complex)
    for j in range(int(n_quad)):
        theta = 2 * np.pi * j / int(n_quad)
        z = centre + radius * np.exp(1j * theta)
        w = radius * np.exp(1j * theta) / int(n_quad)   # dz / (2 pi i), trapezoid
        try:
            X = spla.splu(T(z).tocsc()).solve(V)
        except Exception as exc:
            warning(f'beyn: factorisation failed at z={z:.4g} ({exc}); '
                    'quadrature point skipped')
            continue
        A0 += w * X
        A1 += (w * z) * X

    U, S, Wh = np.linalg.svd(A0, full_matrices=False)
    # Rank = where the singular values fall off a CLIFF, not where they cross a fixed
    # threshold. The count of enclosed eigenvalues shows up as a large ratio between
    # consecutive singular values (measured: 4.1e-2 -> 2.2e-5, a factor of 1900);
    # everything past it is quadrature noise. A fixed tolerance instead keeps that
    # noise, and each retained noise direction contributes a spurious eigenvalue --
    # which is how a trapped mode came back as five, four of them with Im(w) > 0
    # (i.e. growing, and impossible for a passive cavity).
    n_live = int(np.sum(S > rank_tol * max(S[0], 1e-300)))
    if n_live == 0:
        return np.array([], dtype=complex), ell, S
    # Ratios over ALL singular values, not just the live ones: the cliff is often
    # between the LAST real value and the first noise value, and restricting the
    # search to live-only misses it entirely (a 3-eigenvalue control then came back
    # as 2). Capped at n_live so a ratio inside the noise floor cannot win.
    ratios = S[:-1] / np.maximum(S[1:], 1e-300)
    # A cliff is also the evidence that there are ANY modes inside. With an empty
    # contour every singular value is quadrature noise of the same order, no ratio
    # stands out, and taking the largest anyway manufactures eigenvalues out of
    # nothing (measured: an empty contour returned three). Demand a real cliff.
    if float(np.max(ratios)) < min_cliff:
        return np.array([], dtype=complex), ell, S
    keep = min(int(np.argmax(ratios)) + 1, n_live)
    Uk, Sk, Wk = U[:, :keep], S[:keep], Wh[:keep].conj().T
    B = Uk.conj().T @ A1 @ Wk @ np.diag(1.0 / Sk)
    return np.linalg.eigvals(B), ell, S


def _add_solve_beyn():
    """Attach the Beyn-based solver to :class:`PortEigenSolver`."""

    def solve_beyn(self, cav, f_lo_mhz, f_hi_mhz, q_min=3.0, mesh_h=12.0, mesh_p=3,
                   beampipe_length=None, n_probe=24, n_quad=64):
        """Every damped mode in a frequency band, by contour integration.

        The fixed-point iteration in :meth:`solve` converges for trapped modes and
        does **not** converge for damped ones -- measured: 250 passes with the
        frequency still drifting, and the eigenvector jumping between modes because
        a shift-invert cannot track one pole while the shift moves. Beyn sidesteps
        that entirely: it finds every eigenvalue inside a contour at once, from
        independent solves, with no iteration and nothing to track.

        Parameters
        ----------
        f_lo_mhz, f_hi_mhz : float
            Real-frequency band to search.
        q_min : float
            Lowest ``Q_ext`` worth finding. It sets how far into the lower half
            plane the contour reaches, since a mode of quality ``Q`` sits at
            ``Im(lambda)/Re(lambda) ~ -1/Q``.

        Notes
        -----
        **The contour may not enclose a port cutoff.** ``beta_n = sqrt(lambda -
        k_c,n^2)`` has a branch point there, and Beyn's derivation needs ``T`` to be
        analytic inside the contour apart from its eigenvalues. This is checked and
        raises rather than returning a quietly wrong answer -- split the band at the
        cutoff instead.
        """
        mesh, fes, K, M, P, kcs, idx = self._assemble(cav, mesh_h, mesh_p,
                                                      beampipe_length)
        n_free = K.shape[0]

        lam_lo = (2 * np.pi * float(f_lo_mhz) * 1e6 / C0) ** 2
        lam_hi = (2 * np.pi * float(f_hi_mhz) * 1e6 / C0) ** 2
        re_c = 0.5 * (lam_lo + lam_hi)
        im_c = -re_c / (2.0 * float(q_min))
        centre = complex(re_c, im_c)
        radius = 1.15 * max(0.5 * (lam_hi - lam_lo), abs(im_c))

        for k_c in np.unique(kcs):
            lam_cut = float(k_c) ** 2
            if abs(complex(lam_cut, 0.0) - centre) < radius:
                f_cut = k_c * C0 / (2 * np.pi) * 1e-6
                raise ValueError(
                    f"the contour for {f_lo_mhz:.0f}-{f_hi_mhz:.0f} MHz encloses the "
                    f"{f_cut:.1f} MHz port cutoff, where beta = sqrt(lambda - k_c^2) "
                    f"has a branch point. Beyn needs T analytic inside the contour; "
                    f"split the band either side of {f_cut:.1f} MHz.")

        def T(z):
            return K - z * (M + self._port_block(P, kcs, np.sqrt(complex(z)), n_free))

        info(f'beyn: {n_free} dofs, contour centre {centre:.4g} radius {radius:.4g}, '
             f'{n_quad} quadrature points x {n_probe} probes')
        vals, ell, svals = beyn(T, centre, radius, n_probe=n_probe, n_quad=n_quad)

        rows = []
        for lam in vals:
            w = C0 * np.sqrt(complex(lam))
            if w.real < 0:
                w = -w
            f = w.real / (2 * np.pi) * 1e-6
            if not (f_lo_mhz * 0.9 <= f <= f_hi_mhz * 1.1):
                continue                      # outside the band of interest
            q = abs(w.real) / (2 * abs(w.imag)) if w.imag else np.inf
            rows.append({'freq [MHz]': f, 'Im(f) [MHz]': w.imag / (2 * np.pi) * 1e-6,
                         'Q_ext []': q})
        df = pd.DataFrame(rows).sort_values('freq [MHz]').reset_index(drop=True)
        if len(df) >= ell:
            warning(f'beyn found {len(df)} modes with only {ell} probe vectors — the '
                    f'count is probably truncated. Raise n_probe.')
        return df

    PortEigenSolver.solve_beyn = solve_beyn


_add_solve_beyn()
