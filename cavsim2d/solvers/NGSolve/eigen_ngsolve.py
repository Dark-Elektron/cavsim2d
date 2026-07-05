import os.path
import pickle
import sys

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

mu0 = 4 * pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458
SIGMA_COPPER = 5.96e7  # electrical conductivity of copper [S/m]


def surface_resistance(w, conductivity=SIGMA_COPPER, rs=None):
    """Surface resistance [Ohm] at angular frequency *w*.

    Normal conductor: Rs = sqrt(mu0 * w / (2 * conductivity)). For a
    superconductor (or any fixed-Rs material) pass ``rs`` to use that value
    directly instead."""
    if rs is not None:
        return rs
    return np.sqrt(mu0 * w / (2 * conductivity))


def default_direct_solver():
    """Name of the sparse direct-solver backend used for the monopole
    eigenproblem. PARDISO (fast) is only reliably present in the Windows
    NGSolve build, so other platforms default to the always-built-in
    ``sparsecholesky``. Override per run with
    ``eigenmode_config['direct_solver']``."""
    import platform
    return 'pardiso' if platform.system() == 'Windows' else 'sparsecholesky'


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

        mesh_h = 20
        mesh_p = 3
        if eigenmode_config:
            mesh_config = eigenmode_config.get('mesh_config', {})
            mesh_h = mesh_config.get('h', mesh_h) * 1e-3 if 'h' in mesh_config else mesh_h
            mesh_p = mesh_config.get('p', mesh_p)

        pols = parse_polarisations((eigenmode_config or {}).get('polarisation', 0))

        # Wall material for Q / Ploss / Rsh / G (default copper normal
        # conductor; pass 'surface_resistance' for a fixed Rs, e.g. SRF).
        conductivity = (eigenmode_config or {}).get('conductivity', SIGMA_COPPER)
        rs_ohm = (eigenmode_config or {}).get('surface_resistance', None)

        step_geo, ngmesh, bcs = self.load_geo(cav.geo_filepath, maxh=mesh_h)

        for key, bc in bcs.items():
            ngmesh.SetBCName(key - 1, bc)

        mesh = Mesh(ngmesh)
        mesh.Curve(mesh_p)

        # Half-cell length (mm) for the Eacc normalisation. Elliptical cavities
        # store it as 'L_m'; guns/pillboxes have no 'L_m', so allow an explicit
        # 'normalization_length' override and otherwise let evaluate_qois fall
        # back to the on-axis field extent (L_norm=None).
        L_norm = cav.parameters.get('L_m', None)
        if L_norm is None:
            L_norm = (eigenmode_config or {}).get('normalization_length', None)

        if 0 in pols:
            save_dir = os.path.join(cav.self_dir, 'eigenmode', pol_name(0))
            os.makedirs(save_dir, exist_ok=True)
            self.save_mesh(save_dir, mesh)
            freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(cav, save_dir, mesh, mesh_p)

            qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=cav.n_cells, n_cells=cav.n_cells, L=L_norm, save_dir=save_dir,
                                      conductivity=conductivity, surface_resistance_ohm=rs_ohm)
            with open(os.path.join(save_dir, 'qois.json'), "w") as f:
                json.dump(qois, f, indent=4, separators=(',', ': '))

            qois_all_modes = {}
            for ii in range(len(freq_fes)):
                qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii, n_cells=cav.n_cells, L=L_norm, save_dir=save_dir,
                                                        conductivity=conductivity, surface_resistance_ohm=rs_ohm)

            with open(os.path.join(save_dir, 'qois_all_modes.json'), "w") as f:
                json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        for m_pol in [p for p in pols if p > 0]:
            self._solve_mpole_on_mesh(cav, m_pol, mesh, mesh_p, eigenmode_config)

        return True

    def solve_mpole(self, cav, m=1, eigenmode_config=None):
        """Mesh *cav* and solve the m-pole (m >= 1) eigenproblem only.

        Results (mesh, fields, qois.json, qois_all_modes.json, off-axis
        field profiles) are written to ``<cavity>/eigenmode/<pol name>/``,
        e.g. ``eigenmode/dipole/`` for m=1.
        """
        m = pol_number(m)
        assert m >= 1, error('solve_mpole is for m >= 1; use solve() for the monopole.')

        mesh_h = 20
        mesh_p = 3
        if eigenmode_config:
            mesh_config = eigenmode_config.get('mesh_config', {})
            mesh_h = mesh_config.get('h', mesh_h) * 1e-3 if 'h' in mesh_config else mesh_h
            mesh_p = mesh_config.get('p', mesh_p)

        step_geo, ngmesh, bcs = self.load_geo(cav.geo_filepath, maxh=mesh_h)
        for key, bc in bcs.items():
            ngmesh.SetBCName(key - 1, bc)

        mesh = Mesh(ngmesh)
        mesh.Curve(mesh_p)

        self._solve_mpole_on_mesh(cav, m, mesh, mesh_p, eigenmode_config)
        return True

    def _solve_mpole_on_mesh(self, cav, m, mesh, mesh_p, eigenmode_config=None):
        """Solve azimuthal order *m* on an existing mesh and save all results
        to ``<cavity>/eigenmode/<pol name>/``."""
        pol_dir = os.path.join(cav.self_dir, 'eigenmode', pol_name(m))
        os.makedirs(pol_dir, exist_ok=True)
        self.save_mesh(pol_dir, mesh)

        n_modes = (eigenmode_config or {}).get('n_modes', cav.n_cells + 2)
        conductivity = (eigenmode_config or {}).get('conductivity', SIGMA_COPPER)
        rs_ohm = (eigenmode_config or {}).get('surface_resistance', None)
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem_mpole(mesh, mesh_p, m, n_modes, save_dir=pol_dir)

        L_norm = cav.parameters.get('L_m', 1)

        qois = self.evaluate_qois_mpole(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=0,
                                        n_cells=cav.n_cells, L=L_norm, save_dir=pol_dir,
                                        conductivity=conductivity, surface_resistance_ohm=rs_ohm)
        with open(os.path.join(pol_dir, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois_mpole(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=ii,
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

        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(run_save_directory, mesh, mesh_p)

        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=no_of_cells, n_cells=no_of_cells,
                                  L=L, save_dir=run_save_directory)

        with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii, n_cells=no_of_cells, L=L)

        with open(os.path.join(run_save_directory, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    def _solve_eigenproblem(self, cav, save_dir, mesh, mesh_p):
        """Assemble and solve the Maxwell eigenvalue problem. Returns (freqs, E_fields, H_fields)."""
        fes = HCurl(mesh, order=mesh_p, dirichlet="PEC")
        u, v = fes.TnT()

        f_shift = 0
        direct_solver = default_direct_solver()
        if hasattr(cav, 'eigenmode_config') and cav.eigenmode_config:
            f_shift = cav.eigenmode_config.get('f_shift', 0)
            direct_solver = cav.eigenmode_config.get('direct_solver', direct_solver)
        elif isinstance(cav, dict) and 'f_shift' in cav: # Fallback for some legacy calls
             f_shift = cav['f_shift']

        # Support searching around frequency if shift provided
        if f_shift and f_shift != 'default':
            shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
            a = BilinearForm(y * curl(u) * curl(v) * dx - shift_lam * y * u * v * dx)
        else:
            a = BilinearForm(y * curl(u) * curl(v) * dx)

        m = BilinearForm(y * u * v * dx)
        apre = BilinearForm(y * curl(u) * curl(v) * dx + y * u * v * dx)
        pre = Preconditioner(apre, "direct", inverse=direct_solver)

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            gradmat, fesh1 = fes.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()
            math1 = gradmattrans @ m.mat @ gradmat
            math1[0, 0] += 1
            invh1 = math1.Inverse(inverse=direct_solver, freedofs=fesh1.FreeDofs())

            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=cav.n_cells + 2, maxit=20,
                                          printrates=False)

            if f_shift and f_shift != 'default':
                shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
                freq_fes = [c0 * np.sqrt(np.abs(lam + shift_lam)) / (2 * np.pi) * 1e-6 for lam in evals]
            else:
                freq_fes = [c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6 for lam in evals]

            gfu_E = []
            gfu_H = []
            for i in range(len(evecs)):
                w = 2 * pi * freq_fes[i] * 1e6
                gfu = GridFunction(fes)
                gfu.vec.data = evecs[i]
                gfu_E.append(gfu)
                gfu_H.append(1j / (mu0 * w) * curl(gfu))

            self.save_fields(save_dir, gfu_E, gfu_H)

        return freq_fes, gfu_E, gfu_H

    def _solve_eigenproblem_mpole(self, mesh, mesh_p, m, n_modes, save_dir=None):
        """Solve the Maxwell eigenvalue problem for azimuthal order m >= 1.

        The field ansatz is E = (E_r, E_z) cos(m phi) + E_phi sin(m phi) e_phi
        with the scaled azimuthal unknown u_phi = r * E_phi, discretised on
        the product space HCurl x H1 (H1 from ``CreateGradient`` so that the
        gradient kernel (u, u_phi) = (-grad psi, m psi) is exactly
        representable). That kernel is removed from the PINVIT iteration with
        a b-orthogonal projector, mirroring the monopole gradient projection,
        so only ``n_modes`` physical eigenpairs need to be computed.

        Returns (freqs [MHz], gfu_E, gfu_H) where each gfu_E entry is a
        product-space GridFunction (components: in-plane (E_x, E_y), u_phi)
        and each gfu_H entry is a pair (H_inplane_cf, H_phi_cf) of azimuthal
        envelope coefficient functions.
        """
        r = y
        fes_rz = HCurl(mesh, order=mesh_p, dirichlet="PEC")
        Grz, fes_phi = fes_rz.CreateGradient()
        fes = fes_rz * fes_phi
        (u, u_phi), (v, v_phi) = fes.TnT()

        with TaskManager():
            a = BilinearForm((r*curl(u)*curl(v) + 1/r * (m**2*u*v + m*u*grad(v_phi) + m*grad(u_phi)*v + grad(u_phi)*grad(v_phi)))*dx).Assemble()
            b = BilinearForm((r*u*v + 1/r * u_phi*v_phi)*dx).Assemble()
            
            sum_mat = a.mat.CreateMatrix()
            sum_mat.AsVector().data = a.mat.AsVector() + b.mat.AsVector()
            pre = sum_mat.Inverse(fes.FreeDofs())

            # Build projector
            embU   = Embedding(fes.ndof, fes.Range(0))
            embPhi = Embedding(fes.ndof, fes.Range(1))
            G  = (embU @ Grz + (-m) * embPhi).CreateSparseMatrix()

            GT = G.CreateTranspose()
            math1 = GT @ b.mat @ G
            invh1 = math1.Inverse(inverse='pardiso', freedofs=fes_phi.FreeDofs())

            proj = IdentityMatrix(fes.ndof) - G @ invh1 @ GT @ b.mat
            projpre = proj @ pre

            evals_, evecs_ = solvers.PINVIT(a.mat, b.mat, pre=projpre, num=n_modes+4, maxit=10, printrates=False)

            evals = np.array(evals_)[np.array(evals_) > 1]
            evecs = np.array(evecs_)[np.array(evals_) > 1]

            freq_fes = [c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6 for lam in evals]

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
                H_inplane = 1 / (mu0 * w * r) * (m * u_gf + grad(uphi_gf))
                H_phi = 1 / (mu0 * w) * curl(u_gf)
                gfu_H.append((H_inplane, H_phi))

            if save_dir:
                self.save_fields(save_dir, gfu_E, gfu_H)

        return freq_fes, gfu_E, gfu_H

    # ──────────────────────────────────────────────────────────────────────
    # QOI evaluation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, beta=1, save_dir=None, mode_idx=1, n_cells=1, L=1,
                      conductivity=SIGMA_COPPER, surface_resistance_ohm=None):
        """Compute cavity figures of merit for mode *mode_idx*.

        Works for both standard cavities and RF guns - the only difference
        is whether axis nodes come from a linspace or directly from the mesh.

        The wall material enters Q, Ploss, Rsh and G through the surface
        resistance: by default a copper normal conductor (*conductivity* in
        S/m); pass *surface_resistance_ohm* for a fixed Rs (e.g. an SRF
        niobium cavity). The geometric factor G is material-independent.
        """
        w = 2 * pi * freq_fes[mode_idx] * 1e6

        # Active length for the Eacc normalisation. For elliptical cavities L is
        # the half-cell length so the active length is 2*L*n_cells. When L is
        # unknown (guns/pillboxes have no 'L_m'), fall back to the on-axis field
        # extent so Eacc is still a sensible average gradient.
        if L is None:
            axis_pts = np.array(list(get_boundary_nodes(mesh, 'AXI')))
            active_length_m = axis_pts[:, 0].max() - axis_pts[:, 0].min()
            L = active_length_m * 1e3 / (2 * n_cells)  # -> 2*L*n_cells*1e-3 == active length [m]

        # Accelerating voltage and gradient
        Vacc = abs(Integrate(gfu_E[mode_idx][0] * exp(1j * w / (beta * c0) * x), mesh,
                             definedon=mesh.Boundaries('AXI')))
        Eacc = Vacc / (n_cells * L * 1e-3 * 2) # Divisor is ActiveLength = n_cells * L_cell (Note: L is half-cell length)

        # Stored energy and R/Q
        U = 2 * pi * 0.5 * eps0 * Integrate(y * InnerProduct(gfu_E[mode_idx], Conj(gfu_E[mode_idx])), mesh)
        RoQ = Vacc ** 2 / (w * U)

        # Peak surface fields (pointwise maxima over the surface nodes)
        xpnts_surf = get_boundary_nodes(mesh, 'PEC')
        Esurf = [Norm(gfu_E[mode_idx])(mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        Hsurf = [Norm(gfu_H[mode_idx])(mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        Epk = max(Esurf)
        Hpk = max(Hsurf)

        # Surface power loss: exact boundary integral of |H|^2 over the PEC
        # wall (2*pi from the azimuthal revolution, 0.5 from time averaging).
        # NB: integrate the field over the boundary — sampling |H| at the
        # surface nodes and wrapping the list in CF(...) builds a constant
        # vector, which makes the result scale with the mesh node count.
        # NGSolve cannot SIMD-evaluate curl(GridFunction) directly on a
        # boundary, so the azimuthal H = curl(E)/(mu0 w) is first projected
        # into a continuous H1 field whose boundary trace integrates cleanly.
        Rs = surface_resistance(w, conductivity, surface_resistance_ohm)
        H1_space = H1(mesh, order=gfu_E[mode_idx].space.globalorder, complex=True)
        Hphi = GridFunction(H1_space)
        Hphi.Set(1j / (mu0 * w) * curl(gfu_E[mode_idx]))
        Ploss = 2 * pi * 0.5 * Rs * Integrate(
            y * (Hphi * Conj(Hphi)), mesh,
            definedon=mesh.Boundaries('PEC')).real

        # Cell-to-cell coupling
        f_diff = freq_fes[mode_idx] - freq_fes[1]
        f_add = freq_fes[mode_idx] + freq_fes[1]
        kcc = 2 * f_diff / f_add * 100

        Q = w * U / Ploss
        G = Q * Rs

        # Axis field and field flatness
        axis_nodes = np.array(list(get_boundary_nodes(mesh, 'AXI')))
        minz, maxz = axis_nodes[:, 0].min(), axis_nodes[:, 0].max()
        n_ax_pts = int(5000 * (maxz - minz))
        xpnts_ax = np.linspace(minz, maxz, n_ax_pts)
        Ez_0_abs = np.array([Norm(gfu_E[mode_idx])(mesh(xi, 0.0)) for xi in xpnts_ax])

        peaks, _ = find_peaks(Ez_0_abs, distance=n_ax_pts // 100, width=100)
        Ez_0_abs_peaks = Ez_0_abs[peaks]
        try:
            ff = min(Ez_0_abs_peaks) / max(Ez_0_abs_peaks) * 100
        except ValueError:
            ff = 0

        qois = {
            "Normalization Length [mm]": 2 * L,
            "N Cells": n_cells,
            "freq [MHz]": freq_fes[mode_idx],
            "Q []": Q,
            "Vacc [MV]": Vacc * 1e-6,
            "Eacc [MV/m]": Eacc * 1e-6,
            "Epk [MV/m]": Epk * 1e-6,
            "Hpk [A/m]": Hpk,
            "Bpk [mT]": mu0 * Hpk * 1e3,
            "kcc [%]": kcc,
            "ff [%]": ff,
            "Rsh [MOhm]": RoQ * Q * 1e-6,
            "R/Q [Ohm]": RoQ,
            "Epk/Eacc []": Epk / Eacc,
            "Bpk/Eacc [mT/MV/m]": mu0 * Hpk * 1e9 / Eacc,
            "G [Ohm]": G,
            "GR/Q [Ohm^2]": G * RoQ,
            "No of Mesh Elements": mesh.GetNE(VorB.VOL)
        }

        if save_dir:
            Ez_0_abs_df = pd.DataFrame({'z(0, 0)': xpnts_ax, '|Ez(0, 0)|': Ez_0_abs})
            Ez_0_abs_df.to_csv(os.path.join(save_dir, 'Ez_0_abs.csv'),
                               index=False, sep='\t', float_format='%.32f')

        return qois

    @staticmethod
    def evaluate_qois_mpole(mesh, gfu_E, gfu_H, freq_fes, m, beta=1, save_dir=None,
                            mode_idx=0, n_cells=1, L=1,
                            conductivity=SIGMA_COPPER, surface_resistance_ohm=None):
        """Compute figures of merit for m-pole (m >= 1) mode *mode_idx*.

        Same quantities as :meth:`evaluate_qois`, with the voltage-derived
        ones generalised to deflecting modes: the longitudinal voltage is
        integrated along an off-axis line r0 (half the smallest wall radius,
        since E_z ~ r^m vanishes on axis) and converted to the transverse
        kick voltage via Panofsky-Wenzel, V_t = m V_z(r0) / (k r0). The keys
        'Vacc'/'Eacc'/'R/Q' therefore hold the transverse analogues, and
        'R/Q_t [Ohm/m^(2(m-1))]' is additionally normalised by r0^(2(m-1))
        so it is independent of the choice of r0 (equal to 'R/Q [Ohm]' for
        the dipole). Peak surface fields are exact maxima over the azimuth,
        max(|E_inplane|, |E_phi|), because the two groups are 90 degrees
        out of phase in phi.
        """
        w = 2 * pi * freq_fes[mode_idx] * 1e6
        k_wave = w / (beta * c0)

        u_gf, uphi_gf = gfu_E[mode_idx].components
        H_inplane, H_phi = gfu_H[mode_idx]

        # Off-axis evaluation radius from the smallest wall (aperture) radius
        xpnts_surf = np.array(list(get_boundary_nodes(mesh, 'PEC')))
        r_aperture = xpnts_surf[:, 1][xpnts_surf[:, 1] > 1e-9].min()
        r0 = 0.5 * r_aperture

        # Longitudinal voltage along the line r = r0 and Panofsky-Wenzel kick
        axis_nodes = np.array(list(get_boundary_nodes(mesh, 'AXI')))
        minz, maxz = axis_nodes[:, 0].min(), axis_nodes[:, 0].max()
        n_ax_pts = int(5000 * (maxz - minz))
        xpnts_ax = np.linspace(minz, maxz, n_ax_pts)
        Ez_r0 = np.array([u_gf(mesh(xi, r0))[0] for xi in xpnts_ax])

        trapz = getattr(np, 'trapezoid', np.trapz)
        Vz_r0 = abs(trapz(Ez_r0 * np.exp(1j * k_wave * xpnts_ax), xpnts_ax))
        Vt = m * Vz_r0 / (k_wave * r0)
        Et = Vt / (n_cells * L * 1e-3 * 2)

        # Stored energy (phi integral of cos^2/sin^2 gives pi for m >= 1)
        U = pi * 0.5 * eps0 * Integrate(
            y * InnerProduct(u_gf, Conj(u_gf)) + 1 / y * uphi_gf * Conj(uphi_gf), mesh).real
        RoQ = Vt ** 2 / (w * U)
        RoQ_norm = RoQ / r0 ** (2 * (m - 1))

        # Peak surface fields: exact azimuthal maxima
        norm_u = Norm(u_gf)
        norm_H_inplane = Norm(H_inplane)
        norm_H_phi = Norm(H_phi)
        Esurf, Hsurf2 = [], []
        for (xi, yi) in xpnts_surf:
            mip = mesh(xi, yi)
            e_in = norm_u(mip)
            e_phi = abs(uphi_gf(mip)) / yi if yi > 1e-9 else 0.0
            h_in = norm_H_inplane(mip) if yi > 1e-9 else 0.0
            h_phi = norm_H_phi(mip)
            Esurf.append(max(e_in, e_phi))
            Hsurf2.append((h_in, h_phi))
        Epk = max(Esurf)
        Hpk = max(max(h) for h in Hsurf2)

        # Surface power loss: segment-wise line integral over the PEC wall,
        # pi factor from the azimuthal average of cos^2/sin^2
        Rs = surface_resistance(w, conductivity, surface_resistance_ohm)
        Ploss = 0.0
        for el in mesh.Elements(BND):
            if el.mat != 'PEC':
                continue
            p1, p2 = [mesh[vtx].point for vtx in el.vertices]
            dl = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            seg = 0.0
            for (xi, yi) in (p1, p2):
                mip = mesh(xi, yi)
                h_in = norm_H_inplane(mip) if yi > 1e-9 else 0.0
                h_phi = norm_H_phi(mip)
                seg += 0.5 * (h_in ** 2 + h_phi ** 2) * yi
            Ploss += seg * dl
        Ploss *= pi * 0.5 * Rs

        # Cell-to-cell coupling (same convention as the monopole passband)
        if len(freq_fes) > 1:
            kcc = 2 * (freq_fes[mode_idx] - freq_fes[1]) / (freq_fes[mode_idx] + freq_fes[1]) * 100
        else:
            kcc = 0

        Q = w * U / Ploss
        G = Q * Rs

        # Off-axis field profile and field flatness
        Ez_r0_abs = np.abs(Ez_r0)
        peaks, _ = find_peaks(Ez_r0_abs, distance=n_ax_pts // 100, width=100)
        Ez_r0_abs_peaks = Ez_r0_abs[peaks]
        try:
            ff = min(Ez_r0_abs_peaks) / max(Ez_r0_abs_peaks) * 100
        except ValueError:
            ff = 0

        qois = {
            "m": m,
            "polarisation": pol_name(m),
            "r0 [mm]": r0 * 1e3,
            "Normalization Length [mm]": 2 * L,
            "N Cells": n_cells,
            "freq [MHz]": freq_fes[mode_idx],
            "Q []": Q,
            "Vacc [MV]": Vt * 1e-6,
            "Vt [MV]": Vt * 1e-6,
            "Eacc [MV/m]": Et * 1e-6,
            "Et [MV/m]": Et * 1e-6,
            "Epk [MV/m]": Epk * 1e-6,
            "Hpk [A/m]": Hpk,
            "Bpk [mT]": mu0 * Hpk * 1e3,
            "kcc [%]": kcc,
            "ff [%]": ff,
            "Rsh [MOhm]": RoQ * Q * 1e-6,
            "R/Q [Ohm]": RoQ,
            "R/Q_t [Ohm/m^(2(m-1))]": RoQ_norm,
            "Epk/Eacc []": Epk / Et,
            "Bpk/Eacc [mT/MV/m]": mu0 * Hpk * 1e9 / Et,
            "G [Ohm]": G,
            "GR/Q [Ohm^2]": G * RoQ,
            "No of Mesh Elements": mesh.GetNE(VorB.VOL)
        }

        if save_dir:
            Ez_r0_abs_df = pd.DataFrame({'z': xpnts_ax, '|Ez(r0)|': Ez_r0_abs})
            Ez_r0_abs_df.to_csv(os.path.join(save_dir, f'Ez_r0_abs_mode_{mode_idx}.csv'),
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
        """Magnitude coefficient function for plotting; understands both the
        monopole HCurl fields and the m-pole product-space fields, where
        *which* may additionally be 'Ephi' or 'Hphi' for the azimuthal
        envelopes."""
        comps = getattr(gfu_e, 'components', ())
        which = which.lower()
        if len(comps) == 2:  # m-pole HCurl x H1 solution
            u_gf, uphi_gf = comps
            H_inplane, H_phi = gfu_h
            if which in ('ephi', 'e_phi'):
                return Norm(uphi_gf / y)
            if which == 'h':
                return Norm(H_inplane)
            if which in ('hphi', 'h_phi'):
                return Norm(H_phi)
            return Norm(u_gf)
        return Norm(gfu_e) if which == 'e' else Norm(gfu_h)

    def plot_fields(self, folder, mode=1, which='E', plotter='ngsolve'):
        mesh = self.load_mesh(folder)
        gfu_E, gfu_H = self.load_fields(folder, mode)
        field_cf = self._field_cf(gfu_E[mode], gfu_H[mode], which)

        if plotter == 'matplotlib':
            self._plot_field_matplotlib(mesh, field_cf)
        else:
            Draw(field_cf, mesh, order=2, settings={'Objects': {'Wireframe': False}})

    def plot_mesh(self, folder, plotter='ngsolve'):
        mesh = self.load_mesh(folder)

        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            Draw(mesh)

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
