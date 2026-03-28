import os.path
import pickle
import sys

from matplotlib import tri
from cavsim2d.utils.shared_functions import *
from ngsolve import *
from ngsolve import (x, y, dx, pi, Mesh, exp, BND,
                     GridFunction, BilinearForm, InnerProduct, curl,
                     Integrate, TaskManager, HCurl, Preconditioner, solvers, Norm, IdentityMatrix)
from ngsolve.webgui import Draw
from ngsolve.comp import VorB
from netgen.occ import *
from netgen.occ import OCCGeometry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from cavsim2d.utils.printing import *
import gmsh

mu0 = 4 * pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


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
        """Run eigenmode analysis on a single cavity object."""
        eigenmode_folder_structure = {
            'eigenmode': {'monopole': None, 'dipole': None}
        }
        make_dirs_from_dict(eigenmode_folder_structure, cav.self_dir)

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
        self.save_mesh(cav.self_dir, mesh)

        save_dir = os.path.join(cav.self_dir, 'eigenmode', 'monopole')
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(cav, save_dir, mesh, mesh_p)
        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, n=cav.n_cells, save_dir=save_dir)
        with open(os.path.join(save_dir, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, n=ii, save_dir=save_dir)

        with open(os.path.join(save_dir, 'qois_all_modes.json'), "w") as f:
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

        pol_subdir = 'dipole' if pol.lower() != 'monopole' else 'monopole'

        if opt:
            run_save_directory = projectDir / f'Cavities/{fid}/eigenmode'
        elif subdir == '':
            run_save_directory = projectDir / f'Cavities/{fid}/eigenmode/{pol_subdir}'
        else:
            run_save_directory = projectDir / f'Cavities/{subdir}/eigenmode/{fid}/{pol_subdir}'

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

        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes,
                                  save_dir=run_save_directory)

        with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, n=ii)

        with open(os.path.join(run_save_directory, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    def _solve_eigenproblem(self, cav, save_dir, mesh, mesh_p):
        """Assemble and solve the Maxwell eigenvalue problem. Returns (freqs, E_fields, H_fields)."""
        fes = HCurl(mesh, order=mesh_p, dirichlet="PEC")
        u, v = fes.TnT()

        a = BilinearForm(y * curl(u) * curl(v) * dx)
        m = BilinearForm(y * u * v * dx)
        apre = BilinearForm(y * curl(u) * curl(v) * dx + y * u * v * dx)
        pre = Preconditioner(apre, "direct", inverse="pardiso")

        with TaskManager():
            a.Assemble()
            m.Assemble()
            apre.Assemble()

            gradmat, fesh1 = fes.CreateGradient()
            gradmattrans = gradmat.CreateTranspose()
            math1 = gradmattrans @ m.mat @ gradmat
            math1[0, 0] += 1
            invh1 = math1.Inverse(inverse="pardiso", freedofs=fesh1.FreeDofs())

            proj = IdentityMatrix() - gradmat @ invh1 @ gradmattrans @ m.mat
            projpre = proj @ pre.mat
            evals, evecs = solvers.PINVIT(a.mat, m.mat, pre=projpre, num=cav.n_cells + 2, maxit=20,
                                          printrates=False)

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

    # ──────────────────────────────────────────────────────────────────────
    # QOI evaluation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, beta=1, save_dir=None, n=1, L=1):
        """Compute cavity figures of merit for mode *n*.

        Works for both standard cavities and RF guns - the only difference
        is whether axis nodes come from a linspace or directly from the mesh.
        """
        if n == 0:
            n = -1

        w = 2 * pi * freq_fes[n] * 1e6

        # Accelerating voltage and gradient
        Vacc = abs(Integrate(gfu_E[n][0] * exp(1j * w / (beta * c0) * x), mesh,
                             definedon=mesh.Boundaries('AXI')))
        Eacc = Vacc / (L * 1e-3 * 2 * n)

        # Stored energy and R/Q
        U = 2 * pi * 0.5 * eps0 * Integrate(y * InnerProduct(gfu_E[n], Conj(gfu_E[n])), mesh)
        RoQ = Vacc ** 2 / (w * U)

        # Peak surface fields
        xpnts_surf = get_boundary_nodes(mesh, 'PEC')
        Esurf = [Norm(gfu_E[n])(mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        Hsurf = [Norm(gfu_H[n])(mesh(xi, yi)) for (xi, yi) in xpnts_surf]
        Epk = max(Esurf)
        Hpk = max(Hsurf)

        # Surface power loss
        sigma_cond = 5.96e7  # copper conductivity [S/m]
        Rs = np.sqrt(mu0 * w / (2 * sigma_cond))
        Ploss = 2 * pi * 0.5 * Rs * Integrate(
            y * InnerProduct(CF(Hsurf), CF(Conj(Hsurf))), mesh,
            definedon=mesh.Boundaries('PEC'))

        # Cell-to-cell coupling
        f_diff = freq_fes[n] - freq_fes[1]
        f_add = freq_fes[n] + freq_fes[1]
        kcc = 2 * f_diff / f_add * 100

        Q = w * U / Ploss
        G = Q * Rs

        # Axis field and field flatness
        axis_nodes = np.array(list(get_boundary_nodes(mesh, 'AXI')))
        minz, maxz = axis_nodes[:, 0].min(), axis_nodes[:, 0].max()
        n_ax_pts = int(5000 * (maxz - minz))
        xpnts_ax = np.linspace(minz, maxz, n_ax_pts)
        Ez_0_abs = np.array([Norm(gfu_E[n])(mesh(xi, 0.0)) for xi in xpnts_ax])

        peaks, _ = find_peaks(Ez_0_abs, distance=n_ax_pts // 100, width=100)
        Ez_0_abs_peaks = Ez_0_abs[peaks]
        try:
            ff = min(Ez_0_abs_peaks) / max(Ez_0_abs_peaks) * 100
        except ValueError:
            ff = 0

        qois = {
            "Normalization Length [mm]": 2 * L,
            "N Cells": n,
            "freq [MHz]": freq_fes[n],
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

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def save_fields(project_folder, gfu_E, gfu_H):
        with open(os.path.join(project_folder, 'gfu_EH.pkl'), "wb") as f:
            pickle.dump([gfu_E, gfu_H], f)

    @staticmethod
    def save_mesh(cav_dir, mesh):
        with open(os.path.join(cav_dir, "eigenmode", "monopole", "mesh.pkl"), "wb") as f:
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

    def plot_fields(self, folder, mode=1, which='E', plotter='ngsolve'):
        mesh = self.load_mesh(folder)
        gfu_E, gfu_H = self.load_fields(folder, mode)

        if plotter == 'matplotlib':
            self._plot_field_matplotlib(mesh, gfu_E[mode])
        else:
            field = gfu_E[mode] if which == 'E' else gfu_H[mode]
            Draw(Norm(field), mesh, order=2, settings={'Objects': {'Wireframe': False}})

    def plot_mesh(self, folder, plotter='ngsolve'):
        mesh = self.load_mesh(folder)

        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            Draw(mesh)

    @staticmethod
    def _plot_field_matplotlib(mesh, gfu):
        """Render a field magnitude on the mesh using matplotlib tricontourf."""
        mesh_points = mesh.vertices
        E_values, pts = [], []
        for pp in mesh_points:
            pts.append(pp.point)
            E_values.append(Norm(gfu)(mesh(*pp.point)))

        pts = np.array(pts)
        triang = tri.Triangulation(pts[:, 0], pts[:, 1])

        max_radius = 0.02
        triangles = triang.triangles
        xtri = pts[:, 0][triangles] - np.roll(pts[:, 0][triangles], 1, axis=1)
        ytri = pts[:, 1][triangles] - np.roll(pts[:, 1][triangles], 1, axis=1)
        triang.set_mask(np.max(xtri, axis=1) > max_radius)

        plt.tricontourf(triang, E_values, cmap='jet')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    @staticmethod
    def _plot_mesh_matplotlib(mesh):
        """Render the mesh using matplotlib triplot."""
        mesh_points = mesh.vertices
        pts = np.array([pp.point for pp in mesh_points])
        triang = tri.Triangulation(pts[:, 0], pts[:, 1])

        max_radius = 0.02
        triangles = triang.triangles
        xtri = pts[:, 0][triangles] - np.roll(pts[:, 0][triangles], 1, axis=1)
        ytri = pts[:, 1][triangles] - np.roll(pts[:, 1][triangles], 1, axis=1)
        triang.set_mask(np.max(np.sqrt(xtri ** 2 + ytri ** 2), axis=1) > max_radius)

        plt.triplot(triang, lw=1, c='k', zorder=4)
        plt.scatter(pts[:, 0], pts[:, 1])
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
