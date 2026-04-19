from abc import ABC, abstractmethod
from cavsim2d.cavity.base import Cavity
from cavsim2d.constants import *
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.utils.shared_functions import *
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

class Pillbox(Cavity):
    def __init__(self, n_cells, dims, beampipe='none'):
        super().__init__(n_cells, beampipe)

        L, Req, Ri, S, L_bp = dims
        self.n_cells = n_cells
        self.n_modes = n_cells + 1
        self.kind = 'pillbox'
        self.n_modules = 1  # change later to enable module run
        self.L = L
        self.Req = Req
        self.Ri = Ri
        self.S = S
        self.L_bp = L_bp
        self.beampipe = beampipe
        self.bc = 33
        self.cell_parameterisation = 'simplecell'

        self.shape = {
            'IC': [L, Req, Ri, S, L_bp],
            'BP': beampipe
        }
        self.shape_multicell = None

        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            # Create cavity directory directly inside project folder
            self.self_dir = os.path.join(self.projectDir, self.name)
            geo_dir = os.path.join(self.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)

            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)

    def write_geometry(self, parameters, n_cells, beampipe='none', write=None, plot=False, **kwargs):
        """

        Parameters
        ----------
        file_path
        n_cell
        cell_par
        beampipe
        plot
        kwargs

        Returns
        -------

        """

        names = ['L', 'Req', 'Ri', 'S', 'L_bp']
        L, Req, Ri, S, L_bp = (parameters[n]*1e-3 for n in names)

        step = 0.001

        if beampipe.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif beampipe.lower() == 'none':
            L_bp_l = 0.000
            L_bp_r = 0.000
        elif beampipe.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif beampipe.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            shift = (L_bp_l + L_bp_r + n_cells * L + (n_cells - 1) * S) / 2

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            # cav.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # lineTo(start_point, [-shift, Ri], step)
            pt = [-shift, Ri]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)
            # cav.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # add beampipe
            if L_bp_l > 0:
                # lineTo(pt, [-shift + L_bp_l, Ri], step)
                pt = [-shift + L_bp_l, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            for n in range(1, n_cells + 1):
                if n == 1:
                    # lineTo(pt, [-shift + L_bp_l, Req], step)
                    pt = [-shift + L_bp_l, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Req], step)
                    pt = [-shift + L_bp_l + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Ri], step)
                    pt = [-shift + L_bp_l + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L
                elif n > 1:
                    # lineTo(pt, [-shift + L_bp_l + S, Ri], step)
                    pt = [-shift + L_bp_l + S, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S, Req], step)
                    pt = [-shift + L_bp_l + S, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Req], step)
                    pt = [-shift + L_bp_l + S + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Ri], step)
                    pt = [-shift + L_bp_l + S + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L

            if L_bp_r > 0:
                # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri], step)
                pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # END PATH
            # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0], step)
            pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

    def get_geometric_parameters(self):
        self.parameters = {
            'Ri': self.Ri,
            'L': self.L,
            'Req': self.Req,
            'S': self.S,
            'L_bp': self.L_bp
        }

    def plot(self, what, ax=None, **kwargs):
        if what.lower() == 'geometry':
            ax = plot_pillbox_geometry(self.n_cells, self.L, self.Req, self.Ri, self.S, self.L_bp, self.beampipe)
            ax.set_xlabel('$z$ [m]')
            ax.set_ylabel(r"$r$ [m]")
            return ax

        if what.lower() == 'zl':
            if ax:
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
            return ax

        if what.lower() == 'convergence':
            try:
                if ax:
                    self._plot_convergence(ax)
                else:
                    fig, ax = plt.subplot_mosaic([['conv', 'abs_err']], layout='constrained', figsize=(12, 4))
                    self._plot_convergence(ax)
                return ax
            except ValueError:
                info("Convergence data not available.")

    def run_tune(self, tune_variable, cell_type='Mid Cell', freq=None, solver='SLANS', proc=0, resume=False, n_cells=1):
        """
        Tune current cavity geometry

        Parameters
        ----------
        n_cells: int
            Number of cells used for tuning.
        resume: bool
            Option to resume tuning or not. Only for shape space with multiple entries.
        proc: int
            Processor number
        solver: {'SLANS', 'Native'}
            Solver to be used. Native solver is still under development. Results are not as accurate as that of SLANS.
        freq: float
            Reference frequency in MHz
        cell_type: {'mid-cell', 'end-cell', 'single-cell'}
            Type of cell to tune
        tune_variable: {'Req', 'L'}
            Tune variable. Currently supports only the tuning of the equator radius ``Req`` and half-cell length ``L``

        Returns
        -------

        """

        iter_set = ['Linear Interpolation', TUNE_ACCURACY, 10]

        if freq is None:
            # calculate freq from mid cell length
            beta = 1
            freq = beta * c0 / (4 * self.mid_cell[5])
            info("Calculated freq from mid cell half length: ", freq)

        # create new shape space based on cell_type
        # if cell_type.lower() == 'mid cell':
        shape_space = {
            f'{self.name}':
                {
                    'IC': self.shape_space['IC'],
                    'OC': self.shape_space['OC'],
                    'OC_R': self.shape_space['OC_R'],
                    "BP": 'none',
                    'FREQ': freq
                }
        }

        if len(self.slans_tune_res.keys()) != 0:
            run_tune = input("This cavity has already been tuned. Run tune again? (y/N)")
            if run_tune.lower() == 'y':
                self.run_tune_ngsolve(shape_space, resume, proc, self.bc,
                                      SOFTWARE_DIRECTORY, self.projectDir, self.name,
                                      tune_variable, iter_set, cell_type,
                                      progress_list=[], convergence_list=self.convergence_list, n_cells=n_cells)

            # read tune results and update geometry
            try:
                self.get_ngsolve_tune_res(tune_variable, cell_type)
            except FileNotFoundError:
                error("Could not find the tune results. Please run tune again.")
        else:

            self.run_tune_ngsolve(shape_space, resume, proc, self.bc,
                                  SOFTWARE_DIRECTORY, self.projectDir, self.name,
                                  tune_variable, iter_set, cell_type,
                                  progress_list=[], convergence_list=self.convergence_list, n_cells=n_cells)
            # try:
            self.get_ngsolve_tune_res(tune_variable, cell_type)

    @staticmethod
    def run_tune_ngsolve(shape, resume, p, bc, parentDir, projectDir, filename,
                         tune_variable, iter_set, cell_type, progress_list, convergence_list, n_cells):
        tuner.tune_ngsolve(shape, bc, parentDir, projectDir, filename, resume=resume, proc=p,
                           tune_variable=tune_variable, iter_set=iter_set,
                           cell_type=cell_type, sim_folder='Optimisation',
                           progress_list=progress_list, convergence_list=convergence_list,
                           save_last=True,
                           n_cell_last_run=n_cells)  # last_key=last_key This would have to be tested again #val2

    def run_eigenmode(self, solver='ngsolve', freq_shift=0, boundary_cond=None, subdir='',
                      uq_config=None):
        """
        Run eigenmode analysis on cavity

        Parameters
        ----------
        solver: {'SLANS', 'NGSolve'}
            Solver to be used. Native solver is still under development. Results are not as accurate as that of SLANS.
        freq_shift:
            Frequency shift. Eigenmode solver searches for eigenfrequencies around this value
        boundary_cond: int
            Boundary condition of left and right cell/beampipe ends
        subdir: str
            Sub directory to save results to
        uq_config: None | dict
            Provides inputs required for uncertainty quantification. Default is None and disables uncertainty quantification.

        Returns
        -------

        """

        if boundary_cond:
            self.bc = boundary_cond

        self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape_space, self.n_modes, freq_shift,
                          self.bc, SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)
        # load quantities of interest
        try:
            self.get_eigenmode_qois()
        except FileNotFoundError as e:
            error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")

    def run_wakefield(self, MROT=2, MT=10, NFS=10000, wakelength=50, bunch_length=25,
                      DDR_SIG=0.1, DDZ_SIG=0.1, WG_M=None, marker='', operating_points=None, solver='ABCI'):
        """
        Run wakefield analysis on cavity

        Parameters
        ----------
        MROT: {0, 1}
            Polarisation 0 for longitudinal polarization and 1 for transversal polarization
        MT: int
            Number of time steps it takes for a beam to move from one mesh cell to the other
        NFS: int
            Number of frequency samples
        wakelength:
            Wakelength to be analysed
        bunch_length: float
            Length of the bunch
        DDR_SIG: float
            Mesh to bunch length ration in the r axis
        DDZ_SIG: float
            Mesh to bunch length ration in the z axis
        WG_M:
            For module simulation. Specifies the length of the beampipe between two cavities.
        marker: str
            Marker for the cavities. Adds this to the cavity name specified in a shape space json file
        wp_dict: dict
            Python dictionary containing relevant parameters for the wakefield analysis for a specific operating point
        solver: {'ABCI'}
            Only one solver is currently available

        Returns
        -------

        """

        if operating_points is None:
            wp_dict = {}
        exist = False

        if not exist:
            if solver == 'ABCI':
                self._run_abci(self.name, self.n_cells, self.n_modules, self.shape_space,
                               MROT=MROT, MT=MT, NFS=NFS, UBT=wakelength, bunch_length=bunch_length,
                               DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG,
                               parentDir=SOFTWARE_DIRECTORY, projectDir=self.projectDir, WG_M=WG_M, marker=marker,
                               operating_points=operating_points, freq=self.freq, R_Q=self.R_Q)

                try:
                    self.get_abci_data()
                    self.get_wakefield_qois()
                except FileNotFoundError:
                    error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

        else:
            try:
                self.get_abci_data()
                self.get_wakefield_qois()
            except FileNotFoundError:
                error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

    @staticmethod
    def _run_ngsolve(name, n_cells, n_modules, shape, n_modes, f_shift, bc, parentDir, projectDir, sub_dir='',
                     uq_config=None):
        start_time = time.time()
        # create folders for all keys
        ngsolve_mevp.createFolder(name, projectDir, subdir=sub_dir)
        ngsolve_mevp.pillbox(n_cells, n_modules, shape['IC'],
                             n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
                             parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        # run UQ
        if uq_config:
            uq(name, shape, ["freq", "R/Q", "Epk/Eacc", "Bpk/Eacc"],
               n_cells=n_cells, n_modules=n_modules, n_modes=n_modes,
               f_shift=f_shift, bc=bc, parentDir=parentDir, projectDir=projectDir)

        done(f'Done with Cavity {name}. Time: {time.time() - start_time}')

    @staticmethod
    def _run_abci(name, n_cells, n_modules, shape, MROT=0, MT=4.0, NFS=10000, UBT=50.0, bunch_length=20.0,
                  DDR_SIG=0.1, DDZ_SIG=0.1,
                  parentDir=None, projectDir=None,
                  WG_M=None, marker='', operating_points=None, freq=0, R_Q=0):

        # run abci code
        if WG_M is None:
            WG_M = ['']

        start_time = time.time()
        # run both polarizations if MROT == 2
        for ii in WG_M:
            try:
                if MROT == 2:
                    for m in range(2):
                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                         fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                         projectDir=projectDir,
                                         WG_M=ii, marker=ii)
                else:
                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                     fid=name, MROT=MROT, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir, projectDir=projectDir,
                                     WG_M=ii, marker=ii)
            except KeyError:
                if MROT == 2:
                    for m in range(2):
                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                         fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                         projectDir=projectDir,
                                         WG_M=ii, marker=ii)
                else:
                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                     fid=name, MROT=MROT, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir, projectDir=projectDir,
                                     WG_M=ii, marker=ii)

        done(f'Cavity {name}. Time: {time.time() - start_time}')
        if len(operating_points.keys()) > 0:
            try:
                if freq != 0 and R_Q != 0:
                    d = {}
                    # save qois
                    for key, vals in operating_points.items():
                        WP = key
                        I0 = float(vals['I0 [mA]'])
                        Nb = float(vals['Nb [1e11]'])
                        sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                        bl_diff = ['SR', 'BS']

                        # info("Running wakefield analysis for given operating points.")
                        for i, s in enumerate(sigma_z):
                            for ii in WG_M:
                                fid = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
                                try:
                                    for m in range(2):
                                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                                         fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                         bunch_length=s,
                                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                                         projectDir=projectDir,
                                                         WG_M=ii, marker=ii, sub_dir=f"{name}")
                                except KeyError:
                                    for m in range(2):
                                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                                         fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                         bunch_length=s,
                                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                                         projectDir=projectDir,
                                                         WG_M=ii, marker=ii, sub_dir=f"{name}")

                                dirc = os.path.join(projectDir, "Cavities", name, "wakefield")
                                # try:
                                k_loss = abs(ABCIData(dirc, f'{fid}', 0).loss_factor['Longitudinal'])
                                k_kick = abs(ABCIData(dirc, f'{fid}', 1).loss_factor['Transverse'])
                                # except:
                                #     k_loss = 0
                                #     k_kick = 0

                                d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, n_cells)

                    # save qoi dictionary
                    run_save_directory = os.path.join(projectDir, "Cavities", name, "wakefield")
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The working point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()


