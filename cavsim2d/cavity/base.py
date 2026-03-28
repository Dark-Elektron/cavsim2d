from IPython.core.display import HTML, display_html, Math
from IPython.core.display_functions import display
from abc import ABC, abstractmethod
from cavsim2d.constants import *
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.processes import *
from cavsim2d.utils.shared_functions import *
from distutils import dir_util
from ipywidgets import HBox, VBox, Label
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import ast
import ipywidgets as widgets
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import os
import pandas as pd

# Safe arithmetic evaluator for simple expressions
_ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

class Cavity(ABC):
    """
    Command Line Interface module for running analysis.

    .. note::

       Still under development so some functions might not work properly
    """

    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k',
                 plot_label=None, geo_filepath=None):
        """
        Initialise cavity object. You can either specify geometry by dimensions
        (n_cells, mid_cell, end_cell_left, end_cell_right, etc.) *or* by providing
        a path to a geometry file (`geo_filepath`). If `geo_filepath` is not None,
        we load that file and skip the dimension‐based setup.

        Parameters
        ----------
        n_cells: int
            Number of cells (ignored if geo_filepath is provided)
        mid_cell: list or ndarray
            Mid‐cell geometric parameters (ignored if geo_filepath is provided)
        end_cell_left: list or ndarray
            Left end‐cell geometric parameters (ignored if geo_filepath is provided)
        end_cell_right: list or ndarray
            Right end‐cell geometric parameters (ignored if geo_filepath is provided)
        beampipe: {'none', 'both', 'left', 'right'}
            Beampipe options (ignored if geo_filepath is provided)
        name: str
            Name of the cavity
        cell_parameterisation: {'simplecell', 'flattop', ...}
            Parameterisation approach (ignored if geo_filepath is provided)
        color: str
            Colour for plotting
        plot_label: str or None
            Label for plotting; defaults to `name` if None
        geo_filepath: str or None
            If given, load geometry from this file instead of using dimensions.
        """

        # ───────────────────────────────────────────────────────────────────────────────────────────
        #  1) Set all “always‐present” attributes first (so nothing else breaks).
        # ───────────────────────────────────────────────────────────────────────────────────────────

        self.self_dir = None
        self.geo_filepath = None
        self.eigenmode_dir = None
        self.wakefield_dir = None
        self.uq_dir = None

        self.eigenmode_qois_all_modes = {}
        self.Epk_Eacc = None
        self.Bpk_Eacc = None
        self.Q = None
        self.Ez_0_abs = {'z(0, 0)': [], '|Ez(0, 0)|': []}
        self.uq_nodes = None
        self.convergence_df_data = None
        self.convergence_df = None
        self.uq_weights = None
        self.V_rf_config = 0
        self.Eacc_rf_config = 0
        self.rf_performance_qois_uq = {}
        self.rf_performance_qois = {}
        self.uq_hom_results = None
        self.sweep_results = {}
        self.sweep_results_uq = {}
        self.uq_fm_results = None
        self.mesh = None
        self.shape = None
        self.shape_multicell = None

        # plot_label: default to `name` if not provided
        self.name = name
        self.plot_label = plot_label if plot_label is not None else name

        self.projectDir = None
        self.bc = 33
        self.eigenmode_qois = {}
        self.custom_eig_qois = {}
        self.wakefield_qois = {}
        self.wake_op_points = {}
        self.convergence_list = []
        self.tune_results = {}
        self.operating_points = None
        self.Q0 = None
        self.inv_eta = None
        self.neighbours = {}
        self.wall_material = None

        # eigenmode results placeholders
        (self.R_Q,
         self.k_fm,
         self.GR_Q,
         self.freq,
         self.e,
         self.b,
         self.G,
         self.ff,
         self.k_cc,
         self.axis_field,
         self.surface_field) = [0] * 11

        # wakefield results placeholders
        (self.k_fm,
         self.k_loss,
         self.k_kick,
         self.phom,
         self.sigma,
         self.I0) = [{} for _ in range(6)]

        self.geo_filepath = geo_filepath

        # ───────────────────────────────────────────────────────────────────────────────────────────
        #  2) Choose initialisation path: “from file” or “from dimensions”
        # ───────────────────────────────────────────────────────────────────────────────────────────
        self.parameters = {}

    @abstractmethod
    def create(self):
        # If a geometry file is provided, skip dimension‐based setup:
        if self.geo_filepath:
            # make directory paths
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            self._init_from_geo(self.geo_filepath, os.path.join(self.self_dir, 'geometry'))
            self.get_geometric_parameters()

    def _init_from_geo(self, filepath, output_filepath, kind='geo'):
        """
        Load geometry from a file (e.g. a .geo)
        """
        self.step_geo, self.mesh, self.bcs = ngsolve_mevp.load_geo(filepath, output_filepath)

    def get_geometric_parameters(self):
        with open(self.geo_filepath, "r") as file:
            for line in file:
                match = re.match(r'\s*(\w+)\s*=\s*DefineNumber\[\s*([\d.eE+-]+)\s*,\s*Name\s*"Parameters/[^"]*"\s*\];',
                                 line)
                if match:
                    var_name, var_value = match.groups()
                    self.parameters[var_name] = float(var_value)

    def set_name(self, name):
        """
        Set cavity name

        Parameters
        ----------
        name: str
            Name of cavity

        Returns
        -------

        """
        self.name = name

    def set_color(self, color):
        """
        Set cavity name

        Parameters
        ----------
        color: str
            color of cavity

        Returns
        -------

        """
        self.color = color

    def set_parameterisation(self, cell_parameterisation):
        """
        Set cavity name

        Parameters
        ----------
        name: str
            Name of cavity

        Returns
        -------

        """
        self.cell_parameterisation = cell_parameterisation
        self.shape['CELL PARAMETERISATION'] = cell_parameterisation
        self.to_multicell()

    def set_plot_label(self, plot_label):
        """
        Set cavity plot label

        Parameters
        ----------
        plot_label: str
            Cavity plot label

        Returns
        -------

        """

        if plot_label is None:
            self.plot_label = self.name
        else:
            self.plot_label = plot_label

    def set_n_cells(self, n_cells):
        """
        Sets number of cells of cavity

        Parameters
        ----------
        n_cells: int
            Number of cavity cells

        Returns
        -------

        """
        self.n_cells = int(n_cells)
        self.shape['n_cells'] = n_cells
        self.to_multicell()

    def set_mid_cell(self, cell):
        """
        Set mid cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.mid_cell = cell
        self.shape['IC'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_end_cell_left(self, cell):
        """
        Set left end cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.end_cell_left = cell
        self.shape['OC'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_end_cell_right(self, cell):
        """
        Set right end cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.end_cell_right = cell
        self.shape['OC_R'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_boundary_conditions(self, bc):
        """
        Sets boundary conditions for the beampipes of cavity

        Parameters
        ----------
        bc: int
            Boundary condition of left and right cell/beampipe ends

        Returns
        -------

        """
        self.bc = int(bc)

    def set_beampipe(self, bp):
        """
        Set beampipe option of cavity

        Parameters
        ----------
        bp: str
            Beampipe option of cell

        Returns
        -------

        """
        self.beampipe = bp
        self.shape['BP'] = bp
        self.to_multicell()

    def load(self):
        """
        Load existing cavity project folder

        Parameters
        ----------

        Returns
        -------

        """
        pass

    def load_shape_space(self, filepath):
        """
        Get cavity geometric parameters from shape space

        Parameters
        ----------
        filepath: str
            Shape space directory

        Returns
        -------

        """
        pass

    def save_shape_space(self, filepath=None):
        """
        Save current geometric parameters as shape space

        Parameters
        ----------
        filepath: str
            Directory to save shape space to. If no input is given, it is saved to the Cavities directory

        Returns
        -------

        """
        pass

    def sweep(self, sweep_config, which='eigenmode', how='independent', uq_config=None):
        if how == 'cross':
            pass
        else:
            for key, interval_def in sweep_config.items():
                # save nominal variable value form shape space
                current_var = self.shape['IC'][VAR_TO_INDEX_DICT[key]]
                par_vals = np.linspace(interval_def[0], interval_def[1], interval_def[2], endpoint=True)
                self.sweep_results[f'{key}'] = {}
                for val in par_vals:
                    if which == 'eigenmode':
                        # change value
                        self.shape['IC'][VAR_TO_INDEX_DICT[key]] = val
                        res = self.run_eigenmode()
                        if res:
                            self.sweep_results[f'{key}'][val] = copy.deepcopy(self.eigenmode_qois)
                            if uq_config:
                                self.sweep_results_uq[f'{key}'][val] = copy.deepcopy(self.uq_fm_results)

                # replace initial value
                self.shape['IC'][VAR_TO_INDEX_DICT[key]] = current_var

    def study_mesh_convergence(self, h=2, h_passes=7, h_step=1.5, p=2, p_passes=6, p_step=1):
        """

        Parameters
        ----------
        h
        passes
        solver
        type: str
            h or p refinement

        Returns
        -------

        """

        convergence_df_ph = None
        convergence_df_data_ph = None
        # define start value, refinement (adaptive? fixed step?)
        hs = h
        for ip in range(p_passes):
            convergence_dict_h = {}
            for ih in range(h_passes):
                eigenmode_config = {'boundary_conditions': 'mm',
                                    'mesh_config': {
                                        'h': hs,
                                        'p': p
                                    }
                                    }

                run_eigenmode_s({self.name: self}, eigenmode_config, '')
                # read results
                self.get_eigenmode_qois()

                convergence_dict_h[ih] = self.eigenmode_qois
                convergence_dict_h[ih]['h'] = hs
                convergence_dict_h[ih]['p'] = p
                convergence_dict_h[ih]['No of Mesh Elements'] = self.eigenmode_qois['No of Mesh Elements']

                hs /= h_step
            convergence_df_data_h = pd.DataFrame.from_dict(convergence_dict_h, orient='index')
            convergence_df_h = self.calculate_rel_errors(convergence_df_data_h)

            if convergence_df_ph is None:
                convergence_df_data_ph = convergence_df_data_h
                convergence_df_ph = convergence_df_h
            else:
                convergence_df_data_ph = pd.concat([convergence_df_data_ph,
                                                    convergence_df_data_h], ignore_index=True)
                convergence_df_ph = pd.concat([convergence_df_ph,
                                               convergence_df_h], ignore_index=True)
            hs = h
            p += p_step

        # convert to dataframe
        self.convergence_df_data = convergence_df_data_ph
        self.convergence_df = convergence_df_ph

    def calculate_rel_errors(self, df):
        # Specify the columns to exclude
        columns_to_exclude = ['h', 'p', 'No of Mesh Elements']
        df_to_compute = df.drop(columns=columns_to_exclude)
        df_prev = df_to_compute.shift(1)
        relative_errors = (df_to_compute - df_prev).abs() / df_prev.abs()
        relative_errors.columns = [f'rel_error_{col}' for col in df_to_compute.columns]
        excluded_columns = df[columns_to_exclude]
        rel_errors_df = pd.concat([relative_errors, excluded_columns], axis=1)

        return rel_errors_df

    def run_eigenmode(self, solver='ngsolve', freq_shift=0, boundary_cond=None, subdir='', uq_config=None):
        """
        Run eigenmode analysis on cavity

        Parameters ---------- solver: {'SLANS', 'NGSolve'} Solver to be used. Native solver is still under
        development. Results are not as accurate as that of SLANS. freq_shift: Frequency shift. Eigenmode solver
        searches for eigenfrequencies around this value boundary_cond: int Boundary condition of left and right
        cell/beampipe ends subdir: str Sub directory to save results to uq_config: None | dict Provides inputs
        required for uncertainty quantification. Default is None and disables uncertainty quantification.

        Returns
        -------

        """

        if boundary_cond:
            self.bc = boundary_cond

        if self.cell_parameterisation == 'multicell':
            self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape, self.shape_multicell,
                              self.n_modes,
                              freq_shift, self.bc,
                              SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)
        else:
            self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape, self.shape_multicell,
                              self.n_modes,
                              freq_shift, self.bc,
                              SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)

        # load quantities of interest
        try:
            self.get_eigenmode_qois()
            if uq_config:
                self.get_uq_fm_results(
                    os.path.join(self.self_dir, 'eigenmode', 'uq.json'))
            return True
        except FileNotFoundError as e:
            error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")
            return False

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
        :param MROT:
        :param wakelength:
        :param WG_M:
        :param marker:
        :param solver:
        :param operating_points:

        """

        if operating_points is None:
            wp_dict = {}
        exist = False

        # check if R/Q is set
        if self.R_Q == 0:
            self.get_eigenmode_qois()
            self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        if not exist:
            if solver == 'ABCI':
                self._run_abci(self.name, self.n_cells, self.n_modules, self.shape,
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

    def calc_op_freq(self):
        """
        Calculates operating frequency. The operating frequency is used for tuning when a frequency is not given.
        It is advisable to always include the desired tune frequency. Example

        .. py:function:: cav.run_tune('Req', freq=1300)

        Returns
        -------

        """
        if not self.freq:
            self.freq = (c0 / 4 * self.L)

    @staticmethod
    def _run_ngsolve(name, n_cells, n_modules, shape, shape_multi, n_modes, f_shift, bc, parentDir, projectDir,
                     sub_dir='',
                     uq_config=None):
        parallel = False
        start_time = time.time()
        # create folders for all keys
        ngsolve_mevp.createFolder(name, projectDir, subdir=sub_dir)

        if 'OC_R' in shape.keys():
            OC_R = 'OC_R'
        else:
            OC_R = 'OC'

        # ngsolve_mevp.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
        #                     n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
        #                     parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        if shape['CELL PARAMETERISATION'] == 'flattop':
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())

            ngsolve_mevp.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                        n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc,
                                        beampipes=shape['BP'],
                                        parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

        elif shape['CELL PARAMETERISATION'] == 'multicell':
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())
            ngsolve_mevp.cavity_multicell(n_cells, n_modules, shape_multi['IC'], shape_multi['OC'], shape_multi[OC_R],
                                          n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc,
                                          beampipes=shape['BP'],
                                          parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        else:
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())
            ngsolve_mevp.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
                                parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

        # run UQ
        if uq_config:
            objectives = uq_config['objectives']
            solver_dict = {'eigenmode': ngsolve_mevp}
            solver_args_dict = {'eigenmode':
                                    {'n_cells': n_cells, 'n_modules': n_modules, 'f_shift': f_shift, 'bc': bc,
                                     'beampipes': shape['BP']
                                     },
                                'parentDir': parentDir,
                                'projectDir': projectDir,
                                'analysis folder': 'NGSolveMEVP',
                                'cell_type': 'mid cell',
                                'optimisation': False
                                }

            uq_cell_complexity = 'simplecell'
            if 'cell_complexity' in uq_config.keys():
                uq_cell_complexity = uq_config['cell_complexity']

            if uq_cell_complexity == 'multicell':
                shape_space = {name: shape_multi}
                uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict, uq_config)
            else:
                shape_space = {name: shape}
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')

        done(f'Done with Cavity {name}. Time: {time.time() - start_time}')

    def set_wall_material(self, wm):
        self.wall_material = wm

    def optimise(self, optimisation_config, optimiser):
       optimiser(self, optimisation_config)

    def get_ngsolve_tune_res(self):
        """

        Parameters
        ----------
        tune_variable: {'A', 'B', 'a'. 'b', 'Ri', 'L', 'Req'}
            Tune variable.
        cell_type: {'mid cell', 'end-mid cell', 'mid-end cell', 'single cell'}
            Type of cell to tune

        Returns
        -------

        """
        tune_res = 'tune_res.json'
        if os.path.exists(os.path.join(self.self_dir, 'eigenmode', 'tune_res.json')):
            with open(os.path.join(self.self_dir, 'eigenmode', 'tune_res.json'),
                      'r') as json_file:
                self.tune_results = json.load(json_file)
            self.freq = self.tune_results['FREQ']
            # self.shape['IC'] = self.tune_results['IC']
            # self.shape['OC'] = self.tune_results['OC']
            # self.shape['OC_R'] = self.tune_results['OC_R']
            # self.mid_cell = self.shape['IC']
            # self.end_cell_left = self.shape['OC']
            # self.end_cell_right = self.shape['OC_R']

        else:
            error("Tune results not found. Please tune the cavity")

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        qois = 'qois.json'
        assert os.path.exists(
            os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)), (
            error(f"Eigenmode result does not exist {os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)}, please run eigenmode simulation."))
        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole',
                               'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        # with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
        #     self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        self.freq = self.eigenmode_qois['freq [MHz]']
        self.k_cc = self.eigenmode_qois['kcc [%]']
        self.ff = self.eigenmode_qois['ff [%]']
        self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        self.GR_Q = self.eigenmode_qois['GR/Q [Ohm^2]']
        self.G = self.GR_Q / self.R_Q
        self.Q = self.eigenmode_qois['Q []']
        self.e = self.eigenmode_qois['Epk/Eacc []']
        self.b = self.eigenmode_qois['Bpk/Eacc [mT/MV/m]']
        self.Epk_Eacc = self.e
        self.Bpk_Eacc = self.b

    def plot_dispersion(self, ax=None, show_continuous=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        df = pd.DataFrame.from_dict(self.eigenmode_qois_all_modes, orient='index')
        freqs = df['freq [MHz]'][1:self.n_cells + 1]

        k = np.linspace(np.pi / self.n_cells, np.pi, self.n_cells, endpoint=True)
        axis_label = ['$' + f'({i + 1}/{self.n_cells})' + r'\pi$' if i - 1 != self.n_cells else r'$\pi$' for i in
                      range(self.n_cells)]
        ax.plot(k, freqs, marker='o', mec='k', label=self.name, **kwargs)
        ax.set_xticklabels(axis_label)
        ax.set_ylabel('$f$ [MHz]')
        ax.set_xlabel('$k$')

        # if show_continuous:
        #     # calculate continuous
        #     k = np.linspace(0, np.pi, 100, endpoint=True)
        #     f_pi_over_2 = (freqs[0] + freqs[-1])/2
        #     freq_c = np.sqrt(freqs[0]**2 + freqs[-1]**2 - 2*freqs[0]*freqs[-1]*np.cos(k))
        #     ax.plot(k, freq_c, c='k', lw=2)

        return ax

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
            ax.text(
                0.95, 0.05,  # Position (normalized coordinates)
                r'$\eta=' + fr'{self.ff:.2f}\%' + '$',  # Text content
                fontsize=12,  # Font size
                ha='right',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                transform=plt.gca().transAxes  # Use axes-relative positioning
            )
            ax.legend(loc="upper right")
            if show_min_max:
                minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 50, width=100)
                Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                ax.plot(self.Ez_0_abs['z(0, 0)'][peaks], Ez_0_abs_peaks, marker='o', ls='')
                ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                ax.axhline(max(Ez_0_abs_peaks), c='k')
        else:
            if os.path.exists(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')):
                with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
                ax.text(
                    0.95, 0.05,  # Position (normalized coordinates)
                    r'$\eta=$' + fr'{self.ff:.2f}\%' + '$',  # Text content
                    fontsize=12,  # Font size
                    ha='right',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    transform=plt.gca().transAxes  # Use axes-relative positioning
                )
                ax.legend(loc="upper right")
                if show_min_max:
                    minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                    peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 100,
                                          width=100)
                    Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                    ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                    ax.axhline(max(Ez_0_abs_peaks), c='k')
            else:
                error('Axis field plot data not found.')

    def plot_spectra(self, var, ax=None):
        if len(self.uq_fm_results_all_modes) != 0:
            df = pd.DataFrame({k: {sub_k: v[0] for sub_k, v in sub_dict.items()} for k, sub_dict in
                               self.uq_fm_results_all_modes.items()})
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 4))

            # Generate KDE data
            for col in ['freq_1 [MHz]', 'freq_2 [MHz]', 'freq_3 [MHz]', 'freq_4 [MHz]', 'freq_5 [MHz]', 'freq_6 [MHz]',
                        'freq_7 [MHz]', 'freq_8 [MHz]', 'freq_9 [MHz]']:
                mean = df[col]["expe"]
                std = df[col]["stdDev"]
                x_values = np.linspace(mean - 10, mean + 10, 100000)  # Adjust range as needed
                if std / mean < 1e-4:  # set a very narrow range for the x-axis around the mean.
                    x = np.linspace(mean - 10, mean + 10, 100000)
                    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
                    ax.plot(x, y_values * 1e-8, label=col)
                else:
                    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
                    ax.plot(x_values, y_values * 1e-8, label=col)

            # Plot settings
            ax.set_title("KDE Plots for Frequencies")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Density")
            ax.set_xlim(1270, 1310)
            # ax.set_ylim(0, 3)
            ax.legend()

        else:
            error("uq_fm_results_all_modes not found")

    def get_uq_fm_results(self, folder):
        # load uq result
        with open(os.path.join(folder, 'uq.json'), 'r') as json_file:
            self.uq_fm_results = json.load(json_file)

        if os.path.exists(os.path.join(folder, 'uq_all_modes.json')):
            with open(os.path.join(folder, 'uq_all_modes.json'), 'r') as json_file:
                self.uq_fm_results_all_modes = json.load(json_file)

        # get neighbours and all qois
        neighbours = {}
        for dirr in os.listdir(os.path.join(folder, 'Cavities')):
            if 'Q' in dirr.split('_')[-1]:
                with open(os.path.join(folder, 'Cavities', dirr, 'eigenmode', 'monopole', 'qois.json'), 'r') as json_file:
                    neighbour_uq_fm_results = json.load(json_file)
                neighbours[dirr] = neighbour_uq_fm_results

        self.neighbours = pd.DataFrame.from_dict(neighbours, orient='index')

        # nodes, weights = cn_leg_05_2(7)
        # write weights

        # data_table = pd.DataFrame(weights, columns=['weights'])
        # data_table.to_csv(fr'{folder}\weights.csv', index=False, sep='\t', float_format='%.32f')

        # get nodes
        self.uq_nodes = pd.read_csv(os.path.join(folder, 'nodes.csv'), sep='\t')
        # get weights
        self.uq_weights = pd.read_csv(os.path.join(folder, 'weights.csv'))

    def get_uq_hom_results(self, folder):
        with open(folder, 'r') as json_file:
            self.uq_hom_results = json.load(json_file)

    def get_wakefield_qois(self, wakefield_config):
        """
        Get the quantities of interest written by the ABCI code

        Parameters
        ----------
        opt: {'SR', 'BS'}
            SR - Synchrotron radiation bunch length
            BS - Bremsstrahlung

        Returns
        -------

        """

        if os.path.exists(os.path.join(self.self_dir, 'wakefield', 'qois.json')):
            with open(os.path.join(self.self_dir, 'wakefield', 'qois.json')) as json_file:
                all_wakefield_qois = json.load(json_file)

        # get only keys in op_points
        if 'operating_points' in wakefield_config:
            for op_pt in wakefield_config['operating_points'].keys():
                for key, val in all_wakefield_qois.items():
                    if op_pt in key:
                        self.wakefield_qois[key] = val

            for key, val in self.wakefield_qois.items():
                self.k_fm[key] = val['k_FM [V/pC]']
                self.k_loss[key] = val['|k_loss| [V/pC]']
                self.k_kick[key] = val['|k_kick| [V/pC/m]']
                self.phom[key] = val['P_HOM [kW]']
                self.I0[key] = val['I0 [mA]']

    def get_abci_data(self):
        self.abci_data = {'Long': ABCIData(self.wakefield_dir, '', 0),
                          'Trans': ABCIData(self.wakefield_dir, '', 1)}

    def plot_animate_wakefield(self, save=False):
        def plot_contour_for_frame(data, frame_key, ax):
            ax.clear()

            colors = {'DOTS': 'k', 'SOLID': 'k'}

            if frame_key in data:
                for plot_type, df in data[frame_key]:
                    ax.plot(df['X'], df['Y'], linestyle='-', color=colors[plot_type])

            ax.set_xlabel('X-axis (m)')
            ax.set_ylabel('Y-axis (m)')
            ax.set_title(f'Contour Plot for {frame_key}')

        def animate_frames(data):
            fig, ax = plt.subplots(figsize=(18, 4))
            frame_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))  # Sort frames by their order

            def update(frame_key):
                plot_contour_for_frame(data, frame_key, ax)

            ani = FuncAnimation(fig, update, frames=frame_keys, repeat=True, interval=1000)
            plt.close(fig)

            return ani

        top_folder = self.wakefield_dir
        efield_contour = get_wakefield_data(os.path.join(top_folder, 'longitudinal', 'cavity.top'))
        ani = animate_frames(efield_contour)
        display_html(HTML(animate_frames(efield_contour).to_jshtml()))

        if save:
            # Save the animation as an MP4 file
            ani.save(os.path.join(top_folder, f'{self.name}_e_field_animation.mp4'), writer='ffmpeg', dpi=150)

    def get_power_uq(self, rf_config, op_points_list):
        stat_moms = ['expe', 'stdDev', 'skew', 'kurtosis']
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois_uq[op_pt] = {}
            val = self.operating_points[op_pt]
            for kk, vv in val.items():
                if 'sigma' in kk:
                    sig_id = kk.split('_')[-1].split(' ')[0]

                    if 'Eacc [MV/m]' in rf_config.keys():
                        op_field = {'expe': [self.Eacc_rf_config * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    else:
                        op_field = {'expe': [val['Eacc [MV/m]'] * 1e6], 'stdDev': [0]}
                        self.Eacc_rf_config = val['Eacc [MV/m]']

                    if 'V [GV]' in rf_config.keys():
                        v_rf = {'expe': [rf_config['V [GV]'][ii] * 1e9], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    else:
                        v_rf = {'expe': [val['V [GV]'] * 1e9], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}

                    Q0 = {'expe': [self.Q0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    inv_eta = {'expe': [self.inv_eta], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    p_sr = {'expe': [rf_config['SR per turn [MW]'] * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    n_cav = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    p_in = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    p_cryo = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    pdyn = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    pstat = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    p_wp = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}

                    # test
                    n_cav['expe'][0] = int(np.ceil(v_rf['expe'][0] / (self.Eacc_rf_config * 1e6 * self.l_active)))

                    # p_in = rf_config['SR per turn [MW]'] * 1e6 / n_cav * 1e-3  # maximum synchrotron radiation per beam

                    p_cryo_n = 8 / (np.sqrt(self.neighbours['freq [MHz]'] / 500))  # W/m

                    pdyn_n = v_rf['expe'][0] * (self.Eacc_rf_config * 1e6 * self.l_active) / (
                            self.neighbours['R/Q [Ohm]'] * self.Q0 * n_cav['expe'][0])  # per cavity
                    pdyn_expe, pdyn_std, pdyn_skew, pdyn_kurtosis = weighted_mean_obj(
                        np.atleast_2d(pdyn_n.to_numpy()).T, self.uq_weights)
                    pdyn = {'expe': pdyn_expe, 'stdDev': pdyn_std, 'skew': pdyn_skew, 'kurtosis': pdyn_kurtosis}

                    pstat_n = (self.l_cavity * v_rf['expe'][0] / (
                            self.l_active * self.Eacc_rf_config * 1e6 * n_cav['expe'][0])) * p_cryo_n
                    pstat_expe, pstat_std, pstat_skew, pstat_kurtosis = weighted_mean_obj(
                        np.atleast_2d(pstat_n.to_numpy()).T, self.uq_weights)
                    pstat = {'expe': pstat_expe, 'stdDev': pstat_std, 'skew': pstat_skew, 'kurtosis': pstat_kurtosis}

                    p_wp_n = self.inv_eta * (pdyn_n + pstat_n) * 1e-3  # per cavity
                    p_wp_expe, p_wp_std, p_wp_skew, p_wp_kurtosis = weighted_mean_obj(
                        np.atleast_2d(p_wp_n.to_numpy()).T, self.uq_weights)
                    p_wp = {'expe': p_wp_expe, 'stdDev': p_wp_std, 'skew': p_wp_skew, 'kurtosis': p_wp_kurtosis}

                    for stat_mom in stat_moms:
                        if op_field[stat_mom][0] != 0:
                            # n_cav[stat_mom] = [
                            #     int(np.ceil(v_rf[stat_mom][0] / (op_field[stat_mom][0] * self.l_active)))]

                            p_in[stat_mom] = [
                                p_sr[stat_mom][0] / n_cav[stat_mom][0] * 1e-3]  # maximum synchrotron radiation per beam

                        # if self.uq_fm_results['freq [MHz]'][stat_mom][0] != 0:
                        #     p_cryo[stat_mom] = [
                        #         8 / (np.sqrt(self.uq_fm_results['freq [MHz]'][stat_mom][0] / 500))]  # W/m
                        #
                        # if self.uq_fm_results['R/Q [Ohm]'][stat_mom][0] * Q0[stat_mom][0] * n_cav[stat_mom][0] != 0:
                        #     pdyn[stat_mom] = [v_rf[stat_mom][0] * (op_field[stat_mom][0] * self.l_active) / (
                        #             self.uq_fm_results['R/Q [Ohm]'][stat_mom][0] * Q0[stat_mom][0] *
                        #             n_cav[stat_mom][0])]  # per cavity
                        #
                        # if op_field[stat_mom][0] != 0 and n_cav[stat_mom][0] != 0:
                        #     pstat[stat_mom] = [(self.l_cavity * v_rf[stat_mom][0] / (
                        #             self.l_active * op_field[stat_mom][0] * n_cav[stat_mom][0])) * p_cryo[stat_mom][
                        #                            0]]
                        #
                        # if inv_eta[stat_mom][0] != 0:
                        #     p_wp[stat_mom] = [
                        #         (inv_eta[stat_mom][0]) * (pdyn[stat_mom][0] + pstat[stat_mom][0]) * 1e-3]  # per cavity

                    self.rf_performance_qois_uq[op_pt][sig_id] = {
                        r"Ncav": n_cav,
                        r"Q0 []": Q0,
                        r"Pstat/cav [W]": pstat,
                        r"Pdyn/cav [W]": pdyn,
                        r"Pwp/cav [kW]": p_wp,
                        r"Pin/cav [kW]": p_in,
                        r"PHOM/cav [kW]": self.uq_hom_results[fr'P_HOM [kW]_{op_pt}_{sig_id}_{vv}mm']
                    }
        return self.rf_performance_qois_uq

    def get_power(self, rf_config, op_points_list):
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois[op_pt] = {}
            val = self.operating_points[op_pt]
            for kk, vv in val.items():
                if 'sigma' in kk:
                    sig_id = kk.split('_')[-1].split(' ')[0]

                    if 'Eacc [MV/m]' in rf_config.keys():
                        op_field = self.Eacc_rf_config * 1e6
                    else:
                        op_field = val['Eacc [MV/m]'] * 1e6
                        self.Eacc_rf_config = op_field

                    if 'V [GV]' in rf_config.keys():
                        v_rf = rf_config['V [GV]'][ii] * 1e9
                    else:
                        v_rf = val['V [GV]'] * 1e9

                    Q0 = self.Q0
                    inv_eta = self.inv_eta
                    p_sr = rf_config['SR per turn [MW]'] * 1e6

                    n_cav = int(np.ceil(v_rf / (op_field * self.l_active)))
                    p_in = p_sr / n_cav  # maximum synchrotron radiation per beam

                    p_cryo = 8 / (np.sqrt(self.freq / 500))

                    pdyn = v_rf * (op_field * self.l_active) / (self.R_Q * Q0 * n_cav)

                    pstat = (self.l_cavity * v_rf / (self.l_active * op_field * n_cav)) * p_cryo  # per cavity
                    p_wp = (inv_eta) * (pdyn + pstat)  # per cavity

                    self.rf_performance_qois[op_pt][sig_id] = {
                        r"Ncav": n_cav,
                        r"Q0 []": Q0,
                        r"Pstat/cav [W]": pstat,
                        r"Pdyn/cav [W]": pdyn,
                        r"Pwp/cav [kW]": p_wp * 1e-3,
                        r"Pin/cav [kW]": p_in * 1e-3,
                        r"PHOM/cav [kW]": self.phom[fr'{op_pt}_{sig_id}_{vv}mm']
                    }
        return self.rf_performance_qois

    def get_uq_post(self, qoi):
        pass

    def get_fields(self, mode=1):
        gfu_E, gfu_H = ngsolve_mevp.load_fields(os.path.join(self.self_dir,
                                                             'eigenmode', 'monopole'),
                                                mode)
        return gfu_E, gfu_H

    def plot(self, what, ax=None, scale_x=1, **kwargs):
        if what.lower() == 'geometry':
            if 'mid_cell' in kwargs.keys():
                new_kwargs = {key: val for key, val in kwargs.items() if key != 'mid_cell'}
                ax = write_cavity_geometry_cli(self.mid_cell, self.mid_cell, self.mid_cell,
                                               'none', 1, scale=1, ax=ax, plot=True, **new_kwargs)
            elif 'end_cell_left' in kwargs.keys():
                ax = write_cavity_geometry_cli(self.end_cell_left, self.end_cell_left, self.end_cell_left,
                                               'left', 1, scale=1, ax=ax, plot=True, **kwargs)
            elif 'end_cell_right' in kwargs.keys():
                ax = write_cavity_geometry_cli(self.end_cell_right, self.end_cell_right, self.end_cell_right,
                                               'right', 1, scale=1, ax=ax, plot=True, **kwargs)
            else:
                ax = write_cavity_geometry_cli(self.mid_cell, self.end_cell_left, self.end_cell_right,
                                               self.beampipe, self.n_cells, scale=1, ax=ax, plot=True, **kwargs)
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel(r"$r$ [mm]")
            return ax

        if what.lower() == 'zl':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Long.)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Long.)', **kwargs)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Trans.)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Trans.)', **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
            return ax

        if what.lower() == 'wpl':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Long'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Longitudinal wake potentials)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Long'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Longitudinal wake potentials)', **kwargs)

            ax.set_xlabel('Distance from Bunch Head S [m]')
            ax.set_ylabel(r"Scaled Wake Potentials $W (S)$ [V/pC]")
            return ax
        if what.lower() == 'wpt':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Transversal wake potentials)', lw=3, **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)

                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Transversal wake potentials)', lw=3, **kwargs)
            ax.set_xlabel('Distance from Bunch Head S [m]')
            ax.set_ylabel(r"Scaled Wake Potentials $W (S)$ [V/pC/m]")
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

    def plot_mesh(self, plotter='ngsolve'):
        mesh_path = os.path.join(self.self_dir, 'eigenmode', 'monopole')

        if os.path.exists(mesh_path):
            ngsolve_mevp.plot_mesh(mesh_path, plotter=plotter)
        else:
            error('The folder or mesh does not exist. Please check that a mesh file was writtten.')

    def plot_fields(self, mode=1, which='E', plotter='ngsolve'):
        field_path = os.path.join(self.self_dir, 'eigenmode', 'monopole')

        if os.path.exists(field_path):
            ngsolve_mevp.plot_fields(field_path, mode, which, plotter)
        else:
            error('The foler or field file does not exist. Please check that a mesh file was writtten.')

    def _plot_convergence(self, ax):
        keys = list(ax.keys())
        # plot convergence
        conv_filepath = os.path.join(self.self_dir, 'eigenmode', "tune_convergence.json")
        if os.path.exists(conv_filepath):
            with open(conv_filepath, 'r') as f:
                convergence_dict = json.load(f)
            if len(convergence_dict) > 0:
                x, y = convergence_dict[list(convergence_dict.keys())[0]], convergence_dict['freq [MHz]']
                ax[keys[0]].scatter(x, y, ec='k', label=self.name)

                # plot directions
                for i in range(len(x) - 1):
                    dx = x[i + 1] - x[i]
                    dy = y[i + 1] - y[i]
                    ax[keys[0]].quiver(x[i], y[i], dx, dy, ls='--', angles='xy',
                                       scale_units='xy', scale=1, color='red',
                                       width=0.005, units='width', headwidth=3, headlength=5,
                                       headaxislength=4)

        # plot absolute error
        abs_err_filepath = os.path.join(self.self_dir, 'eigenmode', "tune_absolute_error.json")
        abs_err_dict = {}
        if os.path.exists(conv_filepath):
            with open(abs_err_filepath, 'r') as f:
                abs_err_dict = json.load(f)
        if len(abs_err_dict) > 0:
            ax[keys[1]].plot(abs_err_dict['abs_err'], marker='o', mec='k', label=self.name)

        ax[keys[0]].set_xlabel('Parameter')
        ax[keys[0]].set_ylabel(r"Value")
        ax[keys[1]].set_xlabel('Iteration')
        ax[keys[1]].set_ylabel(r"Absolute error")
        ax[keys[1]].set_yscale('log')

    def define_operating_points(self, op):
        self.operating_points = op

    def inspect(self, cell_type='mid-cell', variation=0.2):

        if cell_type == 'mid-cell':
            cell = self.shape['IC']
        elif cell_type == 'end-cell-left':
            cell = self.shape['OC']
        elif cell_type == 'end-cell-right':
            cell = self.shape['OC_R']
        else:
            cell = self.shape['IC']

        if self.cell_parameterisation == 'flattop':
            A_, B_, a_, b_, Ri_, L_, Req_, l_ = cell[:8]

            # Define the function that plots the graph
            def plot_cavity_geometry_flattop(A, B, a, b, Ri, L, Req, l):
                cell = np.array([A, B, a, b, Ri, L, Req, l])
                write_cavity_geometry_cli_flattop(cell, cell, cell, BP='none',
                                                  n_cell=1, tangent_check=True, lw=1,
                                                  plot=True,
                                                  ignore_degenerate=True)

                # Update the sum display
                sum_label.value = f'Sum of A + a + l: {A + a + l:.2f}, L: {L}, delta: {A + a + l - L}'

            # Create sliders for each variable
            A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
                                           description='A')
            B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
                                           description='B')
            a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
                                           description='a')
            b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
                                           description='b')
            Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
                                            description='Ri')
            L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
                                           description='L')
            Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
                                             value=Req_,
                                             description='Req')
            l_slider = widgets.FloatSlider(min=(1 - variation) * l_, max=(1 + variation) * l_, step=0.1, value=l_,
                                           description='l')
            # Create a label to display the sum of A + a
            sum_label = Label()

            # Arrange the sliders in a 3x3 layout
            ui = VBox([
                HBox([A_slider, B_slider, a_slider]),
                HBox([b_slider, Ri_slider, L_slider]),
                HBox([Req_slider, l_slider]),
                sum_label  # Add the sum label to the layout
            ])

            # Create an interactive widget to update the plot
            out = widgets.interactive_output(plot_cavity_geometry_flattop,
                                             {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
                                              'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider, 'l': l_slider})

        else:
            A_, B_, a_, b_, Ri_, L_, Req_ = cell[:7]

            # Define the function that plots the graph
            def plot_cavity_geometry(A, B, a, b, Ri, L, Req):
                cell = np.array([A, B, a, b, Ri, L, Req])
                write_cavity_geometry_cli(cell, cell, cell, BP='none', n_cell=1,
                                          tangent_check=True, lw=1,
                                          plot=True,
                                          ignore_degenerate=True)

                # Update the sum display
                sum_label.value = f'Sum of A + a: {A + a:.2f}, L: {L}, delta: {A + a - L}'

            # Create sliders for each variable
            A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
                                           description='A')
            B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
                                           description='B')
            a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
                                           description='a')
            b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
                                           description='b')
            Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
                                            description='Ri')
            L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
                                           description='L')
            Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
                                             value=Req_,
                                             description='Req')
            # Create a label to display the sum of A + a
            sum_label = Label()

            # Arrange the sliders in a 3x3 layout
            ui = VBox([
                HBox([A_slider, B_slider, a_slider]),
                HBox([b_slider, Ri_slider, L_slider]),
                HBox([Req_slider]),
                sum_label  # Add the sum label to the layout
            ])

            # Create an interactive widget to update the plot
            out = widgets.interactive_output(plot_cavity_geometry,
                                             {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
                                              'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider})

        # Display the layout
        display(out, ui)

    def config_sample(self, kind):
        if kind == 'eigenmode':
            return EIGENMODE_CONFIG

        if kind == 'wakefield':
            return WAKEFIELD_CONFIG

        if kind == 'tune':
            return TUNE_CONFIG

        if kind == 'uq':
            return UQ_CONFIG

        if kind == 'optimisation':
            pass

        if kind == 'sa':
            return
    # def inspect(self, cell_type='mid-cell', variation=0.2, tangent_check=False):
    #
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.set_aspect('equal')
    #
    #     if cell_type == 'mid-cell':
    #         cell = self.shape['IC']
    #     elif cell_type == 'end-cell-left':
    #         cell = self.shape['OC']
    #     elif cell_type == 'end-cell-right':
    #         cell = self.shape['OC_R']
    #     else:
    #         cell = self.shape['IC']
    #
    #     if self.cell_parameterisation == 'flattop':
    #         A_, B_, a_, b_, Ri_, L_, Req_, l_ = cell[:8]
    #
    #         # Define the function that plots the graph
    #         def plot_cavity_geometry_flattop(A, B, a, b, Ri, L, Req, l):
    #             cell = np.array([A, B, a, b, Ri, L, Req, l])
    #             write_cavity_geometry_cli_flattop(cell, cell, cell, BP='none',
    #                                               n_cell=1, tangent_check=True, lw=1,
    #                                               plot=True,
    #                                               ignore_degenerate=True)
    #
    #             # Update the sum display
    #             sum_label.value = f'Sum of A + a + l: {A + a + l:.2f}, L: {L}, delta: {A + a + l - L}'
    #
    #         # Create sliders for each variable
    #         A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
    #                                        description='A')
    #         B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
    #                                        description='B')
    #         a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
    #                                        description='a')
    #         b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
    #                                        description='b')
    #         Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
    #                                         description='Ri')
    #         L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
    #                                        description='L')
    #         Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
    #                                          value=Req_,
    #                                          description='Req')
    #         l_slider = widgets.FloatSlider(min=(1 - variation) * l_, max=(1 + variation) * l_, step=0.1, value=l_,
    #                                        description='l')
    #         # Create a label to display the sum of A + a
    #         sum_label = Label()
    #
    #         # Arrange the sliders in a 3x3 layout
    #         ui = VBox([
    #             HBox([A_slider, B_slider, a_slider]),
    #             HBox([b_slider, Ri_slider, L_slider]),
    #             HBox([Req_slider, l_slider]),
    #             sum_label  # Add the sum label to the layout
    #         ])
    #
    #         # Create an interactive widget to update the plot
    #         out = widgets.interactive_output(plot_cavity_geometry_flattop,
    #                                          {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
    #                                           'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider, 'l': l_slider})
    #
    #     else:
    #         A_, B_, a_, b_, Ri_, L_, Req_ = cell[:7]
    #
    #         # Define the function that plots the graph
    #         def plot_cavity_geometry(A, B, a, b, Ri, L, Req):
    #             cell = np.array([A, B, a, b, Ri, L, Req])
    #             # ax.clear()
    #             write_cavity_geometry_cli(cell, cell, cell, BP='none', n_cell=1,
    #                                       tangent_check=tangent_check, lw=3,
    #                                       plot=True, ax=ax,
    #                                       ignore_degenerate=True)
    #             # Update the sum display
    #             sum_label.value = f'Sum of A + a: {A + a:.2f}, L: {L}, delta: {A + a - L}'
    #
    #         def run_eigenmode(b):
    #             eigenmode_config = {'processes': 1}
    #             boundary_conds = 'mm'
    #             eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[boundary_conds]
    #
    #             shape_space = {}
    #
    #             # run_eigenmode_s({self.name: self.shape}, {self.name: self.shape_multicell}, self.projectDir, eigenmode_config)
    #
    #         # Create sliders for each variable
    #         A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
    #                                        description='A')
    #         B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
    #                                        description='B')
    #         a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
    #                                        description='a')
    #         b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
    #                                        description='b')
    #         Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
    #                                         description='Ri')
    #         L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
    #                                        description='L')
    #         Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
    #                                          value=Req_,
    #                                          description='Req')
    #         # Create a label to display the sum of A + a
    #         sum_label = Label()
    #
    #         # create run tune and run eigenmode button
    #         button_tune = widgets.Button(description="Tune")
    #         button_eigenmode = widgets.Button(description="Eigenmode")
    #         button_eigenmode.on_click(run_eigenmode)
    #
    #         # Arrange the sliders in a 3x3 layout
    #         ui = VBox([
    #             HBox([A_slider, B_slider, a_slider]),
    #             HBox([b_slider, Ri_slider, L_slider]),
    #             HBox([Req_slider]),
    #             sum_label,  # Add the sum label to the layout
    #             # HBox([button_tune, button_eigenmode])
    #         ])
    #
    #         # Create an interactive widget to update the plot
    #         out = widgets.interactive_output(plot_cavity_geometry,
    #                                          {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
    #                                           'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider})
    #
    #     # Display the layout
    #     display(out, ui)

    def eval_expr(self, expr, symbols):
        """Evaluate numeric expression with symbols."""
        node = ast.parse(expr, mode='eval').body

        def _eval(n):
            if isinstance(n, ast.Num): return n.n
            if isinstance(n, ast.Name): return symbols[n.id]
            if isinstance(n, ast.BinOp): return _ops[type(n.op)](_eval(n.left), _eval(n.right))
            if isinstance(n, ast.UnaryOp): return _ops[type(n.op)](_eval(n.operand))
            raise ValueError(f"Unsupported expression: {expr}")

        return _eval(node)

    # Main converter
    def geo_to_abc(self, wakefield_config=None, folder=None, **kwargs):
        # Read input
        with open(self.geo_filepath) as f:
            lines = [l.split('%', 1)[0].strip() for l in f]

        # 1) Build symbol table from DefineNumber
        symbols = {}
        define_re = re.compile(r"(\w+)\s*=\s*DefineNumber\[\s*([^,\]]+)")
        for line in lines:
            m = define_re.search(line)
            if m:
                symbols[m.group(1)] = m.group(2).strip()
        # Evaluate until fixed
        changed = True
        while changed:
            changed = False
            for k, v in list(symbols.items()):
                if isinstance(v, str):
                    try:
                        symbols[k] = self.eval_expr(v, symbols)
                        changed = True
                    except:
                        pass

        # 2) Parse geometry segments
        point_re = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\}")
        ellipse_re = re.compile(r"Ellipse\(\d+\)\s*=\s*\{([^}]+)\}")

        points = {}  # pid -> (r,z)
        segments = []  # list of ('point', r, z) or ('ellipse', end_pid)

        # First pass: collect points structs and z-values
        zvals = []
        for line in lines:
            m = point_re.match(line)
            if m:
                pid = int(m.group(1))
                coords = [p.strip() for p in m.group(2).split(',')]
                z = self.eval_expr(coords[0], symbols)
                r = self.eval_expr(coords[1], symbols)
                points[pid] = (r, z)
                zvals.append(z)

        zmin = min(zvals) if zvals else 0.0

        # Second pass: build segment list
        for line in lines:
            if line.startswith('Point('):
                m = point_re.match(line)
                pid = int(m.group(1))
                r, z = points[pid]
                segments.append(('point', r, z - zmin))
            elif line.startswith('Line('):
                continue
            elif line.startswith('Ellipse('):
                m = ellipse_re.match(line)
                ids = [int(x.strip()) for x in m.group(1).split(',')]
                # remove last 3 point segments
                if len(segments) >= 3:
                    segments = segments[:-3]
                # add ellipse indicator + endpoint
                mid_pid = ids[1]
                end_pid = ids[3]
                # segments.append(('ellipse', mid_pid))
                segments.append(('ellipse', mid_pid, end_pid))

        # Create geometry if not crated
        self.create()

        # 3) Write output
        wakefield_folder_structure = {
            'wakefield': {
                'longitudinal': None,
                'transversal': None
            }
        }
        if folder:
            make_dirs_from_dict(wakefield_folder_structure, folder)
        else:
            make_dirs_from_dict(wakefield_folder_structure, self.self_dir)
            folder = self.self_dir

        MROT = wakefield_config['polarisation']
        if  MROT == 2:
            for m in range(2):
                self._write_abc(points, segments, zmin, symbols, m, wakefield_config, folder, **kwargs)
        else:
            self._write_abc(points, segments, zmin, symbols, MROT, wakefield_config, folder, **kwargs)

    def _write_abc(self, points, segments, zmin, symbols, MROT, wakefield_config, folder, **kwargs):
        # defaults
        RDRIVE, ISIG = 5e-3, 5
        LCRBW = 'F'  # counter-rotating beam
        ZSEP = 0.0
        BSEP = 0
        NBUNCH = 1
        BETA = 1
        LMATPR = '.F.'
        LPRW, LPPW, LSVW, LSVWA, LSVWT, LSVWL, LSVF = '.T.', '.T.', '.T.', '.F.', '.T.', '.T.', '.F.'
        LSAV, LCPUTM = 'F', 'F'
        LCBACK = '.T.'
        LPLE = '.F.'
        NSHOT = 0
        UBT = 50
        SIG = 25e-3
        MT = 4
        mesh_DDR = 0.00125
        mesh_DDZ = 0.00125

        # unpack kwargs
        for key, value in kwargs.items():
            if key == 'RADIAL BEAM OFFSET AT (RDRIVE)':
                RDRIVE = value
            if key == 'NUMBER OF WAKE POTENTIAL POINTS (NW)':
                NW = value
            if key == 'WAKE FOR A COUNTER-ROTATING BEAM (LCRBW)':
                LCRBW = value
            if key == 'VELOCITY OF THE BUNCH / C (BETA)':
                BETA = value
            if key == 'PRINTOUT OF CAVITY SHAPE USED (LMATPR)':
                LMATPR = value
            if key == 'PRINTOUT OF WAKE POTENTIALS (LPRW)':
                LPRW = value
            if key == 'LINE-PRINTER PLOT OF WAKE POT. (LPPW)':
                LPPW = value
            if key == 'SAVE WAKE POTENTIALS IN A FILE (LSVW)':
                LSVW = value
            if key == 'SAVE AZIMUTHAL WAKE IN A FILE (LSVWA)':
                LSVWA = value
            if key == 'SAVE TRANSVERSE WAKE IN A FILE (LSVWT)':
                LSVWT = value
            if key == 'SAVE LONGITUDINAL WAKE IN A FILE (LSVWL)':
                LSVWL = value
            if key == 'SAVE FFT RESULTS IN A FILE (LSVF)':
                LSVF = value
            if key == 'SAVE FIELDS INTO FILE (LSAV)':
                LSAV = value
            if key == 'CPUTIME MONITOR ACTIVE (LCPUTM)':
                LCPUTM = value

        if 'save_fields' in wakefield_config.keys():
            LPLE, LCBACK = 'T', 'F'
            if isinstance(wakefield_config['save_fields'], dict):
                if 'nshot' in wakefield_config['save_fields'].keys():
                    NSHOT = wakefield_config['save_fields']['nshot']

        if 'wake_config' in wakefield_config.keys():
            wake_config = wakefield_config['wake_config']
            if 'MT' in wake_config.keys():
                MT = wake_config['MT']
            if 'counter_rotating'in wakefield_config['wake_config'].keys():
                LCRBW = 'T'
                if 'separation' in wakefield_config['wake_config']['counter_rotating'].keys():
                    ZSEP = wakefield_config['wake_config']['counter_rotating']['separation']

        if 'beam_config' in wakefield_config.keys():
            if 'beam_offset' in wakefield_config['beam_config'].keys():
                RDRIVE = wakefield_config['beam_config']['beam_offset']
            if 'nbunch' in wakefield_config['beam_config'].keys():
                NBUNCH = wakefield_config['beam_config']['nbunch']
            if 'separation' in wakefield_config['beam_config'].keys():
                BSEP = wakefield_config['beam_config']['separation']

        with open(os.path.join(folder, 'wakefield', MROT_DICT[MROT], 'cavity.abc'), 'w') as out:
            # Header
            out.write(f' &FILE LSAV = .{LSAV}., ITEST = 0, LREC = .F., LCPUTM = .{LCPUTM}. &END \n')
            out.write(' SAMPLE INPUT #1 A SIMPLE CAVITY STRUCTURE \n')
            out.write(' &BOUN  IZL = 3, IZR = 3  &END \n')
            out.write(f' &MESH DDR = {mesh_DDR}, DDZ = {mesh_DDZ} &END \n')
            out.write(" #CAVITYSHAPE\n0.\n")

            # Body
            for seg in segments:
                if seg[0] == 'point':
                    _, r, z = seg
                    out.write(f"{r:.16f} {z:.16f}\n")
                elif seg[0] == 'ellipse':
                    _, pid_m, pid_e = seg
                    r_m, z_m = points[pid_m]
                    r_e, z_e = points[pid_e]
                    out.write("-3., 0.000\n")
                    out.write(f"{r_m:.16f} {z_m - zmin:.16f}\n")
                    out.write(f"{r_e:.16f} {z_e - zmin:.16f}\n")

            # Closing
            out.write("0.000 0.000\n9999. 9999.\n")

            # wakefield simulation paramters
            out.write(f' &BEAM  SIG = {SIG}, ISIG = {ISIG}, RDRIVE = {RDRIVE}, MROT = {MROT}, NBUNCH = {NBUNCH}, BSEP = {BSEP} &END \n')
            # out.write(' &BEAM  SIG = {}, MROT = {}, RDRIVE = {}  &END \n'.format(SIG, MROT, 0.005))
            out.write(f' &TIME  MT = {int(MT)} &END \n')
            out.write(
                f' &WAKE  UBT = {int(UBT)}, LCRBW = .{LCRBW}., LCBACK = {LCBACK}, LCRBW = .{LCRBW}., ZSEP = {ZSEP} &END \n')  # , NFS = {NFS}
            # f.write(' &WAKE  UBT = {}, LCHIN = F, LNAPOLY = F, LNONAP = F &END \n'.format(UBT, wake_offset))
            # f.write(' &WAKE R  = {}   &END \n'.format(wake_offset))
            out.write(f' &PLOT  LCAVIN = .T., LCAVUS = .F., LPLW = .T., LFFT = .T., LSPEC = .T., '
                    f'LINTZ = .F., LPATH = .T., LPLE = {LPLE}, LPLC= .F. &END \n')
            out.write(f' &PRIN  LMATPR = {LMATPR}, LPRW = {LPRW}, LPPW = {LPPW}, LSVW = {LSVW}, '
                    f'LSVWA = {LSVWA}, LSVWT = {LSVWT}, LSVWL = {LSVWL},  LSVF = {LSVF}   &END\n')
            out.write('\nSTOP\n')

    def to_multicell(self):
        mid_cell = self.shape['IC']

        self.shape_multicell = {}
        mid_cell_multi = np.array([[[a, a] for _ in range(self.n_cells - 1)] for a in mid_cell])

        self.shape_multicell['OC'] = self.shape['OC']
        self.shape_multicell['OC_R'] = self.shape['OC_R']
        self.shape_multicell['IC'] = mid_cell_multi
        self.shape_multicell['BP'] = self.shape['BP']
        self.shape_multicell['n_cells'] = self.shape['n_cells']
        self.shape_multicell['CELL PARAMETERISATION'] = 'multicell'
        self.shape_multicell['kind'] = self.kind
        self.shape_multicell['geo_file'] = None

    def _create_project(self, overwrite):
        project_name = self.name
        project_dir = self.projectDir

        if project_name != '':

            # check if folder already exist
            e = self._check_if_path_exists(project_dir, project_name, overwrite)

            if e:
                def make_dirs_from_dict(d, current_dir=fr"{project_dir}"):
                    for key, val in d.items():
                        os.mkdir(os.path.join(current_dir, key))
                        if type(val) == dict:
                            make_dirs_from_dict(val, os.path.join(current_dir, key))

                # create project structure in folders
                project_dir_structure = {
                    f'{project_name}':
                        {
                            'Cavities': None,
                            'OperatingPoints': None,
                            'PostData': {
                                'Plots': None,
                                'Data': None,
                            },
                            'Reference': None
                        }
                }
                try:
                    make_dirs_from_dict(project_dir_structure)
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    return True
                except Exception as e:
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    error("An exception occurred in created project: ", e)
                    return False
            else:
                self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                return True
        else:
            info('\tPlease enter a valid project name')
            self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
            return False

    @staticmethod
    def _check_if_path_exists(directory, folder, overwrite=False):
        path = f"{directory}/{folder}"
        if os.path.exists(path):
            if overwrite:
                x = 'y'
            else:
                x = 'n'

            if x == 'y':
                try:
                    directory_list = os.listdir(path)

                    if 'Cavities' in directory_list \
                            and len(directory_list) < 6:
                        shutil.rmtree(path)
                        return True
                    else:
                        info('\tIt seems that the folder specified is not a cavity project folder. Please check folder'
                             'again to avoid deleting important files.')
                        return False

                except Exception as e:
                    error("Exception occurred: ", e)
                    return False
            else:
                return False
        else:
            return True

    def _overwriteFolder(self, invar, projectDir, name):
        path = os.path.join(self.self_dir, 'eigenmode', f'_process_{invar}')
        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}

        os.makedirs(path)

    @staticmethod
    def _copyFiles(invar, parentDir, projectDir, name):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, 'Cavities', name, 'eigenmode', f'_process_{invar}', 'SLANS_exe')

        dir_util.copy_tree(src, dst)

    def __str__(self):
        p = dict()
        p[self.name] = {
            'tune': self.tune_results,
            'fm': self.eigenmode_qois,
            'hom': self.wakefield_qois,
            'uq': {
                'tune': 0,
                'fm': self.uq_fm_results,
                'hom': self.uq_hom_results
            }
        }
        return fr"{json.dumps(p, indent=4)}"

    # @abstractmethod
    def spawn(self, difference, folder):
        spawn = Cavities(folder)
        for key, params_diff in difference.iterrows():
            # load base geometry
            base_geo = self.geo_filepath

            name = key
            scav = Cavity(name=name, geo_filepath=os.path.join(self.self_dir, 'geometry', 'geodata.geo'))
            spawn.add_cavity(scav, names=name, plot_labels=name)

            # modify base_geo parameters according to params_diff
            output_folder = os.path.join(scav.self_dir, "geometry")
            self.update_geo_parameters(base_geo, output_folder, params_diff.to_dict())

        return spawn

    def update_geo_parameters(self, input_geo, output_folder, update_params):
        with open(input_geo, "r") as f:
            content = f.read()

        for name, new_value in update_params.items():
            # Pattern matches DefineNumber[...] form, capturing the part before and after the number
            pattern = rf"(\b{name}\s*=\s*DefineNumber\[)\s*[^,\]]+"
            replacement = rf"\g<1>{new_value}"
            content, n = re.subn(pattern, replacement, content)

            if n == 0:
                print(f"Warning: parameter '{name}' not found in {os.path.basename(input_geo)}")
                info(fr"Here are available parameters: {(self.parameters.keys())}")

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(input_geo))

        with open(output_path, "w") as f:
            f.write(content)

        # print(f"Updated .geo file written to: {output_path}")
        return output_path


