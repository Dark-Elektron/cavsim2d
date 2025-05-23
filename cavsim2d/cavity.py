import copy
import fnmatch
import os.path
import random
import shutil
import subprocess
from paretoset import paretoset
import sys
from distutils import dir_util
from math import floor
import matplotlib
import psutil
import scipy.signal as sps
from matplotlib.animation import FuncAnimation
from scipy.interpolate import griddata
from IPython.core.display import HTML, Image, display_html, Math
from IPython.core.display_functions import display
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull, Delaunay
from scipy.special import jn_zeros, jnp_zeros
import matplotlib as mpl
import scipy.io as spio
import scipy.interpolate as sci
import pandas as pd
from scipy.stats import qmc
from scipy.special import *
from tqdm.auto import tqdm
import time
import datetime
# import oapackage
import multiprocessing as mp
from cavsim2d.analysis.tune.tuner import Tuner
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.analysis.wakefield.abci_geometry import ABCIGeometry
from cavsim2d.utils.shared_functions import *
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label

ngsolve_mevp = NGSolveMEVP()
abci_geom = ABCIGeometry()
tuner = Tuner()

SOFTWARE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))  # str(Path().parents[0])
CUSTOM_COLORS = ['#4b8f63', '#fc6d2d', '#6a7bbf', '#e567a7', '#8cd839', '#ff5f00', '#d1a67a', '#a3a3a3']
VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6, 'l': 7}
TUNE_ACCURACY = 1e-4
DIMENSION = 'm'
DIMENSION_FACTOR = {'mm': 1, 'cm': 1e-1, 'm': 1e-3}
BOUNDARY_CONDITIONS_DICT = {'ee': 11, 'em': 13, 'me': 31, 'mm': 33}
LABELS = {'freq [MHz]': r'$f$ [MHz]', 'R/Q [Ohm]': r"$R/Q ~\mathrm{[\Omega]}$",
          "Epk/Eacc []": r"$E_\mathrm{pk}/E_\mathrm{acc} ~[\cdot]$",
          "Bpk/Eacc [mT/MV/m]": r"$B_\mathrm{pk}/E_\mathrm{acc} ~\mathrm{[mT/MV/m]}$",
          "G [Ohm]": r"$G ~\mathrm{[\Omega]}$", "Q []": r'$Q$ []',
          'kcc [%]': r'$k_\mathrm{cc}$ [%]', 'GR/Q [Ohm^2]': '$G \cdot R/Q \mathrm{[\Omega^2]}$',
          'ff [%]': r'$\eta_ff$ [%]',
          'k_FM [V/pC]': r"$|k_\mathrm{FM}| ~\mathrm{[V/pC]}$",
          '|k_loss| [V/pC]': r"$|k_\parallel| ~\mathrm{[V/pC]}$",
          '|k_kick| [V/pC/m]': r"$|k_\perp| ~\mathrm{[V/pC/m]}$",
          'P_HOM [kW]': r"$P_\mathrm{HOM}/\mathrm{cav} ~\mathrm{[kW]}$",
          'Z_2023': 'Z', 'W_2023': 'W', 'H_2023': 'H', 'ttbar_2023': r'$\mathrm{t \bar t}$',
          'Z_b_2024': 'Z$_\mathrm{b}$', 'W_b_2024': 'W$_\mathrm{b}$',
          'H_b_2024': 'H$_\mathrm{b}$', 'ttbar_b_2024': r'$\mathrm{t \bar t}_\mathrm{b}$',
          'Z_b_2024_FB': 'Z$_\mathrm{b}$[FB]', 'W_b_2024_FB': 'W$_\mathrm{b}$[FB]',
          'H_b_2024_FB': 'H$_\mathrm{b}$[FB]', 'ttbar_b_2024_FB': r'$\mathrm{t \bar t}_\mathrm{b}$[FB]',
          r"Ncav": r"$N_\mathrm{cav}$",
          r"Q0 []": r"$Q_0 ~\mathrm{[]}$",
          r"Pstat/cav [W]": r"$P_\mathrm{stat}$/cav [W]",
          r"Pdyn/cav [W]": r"$P_\mathrm{dyn}$/cav [W]",
          r"Pwp/cav [kW]": r"$P_\mathrm{wp}$/cav [kW]",
          r"Pin/cav [kW]": r"$P_\mathrm{in}$/cav [kW]",
          r"PHOM/cav [kW]": r"$P_\mathrm{HOM}$/cav [kW]"
          }

m0 = 9.1093879e-31
q0 = 1.6021773e-19
c0 = 2.99792458e8
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12


class Optimisation:

    def __init__(self):
        self.poc = 0
        self.eigenmode_config = {}
        self.mid_cell = None
        self.wakefield_config = None
        self.tune_config = None
        self.f2_interp = None
        self.processes_count = None
        self.method = None
        self.mutation_factor = None
        self.crossover_factor = None
        self.elites_to_crossover = None
        self.chaos_factor = None
        self.tune_parameter = None
        self.uq_config = None
        self.constraints = []
        self.df = None
        self.df_global = None
        self.objs_dict = None
        self.constraints_dict = None
        self.n_interp = None
        self.interp_error = None
        self.interp_error_avg = None
        self.cell_type = None
        self.bounds = None
        self.weights = None
        self.objective_vars = None
        self.objectives = None
        self.objectives_unprocessed = None
        self.ng_max = None
        self.tune_freq = None
        self.initial_points = None
        self.projectDir = None
        self.parentDir = None
        self.pareto_history = None
        self.optimisation_config = None
        self.err = None

    def start_optimisation(self, projectDir, config):
        self.err = []
        self.pareto_history = []
        self.optimisation_config = config
        # apply optimisation settings
        self.parentDir = os.getcwd()
        self.projectDir = projectDir
        self.initial_points = config['initial_points']
        self.ng_max = config['no_of_generation']
        self.objectives_unprocessed = config['objectives']
        self.objectives, weights = process_objectives(config['objectives'])
        self.objective_vars = [obj[1] for obj in self.objectives]
        if 'weights' in config.keys():
            self.weights = config['weights']
            assert len(self.weights) == len(weights), \
                ("Length of delta must be equal to the length of the variables. For impedance Z entries, one less than"
                 "the length of the interval list weights are needed. Eg. for ['min', 'ZL', [1, 2, 3]], two weights are"
                 " required. ")
        else:
            self.weights = weights

        self.bounds = config['bounds']
        if 'constraints' in config:
            self.constraints = self.process_constraints(config['constraints'])
        self.processes_count = 1
        if 'processes' in config.keys():
            assert config['processes'] > 0, error('Number of processes must be greater than zero!')
            assert isinstance(config['processes'], int), error('Number of processes must be integer!')
            self.processes_count = config['processes']

        self.method = config['method']
        self.mutation_factor = config['mutation_factor']
        self.crossover_factor = config['crossover_factor']
        self.elites_to_crossover = config['elites_for_crossover']
        self.chaos_factor = config['chaos_factor']

        self.tune_config = config['tune_config']
        tune_config_keys = self.tune_config.keys()
        assert 'freqs' in tune_config_keys, error('Please enter the target tune frequency.')
        assert 'parameters' in tune_config_keys, error('Please enter the tune variable in tune_config_dict')
        assert 'cell_types' in tune_config_keys, error('Please enter the cell_type in tune_config_dict')

        self.cell_type = self.tune_config['cell_types']
        cts = ['end-cell', 'mid-end-cell', 'end-mid-cell', 'end_cell', 'mid_end_cell', 'end_mid_cell']
        if self.cell_type in cts:
            assert 'mid-cell' in config.keys(), error('To optimise an end-cell, mid cell dimensions are required')
            assert len(config['mid-cell']) >= 7, error('Incomplete mid cell dimension.')
            self.mid_cell = config['mid-cell']

        self.tune_parameter = self.tune_config['parameters']
        self.tune_freq = self.tune_config['freqs']

        self.wakefield_config = {}
        if (any(['ZL' in obj for obj in self.objective_vars])
                or any(['ZT' in obj for obj in self.objective_vars])
                or any([obj in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]'] for obj in
                        self.objective_vars])):
            assert 'wakefield_config' in config.keys(), error('Wakefield impedance objective detected in objectives. '
                                                              'Please include a field for wakefield_config. An empty'
                                                              ' config entry implies that default values will be used.')
            self.wakefield_config = config['wakefield_config']

            if 'uq_config' in self.wakefield_config.keys():
                self.uq_config = self.wakefield_config['uq_config']
                # replace objectives with processed objectives
                self.wakefield_config['uq_config']['objectives'] = self.objectives
                self.wakefield_config['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed

                if self.uq_config['delta']:
                    assert len(self.uq_config['delta']) == len(self.uq_config['variables']), error(
                        "The number of deltas must "
                        "be equal to the number of "
                        "variables.")

        if 'eigenmode_config' in config.keys():
            self.eigenmode_config = config['eigenmode_config']

            if 'uq_config' in self.eigenmode_config.keys():
                self.uq_config = self.eigenmode_config['uq_config']
                if self.uq_config['delta']:
                    assert len(self.uq_config['delta']) == len(self.uq_config['variables']), error(
                        "The number of deltas must "
                        "be equal to the number of "
                        "variables.")

        self.df = None

        # interpolation
        self.df_global = pd.DataFrame()
        self.objs_dict = {}
        self.constraints_dict = {}
        self.n_interp = 10000
        self.interp_error = []
        self.interp_error_avg = []
        bar = tqdm(total=self.ng_max)
        self.ea(0, bar)

    def ea(self, n, bar):
        if n == 0:
            # update lists
            self.df = self.generate_first_men(self.initial_points, 0)

            self.f2_interp = [np.zeros(self.n_interp) for _ in range(len(self.objectives))]

            folders = [os.path.join(self.projectDir, 'SimulationData', 'Optimisation'),
                       os.path.join(self.projectDir, 'SimulationData', 'Optimisation')]

            # clear folder to avoid reading from previous optimization attempt
            for folder in folders:
                if os.path.exists(folder):
                    for filename in os.listdir(folder):
                        try:
                            shutil.rmtree(os.path.join(folder, filename))
                        except NotADirectoryError:
                            os.remove(os.path.join(folder, filename))
                else:
                    os.mkdir(folder)

        # optimize by page rank
        # remove the lowest ranking members
        df = self.df

        # compare with global dict and remove duplicates
        compared_cols = ['A', 'B', 'a', 'b', 'Ri']
        if not self.df_global.empty:
            df = df.loc[~df.set_index(compared_cols).index.isin(
                self.df_global.set_index(compared_cols).index)]  # this line of code removes duplicates

        pseudo_shape_space = {}
        for index, row in df.iterrows():
            rw = row.tolist()

            if self.cell_type.lower() == 'mid cell' or self.cell_type.lower() == 'mid-cell' or self.cell_type.lower() == 'mid_cell':
                pseudo_shape_space[rw[0]] = {'IC': rw[1:], 'OC': rw[1:], 'OC_R': rw[1:], 'BP': 'none',
                                             'FREQ': self.tune_freq, 'n_cells': 1,
                                             'CELL PARAMETERISATION': 'simplecell'}

            elif self.cell_type.lower() == 'mid-end cell' or self.cell_type.lower() == 'mid-end-cell' or self.cell_type.lower() == 'mid_end_cell':

                assert 'mid cell' in list(self.optimisation_config.keys()), \
                    ("If cell_type is set as 'mid-end cell', the mid cell geometry parameters must "
                     "be provided in the optimisation_config dictionary.")
                assert len(self.optimisation_config['mid cell']) > 6, ("Incomplete mid cell geometry parameter. "
                                                                       "At least 7 geometric parameters "
                                                                       "[A, B, a, b, Ri, L, Req] required.")
                IC = self.optimisation_config['mid cell']
                # check if mid-cell is not a degenerate geometry
                df = tangent_coords(*np.array(IC)[0:8], 0)
                assert df[-2] == 1, ("The mid-cell geometry dimensions given result in a degenerate geometry. "
                                     "Please check.")

                pseudo_shape_space[rw[0]] = {'IC': IC, 'OC': rw[1:], 'OC_R': rw[1:], 'BP': 'right',
                                             'FREQ': self.tune_freq, 'n_cells': 1,
                                             'CELL PARAMETERISATION': 'simplecell'}

            elif (self.cell_type.lower() == 'end-end cell' or self.cell_type.lower() == 'end-end-cell'
                  or self.cell_type.lower() == 'end_end_cell') or self.cell_type.lower() == 'end end cell':

                pseudo_shape_space[rw[0]] = {'IC': rw[1:], 'OC': rw[1:], 'OC_R': rw[1:], 'BP': 'right',
                                             'FREQ': self.tune_freq, 'n_cells': 1,
                                             'CELL PARAMETERISATION': 'simplecell'}

            else:
                pseudo_shape_space[rw[0]] = {'IC': rw[1:], 'OC': rw[1:], 'OC_R': rw[1:], 'BP': 'both',
                                             'FREQ': self.tune_freq, 'n_cells': 1,
                                             'CELL PARAMETERISATION': 'simplecell'}

        pseudo_shape_space = self.remove_duplicate_values(pseudo_shape_space)

        ############################
        # run tune
        n_cells = 1

        # self.run_tune_parallel(pseudo_shape_space, n_cells)
        self.run_tune_opt(pseudo_shape_space, self.tune_config)
        # get successfully tuned geometries and filter initial generation dictionary
        processed_keys = []
        tune_result = []
        for key in pseudo_shape_space.keys():
            filename = self.projectDir / fr'SimulationData\Optimisation\{key}\tune_res.json'
            try:
                with open(filename, 'r') as file:
                    tune_res = json.load(file)

                # get only tune_variable, alpha_i, and alpha_o and freq but why these quantities
                freq = tune_res['FREQ']
                tune_variable_value = tune_res['IC'][VAR_TO_INDEX_DICT[self.tune_parameter]]
                alpha_i = tune_res['IC'][7]
                alpha_o = tune_res['IC'][7]

                tune_result.append([tune_variable_value, alpha_i, alpha_o, freq])
                processed_keys.append(key)
            except FileNotFoundError:
                pass

        # after removing duplicates, dataframe might change size
        df = df.loc[df['key'].isin(processed_keys)]
        df.loc[:, [self.tune_parameter, 'alpha_i', 'alpha_o', 'freq [MHz]']] = tune_result

        # eigen objective variables
        # for o in self.objectives:
        intersection = set(self.objective_vars).intersection(
            ["freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"])
        if len(intersection) > 0:
            # process tune results
            obj_result = []
            processed_keys = []
            for key in pseudo_shape_space.keys():
                filename = self.projectDir / fr'SimulationData\Optimisation\{key}\monopole\qois.json'
                try:
                    with open(filename, 'r') as file:
                        qois = json.load(file)
                    # extract objectives from tune_res
                    obj = list(
                        {key: val for [key, val] in qois.items() if key in self.objective_vars}.values())

                    obj_result.append(obj)
                    # tune_result.append(list(qois.values()))
                    processed_keys.append(key)
                except FileNotFoundError as e:
                    pass

            # after removing duplicates, dataframe might change size
            if len(processed_keys) == 0:
                error("Unfortunately, none survived. \n"
                      "This is most likely due to all generated initial geometries being degenerate.\n"
                      "Check the variable bounds or increase the number of initial geometries to increase the"
                      "changes of survival. \n"
                      "Can't even say that this was a good run."
                      "Tune ended.")
                return

            df = df.loc[df['key'].isin(processed_keys)]

            obj_eigen = [o[1] for o in self.objectives if
                         o[1] in ["freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"]]
            df[obj_eigen] = obj_result

        # for o in self.objectives:
        #     if o[1] in ["mts monopole", 'mts dipole']:
        #         # process tune results
        #         obj_vars = self.ui.ccb_Populate_Objectives.currentText().split(', ')
        #         for i, obj_var in enumerate(obj_vars):
        #             if obj_var == "mts monopole" or obj_var == "mts dipole":
        #                 goal = self.ui.tw_Objectives.cellWidget(i, 1).currentText()
        #                 if goal == 'equal':
        #                     fshift = float(self.ui.tw_Objectives.item(i, 2).text())
        #
        #         obj_result = []
        #         tune_result = []
        #         processed_keys = []
        #         # run dipole simulation with frequency shift
        #         if o[1] == "mts monopole":
        #             slans_shape_space = self.run_slans_parallel(df, n_cells, fshift, 'monopole')
        #             for key, val in slans_shape_space.items():
        #                 filename = self.projectDir / fr'SimulationData\SLANS_opt\{key}\cavity_33.svl'
        #                 try:
        #                     params = fr.svl_reader(filename)
        #                     obj = self.get_objectives_value(params, self.objectives, norm_length, n_cells)
        #
        #                     obj_result.append(obj)
        #
        #                     df_slans_mts = pd.DataFrame(obj, columns=[key, o[1]])
        #                     df = df.merge(df_slans_mts, on='key', how='inner')
        #                 except FileNotFoundError:
        #                     pass
        #         else:
        #             slans_shape_space = self.run_slans_parallel(df, n_cells, fshift, 'dipole')
        #             for key, val in slans_shape_space.items():
        #                 filename = self.projectDir / fr'SimulationData\SLANS_opt\{key}_n1\cavity_33_2.sv2'
        #                 try:
        #                     params = fr.sv2_reader(filename)
        #                     obj = params['Frequency'][-1]
        #                     obj_result.append([key, obj])
        #                     processed_keys.append(key)
        #                 except FileNotFoundError:
        #                     pass
        #
        #             df_slans_mts = pd.DataFrame(obj_result, columns=['key', o[1]])
        #             df = df.merge(df_slans_mts, on='key', how='inner')

        # wakefield objective variables
        for o in self.objectives:
            if "ZL" in o[1] or "ZT" in o[1] or o[1] in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]',
                                                        'P_HOM [kW]']:
                # run wakefield analysis and return shape space
                wake_shape_space = self.run_wakefield_opt(df, self.wakefield_config)

                # process wakefield results
                df_wake, processed_keys = get_wakefield_objectives_value(wake_shape_space,
                                                                         self.objectives_unprocessed,
                                                                         self.projectDir / fr'SimulationData\ABCI')

                df = df.merge(df_wake, on='key', how='inner')
                break

        # apply UQ
        if self.uq_config:

            # get uq_parameters
            uq_result_dict = {}
            for key in df['key']:
                filename_eigen = self.projectDir / fr'SimulationData\Optimisation\{key}\uq.json'
                filename_abci = self.projectDir / fr'SimulationData\ABCI\{key}\uq.json'
                if os.path.exists(filename_eigen):  # and os.path.exists(filename_abci):
                    uq_result_dict[key] = []
                    with open(filename_eigen, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if o[1] in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                                        "G [Ohm]", "Q []"]:
                                uq_result_dict[key].append(uq_d[o[1]]['expe'][0])
                                uq_result_dict[key].append(uq_d[o[1]]['stdDev'][0])
                                if o[0] == 'min':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] + 6 * uq_d[o[1]]['stdDev'][0])
                                elif o[0] == 'max':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] - 6 * uq_d[o[1]]['stdDev'][0])
                                else:
                                    # for equal, calculate |expected_value - design_value| + 6sigma
                                    uq_result_dict[key].append(
                                        np.abs(uq_d[o[1]]['expe'][0] - o[2]) + uq_d[o[1]]['stdDev'][0])

                if os.path.exists(filename_abci):
                    if key not in uq_result_dict:
                        uq_result_dict[key] = []

                    with open(filename_abci, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if o[1] not in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                                            "G [Ohm]", "Q []"]:
                                uq_result_dict[key].append(uq_d[o[1]]['expe'][0])
                                uq_result_dict[key].append(uq_d[o[1]]['stdDev'][0])
                                if o[0] == 'min':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] + 6 * uq_d[o[1]]['stdDev'][0])
                                elif o[0] == 'max':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] - 6 * uq_d[o[1]]['stdDev'][0])

            uq_column_names = []
            for o in self.objectives:
                uq_column_names.append(fr'E[{o[1]}]')
                uq_column_names.append(fr'std[{o[1]}]')
                if o[0] == 'min':
                    uq_column_names.append(fr'E[{o[1]}] + 6*std[{o[1]}]')
                elif o[0] == 'max':
                    uq_column_names.append(fr'E[{o[1]}] - 6*std[{o[1]}]')
                else:
                    uq_column_names.append(fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]')

            df_uq = pd.DataFrame.from_dict(uq_result_dict, orient='index')

            assert len(df_uq) > 0, error('Unfortunately, no geometry was returned from uq, optimisation terminated.')
            df_uq.columns = uq_column_names
            df_uq.index.name = 'key'
            df_uq.reset_index(inplace=True)
            df = df.merge(df_uq, on='key', how='inner')

        # filter by constraints
        for const in self.constraints:
            c = const.split(" ")

            if c[1] == '>':
                df = df.loc[(df[f'{c[0]}'] > float(c[2]))]
            elif c[1] == '<':
                df = df.loc[(df[f'{c[0]}'] < float(c[2]))]
            elif c[1] == '<=':
                df = df.loc[(df[f'{c[0]}'] <= float(c[2]))]
            elif c[1] == '>=':
                df = df.loc[(df[f'{c[0]}'] >= float(c[2]))]
            elif c[1] == '==':
                df = df.loc[(df[f'{c[0]}'] == float(c[2]))]

        # update with global dataframe
        if not self.df_global.empty:
            df = pd.concat([self.df_global, df], ignore_index=True)

        # reset total rank
        df['total_rank'] = 0

        # rank shapes by objectives
        for i, obj in enumerate(self.objectives):

            if self.uq_config:
                if obj[0] == "min":
                    df[f'rank_E[{obj[1]}] + 6*std[{obj[1]}]'] = df[fr'E[{obj[1]}] + 6*std[{obj[1]}]'].rank() * \
                                                                self.weights[i]
                elif obj[0] == "max":
                    df[f'rank_E[{obj[1]}] - 6*std[{obj[1]}]'] = df[fr'E[{obj[1]}] - 6*std[{obj[1]}]'].rank(
                        ascending=False) * self.weights[i]
                elif obj[0] == "equal":
                    df[fr'rank_|E[{obj[1]}] - {obj[2]}| + std[{obj[1]}]'] = df[
                                                                                fr'|E[{obj[1]}] - {obj[2]}| + std[{obj[1]}]'].rank() * \
                                                                            self.weights[i]

                # if 'total_rank' in df.columns:
                if obj[0] == 'min':
                    df[f'total_rank'] = df[f'total_rank'] + df[f'rank_E[{obj[1]}] + 6*std[{obj[1]}]']
                elif obj[0] == 'max':
                    df[f'total_rank'] = df[f'total_rank'] + df[f'rank_E[{obj[1]}] - 6*std[{obj[1]}]']
                else:
                    df[f'total_rank'] = df[f'total_rank'] + df[fr'rank_|E[{obj[1]}] - {obj[2]}| + std[{obj[1]}]']
                # else:
                #     if obj[0] == 'min':
                #         df[f'total_rank'] = df[f'rank_E[{obj[1]}] + 6*std[{obj[1]}]']
                #     elif obj[0] == 'max':
                #         df[f'total_rank'] = df[f'rank_E[{obj[1]}] - 6*std[{obj[1]}]']
                #     else:
                #         df[f'total_rank'] = df[f'rank_std[{obj[1]}]']
            else:
                if obj[0] == "min":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank() * self.weights[i]
                elif obj[0] == "max":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank(ascending=False) * self.weights[i]
                elif obj[0] == "equal" and obj[1] != 'freq [MHz]':  # define properly later
                    df[f'rank_{obj[1]}'] = (df[obj[1]] - obj[2]).abs().rank() * self.weights[i]

                df[f'total_rank'] = df[f'total_rank'] + df[f'rank_{obj[1]}']

        # reorder
        tot = df.pop(f'total_rank')
        df[f'total_rank'] = tot / sum(self.weights)  # normalize by sum of weights

        # order shapes by rank
        df = df.sort_values(by=['total_rank'])
        df = df.reset_index(drop=True)

        # pareto condition
        reorder_indx, pareto_indx_list = self.pareto_front(df)

        # estimate convergence
        obj_error = []
        obj0 = self.objectives[0][1]
        for i, obj in enumerate(self.objectives):
            if i != 0:
                pareto_shapes = df.loc[pareto_indx_list, [obj0, obj[1]]]
                pareto_shapes_sorted = pareto_shapes.sort_values(obj0)
                f1 = np.linspace(min(pareto_shapes[obj0]), max(pareto_shapes[obj0]), self.n_interp)
                f2_interp = np.interp(f1, pareto_shapes_sorted[obj0], pareto_shapes_sorted[obj[1]])
                rel_error = np.linalg.norm(f2_interp - self.f2_interp[i]) / max(np.abs(f2_interp))
                obj_error.append(rel_error)

                self.f2_interp[i] = f2_interp

        # new error
        # stack previous and current pareto fronts
        # if n == 0:
        #     pareto_shapes = df.loc[pareto_indx_list, self.objective_vars]
        #     self.pareto_history.append(pareto_shapes)
        # else:
        #     pareto_shapes = df.loc[pareto_indx_list, self.objective_vars]
        #     pareto_stack = np.vstack([self.pareto_history[-1], pareto_shapes])
        #
        #     # Compute Delaunay triangulation
        #     delaunay = Delaunay(pareto_stack)
        #     simplices = delaunay.simplices
        #
        #     hypervolumes = self.calculate_hypervolumes(pareto_stack, simplices)
        #     self.err.append(sum(hypervolumes))

        if len(obj_error) != 0:
            self.interp_error.append(max(obj_error))
            self.interp_error_avg.append(np.average(self.interp_error))

        df = df.loc[reorder_indx, :]
        # reset index
        df = df.dropna().reset_index(drop=True)

        # update global
        self.df_global = df

        # check if df_global is empty
        if self.df_global.shape[0] == 0:
            error("Unfortunately, none survived the constraints and the program has to end. "
                  "Can't even say that this was a good run.")
            return
        done(self.df_global)

        # save dataframe
        filename = os.path.join(self.projectDir, 'SimulationData', 'Optimisation', 'Generation{n}.xlsx')
        self.recursive_save(self.df_global, filename, reorder_indx)

        # birth next generation
        # crossover
        if len(df) > 1:
            df_cross = self.crossover(df, n, self.crossover_factor)
        else:
            df_cross = pd.DataFrame()

        # mutation
        df_mutation = self.mutation(df, n, self.mutation_factor)

        # chaos
        df_chaos = self.chaos(self.chaos_factor, n)

        # take elites from previous generation over to next generation
        df_ng = pd.concat([df_cross, df_mutation, df_chaos], ignore_index=True)

        # update dictionary
        self.df = df_ng

        n += 1
        info("=" * 80)
        if n < self.ng_max:
            bar.update(1)
            return self.ea(n, bar)
        else:
            bar.update(1)
            end = datetime.datetime.now()
            info("End time: ", end)
            plt.plot(self.interp_error, marker='P', label='max error')
            plt.plot(self.interp_error_avg, marker='X', label='avereage')
            plt.plot([x + 1 for x in range(len(self.err))], self.err, marker='o', label='convex hull vol')
            plt.yscale('log')
            plt.legend()

            plt.xlabel('Generation $n$')
            plt.ylabel(r"Pareto surface interp. error")
            plt.show()
            return

    def run_uq(self, df, objectives, solver_dict, solver_args_dict, uq_config):
        """

        Parameters
        ----------
        df: pd.DataFrame
            Pandas dataframe containing cavity geometry parameters
        objectives: list|ndarray
            List of objective functions
        solver_dict: dict
            Python dictionary of solver settings
        solver_args_dict: dict
            Python dictionary of solver arguments
        uq_config:
            Python dictionary of uncertainty quantification settings

        Returns
        -------

        """
        proc_count = uq_config['processes']

        # get geometric parameters
        df = df.loc[:, ['key', 'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', "alpha_i", "alpha_o"]]
        shape_space = {}

        df = df.set_index('key')
        for index, row in df.iterrows():
            rw = row.tolist()

            if self.cell_type.lower() == 'mid cell' or self.cell_type.lower() == 'mid-cell' or self.cell_type.lower() == 'mid_cell':
                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw}

            elif self.cell_type.lower() == 'mid-end cell' or self.cell_type.lower() == 'mid-end-cell' or self.cell_type.lower() == 'mid_end_cell':

                assert 'mid cell' in list(self.optimisation_config.keys()), \
                    ("If cell_type is set as 'mid-end cell', the mid cell geometry parameters must "
                     "be provided in the optimisation_config dictionary.")
                assert len(self.optimisation_config['mid cell']) > 6, ("Incomplete mid cell geometry parameter. "
                                                                       "At least 7 geometric parameters "
                                                                       "[A, B, a, b, Ri, L, Req] required.")

                IC = self.optimisation_config['mid cell']
                # check if mid-cell is not a degenerate geometry
                df = tangent_coords(*np.array(IC)[0:8], 0)
                assert df[-2] == 1, ("The mid-cell geometry dimensions given result in a degenerate geometry. "
                                     "Please check.")
                shape_space[f'{index}'] = {'IC': IC, 'OC': rw, 'OC_R': rw}

            elif (self.cell_type.lower() == 'end-end cell' or self.cell_type.lower() == 'end-end-cell'
                  or self.cell_type.lower() == 'end_end_cell') or self.cell_type.lower() == 'end end cell':

                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw}

            else:
                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw}

        if solver_args_dict['eigenmode']:
            if 'uq_config' in solver_args_dict['eigenmode'].keys():
                solver_args_dict['eigenmode']['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')

        if solver_args_dict['wakefield']:
            if 'uq_config' in solver_args_dict['wakefield'].keys():
                solver_args_dict['wakefield']['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'wakefield')

        return shape_space

    @staticmethod
    def calculate_hypervolumes(points, simplices):
        volumes = []
        for simplex in simplices:
            hull = ConvexHull(points[simplex])
            volumes.append(hull.volume)
        return volumes

    def plot_pareto(self, vars, which='last'):
        ################## plot 2d ##################################
        grid_results_folder = fr'D:\Dropbox\CavityDesignHub\KWT_simulations\PostprocessingData\Data\grid_results.xlsx'
        fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(12, 3))

        columns_array = [['Epk/Eacc', 'Bpk/Eacc'], ['Epk/Eacc', 'R/Q'], ['Bpk/Eacc', 'R/Q']]
        columns_par_array = [['A', 'B'], ['a', 'b'], ['A', 'Ri']]
        cmap = matplotlib.colormaps['Pastel2_r']
        norm = matplotlib.colors.Normalize(vmin=0, vmax=49)
        ff = fr'D:\Dropbox\CavityDesignHub\Cavity800\SimulationData'
        for i, (columns, columns_par) in enumerate(zip(columns_array, columns_par_array)):
            for r in [49]:
                for lab, opt_code_result_folder, error_file in zip(['LHS', 'LHS2', 'Random'],
                                                                   [fr'{ff}\SLANS_LHS\Generation{r}.xlsx',
                                                                    fr'{ff}\SLANS_LHS2\Generation{r}.xlsx',
                                                                    fr'{ff}\SLANS_kwt_random1\Generation{r}.xlsx'],
                                                                   [fr'{ff}\SLANS_LHS2\inerp_error_and_average.txt',
                                                                    fr'{ff}\SLANS_LHS2\inerp_error_and_average.txt',
                                                                    fr'{ff}\SLANS_LHS2\inerp_error_and_average.txt']):
                    opt_code_result = pd.read_excel(opt_code_result_folder, 'Sheet1')

                    pareto_shapes = self.pareto_front(opt_code_result, columns, axs[i], show='none',
                                                      label=f"{lab}: g{r} ($n$={len(opt_code_result.index)}).",
                                                      kwargs_dataframe={'facecolors': 'none', 'edgecolor': 'b'},
                                                      kwargs_pareto={'marker': 'o', 'mec': 'k', 'ms': 3},
                                                      # kwargs_pareto={'c': f'{matplotlib.colors.rgb2hex(cmap(norm(r)))}', 'marker': 'o',
                                                      #                'mec': 'k'}
                                                      )
                    # axs[i+3].plot(grid_results[columns[0]], grid_results[columns[1]], marker='o', ms=5, lw=0)
                    axs[i + 3].plot(opt_code_result[columns[0]], opt_code_result[columns[1]], marker='o', ms=5, lw=0)
                    # axs[i+3].plot(qmc_pareto_shapes[columns[0]], qmc_pareto_shapes[columns[1]], marker='o', c='r', label='qmc', ms=5, lw=0)
                    axs[i + 3].plot(pareto_shapes[columns[0]], pareto_shapes[columns[1]], marker='o', c='b', mec='k',
                                    label='ea', ms=5, lw=0)

                    # # load interpolation error
                    # error = pd.read_csv(error_file, header=None, sep='\\s+')
                    # axs[3].plot(error[0], label=lab)
                    # axs[4].plot(error[1], label=lab)

            axs[i].set_xlabel(columns[0])
            axs[i].set_ylabel(columns[1])
            axs[i + 3].set_xlabel(columns_par[0])
            axs[i + 3].set_ylabel(columns_par[1])
        lines, labels = axs[0].get_legend_handles_labels()
        # axs[i].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=10, loc='lower left', mode='expand')
        fig.legend(*axs[0].get_legend_handles_labels(), loc="upper left", mode="expand", ncol=4)
        axs[3].legend()

        axs[3].set_yscale('log')
        axs[4].set_yscale('log')
        axs[3].set_xlabel('Interpolation error')
        # axs[3].set_ylabel()
        # plot error

        plt.tight_layout()
        plt.show()

        # ################### plot surface #########################
        # grid_results_folder = fr'D:\Dropbox\CavityDesignHub\KWT_simulations\PostprocessingData\Data\grid_results.xlsx'
        # opt_code_result_folder_lhs = fr'D:\Dropbox\CavityDesignHub\Cavity800\SimulationData\SLANS_opt_KWT\Generation49.xlsx'
        # opt_code_result_folder_random = fr'D:\Dropbox\CavityDesignHub\Cavity800\SimulationData\SLANS\Generation49.xlsx'
        #
        # grid_results = pd.read_excel(grid_results_folder, 'Sheet1')
        # opt_code_result_lhs = pd.read_excel(opt_code_result_folder_lhs, 'Sheet1')
        # opt_code_result_random = pd.read_excel(opt_code_result_folder_random, 'Sheet1')
        #
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        # ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        # # ax3 = fig.add_subplot(1, 2, 3, projection='3d')
        # axs = [ax1, ax2]
        #
        # grid_pareto_surface = plot_pareto_surface(grid_results, ['Epk/Eacc', 'Bpk/Eacc', 'R/Q'], axs[0],
        #                                           {'cmap': 'gray', 'edgecolor': 'k'})
        # opt_pareto_surface_lhs = plot_pareto_surface(opt_code_result_lhs, ['Epk/Eacc', 'Bpk/Eacc', 'R/Q'], axs[1],
        #                                              {'cmap': 'RdBu', 'edgecolor': 'k'})
        # # opt_pareto_surface_random = plot_pareto_surface(opt_code_result_random, ['Epk/Eacc', 'Bpk/Eacc', 'R/Q'], axs[2],
        # #                                                 {'cmap': 'RdBu', 'edgecolor': 'k'})
        #
        # plt.show()

        # fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(12, 5.5))
        # ani = animation.FuncAnimation(fig=fig, func=create_evolution_animation, frames=49, interval=1000)
        # ani.save(filename="D:\Dropbox\Quick presentation files/ffmpeg_example.gif", writer="pillow")
        # # plt.show()

    # def pareto_front_(self, df, columns, ax=None, show='all', label='', kwargs_dataframe=None, kwargs_pareto=None):
    #     if kwargs_dataframe is None:
    #         kwargs_dataframe = {}
    #     if kwargs_pareto is None:
    #         kwargs_pareto = {}
    #
    #     datapoints = df.loc[:, columns] * (-1)
    #     pareto = oapackage.ParetoDoubleLong()
    #     # ic(datapoints)
    #     for ii in range(0, datapoints.shape[0]):
    #         w = oapackage.doubleVector(tuple(datapoints.iloc[ii].values))
    #         pareto.addvalue(w, ii)
    #     # pareto.show(verbose=1)  # Prints out the results from pareto
    #
    #     lst = pareto.allindices()  # the indices of the Pareto optimal designs
    #     poc = len(lst)  # number of pareto shapes
    #     reorder_idx = list(lst) + [i for i in range(len(df)) if
    #                                i not in lst]  # reordered index putting pareto shapes first
    #
    #     pareto_shapes = df.loc[lst, :]
    #     # sort pareto shapes in ascending x axis data
    #     pareto_shapes = pareto_shapes.sort_values(by=[columns[0]])
    #
    #     if ax:
    #         if show == 'all':
    #             ax.scatter(df[columns[0]], df[columns[1]], **kwargs_dataframe)
    #
    #         ax.plot(pareto_shapes[columns[0]], pareto_shapes[columns[1]], **kwargs_pareto, label=label)
    #
    #     return pareto_shapes

    def plot_pareto_surface(self, df, columns, ax, kwargs=None):
        if kwargs is None:
            kwargs = {}
        pareto = self.pareto_front(df, columns)

        x, y, z = pareto['Epk/Eacc []'], pareto['Bpk/Eacc'], pareto['R/Q']
        xi, yi = np.meshgrid(np.linspace(min(x), max(x), 100),
                             np.linspace(min(y), max(y), 100))
        zi = griddata((x, y), z, (xi, yi), method='cubic')
        surf = ax.plot_surface(xi, yi, zi, antialiased=False, **kwargs)

        return surf

    # def create_evolution_animation(self, frame):
    #     for ax_index in axs:
    #         axs[ax_index].clear()
    #
    #     ################### plot 2d ##################################
    #     grid_results_folder = fr'D:\Dropbox\CavityDesignHub\KWT_simulations\PostprocessingData\Data\grid_results.xlsx'
    #
    #     columns_array = [['Epk/Eacc', 'Bpk/Eacc'], ['Epk/Eacc', 'R/Q'], ['Bpk/Eacc', 'R/Q']]
    #     columns_par_array = [['A', 'B'], ['a', 'b'], ['A', 'Ri']]
    #     cmap = matplotlib.colormaps['Pastel2_r']
    #     norm = matplotlib.colors.Normalize(vmin=0, vmax=49)
    #
    #     for i, (columns, columns_par) in enumerate(zip(columns_array, columns_par_array)):
    #         grid_results = pd.read_excel(grid_results_folder, 'Sheet1')
    #         qmc_pareto_shapes = pareto_front(grid_results, columns, axs[i], show='pareto',
    #                                          label=f'QMC \n({len(grid_results.index)} geoems.)',
    #                                          kwargs_dataframe={'facecolors': 'none', 'edgecolor': 'grey'},
    #                                          kwargs_pareto={'c': 'k', 'marker': 'o', 'mec': 'k'})
    #
    #         for r in [frame]:
    #             for lab, opt_code_result_folder, error_file in zip(['LHS2'],
    #                                                                [
    #                                                                    fr'D:\Dropbox\CavityDesignHub\Cavity800\SimulationData\SLANS\Generation{r}.xlsx'],
    #                                                                [
    #                                                                    fr'D:\Dropbox\CavityDesignHub\Cavity800\SimulationData\SLANS_LHS2\inerp_error_and_average.txt']):
    #                 opt_code_result = pd.read_excel(opt_code_result_folder, 'Sheet1')
    #
    #                 pareto_shapes = pareto_front(opt_code_result, columns, axs[i], show='pareto',
    #                                              label=f"{lab}: G{r} \n({len(opt_code_result.index)} geoms).",
    #                                              kwargs_dataframe={'facecolors': 'none', 'edgecolor': 'b'},
    #                                              kwargs_pareto={'marker': 'o',
    #                                                             'mec': 'k'},
    #                                              # kwargs_pareto={'c': f'{matplotlib.colors.rgb2hex(cmap(norm(r)))}', 'marker': 'o',
    #                                              #                'mec': 'k'}
    #                                              )
    #                 axs[i + 3].scatter(grid_results[columns_par[0]], grid_results[columns_par[1]], s=5)
    #                 axs[i + 3].scatter(opt_code_result[columns_par[0]], opt_code_result[columns_par[1]], s=5)
    #                 axs[i + 3].scatter(qmc_pareto_shapes[columns_par[0]], qmc_pareto_shapes[columns_par[1]], c='r',
    #                                    label='qmc', s=5)
    #                 axs[i + 3].scatter(pareto_shapes[columns_par[0]], pareto_shapes[columns_par[1]], c='b',
    #                                    edgecolor='k',
    #                                    label='ea', s=5)
    #
    #                 # load interpolation error
    #                 error = pd.read_csv(error_file, header=None, sep='\\s+')
    #                 # axs[3].plot(error[0], label=lab)
    #                 # axs[4].plot(error[1], label=lab)
    #
    #         axs[i].set_xlabel(columns[0])
    #         axs[i].set_ylabel(columns[1])
    #         axs[i + 3].set_xlabel(columns_par[0])
    #         axs[i + 3].set_ylabel(columns_par[1])
    #
    #     # axs[i].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=10, loc='lower left', mode='expand')
    #     axs[0].legend()
    #     axs[3].legend()
    #
    #     # axs[3].set_yscale('log')
    #     # axs[4].set_yscale('log')
    #     # axs[3].set_xlabel('Interpolation error')
    #     # axs[3].set_ylabel()
    #     # plot error
    #
    #     # plt.tight_layout()
    #     # plt.show()

    @staticmethod
    def stroud(p):
        # Stroud-3 method
        #
        # Input parameters:
        #  p   number of dimensions
        # Output parameters:
        #  nodes   nodes of quadrature rule in [0,1]^p (column-wise)
        #

        nodes = np.zeros((p, 2 * p))
        coeff = np.pi / p
        fac = np.sqrt(2 / 3)

        for i in range(2 * p):
            for r in range(int(np.floor(0.5 * p))):
                k = 2 * r
                nodes[k, i] = fac * np.cos((k + 1) * (i + 1) * coeff)
                nodes[k + 1, i] = fac * np.sin((k + 1) * (i + 1) * coeff)

            if 0.5 * p != np.floor(0.5 * p):
                nodes[-1, i] = ((-1) ** (i + 1)) / np.sqrt(3)

        # transform nodes from [-1,+1]^p to [0,1]^p
        nodes = 0.5 * nodes + 0.5

        return nodes

    def quad_stroud3(self, rdim, degree):
        # data for Stroud-3 quadrature in [0,1]^k
        # nodes and weights
        nodes = self.stroud(rdim)
        nodestr = 2. * nodes - 1.
        weights = (1 / (2 * rdim)) * np.ones((2 * rdim, 1))

        # evaluation of Legendre polynomials
        bpoly = np.zeros((degree + 1, rdim, 2 * rdim))
        for l in range(rdim):
            for j in range(2 * rdim):
                bpoly[0, l, j] = 1
                bpoly[1, l, j] = nodestr[l, j]
                for i in range(1, degree):
                    bpoly[i + 1, l, j] = ((2 * (i + 1) - 1) * nodestr[l, j] * bpoly[i, l, j] - i * bpoly[
                        i - 1, l, j]) / (i + 1)

        # standardisation of Legendre polynomials
        for i in range(1, degree + 1):
            bpoly[i, :, :] = bpoly[i, :, :] * np.sqrt(2 * (i + 1) - 1)

        return nodes, weights, bpoly

    def weighted_mean_obj(self, tab_var, weights):
        rows_sims_no, cols = np.shape(tab_var)
        no_weights, dummy = np.shape(weights)  # z funckji quadr_stroud wekt columnowy

        if rows_sims_no == no_weights:
            expe = np.zeros((cols, 1))
            outvar = np.zeros((cols, 1))
            for i in range(cols):
                expe[i, 0] = np.dot(tab_var[:, i], weights)
                outvar[i, 0] = np.dot(tab_var[:, i] ** 2, weights)
            stdDev = np.sqrt(outvar - expe ** 2)
        else:
            expe = 0
            stdDev = 0
            error('Cols_sims_no != No_weights')

        return list(expe.T[0]), list(stdDev.T[0])

    def run_tune_opt(self, pseudo_shape_space, tune_config):
        tune_config_keys = tune_config.keys()
        freqs = tune_config['freqs']
        tune_parameters = tune_config['parameters']
        cell_types = tune_config['cell_types']

        # perform all necessary checks
        if 'processes' in tune_config.keys():
            processes = tune_config['processes']
            assert processes > 0, error('Number of proceses must be greater than zero.')
        else:
            processes = 1

        if isinstance(freqs, float) or isinstance(freqs, int):
            freqs = np.array([freqs for _ in range(len(pseudo_shape_space))])
        else:
            assert len(freqs) == len(pseudo_shape_space), error(
                'Number of target frequencies must correspond to the number of cavities')
            freqs = np.array(freqs)

        if isinstance(tune_parameters, str):
            assert tune_config['parameters'] in ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req'], error(
                'Please enter a valid tune parameter')
            tune_variables = np.array([tune_parameters for _ in range(len(pseudo_shape_space))])
            cell_types = np.array([cell_types for _ in range(len(pseudo_shape_space))])
        else:
            assert len(tune_parameters) == len(pseudo_shape_space), error(
                'Number of tune parameters must correspond to the number of cavities')
            assert len(cell_types) == len(pseudo_shape_space), error(
                'Number of cell types must correspond to the number of cavities')
            tune_variables = np.array(tune_parameters)
            cell_types = np.array(cell_types)

        run_tune_parallel(pseudo_shape_space, tune_config, self.projectDir, solver='NGSolveMEVP')

    def run_wakefield_opt(self, df, wakefield_config):
        # # get analysis parameters
        # n_cells = 5
        # n_modules = 1
        #
        # # change later
        # WG_M = ['']  # half length of beam pipe between cavities in module
        #
        # # change all of these later
        # MROT = 2  # run both longitudinal and transverse wakefield analysis
        # MT = 4  # number of time steps for a beam to move one cell to another default = 3
        # bunch_length = 25
        # NFS = 10000  # Number of samples in FFT (max 10000)
        # UBT = 50  # Wakelength in m
        # DDZ_SIG = 0.1
        # DDR_SIG = 0.1
        # proc_count = self.processes_count

        # get geometric parameters

        wakefield_config_keys = wakefield_config.keys()
        MROT = 2
        MT = 10
        NFS = 10000
        wakelength = 50
        bunch_length = 25
        DDR_SIG = 0.1
        DDZ_SIG = 0.1

        # check inputs
        if 'bunch_length' in wakefield_config_keys:
            assert not isinstance(wakefield_config['beam_config']['bunch_length'], str), error(
                'Bunch length must be of type integer or float.')
        else:
            wakefield_config['beam_config']['bunch_length'] = bunch_length
        if 'wakelength' in wakefield_config_keys:
            assert not isinstance(wakefield_config['wake_config']['wakelength'], str), error(
                'Wakelength must be of type integer or float.')
        else:
            wakefield_config['wake_config']['wakelength'] = wakelength

        processes = 1
        if 'processes' in wakefield_config.keys():
            assert wakefield_config['processes'] > 0, error('Number of proceses must be greater than zero.')
            processes = wakefield_config['processes']
        else:
            wakefield_config['processes'] = processes

        rerun = True
        if 'rerun' in wakefield_config_keys:
            if isinstance(wakefield_config['rerun'], bool):
                rerun = wakefield_config['rerun']

        if 'polarisation' in wakefield_config_keys:
            assert wakefield_config['polarisation'] in [0, 1, 2], error('Polarisation should be 0 for longitudinal, '
                                                                        '1 for transverse, 2 for both.')
        else:
            wakefield_config['polarisation'] = MROT

        if 'MT' in wakefield_config_keys:
            assert isinstance(wakefield_config['MT'], int), error('MT must be integer between 4 and 20, with 4 and 20 '
                                                                  'included.')
        else:
            wakefield_config['MT'] = MT

        if 'NFS' in wakefield_config_keys:
            assert isinstance(wakefield_config['NFS'], int), error('NFS must be integer.')
        else:
            wakefield_config['NFS'] = NFS

        if 'DDR_SIG' not in wakefield_config_keys:
            wakefield_config['mesh_config']['DDR_SIG'] = DDR_SIG

        if 'DDZ_SIG' not in wakefield_config_keys:
            wakefield_config['mesh_config']['DDZ_SIG'] = DDZ_SIG

        df = df.loc[:, ['key', 'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', "alpha_i", "alpha_o"]]
        shape_space = {}

        df = df.set_index('key')
        for index, row in df.iterrows():
            rw = row.tolist()
            if self.cell_type.lower() == 'end-mid cell':

                A_i, B_i, a_i, b_i, Ri_i, L_i, Req_i = self.mid_cell

                IC = [A_i, B_i, a_i, b_i, Ri_i, L_i, Req_i]

                shape_space[f'{index}'] = {'IC': IC, 'OC': rw, 'OC_R': rw, 'n_cells': 1, 'BP': 'both',
                                           'CELL PARAMETERISATION': 'simplecell'}
            else:
                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw, 'n_cells': 1, 'BP': 'both',
                                           'CELL PARAMETERISATION': 'simplecell'}

        shape_space_multicell = {}
        for key, shape in shape_space.items():
            shape_space_multicell[key] = to_multicell(1, shape)

        run_wakefield_parallel(shape_space, shape_space_multicell, wakefield_config,
                               self.projectDir, marker='', rerun=rerun)

        return shape_space

    def generate_first_men(self, initial_points, n):

        if list(self.method.keys())[0] == "LHS":
            seed = self.method['LHS']['seed']
            if seed == '' or seed is None:
                seed = None

            columns = list(self.bounds.keys())
            dim = len(columns)
            l_bounds = np.array(list(self.bounds.values()))[:, 0]
            u_bounds = np.array(list(self.bounds.values()))[:, 1]

            const_var = []
            for i in range(dim - 1, -1, -1):
                if l_bounds[i] == u_bounds[i]:
                    const_var.append([columns[i], l_bounds[i]])
                    del columns[i]
                    l_bounds = np.delete(l_bounds, i)
                    u_bounds = np.delete(u_bounds, i)

            reduced_dim = len(columns)
            sampler = qmc.LatinHypercube(d=reduced_dim, scramble=False, seed=seed)
            _ = sampler.reset()
            sample = sampler.random(n=initial_points)
            self.discrepancy = qmc.discrepancy(sample)

            sample = qmc.scale(sample, l_bounds, u_bounds)

            df = pd.DataFrame()
            df['key'] = [f"G{n}_C{i}_P" for i in range(initial_points)]
            df[columns] = sample

            for i in range(len(const_var) - 1, -1, -1):
                df[const_var[i][0]] = np.ones(initial_points) * const_var[i][1]

            df['alpha_i'] = np.zeros(initial_points)
            df['alpha_o'] = np.zeros(initial_points)

            return df

        elif list(self.method.keys())[0] == "Sobol Sequence":
            seed = self.method['LHS']['seed']
            if seed == '' or seed is None:
                seed = None

            columns = list(self.bounds.keys())
            dim = len(columns)
            index = self.method["Sobol Sequence"]['index']
            l_bounds = np.array(list(list(self.bounds.values())))[:, 0]
            u_bounds = np.array(list(list(self.bounds.values())))[:, 1]

            const_var = []
            for i in range(dim - 1, -1, -1):
                if l_bounds[i] == u_bounds[i]:
                    const_var.append([columns[i], l_bounds[i]])
                    del columns[i]
                    l_bounds = np.delete(l_bounds, i)
                    u_bounds = np.delete(u_bounds, i)

            reduced_dim = len(columns)
            sampler = qmc.Sobol(d=reduced_dim, scramble=False, seed=seed)
            _ = sampler.reset()
            sample = sampler.random_base2(m=index)
            sample = qmc.scale(sample, l_bounds, u_bounds)

            df = pd.DataFrame()
            df['key'] = [f"G0_C{i}_P" for i in range(initial_points)]
            df[columns] = sample

            for i in range(len(const_var) - 1, -1, -1):
                df[const_var[i][0]] = np.ones(initial_points) * const_var[i][1]

            df['alpha_i'] = np.zeros(initial_points)
            df['alpha_o'] = np.zeros(initial_points)

            return df
        elif list(self.method.keys())[0] == "Random":
            data = {'key': [f"G0_C{i}_P" for i in range(initial_points)],
                    'A': random.sample(list(
                        np.linspace(list(self.bounds.values())[0][0], list(self.bounds.values())[0][1],
                                    initial_points * 2)),
                        initial_points),
                    'B': random.sample(list(
                        np.linspace(list(self.bounds.values())[1][0], list(self.bounds.values())[1][1],
                                    initial_points * 2)),
                        initial_points),
                    'a': random.sample(list(
                        np.linspace(list(self.bounds.values())[2][0], list(self.bounds.values())[2][1],
                                    initial_points * 2)),
                        initial_points),
                    'b': random.sample(list(
                        np.linspace(list(self.bounds.values())[3][0], list(self.bounds.values())[3][1],
                                    initial_points * 2)),
                        initial_points),
                    'Ri': random.sample(list(
                        np.linspace(list(self.bounds.values())[4][0], list(self.bounds.values())[4][1],
                                    initial_points * 2)),
                        initial_points),
                    'L': random.sample(list(
                        np.linspace(list(self.bounds.values())[5][0], list(self.bounds.values())[5][1],
                                    initial_points * 2)),
                        initial_points),
                    'Req': random.sample(list(
                        np.linspace(list(self.bounds.values())[6][0], list(self.bounds.values())[6][1] + 1,
                                    initial_points * 2)),
                        initial_points),
                    'alpha_i': np.zeros(initial_points),
                    'alpha_o': np.zeros(initial_points)}
            return pd.DataFrame.from_dict(data)
        elif list(self.method.keys())[0] == "Uniform":
            data = {'key': [f"G0_C{i}_P" for i in range(initial_points)],
                    'A': np.linspace(list(self.bounds.values())[0][0], self.bounds[0][1], initial_points),
                    'B': np.linspace(list(self.bounds.values())[1][0], self.bounds[1][1], initial_points),
                    'a': np.linspace(list(self.bounds.values())[2][0], self.bounds[2][1], initial_points),
                    'b': np.linspace(list(self.bounds.values())[3][0], self.bounds[3][1], initial_points),
                    'Ri': np.linspace(list(self.bounds.values())[4][0], list(self.bounds.values())[4][1],
                                      initial_points),
                    'L': np.linspace(list(self.bounds.values())[5][0], list(self.bounds.values())[5][1],
                                     initial_points),
                    'Req': np.linspace(list(self.bounds.values())[6][0], list(self.bounds.values())[6][1] + 1,
                                       initial_points),
                    'alpha_i': np.zeros(initial_points),
                    'alpha_o': np.zeros(initial_points)}
            return pd.DataFrame.from_dict(data)

    def process_constraints(self, constraints):
        processed_constraints = []

        for key, bounds in constraints.items():
            if isinstance(bounds, list):
                if len(bounds) == 2:
                    processed_constraints.append(fr'{key} > {bounds[0]}')
                    processed_constraints.append(fr'{key} < {bounds[1]}')
                else:
                    processed_constraints.append(fr'{key} > {bounds[0]}')
            else:
                processed_constraints.append(fr'{key} = {bounds}')

        return processed_constraints

    def crossover(self, df, generation, f):  # , rq, grq
        elites = {}
        for i, o in enumerate(self.objectives):

            if self.uq_config:
                if o[0] == "min":
                    elites[f'E[{o[1]}] + 6*std[{o[1]}]'] = df.sort_values(f'E[{o[1]}] + 6*std[{o[1]}]')
                elif o[0] == "max":
                    elites[f'E[{o[1]}] - 6*std[{o[1]}]'] = df.sort_values(f'E[{o[1]}] - 6*std[{o[1]}]', ascending=False)
                elif o[0] == "equal":
                    elites[fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'] = df.sort_values(
                        fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]')
            else:
                if o[0] == "min":
                    elites[f'{o[1]}'] = df.sort_values(f'{o[1]}')
                elif o[0] == "max":
                    elites[f'{o[1]}'] = df.sort_values(f'{o[1]}', ascending=False)
                elif o[0] == "equal":
                    elites[f'{o[1]}'] = df.sort_values(f'{o[1]}')

        obj_dict = {}
        for o in self.objectives:
            if self.uq_config:
                if o[0] == 'min':
                    obj_dict[fr'E[{o[1]}] + 6*std[{o[1]}]'] = elites[fr'E[{o[1]}] + 6*std[{o[1]}]']
                elif o[0] == 'max':
                    obj_dict[fr'E[{o[1]}] - 6*std[{o[1]}]'] = elites[fr'E[{o[1]}] - 6*std[{o[1]}]']
                else:
                    obj_dict[fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'] = elites[fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]']
            else:
                # if o[0] != 'equal':
                obj_dict[o[1]] = elites[o[1]]

        obj = {}
        for key, o in obj_dict.items():
            obj[key] = o.reset_index(drop=True)

        # e, b, rq = obj_list
        # e = e.reset_index(drop=True)
        # b = b.reset_index(drop=True)
        # rq = rq.reset_index(drop=True)

        # naming convention G<generation number>_C<cavity number>_<type>
        # type refers to mutation M or crossover C
        df_co = pd.DataFrame(columns=["key", 'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', "alpha_i", "alpha_o"])

        # select only best characteristics
        A_inf = ['All']
        B_inf = ['All']
        a_inf = ['All']
        b_inf = ['All']
        Ri_inf = ['All']
        L_inf = ['All']
        Req_inf = ['All']

        inf_dict = {"A": A_inf, "B": B_inf, "a": a_inf, "b": b_inf, "Ri": Ri_inf, "L": L_inf, "Req": Req_inf}
        for key, influence in inf_dict.items():
            if influence == [''] or influence == ['All']:
                if self.uq_config:
                    ll = []
                    for o in self.objectives:
                        if o[0] == 'min':
                            ll.append(fr'E[{o[1]}] + 6*std[{o[1]}]')
                        elif o[0] == 'max':
                            ll.append(fr'E[{o[1]}] - 6*std[{o[1]}]')
                        else:
                            ll.append(fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]')
                    inf_dict[key] = ll
                else:
                    # inf_dict[key] = [o[1] for o in self.objectives if o[0] != 'equal']
                    inf_dict[key] = self.objective_vars

        n_elites_to_cross = self.elites_to_crossover

        for i in range(f):
            # (<obj>[<rank>][<variable>] -> (b[c[1]][0]

            df_co.loc[i] = [f"G{generation}_C{i}_CO",
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["A"] for key
                                 in inf_dict["A"]]) / len(inf_dict["A"]),  # A
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["B"] for key
                                 in inf_dict["B"]]) / len(inf_dict["B"]),  # B
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["a"] for key
                                 in inf_dict["a"]]) / len(inf_dict["a"]),  # a
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["b"] for key
                                 in inf_dict["b"]]) / len(inf_dict["b"]),  # b
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["Ri"] for
                                 key in inf_dict["Ri"]]) / len(inf_dict["Ri"]),  # Ri
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["L"] for key
                                 in inf_dict["L"]]) / len(inf_dict["L"]),  # L
                            sum([obj[key].loc[np.random.randint(
                                n_elites_to_cross if n_elites_to_cross < df.shape[0] else df.shape[0] - 1)]["Req"] for
                                 key in inf_dict["Req"]]) / len(inf_dict["Req"]),
                            0,
                            0
                            ]
        return df_co

    def mutation(self, df, n, f):

        # get list based on mutation length
        if df.shape[0] < f:
            ml = np.arange(df.shape[0])
        else:
            ml = np.arange(f)

        df_ng_mut = pd.DataFrame(columns=['key', 'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', "alpha_i", "alpha_o"])
        if list(self.bounds.values())[0][0] == list(self.bounds.values())[0][1]:
            df_ng_mut.loc[:, 'A'] = df.loc[ml, "A"]
        else:
            df_ng_mut.loc[:, 'A'] = df.loc[ml, "A"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[1][0] == list(self.bounds.values())[1][1]:
            df_ng_mut.loc[:, 'B'] = df.loc[ml, "B"]
        else:
            df_ng_mut.loc[:, 'B'] = df.loc[ml, "B"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[2][0] == list(self.bounds.values())[2][1]:
            df_ng_mut.loc[:, 'a'] = df.loc[ml, "a"]
        else:
            df_ng_mut.loc[:, 'a'] = df.loc[ml, "a"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[3][0] == list(self.bounds.values())[3][1]:
            df_ng_mut.loc[:, 'b'] = df.loc[ml, "b"]
        else:
            df_ng_mut.loc[:, 'b'] = df.loc[ml, "b"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[4][0] == list(self.bounds.values())[4][1]:
            df_ng_mut.loc[:, 'Ri'] = df.loc[ml, "Ri"]
        else:
            df_ng_mut.loc[:, 'Ri'] = df.loc[ml, "Ri"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[5][0] == list(self.bounds.values())[5][1]:
            df_ng_mut.loc[:, 'L'] = df.loc[ml, "L"]
        else:
            df_ng_mut.loc[:, 'L'] = df.loc[ml, "L"] * random.uniform(0.85, 1.5)

        if list(self.bounds.values())[6][0] == list(self.bounds.values())[6][1]:
            df_ng_mut.loc[:, 'Req'] = df.loc[ml, "Req"]
        else:
            df_ng_mut.loc[:, 'Req'] = df.loc[ml, "Req"] * random.uniform(0.85, 1.5)

        df_ng_mut.loc[:, ["alpha_i", "alpha_o"]] = df.loc[ml, ["alpha_i", "alpha_o"]]

        key1, key2 = [], []
        for i in range(len(df_ng_mut)):
            key1.append(f"G{n}_C{i}_M")

        df_ng_mut.loc[:, 'key'] = key1

        return df_ng_mut

    def chaos(self, f, n):
        df = self.generate_first_men(f, n)
        return df

    @staticmethod
    def remove_duplicate_values(d):
        temp = []
        res = dict()
        for key, val in d.items():
            if val not in temp:
                temp.append(val)
                res[key] = val
        return res

    @staticmethod
    def proof_filename(filepath):
        # check if extension is included
        if filepath.split('.')[-1] != 'json':
            filepath = f'{filepath}.json'

        return filepath

    def recursive_save(self, df, filename, pareto_index):
        styler = self.color_pareto(df, self.poc)
        try:
            styler.to_excel(filename)
            # df.to_excel(filename)
        except PermissionError:
            filename = filename.split('.xlsx')[0]
            filename = fr'{filename}_1.xlsx'
            self.recursive_save(df, filename, pareto_index)

    def pareto_front(self, df):
        # datapoints = np.array([reverse_list(x), reverse_list(y), reverse_list(z)])
        # reverse list or not based on objective goal: minimize or maximize
        # datapoints = [self.negate_list(df.loc[:, o[1]], o[0]) for o in self.objectives]
        sense = []
        if self.uq_config:
            obj = []
            for o in self.objectives:
                if o[0] == 'min':
                    obj.append(fr'E[{o[1]}] + 6*std[{o[1]}]')
                elif o[0] == 'max':
                    obj.append(fr'E[{o[1]}] - 6*std[{o[1]}]')
                elif o[0] == 'equal':
                    obj.append(fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]')

            datapoints = df.loc[:, obj]
        else:
            datapoints = df.loc[:, self.objective_vars]

        for o in self.objectives:
            if o[0] == 'min':
                sense.append('min')
            elif o[0] == "equal":
                sense.append('diff')
            elif o[0] == 'max':
                sense.append('max')

        bool_array = paretoset(datapoints, sense=sense)  # the indices of the Pareto optimal designs
        lst = np.where(bool_array)[0]
        self.poc = len(lst)

        reorder_idx = list(lst) + [i for i in range(len(df)) if i not in lst]

        # return [optimal_datapoints[i, :] for i in range(datapoints.shape[0])]
        return reorder_idx, lst

    # def __pareto_front_oapackage(self, df):
    #
    #     # datapoints = np.array([reverse_list(x), reverse_list(y), reverse_list(z)])
    #     # reverse list or not based on objective goal: minimize or maximize
    #     # datapoints = [self.negate_list(df.loc[:, o[1]], o[0]) for o in self.objectives]
    #
    #     if self.uq_config:
    #         obj = []
    #         for o in self.objectives:
    #             if o[0] == 'min':
    #                 obj.append(fr'E[{o[1]}] + 6*std[{o[1]}]')
    #             elif o[0] == 'max':
    #                 obj.append(fr'E[{o[1]}] - 6*std[{o[1]}]')
    #             elif o[0] == 'equal':
    #                 obj.append(fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]')
    #
    #         datapoints = df.loc[:, obj]
    #     else:
    #         datapoints = df.loc[:, self.objective_vars]
    #
    #     for o in self.objectives:
    #         if o[0] == 'min':
    #             if self.uq_config:
    #                 datapoints[fr'E[{o[1]}] + 6*std[{o[1]}]'] = datapoints[fr'E[{o[1]}] + 6*std[{o[1]}]'] * (-1)
    #             else:
    #                 datapoints[o[1]] = datapoints[o[1]] * (-1)
    #         elif o[0] == "equal":
    #             if self.uq_config:
    #                 datapoints[fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'] = datapoints[
    #                                                                          fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'] * (
    #                                                                          -1)
    #             else:
    #                 datapoints[o[1]] = datapoints[o[1]] * (-1)
    #     # convert datapoints to numpy array
    #
    #     pareto = oapackage.ParetoDoubleLong()
    #     for ii in range(0, datapoints.shape[0]):
    #         w = oapackage.doubleVector(tuple(datapoints.iloc[ii].values))
    #         pareto.addvalue(w, ii)
    #     pareto.show(verbose=1)  # Prints out the results from pareto
    #
    #     lst = pareto.allindices()  # the indices of the Pareto optimal designs
    #     self.poc = len(lst)
    #     reorder_idx = list(lst) + [i for i in range(len(df)) if i not in lst]
    #
    #     # return [optimal_datapoints[i, :] for i in range(datapoints.shape[0])]
    #     return reorder_idx, lst

    @staticmethod
    def negate_list(ll, arg):
        if arg == 'max':
            return ll  # to find the pareto maxima
        else:
            return [-x for x in ll]  # to find the pareto minima

    @staticmethod
    def overwriteFolder(invar, projectDir):
        path = os.path.join(projectDir, 'SimulationData', 'SLANS_opt', f'_process_{invar}')

        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}

        os.makedirs(path)

    @staticmethod
    def copyFiles(invar, parentDir, projectDir):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, 'SimulationData', 'SLANS_opt', f'_process_{invar}', 'SLANS_exe')

        dir_util.copy_tree(src, dst)

    @staticmethod
    def color_pareto(df, no_pareto_optimal):
        def color(row):
            # if row.isnull().values.any():
            if row.iloc[0] in df['key'].tolist()[0:no_pareto_optimal]:
                return ['background-color: #6bbcd1'] * len(row)
            return [''] * len(row)

        # Save Styler Object for Later
        styler = df.style
        # Apply Styles (This can be chained or on separate lines)
        styler.apply(color, axis=1)
        # Export the styler to excel
        return styler


class Cavity:
    """
    Command Line Interface module for running analysis.

    .. note::

       Still under development so some functions might not work properly
    """

    def __init__(self, n_cells, mid_cell, end_cell_left=None, end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k', plot_label=None):
        """
        Initialise cavity object. A cavity object is defined by the number of cells, the cell geometric parameters,
        if it has beampipes or not and the name. These properties could be changed and retrieved later using the
        corresponding ``set`` and ``get`` functions.

        Parameters
        ----------
        n_cells: int
            Number of cells
        mid_cell: list, ndarray
            Mid cell geometric parameters of the cavity
        end_cell_left: list, ndarray
            Left end cell geometric parameters of the cavity
        end_cell_right: list, ndarray
            Right end cell geometric parameters of the cavity
        beampipe: {'none', 'both', 'left', 'right'}
            Beampipe options
        name
        """

        self.kind = 'elliptical cavity'
        self.eigenmode_qois_all_modes = {}
        self.Epk_Eacc = None
        self.Bpk_Eacc = None
        self.Q = None
        self.Ez_0_abs = {'z(0, 0)': [], '|Ez(0, 0)|': []}
        if isinstance(mid_cell, dict):
            # then mid cell parameter is a shape
            end_cell_left = mid_cell['OC']
            end_cell_right = mid_cell['OC_R']
            mid_cell = mid_cell['IC']

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
        if plot_label is None:
            self.plot_label = name
        else:
            self.plot_label = plot_label

        self.n_modes = n_cells + 1
        self.n_modules = 1
        self.projectDir = None
        self.bc = 33
        self.name = name

        # self.n_cav_op_field = {}
        # self.op_field = {}
        # self.p_wp = {}
        # self.pstat = {}
        # self.pdyn = {}
        # self.p_cryo = {}
        # self.p_in = {}
        # self.Q0 = {}

        self.beampipe = beampipe
        self.no_of_modules = 1
        self.eigenmode_qois = {}
        self.custom_eig_qois = {}
        self.wakefield_qois = {}
        self.wake_op_points = {}
        self.convergence_list = []
        self.tune_results = {}
        self.operating_points = None
        self.color = color
        self.Q0 = None
        self.inv_eta = None
        self.neighbours = {}

        # eigenmode results
        self.R_Q, self.k_fm, self.GR_Q, self.freq, self.e, self.b, \
            self.G, self.ff, self.k_cc, self.axis_field, self.surface_field = [0 for _ in range(11)]

        # wakefield results
        self.k_fm, self.k_loss, self.k_kick, self.phom, self.sigma, self.I0 = [{} for _ in range(6)]

        self.wall_material = None
        self.n_cells = n_cells

        self.cell_parameterisation = cell_parameterisation
        if self.cell_parameterisation == 'flattop':
            assert len(mid_cell) > 7, error('Flattop cavity mid-cells require at least 8 input parameters, '
                                            'with the 8th representing length (l).')
            if end_cell_left is not None:
                assert len(end_cell_left) > 7, error(
                    'Flattop cavity left end-cells require at least 8 input parameters, '
                    'with the 8th representing length (l).')
            if end_cell_right is not None:
                assert len(end_cell_right) > 7, error(
                    'Flattop cavity right end-cells require at least 8 input parameters,'
                    'with the 8th representing length (l).')
            self.mid_cell = np.array(mid_cell)[:8]
            self.end_cell_left = np.array(end_cell_left)[:8]
            self.end_cell_right = np.array(end_cell_right)[:8]

            if not (isinstance(end_cell_left, np.ndarray) or isinstance(end_cell_left, list)):
                end_cell_left = mid_cell

            if not (isinstance(end_cell_right, np.ndarray) or isinstance(end_cell_right, list)):
                if not (isinstance(end_cell_left, np.ndarray) or isinstance(end_cell_left, list)):
                    end_cell_right = mid_cell
                else:
                    end_cell_right = end_cell_left

            self.end_cell_left = end_cell_left
            self.end_cell_right = end_cell_right

            self.A, self.B, self.a, self.b, self.Ri, self.L, self.Req, self.l = self.mid_cell[:8]
            self.A_el, self.B_el, self.a_el, self.b_el, self.Ri_el, self.L_el, self.Req_el, self.l_el = self.end_cell_left[
                                                                                                        :8]
            self.A_er, self.B_er, self.a_er, self.b_er, self.Ri_er, self.L_er, self.Req_er, self.l_er = self.end_cell_right[
                                                                                                        :8]

            # active cavity length
            self.l_active = (2 * (self.n_cells - 1) * self.L +
                             (self.n_cells - 2) * self.l + self.L_el + self.l_el +
                             self.L_er + self.l_er) * 1e-3
            self.l_cavity = self.l_active + 8 * (self.L + self.l) * 1e-3

            # get geometric parameters
            self.shape = {
                "IC": update_alpha(self.mid_cell[:8], self.cell_parameterisation),
                "OC": update_alpha(self.end_cell_left[:8], self.cell_parameterisation),
                "OC_R": update_alpha(self.end_cell_right[:8], self.cell_parameterisation),
                "BP": beampipe,
                "n_cells": self.n_cells,
                'CELL PARAMETERISATION': self.cell_parameterisation,
                'kind': self.kind
            }
            self.shape_multicell = {}
            self.to_multicell()  # <- get multicell representation
        else:
            self.mid_cell = np.array(mid_cell)[:7]
            if end_cell_left is not None:
                self.end_cell_left = np.array(end_cell_left)[:7]
            else:
                self.end_cell_left = np.copy(self.mid_cell)
            if end_cell_right is not None:
                self.end_cell_right = np.array(end_cell_right)[:7]
            else:
                self.end_cell_right = np.copy(self.end_cell_left)

            if not (isinstance(end_cell_left, np.ndarray) or isinstance(end_cell_left, list)):
                end_cell_left = mid_cell

            if not (isinstance(end_cell_right, np.ndarray) or isinstance(end_cell_right, list)):
                if not (isinstance(end_cell_left, np.ndarray) or isinstance(end_cell_left, list)):
                    end_cell_right = mid_cell
                else:
                    end_cell_right = end_cell_left

            self.end_cell_left = end_cell_left
            self.end_cell_right = end_cell_right

            self.A, self.B, self.a, self.b, self.Ri, self.L, self.Req = self.mid_cell[:7]
            self.A_el, self.B_el, self.a_el, self.b_el, self.Ri_el, self.L_el, self.Req_el = self.end_cell_left[:7]
            self.A_er, self.B_er, self.a_er, self.b_er, self.Ri_er, self.L_er, self.Req_er = self.end_cell_right[:7]

            # active cavity length
            self.l_active = (2 * (self.n_cells - 1) * self.L + self.L_el + self.L_er) * 1e-3
            self.l_cavity = self.l_active + 8 * self.L * 1e-3

            # get geometric parameters
            self.shape = {
                "IC": update_alpha(self.mid_cell[:7], self.cell_parameterisation),
                "OC": update_alpha(self.end_cell_left[:7], self.cell_parameterisation),
                "OC_R": update_alpha(self.end_cell_right[:7], self.cell_parameterisation),
                "BP": beampipe,
                "n_cells": self.n_cells,
                'CELL PARAMETERISATION': self.cell_parameterisation,
                'kind': self.kind
            }
        self.shape_multicell = {}
        self.to_multicell()  # <- get multicell representation

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

    def study_mesh_convergence(self, h=2, h_passes=10, h_step=1, p=2, p_passes=3, p_step=1):
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
                                    'solver_save_directory': 'ngsolvemevp',
                                    'uq_config': {},
                                    'opt': False,
                                    'mesh_config': {
                                        'h': hs,
                                        'p': p
                                    }
                                    }

                run_eigenmode_s({self.name: self.shape}, {self.name: self.shape_multicell}, self.projectDir,
                                eigenmode_config)
                # read results
                self.get_eigenmode_qois()

                convergence_dict_h[ih] = self.eigenmode_qois
                convergence_dict_h[ih]['h'] = hs
                convergence_dict_h[ih]['p'] = p

                hs += h_step
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
        columns_to_exclude = ['h', 'p']
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
                self.get_uq_fm_results(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', f'{self.name}', 'uq.json'))
            return True
        except FileNotFoundError:
            error("Could not find eigenmode results. Please rerun eigenmode analysis.")
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
        if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'Optimisation', self.name, 'tune_res')):
            with open(os.path.join(self.projectDir, 'SimulationData', 'Optimisation', self.name, 'tune_res'), 'r') as json_file:
                self.tune_results = json.load(json_file)
            self.freq = self.tune_results['FREQ']
            self.shape['IC'] = self.tune_results['IC']
            self.shape['OC'] = self.tune_results['OC']
            self.shape['OC_R'] = self.tune_results['OC_R']
            self.mid_cell = self.shape['IC']
            self.end_cell_left = self.shape['OC']
            self.end_cell_right = self.shape['OC_R']

        else:
            error("Tune results not found. Please tune the cavity")

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        qois = 'qois.json'
        assert os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole', qois)), (
            error('Eigenmode result does not exist, please run eigenmode simulation.'))
        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               'Ez_0_abs.csv')) as csv_file:
            self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

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
                '$\eta=' + fr'{self.ff:.2f}\%' + '$',  # Text content
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
            if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                           'Ez_0_abs.csv')):
                with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                       'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
                ax.text(
                    0.95, 0.05,  # Position (normalized coordinates)
                    '$\eta=$' + fr'{self.ff:.2f}\%' + '$',  # Text content
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
        with open(fr'{folder}\uq.json', 'r') as json_file:
            self.uq_fm_results = json.load(json_file)

        if os.path.exists(fr'{folder}\uq_all_modes.json'):
            with open(fr'{folder}\uq_all_modes.json', 'r') as json_file:
                self.uq_fm_results_all_modes = json.load(json_file)

        # get neighbours and all qois
        neighbours = {}
        for dirr in os.listdir(folder):
            if 'Q' in dirr.split('_')[-1]:
                with open(fr'{folder}\{dirr}\monopole\qois.json', 'r') as json_file:
                    neighbour_uq_fm_results = json.load(json_file)
                neighbours[dirr] = neighbour_uq_fm_results

        self.neighbours = pd.DataFrame.from_dict(neighbours, orient='index')

        nodes, weights = cn_leg_05_2(7)
        # write weights

        data_table = pd.DataFrame(weights, columns=['weights'])
        data_table.to_csv(fr'{folder}\weights.csv', index=False, sep='\t', float_format='%.32f')

        # get weights
        self.uq_weights = pd.read_csv(fr'{folder}\weights.csv')

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
        qois = 'qois.json'

        if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'ABCI', self.name, 'qois.json')):
            with open(os.path.join(self.projectDir, 'SimulationData', 'ABCI', self.name, 'qois.json')) as json_file:
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
        abci_data_dir = os.path.join(self.projectDir, "SimulationData", "ABCI")
        self.abci_data = {'Long': ABCIData(abci_data_dir, self.name, 0),
                          'Trans': ABCIData(abci_data_dir, self.name, 1)}

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

        top_folder = os.path.join(self.projectDir, "SimulationData", "ABCI", self.name)
        efield_contour = get_wakefield_data(fr'{top_folder}/Cavity_MROT_0.top')
        ani = animate_frames(efield_contour)
        display_html(HTML(animate_frames(efield_contour).to_jshtml()))

        if save:
            # Save the animation as an MP4 file
            ani.save(fr'{top_folder}/{self.name}_e_field_animation.mp4', writer='ffmpeg', dpi=150)

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
        gfu_E, gfu_H = ngsolve_mevp.load_fields(os.path.join(self.projectDir,
                                                             'SimulationData', 'NGSolveMEVP',
                                                             self.name, 'monopole'),
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
        ngsolve_mevp.plot_mesh(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole'), plotter=plotter)

    def plot_fields(self, mode=1, which='E', plotter='ngsolve'):
        ngsolve_mevp.plot_fields(os.path.join(self.projectDir,
                                              'SimulationData', 'NGSolveMEVP',
                                              self.name, 'monopole'),
                                 mode, which, plotter)

    def _plot_convergence(self, ax):
        keys = list(ax.keys())
        # plot convergence
        conv_filepath = os.path.join(self.projectDir, 'SimulationData', 'Optimisation', self.name, "convergence.json")
        if os.path.exists(conv_filepath):
            with open(conv_filepath, 'r') as f:
                convergence_dict = json.load(f)
            if len(convergence_dict) > 0:
                x, y = convergence_dict[list(convergence_dict.keys())[0]], convergence_dict['freq [MHz]']
                ax[keys[0]].scatter(x, y, ec='k')

                # plot directions
                for i in range(len(x) - 1):
                    dx = x[i + 1] - x[i]
                    dy = y[i + 1] - y[i]
                    ax[keys[0]].quiver(x[i], y[i], dx, dy, ls='--', angles='xy',
                                       scale_units='xy', scale=1, color='red',
                                       width=0.005, units='width', headwidth=3, headlength=5,
                                       headaxislength=4)

        # plot absolute error
        abs_err_filepath = os.path.join(self.projectDir, 'SimulationData', 'Optimisation', self.name, "absolute_error.json")
        abs_err_dict = {}
        if os.path.exists(conv_filepath):
            with open(abs_err_filepath, 'r') as f:
                abs_err_dict = json.load(f)
        if len(abs_err_dict) > 0:
            ax[keys[1]].plot(abs_err_dict['abs_err'], marker='o', mec='k')

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

    def to_multicell(self):
        mid_cell = self.shape['IC']
        mid_cell_multi = np.array([[[a, a] for _ in range(self.n_cells - 1)] for a in mid_cell])

        self.shape_multicell['OC'] = self.shape['OC']
        self.shape_multicell['OC_R'] = self.shape['OC_R']
        self.shape_multicell['IC'] = mid_cell_multi
        self.shape_multicell['BP'] = self.shape['BP']
        self.shape_multicell['n_cells'] = self.shape['n_cells']
        self.shape_multicell['CELL PARAMETERISATION'] = 'multicell'
        self.shape_multicell['kind'] = self.kind

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
                            'SimulationData': {
                                'SLANS': None,
                                'SLANS_Opt': None,
                                'NGSolveMEVP': None,
                                'NativeEig': None,
                                'ABCI': None,
                                'CavitiesAnalysis': None
                            },
                            'PostprocessingData': {
                                'Plots': None,
                                'Data': None,
                                'CSTData': None
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
                            and 'PostprocessingData' in directory_list \
                            and 'SimulationData' in directory_list and len(directory_list) < 6:
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

    @staticmethod
    def _overwriteFolder(invar, projectDir, name):
        path = os.path.join(self.projectDir, 'SimulationData', 'SLANS', f'_process_{invar}')
        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}

        os.makedirs(path)

    @staticmethod
    def _copyFiles(invar, parentDir, projectDir, name):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, 'SimulationData', 'SLANS', f'_process_{invar}', 'SLANS_exe')

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


class Cavities(Optimisation):
    """
    Cavities object is an object containing several Cavity objects.
    """

    def __init__(self, name=None, cavities_list=None, names_list=None):
        """Constructs all the necessary attributes of the Cavity object

        Parameters
        ----------
        cavities_list: list, array like
            List containing Cavity objects.

        """

        super().__init__()
        self.uq_fm_results_all_modes = {}
        self.rf_config = None
        self.power_qois_uq = {}
        self.shape_space = {}
        self.shape_space_multicell = {}
        self.sweep_results = None
        self.cavities_list = cavities_list
        self.cavities_dict = {}
        if cavities_list is None or cavities_list == []:
            self.cavities_list = []
        else:
            self.add_cavity(cavities_list, names_list)

        self.name = 'cavities'
        if name:
            assert isinstance(name, str), error('Please enter valid project name.')
            self.name = name
        self.eigenmode_qois = {}
        self.eigenmode_qois_all_modes = {}
        self.wakefield_qois = {}
        self.tune_results = {}

        self.uq_fm_results = {}
        self.uq_hom_results = {}

        self.power_qois = {}
        self.fm_results = None
        self.hom_results = None
        self.projectDir = None

        self.operating_points = None
        self.operating_points_threshold = {}

        self.returned_results = None
        self.ls = ['solid', 'dashed', 'dashdot', 'dotted',
                   'solid', 'dashed', 'dashdot', 'dotted',
                   'solid', 'dashed', 'dashdot', 'dotted']

        self.E_acc = np.linspace(0.5, 30, 100) * 1e6  # V/m
        self.set_cavities_field()

    def add_cavity(self, cavs, names=None, plot_labels=None):
        """
        Adds cavity to cavities
        Parameters
        ----------
        plot_labels: list, str
            Plot labels
        cavs: Cavity, list
            Cavity object or list of cavity objects
        names: list, str
            Cavity name or list of cavity names

        Returns
        -------

        """

        if isinstance(cavs, Cavity):
            cavs.projectDir = self.projectDir
            if names:
                cavs.set_name(names)
            else:
                cavs.set_name(f'cav_{len(self.cavities_list)}')

            if isinstance(plot_labels, list):
                cavs.set_plot_label(plot_labels[0])
            else:
                cavs.set_plot_label(plot_labels)

            self.cavities_list.append(cavs)
            self.cavities_dict[cavs.name] = cavs
            self.shape_space[cavs.name] = cavs.shape
            self.shape_space_multicell[cavs.name] = cavs.shape_multicell

        elif isinstance(cavs, RFGun):
            cavs.projectDir = self.projectDir
            if names:
                cavs.set_name(names)
            else:
                cavs.set_name(f'cav_{len(self.cavities_list)}')

            if isinstance(plot_labels, list):
                cavs.set_plot_label(plot_labels[0])
            else:
                cavs.set_plot_label(plot_labels)

            self.cavities_list.append(cavs)
            self.cavities_dict[cavs.name] = cavs
            self.shape_space[cavs.name] = cavs.shape
            self.shape_space_multicell[cavs.name] = cavs.shape_multicell

        else:
            if names is not None:
                assert len(cavs) == len(names), "Number of cavities does not correspond to number of names."
            else:
                names = [f'cav_{ii}' for ii in range(len(self.cavities_list))]

            if plot_labels is not None:
                assert len(cavs) == len(plot_labels), "Number of cavities does not correspond to number of labels."
            else:
                plot_labels = names

            for i1, cav in enumerate(cavs):
                cav.projectDir = self.projectDir
                cav.set_name(names[i1])
                cav.set_plot_label(plot_labels[i1])

                self.cavities_list.append(cav)
                self.cavities_dict[cav.name] = cav
                self.shape_space[cav.name] = cav.shape
                self.shape_space_multicell[cav.name] = cav.shape_multicell

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

    def save(self, project_folder, overwrite=False):
        """
        Set folder to save cavity analysis results

        Parameters
        ----------
        files_path: str
            Save project directory

        Returns
        -------

        """

        if project_folder is None:
            error('Please specify a folder to write the simulation results to.')
            return
        else:
            try:
                self.projectDir = project_folder
                success = self._create_project(overwrite)
                if not success:
                    error(f"Project {project_folder} could not be created. Please check the folder and try again.")
                    return
                else:
                    done(f"Project {project_folder} created successfully/already exists.")
            except Exception as e:
                error("Exception occurred: ", e)
                return

        if project_folder is None:
            self.projectDir = Path(os.getcwd())

    def save_plot_as_json(self, ax):
        ax_children = ax.get_children()

        ax_objects_dict = {
            'ax props': None,
            'fig props': {},
            'lines': {
                # 'line1': line_properties,
            },
            'axvline': {},
            'axhline': {},
            'scatter': {},
            'patches': {
                'text': {},
                'rectangle': {}
            },
        }

        # get axis properties
        axis_properties = self._get_axis_properties(ax)
        ax_objects_dict['ax props'] = axis_properties

        # get figure properties
        fig_props = self._get_figure_properties(ax.get_figure())
        ax_objects_dict['fig props'] = fig_props

        # List of known properties for Line2D
        line_properties = [
            'xdata', 'ydata', 'alpha', 'animated', 'antialiased', 'clip_on', 'clip_path',
            'color', 'dash_capstyle', 'dash_joinstyle', 'drawstyle',
            'label', 'linestyle', 'linewidth', 'marker', 'markeredgecolor',
            'markeredgewidth', 'markerfacecolor', 'markersize', 'path_effects', 'rasterized', 'sketch_params',
            'snap',
            'solid_capstyle',
            'solid_joinstyle', 'url', 'visible', 'zorder',
            # 'transform',
            # 'clip_box',
            # 'figure', 'picker', 'pickradius'
        ]

        for mpl_obj in ax_children:
            if isinstance(mpl_obj, matplotlib.lines.Line2D):
                # Extract properties into a dictionary
                line_properties_data = {prop: getattr(mpl_obj, f'get_{prop}')() for prop in line_properties}
                ax_objects_dict['lines'][fr'{id(mpl_obj)}'] = line_properties_data

        return ax_objects_dict

    def load_plot_from_json(self, filepath, ax=None):
        with open(filepath, 'r') as f:
            plot_config = json.load(f)

        if ax is not None:
            fig, ax = plt.subplot_mosaic([[0]])

        for key, value in plot_config.items():
            if key == 'lines':
                line = ax.plot([], [])
                line.set()
            if key == 'scatter':
                ax.scatter(value['x'], value['y'], value['kwargs'])
            if key == 'patches':
                if value['type'] == 'text':
                    ax.add_text(value['x'], value['y'], value['kwargs'])

    def plot_from_json(self, plot_config, ax=None):
        if ax is None:
            fig, ax = plt.subplot_mosaic([[0]])

        for key, value in plot_config.items():
            if key == 'lines':
                for line_keys, line_values in value.items():
                    line, = ax[0].plot([], [])
                    line.set(**line_values)
            if key == 'scatter':
                pass
                # ax[0].scatter(value['x'], value['y'], value['kwargs'])
            if key == 'patches':
                pass
                # if value['type'] == 'text':
                #     ax[0].add_text(value['x'], value['y'], value['kwargs'])
            if key == 'fig props':
                ax[0].get_figure().set(**value)

            if key == 'ax props':
                ax[0].set(**value)

        if isinstance(ax, dict):
            ax[0].relim()
            ax[0].get_figure().canvas.draw()
            ax[0].get_figure().canvas.flush_events()
            ax[0].get_figure().canvas.draw_idle()
        else:
            ax.relim()
            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()
            ax.get_figure().canvas.draw_idle()

        # plot legend wthout duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    def _get_axis_properties(self, ax):
        # List of common properties for an Axis
        axis_properties = [
            'label', 'ticks', 'scale', 'margin', 'bound', 'aspect'
        ]

        def get_axis_properties(axis, prefix):
            properties = {}
            for prop in axis_properties:
                try:
                    # Use get_* methods dynamically if available
                    getter = getattr(axis, f'get_{prop}', None)
                    if callable(getter):
                        if prop == 'label':
                            properties[fr'{prefix}{prop}'] = getter().get_text()
                        else:
                            properties[fr'{prefix}{prop}'] = getter()
                    else:
                        # Fallback to axis.get_property()
                        properties[fr'{prefix}{prop}'] = axis.get_property(prop)
                except Exception as e:
                    pass
            return properties

        axis_properties = get_axis_properties(ax.xaxis, 'x')
        axis_properties.update(get_axis_properties(ax.yaxis, 'y'))

        return axis_properties

    def _get_figure_properties(self, fig):
        # List of properties to retrieve
        figure_properties = [
            'figwidth', 'figheight', 'dpi', 'tight_layout', 'constrained_layout'
        ]

        def get_figure_properties(fig):
            properties = {}
            for prop in figure_properties:
                getter = getattr(fig, f'get_{prop}', None)
                if callable(getter):
                    properties[prop] = getter()

                else:
                    properties[prop] = 'N/A'
            return properties

        # Get properties of the figure
        fig_size = []
        fig_properties = get_figure_properties(fig)

        return fig_properties

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
                            'SimulationData': {
                                'SLANS': None,
                                'SLANS_Opt': None,
                                'NGSolveMEVP': None,
                                'NativeEig': None,
                                'ABCI': None,
                                'Optimisation': None
                            },
                            'PostprocessingData': {
                                'Plots': None,
                                'Data': None,
                                'CSTData': None
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
                # self.projectDir = os.path.join(project_dir, project_name)
                self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                return True
        else:
            error('\tPlease enter a valid project name')
            self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
            return False

    @staticmethod
    def _check_if_path_exists(directory, folder, overwrite):
        path = f"{directory}/{folder}"
        if os.path.exists(path):
            x = 'n'
            if overwrite:
                x = 'y'

            if x == 'y':
                try:
                    directory_list = os.listdir(path)

                    if 'Cavities' in directory_list \
                            and 'PostprocessingData' in directory_list \
                            and 'SimulationData' in directory_list and len(directory_list) < 6:
                        shutil.rmtree(path)
                        return True
                    else:
                        error('\tIt seems that the folder specified is not a cavity project folder. Please check folder'
                              'again to avoid deleting important files.')
                        return False

                except Exception as e:
                    error("Exception occurred: ", e)
                    return False
            else:
                return False
        else:
            return True

    def sweep(self, sweep_config):
        self.sweep_results = {}
        for cav in self.cavities_list:
            cav.sweep(sweep_config)
            self.sweep_results[cav.name] = cav.sweep_results

    def plot_dispersion(self):
        for cav in self.cavities_list:
            cav.plot_dispersion()

    def check_uq_config(self, uq_config):
        uq_ok = {}
        for cav in self.cavities_list:
            res = cav.check_uq_config(uq_config)
            uq_ok[cav.name] = res
        info(uq_ok)

    def run_tune(self, tune_config=None):
        """

        Parameters
        ----------
        tune_config: dict

            .. code-block:: python

                tune_config = {
                            'freqs': 801.58,
                            'parameters': 'Req',
                            'cell_types': 'mid-cell',
                            'processes': 1,
                            'rerun': True
                        }

        Returns
        -------

        """

        if tune_config is None:
            # set default tune_config
            tune_config = {
                'parameters': ['Req' for _ in self],
                'freqs': [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self],
                'cell_types': ['mid_cell' for _ in self]
            }
            info(f'Tune variable and frequency not entered, defaulting to {json.dumps(tune_config, indent=4)}')

        if 'freqs' not in tune_config.keys():
            # set default tune_config
            tune_config['freqs'] = [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self]
            info(f'Target frequency not entered, defaulting to {tune_config["freqs"]}')

        if 'parameters' not in tune_config.keys():
            # set default tune_config
            tune_config['parameters'] = ['Req' for _ in self]
            info(f'"parameters" not entered, defaulting to {tune_config["parameters"]}')

        if 'cell_types' not in tune_config.keys():
            # set default tune_config
            tune_config['cell_types'] = ['mid_cell' for _ in self]
            info(f'"cell_types" not entered, defaulting to {tune_config["cell_types"]}')

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if 'delta' in uq_config.keys():
                assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                     "be equal to the number of "
                                                                                     "variables.")
            if 'epsilon' in uq_config.keys():
                assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                    "The number of epsilons must "
                    "be equal to the number of "
                    "variables.")

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

        if 'eigenmode_config' in tune_config.keys():
            eigenmode_config = tune_config['eigenmode_config']

            if 'uq_config' in eigenmode_config.keys():
                uq_config_eig = eigenmode_config['uq_config']
                if 'delta' in uq_config_eig.keys():
                    assert len(uq_config_eig['delta']) == len(uq_config_eig['variables']), error(
                        "The number of deltas must "
                        "be equal to the number of "
                        "variables.")
                if 'epsilon' in uq_config_eig.keys():
                    assert len(uq_config_eig['epsilon']) == len(uq_config_eig['variables']), error(
                        "The number of epsilons must "
                        "be equal to the number of "
                        "variables.")

                if 'epsilon' in uq_config_eig.keys() and 'uq_config' in uq_config_eig.keys():
                    info('epsilon and delta are both entered. Epsilon is preferred.')

        rerun = True
        if 'rerun' in tune_config:
            rerun = tune_config['rerun']
            assert isinstance(rerun, bool), error('rerun should be boolean.')
        else:
            tune_config['rerun'] = rerun

        if rerun:
            run_tune_parallel(self.shape_space, tune_config, self.projectDir, solver='NGSolveMEVP', resume=False)

        self.tune_config = tune_config
        # get tune results
        self.get_tune_res()

    def get_tune_res(self):
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_ngsolve_tune_res()
                self.tune_results[cav.name] = cav.tune_results
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    def run_eigenmode(self, eigenmode_config=None):
        """
        Runs the eigenmode analysis with the given configuration.

        Parameters
        ----------
        eigenmode_config : dict
            Configuration for running the eigenmode analysis. Example structure:

            .. code-block:: python

                eigenmode_config = {
                    'processes': 3,
                    'rerun': True,
                    'boundary_conditions': 'mm',
                    'uq_config': {
                        'variables': ['A', 'B', 'a', 'b'],
                        # 'objectives': ["freq [MHz]", "R/Q [Ohm]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "G [Ohm]", "kcc [%]", "ff [%]"],
                        'objectives': ["Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]"],
                        # 'objectives': ["ZL"],
                        'delta': [0.05, 0.05, 0.05, 0.05],
                        'processes': 4,
                        'distribution': 'gaussian',
                        # 'method': ['QMC', 'LHS', 1000],
                        # 'method': ['QMC', 'Sobol', 1000],
                        # 'method': ['Qudrature', 'Gaussian', 1000],
                        'method': ['Quadrature', 'Stroud3'],
                        # 'method': ['Quadrature', 'Stroud5'],
                        # 'gaussian': ['Quadrature', 'Gaussian'],
                        # 'from file': ['<file path>', columns],
                        'cell_type': 'mid-cell',
                        'cell_complexity': 'simplecell'
                    }
                }

        Returns
        -------
        None

        """

        if eigenmode_config is None:
            eigenmode_config = {}

        rerun = True
        if 'rerun' in eigenmode_config.keys():
            if isinstance(eigenmode_config['rerun'], bool):
                rerun = eigenmode_config['rerun']

        uq_config = {}
        if 'uq_config' in eigenmode_config.keys():
            uq_config = eigenmode_config['uq_config']
            if uq_config:
                if 'delta' in uq_config.keys():
                    assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                         "be equal to the number of "
                                                                                         "variables.")

                if 'epsilon' in uq_config.keys():
                    assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                        "The number of epsilons must "
                        "be equal to the number of "
                        "variables.")

                if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                    info('Epsilon and delta are both entered. Epsilon is preferred.')

        else:
            eigenmode_config['uq_config'] = uq_config

        if rerun:
            # add save directory field to eigenmode_config
            eigenmode_config['solver_save_directory'] = 'NGSolveMEVP'
            eigenmode_config['opt'] = False

            run_eigenmode_parallel(self.shape_space, self.shape_space_multicell, eigenmode_config,
                                   self.projectDir)

        self.eigenmode_config = eigenmode_config
        self.get_eigenmode_qois(uq_config)

    def get_eigenmode_qois(self, uq_config):
        # get results
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_eigenmode_qois()
                self.eigenmode_qois[cav.name] = cav.eigenmode_qois
                self.eigenmode_qois_all_modes[cav.name] = cav.eigenmode_qois_all_modes
                if uq_config:
                    cav.get_uq_fm_results(os.path.join(self.projectDir, "SimulationData", "NGSolveMEVP", cav.name))
                    self.uq_fm_results[cav.name] = cav.uq_fm_results
                    self.uq_fm_results_all_modes[cav.name] = cav.uq_fm_results_all_modes
            except FileNotFoundError:
                error("Could not find the eigenmode results. Please rerun eigenmode analysis.")
                return False

    def run_wakefield(self, wakefield_config=None):
        """

        Parameters
        ----------
        wakefield_config:

            .. code-block:: python

                op_points = {
                            "Z": {
                                "freq [MHz]": 400.79,  # Operating frequency
                                "E [GeV]": 45.6,  # <- Beam energy
                                "I0 [mA]": 1280,  # <- Beam current
                                "V [GV]": 0.12,  # <- Total voltage
                                "Eacc [MV/m]": 5.72,  # <- Accelerating field
                                "nu_s []": 0.0370,  # <- Synchrotron oscillation tune
                                "alpha_p [1e-5]": 2.85,  # <- Momentum compaction factor
                                "tau_z [ms]": 354.91,  # <- Longitudinal damping time
                                "tau_xy [ms]": 709.82,  # <- Transverse damping time
                                "f_rev [kHz]": 3.07,  # <- Revolution frequency
                                "beta_xy [m]": 56,  # <- Beta function
                                "N_c []": 56,  # <- Number of cavities
                                "T [K]": 4.5,  # <- Operating tempereature
                                "sigma_SR [mm]": 4.32,  # <- Bunch length
                                "sigma_BS [mm]": 15.2,  # <- Bunch length
                                "Nb [1e11]": 2.76  # <- Bunch population
                            }
                }
                wakefield_config = {
                    'bunch_length': 25,
                    'wakelength': 50,
                    'processes': 2,
                    'rerun': True,
                    'operating_points': op_points,
                }

        Returns
        -------

        """

        if wakefield_config is None:
            wakefield_config = {}

        wakefield_config_keys = wakefield_config.keys()

        if 'operating_points' in wakefield_config_keys:
            self.operating_points = wakefield_config['operating_points']
            for cav in self.cavities_list:
                cav.operating_points = self.operating_points

        rerun = True
        if 'rerun' in wakefield_config_keys:
            if isinstance(wakefield_config['rerun'], bool):
                rerun = wakefield_config['rerun']

        uq_config = {}
        if 'uq_config' in wakefield_config_keys:

            if 'delta' in uq_config.keys():
                assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                     "be equal to the number of "
                                                                                     "variables.")

            if 'epsilon' in uq_config.keys():
                assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                    "The number of epsilons must "
                    "be equal to the number of "
                    "variables.")

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

            assert 'objectives' in wakefield_config['uq_config'].keys(), error('Please enter objectives in uq_config.')

            objectives_unprocessed = []
            # adjust objectives to match signature with optimisation
            for obj in wakefield_config['uq_config']['objectives']:
                if isinstance(obj, list):
                    objectives_unprocessed.append(['', obj[0], obj[1]])
                else:
                    objectives_unprocessed.append(obj)

            objectives, weights = process_objectives(objectives_unprocessed)
            wakefield_config['uq_config']['objectives_unprocessed'] = objectives_unprocessed
            wakefield_config['uq_config']['objectives'] = objectives
            uq_config = wakefield_config['uq_config']
        else:
            wakefield_config['uq_config'] = uq_config

        if rerun:
            MROT = 2
            MT = 10
            NFS = 10000
            wakelength = 50
            bunch_length = 25
            DDR_SIG = 0.1
            DDZ_SIG = 0.1

            processes = 1
            if 'processes' in wakefield_config.keys():
                assert wakefield_config['processes'] > 0, error('Number of proceses must be greater than zero.')
                processes = wakefield_config['processes']
            else:
                wakefield_config['processes'] = processes

            if 'polarisation' in wakefield_config_keys:
                assert wakefield_config['polarisation'] in [0, 1, 2], error(
                    'Polarisation should be 0 for longitudinal, '
                    '1 for transverse, 2 for both.')
            else:
                wakefield_config['polarisation'] = MROT

            if 'MT' in wakefield_config_keys:
                assert isinstance(wakefield_config['MT'], int), error(
                    'MT must be integer between 4 and 20, with 4 and 20 '
                    'included.')
            else:
                wakefield_config['MT'] = MT

            if 'NFS' in wakefield_config_keys:
                assert isinstance(wakefield_config['NFS'], int), error('NFS must be integer.')
            else:
                wakefield_config['NFS'] = NFS

            # check input configs

            # beam config
            wake_config = {}
            beam_config = {}
            mesh_config = {}
            if 'beam_config' not in wakefield_config_keys:
                wakefield_config['beam_config'] = beam_config
            if 'wake_config' not in wakefield_config_keys:
                wakefield_config['wake_config'] = wake_config
            if 'mesh_config' not in wakefield_config_keys:
                wakefield_config['mesh_config'] = mesh_config

            if 'bunch_length' in beam_config.keys():
                assert not isinstance(wakefield_config['beam_config']['bunch_length'], str), error(
                    'Bunch length must be of type integer or float.')
            else:
                beam_config['bunch_length'] = bunch_length

            # wake config
            if 'wakelength' in wake_config.keys():
                assert not isinstance(wake_config['wakelength'], str), error(
                    'Wakelength must be of type integer or float.')
            else:
                wake_config['wakelength'] = wakelength

            if 'DDR_SIG' not in mesh_config.keys():
                mesh_config['DDR_SIG'] = DDR_SIG

            if 'DDZ_SIG' not in mesh_config.keys():
                mesh_config['DDZ_SIG'] = DDZ_SIG

            run_wakefield_parallel(self.shape_space, self.shape_space_multicell, wakefield_config,
                                   self.projectDir, marker='', rerun=rerun)

        self.wakefield_config = wakefield_config
        self.get_wakefield_qois(uq_config)

    def get_wakefield_qois(self, uq_config):
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_abci_data()
                cav.get_wakefield_qois(self.wakefield_config)
                self.wakefield_qois[cav.name] = cav.wakefield_qois
                if uq_config:
                    cav.get_uq_hom_results(os.path.join(self.projectDir, "SimulationData", "ABCI", cav.name, "uq.json"))
                    cav_uq_hom_results = cav.uq_hom_results
                    if 'operating_points' in uq_config:
                        # separate into opearating points
                        op_points = uq_config['operating_points']
                        uq_hom_results_op = {}
                        for op, val in op_points.items():
                            uq_hom_results_op[op] = {}
                            for k, v in val.items():
                                if 'sigma' in k:
                                    sig_id = k.split('_')[-1].split(' ')[0]
                                    ident = fr'_{op}_{sig_id}_{v}mm'
                                    uq_hom_results_op[op][sig_id] = {kk.replace(ident, ''): vv for (kk, vv) in
                                                                     cav_uq_hom_results.items() if ident in kk}
                    else:
                        uq_hom_results_op = cav.uq_hom_results

                    self.uq_hom_results[cav.name] = uq_hom_results_op
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    def plot(self, what, ax=None, scale_x=None, **kwargs):
        for ii, cav in enumerate(self.cavities_list):
            if what.lower() == 'geometry':
                ax = cav.plot('geometry', ax, **kwargs)

            if what.lower() == 'zl':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('zl', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('zl', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'zt':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('zt', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('zt', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'wpl':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('wpl', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('wpl', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'wpt':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('wpt', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('wpt', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'convergence':
                ax = cav.plot('convergence', ax)
        if 'logy' in kwargs:
            if kwargs['logy']:
                ax.set_yscale('log')
        return ax

    def set_cavities_field(self):
        """
        Sets cavities analysis field range.

        Returns
        -------

        """
        for cav in self.cavities_list:
            cav.set_Eacc(Eacc=self.E_acc)

    def compare_power(self, E_acc=None):
        if E_acc is not None:
            self.E_acc = E_acc
            self.set_cavities_field()

        self.p_qois = []
        results = []
        for i, cav in enumerate(self.cavities_list):
            # E_acc_Pin(cavity, op_field[i], ls[i], fig, ax, ax_right, ax_right2)
            results.append(self.qois(cav, cav.op_field * 1e-6, E_acc))

        self.returned_results = results

    def qois_fm(self):
        """
        Retrieves the fundamental mode quantities of interest

        Returns
        -------
        Dictionary containing fundamental mode quantities of interest (normed optional).
        """
        results = []
        for cav in self.cavities_list:
            results.append({
                r"Epk/Eacc []": cav.e,
                r"Bpk/Eacc [mT/MV/m]": cav.b,
                r"kcc [%]": cav.k_cc,
                r"R/Q [Ohm]": cav.R_Q,
                r"G [Ohm]": cav.G,
                r"GR/Q [Ohm^2]": cav.GR_Q
            })

        results_norm_units = []
        for cav in self.cavities_list:
            results_norm_units.append({
                r"$e_\mathrm{pk}$": cav.e,
                r"$b_\mathrm{pk}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$r/q$": cav.R_Q,
                r"$g$": cav.G,
                r"$g\cdot r/q $": cav.GR_Q
            })

        return results

    def qois_hom(self, opt):
        """
        Retrieves the higher-order modes quantities of interest

        Returns
        -------
        Dictionary containing higher-order modes quantities of interest (normed optional).
        """

        results = []
        for cavity in self.cavities_list:
            cavity.get_wakefield_qois(self.wakefield_config)

            results.append({
                r"|k_loss| [V/pC]": cavity.k_loss[opt],
                r"|k_kick| [V/pC/m]": cavity.k_kick[opt],
                r"P_HOM [kW]": cavity.phom[opt]
            })

        return results

    def qois_all(self, opt):
        """
        Retrieves the fundamental mode quantities of interest

        Returns
        -------
        Dictionary containing fundamental mode quantities of interest (normed optional).
        """
        results = []
        for cav in self.cavities_list:
            results.append({
                r"$E_\mathrm{pk}/E_\mathrm{acc} ~[\cdot]$": cav.e,
                r"$B_\mathrm{pk}/E_\mathrm{acc} ~\mathrm{[mT/MV/m]}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$R/Q~ \mathrm{[\Omega]}$": cav.R_Q,
                r"$G ~\mathrm{[\Omega]}$": cav.G,
                r"$G\cdot R/Q ~\mathrm{[10^4\Omega^2]}$": cav.GR_Q * 1e-4,
                r"$|k_\parallel| ~\mathrm{[V/pC]}$": cav.k_loss[opt],
                r"$|k_\perp| ~\mathrm{[V/pC/m]}$": cav.k_kick[opt],
                r"$P_\mathrm{HOM}/cav~ \mathrm{[kW]}$": cav.phom[opt]
            })

        results_norm_units = []
        for cav in self.cavities_list:
            results_norm_units.append({
                r"$e_\mathrm{pk}/e_\mathrm{acc}$": cav.e,
                r"$b_\mathrm{pk}/e_\mathrm{acc}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$r/q$": cav.R_Q,
                r"$g$": cav.G,
                # r"$g\cdot r/q $": cav.GR_Q,
                # r"$|k_\mathrm{FM}|$": cav.k_fm,
                r"$|k_\parallel|$": cav.k_loss[opt],
                r"$k_\perp$": cav.k_kick[opt],
                r"$p_\mathrm{HOM}/cav$": cav.phom[opt]
            })

        return results

    def plot_uq_geometries(self):
        fig, axd = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], layout='constrained')

        for cav, ax in zip(self.cavities_list, axd.values()):
            # plot nominal
            cav.plot('geometry', ax=ax, mid_cell=True, zorder=10)
            directory = os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', cav.name)
            tag = f'{cav.name}_Q'
            uq_geom_folders = self.find_folders_with_tag(directory, tag)

            # load uq geometries
            for uq_geom_folder in uq_geom_folders:
                if os.path.exists(f'{uq_geom_folder}/monopole/geodata.n'):
                    # read geometry
                    cav_geom = pd.read_csv(f'{uq_geom_folder}/monopole/geodata.n', header=None,
                                           skipfooter=1, sep='\\s+', engine='python')

                    cav_geom = cav_geom[[1, 0]]
                    ax.plot(cav_geom[1], cav_geom[0], ls='--', lw=1, c='gray')
            ax.set_title(cav.name)
        return ax

    @staticmethod
    def find_folders_with_tag(directory, tag):
        matching_folders = []

        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for dir_name in dirs:
                # Check if the folder name contains the tag
                if fnmatch.fnmatch(dir_name, f'*{tag}*'):
                    matching_folders.append(os.path.join(root, dir_name))

        return matching_folders

    def plot_power_comparison(self, fig=None, ax_list=None):
        """
        Can be called using ``cavities.plot_power_comparison()``

        Parameters
        ----------
        fig: matplotlib figure
        ax_list: list of matplotlib axes object

        Returns
        -------

        """
        if fig is not None:
            fig = fig
            ax1, ax2, ax3 = ax_list
        else:
            # create figure
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
        # def E_acc_Pin(self, cavity, E_acc, op_field, ls='-', ):

        # ax_right2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        # ax_right2.spines["right"].set_position(("axes", 1.2))
        for i, cavity in enumerate(self.cavities_list):
            ax1.plot(cavity.E_acc * 1e-6, cavity.pstat / cavity.n_cav,
                     ls=self.ls[i], lw=2, c='tab:orange',
                     label=r"$P_\mathrm{static/cav}$" + fr"{cavity.name}")

            ax1.plot(cavity.E_acc * 1e-6, cavity.pdyn / cavity.n_cav,
                     ls=self.ls[i], lw=2, c='tab:blue', label=r"$P_\mathrm{dynamic/cav}$" + fr"{cavity.name}")

            # p1, = ax1.plot(cavity.E_acc * 1e-6, cavity.p_wp/cavity.n_cav,
            #                ls=self.ls[i], lw=2, c='k', label=r"$P_\mathrm{wp/beam}$" + fr"{cavity.name}")

            p2, = ax2.plot(cavity.E_acc * 1e-6, cavity.n_cav, ls=self.ls[i], lw=2, c='tab:red',
                           label=fr"{cavity.name}")

            p3, = ax3.plot(cavity.E_acc * 1e-6, cavity.p_in * 1e-3, ls=self.ls[i], lw=2, c='tab:purple',
                           label=fr"{cavity.name}")

            ax1.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax1.set_ylabel(r"$P_\mathrm{stat, dyn}$/cav [W]")
            ax2.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax2.set_ylabel(r"$N_\mathrm{cav/beam}$")
            ax3.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax3.set_ylabel(r"$P_\mathrm{in/cav}$ [kW]")

            ax1.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax1.text(cavity.op_field * 1e-6 - 1, 0.3, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90, transform=ax1.get_xaxis_transform())
            ax2.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax2.text(cavity.op_field * 1e-6 - 1, 0.5, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90, transform=ax2.get_xaxis_transform())
            ax3.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax3.text(cavity.op_field * 1e-6 - 1, 0.3, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90,
                     transform=ax3.get_xaxis_transform())

            # ax.axvline(7.13, ls='--', c='k')
            # ax.axvline(10, ls='--', c='k')
            # ax.axvline(15, ls='--', c='k')
            # ax_right2.axhline(500, ls='--', c='k')
            # ax_right2.axhline(1000, ls='--', c='k')
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            # ax_right.yaxis.set_major_locator(ticker.MultipleLocator(100))
            # ax_right2.yaxis.set_major_locator(ticker.MultipleLocator(200))

            # ax.yaxis.label.set_color(p1.get_color())
            # ax_right.yaxis.label.set_color(p2.get_color())
            # ax_right2.yaxis.label.set_color(p3.get_color())

            ax1.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            ax2.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            ax3.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            # # ax.set_ylim(0, 50)
            # ax_right.set_ylim(100, 400)f
            # ax_right2.set_ylim(0, 700)
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set_yscale('log')

            # tkw = dict(size=4, width=1.5)
            # ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
            # ax_right.tick_params(axis='y', colors=p2.get_color(), **tkw)
            # ax_right2.tick_params(axis='y', colors=p3.get_color(), **tkw)
            # ax.tick_params(axis='x', **tkw)

            ax1.minorticks_on()
            # ax.grid(True, which='both', axis='both')

        # dummy lines with NO entries, just to create the black style legend
        dummy_lines = []
        for b_idx, b in enumerate(self.cavities_list):
            dummy_lines.append(ax1.plot([], [], c="gray", ls=self.ls[b_idx])[0])

        lines = ax1.get_lines()
        legend1 = ax1.legend([lines[i] for i in range(3)],
                             [r"$P_\mathrm{stat}$", r"$P_\mathrm{dyn}$"], loc=3)
        legend2 = ax1.legend([dummy_lines[i] for i in range(len(self.cavities_list))],
                             [cavity.name for cavity in self.cavities_list],
                             loc=0)
        ax1.add_artist(legend1)

        # ax1.legend(ncol=len(cavities))
        ax2.legend(loc='upper left')
        ax3.legend(loc=3)

        label = [r"$\mathbf{Z^*}$", 'Z', r"$\mathbf{W^*}$", 'W']
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_comparison.png")

        plt.show()

    def plot_compare_bar(self):
        """
        Plots bar chart of power quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (12, 3)
        # plot barchart
        data = np.array([list(d.values()) for d in self.returned_results])
        data_col_max = data.max(axis=0)

        x = list(self.returned_results[0].keys())
        X = np.arange(len(x))

        fig, ax = plt.subplots()
        ax.margins(x=0)
        width = 0.15  # 1 / len(x)
        for i, cav in enumerate(self.cavities_list):
            ax.bar(X + i * width, data[i] / data_col_max, width=width, label=cav.name)

        ax.set_xticks([r + width for r in range(len(x))], x)
        # label = ["C3794_H (2-Cell)", "C3795_H (5-Cell)"]

        ax.axhline(1.05, c='k')
        ax.set_ylim(-0.01, 1.5 * ax.get_ylim()[-1])
        ax.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_comparison_bar.png")

        plt.show()

    def get_power_qois(self, cav, rf_config, op_points_list, uq=False):
        """

        Parameters
        ----------
        cavity: object
            Cavity object
        op_field: float
            Cavity operating field

        Returns
        -------
        Dictionary containing quantities of interest (normed optional).
        """

        if uq:
            qois = cav.get_power_uq(rf_config, op_points_list)
        else:
            qois = cav.get_power(rf_config, op_points_list)

        return qois

    def plot_compare_power_scatter(self, rf_config, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
        """
        Plots bar chart of power quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = figsize

        self.rf_config = rf_config

        assert len(rf_config['Q0 []']) == len(self.cavities_list), error(
            'Lengh of Q0 [] must equal number of Cavity objects.')
        assert len(rf_config['inv_eta []']) == len(self.cavities_list), error(
            'Lengh of inv_eta [] must equal number of Cavity objects.')
        if 'Eacc [MV/m]' in rf_config.keys():
            assert len(rf_config['Eacc [MV/m]']) == len(self.cavities_list), error('length of accelerating field '
                                                                                   'Eacc [MV/m] must equal number '
                                                                                   'of Cavity objects in Cavities.')
        if 'V [GV]' in rf_config.keys():
            assert len(rf_config['V [GV]']) == len(op_points_list), error('length of RF Voltage '
                                                                          'V [GV] must equal number '
                                                                          'of operating points.')

        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        # display formula used
        display(Math(r'Dims: {}x{}m \\ Area: {}m^2 \\ Volume: {}m^3'.format(2, round(3, 2), 5, 5)))

        for ii, cav in enumerate(self.cavities_list):
            # set cavity intrinsic quality factor and inv_eta
            cav.Q0 = rf_config['Q0 []'][ii]
            cav.inv_eta = rf_config['inv_eta []'][ii]
            if 'Eacc [MV/m]' in rf_config.keys():
                cav.Eacc_rf_config = rf_config['Eacc [MV/m]'][ii]  # change this later, not effective way

            self.power_qois[cav.name] = self.get_power_qois(cav, rf_config, op_points_list)
            self.power_qois_uq[cav.name] = self.get_power_qois(cav, rf_config, op_points_list, uq=True)

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:
            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for op_pt in op_points_list:
                # get nominal qois
                dd_nominal = {}
                for cav, metrics in self.power_qois.items():
                    # for metric, values in metrics[op_pt]['SR'].items():
                    dd_nominal[cav] = metrics[op_pt]['SR']

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.power_qois_uq.items():
                    for metric, values in metrics[op_pt]['SR'].items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                    marker=opt_format[ii],
                                                    fc='none', ec=colors[i], lw=2, zorder=100,
                                                    label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=scatter_points.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_scatter_{op_points_list}.png")

    def plot_compare_eigenmode(self, kind='scatter', uq=False, ncols=3):
        if kind == 'scatter' or kind == 's':
            self.plot_compare_fm_scatter(uq=uq, ncols=ncols)
        if kind == 'bar' or kind == 'b':
            self.plot_compare_fm_bar(uq=uq, ncols=ncols)

    def plot_compare_wakefield(self, opt, kind='scatter', uq=False, ncols=3, figsize=(12, 3)):
        if kind == 'scatter' or kind == 's':
            self.plot_compare_hom_scatter(opt, uq=uq, ncols=ncols, figsize=figsize)
        if kind == 'bar' or kind == 'b':
            self.plot_compare_hom_bar(opt, uq=uq, ncols=ncols)

    # def plot_compare_hom_bar_(self, opt, ncols=3, uq=False):
    #     """
    #     Plot bar chart of higher-order mode's quantities of interest
    #
    #     Returns
    #     -------
    #
    #     """
    #     # plt.rcParams["figure.figsize"] = (15 / 27 * 6 * len(self.cavities_list), 3)
    #     plt.rcParams["figure.figsize"] = (12, 4)
    #
    #     if not uq:
    #         # plot barchart
    #         self.hom_results = self.qois_hom(opt)
    #         df = pd.DataFrame.from_dict(self.hom_results)
    #         fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')
    #         # Plot each column in a separate subplot
    #         labels = [cav.plot_label for cav in self.cavities_list]
    #         for key, ax in axd.items():
    #             ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
    #                    edgecolor='k', width=1)
    #             ax.set_xticklabels([])
    #             ax.set_ylabel(key)
    #             h, l = ax.get_legend_handles_labels()
    #     else:
    #         # Step 1: Flatten the dictionary into a DataFrame
    #         rows = []
    #         for cav, metrics in self.uq_hom_results.items():
    #             for metric, values in metrics.items():
    #                 rows.append({
    #                     'cavity': cav,
    #                     'metric': metric,
    #                     'mean': values['expe'][0],
    #                     'std': values['stdDev'][0]
    #                 })
    #
    #         df = pd.DataFrame(rows)
    #
    #         labels = [cav.plot_label for cav in self.cavities_list]
    #
    #         # Step 2: Create a Mosaic Plot
    #         metrics = df['metric'].unique()
    #         num_metrics = len(metrics)
    #
    #         layout = [[metric for metric in metrics]]
    #         fig, axd = plt.subplot_mosaic(layout, layout='constrained')
    #
    #         # Plot each metric on a separate subplot
    #         for metric, ax in axd.items():
    #             sub_df = df[df['metric'] == metric]
    #             ax.bar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], label=labels, capsize=5,
    #                    color=matplotlib.colormaps['Set2'].colors[:len(df)], edgecolor='k',
    #                    width=1)
    #
    #             ax.set_xticklabels([])
    #             ax.set_xticks([])
    #             ax.set_ylabel(metric)
    #             h, l = ax.get_legend_handles_labels()
    #
    #     if not ncols:
    #         ncols = min(4, len(self.cavities_list))
    #     fig.legend(h, l, loc='outside upper center', borderaxespad=0, ncol=ncols)
    #
    #     # fig.set_tight_layout(True)
    #
    #     # save plots
    #     fname = [cav.name for cav in self.cavities_list]
    #     fname = '_'.join(fname)
    #
    #     self.save_all_plots(f"{fname}_hom_bar.png")
    #
    #     return axd

    def plot_compare_hom_bar(self, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        plt.rcParams["figure.figsize"] = figsize
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.bar(df.index, df[key], color=colors[i], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:

            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for opt in op_points_list:

                # get nominal qois
                dd_nominal = {}
                for cav, ops_id in self.wakefield_qois.items():
                    for kk, vv in ops_id.items():
                        if fr'{opt}_SR' in kk:
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in metrics[opt]['SR'].items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        bar = ax.bar(sub_df['cavity'], sub_df['mean'], color=colors[i],
                                     fc='none', ec=colors[i], lw=2, zorder=100,
                                     label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=bar.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_hom_scatter_{op_points_list}.png")

        return axd

    def plot_compare_hom_scatter(self, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        plt.rcParams["figure.figsize"] = figsize
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:

            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for opt in op_points_list:

                # get nominal qois
                dd_nominal = {}
                for cav, ops_id in self.wakefield_qois.items():
                    for kk, vv in ops_id.items():
                        if fr'{opt}_SR' in kk:
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in metrics[opt]['SR'].items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                    marker=opt_format[ii],
                                                    fc='none', ec=colors[i], lw=2, zorder=100,
                                                    label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=scatter_points.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_hom_scatter_{op_points_list}.png")

        return axd

    def plot_compare_fm_bar(self, ncols=3, uq=False):
        """
        Plot bar chart of fundamental mode quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (12, 4)

        if not uq:
            self.fm_results = self.qois_fm()

            df = pd.DataFrame.from_dict(self.fm_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            # Plot each column in a separate subplot
            for key, ax in axd.items():
                ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
                       edgecolor='k',
                       width=1)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS[key])
                h, l = ax.get_legend_handles_labels()
        else:
            self.fm_results = self.uq_fm_results

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cav, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cav,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            labels = [cav.plot_label for cav in self.cavities_list]

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            num_metrics = len(metrics)

            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            for metric, ax in axd.items():
                sub_df = df[df['metric'] == metric]
                ax.bar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], label=labels, capsize=5,
                       color=matplotlib.colormaps['Set2'].colors[:len(df)], edgecolor='k',
                       width=1)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS[metric])
                h, l = ax.get_legend_handles_labels()

        # fig.set_tight_layout(True)
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_fm_bar.png")

        return axd

    def plot_compare_fm_scatter(self, ncols=3, uq=False):
        """
        Plot scatter chart of higher-order mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        plt.rcParams["figure.figsize"] = (12, 3)

        if not uq:
            self.fm_results = self.qois_fm()
            df = pd.DataFrame.from_dict(self.fm_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index, df[key], color=colors[i], label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS[key])

            h, l = ax.get_legend_handles_labels()
        else:
            df_nominal = pd.DataFrame.from_dict(self.eigenmode_qois).T

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cav, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cav,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            labels = [cav.plot_label for cav in self.cavities_list]
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                    scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                fc='none', ec=colors[i], label=label, lw=2, zorder=100)
                    ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], capsize=10, lw=2,
                                color=scatter_points.get_edgecolor()[0])

                    # plot nominal
                    ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none',
                               ec='k', lw=1,
                               zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            h, l = ax.get_legend_handles_labels()

        # by_label = dict(zip(l, h))
        # if 'nominal' in by_label.keys():
        #     nominal_handle = by_label.pop('nominal')  # Remove 'nominal' entry
        #     # Reinsert 'nominal' as the last entry
        #     by_label['nominal'] = nominal_handle

        # Set legend
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_fm_scatter.png")

        return axd

    def plot_compare_all_scatter(self, opt, ncols=3):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        plt.rcParams["figure.figsize"] = (16, 3)

        if not uq:
            self.hom_results = self.qois_hom(opt)

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:
            # get nominal qois
            dd_nominal = {}
            for cav, ops_id in self.wakefield_qois.items():
                for kk, vv in ops_id.items():
                    if fr'{opt}_SR' in kk:
                        dd_nominal[cav] = vv

            dict_all_nominal = {key: {**self.eigenmode_qois.get(key, {}), **dd_nominal.get(key, {})} for key in
                                set(self.eigenmode_config) | set(dd_nominal)}
            df_nominal = pd.DataFrame.from_dict(dict_all_nominal).T

            dict_all = self.uq_fm_results | self.uq_hom_results

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cavity, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cavity,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            for cavity, metrics in self.uq_hom_results.items():
                for metric, values in metrics[opt]['SR'].items():
                    rows.append({
                        'cavity': cavity,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plotting labels and colors
            labels = df['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]

            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                    scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                ec='k', zorder=100, label=label)
                    ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt='o', capsize=10,
                                color=scatter_points.get_facecolor()[0])

                    # plot nominal
                    ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none',
                               ec='k', lw=2,
                               zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        # by_label = dict(zip(l, h))
        # if 'nominal' in by_label.keys():
        #     nominal_handle = by_label.pop('nominal')
        #     # Reinsert 'nominal' as the last entry
        #     by_label['nominal'] = nominal_handle

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_all_scatter_{opt}.png")

        return axd

    def plot_compare_all_bar(self, opt, ncols=3):
        """
        Plot bar chart of fundamental mode quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (15, 3)
        # plot barchart
        self.all_results = self.qois_all(opt)

        df = pd.DataFrame.from_dict(self.all_results)
        fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

        labels = [cav.plot_label for cav in self.cavities_list]
        # Plot each column in a separate subplot
        for key, ax in axd.items():
            ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
                   edgecolor='k', width=1)
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_ylabel(key)
            h, l = ax.get_legend_handles_labels()

        # fig.set_tight_layout(True)
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(h, l, loc='outside upper center', borderaxespad=0, ncol=ncols)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_all_bar_{opt}.png")

        return axd

    def plot_cavities_contour(self, opt='mid', figsize=None):
        """Plot geometric contour of Cavity objects

        Parameters
        ----------
        opt: {"mid", "end", "all"}
            Either plot contour for only mid cells or end cells or the entire cavity

        Returns
        -------

        """
        min_x, max_x, min_y, max_y = [], [], [], []

        if figsize:
            fig, axs = plt.subplot_mosaic([[0]], figsize=figsize,
                                          layout='constrained')
        else:
            fig, axs = plt.subplot_mosaic([[0]], figsize=(12, 4),
                                          layout='constrained')
        ax = axs[0]
        ax.set_aspect('equal', adjustable='box')
        ax.margins(x=0)

        for i, cav in enumerate(self.cavities_list):
            if opt.lower() == 'mid':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['IC'])
                beampipe = 'none'
            elif opt.lower() == 'end':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'right'
            else:
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['OC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'both'

            if cav.freq:
                scale = (cav.freq * 1e6) / c0
            else:
                scale = 1
            if cav.cell_parameterisation == 'flattop':
                write_cavity_geometry_cli_flattop(mid_cell, end_cell_left, end_cell_right,
                                                  BP=beampipe, n_cell=1, ax=ax, scale=scale, plot=True, contour=True,
                                                  lw=4, c=cav.color)
            else:
                write_cavity_geometry_cli(mid_cell, end_cell_left, end_cell_right,
                                          BP=beampipe, n_cell=1, ax=ax, scale=scale, plot=True, contour=True,
                                          lw=4, c=cav.color)
            ax.lines[-1].set_label(cav.name)
            ax.axvline(0, c='k', ls='--')
            ax.legend(loc='upper right')

            x_label = r"$z/\lambda$"
            y_label = r"$r/\lambda$"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            min_x.append(min(ax.lines[-1].get_xdata()))
            min_y.append(min(ax.lines[-1].get_ydata()))
            max_x.append(max(ax.lines[-1].get_xdata()))
            max_y.append(max(ax.lines[-1].get_ydata()))

        ax.set_ylim(0)

        # plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_contour_{opt}.png")

        # fig.show()

    def plot_cavities_contour_dimension(self, opt='mid', figsize=None):
        """Plot geometric contour of Cavity objects

        Parameters
        ----------
        opt: {"mid", "end", "all"}
            Either plot contour for only mid cells or end cells or the entire cavity

        Returns
        -------

        """
        min_x, max_x, min_y, max_y = [], [], [], []

        if figsize:
            fig, axs = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], figsize=figsize,
                                          layout='constrained')
        else:
            fig, axs = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], figsize=(12, 4),
                                          layout='constrained')

        for i, (cav, key) in enumerate(zip(self.cavities_list, axs)):
            ax = axs[key]
            ax.set_aspect('equal')
            if opt.lower() == 'mid':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['IC'])
                beampipe = 'none'
            elif opt.lower() == 'end':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'right'
            else:
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['OC'])
                end_cell_right = np.array(cav.shape['OC_R'])
                beampipe = 'both'

            if cav.cell_parameterisation == 'flattop':
                write_cavity_geometry_cli_flattop(mid_cell * 1e3, end_cell_left * 1e3, end_cell_right * 1e3,
                                                  BP=beampipe, n_cell=1, ax=ax, scale=1, plot=True,
                                                  dimension=True, lw=3, c='k')
            else:
                write_cavity_geometry_cli(mid_cell * 1e3, end_cell_left * 1e3, end_cell_right * 1e3,
                                          BP=beampipe, n_cell=1, ax=ax, scale=1, plot=True,
                                          dimension=True, lw=3, c='k')
            ax.lines[-1].set_label(cav.name)
            ax.legend(loc='upper right')

            x_label = r"$z$ [mm]"
            y_label = r"$r$ [mm]"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            min_x.append(min(ax.lines[-1].get_xdata()))
            min_y.append(min(ax.lines[-1].get_ydata()))
            max_x.append(max(ax.lines[-1].get_xdata()))
            max_y.append(max(ax.lines[-1].get_ydata()))

            ax.set_ylim(0)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_dimension.png")

        # fig.show()

    def plot_axis_fields(self):
        """
        Plot axis fields of cavities

        Returns
        -------

        """
        for cav in self.cavities_list:
            # normalize fields
            e_axis = np.abs(cav.Ez_0_abs['|Ez(0, 0)|'])
            e_axis_norm = e_axis / e_axis.max()

            # shift to mid
            z = cav.Ez_0_abs['z(0, 0)']
            z_shift = z - z.max() / 2
            plt.plot(z_shift, e_axis_norm, label=cav.name)

        plt.xlabel('$z$ [mm]')
        plt.ylabel('$|E_\mathrm{0, z}|/|E_\mathrm{0, z}|_\mathrm{max}$')
        plt.axhline(1.02, c='k')
        plt.ylim(-0.01, 1.5)
        plt.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_axis_fields.png")

        plt.show()

    def plot_surface_fields(self):
        """
        Plot surface fields of cavities

        Returns
        -------

        """
        for cav in self.cavities_list:
            # normalize fields
            e_surf = np.abs(cav.surface_field['0'])
            e_surf_norm = e_surf / e_surf.max()

            plt.plot(e_surf_norm, label=cav.name)

        plt.axhline(1.02, c='k')
        plt.ylim(-0.01, 1.5)
        plt.xlabel('$L_\mathrm{surf}$ [mm]')
        plt.ylabel('$|E_\mathrm{surf}|/|E_\mathrm{surf}|_\mathrm{max}$')
        plt.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_surface_fields.png")

        plt.show()

    def plot_multipac_triplot(self, folders, kind='triplot'):
        """
        Plot Multipac triplot

        Parameters
        ----------
        folders: list, array like
            List of folder to read multipacting results from
        kind

        Notes
        -----
        This will be changed later so that the multipac results will be in the same location as the SLANS and ABCI
        results

        Returns
        -------

        """

        if kind == 'triplot':
            # create figure
            fig = plt.figure()
            gs = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
            axs = [ax1, ax2, ax3]

        else:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]

        mpl.rcParams['figure.figsize'] = [6, 10]

        Eacc_list = [cav.op_field * 1e-6 for cav in self.cavities_list]
        Epk_Eacc_list = [cav.e for cav in self.cavities_list]
        labels = [cav.name for cav in self.cavities_list]
        for Eacc, Epk_Eacc, folder, label in zip(Eacc_list, Epk_Eacc_list, folders, labels):
            # load_output_data
            # files
            fnames = ["Ccounter.mat", "Acounter.mat", "Atcounter.mat", "Efcounter.mat", "param",
                      "geodata.n", "secy1", "counter_flevels.mat", "counter_initials.mat"]
            data = {}
            # files_folder = "D:\Dropbox\multipacting\MPGUI21"
            for f in fnames:
                if ".mat" in f:
                    data[f] = spio.loadmat(fr"{folder}\\{f}")
                else:
                    data[f] = pd.read_csv(fr"{folder}\\{f}", sep='\\s+', header=None)

            A = data["Acounter.mat"]["A"]
            At = data["Atcounter.mat"]["At"]
            C = data["Ccounter.mat"]["C"]
            Ef = data["Efcounter.mat"]["Ef"]
            flevel = data["counter_flevels.mat"]["flevel"]
            initials = data["counter_initials.mat"]["initials"]
            secy1 = data["secy1"].to_numpy()
            Pow = flevel
            n = len(initials[:, 0]) / 2  # number of initials in the bright set
            N = int(data["param"].to_numpy()[4])  # number of impacts
            U = flevel
            Efl = flevel
            q = 1.6021773e-19
            Efq = Ef / q

            e1 = np.min(np.where(secy1[:, 1] >= 1))  # lower threshold
            e2 = np.max(np.where(secy1[:, 1] >= 1))  # upper threshold
            val, e3 = np.max(secy1[:, 1]), np.argmax(secy1[:, 1])  # maximum secondary yield

            cl = 0
            ok, ok1, ok2 = 1, 1, 1
            if ok > 0:
                if n == 0:
                    error('Unable to plot the counters. No initial points.')
                    return

                if ok1 * ok2 == 0:
                    cl = error('Counter functions or impact energy missing.')
                else:
                    # if ss > 0:
                    #     cl = error(np.array(['Plotting the triplot (counter, enhanced ', 'counter and impact energy).']))

                    if kind == 'counter function' or kind == 'triplot':
                        # fig, axs = plt.subplots(3)
                        axs[0].plot(Efl / 1e6, C / n, lw=2, label=label)
                        axs[0].set_ylabel("$c_" + "{" + f"{N}" + "}/ c_0 $")
                        axs[0].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[0].set_title(r'$\mathbf{MultiPac 2.1~~~~~Counter function~~~~}$')
                        axs[0].set_xlim(np.amin(Efl) / 1e6, np.amax(Efl) / 1e6)
                        axs[0].set_ylim(0, np.max([0.1, axs[0].get_ylim()[1]]))

                        # plot peak operating field
                        axs[0].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[0].text(np.round(Eacc * Epk_Eacc, 2) - 1.5, 0.1,
                                    f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                    size=12, rotation=90,
                                    transform=axs[0].get_xaxis_transform())

                        axs[0].minorticks_on()
                        axs[0].legend(loc='upper left')

                    if kind == 'final impact energy' or kind == 'triplot':
                        s = 0
                        if kind == 'final impact energy':
                            s = 1
                        axs[1 - s].semilogy(Efl / 1e6, Efq, lw=2, label=label)

                        # axs[1-s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e1, 0], secy1[e1, 0]], '-r')
                        e0 = sci.interp1d(secy1[0:e1 + 1, 1], secy1[0:e1 + 1, 0])(1)
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [e0, e0], '-r')
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e2, 0], secy1[e2, 0]], '-r')
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e3, 0], secy1[e3, 0]], '--r')

                        axs[1 - s].set_ylabel("$Ef_" + "{" + f"{N}" + "}$")
                        axs[1 - s].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[1-s].set_title('$\mathbf{Final~Impact~Energy~in~eV}$')
                        axs[1 - s].set_xlim(np.min(Efl) / 1e6, np.max(Efl) / 1e6)
                        axs[1 - s].set_ylim(0, axs[1 - s].get_ylim()[1])

                        axs[1 - s].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[1 - s].text(np.round(Eacc * Epk_Eacc, 2) - 1.5, 0.1,
                                        f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                        size=12, rotation=90,
                                        transform=axs[1 - s].get_xaxis_transform())

                        axs[1 - s].minorticks_on()
                        axs[1 - s].legend(loc='upper left')
                    if kind == 'enhanced counter function' or kind == 'triplot':
                        s = 0
                        if kind == 'enhanced counter function':
                            s = 2
                        axs[2 - s].semilogy(Efl / 1e6, (A + 1) / n, lw=2, label=label)
                        axs[2 - s].set_xlabel('$V$ [MV]')
                        axs[2 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [1, 1], '-r')
                        axs[2 - s].set_xlim(np.min(Efl) / 1e6, np.max(Efl) / 1e6)
                        axs[2 - s].set_ylim(np.min((A + 1) / n), axs[2 - s].get_ylim()[1])
                        axs[2 - s].set_ylabel("$e_" + "{" + f"{N}" + "}" + "/ c_0$")
                        axs[2 - s].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[2-s].set_title('$\mathbf{Enhanced~counter~function}$')

                        axs[2 - s].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[2 - s].text(np.round(Eacc * Epk_Eacc, 2) - 1, 0.1,
                                        f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                        size=12, rotation=90,
                                        transform=axs[2 - s].get_xaxis_transform())

                        axs[2 - s].minorticks_on()
                        axs[2 - s].legend(loc='upper left')

        fig.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_{kind.replace(' ', '_')}.png")

        plt.show()

    # def plot_dispersion(self):
    #     """
    #     Plot dispersion curve for the cavities
    #
    #     Returns
    #     -------
    #
    #     """
    #     fig, ax = plt.subplots()
    #     ax.margins(x=0)
    #     for cav in self.cavities_list:
    #         x = range(1, cav.n_cells + 1)
    #         ax.plot(x, cav.d_slans_all_results['FREQUENCY'][0:cav.n_cells], marker='o', mec='k',
    #                 label=f'{cav.name} (kcc={round(cav.k_cc, 2)} %)')
    #         ax.set_xlabel('Mode Number')
    #         ax.set_ylabel('Frequency [MHz]')
    #
    #     plt.legend()
    #
    #     # save plots
    #     fname = [cav.name for cav in self.cavities_list]
    #     fname = '_'.join(fname)
    #
    #     self.save_all_plots(f"{fname}_dispersion.png")
    #
    #     plt.show()
    #
    # # def ql_pin(self, labels, geometry, RF, QOI, Machine, p_data=None):
    # #     """
    # #     Calculate the value of input power as a function of loaded quality factor
    # #
    # #     Parameters
    # #     ----------
    # #     labels: list, array like
    # #         Descriptive labels on matplotlib plot
    # #     geometry: list, array like
    # #         List of grouped geometric input parameters
    # #     RF: list, array like
    # #         List of grouped radio-frequency (RF) properties
    # #     QOI:
    # #         List of quantities of interest for cavities
    # #     Machine:
    # #         List of grouped machine related materials
    # #     p_data:
    # #
    # #
    # #     Returns
    # #     -------
    # #
    # #     """
    # #     # check if entries are of same length
    # #
    # #     it = iter(geometry)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(RF)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(QOI)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(Machine)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     n_cells, l_cells, G, b = [np.array(x) for x in geometry]
    # #     E_acc, Vrf = [np.array(x) for x in RF]
    # #
    # #     fig, ax = plt.subplots()
    # #     ax.margins(x=0)
    # #
    # #     # QOI
    # #     f0, R_Q = [np.array(x) for x in QOI]
    # #
    # #     # Machine
    # #     I0, rho, E0 = [np.array(x) for x in Machine]
    # #
    # #     l_active = 2 * n_cells * l_cells
    # #     l_cavity = l_active + 8 * l_cells
    # #
    # #     # CALCULATED
    # #     v_cav = E_acc * l_active
    # #
    # #     U_loss = 88.46 * E0 ** 4 / rho * 1e-6  # GeV # energy lost per turn per beam
    # #     v_loss = U_loss * 1e9  # V # v loss per beam
    # #
    # #     phi = np.arccos(v_loss / Vrf)
    # #     delta_f = -R_Q * f0 * I0 * np.sin(phi) / (2 * v_cav)  # optimal df
    # #     QL_0_x = v_cav / (R_Q * I0 * np.cos(phi))  # optimal Q loaded
    # #
    # #     QL_0 = np.linspace(1e4, 1e9, 1000000)
    # #
    # #     xy_list = [(0.15, 0.13), (0.1, 0.16), (0.1, 0.19), (0.1, 0.21)]
    # #     for i in range(len(E_acc)):
    # #         f1_2 = f0[i] / (2 * QL_0)  # 380.6
    #
    # #         pin = v_cav[i] ** 2 / (4 * R_Q[i] * QL_0) * \
    # #               ((1 + ((R_Q[i] * QL_0 * I0[i]) / v_cav[i]) * np.cos(phi[i])) ** 2 +
    # #                ((delta_f[i] / f1_2) + ((R_Q[i] * QL_0 * I0[i]) / v_cav[i]) * np.sin(phi[i])) ** 2)
    # #
    # #         # material/ wall power
    # #         e_acc = np.linspace(0.5, 25, 1000) * 1e6  # MV/m
    # #
    # #         txt = labels[i]
    # #
    # #         if "*" in labels[i]:
    # #             l = ax.plot(QL_0, pin * 1e-3, label=txt, lw=4,
    # #                         ls='--')
    # #         else:
    # #             l = ax.plot(QL_0, pin * 1e-3, label=txt, lw=4)
    # #
    # #         # add annotations
    # #
    # #         # annotext = ax.annotate(txt, xy=xy_list[i], xycoords='figure fraction', size=8, rotation=0,
    # #         #                        c=l[0].get_color())
    # #
    # #     if p_data:
    # #         # plot QL with penetration
    # #         ax_2 = ax.twinx()
    # #         data = fr.excel_reader(p_data)
    # #         data_ = data[list(data.keys())[0]]
    # #         ax_2.plot(data_["QL"], data_["penetration"], lw=4)
    # #
    # #     # plot decorations
    # #     ax.set_xlabel(r"$Q_{L,0}$")
    # #     ax.set_ylabel(r"$P_\mathrm{in} ~[\mathrm{kW}]$")
    # #     ax.set_xscale('log')
    # #     ax.set_xlim(5e3, 1e9)
    # #     ax.set_ylim(0, 3000)
    # #     ax.legend(loc='upper left')  #
    # #     ax.minorticks_on()
    # #     # ax.grid(which='both')
    # #     fig.show()

    def run_abci(self):
        for cav in self.cavities_list:
            cav.run_abci()

    def run_multipacting(self):
        for cav in self.cavities_list:
            cav.run_multipacting()

    def define_operating_points(self, ops):
        self.operating_points = ops

    @staticmethod
    def linspace(start, stop, step=1.):
        """
        Like np.linspace but uses step instead of num
        This is inclusive to stop, so if start=1, stop=3, step=0.5
        Output is: array([1., 1.5, 2., 2.5, 3.])
        """
        if start < stop:
            ll = np.linspace(start, stop, int((stop - start) / abs(step) + 1))
            if stop not in ll:
                ll = np.append(ll, stop)

            return ll
        else:
            ll = np.linspace(stop, start, int((start - stop) / abs(step) + 1))
            if start not in ll:
                ll = np.append(ll, start)
            return ll

    @staticmethod
    def lineTo(prevPt, nextPt, step):
        if prevPt[0] == nextPt[0]:
            # vertical line
            # chwxk id nextPt is greater
            if prevPt[1] < nextPt[1]:
                py = np.linspace(prevPt[1], nextPt[1], step)
            else:
                py = np.linspace(nextPt[1], prevPt[1], step)
                py = py[::-1]
            px = np.ones(len(py)) * prevPt[0]

        elif prevPt[1] == nextPt[1]:
            # horizontal line
            if prevPt[0] < nextPt[1]:
                px = np.linspace(prevPt[0], nextPt[0], step)
            else:
                px = np.linspace(nextPt[0], prevPt[0], step)

            py = np.ones(len(px)) * prevPt[1]
        else:
            # calculate angle to get appropriate step size for x and y
            ang = np.arctan((nextPt[1] - prevPt[1]) / (nextPt[0] - prevPt[0]))
            if prevPt[0] < nextPt[0] and prevPt[1] < nextPt[1]:
                px = np.arange(prevPt[0], nextPt[0], step * np.cos(ang))
                py = np.arange(prevPt[1], nextPt[1], step * np.sin(ang))
            elif prevPt[0] > nextPt[0] and prevPt[1] < nextPt[1]:
                px = np.arange(nextPt[0], prevPt[0], step * np.cos(ang))
                px = px[::-1]
                py = np.arange(prevPt[1], nextPt[1], step * np.sin(ang))
            elif prevPt[0] < nextPt[0] and prevPt[1] > nextPt[1]:
                px = np.arange(prevPt[0], nextPt[0], step * np.cos(ang))
                py = np.arange(nextPt[1], prevPt[1], step * np.sin(ang))
                py = py[::-1]
            else:
                px = np.arange(nextPt[0], prevPt[0], step * np.cos(ang))
                px = px[::-1]
                py = np.arange(nextPt[1], prevPt[1], step * np.sin(ang))
                py = py[::-1]

        # plt.plot(px, py)

    @staticmethod
    def arcTo2(x_center, y_center, a, b, step, start_angle, end_angle):
        u = x_center  # x-position of the center
        v = y_center  # y-position of the center
        a = a  # radius on the x-axis
        b = b  # radius on the y-axis
        sa = (start_angle / 360) * 2 * np.pi  # convert angle to radians
        ea = (end_angle / 360) * 2 * np.pi  # convert angle to radians

        if ea < sa:
            # end point of curve
            x_end, y_end = u + a * np.cos(sa), v + b * np.sin(sa)

            t = np.arange(ea, sa, np.pi / 100)
            # t = np.linspace(ea, sa, 100)
            # check if end angle is included, include if not
            if sa not in t:
                t = np.append(t, sa)
            t = t[::-1]
        else:
            # end point of curve
            x_end, y_end = u + a * np.cos(ea), v + b * np.sin(ea)

            t = np.arange(sa, ea, np.pi / 100)
            # t = np.linspace(ea, sa, 100)
            if ea not in t:
                t = np.append(t, ea)

        return [x_end, y_end]

    @staticmethod
    def arcTo(x_center, y_center, a, b, step, start, end):
        u = x_center  # x-position of the center
        v = y_center  # y-position of the center
        a = a  # radius on the x-axis
        b = b  # radius on the y-axis

        t = np.arange(0, 2 * np.pi, np.pi / 100)

        x = u + a * np.cos(t)
        y = v + b * np.sin(t)
        pts = np.column_stack((x, y))
        inidx = np.all(np.logical_and(np.array(start) < pts, pts < np.array(end)), axis=1)
        inbox = pts[inidx]
        inbox = inbox[inbox[:, 0].argsort()]

        # plt.plot(inbox[:, 0], inbox[:, 1])

        return inbox

    def make_latex_summary_tables(self, which='geometry', op_pts_list=None):
        fname = [cav.name for cav in self.cavities_list]
        # create new sub directory
        if not os.path.exists(fr"{self.projectDir}\PostprocessingData\Data\{'_'.join(fname)}"):
            os.mkdir(fr"{self.projectDir}\PostprocessingData\Data\{'_'.join(fname)}")
        try:
            if which == 'geometry':
                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters of " + fr"{', '.join(fname)}" + " cavity geometries.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{''.join(['c' for i in self.cavities_list])}" + "|}"
                toprule = r"\toprule"
                header = r" ".join([fr"& {cav.name} " for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"
                A = r"$A$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][0], 2)}/{round(cav.shape['OC'][0], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                B = r"$B$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][1], 2)}/{round(cav.shape['OC'][1], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                a = r"$a$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][2], 2)}/{round(cav.shape['OC'][2], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                b = r"$b$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][3], 2)}/{round(cav.shape['OC'][3], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Ri = r"$R_\mathrm{i}$ " + "".join(
                    [fr"& {round(cav.shape['IC'][4], 2)}/{round(cav.shape['OC'][4], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                L = r"$L$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][5], 2)}/{round(cav.shape['OC'][5], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Req = r"$R_\mathrm{eq}$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][6], 2)}/{round(cav.shape['OC'][6], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                alpha = r"$ \alpha [^\circ]$" + "".join(
                    [fr"& {round(cav.shape['IC'][7], 2)}/{round(cav.shape['OC'][7], 2)} " for cav in
                     self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} geometric properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header,
                                     hhline,
                                     A, B, a, b, Ri, L, Req, alpha,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_geom_latex.tex", 'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'fm':
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{FM QoIs of " + fr"{', '.join(fname)}" + " cavity geometries.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{''.join(['c' for i in self.cavities_list])}" + "|}"
                toprule = r"\toprule"
                header = r" ".join([fr"& {cav.name} " for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. [MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for cav in self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~[\Omega$] " + "".join([fr"& {round(cav.R_Q, 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(cav.G, 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(cav.k_cc, 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(cav.e, 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for cav in self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} fm properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header,
                                     hhline,
                                     rf_freq, R_Q, G, GR_Q, kcc, epk, bpk,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)
                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_fm_latex.tex", 'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'hom':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())
                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{HOM QoIs of " + fr"{', '.join(fname)}" + " cavity geometries for the " + fr"{', '.join([LABELS[op_pt] for op_pt in op_pts_list])}" + " operating points.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c|}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                kfm = r"$|k_\mathrm{FM}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} hom properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_hom_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'qois':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters and QoIs of cavities.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. ~[MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for cav in self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~\mathrm{[\Omega$]} " + "".join(
                    [fr"& {round(cav.R_Q, 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(cav.G, 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(cav.k_cc, 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(cav.e, 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for cav in self.cavities_list]) + r" \\"

                kfm = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}~[]$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ ~[kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: selected shape}"
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     rf_freq, R_Q, G, GR_Q, kcc, epk, bpk,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hhline, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom, PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_qois_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'power':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())

                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Static, dynamic and HOM power loss and input power per cavity for " + fr"{', '.join(fname)}" + " cavity geometries for the " + fr"{', '.join([LABELS[op_pt] for op_pt in op_pts_list])}" + " operating points.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c|}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. [MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"

                if 'V [GV]' in self.rf_config.keys():
                    Vrf = r"$V_\mathrm{RF}$ [GV] " + "".join(
                        [fr"& {round(self.rf_config['V [GV]'][ii], 2):.2f} " for ii in range(len(op_pts_list)) for _ in
                         self.cavities_list]) + r" \\"
                else:
                    Vrf = r"$V_\mathrm{RF}$ [GV] " + "".join(
                        [fr"& {round(self.operating_points[op_pt]['V [GV]'], 2):.2f} " for op_pt in op_pts_list for _ in
                         self.cavities_list]) + r" \\"

                Eacc = r"$E_\mathrm{acc}$ [MV/m] " + "".join(
                    [fr"& {round(cav.Eacc_rf_config, 2)} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$~[]" + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} ~[\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} power properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline, rf_freq, Vrf, Eacc, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom,
                                     # PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_power_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            else:
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters and QoIs of cavities.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"
                A = r"$A$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][0], 2)}/{round(cav.shape['OC'][0], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                B = r"$B$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][1], 2)}/{round(cav.shape['OC'][1], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                a = r"$a$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][2], 2)}/{round(cav.shape['OC'][2], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                b = r"$b$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][3], 2)}/{round(cav.shape['OC'][3], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Ri = r"$R_\mathrm{i}$ " + "".join(
                    [fr"& {round(cav.shape['IC'][4], 2)}/{round(cav.shape['OC'][4], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                L = r"$L$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][5], 2)}/{round(cav.shape['OC'][5], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Req = r"$R_\mathrm{eq}$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][6], 2)}/{round(cav.shape['OC'][6], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                alpha = r"$ \alpha [^\circ]$" + "".join(
                    [fr"& {round(cav.shape['IC'][7], 2)}/{round(cav.shape['OC'][7], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                rf_freq = r"RF Freq. ~[MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                Vrf = r"$V_\mathrm{RF}$ ~[GV] " + "".join(
                    [fr"& {round(self.operating_points[op_pt]['V [GV]'], 2):.2f} " for op_pt in op_pts_list for _ in
                     self.cavities_list]) + r" \\"

                Eacc = r"$E_\mathrm{acc}$ [MV/m] " + "".join(
                    [fr"& {round(self.operating_points[op_pt]['Eacc [MV/m]'], 2)} " for op_pt in op_pts_list for _ in
                     self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~[\Omega$] " + "".join(
                    [fr"& {round(cav.R_Q, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                G = r"$G$ [$\Omega$] " + "".join(
                    [fr"& {round(cav.G, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ [] " + "".join(
                    [fr"& {round(cav.e, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} \left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"

                kfm = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                #     [fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in cav.k_loss])} " for cav in
                #      self.cavities_list]) + r" \\"
                #
                # kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                #     [fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in cav.k_kick])} " for cav in
                #      self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                #     [fr"& {'/'.join([str(round(phom, 4)) for phom in cav.phom])} " for cav in self.cavities_list]) + r" \\"

                phom_values = np.array([[phom for phom in cav.phom] for cav in self.cavities_list])

                # if len(phom_values) % 2 == 0:
                #     result_array = phom_values[1::2] - phom_values[::2]
                #
                #     Phom_diff = r"$\Delta P_\mathrm{HOM}$/cav [W] & \multicolumn{2}{c|}{" + " & \multicolumn{2}{c|}{".join(
                #         [fr"{'/'.join([str(round(val * 1e3, 2)) for val in arr])} " + '}' for arr in result_array]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                #     [fr"& {'/'.join([str(round(phom * cav.n_cav_op_field, 2)) for phom in cav.phom])} " for cav in
                #      self.cavities_list]) + r" \\"

                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: selected shape}"
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     A, B, a, b, Ri, L, Req, alpha,
                                     hhline,
                                     rf_freq, Vrf, Eacc, R_Q, G, GR_Q, epk, bpk,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hhline, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom, PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_all_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')

        except KeyError as e:
            error("Either NGSolve or ABCI results not available. Please use '<cav>.set_eigenmode_qois(<folder>)' "
                  "or '<cav>.set_wakefield_qois(<folder>)' to fix this. Error: ", e)

    def make_excel_summary(self):
        try:
            data = {'Name': [cav.name for cav in self.cavities_list],
                    'Project': [cav.project for cav in self.cavities_list],
                    'Type': [cav.type for cav in self.cavities_list],
                    'CW/Pulsed': [cav.cw_pulsed for cav in self.cavities_list],
                    'Material': [cav.material for cav in self.cavities_list],
                    'N_cells': [cav.n_cells for cav in self.cavities_list],
                    'Freq [MHz]': [cav.op_freq for cav in self.cavities_list],
                    'Beta': [cav.beta for cav in self.cavities_list],
                    'T_oper [K]': [cav.op_temp for cav in self.cavities_list],
                    'I0 [mA]': [cav.I0 for cav in self.cavities_list],
                    'sigma [mm]': [cav.sigma for cav in self.cavities_list],
                    'A_i [mm]': [round(cav.shape_space['IC'][0], 2) for cav in self.cavities_list],
                    'B_i [mm]': [round(cav.shape_space['IC'][1], 2) for cav in self.cavities_list],
                    'a_i [mm]': [round(cav.shape_space['IC'][2], 2) for cav in self.cavities_list],
                    'b_i [mm]': [round(cav.shape_space['IC'][3], 2) for cav in self.cavities_list],
                    'R_i [mm]': [round(cav.shape_space['IC'][4], 2) for cav in self.cavities_list],
                    'L_i [mm]': [round(cav.shape_space['IC'][5], 2) for cav in self.cavities_list],
                    'Req [mm]': [round(cav.shape_space['IC'][6], 2) for cav in self.cavities_list],
                    'alpha_i [deg]': [round(cav.shape_space['IC'][7], 2) for cav in self.cavities_list],
                    'A_el [mm]': [round(cav.shape_space['OC'][0], 2) for cav in self.cavities_list],
                    'B_el [mm]': [round(cav.shape_space['OC'][1], 2) for cav in self.cavities_list],
                    'a_el [mm]': [round(cav.shape_space['OC'][2], 2) for cav in self.cavities_list],
                    'b_el [mm]': [round(cav.shape_space['OC'][3], 2) for cav in self.cavities_list],
                    'R_el [mm]': [round(cav.shape_space['OC'][4], 2) for cav in self.cavities_list],
                    'L_el [mm]': [round(cav.shape_space['OC'][5], 2) for cav in self.cavities_list],
                    # 'Req [mm]': [round(cav.shape_space['OC'][6], 2) for cav in self.cavities_list],
                    'alpha__el [deg]': [round(cav.shape_space['OC'][7], 2) for cav in self.cavities_list],
                    'A_er [mm]': [round(cav.shape_space['OC'][0], 2) for cav in self.cavities_list],
                    'B_er [mm]': [round(cav.shape_space['OC'][1], 2) for cav in self.cavities_list],
                    'a_er [mm]': [round(cav.shape_space['OC'][2], 2) for cav in self.cavities_list],
                    'b_er [mm]': [round(cav.shape_space['OC'][3], 2) for cav in self.cavities_list],
                    'R_er [mm]': [round(cav.shape_space['OC'][4], 2) for cav in self.cavities_list],
                    'L_er [mm]': [round(cav.shape_space['OC'][5], 2) for cav in self.cavities_list],
                    # 'Req [mm]': [round(cav.shape_space['OC'][6], 2) for cav in self.cavities_list],
                    'alpha_er [deg]': [round(cav.shape_space['OC'][7], 2) for cav in self.cavities_list],
                    'R_shunt [Ohm]': ['' for cav in self.cavities_list],
                    'R/Q [Ohm]': [cav.R_Q for cav in self.cavities_list],
                    'k_cc [%]': [cav.k_cc for cav in self.cavities_list],
                    'field flatness [%]': [cav.ff for cav in self.cavities_list],
                    'L_active [m]': [cav.l_active for cav in self.cavities_list],
                    'Epk/Eacc []': [cav.e for cav in self.cavities_list],
                    'Bpk/Eacc [mT/MV/m]': [cav.b for cav in self.cavities_list],
                    'G [Ohm]': [cav.G for cav in self.cavities_list],
                    'R/Q.G [Ohm^2]': [cav.GR_Q for cav in self.cavities_list],
                    '|k_loss| [V/pC]': [cav.k_loss for cav in self.cavities_list],
                    '|k_kick| [V/pC/m]': [cav.k_kick for cav in self.cavities_list],
                    'P_HOM/cav [kW]': [cav.phom for cav in self.cavities_list],
                    'Reference': [cav.reference for cav in self.cavities_list]
                    }

            df = pd.DataFrame.from_dict(data)
            df.to_excel(
                os.path.join(self.projectDir, "SimulationData", "Data", f"{self.name}_excel_summary.xlsx"),
                sheet_name='Cavities')
        except Exception as e:
            error("Either SLANS or ABCI results not available. Please use '<cav>.set_slans_qois(<folder>)' "
                  "or '<cav>.set_wakefield_qois(<folder>)' to fix this.")

    def remove_cavity(self, cav):
        """
        Removes cavity from cavity list
        Parameters
        ----------
        cav: object
            Cavity object

        Returns
        -------

        """
        self.cavities_list.remove(cav)

    def save_all_plots(self, plot_name):
        """
        Save all plots
        Parameters
        ----------
        plot_name: str
            Name of saved plot

        Returns
        -------

        """
        fname = [cav.name for cav in self.cavities_list]
        if self.projectDir != '':
            # check if folder exists
            if os.path.exists(fr"{self.projectDir}\PostProcessingData\Plots"):
                # create new subdirectory
                if not os.path.exists(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}"):
                    os.mkdir(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}")

                save_folder = fr"{self.projectDir}\PostProcessingData\Plots\{'_'.join(fname)}"
                plt.savefig(f"{save_folder}/{plot_name}", dpi=300)
            else:
                if not os.path.exists(fr"{self.projectDir}\PostProcessingData"):
                    os.mkdir(fr"{self.projectDir}\PostProcessingData")
                    os.mkdir(fr"{self.projectDir}\PostProcessingData\Plots")
                    os.mkdir(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}")

                save_folder = fr"{self.projectDir}\PostProcessingData\Plots\{'_'.join(fname)}"
                plt.savefig(f"{save_folder}/{plot_name}", dpi=300)

    def calc_limits(self, which, selection):
        if self.operating_points is not None:
            cols = selection
            E0 = [self.operating_points[col]["E [GeV]"] for col in cols]
            nu_s = [self.operating_points[col]["nu_s []"] for col in cols]
            I0 = [self.operating_points[col]["I0 [mA]"] for col in cols]
            alpha_c = [self.operating_points[col]["alpha_p [1e-5]"] for col in cols]
            tau_z = [self.operating_points[col]["tau_z [ms]"] for col in cols]
            tau_xy = [self.operating_points[col]["tau_xy [ms]"] for col in cols]
            f_rev = [self.operating_points[col]["f_rev [kHz]"] for col in cols]
            beta_xy = [self.operating_points[col]["beta_xy [m]"] for col in cols]
            n_cav = [self.operating_points[col]["N_c []"] for col in cols]

            unit = {'MHz': 1e6,
                    'GHz': 1e9}

            Z_list, ZT_le = [], []
            if which == 'longitudinal':
                f_list = np.linspace(0, 10000, num=1000) * 1e6
                Z_le = []
                try:
                    for i, n in enumerate(n_cav):
                        Z = [(2 * E0[i] * 1e9 * nu_s[i])
                             / (n * I0[i] * 1e-3 * alpha_c[i] * 1e-5 * tau_z[i] * 1e-3 * f)
                             if f > 1e-8 else 1e5 for f in f_list]

                        Z_le.append(np.round((2 * E0[i] * 1e9 * nu_s[i])
                                          / (n * I0[i] * 1e-3 * alpha_c[i] * 1e-5
                                             * tau_z[i] * 1e-3) * 1e-9 * 1e-3, 2))

                        Z_list.append(np.array(Z) * 1e-3)  # convert to kOhm

                except ZeroDivisionError:
                    error("ZeroDivisionError, check input")
                return f_list * 1e-6, Z_list, cols

            elif which == 'transversal':

                f_list = np.linspace(0, 10000, num=1000) * 1e6
                try:
                    # print(E0, beta_xy, tau_xy, n_cav, f_rev, I0)
                    for i, n in enumerate(n_cav):
                        # print(E0[i], beta_xy[i], tau_xy[i], n_cav[i], f_rev[i], I0[i])
                        ZT = (2 * E0[i]) * 1e9 / (n * I0[i] * 1e-3 * beta_xy[i] * tau_xy[i] * 1e-3 * f_rev[i] * 1e3)
                        ZT_le.append(np.round(ZT * 1e-3, 2))
                        Z_list.append(np.array(ZT) * 1e-3)  # convert to kOhm/m

                except Exception as e:
                    error("ZeroDivisionError, check input", e)

                return f_list * 1e-6, Z_list, cols
        else:
            error("Please load a valid operating point(s) file.")

    def plot_thresholds(self, which, selection, ax=None, ncav_mod=1):
        labels_info = []

        def update_text_positions(event):
            """Callback function to update the text positions based on current limits."""
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for info_ in labels_info:
                # Check if the label is out of bounds
                if not (xlim[0] <= info_['x'] <= xlim[1]) or not (ylim[0] <= info_['y'] <= ylim[1]):
                    # Reposition the label to the center of the current view
                    axes_coord = (0.01, 0.5)
                    # Transform axes coordinate to display coordinate
                    display_coord = ax.transAxes.transform(axes_coord)
                    # Transform display coordinate to data coordinate
                    data_coord = ax.transData.inverted().transform(display_coord)
                    new_x = data_coord[0]
                    new_y = info_['line'].get_ydata()[np.abs(info_['line'].get_xdata() - new_x).argmin()]
                    info_['text_obj'].set_position((new_x, new_y))

        self.threshold_texts_objects = []
        if ax is None:
            fig, ax = plt.subplots()
            ax.margins(x=0)

        if which.lower() in 'longitudinal':
            # calculate limits
            f_list, Z_list, labels = self.calc_limits('longitudinal', selection)

            # plot baselines
            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                aa, = ax.plot(f_list, ncav_mod * z, ls='--', c='k')
                labels_info.append({'line': aa})

            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                pos = axis_data_coords_sys_transform(ax, 0.01, 0.5)
                indx = np.argmin(abs(f_list - pos[0]))
                # x, y = axis_data_coords_sys_transform(ax, f_list[indx], z[indx], True)
                txt = LABELS[lab]

                ab = ax.text(f_list[indx], ncav_mod * z[indx], txt, color='k', fontsize=12, ha='left',
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                labels_info[i] = labels_info[i] | {'x': f_list[indx], 'y': ncav_mod * z[indx], 'label': txt,
                                                   'text_obj': ab}
                # ab = add_text(ax, txt, box="Square", xy=(x, y), xycoords='axes fraction', size=12)
                self.threshold_texts_objects.append(ab)

        else:
            # calculate limits
            f_list, Z_list, labels = self.calc_limits('transversal', selection)

            # plot baselines
            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                aa = ax.axhline(ncav_mod * z, ls='--', c='k')
                labels_info.append({'line': aa})

            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                pos = axis_data_coords_sys_transform(ax, 0.01, 0.5)
                indx = np.argmin(abs(f_list - pos[0]))
                # x, y = axis_data_coords_sys_transform(ax, f_list[indx], z, True)

                txt = LABELS[lab]

                ab = ax.text(f_list[indx], ncav_mod * z, txt, color='k', fontsize=12, ha='left',
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                labels_info[i] = labels_info[i] | {'x': f_list[indx], 'y': ncav_mod * z, 'label': txt, 'text_obj': ab}
                self.threshold_texts_objects.append(ab)

        # Attach the callback to limit changes
        ax.callbacks.connect('xlim_changed', update_text_positions)
        ax.callbacks.connect('ylim_changed', update_text_positions)

    def _adjust_texts(self, texts, ax, separation=0.1, iterations=20):
        renderer = ax.figure.canvas.get_renderer()

        def get_text_bbox(text):
            bbox = text.get_window_extent(renderer=renderer)
            bbox_axes_coords = bbox.transformed(ax.transAxes.inverted())
            return bbox_axes_coords

        def calculate_adjustment(bbox1, bbox2):
            center1 = np.array([bbox1.x0 + bbox1.width / 2, bbox1.y0 + bbox1.height / 2])
            center2 = np.array([bbox2.x0 + bbox2.width / 2, bbox2.y0 + bbox2.height / 2])

            pos1 = np.array(text1.get_position())
            pos2 = np.array(text2.get_position())

            if bbox1.overlaps(bbox2):
                overlap_x = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
                overlap_y = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))

                if overlap_x > 0:
                    midpoint_x = (center1[0] + center2[0]) / 2
                    if bbox1.x1 > bbox2.x1:
                        pos1[0] = midpoint_x + bbox1.width / 2 + separation
                        pos2[0] = midpoint_x - bbox2.width / 2 - separation
                    else:
                        pos1[0] = midpoint_x - bbox1.width / 2 - separation
                        pos2[0] = midpoint_x + bbox2.width / 2 + separation

                # if overlap_y > 0:
                #     midpoint_y = (center1[1] + center2[1]) / 2
                #     if bbox1.y1 > bbox2.y1:
                #         pos1[1] = midpoint_y + bbox1.height / 2 + separation
                #         pos2[1] = midpoint_y - bbox2.height / 2 - separation
                #     else:
                #         pos1[1] = midpoint_y - bbox1.height / 2 - separation
                #         pos2[1] = midpoint_y + bbox2.height / 2 + separation

            return pos1, pos2

        # Run the adjustment algorithm
        for _ in range(iterations):
            for i, text1 in enumerate(texts):
                bbox1 = get_text_bbox(text1)
                for j, text2 in enumerate(texts):
                    if i == j:
                        continue
                    bbox2 = get_text_bbox(text2)

                    if bbox1.overlaps(bbox2):
                        pos1, pos2 = calculate_adjustment(bbox1, bbox2)
                        text1.set_position((pos1[0], pos1[1]))
                        text2.set_position((pos2[0], pos2[1]))

        # Ensure the texts are within the plot limits
        for text in texts:
            x, y = text.get_position()
            x = np.clip(x, ax.get_xlim()[0], ax.get_xlim()[1])
            y = np.clip(y, ax.get_ylim()[0], ax.get_ylim()[1])
            text.set_position((x, y))

    def plot_cutoff(self, Ri_list, which, ax=None):
        labels_info = []

        def update_text_positions(event):
            """Callback function to update the text positions based on current limits."""
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for info_ in labels_info:
                # Check if the label is out of bounds
                if not (xlim[0] <= info_['x'] <= xlim[1]) or not (ylim[0] <= info_['y'] <= ylim[1]):
                    # Reposition the label to the center of the current view
                    axes_coord = (0.01, 0.05)
                    # Transform axes coordinate to display coordinate
                    display_coord = ax.transAxes.transform(axes_coord)
                    # Transform display coordinate to data coordinate
                    data_coord = ax.transData.inverted().transform(display_coord)
                    new_y = data_coord[1]
                    new_x = info_['line'].get_xdata()[np.abs(info_['line'].get_ydata() - new_y).argmin()]
                    info_['text_obj'].set_position((new_x, new_y))

        if ax is None:
            fig, ax = plt.subplots()
            ax.margins(x=0)

        f_list = self.calculate_beampipe_cutoff(Ri_list, which)

        i = 0
        for Ri in Ri_list:
            for mode_type, freq in zip(which, f_list[f'{Ri}']):
                vl = ax.axvline(freq, ls='--', c='k', zorder=900)  # label=f"{sc[0]} cutoff (Ri={sc[1]})",
                labels_info.append({'line': vl})

                # get y text position from axis position. Position x is not used
                pos = axis_data_coords_sys_transform(ax, freq, 0.05, inverse=False)

                txt = r"$f_\mathrm{c," + f"{mode_type}" + r"} (R_\mathrm{i} = " + f"{Ri}" + r" ~\mathrm{mm}) $"
                ab = ax.text(freq, pos[1], txt, color='k', fontsize=16, ha='left', rotation=90, zorder=1000)
                labels_info[i] = labels_info[i] | {'x': freq, 'y': pos[1], 'label': txt, 'text_obj': ab}
                i += 1

                # ab = add_text(ax, r"$f_\mathrm{c," + f"{mode_type}" + r"} (R_\mathrm{i} = "
                #               + f"{Ri}" + r" ~\mathrm{mm}) $",
                #               box="None", xy=(freq, pos[1]),
                #               xycoords='data', size=14, rotation=90)

        # Attach the callback to limit changes
        ax.callbacks.connect('xlim_changed', update_text_positions)
        ax.callbacks.connect('ylim_changed', update_text_positions)

    @staticmethod
    def calculate_beampipe_cutoff(Ri_list, which):
        c = 299792458
        mode_list = {}
        f_list = {}

        mode_dict = {}
        for mode in which:
            mode_type = mode[:2]
            m = int(mode[2:3])
            n = int(mode[3:])
            mode_dict[mode] = {'type': mode_type, 'm': m, 'n': n}

        for Ri in Ri_list:
            f_list[f'{Ri}'] = []
            for mode, indices in mode_dict.items():
                m, n = indices['m'], indices['n']
                # get jacobian
                if mode_dict[mode]['type'].lower() == 'tm':
                    j = jn_zeros(m, n)[n - 1]
                else:
                    j = jnp_zeros(m, n)[n - 1]

                f = c / (2 * np.pi) * (j / (Ri * 1e-3))

                # append to f list
                f_list[f'{Ri}'].append(f * 1e-6)

        return f_list

    def run_optimisation(self, optimisation_config):
        self.start_optimisation(self.projectDir, optimisation_config)

    @staticmethod
    def calc_cutoff(Ri, mode):
        # calculate frequency from Ri
        p_TM01, p_TE11 = 2.405, 1.841
        c = 299792458  # m/s

        if mode == 'TM01':
            freq = (c * p_TM01) / (2 * np.pi * Ri * 1e9) * 1e3
        else:
            freq = (c * p_TE11) / (2 * np.pi * Ri * 1e9) * 1e3

        return freq

    def __str__(self):
        p = dict()
        for name, cav in self.cavities_dict.items():
            p[name] = {
                'tune': cav.tune_results,
                'fm': cav.eigenmode_qois,
                'hom': cav.wakefield_qois,
                'uq': {
                    'tune': 0,
                    'fm': cav.uq_fm_results,
                    'hom': cav.uq_hom_results
                }
            }
        return fr"{json.dumps(p, indent=4)}"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.cavities_list[key]
        elif isinstance(key, str):
            if key in self.cavities_dict.keys():
                return self.cavities_dict[key]
            else:
                error('Invalid key. Cavities does not contain a Cavity named {key}.')
        else:
            raise TypeError("Invalid argument type. Must be int or str.")


class Pillbox(Cavity):
    def __init__(self, n_cells, L, Req, Ri, S, L_bp, beampipe='none'):
        self.n_cells = n_cells
        self.n_modes = n_cells + 1
        self.n_modules = 1  # change later to enable module run
        self.L = L
        self.Req = Req
        self.Ri = Ri
        self.S = S
        self.L_bp = L_bp
        self.beampipe = beampipe
        self.bc = 33

        self.shape_space = {
            'IC': [L, Req, Ri, S, L_bp],
            'BP': beampipe
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
        cell_type: {'mid cell', 'end-mid cell', 'mid-end cell', 'end-end cell', 'single cell'}
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
            try:
                self.get_ngsolve_tune_res(tune_variable, cell_type)
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

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
        except FileNotFoundError:
            error("Could not find eigenmode results. Please rerun eigenmode analysis.")

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

                                dirc = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                                # try:
                                k_loss = abs(ABCIData(dirc, f'{fid}', 0).loss_factor['Longitudinal'])
                                k_kick = abs(ABCIData(dirc, f'{fid}', 1).loss_factor['Transverse'])
                                # except:
                                #     k_loss = 0
                                #     k_kick = 0

                                d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, n_cells)

                    # save qoi dictionary
                    run_save_directory = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The working point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()


class RFGun(Cavity):
    def __init__(self, shape, name='gun'):
        # self.shape_space = {
        #     'IC': [L, Req, Ri, S, L_bp],
        #     'BP': beampipe
        # }
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name
        self.n_cells = 1
        self.n_modules = 1
        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = 'vhf gun'

        self.shape = {
                "geometry": shape['geometry'],
                "n_cells": self.n_cells,
                'CELL PARAMETERISATION': self.cell_parameterisation,
                'kind': self.kind}
        self.shape_multicell = {'kind': self.kind}

    def plot(self, what, ax=None, **kwargs):
        # file_path = os.path.join(self.projectDir, "SimulationData", "NGSolveMEVP", self.name, "monopole", "geodata.n")
        if what.lower() == 'geometry':
            ax = write_gun_geometry(self.shape['geometry'])
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

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        print('it is here')
        qois = 'qois.json'
        assert os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole', qois)), (
            error('Eigenmode result does not exist, please run eigenmode simulation.'))
        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                               'Ez_0_abs.csv')) as csv_file:
            self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        self.freq = self.eigenmode_qois['freq [MHz]']
        self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        self.GR_Q = self.eigenmode_qois['GR/Q [Ohm^2]']
        self.G = self.GR_Q / self.R_Q
        self.Q = self.eigenmode_qois['Q []']
        self.e = self.eigenmode_qois['Epk/Eacc []']
        self.b = self.eigenmode_qois['Bpk/Eacc [mT/MV/m]']
        self.Epk_Eacc = self.e
        self.Bpk_Eacc = self.b

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')

            ax.legend(loc="upper right")
            if show_min_max:
                minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 50, width=100)
                Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                ax.plot(self.Ez_0_abs['z(0, 0)'][peaks], Ez_0_abs_peaks, marker='o', ls='')
                ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                ax.axhline(max(Ez_0_abs_peaks), c='k')
        else:
            if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                           'Ez_0_abs.csv')):
                with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                       'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
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

    def __str__(self):
        p = dict()
        # p[self.name] = {
        #     'tune': self.tune_results,
        #     'fm': self.eigenmode_qois,
        #     'hom': self.wakefield_qois,
        #     'uq': {
        #         'tune': 0,
        #         'fm': self.uq_fm_results,
        #         'hom': self.uq_hom_results
        #     }
        # }
        return fr"{json.dumps(p, indent=4)}"


class Dakota:
    def __init__(self, folder, name, scripts_folder=None):
        self.nodes = None
        self.sim_results = None

        assert f'/' in folder, error('Please ensure directory paths use forward slashes.')
        self.projectDir = folder
        if scripts_folder is None:
            self.scripts_folder = r'D:/Dropbox/CavityDesignHub/analysis_modules/uq/dakota_scripts'
        else:
            self.scripts_folder = scripts_folder
        self.name = name

    def write_input_file(self, **kwargs):
        keys = kwargs.keys()
        assert 'variables_config' in keys, error('please enter keyword "variables config"')
        assert 'interface_config' in keys, error('please enter keyword "interface config"')
        assert 'method_config' in keys, error('please enter keyword "method config"')

        variables_config = kwargs['variables_config']
        interface_config = kwargs['interface_config']
        method_config = kwargs['method_config']

        # check if folder exists, if not, create folder
        if not os.path.exists(os.path.join(self.projectDir, self.name)):
            try:
                os.mkdir(os.path.join(self.projectDir, self.name))
            except FileExistsError:
                error("Could not create folder. Make sure target location exists.")

        with open(os.path.join(self.projectDir, self.name, f'{self.name}.in'), 'w') as f:
            self.environment(f)
            self.method(f, **method_config)
            self.variables(f, **variables_config)
            self.interface(f, **interface_config)

    def environment(self, f):
        f.write('environment\n')
        f.write('\ttabular_data\n')
        f.write("\t\ttabular_data_file = 'sim_result_table.dat'\n")
        f.write('\tresults_output\n')
        f.write("\t\tresults_output_file = 'result_output_file.dat'\n")

        f.write('\n')

    def method(self, f, **kwargs):
        keys = kwargs.keys()
        assert 'method' in keys, error('Please enter "method" in "method config".')
        method = kwargs['method']

        f.write('method\n')
        f.write(f'\t{method}\n')
        f.write("\t\texport_expansion_file ='expansion_file.dat'\n")
        f.write("\t\tcubature_integrand = 3\n")
        f.write("\t\tsamples_on_emulator = 10000\n")
        f.write("\t\tseed = 12347\n")
        f.write("\t\tvariance_based_decomp interaction_order = 1\n")

        f.write('\n')

    def variables(self, f, **kwargs):
        """

        Parameters
        ----------
        f: File
            File
        kind: {'uniform_uncertain', 'normal_uncertain', 'beta_uncertain'}
            Type of distribution of the variables
        descriptors: list, ndarray
            Uncertain variable names
        kwargs: kwargs
            Other relevant arguments eg. {means: [], 'std_deviations': [], 'lower_bounds': [], 'upper_bounds': []}


        Returns
        -------

        """

        keys = kwargs.keys()
        assert 'kind' in keys, error('Please enter "kind"')
        assert 'lower_bounds' in keys, error('Please enter keyword "lower_bounds"')
        assert 'upper_bounds' in keys, error('Please enter keyword "upper bounds"')
        kind = kwargs['kind']
        upper_bounds = kwargs['upper_bounds']

        assert len(upper_bounds) == len(lower_bounds), error("Length of upper and lower bounds must be equal.")

        if "descriptors" in keys:
            descriptors = kwargs['descriptors']
        else:
            info('"descriptors" not entered. Using default parameter labelling.')
            descriptors = [f'p{n}' for n in range(len(upper_bounds))]
        assert len(descriptors) == len(kwargs['upper_bounds'])

        f.write("variables\n")
        f.write(f"\t{kind} = {len(descriptors)}\n")
        f.write(
            "\tdescriptors       =   " + '\t\t\t'.join(['"' + descriptor + '"' for descriptor in descriptors]) + '\n')

        if 'means' in kwargs.keys():
            assert len(descriptors) == len(kwargs['means'])
            f.write("\tmeans      =   " + '\t\t\t'.join([str(mean) for mean in kwargs['means']]) + '\n')

        if 'std_deviations' in kwargs.keys():
            assert len(descriptors) == len(kwargs['std_deviations'])
            f.write("\tstd_deviations      =   " + '\t\t\t'.join([str(std) for std in kwargs['std_deviations']]) + '\n')

        if 'lower_bounds' in kwargs.keys():
            f.write("\tlower_bounds      =   " + '\t\t\t'.join([str(lb) for lb in kwargs['lower_bounds']]) + '\n')

        if 'upper_bounds' in kwargs.keys():
            f.write("\tupper_bounds      =   " + '\t\t\t'.join([str(ub) for ub in kwargs['upper_bounds']]) + '\n')

        f.write('\n')

    def interface(self, f, **kwargs):

        keys = kwargs.keys()
        assert 'analysis_driver' in keys, error('please enter keyword "analysis driver"')
        assert 'responses' in keys, error('Please enter "responses"')

        analysis_driver = kwargs['analysis_driver']
        responses = kwargs['responses']

        nodes_only = False
        if 'nodes_only' in keys:
            nodes_only = kwargs['nodes_only']
            responses = ['f1']

        processes = 1
        if 'processes' in keys:
            processes = kwargs['processes']

        f.write("interface\n")
        f.write("#\tcommon options\n")
        f.write("#\tfork\n")
        f.write("\tparameters_file = 'params.in'\n")
        f.write("\tresults_file    = 'results.out'\n")
        f.write(f"\tsystem asynchronous evaluation_concurrency = {processes}\n")
        f.write(f"\tanalysis_driver = '{analysis_driver} {len(responses)} {nodes_only} {self.scripts_folder}'\n")
        f.write("#\tparameters_file = 'params.in'\n")
        f.write("#\tresults_file    = 'results.out'\n")
        f.write("#\tfile_tag\n")
        f.write("#\tfile_save\n")
        f.write("#\taprepro\n")
        f.write('\n')

        self.responses(f, responses)

    def responses(self, f, responses):
        f.write("responses\n")
        f.write(f"\tresponse_functions = {len(responses)}\n")
        f.write("\tno_gradients\n")
        f.write("\tno_hessians\n")

        f.write('\n')

    def nodes_to_cst_sweep_input(self, partitions=1):
        # save parts
        row_partition = len(self.nodes.index) // partitions
        for i in range(partitions):
            if i < partitions - 1:
                df_part = self.nodes.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = self.nodes.loc[i * row_partition:]

            df_part.to_csv(fr"{self.projectDir}/{self.name}/cst_sweep_files/cst_par_in_{i + 1}.txt", sep="\t",
                           index=None)

    def run_analysis(self, write_cst=True, partitions=1):
        cwd = os.path.join(self.projectDir, self.name)
        dakota_in = f'{os.path.join(self.projectDir, self.name, f"{self.name}.in")}'
        dakota_out = f'{os.path.join(self.projectDir, self.name, f"{self.name}.out")}'

        subprocess.run(['dakota', '-i', dakota_in, '-o', dakota_out], cwd=cwd, shell=True)

        # read results
        filepath = fr"{self.projectDir}/{self.name}/sim_result_table.dat"
        self.sim_results = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        self.nodes = self.sim_results.drop(self.sim_results.filter(regex='response|interface|eval_id').columns, axis=1)
        self.sim_results.to_excel(fr"{self.projectDir}/{self.name}/nodes.xlsx", index=False)

        if write_cst:
            # check if folder exist and clear
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
            else:
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

            # post processes
            self.nodes_to_cst_sweep_input(partitions)
        else:
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

    def plot_sobol_indices(self, filepath, objectives, which=None, kind='stacked', orientation='vertical',
                           normalise=True,
                           group=None, reorder_index=None,
                           selection_index=None, figsize=None):

        if figsize is None:
            figsize = (8, 4)

        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall("\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall("\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []

        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=['main', 'total', 'vars'])
            df = df.astype({'main': 'float', 'total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        df_merge_interaction = pd.DataFrame(columns=['interaction', 'var1', 'var2'])
        # ic(result_interaction.items())
        for i, (k, v) in enumerate(result_interaction.items()):
            df = pd.DataFrame(v, columns=['interaction', 'var1', 'var2'])
            df = df.astype({'interaction': 'float'})
            if i == 0:
                df_merge_interaction = df
            else:
                # df_merge_interaction = pd.merge(df_merge_interaction, df, on=['var1', 'var2'])
                pass
            # ic(df_merge_interaction)

            # combine var columns
            # df_merge_interaction["vars"] = df_merge_interaction[["var1", "var2"]].agg(','.join, axis=1)
            # df.plot.bar(x='var', y='Main')

        # ic(df_merge)

        # group columns
        if group:
            df_merge_T = df_merge.T
            df_merge_T.columns = df_merge_T.loc['vars']
            df_merge_T = df_merge_T.drop('vars')
            for g in group:
                df_merge_T[','.join(g)] = df_merge_T[g].sum(axis=1)

                # drop columns
                df_merge_T = df_merge_T.drop(g, axis=1)

            # reorder index
            if reorder_index:
                df_merge_T = df_merge_T[reorder_index]
            df_merge = df_merge_T.T.reset_index()
            # ic(df_merge)

        # ic(df_merge_interaction)

        if normalise:
            # normalise dataframe columns
            for column in df_merge.columns:
                if 'main' in column or 'total' in column:
                    df_merge[column] = df_merge[column].abs() / df_merge[column].abs().sum()

        # ic(df_merge)
        # filter df
        for w in which:
            if w.lower() == 'main' or w.lower() == 'total':
                dff = df_merge.filter(regex=f'{w.lower()}|vars')
            else:
                # create new column which is a combination of the two variable names
                dff = df_merge_interaction.filter(regex=f'{w.lower()}|vars')
                if not dff.empty:
                    dff['vars'] = df_merge_interaction[['var1', 'var2']].apply(lambda x: '_'.join(x), axis=1)

            cmap = 'tab20'

            if not dff.empty:
                if kind.lower() == 'stacked':
                    dff_T = dff.set_index('vars').T
                    if orientation == 'vertical':
                        ax = dff_T.plot.bar(stacked=True, rot=0, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff_T.plot.barh(stacked=True, rot=0, edgecolor='k', figsize=(8, 4))  # , cmap=cmap
                        ax.invert_yaxis()
                        # for bars in ax.containers:
                        #     ax.bar_label(bars, fmt='%.2f', label_type='center', color='white', fontsize=5)

                        ax.set_xlim(left=0)
                        ax.set_yticklabels(objectives)
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')
                else:
                    if orientation == 'vertical':
                        ax = dff.plot.bar(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axhline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff.plot.barh(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axvline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')

            else:
                error(f"No {w} found.")

    def get_sobol_indices(self, filepath, objectives, which=None, selection_index=None):
        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall("\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall("\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []
        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=[f'{objectives[i].replace("$", "")}_main',
                                          f'{objectives[i].replace("$", "")}_total', 'vars'])
            df = df.astype(
                {f'{objectives[i].replace("$", "")}_main': 'float', f'{objectives[i].replace("$", "")}_total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        return df_merge

    def quadrature_nodes_to_cst_par_input(self, filefolder, n=2):
        filepath = fr"{filefolder}\sim_result_table.dat"
        df = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        df.drop(df.filter(regex='response|interface|eval_id').columns, axis=1, inplace=True)
        df.to_excel(fr"{filefolder}\cubature_nodes_pars.xlsx", index=False)

        # save parts
        row_partition = len(df.index) // n
        for i in range(n):
            if i < n - 1:
                df_part = df.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = df.loc[i * row_partition:]

            df_part.to_csv(fr"{filefolder}\cst_par_in_{i + 1}.txt", sep="\t", index=None)

    def get_pce(self, filefolder):

        filepath = fr"{filefolder}\uq_pce_expansion.dat"
        df = pd.read_csv(filepath, sep='\\s+', header=None)

        poly = 0
        for row in df.iterrows():
            poly += 0

    def quote_to_float(self, value):
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            return float(value[1:-1])
        else:
            return value

    def combine_params_output(self, folder, N):
        for i in range(N):
            if i == 0:
                df = pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)
            else:
                df = pd.concat([df, pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)])

        # rearrange column according to the reference column order
        df_reference = pd.read_excel(fr"{folder}\cubature_nodes_pars.xlsx")
        columns = list(df_reference.columns)
        # check if 3D Run ID in column and drop if yes
        if ' 3D Run ID' in list(df.columns):
            columns.append(' 3D Run ID')

        columns = list(df_reference.columns) + (df.columns.drop(columns).tolist())

        df = df[columns]

        df.to_excel(fr"{folder}\cubature_nodes.xlsx", index=False)

    def plot_sobol(self, config):
        obj = config['obj']
        group = config['group']
        reorder_index = ['reorder_index']
        filefolder = config['folder']
        kind = config['kind']
        normalise = config['normalise']
        which = config['which']
        orientation = config['orientation']
        selection_index = config['selection_index']

        # obj = [r"$Q_\mathrm{ext, FM}$", r"$\max(Q_\mathrm{ext, dip})$"]
        # obj = [r"$S_\mathrm{max}~\mathrm{[dB]}$", r"$S_\mathrm{min}~\mathrm{[dB]}$", r"$f(S_\mathrm{max})~\mathrm{[MHz]}$", r"$f(S_\mathrm{min})~\mathrm{[MHz]}$"]
        # obj =
        # obj = [r"$freq [MHz]$",	fr"$R/Q [Ohm]$", r"$Epk/Eacc []$",	r"$Bpk/Eacc [mT/MV/m]$", 'G', 'kcc', 'ff']
        # obj =
        # obj = [r"$freq [MHz]$", 'kcc']
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', group=group, reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', selection_index=selection_index, normalise=False)#

        self.plot_sobol_indices(fr"{filefolder}\dakota.out", obj, which=which, kind=kind,
                                selection_index=selection_index, orientation=orientation, normalise=normalise)  #


class OperationPoints:
    def __init__(self, filepath=None):
        self.op_points = {}

        if filepath:
            if os.path.exists(filepath):
                self.op_points = self.load_operating_point(filepath)

    def load_operating_point(self, filepath):
        with open(filepath, 'r') as f:
            op_points = json.load(f)

        self.op_points = op_points
        return op_points

    def get_default_operating_points(self):
        self.op_points = pd.DataFrame({
            "Z_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1280,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 56,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 132,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2023": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1400,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 112,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_booster_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 45.6,
                "I0 [mA]": 128,
                "V [GV]": 0.14,
                "Eacc [MV/m]": 6.23,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 120,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 0.276
            },
            "Z_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1390,
                "V [GV]": 0.10,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.025,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 424.6,
                "tau_xy [ms]": 849.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.5,
                "sigma_BS [mm]": 12.1,
                "Nb [1e11]": 1.7
            },
            "W_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 147,
                "V [GV]": 0.75,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0506,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 78.7,
                "tau_xy [ms]": 157.4,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.0,
                "sigma_BS [mm]": 6.0,
                "Nb [1e11]": 1.5
            },
            "H_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 29,
                "V [GV]": 2.0,
                "Eacc [MV/m]": 11.87,
                "nu_s []": 0.036,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 23.4,
                "tau_xy [ms]": 46.8,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 136,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.15,
                "sigma_BS [mm]": 5.3,
                "Nb [1e11]": 1.8
            },
            "ttbar_2018": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10.8,
                "V [GV]": 10.93,
                "Eacc [MV/m]": 24.72,
                "nu_s []": 0.087,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 6.8,
                "tau_xy [ms]": 13.6,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 584,
                "T [K]": 2,
                "sigma_SR [mm]": 1.97,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.3
            }
        })


class QuickTools:
    def __init__(self):
        pass

    @staticmethod
    def cwg_cutoff(r, l=0, mode=None):
        """

        Parameters
        ----------
        r: float, list
            radius or list of radii in nmm

        Returns
        -------
        f_cutoff: float, list
            cutoff frequency or list of cutoff frequencies in MHz
        """

        if isinstance(r, float) or isinstance(r, int):
            r = [r]
        if isinstance(l, float) or isinstance(l, int):
            l = [l]

        f_cutoff = {}
        mode_dict = {'te11': 1.841, 'tm01': 2.405, 'te21': 3.054, 'te01': 3.832, 'tm11': 3.832, 'tm21': 5.135,
                     'te12': 5.331, 'tm02': 5.520, 'te22': 6.706, 'te02': 7.016, 'tm12': 7.016, 'tm22': 8.417,
                     'te13': 8.536, 'tm03': 8.654, 'te23': 9.970, 'te03': 10.174, 'tm13': 10.174, 'tm23': 11.620}

        if mode is None:
            for radius in r:
                f_cutoff[f'{radius}'] = {}
                for mode, j in mode_dict.items():
                    f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                    f_cutoff[f'{radius}'][mode] = f * 1e-6
        else:
            if isinstance(mode, list):
                for mode_ in mode:
                    for radius in r:
                        f_cutoff[f'{radius}'] = {}
                        try:
                            j = mode_dict[mode_]
                        except KeyError:
                            error("One or more mode names is wrong. Please check mode names.")
                            j = 0
                        f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                        f_cutoff[f'{radius}'][mode_] = f * 1e-6
            else:
                if isinstance(mode, str):
                    for radius in r:
                        f_cutoff[f'{radius}'] = {}
                        try:
                            j = mode_dict[mode]
                        except KeyError:
                            error("One or more mode names is wrong. Please check mode names.")
                            j = 0
                        f = c0 / (2 * np.pi) * (j / (radius * 1e-3))
                        f_cutoff[f'{radius}'][mode] = f * 1e-6
                else:
                    error("One or more mode names is wrong. Please check mode names.")

        return f_cutoff

    @staticmethod
    def rwg_cutoff(a, b, mn=None, l=0, p=None):
        if isinstance(a, float) or isinstance(a, int):
            a = [a]
        if isinstance(b, float) or isinstance(b, int):
            b = [b]
        if isinstance(l, float) or isinstance(l, int):
            l = [l]

        if mn is None:
            mn = [[0, 1]]

        if len(np.array(mn).shape) == 1:
            mn = [mn]

        if isinstance(p, int):
            p = [p]

        if p is None:
            p = 0

        f_cutoff = {}
        try:
            for a_ in a:
                f_cutoff[f'a: {a_} mm'] = {}
                for b_ in b:
                    f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'] = {}
                    for l_ in l:
                        f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'] = {}
                        for mn_ in mn:
                            m, n = mn_
                            if l_ == 0:
                                f = (c0 / (2 * np.pi)) * (
                                        (m * np.pi / (a_ * 1e-3)) ** 2 + (n * np.pi / (b_ * 1e-3)) ** 2) ** 0.5
                                f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'TE/TM({m},{n})'] = f * 1e-6
                            else:
                                f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'] = {}
                                for p_ in p:
                                    f = (c0 / (2 * np.pi)) * (
                                            (m * np.pi / (a_ * 1e-3)) ** 2 + (n * np.pi / (b_ * 1e-3)) ** 2 + (
                                            p_ * np.pi / (l_ * 1e-3)) ** 2) ** 0.5
                                    f_cutoff[f'a: {a_} mm'][f'b: {b_} mm'][f'l: {l_} mm'][
                                        f'TE/TM({m},{n},{p_})'] = f * 1e-6
            return f_cutoff
        except ValueError:
            print("Please enter a valid number.")

    @staticmethod
    def coaxial_tline(D, d, x=0, epsr=1):
        D, d, x = D * 1e-3, d * 1e-3, x * 1e-3

        Z = 60 * np.arccosh((d ** 2 + D ** 2 - x ** 2) / (2 * d * D))
        C = 2 * np.pi * eps0 * epsr / (np.arccosh((d ** 2 + D ** 2 - x ** 2) / (2 * d * D))) * 1e12
        L = mu0 / (2 * np.pi) * np.arccosh((d ** 2 + D ** 2 - x ** 2) / (2 * d * D)) * 1e9
        return {"L' [nH/m]": L, "C' [pF/m]": C, "Z' [Ohm/m]": Z}

    @staticmethod
    def parallel_plate_capacitor(l, b, d, epsr=1):
        l, b = l * 1e-3, b * 1e-3
        C = epsr * l * b / d
        return {"C' [pF]": C}

    @staticmethod
    def parallel_disc_capacitor(D, d, epsr=1):
        D, d = D * 1e-3, d * 1e-3
        C = epsr * np.pi * D ** 2 / (4 * d)
        return {"C' [pF]": C}

    @staticmethod
    def cwg_analytical(m, n, kind='te', R=None, pol=None, component='abs'):
        if R is None:
            R = 1
        if not pol:
            pol = 0
        else:
            pol = 0

        r_ = np.linspace(1e-6, R, 500)
        t_ = np.linspace(0, 2 * np.pi, 500)
        radius, theta = np.meshgrid(r_, t_)
        A = 1
        k = 0  # no propagation in z

        if kind.lower() == 'te':
            j_mn_p = jnp_zeros(m, n)[n - 1]
            kc = j_mn_p / R
            w = kc / np.sqrt(mu0 * eps0)

            beta = np.sqrt(k ** 2 - kc ** 2)

            Er = -1j * w * mu0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(
                m,
                kc * radius)
            Et = 1j * w * mu0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Ez = 0
            Hr = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Ht = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                    kc * radius)
            Hz = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)

            Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
            Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))
        elif kind.lower() == 'tm':
            j_mn = jn_zeros(m, n)[n - 1]
            kc = j_mn / R
            w = kc / np.sqrt(mu0 * eps0)
            beta = np.sqrt(k ** 2 - kc ** 2)
            Ez = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)
            Er = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Et = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                    kc * radius)
            Hr = 1j * w * eps0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(
                m, kc * radius)
            Ht = -1j * w * eps0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
            Hz = 0

            Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
            Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))
        else:
            raise RuntimeError('Please enter valid cicular waveguide mode kind.')

        if component.lower() == 'abs':
            return radius, theta, Emag, Hmag
        if component.lower() == 'azimuthal':
            return radius, theta, Et, Ht
        if component.lower() == 'radial':
            return radius, theta, Er, Hr
        if component.lower() == 'longitudinal':
            return radius, theta, Ez, Hz

    @staticmethod
    def cwg_tm_analytical(m, n, theta, radius, R=None, pol=None, component='abs'):
        if R is None:
            R = 1
        if pol is None:
            pol = 0

        r_ = np.linspace(1e-6, R, 500)
        t_ = np.linspace(0, 2 * np.pi, 500)
        radius, theta = np.meshgrid(r_, t_)

        j_mn = jn_zeros(m, n)[n - 1]
        A = 1
        k = 0  # no propagation in z
        kc = j_mn / R
        w = kc / np.sqrt(mu0 * eps0)
        beta = np.sqrt(k ** 2 - kc ** 2)
        Ez = A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jv(m, kc * radius)
        Er = -1j * beta / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
        Et = -1j * beta * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                kc * radius)
        Hr = 1j * w * eps0 * m / (kc ** 2 * radius) * A * (np.cos(m * theta + pol) - np.sin(m * theta + pol)) * jv(m,
                                                                                                                   kc * radius)
        Ht = -1j * w * eps0 / kc * A * (np.sin(m * theta + pol) + np.cos(m * theta + pol)) * jvp(m, kc * radius)
        Hz = 0

        Emag = np.abs(np.sqrt(Er ** 2 + Et ** 2 + Ez ** 2))
        Hmag = np.abs(np.sqrt(Hr ** 2 + Ht ** 2 + Hz ** 2))

        if component.lower() == 'abs':
            return radius, theta, Emag, Hmag
        if component.lower() == 'azimuthal':
            return radius, theta, Et, Ht
        if component.lower() == 'radial':
            return radius, theta, Er, Hr
        if component.lower() == 'longitudinal':
            return radius, theta, Ez, Hz


def run_tune_parallel(shape_space, tune_config, projectDir, solver='NGSolveMEVP',
                      resume=False):
    tune_config_keys = tune_config.keys()
    if 'processes' in tune_config_keys:
        processes = tune_config['processes']
        assert processes > 0, error('Number of proceses must be greater than zero.')
    else:
        processes = 1

    assert 'freqs' in tune_config_keys, error('Please enter the target tune "freqs" in tune_config.')
    assert 'parameters' in tune_config_keys, error('Please enter the tune "parameters"  in tune_config')
    assert 'cell_types' in tune_config_keys, error('Please enter the "cell_types" in tune_config')
    freqs = tune_config['freqs']
    tune_parameters = tune_config['parameters']
    cell_types = tune_config['cell_types']

    if isinstance(freqs, float) or isinstance(freqs, int):
        freqs = np.array([freqs for _ in range(len(shape_space))])
    else:
        assert len(freqs) == len(shape_space), error(
            'Number of target frequencies must correspond to the number of cavities')
        freqs = np.array(freqs)

    if isinstance(tune_parameters, str):
        assert tune_config['parameters'] in ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req'], error(
            'Please enter a valid tune parameter')
        tune_parameters = np.array([tune_parameters for _ in range(len(shape_space))])
        cell_types = np.array([cell_types for _ in range(len(shape_space))])
    else:
        assert len(tune_parameters) == len(shape_space), error(
            'Number of tune parameters must correspond to the number of cavities')
        assert len(cell_types) == len(shape_space), error(
            'Number of cell types must correspond to the number of cavities')
        tune_parameters = np.array(tune_parameters)
        cell_types = np.array(cell_types)

    # split shape_space for different processes/ MPI share process by rank
    keys = list(shape_space.keys())

    # check if number of processors selected is greater than the number of keys in the pseudo shape space
    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    # share = int(round(shape_space_len / processes))
    jobs = []
    # for p in range(processes):
    #     # try:
    #     if p < processes - 1:
    #         proc_keys_list = keys[p * share:p * share + share]
    #         proc_tune_variables = tune_parameters[p * share:p * share + share]
    #         proc_freqs = freqs[p * share:p * share + share]
    #         proc_cell_types = cell_types[p * share:p * share + share]
    #     else:
    #         proc_keys_list = keys[p * share:]
    #         proc_tune_variables = tune_parameters[p * share:]
    #         proc_freqs = freqs[p * share:]
    #         proc_cell_types = cell_types[p * share:]

    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0
    for p in range(processes):
        # Determine the size of the current chunk
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        proc_tune_variables = tune_parameters[start_idx:start_idx + current_chunk_size]
        proc_freqs = freqs[start_idx:start_idx + current_chunk_size]
        proc_cell_types = cell_types[start_idx:start_idx + current_chunk_size]

        start_idx += current_chunk_size

        processor_shape_space = {key: shape_space[key] for key in proc_keys_list}
        service = mp.Process(target=run_tune_s, args=(processor_shape_space, proc_tune_variables,
                                                      proc_freqs, proc_cell_types, tune_config, projectDir, resume, p))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_tune_s(processor_shape_space, proc_tune_variables, proc_freqs, proc_cell_types, tune_config, projectDir, resume,
               p, sim_folder='Optimisation'):
    # perform necessary checks
    if tune_config is None:
        tune_config = {}
    tune_config_keys = tune_config.keys()

    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    def _run_tune():
        tuned_shape_space, d_tune_res, conv_dict, abs_err_dict = tuner.tune_ngsolve({key: shape}, 33,
                                                                                    SOFTWARE_DIRECTORY,
                                                                                    projectDir, key,
                                                                                    resume=resume, proc=p,
                                                                                    tune_variable=
                                                                                    proc_tune_variables[i],
                                                                                    cell_type=proc_cell_types[
                                                                                        i],
                                                                                    sim_folder=sim_folder,
                                                                                    tune_config=tune_config)

        if d_tune_res:
            n_cells = processor_shape_space[key]['n_cells']
            tuned_shape_space[key]['n_cells'] = n_cells
            tuned_shape_space[key]['CELL PARAMETERISATION'] = processor_shape_space[key]['CELL PARAMETERISATION']
            tuned_shape_space_multi = {kk: to_multicell(n_cells, tuned_shape) for kk, tuned_shape in
                                       tuned_shape_space.items()}

            eigenmode_config = {}
            if 'eigenmode_config' in tune_config_keys:
                eigenmode_config = tune_config['eigenmode_config']
            else:
                info('tune_config does not contain eigenmode_config. Default values are used for eigenmode analysis.')

            eigenmode_config['solver_save_directory'] = sim_folder.split(os.sep)[0]
            if sim_folder == 'Optimisation':
                eigenmode_config['opt'] = True
            else:
                eigenmode_config['opt'] = False

            # might cause some problems later
            if len(sim_folder.split(os.sep)) > 1:
                subdir = os.path.basename(sim_folder)
            else:
                subdir = ''
            run_eigenmode_parallel(tuned_shape_space, tuned_shape_space_multi, eigenmode_config, projectDir,
                                   subdir=subdir)

            # save tune results
            save_tune_result(d_tune_res, 'tune_res.json', projectDir, key, sim_folder)

            # save convergence information
            save_tune_result(conv_dict, 'convergence.json', projectDir, key, sim_folder)
            save_tune_result(abs_err_dict, 'absolute_error.json', projectDir, key, sim_folder)

    for i, (key, shape) in enumerate(processor_shape_space.items()):
        shape['FREQ'] = proc_freqs[i]
        if os.path.exists(os.path.join(projectDir, "SimulationData", sim_folder, key)):
            if rerun:
                # clear previous results
                shutil.rmtree(os.path.join(projectDir, "SimulationData", sim_folder, key))
                os.mkdir(os.path.join(projectDir, "SimulationData", sim_folder, key))
                _run_tune()
        else:
            _run_tune()


def run_eigenmode_parallel(shape_space, shape_space_multi, eigenmode_config,
                           projectDir, subdir=''):
    if 'processes' in eigenmode_config.keys():
        processes = eigenmode_config['processes']
        assert processes > 0, error('Number of proceses must be greater than zero.')
    else:
        processes = 1

    # split shape_space for different processes/ MPI share process by rank
    keys = list(shape_space.keys())

    # check if number of processors selected is greater than the number of keys in the pseudo shape space
    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    share = int(round(shape_space_len / processes))

    jobs = []
    # for p in range(processes):
    #     if p < processes - 1:
    #         proc_keys_list = keys[p * share:p * share + share]
    #     else:
    #         proc_keys_list = keys[p * share:]
    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0
    for p in range(processes):
        # Determine the size of the current chunk
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        start_idx += current_chunk_size

        processor_shape_space = {key: shape_space[key] for key in proc_keys_list}
        processor_shape_space_multi = {key: shape_space_multi[key] for key in proc_keys_list}
        service = mp.Process(target=run_eigenmode_s, args=(processor_shape_space, processor_shape_space_multi,
                                                           projectDir, eigenmode_config, subdir))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_eigenmode_s(shape_space, shape_space_multi, projectDir, eigenmode_config, subdir):
    """
    Run eigenmode analysis

    Parameters
    ----------
    shape_space
    shape_space_multi
    projectDir
    eigenmode_config

    Returns
    -------

    """

    rerun = True
    if 'rerun' in eigenmode_config.keys():
        assert isinstance(eigenmode_config['rerun'], bool), error('rerun must be boolean.')
        rerun = eigenmode_config['rerun']

    # perform all necessary checks
    processes = 1
    if 'processes' in eigenmode_config.keys():
        assert eigenmode_config['processes'] > 0, error('Number of proceses must be greater than zero.')
        assert isinstance(eigenmode_config['processes'], int), error('Number of proceses must be integer.')
    else:
        eigenmode_config['processes'] = processes

    freq_shifts = 0
    if 'f_shifts' in eigenmode_config.keys():
        freq_shifts = eigenmode_config['f_shifts']

    boundary_conds = 'mm'
    if 'boundary_conditions' in eigenmode_config.keys():
        eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[eigenmode_config['boundary_conditions']]
    else:
        eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[boundary_conds]

    # uq_config = None
    # if 'uq_config' in eigenmode_config.keys():
    #     uq_config = eigenmode_config['uq_config']
    #     if uq_config:
    #         assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
    #                                                                              "be equal to the number of "
    #                                                                              "variables.")
    # else:
    #     eigenmode_config['uq_config'] = uq_config

    def _run_ngsolve(name, shape, shape_multi, eigenmode_config, projectDir, subdir):

        start_time = time.time()
        if shape['kind'] == 'elliptical cavity':
            n_cells = shape['n_cells']
            n_modules = 1
            bc = eigenmode_config['boundary_conditions']
            # create folders for all keys
            ngsolve_mevp.createFolder(name, projectDir, subdir=subdir, opt=eigenmode_config['opt'])

            if 'OC_R' in shape.keys():
                OC_R = 'OC_R'
            else:
                OC_R = 'OC'

            if shape['CELL PARAMETERISATION'] == 'flattop':
                write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
                                    projectDir=projectDir, cell_type="None",
                                    solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)

                ngsolve_mevp.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                            n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc,
                                            beampipes=shape['BP'], sim_folder=solver_save_dir,
                                            parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
                                            eigenmode_config=eigenmode_config)

            elif shape['CELL PARAMETERISATION'] == 'multicell':
                write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
                                    projectDir=projectDir, cell_type="None",
                                    solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)
                ngsolve_mevp.cavity_multicell(n_cells, n_modules, shape_multi['IC'], shape_multi['OC'],
                                              shape_multi[OC_R],
                                              n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc,
                                              beampipes=shape['BP'], sim_folder=solver_save_dir,
                                              parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
                                              eigenmode_config=eigenmode_config)
            else:
                write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
                                    projectDir=projectDir, cell_type="None",
                                    solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)

                ngsolve_mevp.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                    n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc, beampipes=shape['BP'],
                                    sim_folder=solver_save_dir,
                                    parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
                                    eigenmode_config=eigenmode_config)
        elif shape['kind'] == 'vhf gun':
            ngsolve_mevp.createFolder(name, projectDir, subdir=subdir, opt=eigenmode_config['opt'])
            ngsolve_mevp.vhf_gun(fid=f"{name}", shape=shape, sim_folder=solver_save_dir,
                                 parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
                                 eigenmode_config=eigenmode_config)

        elif shape['kind'] == 'pillbox cavity':
            pass
        else:
            pass

        # run UQ
        if 'uq_config' in eigenmode_config.keys():
            uq_config = eigenmode_config['uq_config']
            if uq_config:
                opt = False
                if eigenmode_config['solver_save_directory'].lower() == 'optimisation':
                    opt = True
                objectives = uq_config['objectives']
                solver_dict = {'ngsolvemevp': ngsolve_mevp}
                solver_args_dict = {'eigenmode': eigenmode_config,
                                    'n_cells': n_cells,
                                    'n_modules': n_modules,
                                    'parentDir': SOFTWARE_DIRECTORY,
                                    'projectDir': projectDir,
                                    'analysis folder': eigenmode_config['solver_save_directory'],
                                    'cell_type': 'mid-cell',
                                    'cell_parameterisation': shape['CELL PARAMETERISATION'],
                                    'optimisation': opt
                                    }

                uq_cell_complexity = 'simplecell'
                if 'cell_complexity' in uq_config.keys():
                    uq_cell_complexity = uq_config['cell_complexity']

                if uq_cell_complexity == 'multicell':
                    shape_space = {name: shape_multi}
                    uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict)
                else:
                    shape_space = {name: shape}
                    uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')

        done(f'Done with Cavity {name}. Time: {time.time() - start_time}')

    for i, (key, shape) in enumerate(list(shape_space.items())):
        # if isinstance(freq_shifts, int) or isinstance(freq_shifts, float):
        #     freq_shift = freq_shifts
        # else:
        #     freq_shift = freq_shifts[i]
        #
        # if isinstance(boundary_conds, str) or boundary_conds is None:
        #     boundary_cond = boundary_conds
        # else:
        #     boundary_cond = boundary_conds[i]

        solver_save_dir = eigenmode_config['solver_save_directory']

        if os.path.exists(os.path.join(projectDir, "SimulationData", solver_save_dir, key)):
            if rerun:
                # delete old results
                shutil.rmtree(os.path.join(projectDir, "SimulationData", solver_save_dir, key))
                _run_ngsolve(key, shape, shape_space_multi[key], eigenmode_config, projectDir, subdir)

            else:
                # check if eigenmode analysis results exist
                if os.path.exists(os.path.join(projectDir, "SimulationData", solver_save_dir, key, "monopole",
                                               "qois.json")):
                    pass
                else:
                    shutil.rmtree(os.path.join(projectDir, "SimulationData", solver_save_dir, key))
                    _run_ngsolve(key, shape, shape_space_multi[key], eigenmode_config, projectDir, subdir)
        else:
            _run_ngsolve(key, shape, shape_space_multi[key], eigenmode_config, projectDir, subdir)


def run_wakefield_parallel(shape_space, shape_space_multi, wakefield_config, projectDir, marker='', rerun=True):
    processes = wakefield_config['processes']
    # split shape_space for different processes/ MPI share process by rank
    keys = list(shape_space.keys())

    # # check if number of processors selected is greater than the number of keys in the pseudo shape space
    # if processes > len(keys):
    #     processes = len(keys)

    shape_space_len = len(keys)
    # share = int(round(shape_space_len / processes))
    jobs = []
    # for p in range(processes):
    #     # try:
    #     if p < processes - 1:
    #         proc_keys_list = keys[p * share:p * share + share]
    #     else:
    #         proc_keys_list = keys[p * share:]

    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0
    for p in range(processes):
        # Determine the size of the current chunk
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        start_idx += current_chunk_size

        processor_shape_space = {key: shape_space[key] for key in proc_keys_list}
        processor_shape_space_multi = {key: shape_space_multi[key] for key in proc_keys_list}
        service = mp.Process(target=run_wakefield_s, args=(processor_shape_space, processor_shape_space_multi,
                                                           wakefield_config, projectDir, marker, rerun))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_wakefield_s(shape_space, shape_space_multi, wakefield_config, projectDir, marker, rerun):
    MROT = wakefield_config['polarisation']
    MT = wakefield_config['MT']
    NFS = wakefield_config['NFS']
    UBT = wakefield_config['wake_config']['wakelength']
    bunch_length = wakefield_config['beam_config']['bunch_length']
    DDR_SIG = wakefield_config['mesh_config']['DDR_SIG']
    DDZ_SIG = wakefield_config['mesh_config']['DDZ_SIG']

    operating_points = None
    if 'operating_points' in wakefield_config.keys():
        operating_points = wakefield_config['operating_points']

    uq_config = wakefield_config['uq_config']
    # if uq_config:
    #     assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
    #                                                                          "be equal to the number of "
    #                                                                          "variables.")

    WG_M = None

    def _run_abci(name, n_cells, n_modules, shape, shape_multi, wakefield_config, projectDir, WG_M=None, marker=''):

        freq = 0
        R_Q = 0
        # run abci code
        if WG_M is None:
            WG_M = ['']

        start_time = time.time()
        # run both polarizations if MROT == 2
        for ii in WG_M:
            # run abci code
            # run both polarizations if MROT == 2
            if 'OC_R' in list(shape.keys()):
                OC_R = 'OC_R'
            else:
                OC_R = 'OC'

            if MROT == 2:
                for m in range(2):
                    if shape['kind'] == 'elliptical cavity':
                        if shape['CELL PARAMETERISATION'] == 'simplecell':
                            abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                             fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                             DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                             projectDir=projectDir,
                                             WG_M=ii, marker=ii, wakefield_config=wakefield_config)
                        if shape['CELL PARAMETERISATION'] == 'flattop':
                            abci_geom.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                                     fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                                     projectDir=projectDir,
                                                     WG_M=ii, marker=ii, wakefield_config=wakefield_config)

            else:
                for m in range(2):
                    if shape['kind'] == 'elliptical cavity':
                        if shape['CELL PARAMETERISATION'] == 'simplecell':
                            abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                             fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                             DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                             projectDir=projectDir,
                                             WG_M=ii, marker=ii, wakefield_config=wakefield_config)
                        if shape['CELL PARAMETERISATION'] == 'flattop':
                            abci_geom.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                                     fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                                     projectDir=projectDir,
                                                     WG_M=ii, marker=ii, wakefield_config=wakefield_config)

        done(f'Cavity {name}. Time: {time.time() - start_time}')

        if uq_config:
            objectives = uq_config['objectives']
            solver_dict = {'abci': abci_geom}
            solver_args_dict = {'wakefield': wakefield_config,
                                'n_cells': n_cells,
                                'n_modules': n_modules,
                                'parentDir': SOFTWARE_DIRECTORY,
                                'projectDir': projectDir,
                                'analysis folder': 'ABCI',
                                # 'cell_type': cell_type,
                                'cell_parameterisation': shape['CELL PARAMETERISATION'],
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
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'wakefield')

        if operating_points:
            try:
                # check folder for freq and R/Q
                folder = os.path.join(projectDir, 'SimulationData', 'NGSolveMEVP', name, 'monopole', 'qois.json')
                if os.path.exists(folder):
                    try:
                        with open(folder, 'r') as json_file:
                            fm_results = json.load(json_file)
                        freq = fm_results['freq [MHz]']
                        R_Q = fm_results['R/Q [Ohm]']
                    except OSError:
                        info("To run analysis for working points, eigenmode simulation has to be run first"
                             "to obtain the cavity operating frequency and R/Q")

                if freq != 0 and R_Q != 0:
                    d = {}
                    # save qois
                    for key_op, vals in operating_points.items():
                        WP = key_op
                        I0 = float(vals['I0 [mA]'])
                        Nb = float(vals['Nb [1e11]'])
                        sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                        bl_diff = ['SR', 'BS']

                        # info("Running wakefield analysis for given operating points.")
                        for i, s in enumerate(sigma_z):
                            for ii in WG_M:
                                fid = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
                                OC_R = 'OC'
                                if 'OC_R' in shape.keys():
                                    OC_R = 'OC_R'
                                for m in range(2):
                                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                                     fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                     bunch_length=s,
                                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                                     projectDir=projectDir,
                                                     WG_M=ii, marker=ii, sub_dir=f"{name}")

                                dirc = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                                # try:
                                k_loss = abs(ABCIData(dirc, f'{fid}', 0).loss_factor['Longitudinal'])
                                k_kick = abs(ABCIData(dirc, f'{fid}', 1).loss_factor['Transverse'])
                                # except:
                                #     k_loss = 0
                                #     k_kick = 0

                                d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, n_cells)

                    # save qoi dictionary
                    run_save_directory = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The working point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()

    for i, (key, shape) in enumerate(shape_space.items()):
        if os.path.exists(os.path.join(projectDir, "SimulationData", "ABCI", key)):
            if rerun:
                # remove old simulation results
                shutil.rmtree(os.path.join(projectDir, "SimulationData", "ABCI", key))
                os.mkdir(os.path.join(projectDir, "SimulationData", "ABCI", key))

                _run_abci(key, shape['n_cells'], 1, shape, shape_space_multi[key], wakefield_config,
                          projectDir, WG_M, marker)
            else:
                # check if eigenmode analysis results exist
                if os.path.exists(os.path.join(projectDir, "SimulationData", "ABCI", key, "qois.json")):
                    pass
                else:
                    _run_abci(key, shape['n_cells'], 1, shape, shape_space_multi[key], wakefield_config, projectDir,
                              WG_M, marker)
        else:
            _run_abci(key, shape['n_cells'], 1, shape, shape_space_multi[key], wakefield_config, projectDir, WG_M,
                      marker)


def uq_parallel(shape_space, objectives, solver_dict, solver_args_dict,
                solver):
    """

    Parameters
    ----------
    key: str | int
        Cavity geomery identifier
    shape: dict
        Dictionary containing geometric dimensions of cavity geometry
    qois: list
        Quantities of interest considered in uncertainty quantification
    n_cells: int
        Number of cavity cells
    n_modules: int
        Number of modules
    n_modes: int
        Number of eigenmodes to be calculated
    f_shift: float
        Since the eigenmode solver uses the power method, a shift can be provided
    bc: int
        Boundary conditions {1:inner contour, 2:Electric wall Et = 0, 3:Magnetic Wall En = 0, 4:Axis, 5:metal}
        bc=33 means `Magnetic Wall En = 0` boundary condition at both ends
    pol: int {Monopole, Dipole}
        Defines whether to calculate for monopole or dipole modes
    parentDir: str | path
        Parent directory
    projectDir: str|path
        Project directory

    Returns
    -------
    :param select_solver:

    """

    if solver == 'eigenmode':
        # parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        uq_config = solver_args_dict['eigenmode']['uq_config']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        # opt = solver_args_dict['optimisation']
        # delta = uq_config['delta']

        method = ['stroud3']
        if 'method' in uq_config.keys():
            method = uq_config['method']

        uq_vars = uq_config['variables']
        # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

        for key, shape in shape_space.items():
            # n_cells = shape['n_cells']
            uq_path = projectDir / fr'SimulationData\{analysis_folder}\{key}'

            result_dict_eigen = {}
            result_dict_eigen_all_modes = {}
            # eigen_obj_list = []

            # for o in objectives:
            #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
            #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
            #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            #         eigen_obj_list.append(o)

            rdim = len(uq_vars)
            degree = 1

            if isinstance(method, str):
                flag = method
            else:
                flag = method[0]

            if flag.lower() == 'stroud3':
                nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.
                # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this
                # gives a less accurate answer. the nodes are not the same as the custom function
            elif flag.lower() == 'stroud5':
                nodes_, weights_ = cn_leg_05_2(rdim)
            elif flag.lower() == 'cn_gauss':
                nodes_, weights_ = cn_gauss(rdim, 2)
            elif flag.lower() == 'lhc':
                sampler = qmc.LatinHypercube(d=rdim)
                _ = sampler.reset()
                nsamp = 2500
                sample = sampler.random(n=nsamp)
                # ic(qmc.discrepancy(sample))
                l_bounds = [-1, -1, -1, -1, -1, -1]
                u_bounds = [1, 1, 1, 1, 1, 1]
                sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

                nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
            else:
                # issue warning
                warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
                nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.

            # save nodes and weights
            data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
            data_table.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'nodes.csv'),
                              index=False, sep='\t', float_format='%.32f')

            data_table_w = pd.DataFrame(weights_, columns=['weights'])
            data_table_w.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'weights.csv'),
                                index=False,
                                sep='\t', float_format='%.32f')

            no_parm, no_sims = np.shape(nodes_)
            # if delta is None:
            #     delta = [0.05 for _ in range(len(uq_vars))]

            sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

            proc_count = 1
            if 'processes' in uq_config.keys():
                assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
                assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
                proc_count = uq_config['processes']
            if proc_count > no_sims:
                proc_count = no_sims

            share = int(round(no_sims / proc_count))
            jobs = []
            # for p in range(proc_count):
            #     # try:
            #     end_already = False
            #     if p != proc_count - 1:
            #         if (p + 1) * share < no_sims:
            #             proc_keys_list = np.arange(p * share, p * share + share)
            #         else:
            #             proc_keys_list = np.arange(p * share, no_sims)
            #             end_already = True
            #
            #     if p == proc_count - 1 and not end_already:
            #         proc_keys_list = np.arange(p * share, no_sims)

            base_chunk_size = no_sims // proc_count
            remainder = no_sims % proc_count

            start_idx = 0
            for p in range(proc_count):
                # Determine the size of the current chunk
                current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
                proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
                start_idx += current_chunk_size

                processor_nodes = nodes_[:, proc_keys_list]
                processor_weights = weights_[proc_keys_list]
                service = mp.Process(target=uq, args=(key, objectives, uq_config, uq_path,
                                                      solver_args_dict, sub_dir,
                                                      proc_keys_list, processor_nodes, p, shape, solver))

                service.start()
                jobs.append(service)

            for job in jobs:
                job.join()

            # combine results from processes
            # qois_result_dict = {}
            # Ttab_val_f = []
            # keys = []
            for i1 in range(proc_count):
                if i1 == 0:
                    df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
                    df_all_modes = pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')
                else:
                    df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])
                    df_all_modes = pd.concat(
                        [df_all_modes, pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')])

            df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
            df.to_excel(uq_path / 'table.xlsx', index=False)

            Ttab_val_f = df.to_numpy()
            # print(Ttab_val_f.shape, weights_.shape)
            mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

            # # append results to dict
            # for i, o in enumerate(eigen_obj_list):
            #     result_dict_eigen[o]['expe'].append(mean_obj[i])
            #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
            for i, o in enumerate(df.columns):
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                result_dict_eigen[o]['expe'].append(mean_obj[i])
                result_dict_eigen[o]['stdDev'].append(std_obj[i])
                result_dict_eigen[o]['skew'].append(skew_obj[i])
                result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
            with open(uq_path / fr'uq.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))

            # for all modes
            df_all_modes.to_csv(uq_path / 'table_all_modes.csv', index=False, sep='\t', float_format='%.32f')
            df_all_modes.to_excel(uq_path / 'table_all_modes.xlsx', index=False)

            Ttab_val_f_all_modes = df_all_modes.to_numpy()
            # print(Ttab_val_f_all_modes.shape, weights_.shape)
            # print()
            mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
                Ttab_val_f_all_modes, weights_)
            # print(mean_obj_all_modes)

            # # append results to dict
            # for i, o in enumerate(eigen_obj_list):
            #     result_dict_eigen[o]['expe'].append(mean_obj[i])
            #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
            for i, o in enumerate(df_all_modes.columns):
                result_dict_eigen_all_modes[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                result_dict_eigen_all_modes[o]['expe'].append(mean_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['stdDev'].append(std_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['skew'].append(skew_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['kurtosis'].append(kurtosis_obj_all_modes[i])

            with open(uq_path / fr'uq_all_modes.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))

    elif solver == 'wakefield':
        # parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        solver_args = solver_args_dict['wakefield']
        # n_cells = solver_args['n_cells']
        uq_config = solver_args['uq_config']
        # delta = uq_config['delta']

        method = 'stroud3'
        if 'method' in uq_config.keys():
            method = uq_config['method']

        uq_vars = uq_config['variables']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']

        # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

        for key, shape in shape_space.items():
            # n_cells = shape['n_cells']
            uq_path = projectDir / fr'SimulationData\ABCI\{key}'
            result_dict_wakefield = {}
            # wakefield_obj_list = []

            # for o in objectives:
            #     if isinstance(o, list):
            #         if o[1].split(' ')[0] in ['ZL', 'ZT']:
            #             result_dict_wakefield[o[1]] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            #             wakefield_obj_list.append(o[1])
            #     else:
            #         if o in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
            #             result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            #             wakefield_obj_list.append(o)

            rdim = len(uq_vars)
            degree = 1

            if isinstance(method, str):
                flag = method
            else:
                flag = method[0]

            if flag.lower() == 'stroud3':
                nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.
                # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this
                # gives a less accurate answer. the nodes are not the same as the custom function
            elif flag.lower() == 'stroud5':
                nodes_, weights_ = cn_leg_05_2(rdim)
            elif flag.lower() == 'cn_gauss':
                nodes_, weights_ = cn_gauss(rdim, 2)
            elif flag.lower() == 'lhc':
                sampler = qmc.LatinHypercube(d=rdim)
                _ = sampler.reset()
                nsamp = 2500
                sample = sampler.random(n=nsamp)
                # ic(qmc.discrepancy(sample))
                l_bounds = [-1, -1, -1, -1, -1, -1]
                u_bounds = [1, 1, 1, 1, 1, 1]
                sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

                nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
            else:
                # issue warning
                warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
                nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.

            # save nodes
            data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
            data_table.to_csv(os.path.join(projectDir, "SimulationData", analysis_folder, key, 'nodes.csv'),
                              index=False, sep='\t', float_format='%.32f')

            data_table_w = pd.DataFrame(weights_, columns=['weights'])
            data_table_w.to_csv(os.path.join(projectDir, "SimulationData", analysis_folder, key, 'weights.csv'), index=False,
                                sep='\t', float_format='%.32f')

            no_parm, no_sims = np.shape(nodes_)
            # if delta is None:
            #     delta = [0.05 for _ in range(len(uq_vars))]

            sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

            proc_count = 1
            if 'processes' in uq_config.keys():
                assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
                assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
                proc_count = uq_config['processes']

            jobs = []

            # if proc_count > no_sims:
            #     proc_count = no_sims
            #
            # share = int(round(no_sims / proc_count))
            # for p in range(proc_count):
            #     # try:
            #     end_already = False
            #     if p != proc_count - 1:
            #         if (p + 1) * share < no_sims:
            #             proc_keys_list = np.arange(p * share, p * share + share)
            #         else:
            #             proc_keys_list = np.arange(p * share, no_sims)
            #             end_already = True
            #
            #     if p == proc_count - 1 and not end_already:
            #         proc_keys_list = np.arange(p * share, no_sims)

            base_chunk_size = no_sims // proc_count
            remainder = no_sims % proc_count

            start_idx = 0
            for p in range(proc_count):
                # Determine the size of the current chunk
                current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
                proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
                start_idx += current_chunk_size

                processor_nodes = nodes_[:, proc_keys_list]
                processor_weights = weights_[proc_keys_list]

                service = mp.Process(target=uq, args=(key, objectives, uq_config, uq_path,
                                                      solver_args_dict, sub_dir, proc_keys_list, processor_nodes,
                                                      p, shape, solver))

                service.start()
                jobs.append(service)

            for job in jobs:
                job.join()

            # combine results from processes
            # qois_result_dict = {}
            # keys = []
            # Ttab_val_f = []
            for i1 in range(proc_count):
                if i1 == 0:
                    df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
                else:
                    df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])

            df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
            df.to_excel(uq_path / 'table.xlsx', index=False)
            Ttab_val_f = df.to_numpy()
            mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)
            # # append results to dict
            # for i, o in enumerate(wakefield_obj_list):
            #     result_dict_wakefield[o]['expe'].append(mean_obj[i])
            #     result_dict_wakefield[o]['stdDev'].append(std_obj[i])
            #
            for i, o in enumerate(df.columns):
                result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                result_dict_wakefield[o]['expe'].append(mean_obj[i])
                result_dict_wakefield[o]['stdDev'].append(std_obj[i])
                result_dict_wakefield[o]['skew'].append(skew_obj[i])
                result_dict_wakefield[o]['kurtosis'].append(kurtosis_obj[i])

            with open(uq_path / fr'uq.json', 'w') as f:
                f.write(json.dumps(result_dict_wakefield, indent=4, separators=(',', ': ')))

    else:
        pass


def uq(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
       proc_keys_list, processor_nodes, proc_num, shape, solver):
    """

    Parameters
    ----------
    shape_space: dict
        Cavity geometry parameter space
    objectives: list | ndarray
        Array of objective functions
    solver_dict: dict
        Python dictionary of solver settings
    solver_args_dict: dict
        Python dictionary of solver arguments
    uq_config:
        Python dictionary of uncertainty quantification settings

    Returns
    -------
    :param n_cells:

    """

    print(processor_nodes)
    if solver == 'eigenmode':
        parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        opt = solver_args_dict['optimisation']

        epsilon, delta = [], []
        if 'epsilon' in uq_config.keys():
            epsilon = uq_config['epsilon']
        if 'delta' in uq_config.keys():
            delta = uq_config['delta']
        if 'perturbed_cell' in uq_config.keys():
            perturbed_cell = uq_config['perturbed_cell']
        else:
            perturbed_cell = 'mid-cell'

        # method = uq_config['method']
        uq_vars = uq_config['variables']
        cell_parameterisation = solver_args_dict['cell_parameterisation']
        err = False
        result_dict_eigen = {}
        Ttab_val_f = []
        Ttab_val_f_all_modes = []

        # eigen_obj_list = objectives
        eigen_obj_list = []

        for o in objectives:
            if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                     "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                eigen_obj_list.append(o)

        # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
        #     cell_node = shape['IC']
        # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
        #     cell_node = shape['OC']
        # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
        #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
        #     cell_node = shape['OC']
        # else:
        #     cell_node = shape['OC']
        cell_node = shape['IC']
        cell_node_left = shape['OC']
        cell_node_right = shape['OC_R']
        print('before: ', cell_node, cell_node_left, cell_node_right)
        perturbed_cell_node = np.array(cell_node)
        perturbed_cell_node_left = np.array(cell_node_left)
        perturbed_cell_node_right = np.array(cell_node_right)

        for i1, proc_key in enumerate(proc_keys_list):
            skip = False
            for j, uq_var in enumerate(uq_vars):
                uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
                if epsilon:
                    if perturbed_cell.lower() == 'mid cell' or perturbed_cell.lower() == 'mid-cell' or perturbed_cell.lower() == 'mid_cell':
                        perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
                    elif perturbed_cell.lower() == 'end cell' or perturbed_cell.lower() == 'end-cell' or perturbed_cell.lower() == 'end_cell':
                        perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * \
                                                                processor_nodes[
                                                                    j, i1]
                        perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
                                                                 processor_nodes[j, i1]
                    else:
                        perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
                        perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * \
                                                                processor_nodes[
                                                                    j, i1]
                        perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
                                                                 processor_nodes[j, i1]

                else:
                    if perturbed_cell.lower() == 'mid cell' or perturbed_cell.lower() == 'mid-cell' or perturbed_cell.lower() == 'mid_cell':
                        perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])
                    elif perturbed_cell.lower() == 'end cell' or perturbed_cell.lower() == 'end-cell' or perturbed_cell.lower() == 'end_cell':
                        perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])
                        perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])
                    else:
                        perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])
                        perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])
                        perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
                                1 + delta[j] * processor_nodes[j, i1])

            print('\tafter: ', perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)

            # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
            #     # cell_node = shape['IC']
            #     mid = perturbed_cell_node
            #     left = shape['OC']
            #     right = shape['OC_R']
            # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
            #     mid = shape['IC']
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
            #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
            #     mid = perturbed_cell_node
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # elif cell_type.lower() == 'end cell' or cell_type.lower() == 'end-cell' or cell_type.lower() == 'end_cell':
            #     mid = shape['IC']
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # else:
            #     mid = perturbed_cell_node
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node

            enforce_Req_continuity(perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)

            # perform checks on geometry
            fid = fr'{key}_Q{proc_key}'

            # check if folder exists and skip if it does
            if os.path.exists(os.path.join(projectDir, 'SimulationData', analysis_folder, key, fid)):
                skip = True
                info(["Skipped: ", fid, os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)])

            if not skip:
                ngsolve_mevp.createFolder(fid, projectDir, subdir=sub_dir, opt=opt)
                # it does not seem to make sense to perform uq on a multi cell by repeating the same perturbation
                # to all multi cells at once. For multicells, the uq_multicell option is more suitable as it creates
                # independent perturbations to all cells individually

                if 'tune_config' in uq_config.keys():
                    tune_config = uq_config['tune_config']
                    # tune first before running
                    processor_shape_space = {fid:
                                                 {'IC': perturbed_cell_node,
                                                  'OC': perturbed_cell_node_left,
                                                  'OC_R': perturbed_cell_node_right,
                                                  'BP': 'both',
                                                  "n_cells": 9,
                                                  'CELL PARAMETERISATION': 'simplecell'
                                                  },
                                             }
                    # save tune results to uq cavity folders
                    sim_folder = os.path.join(analysis_folder, key)

                    run_tune_s(processor_shape_space, tune_config['parameters'], tune_config['freqs'],
                               tune_config['cell_types'], tune_config, projectDir, False, fid, sim_folder)

                else:
                    if cell_parameterisation == 'simplecell':
                        ngsolve_mevp.cavity(shape['n_cells'], 1, perturbed_cell_node,
                                            perturbed_cell_node_left, perturbed_cell_node_right,
                                            f_shift=0, bc=33, beampipes=shape['BP'],
                                            fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
                                            projectDir=projectDir,
                                            subdir=sub_dir)
                    if cell_parameterisation == 'flattop':
                        ngsolve_mevp.cavity_flattop(shape['n_cells'], 1, perturbed_cell_node,
                                                    perturbed_cell_node_left, perturbed_cell_node_right,
                                                    f_shift=0, bc=33,
                                                    beampipes=shape['BP'],
                                                    fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
                                                    projectDir=projectDir,
                                                    subdir=sub_dir)

            filename = uq_path / f'{fid}/monopole/qois.json'
            filename_all_modes = uq_path / f'{fid}/monopole/qois_all_modes.json'
            if os.path.exists(filename):
                qois_result_dict = dict()

                with open(filename) as json_file:
                    qois_result_dict.update(json.load(json_file))
                qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)

                tab_val_f = qois_result
                Ttab_val_f.append(tab_val_f)
            else:
                err = True

            # for all modes
            if os.path.exists(filename_all_modes):
                qois_result_dict_all_modes = dict()
                qois_result_all_modes = {}

                with open(filename_all_modes) as json_file:
                    qois_result_dict_all_modes.update(json.load(json_file))

                tab_val_f_all_modes = []
                for kk, val in qois_result_dict_all_modes.items():
                    qois_result_all_modes[kk] = get_qoi_value(val, eigen_obj_list)

                    tab_val_f_all_modes.append(qois_result_all_modes[kk])

                tab_val_f_all_modes_flat = [item for sublist in tab_val_f_all_modes for item in sublist]
                Ttab_val_f_all_modes.append(tab_val_f_all_modes_flat)
            else:
                err = True

        data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
        data_table.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')

        # for all modes
        keys = qois_result_dict_all_modes.keys()
        eigen_obj_list_all_modes = [f"{name.split(' ')[0]}_{i} {name.split(' ', 1)[1]}" for i in keys for name in
                                    eigen_obj_list]
        data_table = pd.DataFrame(Ttab_val_f_all_modes, columns=eigen_obj_list_all_modes)
        data_table.to_csv(uq_path / fr'table_{proc_num}_all_modes.csv', index=False, sep='\t', float_format='%.32f')

    elif solver == 'wakefield':
        Ttab_val_f = []
        parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        solver_args = solver_args_dict['wakefield']
        # n_cells = solver_args_dict['n_cells']
        # n_modules = solver_args_dict['n_modules']
        MROT = solver_args['polarisation']
        MT = solver_args['MT']
        NFS = solver_args['NFS']
        UBT = solver_args['wakelength']
        bunch_length = solver_args['bunch_length']
        DDR_SIG = solver_args['DDR_SIG']
        DDZ_SIG = solver_args['DDZ_SIG']
        # WG_M = solver_args['WG_M']

        epsilon, delta = [], []
        if 'epsilon' in uq_config.keys():
            epsilon = uq_config['epsilon']
        if 'delta' in uq_config.keys():
            delta = uq_config['delta']

        # method = uq_config['method']
        cell_parameterisation = solver_args_dict['cell_parameterisation']
        uq_vars = uq_config['variables']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        # marker = solver_args['marker']

        result_dict_wakefield = {}
        # wakefield_obj_list = objectives
        wakefield_obj_list = []

        for o in objectives:
            if isinstance(o, list):
                if o[1].split(' ')[0] in ['ZL', 'ZT']:
                    result_dict_wakefield[o[1]] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                    wakefield_obj_list.append(o[1])
            else:
                if o in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
                    result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                    wakefield_obj_list.append(o)

        # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
        #     cell_node = shape['IC']
        # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
        #     cell_node = shape['OC']
        # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
        #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
        #     cell_node = shape['OC']
        # else:
        #     cell_node = shape['OC']

        cell_node = shape['IC']
        cell_node_left = shape['OC']
        cell_node_right = shape['OC_R']
        perturbed_cell_node = np.array(cell_node)
        perturbed_cell_node_left = np.array(cell_node_left)
        perturbed_cell_node_right = np.array(cell_node_right)

        uq_shape_space = {}
        d_uq_op = {}
        for i1, proc_key in enumerate(proc_keys_list):
            skip = False
            for j, uq_var in enumerate(uq_vars):
                uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
                if epsilon:
                    perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
                    perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * processor_nodes[
                        j, i1]
                    perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
                                                             processor_nodes[j, i1]
                else:
                    perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (1 + delta[j] * processor_nodes[j, i1])
                    perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
                            1 + delta[j] * processor_nodes[j, i1])
                    perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
                            1 + delta[j] * processor_nodes[j, i1])

            # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
            #     # cell_node = shape['IC']
            #     mid = perturbed_cell_node
            #     left = shape['OC']
            #     right = shape['OC_R']
            # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
            #     mid = shape['IC']
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
            #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
            #     mid = perturbed_cell_node
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # elif cell_type.lower() == 'end cell' or cell_type.lower() == 'end-cell' or cell_type.lower() == 'end_cell':
            #     mid = shape['IC']
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node
            # else:
            #     mid = perturbed_cell_node
            #     left = perturbed_cell_node
            #     right = perturbed_cell_node

            enforce_Req_continuity(perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)

            # perform checks on geometry
            fid = fr'{key}_Q{proc_key}'

            # check if folder exists and skip if it does
            if os.path.exists(os.path.join(projectDir, 'SimulationData', analysis_folder, key, fid)):
                skip = True
                info(["Skipped: ", fid, os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)])

            if not skip:
                abci_geom.createFolder(fid, projectDir, subdir=sub_dir)
                for wi in range(MROT):
                    if cell_parameterisation == 'simplecell':
                        abci_geom.cavity(shape['n_cells'], 1, perturbed_cell_node,
                                         perturbed_cell_node_left, perturbed_cell_node_right, fid=fid, MROT=wi,
                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, beampipes=shape['BP'],
                                         bunch_length=bunch_length,
                                         MT=MT, NFS=NFS, UBT=UBT,
                                         parentDir=parentDir, projectDir=projectDir, WG_M='',
                                         marker='', sub_dir=sub_dir
                                         )
                    if cell_parameterisation == 'flattop':
                        abci_geom.cavity_flattop(shape['n_cells'], 1, perturbed_cell_node,
                                                 perturbed_cell_node_left, perturbed_cell_node_right, fid=fid, MROT=wi,
                                                 DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, beampipes=shape['BP'],
                                                 bunch_length=bunch_length,
                                                 MT=MT, NFS=NFS, UBT=UBT,
                                                 parentDir=parentDir, projectDir=projectDir, WG_M='',
                                                 marker='', sub_dir=sub_dir
                                                 )

            uq_shape = {'IC': perturbed_cell_node, 'OC': perturbed_cell_node_left,
                        'OC_R': perturbed_cell_node_right, 'n_cells': shape['n_cells'], 'BP': shape['BP']}
            uq_shape_space[fid] = uq_shape

            # calculate uq for operating points
            if 'operating_points' in uq_config.keys():
                operating_points = uq_config['operating_points']
                try:
                    # check folder for freq and R/Q
                    freq, R_Q = 0, 0
                    folder = os.path.join(projectDir, 'SimulationData', 'NGSolveMEVP', key, fid, 'monopole', 'qois.json')
                    if os.path.exists(folder):
                        try:
                            with open(folder, 'r') as json_file:
                                fm_results = json.load(json_file)
                            freq = fm_results['freq [MHz]']
                            R_Q = fm_results['R/Q [Ohm]']
                        except OSError:
                            info("To run analysis for working points, eigenmode simulation has to be run first"
                                 "to obtain the cavity operating frequency and R/Q")

                    if freq != 0 and R_Q != 0:
                        d = {}
                        # save qois
                        for key_op, vals in operating_points.items():
                            WP = key_op
                            I0 = float(vals['I0 [mA]'])
                            Nb = float(vals['Nb [1e11]'])
                            sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                            bl_diff = ['SR', 'BS']

                            # info("Running wakefield analysis for given operating points.")
                            for i, s in enumerate(sigma_z):
                                for ii in ['']:
                                    fid_op = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
                                    OC_R = 'OC'
                                    if 'OC_R' in uq_shape.keys():
                                        OC_R = 'OC_R'
                                    for m in range(2):
                                        abci_geom.cavity(shape['n_cells'], 1, uq_shape['IC'], uq_shape['OC'],
                                                         uq_shape[OC_R],
                                                         fid=fid_op, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                         bunch_length=s,
                                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                                         projectDir=projectDir,
                                                         WG_M=ii, marker=ii, sub_dir=fr"{key}\{fid}")

                                    dirc = os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)
                                    # try:
                                    k_loss = abs(ABCIData(dirc, f'{fid_op}', 0).loss_factor['Longitudinal'])
                                    k_kick = abs(ABCIData(dirc, f'{fid_op}', 1).loss_factor['Transverse'])
                                    # except:
                                    #     k_loss = 0
                                    #     k_kick = 0

                                    d[fid_op] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, shape['n_cells'])

                        d_uq_op[fid] = d
                        # save qoi dictionary
                        run_save_directory = os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)
                        with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
                            json.dump(d, f, indent=4, separators=(',', ': '))

                        done("Done with the secondary analysis for working points")
                    else:
                        info("To run analysis for working points, eigenmode simulation has to be run first"
                             "to obtain the cavity operating frequency and R/Q")
                except KeyError:
                    error('The working point entered is not valid. See below for the proper input structure.')
                    show_valid_operating_point_structure()

        df_uq_op = _data_uq_op(d_uq_op)

        wakefield_folder = os.path.join(projectDir, 'SimulationData', 'ABCI', key)
        data_table, _ = get_wakefield_objectives_value(uq_shape_space, uq_config['objectives_unprocessed'],
                                                       wakefield_folder)

        # merge with data table
        if d_uq_op:
            data_table_merged = pd.merge(df_uq_op, data_table, on='key', how='inner')
        else:
            data_table_merged = data_table
        data_table_merged = data_table_merged.set_index('key')
        data_table_merged.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')


def uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict):
    """

    Parameters
    ----------
    shape_space: dict
        Cavity geometry parameter space
    objectives: list | ndarray
        Array of objective functions
    solver_dict: dict
        Python dictionary of solver settings
    solver_args_dict: dict
        Python dictionary of solver arguments
    uq_config:
        Python dictionary of uncertainty quantification settings

    Returns
    -------

    """
    parentDir = solver_args_dict['parentDir']
    projectDir = solver_args_dict['projectDir']
    uq_config = solver_args_dict['eigenmode']['uq_config']
    cell_type = uq_config['cell_type']
    analysis_folder = solver_args_dict['analysis folder']
    opt = solver_args_dict['optimisation']
    delta = uq_config['delta']

    method = 'stroud3'
    if 'method' in uq_config.keys():
        method = uq_config['method']

    uq_vars = uq_config['variables']
    n_cells = solver_args_dict['eigenmode']['n_cells']
    assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

    for key, shape in shape_space.items():
        # n_cells = shape['IC'].shape[1] + 1
        uq_path = projectDir / fr'SimulationData\NGSolveMEVP\{key}'
        err = False
        result_dict_eigen, result_dict_abci = {}, {}
        run_eigen, run_abci = False, False
        eigen_obj_list, abci_obj_list = [], []

        for o in objectives:
            if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                     "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                run_eigen = True
                eigen_obj_list.append(o)

            if o.split(' ')[0] in ['ZL', 'ZT', 'k_loss', 'k_kick']:
                result_dict_abci[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                run_abci = True
                abci_obj_list.append(o)

        cav_var_list = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        midcell_var_dict = dict()
        for i1 in range(len(cav_var_list)):
            for i2 in range(n_cells):
                for i3 in range(2):
                    midcell_var_dict[f'{cav_var_list[i1]}_{i2}_m{i3}'] = [i1, i2, i3]

        # create random variables
        multicell_mid_vars = shape['IC']

        if n_cells == 1:
            # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            p_true = [np.array(shape['OC'])[:7], np.array(shape['OC_R'])[:7]]
            rdim = len(np.array(shape['OC'])[:7]) + len(
                np.array(shape['OC_R'])[:7])  # How many variabels will be considered as random in our case 5
        else:
            # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            p_true = [np.array(shape['OC'])[:7], multicell_mid_vars, np.array(shape['OC_R'])[:7]]
            rdim = len(np.array(shape['OC'])[:7]) + multicell_mid_vars.size + len(
                np.array(shape['OC_R'])[:7])  # How many variabels will be considered as random in our case 5
            # rdim = rdim - (n_cells*2 - 1)  # <- reduce dimension by making iris and equator radii to be equal

        # ic(rdim, multicell_mid_vars.size)
        # rdim = n_cells*3  # How many variables will be considered as random in our case 5
        degree = 1

        if isinstance(method, str):
            flag = method
        else:
            flag = method[0]

        if flag == 'stroud3':
            nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
            nodes_ = 2. * nodes_ - 1.
            # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this gives a less accurate answer.
            # the nodes are not the same as the custom function
        elif flag == 'stroud5':
            nodes_, weights_ = cn_leg_05_2(rdim)
        elif flag == 'cn_gauss':
            nodes_, weights_ = cn_gauss(rdim, 2)
        elif flag == 'lhc':
            sampler = qmc.LatinHypercube(d=rdim)
            _ = sampler.reset()
            nsamp = 3000
            sample = sampler.random(n=nsamp)
            # ic(qmc.discrepancy(sample))
            l_bounds = -np.ones(rdim)
            u_bounds = np.ones(rdim)
            sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

            nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
        elif flag == 'load_from_file':
            nodes_ = pd.read_csv(fr'C:\Users\sosoho\DakotaProjects\Cavity\C3795_lhs\sim_result_table.dat',
                                 sep='\\s+').iloc[:, 2:-2]
            nodes_ = nodes_.to_numpy().T
            weights_ = np.ones((nodes_.shape[1], 1))
        else:
            # defaults to Stroud3
            nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
            nodes_ = 2. * nodes_ - 1.

        # save nodes
        data_table = pd.DataFrame(nodes_.T)
        data_table.to_csv(uq_path / 'nodes.csv', index=False, sep='\t', float_format='%.32f')

        data_table_w = pd.DataFrame(weights_, columns=['weights'])
        data_table_w.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'weights.csv'), index=False, sep='\t',
                            float_format='%.32f')

        #  mean value of geometrical parameters
        no_parm, no_sims = np.shape(nodes_)

        sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

        proc_count = 1
        if 'processes' in uq_config.keys():
            assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
            assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
            proc_count = uq_config['processes']

        # if proc_count > no_sims:
        #     proc_count = no_sims
        #
        # share = int(round(no_sims / proc_count))
        # for p in range(proc_count):
        #     # try:
        #     end_already = False
        #     if p != proc_count - 1:
        #         if (p + 1) * share < no_sims:
        #             proc_keys_list = np.arange(p * share, p * share + share)
        #         else:
        #             proc_keys_list = np.arange(p * share, no_sims)
        #             end_already = True
        #
        #     if p == proc_count - 1 and not end_already:
        #         proc_keys_list = np.arange(p * share, no_sims)
        jobs = []

        base_chunk_size = no_sims // proc_count
        remainder = no_sims % proc_count

        start_idx = 0
        for p in range(proc_count):
            # Determine the size of the current chunk
            current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
            proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
            start_idx += current_chunk_size

            processor_nodes = nodes_[:, proc_keys_list]
            processor_weights = weights_[proc_keys_list]

            service = mp.Process(target=uq_multicell_s, args=(
                n_cells, 1, shape, objectives, n_cells, 0, 33, 'monopole', parentDir,
                projectDir, sub_dir, key, uq_path,
                proc_keys_list, processor_nodes, processor_weights, p, p_true))

            service.start()
            jobs.append(service)

        for job in jobs:
            job.join()

        # combine results from processes
        qois_result_dict = {}
        Ttab_val_f = []
        keys = []
        for i1 in range(no_sims):
            if i1 == 0:
                df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
            else:
                try:
                    df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])
                except:
                    pass

        df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
        df.to_excel(uq_path / 'table.xlsx', index=False)

        Ttab_val_f = df.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

        # append results to dict
        for i, o in enumerate(eigen_obj_list):
            result_dict_eigen[o]['expe'].append(mean_obj[i])
            result_dict_eigen[o]['stdDev'].append(std_obj[i])
            result_dict_eigen[o]['skew'].append(skew_obj[i])
            result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])

        with open(uq_path / fr'uq.json', 'w') as file:
            file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))


def uq_multicell_s(n_cells, n_modules, shape, qois, n_modes, f_shift, bc, pol, parentDir, projectDir, sub_dir,
                   key, uq_path, proc_keys_list, processor_nodes,
                   proc_num, p_true):
    start = time.time()
    err = False
    result_dict_eigen = {}
    Ttab_val_f = []
    eigen_obj_list = qois

    for o in qois:
        result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}

    for i1 in proc_keys_list:
        skip = False
        if n_cells == 1:
            p_init_el = p_true[0] + processor_nodes[0:len(p_true[0]), i1 - min(proc_keys_list)]
            p_init_er = p_true[1] + processor_nodes[len(p_true[0]):, i1 - min(proc_keys_list)]
            par_mid = p_init_el
        else:
            proc_node = processor_nodes[:, i1 - min(proc_keys_list)]
            # # one dimension of nodes_ is dimension of number of variables. The variables must be expanded to the unreduced dimension
            # # by filling the missing slots with radius values. Insert from end of list, index(Req) + 7 and index(Ri) + 6
            # moved_val_indx = 7
            # proc_nodes_len = len(processor_nodes)
            # for i2 in range(2 * n_cells - 1):
            #     if i2 % 2 == 0:
            #         proc_node = np.insert(proc_node, proc_nodes_len - moved_val_indx + 7, proc_node[proc_nodes_len - moved_val_indx])
            #         # update index
            #         moved_val_indx += 7
            #     else:
            #         proc_node = np.insert(proc_node, proc_nodes_len - moved_val_indx + 6, proc_node[proc_nodes_len - moved_val_indx])
            #         moved_val_indx += 5

            # p_init_el = processor_nodes[0:len(p_true[0]), i1 - min(proc_keys_list)]
            # p_init_m = processor_nodes[len(p_true[0]):len(p_true[0]) + p_true[1].size,
            #            i1 - min(proc_keys_list)].reshape(np.shape(p_true[1])[::-1]).T
            # p_init_er = processor_nodes[len(p_true[0]) + p_true[1].size:, i1 - min(proc_keys_list)]

            p_init_el = p_true[0] + processor_nodes[0:len(p_true[0]), i1 - min(proc_keys_list)]
            p_init_m = p_true[1] + processor_nodes[len(p_true[0]):len(p_true[0]) + p_true[1].size,
                                   i1 - min(proc_keys_list)].reshape(np.shape(p_true[1]))
            p_init_er = p_true[2] + processor_nodes[len(p_true[0]) + p_true[1].size:, i1 - min(proc_keys_list)]
            # ic(proc_node, proc_node.shape)

            # p_init_el = p_true[0] + proc_node[0:len(p_true[0])]
            # p_init_m = p_true[1] + proc_node[len(p_true[0]):len(p_true[0])+p_true[1].size].reshape(np.shape(p_true[1]))
            # p_init_er = p_true[2] + proc_node[len(p_true[0])+p_true[1].size:]

            par_mid = p_init_m

        par_end_l = p_init_el
        par_end_r = p_init_er

        fid = fr'{key}_Q{i1}'

        # check if folder already exist (simulation already completed)

        if os.path.exists(uq_path / f'{fid}/monopole/qois.json'):
            skip = True
            info(f'processor {proc_num} skipped ', fid, 'Result already exists.')

        # skip analysis if folder already exists.
        if not skip:
            solver = ngsolve_mevp
            #  run model using SLANS or CST
            # # create folders for all keys
            solver.createFolder(fid, projectDir, subdir=sub_dir)

            solver.cavity_multicell(n_cells, n_modules, par_mid, par_end_l, par_end_r,
                                    n_modes=n_modes, fid=fid, f_shift=f_shift, bc=bc, pol=pol,
                                    beampipes=shape['BP'],
                                    parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

        filename = uq_path / f'{fid}/monopole/qois.json'
        if os.path.exists(filename):
            qois_result_dict = dict()

            with open(filename) as json_file:
                qois_result_dict.update(json.load(json_file))

            qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)
            # sometimes some degenerate shapes are still generated and the solver returns zero
            # for the objective functions, such shapes are considered invalid
            # for objr in qois_result:
            #     if objr == 0:
            #         # skip key
            #         err = True
            #         break

            tab_val_f = qois_result
            Ttab_val_f.append(tab_val_f)
        else:
            err = True

    # save table
    data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
    data_table.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')


def get_wakefield_objectives_value(d, objectives_unprocessed, abci_data_dir):
    k_loss_array_transverse = []
    k_loss_array_longitudinal = []
    k_loss_M0 = []
    key_list = []

    # create list to hold Z
    Zmax_mon_list = []
    Zmax_dip_list = []
    xmax_mon_list = []
    xmax_dip_list = []
    processed_keys_mon = []
    processed_keys_dip = []

    def calc_k_loss():
        for key, value in d.items():
            abci_data_long = ABCIData(abci_data_dir, key, 0)
            abci_data_trans = ABCIData(abci_data_dir, key, 1)

            # trans
            x, y, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            # long
            x, y, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            abci_data_long.get_data('Loss Factor Spectrum Integrated up to F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            # append only after successful run
            k_loss_M0.append(k_M0)
            k_loss_array_longitudinal.append(k_loss_HOM)
            k_loss_array_transverse.append(k_loss_trans)

        return [k_loss_M0, k_loss_array_longitudinal, k_loss_array_transverse]

    def get_Zmax_L(mon_interval=None):
        if mon_interval is None:
            mon_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                abci_data_mon = ABCIData(abci_data_dir, f"{key}", 0)

                # get longitudinal and transverse impedance plot data
                xr_mon, yr_mon, _ = abci_data_mon.get_data('Real Part of Longitudinal Impedance')
                xi_mon, yi_mon, _ = abci_data_mon.get_data('Imaginary Part of Longitudinal Impedance')

                # Zmax
                if mon_interval is None:
                    mon_interval = [[0.0, 10]]

                # calculate magnitude
                ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]

                # get peaks
                peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
                xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

                for i, z_bound in enumerate(mon_interval):
                    # get mask
                    msk_mon = [(z_bound[0] < x < z_bound[1]) for x in xp_mon]

                    if len(yp_mon[msk_mon]) != 0:
                        Zmax_mon = max(yp_mon[msk_mon])

                        Zmax_mon_list[i].append(Zmax_mon)
                    elif len(yp_mon) != 0:
                        Zmax_mon_list[i].append(0)
                    else:
                        error("skipped, yp_mon = [], raise exception")
                        raise Exception()

                processed_keys_mon.append(key)
            except:
                info("skipped, yp_mon = []")
                # for i, z_bound in enumerate(mon_interval):
                #     Zmax_mon_list[i].append(-1)

        return Zmax_mon_list

    def get_Zmax_T(dip_interval=None):
        if dip_interval is None:
            dip_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                abci_data_dip = ABCIData(abci_data_dir, f"{key}", 1)

                xr_dip, yr_dip, _ = abci_data_dip.get_data('Real Part of Transverse Impedance')
                xi_dip, yi_dip, _ = abci_data_dip.get_data('Imaginary Part of Transverse Impedance')

                # Zmax
                if dip_interval is None:
                    dip_interval = [[0.0, 10]]

                # calculate magnitude
                ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

                # get peaks
                peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
                xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

                for i, z_bound in enumerate(dip_interval):
                    # get mask
                    msk_dip = [(z_bound[0] < x < z_bound[1]) for x in xp_dip]

                    if len(yp_dip[msk_dip]) != 0:
                        Zmax_dip = max(yp_dip[msk_dip])

                        Zmax_dip_list[i].append(Zmax_dip)
                    elif len(yp_dip) != 0:
                        Zmax_dip_list[i].append(0)
                    else:
                        error("skipped, yp_dip = [], raise exception")
                        raise Exception()

                processed_keys_dip.append(key)
            except:
                error("skipped, yp_dip = []")
                # for i, z_bound in enumerate(dip_interval):
                #     Zmax_dip_list[i].append(-1)

        return Zmax_dip_list

    def all(mon_interval, dip_interval):
        for key, value in d.items():
            abci_data_long = ABCIData(abci_data_dir, f"{key}_", 0)
            abci_data_trans = ABCIData(abci_data_dir, f"{key}_", 1)

            # get longitudinal and transverse impedance plot data
            xr_mon, yr_mon, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            xi_mon, yi_mon, _ = abci_data_long.get_data('Imaginary Part of Longitudinal Impedance')

            xr_dip, yr_dip, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            xi_dip, yi_dip, _ = abci_data_trans.get_data('Imaginary Part of Transverse Impedance')

            # loss factors
            # trans
            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            # long
            abci_data_long.get_data('Loss Factor Spectrum Integrated upto F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            # calculate magnitude
            ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]
            ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

            # get peaks
            peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
            xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

            peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
            xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

            for i, z_bound in enumerate(mon_interval):
                # get mask
                msk_mon = [(z_bound[0] < x < z_bound[1]) for x in xp_mon]

                if len(yp_mon[msk_mon]) != 0:
                    Zmax_mon = max(yp_mon[msk_mon])
                    xmax_mon = xp_mon[np.where(yp_mon == Zmax_mon)][0]

                    Zmax_mon_list[i].append(Zmax_mon)
                    xmax_mon_list[i].append(xmax_mon)
                elif len(yp_mon) != 0:
                    Zmax_mon_list[i].append(0.0)
                    xmax_mon_list[i].append(0.0)
                else:
                    continue

            for i, z_bound in enumerate(dip_interval):
                # get mask
                msk_dip = [(z_bound[0] < x < z_bound[1]) for x in xp_dip]

                if len(yp_dip[msk_dip]) != 0:
                    Zmax_dip = max(yp_dip[msk_dip])
                    xmax_dip = xp_dip[np.where(yp_dip == Zmax_dip)][0]

                    Zmax_dip_list[i].append(Zmax_dip)
                    xmax_dip_list[i].append(xmax_dip)
                elif len(yp_dip) != 0:
                    Zmax_dip_list[i].append(0.0)
                    xmax_dip_list[i].append(0.0)
                else:
                    continue

            # append only after successful run

            k_loss_M0.append(k_M0)
            k_loss_array_longitudinal.append(k_loss_HOM)
            k_loss_array_transverse.append(k_loss_trans)

    ZL, ZT = [], []
    df_ZL, df_ZT = pd.DataFrame(), pd.DataFrame()
    for obj in objectives_unprocessed:
        if "ZL" in obj[1]:
            freq_range = process_interval(obj[2])
            for i in range(len(freq_range)):
                Zmax_mon_list.append([])
                xmax_mon_list.append([])
                df_ZL[f"{obj[1]} [max({freq_range[i][0]}<f<{freq_range[i][1]})]"] = 0

            ZL = get_Zmax_L(freq_range)

        elif "ZT" in obj[1]:
            freq_range = process_interval(obj[2])

            for i in range(len(freq_range)):
                Zmax_dip_list.append([])
                xmax_dip_list.append([])
                # df_ZT[obj[1]] = 0
                df_ZT[f"{obj[1]} [max({freq_range[i][0]}<f<{freq_range[i][1]})]"] = 0

            ZT = get_Zmax_T(freq_range)

        elif obj[1] == "k_loss":
            pass
        elif obj[1] == "k_kick":
            pass

    # create dataframes from list
    df_ZL.loc[:, :] = np.array(ZL).T
    df_ZT.loc[:, :] = np.array(ZT).T
    df_ZL['key'] = processed_keys_mon
    df_ZT['key'] = processed_keys_dip

    processed_keys = list(set(processed_keys_mon) & set(processed_keys_dip))

    # ZL, ZT = np.array(ZL).T, np.array(ZT).T

    if len(ZL) != 0 and len(ZT) != 0:
        df_wake = df_ZL.merge(df_ZT, on='key', how='inner')
        # obj_result = np.hstack((ZL, ZT))
    elif len(ZL) != 0:
        df_wake = df_ZL
        # obj_result = ZL
    else:
        df_wake = df_ZT
        # obj_result = ZT

    return df_wake, processed_keys


def process_interval(interval_list):
    interval = []
    for i in range(len(interval_list) - 1):
        interval.append([interval_list[i], interval_list[i + 1]])

    return interval


def get_qois_value(f_fm, R_Q, k_loss, k_kick, sigma_z, I0, Nb, n_cell):
    c = 299792458
    w_fm = 2 * np.pi * f_fm * 1e6
    e = 1.602e-19

    k_fm = (w_fm / 4) * R_Q * np.exp(-(w_fm * sigma_z * 1e-3 / c) ** 2) * 1e-12
    k_hom = k_loss - k_fm
    p_hom = (k_hom * 1e12) * (I0 * 1e-3) * e * (Nb * 1e11)

    d = {
        "n cell": n_cell,
        # "freq [MHz]": f_fm,
        "R/Q [Ohm]": R_Q,
        "k_FM [V/pC]": k_fm,
        "I0 [mA]": I0,
        "sigma_z [mm]": sigma_z,
        "Nb [1e11]": Nb,
        "|k_loss| [V/pC]": k_loss,
        "|k_kick| [V/pC/m]": k_kick,
        "P_HOM [kW]": p_hom * 1e-3
    }
    return d


def process_objectives(objectives):
    processed_objectives = []
    weights = []
    for i, obj in enumerate(objectives):
        if obj[1] == "ZL" or obj[1] == "ZT":
            goal = obj[0]
            freq_ranges = process_interval(obj[2])
            for f in freq_ranges:
                processed_objectives.append([goal, f"{obj[1]} [max({f[0]}<f<{f[1]})]", f])
                weights.append(1)
        else:
            goal = obj[0]
            if goal == 'equal':
                processed_objectives.append(obj)
            else:
                processed_objectives.append(obj)
            weights.append(1)

    return processed_objectives, weights


def show_valid_operating_point_structure():
    dd = {
        '<wp1>': {
            'I0 [mA]': '<value>',
            'Nb [1e11]': '<value>',
            'sigma_z (SR/BS) [mm]': '<value>'
        },
        '<wp2>': {
            'I0 [mA]': '<value>',
            'Nb [1e11]': '<value>',
            'sigma_z (SR/BS) [mm]': '<value>'
        }
    }

    info(dd)


def get_surface_resistance(Eacc, b, m, freq, T):
    Rs_dict = {
        "Rs_NbCu_2K_400.79Mhz": 0.57 * (Eacc * 1e-6 * b) + 28.4,  # nOhm
        "Rs_NbCu_4.5K_400.79Mhz": 39.5 * np.exp(0.014 * (Eacc * 1e-6 * b)) + 27,  # nOhm
        "Rs_bulkNb_2K_400.79Mhz": (2.33 / 1000) * (Eacc * 1e-6 * b) ** 2 + 26.24,  # nOhm
        "Rs_bulkNb_4.5K_400.79Mhz": 0.0123 * (Eacc * 1e-6 * b) ** 2 + 62.53,  # nOhm

        "Rs_NbCu_2K_801.58Mhz": 1.45 * (Eacc * 1e-6 * b) + 92,  # nOhm
        "Rs_NbCu_4.5K_801.58Mhz": 50 * np.exp(0.033 * (Eacc * 1e-6 * b)) + 154,  # nOhm
        "Rs_bulkNb_2K_801.58Mhz": (16.4 + Eacc * 1e-6 * b * 0.092) * (800 / 704) ** 2,  # nOhm
        "Rs_bulkNb_4.5K_801.58Mhz": 4 * (62.7 + (Eacc * 1e-6 * b) ** 2 * 0.012)  # nOhm
    }
    if freq < 600:
        freq = 400.79

    if freq >= 600:
        freq = 801.58

    rs = Rs_dict[fr"Rs_{m}_{T}K_{freq}Mhz"]

    return rs


def axis_data_coords_sys_transform(axis_obj_in, xin, yin, inverse=False):
    """ inverse = False : Axis => Data
                    = True  : Data => Axis
        """
    if axis_obj_in.get_yscale() == 'log':
        xlim = axis_obj_in.get_xlim()
        ylim = axis_obj_in.get_ylim()

        x_delta = xlim[1] - xlim[0]

        if not inverse:
            x_out = xlim[0] + xin * x_delta
            y_out = ylim[0] ** (1 - yin) * ylim[1] ** yin
        else:
            x_delta2 = xin - xlim[0]
            x_out = x_delta2 / x_delta
            y_out = np.log(yin / ylim[0]) / np.log(ylim[1] / ylim[0])

    else:
        xlim = axis_obj_in.get_xlim()
        ylim = axis_obj_in.get_ylim()

        x_delta = xlim[1] - xlim[0]
        y_delta = ylim[1] - ylim[0]

        if not inverse:
            x_out = xlim[0] + xin * x_delta
            y_out = ylim[0] + yin * y_delta
        else:
            x_delta2 = xin - xlim[0]
            y_delta2 = yin - ylim[0]
            x_out = x_delta2 / x_delta
            y_out = y_delta2 / y_delta

    return x_out, y_out


def _get_nodes_and_weights(uq_config, rdim, degree):
    method = uq_config['method']
    uq_vars = uq_config['variables']

    if method[1].lower() == 'stroud3':
        nodes, weights, bpoly = quad_stroud3(rdim, degree)
        nodes = 2. * nodes - 1.
        # nodes, weights = cn_leg_03_1(rdim)
    elif method[1].lower() == 'stroud5':
        nodes, weights = cn_leg_05_2(rdim)
    elif method[1].lower() == 'gaussian':
        nodes, weights = cn_gauss(rdim, 2)
    elif method[1].lower() == 'lhs':
        sampler = qmc.LatinHypercube(d=rdim)
        _ = sampler.reset()
        nsamp = uq_config['integration'][2]
        sample = sampler.random(n=nsamp)

        l_bounds = [-1 for _ in range(len(uq_vars))]
        u_bounds = [1 for _ in range(len(uq_vars))]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

        nodes, weights = sample_scaled.T, np.ones((nsamp, 1))
    elif method[0].lower() == 'from file':
        if len(method) == 2:
            nodes = pd.read_csv(method[1], sep='\\s+').iloc[:, method[1]]
        else:
            nodes = pd.read_csv(method[1], sep='\\s+')

        nodes = nodes.to_numpy().T
        weights = np.ones((nodes.shape[1], 1))
    else:
        # issue warning
        warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
        nodes, weights, bpoly = quad_stroud3(rdim, degree)
        nodes = 2. * nodes - 1.

    return nodes, weights


def add_text(ax, text, box, xy=(0.5, 0.5), xycoords='data', xytext=None, textcoords='data',
             size=14, rotation=0, arrowprops=None):
    """

    Parameters
    ----------
    text: str
        Matplotlib annotation text
    box
    xy: tuple
        Coordinates of annotation text
    xycoords: str {data, axis}
        Coordinate system reference
    xytext
    textcoords
    size
    rotation: float
        Annotation text rotation
    arrowprops

    Returns
    -------

    """
    if text.strip("") == "":
        return

    # add text
    if xytext:
        bbox_props = dict(boxstyle='{}'.format(box), fc='w', ec='k')
        annotext = ax.annotate(text, xy=xy, xycoords=xycoords,
                               xytext=xytext, textcoords=textcoords, bbox=bbox_props, fontsize=size,
                               rotation=rotation, arrowprops=arrowprops, zorder=500)
    else:
        if box == "None":
            annotext = ax.annotate(text, xy=xy, xycoords=xycoords, fontsize=size,
                                   rotation=rotation, arrowprops=arrowprops, zorder=500)
        else:
            bbox_props = dict(boxstyle='{}'.format(box), fc='w', ec='k')
            annotext = ax.annotate(text, xy=xy, xycoords=xycoords, bbox=bbox_props, fontsize=size,
                                   rotation=rotation, arrowprops=arrowprops, zorder=500)

    ax.get_figure().canvas.draw_idle()
    ax.get_figure().canvas.flush_events()
    return annotext


def _data_uq_op(d_uq_op):
    # Initialize an empty list to store the rows
    rows = []

    # Iterate over the dictionary
    for main_key, sub_dict in d_uq_op.items():
        row = {'key': main_key}
        for sub_key, metrics in sub_dict.items():
            for metric, value in metrics.items():
                # Check if the metric is one of the desired ones
                if metric in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
                    # Create a new key combining the metric and the sub_key
                    new_key = f"{metric}_{sub_key}"
                    # Add the value to the row
                    row[new_key] = value
        # Add the row to the list
        rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df


def show_welcome():
    import base64
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    image_path = os.path.join(filename, 'docs/images/cavsim2d_logo_1x.png')

    # Convert the image to base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    # HTML and CSS to display image and colourful text on the same line
    message = f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_image}" style="height: 32px;">
        <p style="margin: 0; font-size: 16px; 
                  background: -webkit-linear-gradient(left, #EFC3CA, #5DE2E7, #FE9900, #E7DDFF, #FFDE59);
                  -webkit-background-clip: text; 
                  -webkit-text-fill-color: transparent;">
            <b>CAV-SIM-2D</b> loaded successfully!
        </p>
    </div>
    """

    # Display the HTML
    display(HTML(message))
