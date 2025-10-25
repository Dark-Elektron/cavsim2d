import datetime
import json
import os
import random
import shutil
from distutils import dir_util

import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
from paretoset import paretoset
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from scipy.stats import qmc
from tqdm import tqdm

from cavsim2d.constants import *
from cavsim2d.processes import *
from cavsim2d.utils.printing import *
from cavsim2d.utils.shared_functions import *


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

    def optimiser(self, cav, config):
        self.cav = cav
        self.err = []
        self.pareto_history = []
        self.optimisation_config = config
        # apply optimisation settings
        self.parentDir = SOFTWARE_DIRECTORY
        self.projectDir = cav.projectDir
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

            folder = os.path.join(self.cav.self_dir, 'optimisation')

            # clear folder to avoid reading from previous optimization attempt
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
        compared_cols = list(self.bounds.keys())
        if not self.df_global.empty:
            df = df.loc[~df.set_index(compared_cols).index.isin(
                self.df_global.set_index(compared_cols).index)]  # this line of code removes duplicates

        # pseudo_shape_space = {}
        for index, row in df.iterrows():
            rw = row.tolist()

        n_cells = 1
        # print(df)
        cavs_object = self.cav.spawn(df, os.path.join(self.cav.self_dir, 'optimisation'))
        cavs_dict = cavs_object.cavities_dict
        self.run_tune_opt(cavs_dict, self.tune_config)

        # get successfully tuned geometries and filter initial generation dictionary
        processed_keys = []
        tune_result = []
        for key, scav in cavs_dict.items():
            filename = os.path.join(scav.self_dir, 'eigenmode', 'tune_res.json')
            try:
                with open(filename, 'r') as file:
                    tune_res = json.load(file)

                # get only tune_variable, alpha_i, and alpha_o and freq but why these quantities
                freq = tune_res['FREQ']
                tune_variable_value = tune_res['parameters'][tune_res['TUNED VARIABLE']]

                tune_result.append([tune_variable_value, freq])
                processed_keys.append(key)
            except FileNotFoundError:
                info(f'Results not found for {scav.self_dir}, tuning probably failed.')

        # after removing duplicates, dataframe might change size
        df = df.loc[processed_keys]
        df.loc[:, [self.tune_parameter, 'freq [MHz]']] = tune_result
        # print(df)
        # eigen objective variables
        # for o in self.objectives:
        intersection = set(self.objective_vars).intersection(
            ["freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"])

        if len(intersection) > 0:
            # process tune results
            obj_result = []
            processed_keys = []
            for key, scav in cavs_dict.items():
                filename = os.path.join(scav.eigenmode_dir, 'monopole', 'qois.json')
                try:
                    with open(filename, 'r') as file:
                        qois = json.load(file)

                    # extract objectives from tuned cavity
                    obj = list(
                        {key: val for [key, val] in qois.items() if key in self.objective_vars}.values())

                    obj_result.append(obj)

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

            df = df.loc[processed_keys]

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
        filename = os.path.join(self.cav.self_dir, 'optimisation', f'g{n}.xlsx')
        self.recursive_save(self.df_global, filename, reorder_indx)

        # birth next generation
        # crossover
        if len(df) > 1:
            df_cross = self.crossover(df, n, self.crossover_factor)
        else:
            df_cross = pd.DataFrame()
        # print('cross', df_cross)
        # mutation
        df_mutation = self.mutation(df, n, self.mutation_factor)
        # print('mutation', df_mutation)

        # chaos
        df_chaos = self.chaos(self.chaos_factor, n)
        # print('chaos', df_chaos)

        # take elites from previous generation over to next generation
        df_ng = pd.concat([df_cross, df_mutation, df_chaos])
        # print('combined dict', df_ng)

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

    def run_tune_opt(self, cav_dict, tune_config):
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

        # if isinstance(freqs, float) or isinstance(freqs, int):
        #     freqs = np.array([freqs for _ in range(len(pseudo_shape_space))])
        # else:
        #     assert len(freqs) == len(pseudo_shape_space), error(
        #         'Number of target frequencies must correspond to the number of cavities')
        #     freqs = np.array(freqs)
        #
        # if isinstance(tune_parameters, str):
        #     assert tune_config['parameters'] in ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req'], error(
        #         'Please enter a valid tune parameter')
        #     tune_variables = np.array([tune_parameters for _ in range(len(pseudo_shape_space))])
        #     cell_types = np.array([cell_types for _ in range(len(pseudo_shape_space))])
        # else:
        #     assert len(tune_parameters) == len(pseudo_shape_space), error(
        #         'Number of tune parameters must correspond to the number of cavities')
        #     assert len(cell_types) == len(pseudo_shape_space), error(
        #         'Number of cell types must correspond to the number of cavities')
            tune_variables = np.array(tune_parameters)
            cell_types = np.array(cell_types)

        run_tune_parallel(cav_dict, tune_config)

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

        # run_wakefield_parallel(shape_space, shape_space_multicell, wakefield_config,
        #                        self.projectDir, marker='', rerun=rerun)

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


            return df.set_index('key')

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
        df_co = pd.DataFrame(columns=self.bounds.keys())

        # influence dictionary: select only best characteristics # must be reviewed later.
        inf_dict = {var: ['All'] for var in self.bounds.keys()}

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
            row_values = []

            for var, keys in inf_dict.items():
                vals = [
                    obj[key].loc[
                        np.random.randint(min(n_elites_to_cross, df.shape[0]))
                    ][var]
                    for key in keys
                ]
                row_values.append(sum(vals) / len(vals))

            df_co.loc[f"G{generation}_C{i}_CO"] = row_values
        df_co.index.name = 'key'
        return df_co

    def mutation(self, df, n, f):

        # get list based on mutation length
        if df.shape[0] < f:
            ml = np.arange(df.shape[0])
        else:
            ml = np.arange(f)

        df_ng_mut = pd.DataFrame(columns=self.bounds.keys())

        for var, bound in self.bounds.items():
            if bound[0] == bound[1]:
                df_ng_mut.loc[:, var] = df.loc[ml, var]
            else:
                df_ng_mut.loc[:, var] = df.loc[ml, var] * random.uniform(0.85, 1.5)

        key1, key2 = [], []
        for i in range(len(df_ng_mut)):
            key1.append(f"G{n}_C{i}_M")

        df_ng_mut.loc[:, 'key'] = key1

        return df_ng_mut.set_index('key')

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
            if row.iloc[0] in df.index.tolist()[0:no_pareto_optimal]:
                return ['background-color: #6bbcd1'] * len(row)
            return [''] * len(row)

        # Save Styler Object for Later
        styler = df.style
        # Apply Styles (This can be chained or on separate lines)
        styler.apply(color, axis=1)
        # Export the styler to excel
        return styler

