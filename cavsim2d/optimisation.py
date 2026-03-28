import datetime
import json
import os
import random
import shutil
from distutils import dir_util

import pandas as pd
from matplotlib import pyplot as plt
from paretoset import paretoset
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
        self.parentDir = SOFTWARE_DIRECTORY
        self.projectDir = cav.projectDir
        self.initial_points = config['initial_points']
        self.ng_max = config['no_of_generation']
        self.objectives_unprocessed = config['objectives']
        self.objectives, weights = process_objectives(config['objectives'])
        self.objective_vars = [obj[1] for obj in self.objectives]
        if 'weights' in config:
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
        if 'processes' in config:
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
            assert 'mid-cell' in config, error('To optimise an end-cell, mid cell dimensions are required')
            assert len(config['mid-cell']) >= 7, error('Incomplete mid cell dimension.')
            self.mid_cell = config['mid-cell']

        self.tune_parameter = self.tune_config['parameters']
        self.tune_freq = self.tune_config['freqs']

        self.wakefield_config = {}
        if (any(['ZL' in obj for obj in self.objective_vars])
                or any(['ZT' in obj for obj in self.objective_vars])
                or any([obj in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]'] for obj in
                        self.objective_vars])):
            assert 'wakefield_config' in config, error('Wakefield impedance objective detected in objectives. '
                                                       'Please include a field for wakefield_config.')
            self.wakefield_config = config['wakefield_config']

            if 'uq_config' in self.wakefield_config:
                self.uq_config = self.wakefield_config['uq_config']
                self.wakefield_config['uq_config']['objectives'] = self.objectives
                self.wakefield_config['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed

                if self.uq_config['delta']:
                    assert len(self.uq_config['delta']) == len(self.uq_config['variables']), error(
                        "The number of deltas must be equal to the number of variables.")

        if 'eigenmode_config' in config:
            self.eigenmode_config = config['eigenmode_config']

            if 'uq_config' in self.eigenmode_config:
                self.uq_config = self.eigenmode_config['uq_config']
                if self.uq_config['delta']:
                    assert len(self.uq_config['delta']) == len(self.uq_config['variables']), error(
                        "The number of deltas must be equal to the number of variables.")

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
            self.df = self.generate_first_men(self.initial_points, 0)
            self.f2_interp = [np.zeros(self.n_interp) for _ in range(len(self.objectives))]

            folder = os.path.join(self.cav.self_dir, 'optimisation')

            if os.path.exists(folder):
                for filename in os.listdir(folder):
                    try:
                        shutil.rmtree(os.path.join(folder, filename))
                    except NotADirectoryError:
                        os.remove(os.path.join(folder, filename))
            else:
                os.mkdir(folder)

        df = self.df

        # Remove duplicates already in global dataframe
        compared_cols = list(self.bounds.keys())
        if not self.df_global.empty:
            df = df.loc[~df.set_index(compared_cols).index.isin(
                self.df_global.set_index(compared_cols).index)]

        for index, row in df.iterrows():
            rw = row.tolist()

        n_cells = 1
        cavs_object = self.cav.spawn(df, os.path.join(self.cav.self_dir, 'optimisation'))
        cavs_dict = cavs_object.cavities_dict
        cavs_dict.run_eigenmode(self.eigenmode_config)

        # Get successfully tuned geometries
        processed_keys = []
        tune_result = []
        for key, scav in cavs_dict.items():
            filename = os.path.join(scav.self_dir, 'eigenmode', 'tune_res.json')
            try:
                with open(filename, 'r') as file:
                    tune_res = json.load(file)

                freq = tune_res['FREQ']
                tune_variable_value = tune_res['parameters'][tune_res['TUNED VARIABLE']]

                tune_result.append([tune_variable_value, freq])
                processed_keys.append(key)
            except FileNotFoundError:
                info(f'Results not found for {scav.self_dir}, tuning probably failed.')

        df = df.loc[processed_keys]
        df.loc[:, [self.tune_parameter, 'freq [MHz]']] = tune_result

        # Eigenmode objective variables
        intersection = set(self.objective_vars).intersection(
            {"freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"})

        if len(intersection) > 0:
            obj_result = []
            processed_keys = []
            for key, scav in cavs_dict.items():
                filename = os.path.join(scav.eigenmode_dir, 'monopole', 'qois.json')
                try:
                    with open(filename, 'r') as file:
                        qois = json.load(file)

                    obj = list(
                        {key: val for [key, val] in qois.items() if key in self.objective_vars}.values())
                    obj_result.append(obj)
                    processed_keys.append(key)
                except FileNotFoundError:
                    pass

            if len(processed_keys) == 0:
                error("Unfortunately, none survived. \n"
                      "This is most likely due to all generated initial geometries being degenerate.\n"
                      "Check the variable bounds or increase the number of initial geometries.\n"
                      "Tune ended.")
                return

            df = df.loc[processed_keys]

            obj_eigen = [o[1] for o in self.objectives if
                         o[1] in {"freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"}]
            df[obj_eigen] = obj_result

        # Wakefield objective variables
        for o in self.objectives:
            if "ZL" in o[1] or "ZT" in o[1] or o[1] in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]',
                                                        'P_HOM [kW]']:
                wake_shape_space = self.run_wakefield_opt(df, self.wakefield_config)

                df_wake, processed_keys = get_wakefield_objectives_value(wake_shape_space,
                                                                         self.objectives_unprocessed,
                                                                         self.projectDir / 'Cavities')
                df = df.merge(df_wake, on='key', how='inner')
                break

        # Apply UQ
        if self.uq_config:
            uq_result_dict = {}
            for key in df['key']:
                filename_eigen = self.projectDir / f'Cavities/{key}/eigenmode/uq.json'
                filename_abci = self.projectDir / f'Cavities/{key}/wakefield/uq.json'
                if os.path.exists(filename_eigen):
                    uq_result_dict[key] = []
                    with open(filename_eigen, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if o[1] in {"Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                                        "G [Ohm]", "Q []"}:
                                uq_result_dict[key].append(uq_d[o[1]]['expe'][0])
                                uq_result_dict[key].append(uq_d[o[1]]['stdDev'][0])
                                if o[0] == 'min':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] + 6 * uq_d[o[1]]['stdDev'][0])
                                elif o[0] == 'max':
                                    uq_result_dict[key].append(uq_d[o[1]]['expe'][0] - 6 * uq_d[o[1]]['stdDev'][0])
                                else:
                                    uq_result_dict[key].append(
                                        np.abs(uq_d[o[1]]['expe'][0] - o[2]) + uq_d[o[1]]['stdDev'][0])

                if os.path.exists(filename_abci):
                    if key not in uq_result_dict:
                        uq_result_dict[key] = []
                    with open(filename_abci, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if o[1] not in {"Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                                            "G [Ohm]", "Q []"}:
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

        # Filter by constraints
        for const in self.constraints:
            c = const.split(" ")
            op = c[1]
            val = float(c[2])
            col = c[0]

            if op == '>':
                df = df.loc[df[col] > val]
            elif op == '<':
                df = df.loc[df[col] < val]
            elif op == '<=':
                df = df.loc[df[col] <= val]
            elif op == '>=':
                df = df.loc[df[col] >= val]
            elif op == '==':
                df = df.loc[df[col] == val]

        # Update with global dataframe
        if not self.df_global.empty:
            df = pd.concat([self.df_global, df], ignore_index=True)

        # Rank shapes by objectives
        df['total_rank'] = 0

        for i, obj in enumerate(self.objectives):
            if self.uq_config:
                if obj[0] == "min":
                    col = fr'E[{obj[1]}] + 6*std[{obj[1]}]'
                    df[f'rank_{col}'] = df[col].rank() * self.weights[i]
                elif obj[0] == "max":
                    col = fr'E[{obj[1]}] - 6*std[{obj[1]}]'
                    df[f'rank_{col}'] = df[col].rank(ascending=False) * self.weights[i]
                elif obj[0] == "equal":
                    col = fr'|E[{obj[1]}] - {obj[2]}| + std[{obj[1]}]'
                    df[f'rank_{col}'] = df[col].rank() * self.weights[i]

                df['total_rank'] = df['total_rank'] + df[f'rank_{col}']
            else:
                if obj[0] == "min":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank() * self.weights[i]
                elif obj[0] == "max":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank(ascending=False) * self.weights[i]
                elif obj[0] == "equal" and obj[1] != 'freq [MHz]':
                    df[f'rank_{obj[1]}'] = (df[obj[1]] - obj[2]).abs().rank() * self.weights[i]

                df['total_rank'] = df['total_rank'] + df[f'rank_{obj[1]}']

        # Normalize and sort
        tot = df.pop('total_rank')
        df['total_rank'] = tot / sum(self.weights)

        df = df.sort_values(by=['total_rank']).reset_index(drop=True)

        # Pareto front
        reorder_indx, pareto_indx_list = self.pareto_front(df)

        # Estimate convergence
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

        if len(obj_error) != 0:
            self.interp_error.append(max(obj_error))
            self.interp_error_avg.append(np.average(self.interp_error))

        df = df.loc[reorder_indx, :]
        df = df.dropna().reset_index(drop=True)

        self.df_global = df

        if self.df_global.shape[0] == 0:
            error("Unfortunately, none survived the constraints and the program has to end.")
            return
        done(self.df_global)

        # Save dataframe
        filename = os.path.join(self.cav.self_dir, 'optimisation', f'g{n}.xlsx')
        self.recursive_save(self.df_global, filename, reorder_indx)

        # Birth next generation
        if len(df) > 1:
            df_cross = self.crossover(df, n, self.crossover_factor)
        else:
            df_cross = pd.DataFrame()

        df_mutation = self.mutation(df, n, self.mutation_factor)
        df_chaos = self.chaos(self.chaos_factor, n)

        df_ng = pd.concat([df_cross, df_mutation, df_chaos])
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
            plt.plot(self.interp_error_avg, marker='X', label='average')
            plt.plot([x + 1 for x in range(len(self.err))], self.err, marker='o', label='convex hull vol')
            plt.yscale('log')
            plt.legend()
            plt.xlabel('Generation $n$')
            plt.ylabel(r"Pareto surface interp. error")
            plt.show()
            return

    def run_uq(self, df, objectives, solver_dict, solver_args_dict, uq_config):
        """Run UQ for eigenmode and/or wakefield solvers."""
        df = df.loc[:, ['key', 'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', "alpha_i", "alpha_o"]]
        shape_space = {}

        df = df.set_index('key')
        for index, row in df.iterrows():
            rw = row.tolist()
            ct = self.cell_type.lower().replace('-', ' ').replace('_', ' ')

            if ct == 'mid cell':
                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw}
            elif ct == 'mid end cell':
                assert 'mid cell' in list(self.optimisation_config.keys()), \
                    ("If cell_type is set as 'mid-end cell', the mid cell geometry parameters must "
                     "be provided in the optimisation_config dictionary.")
                assert len(self.optimisation_config['mid cell']) > 6, ("Incomplete mid cell geometry parameter. "
                                                                       "At least 7 geometric parameters required.")
                IC = self.optimisation_config['mid cell']
                df_check = tangent_coords(*np.array(IC)[0:8], 0)
                assert df_check[-2] == 1, ("The mid-cell geometry dimensions given result in a degenerate geometry.")
                shape_space[f'{index}'] = {'IC': IC, 'OC': rw, 'OC_R': rw}
            else:
                shape_space[f'{index}'] = {'IC': rw, 'OC': rw, 'OC_R': rw}

        if solver_args_dict['eigenmode']:
            if 'uq_config' in solver_args_dict['eigenmode']:
                solver_args_dict['eigenmode']['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')

        if solver_args_dict['wakefield']:
            if 'uq_config' in solver_args_dict['wakefield']:
                solver_args_dict['wakefield']['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'wakefield')

        return shape_space

    def run_tune_opt(self, cav_dict, tune_config):
        freqs = tune_config['freqs']
        tune_parameters = tune_config['parameters']
        cell_types = tune_config['cell_types']

        processes = tune_config.get('processes', 1)
        if processes <= 0:
            error('Number of processes must be greater than zero.')
            processes = 1

        run_tune_parallel(cav_dict, tune_config)

    def run_wakefield_opt(self, df, wakefield_config):
        wakefield_config_keys = wakefield_config.keys()
        MROT = 2
        MT = 10
        NFS = 10000
        wakelength = 50
        bunch_length = 25
        DDR_SIG = 0.1
        DDZ_SIG = 0.1

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

        processes = wakefield_config.get('processes', 1)
        if processes <= 0:
            error('Number of processes must be greater than zero.')
            processes = 1
        wakefield_config['processes'] = processes

        if 'polarisation' in wakefield_config_keys:
            assert wakefield_config['polarisation'] in [0, 1, 2], error('Polarisation should be 0, 1, or 2.')
        else:
            wakefield_config['polarisation'] = MROT

        if 'MT' not in wakefield_config_keys:
            wakefield_config['MT'] = MT
        if 'NFS' not in wakefield_config_keys:
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

        return shape_space

    def generate_first_men(self, initial_points, n):
        method_name = list(self.method.keys())[0]

        if method_name == "LHS":
            seed = self.method['LHS'].get('seed') or None

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

        elif method_name == "Sobol Sequence":
            seed = self.method['LHS'].get('seed') or None

            columns = list(self.bounds.keys())
            dim = len(columns)
            index = self.method["Sobol Sequence"]['index']
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

        elif method_name == "Random":
            data = {'key': [f"G0_C{i}_P" for i in range(initial_points)]}
            for j, (var, bounds) in enumerate(self.bounds.items()):
                data[var] = random.sample(
                    list(np.linspace(bounds[0], bounds[1] + (1 if var == 'Req' else 0), initial_points * 2)),
                    initial_points)
            data['alpha_i'] = np.zeros(initial_points)
            data['alpha_o'] = np.zeros(initial_points)
            return pd.DataFrame.from_dict(data)

        elif method_name == "Uniform":
            data = {'key': [f"G0_C{i}_P" for i in range(initial_points)]}
            for j, (var, bounds) in enumerate(self.bounds.items()):
                data[var] = np.linspace(bounds[0], bounds[1] + (1 if var == 'Req' else 0), initial_points)
            data['alpha_i'] = np.zeros(initial_points)
            data['alpha_o'] = np.zeros(initial_points)
            return pd.DataFrame.from_dict(data)

    @staticmethod
    def process_constraints(constraints):
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

    def crossover(self, df, generation, f):
        elites = {}
        for i, o in enumerate(self.objectives):
            if self.uq_config:
                if o[0] == "min":
                    col = fr'E[{o[1]}] + 6*std[{o[1]}]'
                elif o[0] == "max":
                    col = fr'E[{o[1]}] - 6*std[{o[1]}]'
                else:
                    col = fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'
                elites[col] = df.sort_values(col, ascending=(o[0] != 'max'))
            else:
                elites[o[1]] = df.sort_values(o[1], ascending=(o[0] != 'max'))

        obj_dict = {}
        for o in self.objectives:
            if self.uq_config:
                if o[0] == 'min':
                    col = fr'E[{o[1]}] + 6*std[{o[1]}]'
                elif o[0] == 'max':
                    col = fr'E[{o[1]}] - 6*std[{o[1]}]'
                else:
                    col = fr'|E[{o[1]}] - {o[2]}| + std[{o[1]}]'
                obj_dict[col] = elites[col]
            else:
                obj_dict[o[1]] = elites[o[1]]

        obj = {key: o.reset_index(drop=True) for key, o in obj_dict.items()}

        df_co = pd.DataFrame(columns=self.bounds.keys())

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
                    inf_dict[key] = self.objective_vars

        n_elites_to_cross = self.elites_to_crossover

        for i in range(f):
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

        key1 = [f"G{n}_C{i}_M" for i in range(len(df_ng_mut))]
        df_ng_mut.loc[:, 'key'] = key1

        return df_ng_mut.set_index('key')

    def chaos(self, f, n):
        return self.generate_first_men(f, n)

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
        if filepath.split('.')[-1] != 'json':
            filepath = f'{filepath}.json'
        return filepath

    def recursive_save(self, df, filename, pareto_index):
        styler = self.color_pareto(df, self.poc)
        try:
            styler.to_excel(filename)
        except PermissionError:
            filename = filename.split('.xlsx')[0]
            filename = fr'{filename}_1.xlsx'
            self.recursive_save(df, filename, pareto_index)

    def pareto_front(self, df):
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

        bool_array = paretoset(datapoints, sense=sense)
        lst = np.where(bool_array)[0]
        self.poc = len(lst)

        reorder_idx = list(lst) + [i for i in range(len(df)) if i not in lst]
        return reorder_idx, lst

    @staticmethod
    def negate_list(ll, arg):
        if arg == 'max':
            return ll
        else:
            return [-x for x in ll]

    @staticmethod
    def overwriteFolder(invar, projectDir):
        path = os.path.join(projectDir, 'Cavities', '_optimisation', f'_process_{invar}')
        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}
        os.makedirs(path)

    @staticmethod
    def copyFiles(invar, parentDir, projectDir):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, 'Cavities', '_optimisation', f'_process_{invar}', 'SLANS_exe')
        dir_util.copy_tree(src, dst)

    @staticmethod
    def color_pareto(df, no_pareto_optimal):
        def color(row):
            if row.iloc[0] in df.index.tolist()[0:no_pareto_optimal]:
                return ['background-color: #6bbcd1'] * len(row)
            return [''] * len(row)

        styler = df.style
        styler.apply(color, axis=1)
        return styler
