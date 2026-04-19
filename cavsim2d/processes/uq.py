"""Parallel uncertainty quantification process functions."""
import json
import multiprocessing as mp
import os.path
import re
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.analysis.wakefield.abci_geometry import ABCIGeometry
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.solvers.ABCI.abci import ABCI
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

ngsolve_mevp = NGSolveMEVP()
abci = ABCI()
abci_geom = ABCIGeometry()

def uq_parallel(cav, eigenmode_config, solver='eigenmode'):
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
    if solver == 'eigenmode':

        if not os.path.exists(cav.uq_dir):
            os.mkdir(cav.uq_dir)

        result_dict_eigen = {}
        result_dict_eigen_all_modes = {}
        # eigen_obj_list = []

        # for o in objectives:
        #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
        #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
        #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
        #         eigen_obj_list.append(o)

        # print("multicell shape space", shape_space)
        # cav_var_list = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        # midcell_var_dict = dict()
        # for i1 in range(len(cav_var_list)):
        #     for i2 in range(n_cells):
        #         for i3 in range(2):
        #             midcell_var_dict[f'{cav_var_list[i1]}_{i2}_m{i3}'] = [i1, i2, i3]

        # # create random variables
        # multicell_mid_vars = shape['IC']
        # print(multicell_mid_vars)

        # if n_cells == 1:
        #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
        #     p_true = [np.array(shape['OC'])[:7], np.array(shape['OC_R'])[:7]]
        #     rdim = len(np.array(shape['OC'])[:7]) + len(
        #         np.array(shape['OC_R'])[:7])
        # else:
        #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
        #     p_true = [np.array(shape['OC'])[:7], multicell_mid_vars, np.array(shape['OC_R'])[:7]]
        #     rdim = len(np.array(shape['OC'])[:7]) + multicell_mid_vars.size + len(
        #         np.array(shape['OC_R'])[:7])
        #     # rdim = rdim - (n_cells*2 - 1)  # <- reduce dimension by making iris and equator radii to be equal
        #
        # ic(rdim, multicell_mid_vars.size)
        # rdim = n_cells*3  # How many variables will be considered as random in our case 5

        perturbed_cavities, weights_ = perturb_geometry(cav, eigenmode_config)

        nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
        # save nodes
        nodes_perturbed.to_csv(os.path.join(cav.uq_dir, 'nodes_before_continuity.csv'), index=False, sep='\t', float_format='%.32f')

        # enforce continuity
        # nodes_ = enforce_continuity_df(nodes_perturbed)
        nodes_ = nodes_perturbed
        # save nodes
        nodes_.to_csv(os.path.join(cav.uq_dir, 'nodes.csv'), index=False, sep='\t', float_format='%.32f')

        # spawn cavity objects
        cavs_object = cav.spawn(nodes_, os.path.join(cav.self_dir, 'uq'))

        uq_cfg = eigenmode_config['uq_config']
        
        # Save UQ config for traceability
        with open(os.path.join(cav.uq_dir, 'uq_config.json'), 'w') as f:
            json.dump(eigenmode_config, f, indent=4, default=str)
        if 'tune_config' in uq_cfg and uq_cfg['tune_config']:
            # Refit each perturbed variant (e.g. retune Req) before measuring
            # QoIs, then pull QoIs off the tuned cavities.
            cavs_object.run_tune(uq_cfg['tune_config'])
        else:
            # Plain UQ: run eigenmode on each perturbed variant. Strip the
            # nested uq_config so we don't recurse into another UQ sweep.
            eig_cfg = {k: v for k, v in eigenmode_config.items() if k != 'uq_config'}
            cavs_object.run_eigenmode(eig_cfg)

        uq_df = pd.DataFrame.from_dict(cavs_object.eigenmode_qois).T

        records = []
        for outer_key, mid_dict in cavs_object.eigenmode_qois_all_modes.items():
            row = {}
            for mid_key, inner_dict in mid_dict.items():
                for k, v in inner_dict.items():
                    # create column name combining key numbers
                    m = re.search(r"([\w\s/\-()]+?)\s*(\[.*?\])?$", k)
                    if m:
                        name = m.group(1).strip()  # e.g. "GR/Q"
                        unit = m.group(2) or ""  # e.g. "[Ohm^2]" or "" if none
                        col_name = f"{name}_{mid_key} {unit}".strip()
                    else:
                        col_name = f"{k}_{mid_key}"
                    row[col_name] = v
            # include the outer key as index
            row['outer_index'] = outer_key
            records.append(row)

        # Create DataFrame
        uq_df_all = pd.DataFrame(records)
        uq_df_all.set_index('outer_index', inplace=True)

        data_table_w = pd.DataFrame(weights_, columns=['weights'])
        data_table_w.to_csv(os.path.join(cav.uq_dir, 'weights.csv'),
                            index=False, sep='\t',
                            float_format='%.32f')

        uq_df.to_csv(os.path.join(cav.uq_dir, 'table.csv'), index=False, sep='\t', float_format='%.32f')
        uq_df.to_excel(os.path.join(cav.uq_dir, 'table.xlsx'), index=False)

        Ttab_val_f = uq_df.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

        for i, o in enumerate(uq_df.columns):
            result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            result_dict_eigen[o]['expe'].append(mean_obj[i])
            result_dict_eigen[o]['stdDev'].append(std_obj[i])
            result_dict_eigen[o]['skew'].append(skew_obj[i])
            result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
        with open(os.path.join(cav.uq_dir, fr'uq.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))

        # for all modes
        uq_df_all.to_csv(os.path.join(cav.uq_dir, 'table_all_modes.csv'), index=False, sep='\t', float_format='%.32f')
        uq_df_all.to_excel(os.path.join(cav.uq_dir, 'table_all_modes.xlsx'), index=False)

        Ttab_val_f_all_modes = uq_df_all.to_numpy()

        mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
            Ttab_val_f_all_modes, weights_)

        for i, o in enumerate(uq_df_all.columns):
            # Extract mode number and base property name from column like 'freq_0 [MHz]'
            match = re.search(r'^(.*?)_(\d+)(\s*\[.*\])?$', o)
            if match:
                prop_base = match.group(1).strip()
                mode_idx = match.group(2)
                unit = match.group(3).strip() if match.group(3) else ""
                base_name = f"{prop_base} {unit}".strip()
            else:
                base_name = o
                mode_idx = "0"

            if mode_idx not in result_dict_eigen_all_modes:
                result_dict_eigen_all_modes[mode_idx] = {}

            result_dict_eigen_all_modes[mode_idx][base_name] = {
                'expe': [mean_obj_all_modes[i]],
                'stdDev': [std_obj_all_modes[i]],
                'skew': [skew_obj_all_modes[i]],
                'kurtosis': [kurtosis_obj_all_modes[i]]
            }

        with open(os.path.join(cav.uq_dir, 'uq_all_modes.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))


def uq(proc_cavs_dict, eigenmode_config, proc_num):
    result_dict_eigen = {}
    Ttab_val_f = []
    Ttab_val_f_all_modes = []

    qois_result_dict_all_modes = dict()
    qois_result_all_modes = {}

    # eigen_obj_list = objectives
    eigen_obj_list = []

    for o in eigenmode_config['uq_config']['objectives']:
        if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                 "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
            result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            eigen_obj_list.append(o)

    for name, cav in proc_cavs_dict.items():

        filename = os.path.join(cav.eigenmode_dir, 'qois.json')
        filename_all_modes = os.path.join(cav.eigenmode_dir, 'qois_all_modes.json')

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
    data_table.to_csv(os.path.join(cav.projectDir, fr'table.csv'), index=False, sep='\t', float_format='%.32f')

    # for all modes
    keys = qois_result_dict_all_modes.keys()
    eigen_obj_list_all_modes = [f"{name.split(' ')[0]}_{i} {name.split(' ', 1)[1]}" for i in keys for name in
                                eigen_obj_list]
    data_table = pd.DataFrame(Ttab_val_f_all_modes, columns=eigen_obj_list_all_modes)
    data_table.to_csv(os.path.join(cav.projectDir, fr'table_{proc_num}_all_modes.csv'), index=False, sep='\t', float_format='%.32f')

def uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict, solver):
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
    if solver == 'eigenmode':
        # parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        uq_config = solver_args_dict['eigenmode']['uq_config']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        uq_vars = uq_config['variables']
        which_cell = uq_config.get('cell', 'all')
        # opt = solver_args_dict['optimisation']
        # delta = uq_config['delta']

        method = uq_config['method']
        if 'perturbation_mode' not in uq_config:
            # default: additive perturbation with bound from delta or 0.01
            uq_config['perturbation_mode'] = ['add', uq_config.get('delta', 0.01)]

        perturbation_mode = uq_config['perturbation_mode']
        if len(perturbation_mode) < 2:
            perturbation_mode.append(uq_config.get('delta', 0.01))

        if not isinstance(perturbation_mode[1], list):
            perturbation_mode[1] = [perturbation_mode[1]] * len(uq_vars)

        # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

        # print('shape_space', shape_space)
        for key, shape in shape_space.items():
            n_cells = shape['n_cells']

            uq_path = projectDir / f'{key}/eigenmode'

            result_dict_eigen = {}
            result_dict_eigen_all_modes = {}
            # eigen_obj_list = []

            # for o in objectives:
            #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
            #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
            #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            #         eigen_obj_list.append(o)

            # print("multicell shape space", shape_space)
            # cav_var_list = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
            # midcell_var_dict = dict()
            # for i1 in range(len(cav_var_list)):
            #     for i2 in range(n_cells):
            #         for i3 in range(2):
            #             midcell_var_dict[f'{cav_var_list[i1]}_{i2}_m{i3}'] = [i1, i2, i3]

            # # create random variables
            # multicell_mid_vars = shape['IC']
            # print(multicell_mid_vars)

            # if n_cells == 1:
            #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            #     p_true = [np.array(shape['OC'])[:7], np.array(shape['OC_R'])[:7]]
            #     rdim = len(np.array(shape['OC'])[:7]) + len(
            #         np.array(shape['OC_R'])[:7])
            # else:
            #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            #     p_true = [np.array(shape['OC'])[:7], multicell_mid_vars, np.array(shape['OC_R'])[:7]]
            #     rdim = len(np.array(shape['OC'])[:7]) + multicell_mid_vars.size + len(
            #         np.array(shape['OC_R'])[:7])
            #     # rdim = rdim - (n_cells*2 - 1)  # <- reduce dimension by making iris and equator radii to be equal
            #
            # ic(rdim, multicell_mid_vars.size)
            # rdim = n_cells*3  # How many variables will be considered as random in our case 5

            perturbed_cavities, weights_ = generate_perturbed_shapes(shape,
                                                                     cells=which_cell,
                                                                     variables=uq_vars,
                                                                     mode=perturbation_mode,
                                                                     node_type=method)

            nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
            # save nodes
            nodes_perturbed.to_csv(uq_path / 'nodes_before_continuity.csv', index=False, sep='\t', float_format='%.32f')

            # enforce continuity
            nodes_ = enforce_continuity_df(nodes_perturbed)

            # save nodes
            nodes_.to_csv(uq_path / 'nodes.csv', index=False, sep='\t', float_format='%.32f')

            data_table_w = pd.DataFrame(weights_, columns=['weights'])
            data_table_w.to_csv(os.path.join(projectDir, key, 'eigenmode', 'weights.csv'),
                                index=False, sep='\t',
                                float_format='%.32f')

            #  mean value of geometrical parameters
            no_sims, no_parm = np.shape(nodes_)

            sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

            proc_count = 1
            if 'processes' in uq_config.keys():
                assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
                assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
                proc_count = uq_config['processes']

            jobs = []

            base_chunk_size = no_sims // proc_count
            remainder = no_sims % proc_count

            start_idx = 0
            for p in range(proc_count):
                # Determine the size of the current chunk
                current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
                proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
                start_idx += current_chunk_size

                processor_nodes = nodes_.iloc[proc_keys_list, :]
                # processor_weights = weights_[proc_keys_list]

                service = mp.Process(target=uq_multicell_s, args=(key, objectives, uq_config, uq_path,
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
                # Extract mode number and base property name from column like 'freq_0 [MHz]'
                match = re.search(r'^(.*?)_(\d+)(\s*\[.*\])?$', o)
                if match:
                    prop_base = match.group(1).strip()
                    mode_idx = match.group(2)
                    unit = match.group(3).strip() if match.group(3) else ""
                    base_name = f"{prop_base} {unit}".strip()
                else:
                    base_name = o
                    mode_idx = "0"

                if mode_idx not in result_dict_eigen_all_modes:
                    result_dict_eigen_all_modes[mode_idx] = {}

                result_dict_eigen_all_modes[mode_idx][base_name] = {
                    'expe': [mean_obj_all_modes[i]],
                    'stdDev': [std_obj_all_modes[i]],
                    'skew': [skew_obj_all_modes[i]],
                    'kurtosis': [kurtosis_obj_all_modes[i]]
                }

            with open(uq_path / fr'uq_all_modes.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))


def uq_multicell_s(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
                   proc_keys_list, processor_multicell_nodes, proc_num, shape, solver):
    if solver == 'eigenmode':
        parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        opt = solver_args_dict['optimisation']

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

        multicell_keys, multicell_values = processor_multicell_nodes.index.to_numpy(), processor_multicell_nodes.to_numpy()
        # print('multicell_keys', multicell_keys)
        for i1, shape in zip(multicell_keys, multicell_values):

            fid = fr'{key}_Q{i1}'

            # check if folder already exist (simulation already completed)

            skip = False
            if os.path.exists(uq_path / f'{fid}/qois.json'):
                skip = True
                info(f'processor {proc_num} skipped ', fid, 'Result already exists.')

            # skip analysis if folder already exists.
            if not skip:
                solver = ngsolve_mevp
                #  run model using SLANS or CST
                # # create folders for all keys

                if 'tune_config' in uq_config.keys():
                    tune_config = uq_config['tune_config']
                    # tune first before running
                    shape_space = {fid: shape}
                    # save tune results to uq cavity folders
                    sim_folder = os.path.join(analysis_folder, key)

                    tuned_shape_space = run_tune_s_multicell(shape_space, tune_config['parameters'],
                                                             tune_config, projectDir, False, fid, sim_folder)

                    shape = tuned_shape_space[fid]

                solver.createFolder(fid, projectDir, subdir=sub_dir)
                # print("multicell after tuningsdf", shape)
                n_cells = int(len(shape) / (2 * 8))

                solver.cavity_multicell(n_cells, 1, shape,
                                        n_modes=n_cells, fid=fid, f_shift=0,
                                        beampipes='both',
                                        parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

                filename = uq_path / f'{fid}/qois.json'
                filename_all_modes = uq_path / f'{fid}/qois_all_modes.json'
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


def sa_parallel():
    pass


def sa():
    pass



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
    ax
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
