"""Parallel tuning process functions."""
import multiprocessing as mp
import os.path
import time

import numpy as np
from cavsim2d.analysis.tune.tuner import Tuner
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

tuner = Tuner()

def run_tune_parallel(cavs_dict, tune_config, solver='NGSolveMEVP',
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
        freqs = np.array([freqs for _ in range(len(cavs_dict))])
    else:
        assert len(freqs) == len(cavs_dict), error(
            'Number of target frequencies must correspond to the number of cavities')
        freqs = np.array(freqs)

    if isinstance(tune_parameters, str):
        for key, cav in cavs_dict.items():
            assert tune_config['parameters'] in cav.parameters.keys(), error(
                fr'Please enter a valid tune parameter from \n\t{cav.name}.parameters: {cav.parameters.keys()}' )
        tune_parameters = np.array([tune_parameters for _ in range(len(cavs_dict))])
    else:
        assert len(tune_parameters) == len(cavs_dict), error(
            'Number of tune parameters must correspond to the number of cavities')
        assert len(cell_types) == len(cavs_dict), error(
            'Number of cell types must correspond to the number of cavities')
        tune_parameters = np.array(tune_parameters)

    if isinstance(cell_types, str):
        cell_types = np.array([cell_types for _ in range(len(cavs_dict))])
    else:
        assert len(cell_types) == len(cavs_dict), error(
            'Number of cell types must correspond to the number of cavities')
        cell_types = np.array(cell_types)

    # split shape_space for different processes/ MPI share process by rank
    keys = list(cavs_dict.keys())

    # check if number of processors selected is greater than the number of keys in the pseudo shape space
    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    # share = int(round(shape_space_len / processes))
    jobs = []

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

        # modify tune config
        proc_tune_config = {}
        for key, val in tune_config.items():
            if key == 'freqs':
                proc_tune_config[key] = proc_freqs
            if key == 'parameters':
                proc_tune_config[key] = proc_tune_variables

        processor_cavs_dict = {key: cavs_dict[key] for key in proc_keys_list}
        service = mp.Process(target=run_tune_s, args=(processor_cavs_dict, proc_tune_config, p))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_tune_s(processor_cavs_dict, tune_config, p):
    proc_tune_variables = tune_config['parameters']
    proc_freqs = tune_config['freqs']

    # perform necessary checks
    if tune_config is None:
        tune_config = {}
    tune_config_keys = tune_config.keys()

    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    def _run_tune():
        cav_tune_config = tune_config.copy()
        cav_tune_config['freqs'] = tune_config['freqs'][i]
        cav_tune_config['parameters'] = tune_config['parameters'][i]
        tuned_shape_space, d_tune_res, conv_dict, abs_err_dict = tuner.tune_ngsolve({key: cav}, 33,
                                                                                    proc=p,
                                                                                    tune_variable=proc_tune_variables[i],
                                                                                    tune_config=cav_tune_config)

        if d_tune_res:
            n_cells = processor_cavs_dict[key].n_cells
            tuned_shape_space[key]['n_cells'] = n_cells
            tuned_shape_space[key]['CELL PARAMETERISATION'] = processor_cavs_dict[key].cell_parameterisation

            eigenmode_config = {'target': run_eigenmode_s}
            if 'eigenmode_config' in tune_config_keys:
                eigenmode_config = tune_config['eigenmode_config']
                eigenmode_config['target']= run_eigenmode_s
            else:
                info('tune_config does not contain eigenmode_config. Default values are used for eigenmode analysis.')

            run_eigenmode_parallel({key: cav}, eigenmode_config)

            # save tune results
            save_tune_result(d_tune_res, cav.self_dir, 'tune_res.json')

            # save convergence information
            save_tune_result(conv_dict, cav.self_dir, 'tune_convergence.json',)
            save_tune_result(abs_err_dict, cav.self_dir, 'tune_absolute_error.json')

    for i, (key, cav) in enumerate(processor_cavs_dict.items()):
        cav.shape['FREQ'] = proc_freqs[i]
        if os.path.exists(os.path.join(cav.eigenmode_dir, key)):
            if rerun:
                # clear previous results
                shutil.rmtree(os.path.join(cav.eigenmode_dir, 'monopole'))
                os.mkdir(os.path.join(cav.eigenmode_dir, 'monopole'))
                _run_tune()
        else:
            _run_tune()


def run_tune_s_multicell(processor_shape_space, proc_tune_variables, tune_config, projectDir, resume,
                         p, sim_folder='Optimisation'):
    # perform necessary checks
    if tune_config is None:
        tune_config = {}
    tune_config_keys = tune_config.keys()

    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    def _run_tune(key, shape):
        tuned_shape_space, d_tune_res, conv_dict, abs_err_dict = tuner.tune_ngsolve_multicell({key: shape}, 33,
                                                                                              SOFTWARE_DIRECTORY,
                                                                                              projectDir, key,
                                                                                              resume=resume, proc=p,
                                                                                              tune_variable=
                                                                                              proc_tune_variables[i],
                                                                                              sim_folder=sim_folder,
                                                                                              tune_config=tune_config)

        # n_cells = processor_shape_space[key]['n_cells']
        # tuned_shape_space[key]['n_cells'] = n_cells
        # tuned_shape_space[key]['CELL PARAMETERISATION'] = processor_shape_space[key]['CELL PARAMETERISATION']

        return tuned_shape_space[key]

    processor_shape_space_tuned = {}
    for i, (key, shape) in enumerate(processor_shape_space.items()):
        # shape['FREQ'] = proc_freqs[i]

        if os.path.exists(os.path.join(projectDir, "Cavities", key, "eigenmode")):
            # clear previous results
            shutil.rmtree(os.path.join(projectDir, "Cavities", key, "eigenmode"))
            os.mkdir(os.path.join(projectDir, "Cavities", key, "eigenmode"))
            tuned_shape = _run_tune(key, shape)
        else:
            tuned_shape = _run_tune(key, shape)

        processor_shape_space_tuned[key] = tuned_shape

    # print('processor_shape_space_tuned', processor_shape_space_tuned)
    return processor_shape_space_tuned


