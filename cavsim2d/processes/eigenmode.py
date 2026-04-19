"""Parallel eigenmode analysis process functions."""
import multiprocessing as mp
import os.path
import shutil
import time
from pathlib import Path

from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.constants import *
from cavsim2d.processes.uq import uq_parallel, uq_parallel_multicell
from cavsim2d.utils.shared_functions import *

ngsolve_mevp = NGSolveMEVP()


def run_eigenmode_parallel(cavs_dict, solver_config, subdir=''):

    processes = solver_config.get('processes', 1)
    if processes <= 0:
        error('Number of processes must be greater than zero.')
        processes = 1

    keys = list(cavs_dict.keys())

    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    jobs = []
    start_idx = 0

    for p in range(processes):
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        start_idx += current_chunk_size

        processor_cavs_dict = {key: cavs_dict[key] for key in proc_keys_list}

        if processes == 1:
            # Inline path: avoids Windows spawn guard requirement and
            # surfaces stdout/stderr directly in Jupyter.
            solver_config['target'](processor_cavs_dict, solver_config, subdir)
        else:
            service = mp.Process(target=solver_config['target'], args=(processor_cavs_dict, solver_config, subdir))
            service.start()
            jobs.append(service)

    for job in jobs:
        job.join()


def run_eigenmode_s(cavs_dict, eigenmode_config, subdir):
    """Run eigenmode analysis for each cavity in the dictionary."""

    rerun = True
    if 'rerun' in eigenmode_config:
        assert isinstance(eigenmode_config['rerun'], bool), error('rerun must be boolean.')
        rerun = eigenmode_config['rerun']

    processes = 1
    if 'processes' in eigenmode_config:
        assert eigenmode_config['processes'] > 0, error('Number of processes must be greater than zero.')
        assert isinstance(eigenmode_config['processes'], int), error('Number of processes must be integer.')
    else:
        eigenmode_config['processes'] = processes

    if 'boundary_conditions' in eigenmode_config:
        if isinstance(eigenmode_config['boundary_conditions'], str):
            eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[eigenmode_config['boundary_conditions']]
    else:
        eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT['mm']

    def _run_ngsolve(cav, eigenmode_config):
        start_time = time.time()

        cav.create()
        ngsolve_mevp.solve(cav, eigenmode_config=eigenmode_config)

        # Run UQ if configured
        if 'uq_config' in eigenmode_config:
            uq_config = eigenmode_config['uq_config']

            uq_cell_complexity = uq_config.get('cell_complexity', 'simplecell')

            if uq_cell_complexity == 'multicell':
                uq_parallel_multicell(cav, eigenmode_config, 'eigenmode')
            else:
                uq_parallel(cav, eigenmode_config, 'eigenmode')

        done(f'Done with Cavity {cav.name}. Time: {time.time() - start_time}')

    for i, (key, cav) in enumerate(list(cavs_dict.items())):
        cav_path = Path(cav.self_dir)
        if cav_path.exists():
            if rerun:
                eigenmode_path = cav_path / "eigenmode"
                if eigenmode_path.exists():
                    shutil.rmtree(eigenmode_path)
                _run_ngsolve(cav, eigenmode_config)
            else:
                qois_path = cav_path / "eigenmode" / "qois.json"
                if qois_path.exists():
                    pass
                else:
                    if cav_path.exists():
                        shutil.rmtree(cav_path)
                    _run_ngsolve(cav, eigenmode_config)
        else:
            _run_ngsolve(cav, eigenmode_config)
