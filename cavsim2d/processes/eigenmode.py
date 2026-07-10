"""Parallel eigenmode analysis process functions."""
import multiprocessing as mp
import os.path
import shutil
import time
from pathlib import Path

from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP, parse_polarisations
from cavsim2d.solvers.eigenmode_result import pol_name
from cavsim2d.constants import *
from cavsim2d.processes.uq import uq_parallel
from cavsim2d.utils.shared_functions import *
from cavsim2d.utils.config_validation import require

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
        require(isinstance(eigenmode_config['rerun'], bool), 'rerun must be boolean.')
        rerun = eigenmode_config['rerun']

    processes = 1
    if 'processes' in eigenmode_config:
        require(eigenmode_config['processes'] > 0, 'Number of processes must be greater than zero.')
        require(isinstance(eigenmode_config['processes'], int), 'Number of processes must be integer.')
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

        # Run UQ if configured. With uq_config['cell_complexity'] = 'multicell',
        # every half-cell becomes an independent random variable (subject to the
        # equator/iris continuity constraints); uq_parallel branches on it.
        if 'uq_config' in eigenmode_config:
            uq_parallel(cav, eigenmode_config, 'eigenmode')

        done(f'Done with Cavity {cav.name}. Time: {time.time() - start_time}')

    # legacy flat-layout monopole artefacts (pre-``monopole/`` subfolder)
    legacy_monopole_files = ('qois.json', 'qois_all_modes.json', 'gfu_EH.pkl',
                             'mesh.pkl', 'Ez_0_abs.csv')

    for i, (key, cav) in enumerate(list(cavs_dict.items())):
        cav_path = Path(cav.self_dir)
        eigenmode_path = cav_path / "eigenmode"
        pols = parse_polarisations(eigenmode_config.get('polarisation', 0))
        if cav_path.exists():
            # Guard against silently serving cached results that no longer
            # match the in-memory geometry. ``rerun=True`` will regenerate
            # the geometry below; ``rerun=False`` will only warn.
            cav._check_geometry_mismatch('eigenmode')
            if rerun:
                # Only clear the polarisations being rerun, so e.g. a dipole
                # rerun never wipes existing monopole results (and vice versa).
                for m in pols:
                    pol_path = eigenmode_path / pol_name(m)
                    if pol_path.exists():
                        shutil.rmtree(pol_path)
                if 0 in pols:
                    for legacy in legacy_monopole_files:
                        legacy_path = eigenmode_path / legacy
                        if legacy_path.exists():
                            legacy_path.unlink()
                    # UQ results are computed from the monopole solve, so drop
                    # stale UQ artefacts on a monopole rerun — otherwise the old
                    # uq.json (and perturbed-cavity sims) outlive the eigenmode
                    # results they were derived from and could be served as
                    # current (P2-6). A fresh uq_config regenerates them.
                    for uq_file in ('uq.json', 'uq_all_modes.json'):
                        uq_path = eigenmode_path / uq_file
                        if uq_path.exists():
                            uq_path.unlink()
                    uq_dir = cav_path / 'uq'
                    if uq_dir.exists():
                        shutil.rmtree(uq_dir)
                _run_ngsolve(cav, eigenmode_config)
            else:
                missing = [m for m in pols
                           if not (eigenmode_path / pol_name(m) / 'qois.json').exists()]
                # legacy flat monopole results still count as present
                if 0 in missing and (eigenmode_path / 'qois.json').exists():
                    missing.remove(0)
                if missing:
                    config_missing = dict(eigenmode_config)
                    config_missing['polarisation'] = missing
                    _run_ngsolve(cav, config_missing)
        else:
            _run_ngsolve(cav, eigenmode_config)
