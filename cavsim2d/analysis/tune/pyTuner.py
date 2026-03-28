import json
import os.path
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.stats import qmc

from cavsim2d.constants import SOFTWARE_DIRECTORY
from cavsim2d.utils.shared_functions import *
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
import shutil
import multiprocessing as mp

ngsolve_mevp = NGSolveMEVP()
file_color = 'cyan'

DEBUG = True


def print_(*arg):
    if DEBUG:
        print(colored(f'\t\t\t{arg}', file_color))


VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6, 'l': 7}
TUNE_VAR_STEP_DIRECTION_DICT = {'A': -1, 'B': 1, 'a': -1, 'b': 1, 'Ri': 1, 'L': 1, 'Req': -1, 'l': 1}
MAX_TUNE_ITERATION = 10

EIGENMODE_OBJECTIVES = {"Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]",
                         "R/Q [Ohm]", "G [Ohm]", "Q []", "kcc [%]", "ff [%]"}


def _resolve_cell_type(cell_type, shape, perturbed=None):
    """Map cell_type string to (mid, left, right, beampipes) tuple.

    Parameters
    ----------
    cell_type : str
        One of 'mid cell', 'mid-end cell', 'end-end cell', or other.
    shape : dict
        Shape dictionary with 'IC' and 'OC' keys.
    perturbed : np.ndarray, optional
        If provided, used as the perturbed cell node (for UQ).
        If None, returns the base cell node from shape.

    Returns
    -------
    tuple : (mid, left, right, beampipes, cell_node)
    """
    ct = cell_type.lower().replace('-', ' ').replace('_', ' ')

    if ct == 'mid cell':
        cell_node = shape['IC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'none', shape['IC']
    elif ct == 'mid end cell':
        base = shape['IC'] if perturbed is None else shape['IC']
        end = shape['OC'] if perturbed is None else perturbed
        return base, base, end, 'right', shape['OC']
    elif ct == 'end end cell':
        cell_node = shape['OC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'right', shape['OC']
    else:
        cell_node = shape['OC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'both', shape['OC']


class PyTuneNGSolve:
    def __init__(self):
        self.plot = None
        self.beampipe = None

    def tune(self, cav, tune_config=None):
        self.cav = cav
        self.tune_config = tune_config
        self.target_freq = self.cav.shape['FREQ']
        self.tune_var = self.tune_config['parameters']

        tol = tune_config.get('tol', 1e-4)
        maxiter = tune_config.get('maxiter', 10)
        method = tune_config.get('method', 'newton')

        if cav.kind == 'elliptical cavity':
            if '_m' in self.tune_var:
                self.beampipe = 'none'
            if '_el' in self.tune_var:
                self.beampipe = 'left'
            if '_er' in self.tune_var:
                self.beampipe = 'right'

        self.freq_list = []
        self.tv_list = []
        self.abs_err_list = []
        self.convergence_list = []

        res = root_scalar(
            self.tune_function,
            method=method,
            x0=cav.parameters[self.tune_var],
            xtol=tol,
            rtol=tol,
            maxiter=maxiter
        )

        self.tune_function(res.root)

        self.convergence_list.extend([self.tv_list, self.freq_list])
        self.conv_dict = {f'{self.tune_var}': self.convergence_list[0], 'freq [MHz]': self.convergence_list[1]}

        if res:
            return res.root, self.freq_list[self.tv_list.index(res.root)], self.conv_dict, self.abs_err_list
        else:
            return 0, 0, self.conv_dict, self.abs_err_list

    def tune_function(self, x):
        x_val = x if isinstance(x, float) else x[0]
        self.cav.parameters[self.tune_var] = x_val
        self.tv_list.append(x_val)

        bp = self.beampipe if self.beampipe else 'none'
        self.cav.create(1, bp, mode='tune')

        res = ngsolve_mevp.solve(self.cav)
        if not res:
            error('Cannot continue with tuning -> Skipping degenerate geometry')
            return 0, 0, [], []

        if 'uq_config' in self.tune_config:
            uq_config = self.tune_config['uq_config']
            if uq_config:
                run_tune_uq(self.cav, self.tune_config)

            with open(os.path.join(self.cav.self_dir, 'eigenmode', 'uq.json')) as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            with open(os.path.join(self.cav.self_dir, 'eigenmode', 'monopole', 'qois.json')) as json_file:
                eigenmode_qois = json.load(json_file)

        freq = eigenmode_qois['freq [MHz]']
        self.freq_list.append(freq)

        diff = abs(freq - self.target_freq)
        self.abs_err_list.append(diff)

        return diff

    def tune_multicell(self, multicell, tune_var, target_freq, bc,
                       sim_folder, parentDir, projectDir, proc=0, tune_config=None):
        if tune_config is None:
            tune_config = {}

        n_half_cells = len(multicell) // 8
        convergence_dict = {}
        tv_dict, freq_dict = {}, {}

        for ii in range(n_half_cells):
            cell = multicell[ii * 8:(ii + 1) * 8]
            bp = ii == 0 or ii == n_half_cells - 1

            convergence_list = []
            indx = VAR_TO_INDEX_DICT['L']
            fid = f'_process_{proc}' if proc != '' else '_process_0'

            sim_path = os.path.join(projectDir, 'Cavities', '_tune_temp', fid)
            if os.path.exists(sim_path):
                shutil.rmtree(sim_path)
            os.mkdir(sim_path)

            freq_list = []
            tv_list = []
            abs_err_list = []
            err = 1

            res = ngsolve_mevp.cavity_quarter(cell, bp=False, fid=fid, f_shift=0, bc=bc,
                                              sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
            if not res:
                error('\tCannot continue with tuning -> Skipping degenerate geometry')
                return 0, 0, [], []

            tv = cell[indx]

            with open(os.path.join(sim_path, 'monopole', 'qois.json')) as json_file:
                eigenmode_qois = json.load(json_file)
                freq = eigenmode_qois['freq [MHz]']

            freq_list.append(freq)
            tv_list.append(tv)

            # Initial perturbation
            tv = tv + TUNE_VAR_STEP_DIRECTION_DICT[tune_var] * 0.05 * cell[indx]
            cell[indx] = tv

            res = ngsolve_mevp.cavity_quarter(cell, bp, fid=fid, f_shift=0, bc=bc,
                                              sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
            if not res:
                error('Cannot continue with tuning -> Skipping degenerate geometry')
                return 0, 0, [], []

            with open(os.path.join(sim_path, 'monopole', 'qois.json')) as json_file:
                eigenmode_qois = json.load(json_file)
                freq = eigenmode_qois['freq [MHz]']

            freq_list.append(freq)
            tv_list.append(tv)

            tol = tune_config.get('tolerance', 1e-2)

            n = 1
            while abs(err) > tol and n < MAX_TUNE_ITERATION:
                # Linear interpolation to find next tune variable value
                mat = np.array([np.array(freq_list)[-2:], np.ones(2)]).T
                coeffs = np.linalg.solve(mat, np.array(tv_list)[-2:])

                max_step = 0.2 * cell[indx]
                step = coeffs[0] * target_freq - (tv - coeffs[1])
                if step > max_step:
                    coeffs[1] = tv + max_step - coeffs[0] * target_freq
                if step < -max_step:
                    coeffs[1] = tv - max_step - coeffs[0] * target_freq

                tv = coeffs[0] * target_freq + coeffs[1]
                cell[indx] = tv

                res = ngsolve_mevp.cavity_quarter(cell, bp, fid=fid, f_shift=0, bc=bc,
                                                  sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
                if not res:
                    error('Cannot continue with tuning -> Skipping degenerate geometry')
                    return 0, 0, 0

                with open(os.path.join(sim_path, 'monopole', 'qois.json')) as json_file:
                    eigenmode_qois = json.load(json_file)
                    freq = eigenmode_qois['freq [MHz]']

                freq_list.append(freq)
                tv_list.append(tv)

                err = target_freq - freq_list[-1]
                abs_err_list.append(abs(err))

                if n == MAX_TUNE_ITERATION:
                    info('Maximum number of iterations exceeded. No solution found.')
                    break

                if self.all_equal(freq_list[-2:]):
                    error("Converged. Solution found.")
                    break

                if tv_list[-1] < 0:
                    error("Negative value encountered. It is possible that there no solution for the parameter input set.")
                    break

                n += 1

            min_error = [abs(x - target_freq) for x in freq_list]
            key = min_error.index(min(min_error))

            convergence_list.extend([tv_list, freq_list])
            tv_dict[ii] = tv_list
            freq_dict[ii] = freq_list

            conv_dict = {f'{tune_var}': convergence_list[0], 'freq [MHz]': convergence_list[1]}
            convergence_dict[f'halfcell {ii}'] = convergence_dict

        return tv_dict, freq_dict, conv_dict, abs_err_list

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    @staticmethod
    def write_output(tv_list, freq_list, fid, projectDir):
        dd = {"tv": tv_list, "freq": freq_list}
        with open(os.path.join(projectDir, 'Cavities', '_tune_temp', fid, "convergence_output.json"), "w") as outfile:
            json.dump(dd, outfile, indent=4, separators=(',', ': '))


def run_tune_uq(cav, tune_config):
    uq_config = tune_config['uq_config']

    objectives = uq_config['objectives']
    solver_dict = {'ngsolvemevp': ngsolve_mevp}
    solver_args_dict = {'eigenmode': tune_config,
                        'n_cells': 1,
                        'n_modules': 1,
                        'analysis folder': 'Optimisation',
                        'cell_type': 'mid-cell',
                        'optimisation': True
                        }

    uq_cell_complexity = uq_config.get('cell_complexity', 'simplecell')
    if uq_cell_complexity == 'multicell':
        pass
    else:
        shape_space = {name: shape}
        uq_parallel_tuner(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')


def uq_parallel_tuner(shape_space, objectives, solver_dict, solver_args_dict, solver):
    """Run UQ in parallel across quadrature points for tuning.

    Parameters
    ----------
    shape_space : dict
        Cavity geometry parameter space.
    objectives : list
        Quantities of interest for UQ.
    solver_dict : dict
        Solver instances.
    solver_args_dict : dict
        Solver configuration arguments.
    solver : str
        Solver type identifier.
    """
    if solver != 'eigenmode':
        return

    parentDir = solver_args_dict['parentDir']
    projectDir = solver_args_dict['projectDir']
    uq_config = solver_args_dict['eigenmode']['uq_config']
    cell_type = uq_config['cell_type']
    analysis_folder = solver_args_dict['analysis folder']
    opt = solver_args_dict['optimisation']
    delta = uq_config['delta']
    method = uq_config['method']
    uq_vars = uq_config['variables']
    assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

    for key, shape in shape_space.items():
        uq_path = projectDir / f'Cavities/{key}/eigenmode'
        result_dict_eigen = {}
        eigen_obj_list = []

        for o in objectives:
            if o in EIGENMODE_OBJECTIVES:
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                eigen_obj_list.append(o)

        rdim = len(uq_vars)
        degree = 1

        # Compute quadrature nodes and weights
        flag_stroud = 'stroud3'
        if flag_stroud == 'stroud3':
            nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
            nodes_ = 2. * nodes_ - 1.
        elif flag_stroud == 'stroud5':
            nodes_, weights_ = cn_leg_05_2(rdim)
        elif flag_stroud == 'cn_gauss':
            nodes_, weights_ = cn_gauss(rdim, 2)
        elif flag_stroud == 'lhc':
            sampler = qmc.LatinHypercube(d=rdim)
            _ = sampler.reset()
            nsamp = 2500
            sample = sampler.random(n=nsamp)
            l_bounds = [-1] * rdim
            u_bounds = [1] * rdim
            sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
            nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
        else:
            warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
            nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
            nodes_ = 2. * nodes_ - 1.

        data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
        data_table.to_csv(uq_path / 'nodes.csv', index=False, sep='\t', float_format='%.32f')

        _, _, _, _, cell_node = _resolve_cell_type(cell_type, shape)

        no_parm, no_sims = np.shape(nodes_)
        if delta is None:
            delta = [0.05] * len(uq_vars)

        sub_dir = fr'{key}'
        proc_count = uq_config.get('processes', 1)
        if proc_count > 0:
            assert isinstance(proc_count, int), error('Number of processes must be integer')
        else:
            error('Number of processes must be greater than zero')
            proc_count = 1
        proc_count = min(proc_count, no_sims)

        share = round(no_sims / proc_count)
        jobs = []
        for p in range(proc_count):
            end_already = False
            if p != proc_count - 1:
                if (p + 1) * share < no_sims:
                    proc_keys_list = np.arange(p * share, p * share + share)
                else:
                    proc_keys_list = np.arange(p * share, no_sims)
                    end_already = True

            if p == proc_count - 1 and not end_already:
                proc_keys_list = np.arange(p * share, no_sims)

            processor_nodes = nodes_[:, proc_keys_list]
            service = mp.Process(target=uq_tuner, args=(key, objectives, uq_config, uq_path,
                                                        solver_args_dict, sub_dir,
                                                        proc_keys_list, processor_nodes, p, cell_node, solver))
            service.start()
            jobs.append(service)

        for job in jobs:
            job.join()

        # Combine results from processes
        df = pd.read_csv(uq_path / 'table_0.csv', sep='\t', engine='python')
        for i1 in range(1, proc_count):
            df = pd.concat([df, pd.read_csv(uq_path / f'table_{i1}.csv', sep='\t', engine='python')])

        df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
        df.to_excel(uq_path / 'table.xlsx', index=False)

        Ttab_val_f = df.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

        for i, o in enumerate(eigen_obj_list):
            result_dict_eigen[o]['expe'].append(mean_obj[i])
            result_dict_eigen[o]['stdDev'].append(std_obj[i])
            result_dict_eigen[o]['skew'].append(skew_obj[i])
            result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])

        with open(uq_path / 'uq.json', 'w') as file:
            file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))


def uq_tuner(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
             proc_keys_list, processor_nodes, proc_num, cell_node, solver):
    """Run UQ eigenmode solves for a subset of quadrature points.

    Parameters
    ----------
    key : str
        Cavity geometry identifier.
    objectives : list
        QOIs for UQ.
    uq_config : dict
        UQ configuration.
    uq_path : Path
        Output directory.
    solver_args_dict : dict
        Solver arguments.
    sub_dir : str
        Subdirectory for results.
    proc_keys_list : np.ndarray
        Indices of quadrature points for this process.
    processor_nodes : np.ndarray
        Quadrature nodes for this process.
    proc_num : int
        Process number.
    cell_node : np.ndarray
        Base cell parameters.
    solver : str
        Solver type.
    """
    if solver != 'eigenmode':
        return

    parentDir = solver_args_dict['parentDir']
    projectDir = solver_args_dict['projectDir']
    cell_type = uq_config['cell_type']
    analysis_folder = solver_args_dict['analysis folder']
    opt = solver_args_dict['optimisation']
    delta = uq_config['delta']
    uq_vars = uq_config['variables']

    eigen_obj_list = [o for o in objectives if o in EIGENMODE_OBJECTIVES]
    Ttab_val_f = []

    perturbed_cell_node = np.array(cell_node)
    for i1, proc_key in enumerate(proc_keys_list):
        for j, uq_var in enumerate(uq_vars):
            uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
            perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (1 + delta[j] * processor_nodes[j, i1])

        ct = cell_type.lower().replace('-', ' ').replace('_', ' ')
        if ct == 'mid cell':
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'none'
        elif ct == 'mid end cell':
            mid, left, right, beampipes = cell_node, cell_node, perturbed_cell_node, 'right'
        elif ct == 'end end cell':
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'right'
        else:
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'both'

        enforce_Req_continuity(mid, left, right, cell_type)

        fid = fr'{key}_Q{proc_key}'
        ngsolve_mevp.createFolder(fid, projectDir, subdir=sub_dir, opt=opt)
        ngsolve_mevp.cavity(1, 1, mid, left, right, f_shift=0, bc=33, beampipes=beampipes,
                            fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
                            projectDir=projectDir, subdir=sub_dir)

        filename = uq_path / f'{fid}/monopole/qois.json'
        if os.path.exists(filename):
            with open(filename) as json_file:
                qois_result_dict = json.load(json_file)
            qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)
            Ttab_val_f.append(qois_result)

    data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
    data_table.to_csv(uq_path / f'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')
