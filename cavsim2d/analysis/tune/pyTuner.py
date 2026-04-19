import json
import os.path
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.stats import qmc

from cavsim2d.constants import SOFTWARE_DIRECTORY
from cavsim2d.utils.shared_functions import *
from cavsim2d.utils.printing import suppress_errors
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
        One of 'mid-cell', 'end-cell', 'single-cell'.
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
    elif ct == 'end cell':
        base = shape['IC'] if perturbed is None else shape['IC']
        end = shape['OC'] if perturbed is None else perturbed
        return base, base, end, 'right', shape['OC']
    elif ct == 'single cell':
        cell_node = shape['OC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'both', shape['OC']
    else:
        cell_node = shape['IC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'none', shape['IC']


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

        if cav.kind == 'elliptical cavity':
            # If tune_var is a bare name (no suffix), map it based on cell_types
            if '_m' not in self.tune_var and '_el' not in self.tune_var and '_er' not in self.tune_var:
                ct = tune_config.get('cell_types', 'mid-cell').lower().replace('-', ' ').replace('_', ' ')
                if ct in ('mid cell',):
                    self.tune_var = f'{self.tune_var}_m'
                elif ct in ('end cell r', 'end cell right'):
                    self.tune_var = f'{self.tune_var}_er'
                elif ct in ('end cell', 'end cell l', 'end cell left', 'single cell'):
                    self.tune_var = f'{self.tune_var}_el'
                else:
                    self.tune_var = f'{self.tune_var}_m'

            # Use suffix-based dispatch — substring matching breaks on names
            # like 'Req_er' (which contains '_e').
            if self.tune_var.endswith('_m'):
                self.beampipe = 'none'
            elif self.tune_var.endswith('_el'):
                self.beampipe = 'left'
            elif self.tune_var.endswith('_er'):
                self.beampipe = 'right'

        self.freq_list = []
        self.tv_list = []
        self.abs_err_list = []
        self.convergence_list = []

        x0 = cav.parameters[self.tune_var]
        self.x0_initial = x0  # Store for sanity check in tuner
        # Secant method needs two starting points
        x1 = x0 * 1.05  # 5% perturbation

        self._degen_count = 0
        # Secant probing routinely lands on parameter sets that yield
        # degenerate geometry — the tuner handles these via a penalty and
        # retreats. Silence the per-iteration "Parameter set leads to
        # degenerate geometry." error so only the final stage result is
        # reported. A terminal degeneracy still surfaces via the
        # ValueError branch below.
        try:
            with suppress_errors('Parameter set leads to degenerate geometry'):
                res = root_scalar(
                    self.tune_function,
                    method='secant',
                    x0=x0,
                    x1=x1,
                    xtol=tol,
                    rtol=tol,
                    maxiter=maxiter
                )
                # Final evaluation at converged root
                self.tune_function(res.root)
            converged = bool(res.converged) if hasattr(res, 'converged') else True
            root_val = res.root
        except ValueError as e:
            # Tuning aborted mid-iteration (e.g., repeated degeneracy).
            error(f'Tune aborted: {e}')
            converged = False
            # Best-effort: pick the tv with smallest |diff| from the valid points collected.
            root_val = self._best_valid_tv_or_zero()

        self.convergence_list.extend([self.tv_list, self.freq_list])
        self.conv_dict = {f'{self.tune_var}': self.convergence_list[0], 'freq [MHz]': self.convergence_list[1]}

        # Diagnose unreachable targets: if the final achievable freq is
        # far from the target, surface that clearly rather than returning
        # a spurious result.
        if self._valid_freqs():
            best_freq, best_diff = self._best_valid_result()
            freq_tol_mhz = max(1e-2, 1e-3 * self.target_freq)  # 0.001% or 10 kHz
            if abs(best_diff) > freq_tol_mhz:
                min_f = min(self._valid_freqs())
                max_f = max(self._valid_freqs())
                error(
                    f"Tune did not reach target {self.target_freq} MHz. "
                    f"Best |diff| = {abs(best_diff):.3f} MHz at {self.tune_var}={root_val}. "
                    f"Achievable freq range (sampled): [{min_f:.3f}, {max_f:.3f}] MHz. "
                    f"Target may be unreachable by varying {self.tune_var} alone — "
                    f"consider adjusting other geometry parameters (e.g., Req)."
                )
                return 0, 0, self.conv_dict, self.abs_err_list

        if converged:
            # Find closest match in recorded tv_list (float eq after re-eval may differ)
            if root_val in self.tv_list:
                idx = self.tv_list.index(root_val)
            else:
                idx = int(np.argmin(np.abs(np.array(self.tv_list) - root_val)))
            return root_val, self.freq_list[idx], self.conv_dict, self.abs_err_list
        else:
            return 0, 0, self.conv_dict, self.abs_err_list

    def _valid_freqs(self):
        """Recorded frequencies from steps that weren't flagged degenerate."""
        return [f for f, ok in zip(self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]

    def _best_valid_result(self):
        """Return (freq, diff) for the valid step closest to target."""
        valid = [(f, f - self.target_freq) for f, ok in
                 zip(self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]
        if not valid:
            return self.target_freq, 0.0
        return min(valid, key=lambda fd: abs(fd[1]))

    def _best_valid_tv_or_zero(self):
        """Return the tune variable value of the best valid step, or 0 if none."""
        valid_pairs = [(tv, f) for tv, f, ok in
                       zip(self.tv_list, self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]
        if not valid_pairs:
            return 0
        best = min(valid_pairs, key=lambda tf: abs(tf[1] - self.target_freq))
        return best[0]

    def tune_function(self, x):
        x_val = float(x if isinstance(x, (float, int, np.floating)) else x[0])
        self.cav.parameters[self.tune_var] = x_val
        self.tv_list.append(x_val)
        if not hasattr(self, '_valid_mask'):
            self._valid_mask = []

        bp = self.beampipe if self.beampipe else 'none'

        # End-cell stages tune the end-cell in its real context (beampipe +
        # end-cell + adjacent mid-cell half, PMC at both outer ends). The
        # quarter-cell geometry used for mid-cell tuning mirrors the end-cell
        # to itself, which excludes the pi-mode frequency the real multicell
        # cavity has — i.e. the target is unreachable.
        endcell_tune = bp in ('left', 'right')
        tune_mode = 'tune-endcell' if endcell_tune else 'tune'

        # Geometry creation and solve both can fail on degenerate parameters.
        # Rather than aborting the entire tune, record a penalty and let the
        # root finder retreat toward the valid region.
        degenerate = False
        try:
            self.cav.create(1, bp, mode=tune_mode)
        except Exception as e:
            degenerate = True

        if not degenerate:
            orig_n_cells = self.cav.n_cells
            self.cav.n_cells = 1
            try:
                res = ngsolve_mevp.solve(self.cav)
            except Exception:
                res = False
            finally:
                self.cav.n_cells = orig_n_cells
            if not res:
                degenerate = True

        if degenerate:
            return self._degenerate_penalty(x_val)

        if 'uq_config' in self.tune_config and self.tune_config['uq_config']:
            run_tune_uq(self.cav, self.tune_config)

            with open(os.path.join(self.cav.self_dir, 'eigenmode', 'uq.json')) as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            with open(os.path.join(self.cav.self_dir, 'eigenmode', 'qois.json')) as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']

        self.freq_list.append(freq)
        self._valid_mask.append(True)
        self._degen_count = 0

        diff = freq - self.target_freq
        self.abs_err_list.append(abs(diff))

        return diff

    def _degenerate_penalty(self, x_val):
        """Handle degenerate geometry step by returning a penalty that
        nudges the secant method back toward the valid region.

        Strategy: keep the sign of the last valid diff and inflate magnitude
        so the secant extrapolation retreats. After too many consecutive
        failures, raise to terminate tuning.
        """
        self._degen_count = getattr(self, '_degen_count', 0) + 1

        # Find the last valid freq (if any)
        last_valid_freq = None
        for f, ok in zip(reversed(self.freq_list), reversed(self._valid_mask)):
            if ok:
                last_valid_freq = f
                break

        if last_valid_freq is not None:
            last_diff = last_valid_freq - self.target_freq
            # Keep sign, amplify magnitude so secant retreats toward last valid x.
            sign = 1.0 if last_diff >= 0 else -1.0
            penalty = sign * max(abs(last_diff) * 10.0, 1e3)
            placeholder_freq = last_valid_freq
        else:
            # No valid anchor yet — default to a large positive penalty.
            penalty = 1e3
            placeholder_freq = self.target_freq + penalty

        # Record the failed step in all tracking lists to keep them aligned.
        self.freq_list.append(placeholder_freq)
        self._valid_mask.append(False)
        self.abs_err_list.append(abs(penalty))

        if self._degen_count >= 3:
            raise ValueError(
                f'Geometry degenerated {self._degen_count} consecutive times at '
                f'{self.tune_var}≈{x_val:.4g}. Target frequency may be unreachable.'
            )
        return penalty

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

            sim_path = os.path.join(projectDir, '_tune_temp', fid)
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
        with open(os.path.join(projectDir, '_tune_temp', fid, "convergence_output.json"), "w") as outfile:
            json.dump(dd, outfile, indent=4, separators=(',', ': '))


def run_tune_uq(cav, tune_config):
    """Run eigenmode UQ at the current tune iteration.

    Perturbs the requested variables around ``cav.parameters`` over a
    quadrature rule, solves eigenmode at each node, and writes the
    weighted mean/std/skew/kurtosis of ``freq [MHz]`` to
    ``<cav.self_dir>/eigenmode/uq.json``. The tune loop then reads the
    expected frequency from that file in place of a single deterministic
    solve.
    """
    uq_config = tune_config['uq_config']

    uq_cell_complexity = uq_config.get('cell_complexity',
                                       uq_config.get('cell complexity', 'simplecell'))
    if uq_cell_complexity == 'multicell':
        # Multicell UQ during tuning is not wired up in the current flow.
        warning('Multicell UQ during tuning is not supported; skipping UQ step.')
        return

    uq_vars = uq_config['variables']
    delta = uq_config.get('delta', [0.05] * len(uq_vars))
    method = uq_config.get('method', ['Quadrature', 'Stroud3'])

    # Resolve bare variable names (A, B, ...) to the suffixed parameters
    # that actually live on cav.parameters (A_m, A_el, ...). During a
    # multi-stage tune the per-stage config passes ``cell_types`` (singular
    # value) to pyTuner; UQ overrides via ``uq_config['cell_type']``.
    cell_type = uq_config.get('cell_type', tune_config.get('cell_types', 'mid-cell'))
    ct = cell_type.lower().replace('-', ' ').replace('_', ' ')
    if ct in ('end cell r', 'end cell right'):
        suffix = '_er'
    elif ct in ('end cell', 'end cell l', 'end cell left', 'single cell'):
        suffix = '_el'
    else:
        suffix = '_m'

    def _resolve(v):
        if v.endswith(('_m', '_el', '_er')):
            return v
        return f'{v}{suffix}'

    resolved_vars = [_resolve(v) for v in uq_vars]

    # Beampipe setting matching the suffix of the perturbed cell.
    bp_map = {'_m': 'none', '_el': 'left', '_er': 'right'}
    bp = bp_map.get(suffix, 'none')

    # Quadrature nodes / weights.
    rdim = len(resolved_vars)
    method_name = method[1].lower() if len(method) > 1 else 'stroud3'
    if method_name == 'stroud3':
        nodes_, weights_, _ = quad_stroud3(rdim, 1)
        nodes_ = 2. * nodes_ - 1.
    elif method_name == 'stroud5':
        nodes_, weights_ = cn_leg_05_2(rdim)
    elif method_name == 'gaussian':
        nodes_, weights_ = cn_gauss(rdim, 2)
    else:
        warning(f'UQ integration method {method!r} not recognised; '
                'defaulting to Stroud3.')
        nodes_, weights_, _ = quad_stroud3(rdim, 1)
        nodes_ = 2. * nodes_ - 1.

    n_sims = nodes_.shape[1]
    base_params = dict(cav.parameters)
    freqs = []

    eigenmode_dir = os.path.join(cav.self_dir, 'eigenmode')
    os.makedirs(eigenmode_dir, exist_ok=True)

    with suppress_errors('Parameter set leads to degenerate geometry'):
        for j in range(n_sims):
            # Perturb in-place; cav is restored after the loop.
            for i, rvar in enumerate(resolved_vars):
                cav.parameters[rvar] = base_params[rvar] * (1 + delta[i] * nodes_[i, j])

            ok = False
            orig_n_cells = cav.n_cells
            cav.n_cells = 1
            # End-cell UQ uses the hybrid (end-cell + adjacent mid-cell half)
            # geometry for the same reason the non-UQ tune path does.
            uq_tune_mode = 'tune-endcell' if bp in ('left', 'right') else 'tune'
            try:
                cav.create(1, bp, mode=uq_tune_mode)
                ok = bool(ngsolve_mevp.solve(cav))
            except Exception as e:
                info(f'UQ node {j} failed to solve ({e!r}); dropping from quadrature.')
                ok = False
            finally:
                cav.n_cells = orig_n_cells

            if ok:
                with open(os.path.join(eigenmode_dir, 'qois.json')) as f:
                    q = json.load(f)
                freqs.append(q['freq [MHz]'])
            else:
                freqs.append(np.nan)

    # Restore the base parameters so the outer tune loop continues from
    # the same state it entered with.
    cav.parameters.update(base_params)

    freqs_arr = np.array(freqs, dtype=float).reshape(-1, 1)
    if np.isnan(freqs_arr).all():
        raise ValueError('All UQ eigenmode solves failed — cannot aggregate.')
    valid_mask = ~np.isnan(freqs_arr[:, 0])
    valid_freqs = freqs_arr[valid_mask]
    valid_weights = np.asarray(weights_)[valid_mask]
    # Renormalise weights over surviving nodes.
    valid_weights = valid_weights / valid_weights.sum()

    mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(valid_freqs, valid_weights)

    result = {
        'freq [MHz]': {
            'expe': [float(mean_obj[0])],
            'stdDev': [float(std_obj[0])],
            'skew': [float(skew_obj[0])],
            'kurtosis': [float(kurtosis_obj[0])],
        }
    }
    with open(os.path.join(eigenmode_dir, 'uq.json'), 'w') as f:
        json.dump(result, f, indent=4)

    # Diagnostic dump: per-node freqs + nodes + weights for this tune iter.
    debug_rows = {
        'node': list(range(n_sims)),
        'freq [MHz]': [float(v) if np.isfinite(v) else None for v in freqs],
        'weight': [float(w) for w in np.asarray(weights_).ravel()],
    }
    for i, rv in enumerate(resolved_vars):
        debug_rows[rv] = [float(base_params[rv] * (1 + delta[i] * nodes_[i, j]))
                          for j in range(n_sims)]
    debug_rows['Req_m (base)'] = [float(base_params.get('Req_m', np.nan))] * n_sims
    debug_rows['mean freq [MHz]'] = [float(mean_obj[0])] * n_sims
    pd.DataFrame(debug_rows).to_csv(os.path.join(eigenmode_dir, 'uq_nodes.csv'),
                                    index=False, sep='\t', float_format='%.6f')


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
        uq_path = projectDir / f'{key}/eigenmode'
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
        elif ct == 'end cell':
            mid, left, right, beampipes = cell_node, cell_node, perturbed_cell_node, 'right'
        elif ct == 'single cell':
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'both'
        else:
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'none'

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
