import os
import time

from cavsim2d.analysis.tune.pyTuner import PyTuneNGSolve
from cavsim2d.utils.shared_functions import *
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
import shutil
import json
from cavsim2d.constants import *

ngsolve_mevp = NGSolveMEVP()

VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6, 'l': 7}


class Tuner:
    def __init__(self):
        pass

    def tune_ngsolve(self, pseudo_cavs_dict, bc, resume="No",
                     proc=0, tune_variable='Req', tune_config=None):

        if tune_config is None:
            tune_config = {}

        pytune_ngsolve = PyTuneNGSolve()

        start = time.time()
        tuned_shape_space = {}
        all_tune_res = {}
        all_conv = {}
        all_abs_err = {}
        existing_keys = []

        tol = tune_config.get('tol', 1e-4)

        for key, cav in pseudo_cavs_dict.items():
            target_freq = cav.shape['FREQ']
            freq = 0
            tune_var = 0
            abs_err_list = []
            conv_dict = []

            if resume == "Yes" and os.path.exists(os.path.join(cav.self_dir, key)):
                # Attempt to load previous tuning result
                prev_result_path = os.path.join(cav.self_dir, key, 'tune_res.json')
                if os.path.exists(prev_result_path):
                    try:
                        with open(prev_result_path, 'r') as f:
                            prev_result = json.load(f)

                        # Determine the cell-type key: check new keyed format first,
                        # then fall back to flat (legacy) format.
                        ct_label = tune_config.get('cell_types', '')
                        if ct_label and ct_label in prev_result:
                            prev_data = prev_result[ct_label]
                        elif 'parameters' in prev_result:
                            prev_data = prev_result
                        else:
                            # Pick the first key if structure is unrecognised
                            first_key = next(iter(prev_result))
                            prev_data = prev_result[first_key]

                        freq = prev_data.get('FREQ', 0)
                        tune_var = prev_data.get('TUNED VARIABLE', tune_variable)

                        if freq != 0:
                            accuracy = abs(freq - target_freq)
                            if accuracy <= tol:
                                # Valid previous result — use it
                                tuned_shape_space[key] = {
                                    "parameter": prev_data.get('parameters', {}),
                                    'FREQ': freq
                                }
                                all_tune_res[key] = prev_data
                                all_conv[key] = prev_data.get('convergence', [])
                                all_abs_err[key] = prev_data.get('abs_err', [])

                                done(f'Resumed previous tuning result for {key}: '
                                     f'freq={freq}, var={tune_var}')
                                continue
                            else:
                                info(f'Previous result for {key} did not meet tolerance '
                                     f'({accuracy:.2e} > {tol:.2e}). Re-tuning.')
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        warning(f'Could not load previous result for {key}: {e}. Re-tuning.')
                else:
                    info(f'No previous result file found for {key}. Tuning from scratch.')

            # Clean up old eigenmode directory if present
            eigenmode_dir = os.path.join(cav.self_dir, 'eigenmode', key)
            if os.path.exists(eigenmode_dir):
                shutil.rmtree(eigenmode_dir)

            if key not in existing_keys:
                try:
                    tune_var, freq, conv_dict, abs_err_list = pytune_ngsolve.tune(
                        cav, tune_config=tune_config
                    )
                except (FileNotFoundError, ValueError) as e:
                    error(f'Tuning failed for {key}: {e}')
                    tune_var, freq = 0, 0

                # Sanity check: reject wild tune values from secant divergence
                if tune_var != 0:
                    x0_initial = getattr(pytune_ngsolve, 'x0_initial', None)
                    if x0_initial is not None and x0_initial != 0:
                        if abs(tune_var) > 10 * abs(x0_initial) or tune_var < 0:
                            error(f'Tuned value {tune_var:.4e} is out of bounds '
                                  f'(original: {x0_initial:.4e}). Treating as failed.')
                            tune_var, freq = 0, 0

            # Build result for this cavity
            d_tune_res = {}
            abs_err_dict = {}

            if tune_var != 0 and freq != 0:
                accuracy = abs(freq - target_freq)
                if accuracy <= tol:
                    result = f"Success: {target_freq, freq}"
                else:
                    result = (f"Failed: Accuracy of {tol:.2e} could not be reached. "
                              f"Accuracy of {accuracy:.2e} reached.")

                tuned_shape_space[key] = {"parameter": dict(cav.parameters), 'FREQ': freq}

                resolved_tune_var = getattr(pytune_ngsolve, 'tune_var', tune_variable)
                d_tune_res = {
                    'parameters': dict(cav.parameters),
                    'TUNED VARIABLE': resolved_tune_var,
                    'FREQ': freq
                }
                abs_err_dict = {'abs_err': abs_err_list}
            else:
                result = "Failed"

            # Log with stage info
            ct_label = tune_config.get('cell_types', '') or ''
            var_label = getattr(pytune_ngsolve, 'tune_var', tune_variable)
            stage_tag = f'[{ct_label}: {var_label}]' if ct_label else f'[{var_label}]'

            if "Success" in result:
                done(f'Done Tuning Cavity {key} {stage_tag}: {result}')
            else:
                error(f'Done Tuning Cavity {key} {stage_tag}: {result}')

            # Accumulate results for all cavities
            all_tune_res[key] = d_tune_res
            all_conv[key] = conv_dict
            all_abs_err[key] = abs_err_dict

        end = time.time()
        runtime = end - start
        info(f'\tProcessor {proc} runtime: {runtime}s')

        return tuned_shape_space, all_tune_res, all_conv, all_abs_err

    def tune_ngsolve_multicell(self, pseudo_shape_space, bc, parentDir, projectDir, filename, resume="No",
                               proc=0, sim_folder='NGSolveMEVP', tune_variable='Req', cell_type='Mid Cell',
                               tune_config=None):

        if tune_config is None:
            tune_config = {}
        target_freq = tune_config['freq']
        abs_err_list, conv_dict = [], []
        pytune_ngsolve = PyTuneNGSolve()

        tuned_multicell_shape_space = {}
        for key, multicell in pseudo_shape_space.items():

            if os.path.exists(os.path.join(projectDir, key, "eigenmode")):
                shutil.rmtree(os.path.join(projectDir, key, "eigenmode"))

            tune_var_dict, freq_dict, conv_dict, abs_err_list = pytune_ngsolve.tune_multicell(
                multicell, tune_variable, target_freq, bc,
                sim_folder, parentDir, projectDir,
                proc=proc, tune_config=tune_config)

            tuned_multicell_shape_space[key] = multicell  # tuning done in place

            # Clear processor folder to avoid stale results
            proc_fold = os.path.join(projectDir, '_tune_temp', f'_process_{proc}')
            shutil.rmtree(proc_fold)

            result = "Passed"
            d_tune_res = {}
            abs_err_dict = {}
            end_freq_list = []
            for fkey, freq_list in freq_dict.items():
                end_freq_list.append(freq_list[-1])
                if not (1 - 0.001) * target_freq < round(freq_list[-1], 2) < (1 + 0.001) * target_freq:
                    result = 'Failed'

            done(f'Done Tuning Cavity {key}: {result}:: {end_freq_list}')

        return tuned_multicell_shape_space, d_tune_res, conv_dict, abs_err_dict
