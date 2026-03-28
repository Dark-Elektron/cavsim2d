import os
import time

from cavsim2d.analysis.tune.pyTuner import PyTuneNGSolve
from cavsim2d.utils.shared_functions import *
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
import shutil
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

        abs_err_list, conv_dict = [], []
        pytune_ngsolve = PyTuneNGSolve()

        start = time.time()
        tuned_shape_space = {}
        existing_keys = []

        tol = tune_config.get('tol', 1e-4)

        for key, cav in pseudo_cavs_dict.items():
            target_freq = cav.shape['FREQ']
            freq = 0
            tune_var = 0

            if resume == "Yes" and os.path.exists(os.path.join(cav.projectDir, key)):
                pass
            else:
                if os.path.exists(fr"{cav.self_dir}/eigenmode/{key}"):
                    shutil.rmtree(fr"{cav.self_dir}/eigenmode/{key}")

                if key not in existing_keys:
                    try:
                        tune_var, freq, conv_dict, abs_err_list = pytune_ngsolve.tune(cav, tune_config=tune_config)
                    except FileNotFoundError:
                        tune_var, freq = 0, 0

            result = "Failed"
            d_tune_res = {}
            abs_err_dict = {}

            if tune_var != 0 and freq != 0:
                accuracy = abs(freq - target_freq)
                if accuracy <= tol:
                    result = f"Success: {target_freq, freq}"
                else:
                    result = f"Failed: Accuracy of {tol:.2e} could not reached. Accuracy of {accuracy:.2e} reached."

                tuned_shape_space[key] = {"parameter": cav.parameters, 'FREQ': freq}
                d_tune_res = {'parameters': cav.parameters,
                              'TUNED VARIABLE': tune_variable, 'FREQ': freq}
                abs_err_dict = {'abs_err': abs_err_list}

            if result:
                done(f'Done Tuning Cavity {key}: {result}')
            else:
                error(f'Done Tuning Cavity {key}: {result}')

        end = time.time()
        runtime = end - start
        info(f'\tProcessor {proc} runtime: {runtime}s')

        return tuned_shape_space, d_tune_res, conv_dict, abs_err_dict

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

            if os.path.exists(os.path.join(projectDir, "Cavities", key, "eigenmode")):
                shutil.rmtree(os.path.join(projectDir, "Cavities", key, "eigenmode"))

            tune_var_dict, freq_dict, conv_dict, abs_err_list = pytune_ngsolve.tune_multicell(
                multicell, tune_variable, target_freq, bc,
                sim_folder, parentDir, projectDir,
                proc=proc, tune_config=tune_config)

            tuned_multicell_shape_space[key] = multicell  # tuning done in place

            # Clear processor folder to avoid stale results
            proc_fold = os.path.join(projectDir, 'Cavities', '_tune_temp', f'_process_{proc}')
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
