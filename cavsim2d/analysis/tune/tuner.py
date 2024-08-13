import copy
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
                     proc=0,  tune_variable='Req', tune_config=None):

        parentDir = SOFTWARE_DIRECTORY
        # tuner
        if tune_config is None:
            tune_config = {}

        abs_err_list, conv_dict = [], []
        pytune_ngsolve = PyTuneNGSolve()

        start = time.time()
        tuned_shape_space = {}
        total_no_of_shapes = len(pseudo_cavs_dict)

        # check for already processed shapes
        existing_keys = []

        # if resume == "Yes":
        #     # check if value set is already written. This is to enable continuation in case of break in program
        #     if os.path.exists(os.path.join(cav.self_dir, 'Cavities', filename)):
        #         tuned_shape_space = json.load(open(os.path.join(cav.self_dir, 'Cavities', filename), 'r'))
        #
        #         existing_keys = list(tuned_shape_space.keys())

        error_msg1 = 1
        error_msg2 = 1

        for key, cav in pseudo_cavs_dict.items():
            # A_i, B_i, a_i, b_i, Ri_i, L_i, Req = np.array(cav.shape['IC'])[:7]
            # A_o, B_o, a_o, b_o, Ri_o, L_o, Req_o = np.array(cav.shape['OC'])[:7]  # Req here is none but required

            target_freq = cav.shape['FREQ']

            # Check if simulation is already run
            freq = 0
            alpha_i = 0
            alpha_o = 0
            if resume == "Yes" and os.path.exists(os.path.join(cav.projectDir, key)):

                # alpha_i, error_msg1 = calculate_alpha(A_i, B_i, a_i, b_i, Ri_i, L_i, Req, 0)
                # alpha_o, error_msg2 = calculate_alpha(A_o, B_o, a_o, b_o, Ri_o, L_o, Req, 0)
                #
                # inner_cell = [A_i, B_i, a_i, b_i, Ri_i, L_i, Req, alpha_i]
                # outer_cell = [A_o, B_o, a_o, b_o, Ri_o, L_o, Req, alpha_o]
                pass
            else:
                # remove any existing folder to avoid copying the wrong retults
                if os.path.exists(fr"{cav.self_dir}/eigenmode/{key}"):
                    shutil.rmtree(fr"{cav.self_dir}/eigenmode/{key}")

                # inner_cell = [A_i, B_i, a_i, b_i, Ri_i, L_i, Req, 0]
                # outer_cell = [A_o, B_o, a_o, b_o, Ri_o, L_o, Req_o, 0]

                # edit to check for key later
                if key not in existing_keys:
                    try:
                        tune_var, freq, conv_dict, abs_err_list = pytune_ngsolve.tune(cav, tune_config=tune_config)
                    except FileNotFoundError:
                        tune_var, freq = 0, 0
                # tune_variable = tune_variable.split('_')[0]
                # if tune_var != 0 and freq != 0:
                #     if (cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell'
                #             or cell_type.lower() == 'mid_cell'):
                #         tuned_mid_cell = cav.shape['IC'][:7]
                #         tuned_mid_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                #         tuned_end_cell = cav.shape['OC'][:7]
                #         # enforce equator continuity
                #         tuned_end_cell[6] = tuned_mid_cell[6]
                #     elif (cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell'
                #           or cell_type.lower() == 'mid_end_cell'):
                #         tuned_mid_cell = cav.shape['IC'][:7]
                #         tuned_end_cell = cav.shape['OC'][:7]
                #         tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                #     elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
                #           or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
                #         tuned_mid_cell = cav.shape['IC'][:7]
                #         tuned_end_cell = cav.shape['OC'][:7]
                #         tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                #         tuned_mid_cell = copy.deepcopy(tuned_end_cell)
                #     elif cell_type.lower() == 'single_cell' or cell_type.lower() == 'single cell':
                #         tuned_mid_cell = cav.shape['IC'][:7]
                #         tuned_end_cell = cav.shape['OC'][:7]
                #         tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                #         tuned_mid_cell = copy.deepcopy(tuned_end_cell)
                #     else:
                #         info('Valid cell_type not selected. Defaulting to end_cell')
                #         tuned_mid_cell = cav.shape['IC'][:7]
                #         tuned_mid_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                #         tuned_end_cell = cav.shape['OC'][:7]
                #         # enforce equator continuity
                #         tuned_end_cell[6] = tuned_mid_cell[6]
                #
                #         # # enforce equator continuity
                #         # tuned_mid_cell[6] = tuned_end_cell[6]
                #
                #     alpha_i, error_msg1 = calculate_alpha(*tuned_mid_cell, 0)
                #     alpha_o, error_msg2 = calculate_alpha(*tuned_end_cell, 0)
                #
                #     # update cells with alpha
                #     tuned_mid_cell = np.append(tuned_mid_cell, alpha_i)
                #     tuned_end_cell = np.append(tuned_end_cell, alpha_o)
                #
                #     inner_cell = [*tuned_mid_cell, alpha_i]
                #     outer_cell = [*tuned_end_cell, alpha_o]

            # clear folder after every run. This is to avoid copying of wrong values to save folder
            # processor folder
            # print(tune_var, freq)
            # proc_fold = os.path.join(cav.self_dir, 'eigenmode', f'_process_{proc}')
            # shutil.rmtree(proc_fold)

            result = "Failed"
            d_tune_res = {}
            abs_err_dict = {}
            tol = 1e-4

            if 'tol' in tune_config.keys():
                tol = tune_config['tol']

            if tune_var != 0 and freq != 0:
                accuracy = abs(freq - target_freq)
                if accuracy <= tol:
                    result = f"Success: {target_freq, freq}"
                else:
                    result = f"Failed: Accuracy of {tol:.2e} could not reached. Accuracy of {accuracy:.2e} reached."

                tuned_shape_space[key] = {"parameter": cav.parameters, 'FREQ': freq}
                # write tune results
                d_tune_res = {'parameters': cav.parameters,
                              'TUNED VARIABLE': tune_variable, 'FREQ': freq}

                abs_err_dict = {'abs_err': abs_err_list}

            if result:
                done(f'Done Tuning Cavity {key}: {result}')
            else:
                error(f'Done Tuning Cavity {key}: {result}')

            return tuned_shape_space, d_tune_res, conv_dict, abs_err_dict

        end = time.time()

        runtime = end - start
        info(f'\tProcessor {proc} runtime: {runtime}s')

    def tune_ngsolve_multicell(self, pseudo_shape_space, bc, parentDir, projectDir, filename, resume="No",
                     proc=0, sim_folder='NGSolveMEVP', tune_variable='Req', cell_type='Mid Cell', tune_config=None):

        # tuner
        if tune_config is None:
            tune_config = {}
        target_freq = tune_config['freq']
        abs_err_list, conv_dict = [], []
        pytune_ngsolve = PyTuneNGSolve()

        tuned_multicell_shape_space = {}
        for key, multicell in pseudo_shape_space.items():

            if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{key}"):
                shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{key}")
            # print(multicell)
            tune_var_dict, freq_dict, conv_dict, abs_err_list = pytune_ngsolve.tune_multicell(multicell,
                                                                          tune_variable,
                                                                          target_freq, bc,
                                                                          sim_folder,
                                                                          parentDir, projectDir,
                                                                          proc=proc, tune_config=tune_config)
            # print('before tuning', multicell)
            # tuned_multicell = pytune_ngsolve.tune_multicell(multicell,
            #                                                 tune_variable,
            #                                                 target_freq, bc,
            #                                                 sim_folder,
            #                                                 parentDir, projectDir,
            #                                                 proc=proc, tune_config=tune_config)
            # print('after tuning', multicell)

            tuned_multicell_shape_space[key] = multicell  # tuning done in place

            # clear folder after every run. This is to avoid copying of wrong values to save folder
            # processor folder
            proc_fold = os.path.join(projectDir, 'SimulationData', f'{sim_folder}', f'_process_{proc}')
            shutil.rmtree(proc_fold)

            result = "Passed"
            d_tune_res = {}
            abs_err_dict = {}
            end_freq_list = []
            for key, freq_list in freq_dict.items():
                end_freq_list.append(freq_list[-1])
                if not (1 - 0.001) * target_freq < round(freq_list[-1], 2) < (1 + 0.001) * target_freq:
                    result = 'Failed'

            done(f'Done Tuning Cavity {key}: {result}:: {end_freq_list}')

        return tuned_multicell_shape_space, d_tune_res, conv_dict, abs_err_dict


    # if __name__ == '__main__':
#     #
#     tune = Tuner()
#
#     tune_var =
#     par_mid =
#     par_end =
#     target_freq = 400  # MHz
#     beampipes =
#     bc =  # boundary conditions
#     parentDir = ""  # location of slans code. See folder structure in the function above
#     projectDir = ""  # location to write results to
#     iter_set =
#     proc = 0
#     tune.tune(self, tune_var, par_mid, par_end, target_freq, beampipes, bc, parentDir, projectDir, iter_set, proc=0):
