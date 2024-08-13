import copy
import time
from cavsim2d.analysis.tune.pyTuner import PyTuneNGSolve
from cavsim2d.utils.shared_functions import *
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
import shutil

ngsolve_mevp = NGSolveMEVP()

VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6}


class Tuner:
    def __init__(self):
        pass

    def tune_ngsolve(self, pseudo_shape_space, bc, parentDir, projectDir, filename, resume="No",
                     proc=0, sim_folder='NGSolveMEVP', tune_variable='Req', iter_set=None, cell_type='Mid Cell',
                     progress_list=None, convergence_list=None, save_last=True, n_cell_last_run=1):

        # tuner
        pytune_ngsolve = PyTuneNGSolve()

        start = time.time()
        population = {}
        total_no_of_shapes = len(list(pseudo_shape_space.keys()))

        # check for already processed shapes
        existing_keys = []

        if resume == "Yes":
            # check if value set is already written. This is to enable continuation in case of break in program
            if os.path.exists(os.path.join(projectDir, 'Cavities', filename)):
                population = json.load(open(os.path.join(projectDir, 'Cavities', filename), 'r'))

                existing_keys = list(population.keys())

        progress = 0
        error_msg1 = 1
        error_msg2 = 1

        for key, pseudo_shape in pseudo_shape_space.items():
            A_i, B_i, a_i, b_i, Ri_i, L_i, Req = pseudo_shape['IC'][:7]
            A_o, B_o, a_o, b_o, Ri_o, L_o, Req_o = pseudo_shape['OC'][:7]  # Req here is none but required

            beampipes = pseudo_shape['BP']
            target_freq = pseudo_shape['FREQ']

            # Check if simulation is already run
            freq = 0
            alpha_i = 0
            alpha_o = 0
            if resume == "Yes" and os.path.exists(os.path.join(projectDir, 'SimulationData', f'{sim_folder}', key)):

                alpha_i, error_msg1 = calculate_alpha(A_i, B_i, a_i, b_i, Ri_i, L_i, Req, 0)
                alpha_o, error_msg2 = calculate_alpha(A_o, B_o, a_o, b_o, Ri_o, L_o, Req, 0)

                inner_cell = [A_i, B_i, a_i, b_i, Ri_i, L_i, Req, alpha_i]
                outer_cell = [A_o, B_o, a_o, b_o, Ri_o, L_o, Req, alpha_o]
            else:
                # remove any existing folder to avoid copying the wrong retults
                if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{key}"):
                    shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{key}")

                inner_cell = [A_i, B_i, a_i, b_i, Ri_i, L_i, Req, 0]
                outer_cell = [A_o, B_o, a_o, b_o, Ri_o, L_o, Req_o, 0]

                # edit to check for key later
                if key not in existing_keys:
                    try:
                        tune_var, freq, abs_err_list = pytune_ngsolve.tune(inner_cell, outer_cell, tune_variable,
                                                                           target_freq,
                                                                           cell_type, beampipes, bc, sim_folder,
                                                                           parentDir, projectDir, iter_set=iter_set,
                                                                           proc=proc,
                                                                           conv_list=convergence_list)
                    except FileNotFoundError:
                        tune_var, freq = 0, 0
                if tune_var != 0 and freq != 0:
                    if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
                        tuned_mid_cell = pseudo_shape['IC'][:7]
                        tuned_mid_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                        tuned_end_cell = pseudo_shape['OC'][:7]
                        # enforce equator continuity
                        tuned_end_cell[6] = tuned_mid_cell[6]
                    elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
                        tuned_mid_cell = pseudo_shape['IC'][:7]
                        tuned_end_cell = pseudo_shape['OC'][:7]
                        tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                    elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
                          or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
                        tuned_mid_cell = pseudo_shape['IC'][:7]
                        tuned_end_cell = pseudo_shape['OC'][:7]
                        tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                        tuned_mid_cell = copy.deepcopy(tuned_end_cell)
                    else:
                        tuned_mid_cell = pseudo_shape['IC'][:7]
                        tuned_end_cell = pseudo_shape['OC'][:7]
                        tuned_end_cell[VAR_TO_INDEX_DICT[tune_variable]] = tune_var
                        tuned_mid_cell = copy.deepcopy(tuned_end_cell)

                        # # enforce equator continuity
                        # tuned_mid_cell[6] = tuned_end_cell[6]

                    alpha_i, error_msg1 = calculate_alpha(*tuned_mid_cell, 0)
                    alpha_o, error_msg2 = calculate_alpha(*tuned_end_cell, 0)

                    # update cells with alpha
                    tuned_mid_cell = np.append(tuned_mid_cell, alpha_i)
                    tuned_end_cell = np.append(tuned_end_cell, alpha_o)

                    inner_cell = [*tuned_mid_cell, alpha_i]
                    outer_cell = [*tuned_end_cell, alpha_o]

            result = "Failed"
            if tune_var != 0 and freq != 0:
                if (1 - 0.001) * target_freq < round(freq, 2) < (1 + 0.001) * target_freq \
                        and (90.0 <= alpha_i <= 180) \
                        and (90.0 <= alpha_o <= 180) and error_msg1 == 1 and error_msg2 == 1:
                    result = f"Success: {target_freq, freq}"

                    population[key] = {"IC": inner_cell, "OC": outer_cell, "BP": 'both', 'FREQ': freq}

                    # save last slans run if activated. Save only when condition is fulfilled
                    if save_last:
                        beampipes = 'both'
                        if 'mid' in cell_type.lower():
                            beampipes = 'none'

                        # make directory
                        if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{key}"):
                            shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{key}")

                        os.mkdir(fr"{projectDir}/SimulationData/{sim_folder}/{key}")

                        ngsolve_mevp.cavity(n_cell_last_run, 1, tuned_mid_cell, tuned_end_cell, tuned_end_cell,
                                            f_shift=0, bc=bc,
                                            beampipes=beampipes, n_modes=n_cell_last_run + 1, fid=key,
                                            sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)

                        # write cst_studio parameters
                        write_cst_paramters(key, tuned_mid_cell, tuned_end_cell, tuned_end_cell, projectDir, cell_type,
                                            solver='Optimisation')

                        # write tune results
                        d_tune_res = {'IC': list(tuned_mid_cell), 'OC': list(tuned_end_cell),
                                      'OC_R': list(tuned_end_cell),
                                      'TUNED VARIABLE': tune_variable, 'CELL TYPE': cell_type, 'FREQ': freq}
                        self.save_tune_result(d_tune_res, 'tune_res.json', projectDir, key, sim_folder)

                        # save convergence information
                        if convergence_list:
                            conv_dict = {f'{tune_variable}': convergence_list[0], 'freq [MHz]': convergence_list[1]}
                            abs_err_dict = {'abs_err': abs_err_list}
                            self.save_tune_result(conv_dict, 'convergence.json', projectDir, key, sim_folder)
                            self.save_tune_result(abs_err_dict, 'absolute_error.json', projectDir, key, sim_folder)

            done(f'Done Tuning Cavity {key}: {result}')

            # clear folder after every run. This is to avoid copying of wrong values to save folder
            # processor folder
            proc_fold = os.path.join(projectDir, 'SimulationData', f'{sim_folder}', f'_process_{proc}')
            keep = ['SLANS_exe']
            for item in os.listdir(proc_fold):
                if item not in keep:  # If it isn't in the list for retaining
                    try:
                        os.remove(item)
                    except FileNotFoundError:
                        continue

            # update progress
            progress_list.append((progress + 1) / total_no_of_shapes)

            # Update progressbar
            progress += 1

            # print("Saving Dictionary", f"shape_space{proc}.json")◙
            # print("Done saving")

        end = time.time()

        runtime = end - start
        info(f'\tProcessor {proc} runtime: {runtime} s')

    @staticmethod
    def save_tune_result(d, filename, projectDir, key, sim_folder='SLAN_Opt'):
        with open(fr"{projectDir}\SimulationData\{sim_folder}\{key}\{filename}", 'w') as file:
            file.write(json.dumps(d, indent=4, separators=(',', ': ')))

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
