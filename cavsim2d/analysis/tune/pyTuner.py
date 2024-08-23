import os.path
from itertools import groupby

from scipy.stats import qmc

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


class PyTuneNGSolve:
    def __init__(self):
        self.plot = None

    def tune(self, par_mid, par_end, tune_var, target_freq, cell_type, beampipes, bc,
             sim_folder, parentDir, projectDir, proc=0, tune_config=None):
        if tune_config is None:
            tune_config = {}

        convergence_list = []
        # tv => tune variable
        indx = VAR_TO_INDEX_DICT[tune_var]

        if proc == '':
            fid = '_process_0'
        else:
            fid = f'_process_{proc}'

        # make directory
        if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{fid}"):
            shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")
        os.mkdir(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")

        # get parameters
        freq_list = []
        tv_list = []
        abs_err_list = []
        err = 1

        if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
            tuned_cell = par_mid
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'none'
        elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
            mid = par_mid
            left = par_mid
            tuned_cell = par_end
            right = tuned_cell
            beampipes = 'right'
        elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
              or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
            tuned_cell = par_end
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'right'
        else:
            tuned_cell = par_end
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'both'

        res = ngsolve_mevp.cavity(1, 1, mid, left, right,
                                  n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                  sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)

        if not res:
            # make functionality later for restart for the tune variable
            error('\tCannot continue with tuning -> Skipping degenerate geometry')
            return 0, 0, [], []

        tv = tuned_cell[indx]

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if uq_config:
                shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell', 'BP': beampipes}
                run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

            # get uq results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            # get results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                eigenmode_qois = json.load(json_file)

            freq = eigenmode_qois['freq [MHz]']

        freq_list.append(freq)
        tv_list.append(tv)

        # first shot
        tv = tv + TUNE_VAR_STEP_DIRECTION_DICT[tune_var] * 0.05 * tuned_cell[indx]
        tuned_cell[indx] = tv

        enforce_Req_continuity(mid, left, right, cell_type)

        # run
        res = ngsolve_mevp.cavity(1, 1, mid, left, right,
                                  n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                  sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
        if not res:
            # make functionality later for restart for the tune variable
            error('Cannot continue with tuning -> Skipping degenerate geometry')
            return 0, 0, [], []

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if uq_config:
                shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell', 'BP': beampipes}
                run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

            # get uq results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            # get results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                eigenmode_qois = json.load(json_file)

            freq = eigenmode_qois['freq [MHz]']

        freq_list.append(freq)
        tv_list.append(tv)

        tol = 1e-2  # for testing purposes to reduce tuning time.
        # max_iter = iter_set[2]

        n = 1
        while abs(err) > tol and n < MAX_TUNE_ITERATION:
            # create matrix
            mat = [np.array(freq_list)[-2:], np.ones(2)]
            mat = np.array(mat).T

            # solve for coefficients
            coeffs = np.linalg.solve(mat, np.array(tv_list)[-2:])

            max_step = 0.2 * tuned_cell[indx]  # control order of convergence/stability with maximum step
            # bound_factor = 0.1
            # bound the maximum step
            if coeffs[0] * target_freq - (tv - coeffs[1]) > max_step:
                coeffs[1] = tv + max_step - coeffs[0] * target_freq
            if coeffs[0] * target_freq - (tv - coeffs[1]) < -max_step:
                coeffs[1] = tv - max_step - coeffs[0] * target_freq

            # define function
            func = lambda x: coeffs[0] * x + coeffs[1]

            # calculate approximate tv
            tv = func(target_freq)

            # change tv
            tuned_cell[indx] = tv
            enforce_Req_continuity(mid, left, right, cell_type)

            # run
            res = ngsolve_mevp.cavity(1, 1, mid, left, right,
                                      n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                      sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
            if not res:
                # make functionality later for restart for the tune variable
                error('Cannot continue with tuning -> Skipping degenerate geometry')
                return 0, 0, 0

            if 'uq_config' in tune_config.keys():
                uq_config = tune_config['uq_config']
                if uq_config:
                    shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell',
                             'BP': beampipes}
                    run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

                # get uq results and compare with set value
                with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                    eigenmode_qois = json.load(json_file)
                freq = eigenmode_qois['freq [MHz]']['expe'][0]
            else:
                # get results and compare with set value
                with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                    eigenmode_qois = json.load(json_file)

                freq = eigenmode_qois['freq [MHz]']

            freq_list.append(freq)
            tv_list.append(tv)

            # if equal, break else continue with new shape
            err = target_freq - freq_list[-1]
            abs_err_list.append(abs(err))

            if n == MAX_TUNE_ITERATION:
                info('Maximum number of iterations exceeded. No solution found.')
                break

            # condition for repeated last four values
            if self.all_equal(freq_list[-2:]):
                error("Converged. Solution found.")
                break

            if tv_list[-1] < 0:
                error("Negative value encountered. It is possible that there no solution for the parameter input set.")
                break

            n += 1

        # return best answer from iteration
        min_error = [abs(x - target_freq) for x in freq_list]
        key = min_error.index(min(min_error))

        # print(tv_list, freq_list)
        # import matplotlib.pyplot as plt
        # plt.scatter(tv_list, freq_list)
        # plt.show()
        # update convergence list
        convergence_list.extend([tv_list, freq_list])

        # save convergence information
        conv_dict = {f'{tune_var}': convergence_list[0], 'freq [MHz]': convergence_list[1]}
        return tv_list[key], freq_list[key], conv_dict, abs_err_list

    def tune_flattop(self, par_mid, par_end, tune_var, target_freq, cell_type, beampipes, bc,
                     sim_folder, parentDir, projectDir, proc=0, tune_config=None):
        if tune_config is None:
            tune_config = {}

        convergence_list = []
        # tv => tune variable
        indx = VAR_TO_INDEX_DICT[tune_var]

        if proc == '':
            fid = '_process_0'
        else:
            fid = f'_process_{proc}'

        # make directory
        if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{fid}"):
            shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")
        os.mkdir(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")

        # get parameters
        freq_list = []
        tv_list = []
        abs_err_list = []
        err = 1

        if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
            tuned_cell = par_mid
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'none'
        elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
            mid = par_mid
            left = par_mid
            tuned_cell = par_end
            right = tuned_cell
            beampipes = 'right'
        elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
              or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
            tuned_cell = par_end
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'right'
        else:
            tuned_cell = par_end
            mid = tuned_cell
            left = tuned_cell
            right = tuned_cell
            beampipes = 'both'

        res = ngsolve_mevp.cavity_flattop(1, 1, mid, left, right,
                                          n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                          sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)

        if not res:
            # make functionality later for restart for the tune variable
            error('\tCannot continue with tuning -> Skipping degenerate geometry')
            return 0, 0, [], []

        tv = tuned_cell[indx]

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if uq_config:
                shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell', 'BP': beampipes}
                run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

            # get uq results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            # get results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                eigenmode_qois = json.load(json_file)

            freq = eigenmode_qois['freq [MHz]']

        freq_list.append(freq)
        tv_list.append(tv)

        # first shot
        tv = tv + TUNE_VAR_STEP_DIRECTION_DICT[tune_var] * 0.05 * tuned_cell[indx]
        tuned_cell[indx] = tv

        enforce_Req_continuity(mid, left, right, cell_type)

        # run
        res = ngsolve_mevp.cavity_flattop(1, 1, mid, left, right,
                                          n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                          sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
        if not res:
            # make functionality later for restart for the tune variable
            error('Cannot continue with tuning -> Skipping degenerate geometry')
            return 0, 0, [], []

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if uq_config:
                shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell', 'BP': beampipes}
                run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

            # get uq results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            # get results and compare with set value
            with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                eigenmode_qois = json.load(json_file)

            freq = eigenmode_qois['freq [MHz]']

        freq_list.append(freq)
        tv_list.append(tv)

        tol = 1e-2  # for testing purposes to reduce tuning time.
        # max_iter = iter_set[2]

        n = 1
        while abs(err) > tol and n < MAX_TUNE_ITERATION:
            # create matrix
            mat = [np.array(freq_list)[-2:], np.ones(2)]
            mat = np.array(mat).T

            # solve for coefficients
            coeffs = np.linalg.solve(mat, np.array(tv_list)[-2:])

            max_step = 0.2 * tuned_cell[indx]  # control order of convergence/stability with maximum step
            # bound_factor = 0.1
            # bound the maximum step
            if coeffs[0] * target_freq - (tv - coeffs[1]) > max_step:
                coeffs[1] = tv + max_step - coeffs[0] * target_freq
            if coeffs[0] * target_freq - (tv - coeffs[1]) < -max_step:
                coeffs[1] = tv - max_step - coeffs[0] * target_freq

            # define function
            func = lambda x: coeffs[0] * x + coeffs[1]

            # calculate approximate tv
            tv = func(target_freq)

            # change tv
            tuned_cell[indx] = tv
            enforce_Req_continuity(mid, left, right, cell_type)

            # run
            res = ngsolve_mevp.cavity_flattop(1, 1, mid, left, right,
                                              n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
                                              sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
            if not res:
                # make functionality later for restart for the tune variable
                error('Cannot continue with tuning -> Skipping degenerate geometry')
                return 0, 0, 0

            if 'uq_config' in tune_config.keys():
                uq_config = tune_config['uq_config']
                if uq_config:
                    shape = {'IC': mid, 'OC': left, 'OC_R': right, 'n_cells': 1, 'CELL TYPE': 'simplecell',
                             'BP': beampipes}
                    run_tune_uq(fid, shape, tune_config, parentDir, projectDir)

                # get uq results and compare with set value
                with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/uq.json") as json_file:
                    eigenmode_qois = json.load(json_file)
                freq = eigenmode_qois['freq [MHz]']['expe'][0]
            else:
                # get results and compare with set value
                with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
                    eigenmode_qois = json.load(json_file)

                freq = eigenmode_qois['freq [MHz]']

            freq_list.append(freq)
            tv_list.append(tv)

            # if equal, break else continue with new shape
            err = target_freq - freq_list[-1]
            abs_err_list.append(abs(err))

            if n == MAX_TUNE_ITERATION:
                info('Maximum number of iterations exceeded. No solution found.')
                break

            # condition for repeated last four values
            if self.all_equal(freq_list[-2:]):
                error("Converged. Solution found.")
                break

            if tv_list[-1] < 0:
                error("Negative value encountered. It is possible that there no solution for the parameter input set.")
                break

            n += 1

        # return best answer from iteration
        min_error = [abs(x - target_freq) for x in freq_list]
        key = min_error.index(min(min_error))

        # print(tv_list, freq_list)
        # import matplotlib.pyplot as plt
        # plt.scatter(tv_list, freq_list)
        # plt.show()
        # update convergence list
        convergence_list.extend([tv_list, freq_list])

        # save convergence information
        conv_dict = {f'{tune_var}': convergence_list[0], 'freq [MHz]': convergence_list[1]}
        return tv_list[key], freq_list[key], conv_dict, abs_err_list

    # def tune_last_gultig(self, par_mid, par_end, tune_var, target_freq, cell_type, beampipes, bc,
    #          sim_folder, parentDir, projectDir, proc=0):
    #     convergence_list = []
    #     # tv => tune variable
    #     indx = VAR_TO_INDEX_DICT[tune_var]
    #
    #     if proc == '':
    #         fid = '_process_0'
    #     else:
    #         fid = f'_process_{proc}'
    #
    #     # make directory
    #     if os.path.exists(fr"{projectDir}/SimulationData/{sim_folder}/{fid}"):
    #         shutil.rmtree(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")
    #     os.mkdir(fr"{projectDir}/SimulationData/{sim_folder}/{fid}")
    #
    #     # get parameters
    #     freq_list = []
    #     tv_list = []
    #     abs_err_list = []
    #     err = 1
    #
    #     if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
    #         tuned_cell = par_mid
    #         mid = tuned_cell
    #         left = tuned_cell
    #         right = tuned_cell
    #         beampipes = 'none'
    #     elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
    #         mid = par_mid
    #         left = par_mid
    #         tuned_cell = par_end
    #         right = tuned_cell
    #         beampipes = 'right'
    #     elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
    #           or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
    #         tuned_cell = par_end
    #         mid = tuned_cell
    #         left = tuned_cell
    #         right = tuned_cell
    #         beampipes = 'right'
    #     else:
    #         tuned_cell = par_end
    #         mid = tuned_cell
    #         left = tuned_cell
    #         right = tuned_cell
    #         beampipes = 'both'
    #
    #     res = ngsolve_mevp.cavity(1, 1, mid, left, right,
    #                               n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
    #                               sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
    #     if not res:
    #         # make functionality later for restart for the tune variable
    #         error('\tCannot continue with tuning -> Skipping degenerate geometry')
    #         return 0, 0, [], []
    #
    #     tv = tuned_cell[indx]
    #
    #     with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
    #         eigenmode_qois = json.load(json_file)
    #
    #     freq = eigenmode_qois['freq [MHz]']
    #     freq_list.append(freq)
    #     tv_list.append(tv)
    #
    #     # first shot
    #     tv = tv + TUNE_VAR_STEP_DIRECTION_DICT[tune_var] * 0.05 * tuned_cell[indx]
    #     tuned_cell[indx] = tv
    #
    #     enforce_Req_continuity(mid, left, right, cell_type)
    #
    #     # run
    #     res = ngsolve_mevp.cavity(1, 1, mid, left, right,
    #                               n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
    #                               sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
    #     if not res:
    #         # make functionality later for restart for the tune variable
    #         error('Cannot continue with tuning -> Skipping degenerate geometry')
    #         return 0, 0, [], []
    #
    #     # get results and compare with set value
    #     with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
    #         eigenmode_qois = json.load(json_file)
    #
    #     freq = eigenmode_qois['freq [MHz]']
    #     freq_list.append(freq)
    #     tv_list.append(tv)
    #
    #     tol = 1e-2  # for testing purposes to reduce tuning time.
    #     # max_iter = iter_set[2]
    #
    #     n = 1
    #     while abs(err) > tol and n < MAX_TUNE_ITERATION:
    #         # create matrix
    #         mat = [np.array(freq_list)[-2:], np.ones(2)]
    #         mat = np.array(mat).T
    #
    #         # solve for coefficients
    #         coeffs = np.linalg.solve(mat, np.array(tv_list)[-2:])
    #
    #         max_step = 0.2 * tuned_cell[indx]  # control order of convergence/stability with maximum step
    #         # bound_factor = 0.1
    #         # bound the maximum step
    #         if coeffs[0] * target_freq - (tv - coeffs[1]) > max_step:
    #             coeffs[1] = tv + max_step - coeffs[0] * target_freq
    #         if coeffs[0] * target_freq - (tv - coeffs[1]) < -max_step:
    #             coeffs[1] = tv - max_step - coeffs[0] * target_freq
    #
    #         # define function
    #         func = lambda x: coeffs[0] * x + coeffs[1]
    #
    #         # calculate approximate tv
    #         tv = func(target_freq)
    #
    #         # change tv
    #         tuned_cell[indx] = tv
    #         enforce_Req_continuity(mid, left, right, cell_type)
    #
    #         # run
    #         res = ngsolve_mevp.cavity(1, 1, mid, left, right,
    #                                   n_modes=1, fid=fid, f_shift=0, bc=bc, beampipes=beampipes,
    #                                   sim_folder=sim_folder, parentDir=parentDir, projectDir=projectDir)
    #         if not res:
    #             # make functionality later for restart for the tune variable
    #             error('Cannot continue with tuning -> Skipping degenerate geometry')
    #             return 0, 0, 0
    #
    #         # get results and compare with set value
    #         with open(fr"{projectDir}/SimulationData/{sim_folder}/{fid}/monopole/qois.json") as json_file:
    #             eigenmode_qois = json.load(json_file)
    #
    #         freq = eigenmode_qois['freq [MHz]']
    #         freq_list.append(freq)
    #         tv_list.append(tv)
    #
    #         # if equal, break else continue with new shape
    #         err = target_freq - freq_list[-1]
    #         abs_err_list.append(abs(err))
    #
    #         if n == MAX_TUNE_ITERATION:
    #             info('Maximum number of iterations exceeded. No solution found.')
    #             break
    #
    #         # condition for repeated last four values
    #         if self.all_equal(freq_list[-2:]):
    #             error("Converged. Solution found.")
    #             break
    #
    #         if tv_list[-1] < 0:
    #             error("Negative value encountered. It is possible that there no solution for the parameter input set.")
    #             break
    #
    #         n += 1
    #
    #     # return best answer from iteration
    #     min_error = [abs(x - target_freq) for x in freq_list]
    #     key = min_error.index(min(min_error))
    #
    #     # print(tv_list, freq_list)
    #     # import matplotlib.pyplot as plt
    #     # plt.scatter(tv_list, freq_list)
    #     # plt.show()
    #     # update convergence list
    #     convergence_list.extend([tv_list, freq_list])
    #
    #     # save convergence information
    #     conv_dict = {f'{tune_var}': convergence_list[0], 'freq [MHz]': convergence_list[1]}
    #     return tv_list[key], freq_list[key], conv_dict, abs_err_list

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    @staticmethod
    def write_output(tv_list, freq_list, fid, projectDir):
        dd = {"tv": tv_list, "freq": freq_list}

        with open(fr"{projectDir}\SimulationData\SLANS_opt\{fid}\convergence_output.json", "w") as outfile:
            json.dump(dd, outfile, indent=4, separators=(',', ': '))


def run_tune_uq(name, shape, tune_config, parentDir, projectDir):
    uq_config = tune_config['uq_config']

    objectives = uq_config['objectives']
    solver_dict = {'ngsolvemevp': ngsolve_mevp}
    solver_args_dict = {'eigenmode': tune_config,
                        'n_cells': 1,
                        'n_modules': 1,
                        'parentDir': parentDir,
                        'projectDir': projectDir,
                        'analysis folder': 'Optimisation',
                        'cell_type': 'mid-cell',
                        'optimisation': True
                        }

    uq_cell_complexity = 'simplecell'
    if 'cell_complexity' in uq_config.keys():
        uq_cell_complexity = uq_config['cell_complexity']
    if uq_cell_complexity == 'multicell':
        # shape_space = {name: shape_multi}
        # shape_space_multi = {name: to_multicell(1, shape_space[name])}
        # uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict, uq_config)
        pass
    else:
        shape_space = {name: shape}
        uq_parallel_tuner(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')


def uq_parallel_tuner(shape_space, objectives, solver_dict, solver_args_dict,
                      solver):
    """

    Parameters
    ----------
    key: str | int
        Cavity geomery identifier
    shape: dict
        Dictionary containing geometric dimensions of cavity geometry
    qois: list
        Quantities of interest considered in uncertainty quantification
    n_cells: int
        Number of cavity cells
    n_modules: int
        Number of modules
    n_modes: int
        Number of eigenmodes to be calculated
    f_shift: float
        Since the eigenmode solver uses the power method, a shift can be provided
    bc: int
        Boundary conditions {1:inner contour, 2:Electric wall Et = 0, 3:Magnetic Wall En = 0, 4:Axis, 5:metal}
        bc=33 means `Magnetic Wall En = 0` boundary condition at both ends
    pol: int {Monopole, Dipole}
        Defines whether to calculate for monopole or dipole modes
    parentDir: str | path
        Parent directory
    projectDir: str|path
        Project directory

    Returns
    -------
    :param select_solver:

    """

    if solver == 'eigenmode':
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
            # n_cells = shape['IC'].shape[1] + 1
            uq_path = projectDir / fr'SimulationData\{analysis_folder}\{key}'
            result_dict_eigen, result_dict_abci = {}, {}
            eigen_obj_list, abci_obj_list = [], []

            for o in objectives:
                if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                         "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
                    result_dict_eigen[o] = {'expe': [], 'stdDev': []}
                    eigen_obj_list.append(o)

                # if o.split(' ')[0] in ['ZL', 'ZT', 'k_loss', 'k_kick']:
                #     result_dict_abci[o] = {'expe': [], 'stdDev': []}
                #     run_abci = True
                #     abci_obj_list.append(o)

            rdim = len(uq_vars)
            degree = 1
            flag_stroud = 'stroud3'

            if flag_stroud == 'stroud3':
                nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.
                # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this
                # gives a less accurate answer. the nodes are not the same as the custom function
            elif flag_stroud == 'stroud5':
                nodes_, weights_ = cn_leg_05_2(rdim)
            elif flag_stroud == 'cn_gauss':
                nodes_, weights_ = cn_gauss(rdim, 2)
            elif flag_stroud == 'lhc':
                sampler = qmc.LatinHypercube(d=rdim)
                _ = sampler.reset()
                nsamp = 2500
                sample = sampler.random(n=nsamp)
                # ic(qmc.discrepancy(sample))
                l_bounds = [-1, -1, -1, -1, -1, -1]
                u_bounds = [1, 1, 1, 1, 1, 1]
                sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

                nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
            else:
                # issue warning
                warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
                nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
                nodes_ = 2. * nodes_ - 1.

            # save nodes
            data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
            data_table.to_csv(fr'{projectDir}\SimulationData\{analysis_folder}\{key}\nodes.csv',
                              index=False, sep='\t', float_format='%.32f')

            if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
                cell_node = shape['IC']
            elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
                cell_node = shape['OC']
            elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
                  or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
                cell_node = shape['OC']
            else:
                cell_node = shape['OC']

            no_parm, no_sims = np.shape(nodes_)
            if delta is None:
                delta = [0.05 for _ in range(len(uq_vars))]

            sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

            proc_count = 1
            if 'processes' in uq_config.keys():
                assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
                assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
                proc_count = uq_config['processes']
            if proc_count > no_sims:
                proc_count = no_sims

            share = round(no_sims / proc_count)
            jobs = []
            for p in range(proc_count):
                # try:
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
                processor_weights = weights_[proc_keys_list]
                service = mp.Process(target=uq_tuner, args=(key, objectives, uq_config, uq_path,
                                                            solver_args_dict, sub_dir,
                                                            proc_keys_list, processor_nodes, p, cell_node, solver))

                service.start()
                jobs.append(service)

            for job in jobs:
                job.join()

            # combine results from processes
            qois_result_dict = {}
            Ttab_val_f = []
            keys = []
            for i1 in range(proc_count):
                if i1 == 0:
                    df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
                else:
                    df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])

            df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
            df.to_excel(uq_path / 'table.xlsx', index=False)

            Ttab_val_f = df.to_numpy()
            v_expe_fobj, v_stdDev_fobj = weighted_mean_obj(Ttab_val_f, weights_)

            # append results to dict
            for i, o in enumerate(eigen_obj_list):
                result_dict_eigen[o]['expe'].append(v_expe_fobj[i])
                result_dict_eigen[o]['stdDev'].append(v_stdDev_fobj[i])

            with open(uq_path / fr'uq.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))


def uq_tuner(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
             proc_keys_list, processor_nodes, proc_num, cell_node, solver):
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
    :param n_cells:

    """

    if solver == 'eigenmode':
        parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        opt = solver_args_dict['optimisation']
        delta = uq_config['delta']
        method = uq_config['method']
        uq_vars = uq_config['variables']
        err = False
        result_dict_eigen = {}
        Ttab_val_f = []

        # eigen_obj_list = objectives
        eigen_obj_list = []

        for o in objectives:
            if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                     "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
                result_dict_eigen[o] = {'expe': [], 'stdDev': []}
                eigen_obj_list.append(o)

        # for o in objectives:
        #     result_dict_eigen[o] = {'expe': [], 'stdDev': []}

        perturbed_cell_node = np.array(cell_node)
        for i1, proc_key in enumerate(proc_keys_list):
            skip = False
            for j, uq_var in enumerate(uq_vars):
                uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
                perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (1 + delta[j] * processor_nodes[j, i1])

            if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
                # cell_node = shape['IC']
                mid = perturbed_cell_node
                left = perturbed_cell_node
                right = perturbed_cell_node
                beampipes = 'none'
            elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
                mid = cell_node
                left = cell_node
                right = perturbed_cell_node
                beampipes = 'right'
            elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
                  or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
                mid = perturbed_cell_node
                left = perturbed_cell_node
                right = perturbed_cell_node
                beampipes = 'right'
            else:
                mid = perturbed_cell_node
                left = perturbed_cell_node
                right = perturbed_cell_node
                beampipes = 'both'

            enforce_Req_continuity(mid, left, right, cell_type)

            # perform checks on geometry
            fid = fr'{key}_Q{proc_key}'

            ngsolve_mevp.createFolder(fid, projectDir, subdir=sub_dir, opt=opt)
            # it does not seem to make sense to perform uq on a multi cell by repeating the same perturbation
            # to all multi cells at once. For multicells, the uq_multicell option is more suitable as it creates
            # independent perturbations to all cells individually
            ngsolve_mevp.cavity(1, 1, mid, left, right, f_shift=0, bc=33, beampipes=beampipes,
                                fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
                                projectDir=projectDir,
                                subdir=sub_dir)

            filename = uq_path / f'{fid}/monopole/qois.json'
            if os.path.exists(filename):
                qois_result_dict = dict()

                with open(filename) as json_file:
                    qois_result_dict.update(json.load(json_file))
                qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)

                tab_val_f = qois_result
                Ttab_val_f.append(tab_val_f)
            else:
                err = True

        data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
        data_table.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')
