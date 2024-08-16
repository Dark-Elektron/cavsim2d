import os.path
from itertools import groupby
from cavsim2d.utils.shared_functions import *
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
import shutil

ngsolve_mevp = NGSolveMEVP()
file_color = 'cyan'

DEBUG = True


def print_(*arg):
    if DEBUG:
        print(colored(f'\t\t\t{arg}', file_color))


VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6}
TUNE_VAR_STEP_DIRECTION_DICT = {'A': -1, 'B': 1, 'a': -1, 'b': 1, 'Ri': 1, 'L': 1, 'Req': -1}
MAX_TUNE_ITERATION = 10


class PyTuneNGSolve:
    def __init__(self):
        self.plot = None

    def tune(self, par_mid, par_end, tune_var, target_freq, cell_type, beampipes, bc,
             sim_folder, parentDir, projectDir, proc=0):
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
            error('\tCannot continue with the tuning geometry -> Skipping degenerate geometry')
            return 0, 0, [], []

        tv = tuned_cell[indx]

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
            error('Cannot continue with the tuning geometry -> Skipping degenerate geometry')
            return 0, 0, [], []

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
                error('Cannot continue with the tuning of this geometry -> Skipping degenerate geometry')
                return 0, 0, 0

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

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    @staticmethod
    def write_output(tv_list, freq_list, fid, projectDir):
        dd = {"tv": tv_list, "freq": freq_list}

        with open(fr"{projectDir}\SimulationData\SLANS_opt\{fid}\convergence_output.json", "w") as outfile:
            json.dump(dd, outfile, indent=4, separators=(',', ': '))
