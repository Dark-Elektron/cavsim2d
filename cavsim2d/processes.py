import multiprocessing as mp
import os.path
import shutil
import time

from cavsim2d.analysis.tune.tuner import Tuner
from cavsim2d.analysis.wakefield.abci_geometry import ABCIGeometry
from cavsim2d.solvers.ABCI.abci import ABCI
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

abci = ABCI()
ngsolve_mevp = NGSolveMEVP()
abci_geom = ABCIGeometry()
tuner = Tuner()

def run_tune_parallel(cavs_dict, tune_config, solver='NGSolveMEVP',
                      resume=False):

    tune_config_keys = tune_config.keys()
    if 'processes' in tune_config_keys:
        processes = tune_config['processes']
        assert processes > 0, error('Number of proceses must be greater than zero.')
    else:
        processes = 1

    assert 'freqs' in tune_config_keys, error('Please enter the target tune "freqs" in tune_config.')
    assert 'parameters' in tune_config_keys, error('Please enter the tune "parameters"  in tune_config')
    assert 'cell_types' in tune_config_keys, error('Please enter the "cell_types" in tune_config')
    freqs = tune_config['freqs']
    tune_parameters = tune_config['parameters']
    cell_types = tune_config['cell_types']

    if isinstance(freqs, float) or isinstance(freqs, int):
        freqs = np.array([freqs for _ in range(len(cavs_dict))])
    else:
        assert len(freqs) == len(cavs_dict), error(
            'Number of target frequencies must correspond to the number of cavities')
        freqs = np.array(freqs)

    if isinstance(tune_parameters, str):
        for key, cav in cavs_dict.items():
            assert tune_config['parameters'] in cav.parameters.keys(), error(
                'Please enter a valid tune parameter. \n\tUse `Cavity.parameters to see valid parameters' )
        tune_parameters = np.array([tune_parameters for _ in range(len(cavs_dict))])
    else:
        assert len(tune_parameters) == len(cavs_dict), error(
            'Number of tune parameters must correspond to the number of cavities')
        assert len(cell_types) == len(cavs_dict), error(
            'Number of cell types must correspond to the number of cavities')
        tune_parameters = np.array(tune_parameters)

    if isinstance(cell_types, str):
        cell_types = np.array([cell_types for _ in range(len(cavs_dict))])
    else:
        assert len(cell_types) == len(cavs_dict), error(
            'Number of cell types must correspond to the number of cavities')
        cell_types = np.array(cell_types)

    # split shape_space for different processes/ MPI share process by rank
    keys = list(cavs_dict.keys())

    # check if number of processors selected is greater than the number of keys in the pseudo shape space
    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    # share = int(round(shape_space_len / processes))
    jobs = []

    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0
    for p in range(processes):
        # Determine the size of the current chunk
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        proc_tune_variables = tune_parameters[start_idx:start_idx + current_chunk_size]
        proc_freqs = freqs[start_idx:start_idx + current_chunk_size]

        proc_cell_types = cell_types[start_idx:start_idx + current_chunk_size]

        start_idx += current_chunk_size

        # modify tune config
        proc_tune_config = {}
        for key, val in tune_config.items():
            if key == 'freqs':
                proc_tune_config[key] = proc_freqs
            if key == 'parameters':
                proc_tune_config[key] = proc_tune_variables

        processor_cavs_dict = {key: cavs_dict[key] for key in proc_keys_list}
        service = mp.Process(target=run_tune_s, args=(processor_cavs_dict, proc_tune_config, p))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_tune_s(processor_cavs_dict, tune_config, p):
    proc_tune_variables = tune_config['parameters']
    proc_freqs = tune_config['freqs']

    # perform necessary checks
    if tune_config is None:
        tune_config = {}
    tune_config_keys = tune_config.keys()

    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    def _run_tune():
        cav_tune_config = tune_config.copy()
        cav_tune_config['freqs'] = tune_config['freqs'][i]
        cav_tune_config['parameters'] = tune_config['parameters'][i]

        tuned_shape_space, d_tune_res, conv_dict, abs_err_dict = tuner.tune_ngsolve({key: cav}, 33,
                                                                                    proc=p,
                                                                                    tune_variable=proc_tune_variables[i],
                                                                                    tune_config=cav_tune_config)

        if d_tune_res:
            n_cells = processor_cavs_dict[key].n_cells
            tuned_shape_space[key]['n_cells'] = n_cells
            tuned_shape_space[key]['CELL PARAMETERISATION'] = processor_cavs_dict[key].cell_parameterisation

            eigenmode_config = {'target': run_eigenmode_s}
            if 'eigenmode_config' in tune_config_keys:
                eigenmode_config = tune_config['eigenmode_config']
                eigenmode_config['target']= run_eigenmode_s
            else:
                info('tune_config does not contain eigenmode_config. Default values are used for eigenmode analysis.')

            run_parallel({key: cav}, eigenmode_config)

            # save tune results
            save_tune_result(d_tune_res, cav.self_dir, 'tune_res.json')

            # save convergence information
            save_tune_result(conv_dict, cav.self_dir, 'tune_convergence.json',)
            save_tune_result(abs_err_dict, cav.self_dir, 'tune_absolute_error.json')

    for i, (key, cav) in enumerate(processor_cavs_dict.items()):
        cav.shape['FREQ'] = proc_freqs[i]
        if os.path.exists(os.path.join(cav.eigenmode_dir, key)):
            if rerun:
                # clear previous results
                shutil.rmtree(os.path.join(cav.eigenmode_dir, 'monopole'))
                os.mkdir(os.path.join(cav.eigenmode_dir, 'monopole'))
                _run_tune()
        else:
            _run_tune()


def run_tune_s_multicell(processor_shape_space, proc_tune_variables, tune_config, projectDir, resume,
                         p, sim_folder='Optimisation'):
    # perform necessary checks
    if tune_config is None:
        tune_config = {}
    tune_config_keys = tune_config.keys()

    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    def _run_tune(key, shape):
        tuned_shape_space, d_tune_res, conv_dict, abs_err_dict = tuner.tune_ngsolve_multicell({key: shape}, 33,
                                                                                              SOFTWARE_DIRECTORY,
                                                                                              projectDir, key,
                                                                                              resume=resume, proc=p,
                                                                                              tune_variable=
                                                                                              proc_tune_variables[i],
                                                                                              sim_folder=sim_folder,
                                                                                              tune_config=tune_config)

        # n_cells = processor_shape_space[key]['n_cells']
        # tuned_shape_space[key]['n_cells'] = n_cells
        # tuned_shape_space[key]['CELL PARAMETERISATION'] = processor_shape_space[key]['CELL PARAMETERISATION']

        return tuned_shape_space[key]

    processor_shape_space_tuned = {}
    for i, (key, shape) in enumerate(processor_shape_space.items()):
        # shape['FREQ'] = proc_freqs[i]

        if os.path.exists(os.path.join(projectDir, "SimulationData", sim_folder, key)):
            # if rerun:
            # clear previous results
            shutil.rmtree(os.path.join(projectDir, "SimulationData", sim_folder, key))
            os.mkdir(os.path.join(projectDir, "SimulationData", sim_folder, key))
            tuned_shape = _run_tune(key, shape)
        else:
            tuned_shape = _run_tune(key, shape)

        processor_shape_space_tuned[key] = tuned_shape

    # print('processor_shape_space_tuned', processor_shape_space_tuned)
    return processor_shape_space_tuned


def run_parallel(cavs_dict, solver_config, subdir=''):

    if 'processes' in solver_config.keys():
        processes = solver_config['processes']
        assert processes > 0, error('Number of proceses must be greater than zero.')
    else:
        processes = 1

    # split shape_space for different processes/ MPI share process by rank
    keys = list(cavs_dict.keys())

    # check if number of processors selected is greater than the number of keys in the pseudo shape space
    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    share = int(round(shape_space_len / processes))

    jobs = []
    # for p in range(processes):
    #     if p < processes - 1:
    #         proc_keys_list = keys[p * share:p * share + share]
    #     else:
    #         proc_keys_list = keys[p * share:]
    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0

    for p in range(processes):
        # Determine the size of the current chunk
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        start_idx += current_chunk_size

        processor_cavs_dict = {key: cavs_dict[key] for key in proc_keys_list}
        # processor_shape_space_multi = {key: shape_space_multi[key] for key in proc_keys_list}
        service = mp.Process(target=solver_config['target'], args=(processor_cavs_dict, solver_config, subdir))

        service.start()
        jobs.append(service)

    for job in jobs:
        job.join()


def run_eigenmode_s(cavs_dict, eigenmode_config, subdir):
    """
    Run eigenmode analysis

    Parameters
    ----------
    shape_space
    shape_space_multi
    projectDir
    eigenmode_config

    Returns
    -------

    """

    rerun = True
    if 'rerun' in eigenmode_config.keys():
        assert isinstance(eigenmode_config['rerun'], bool), error('rerun must be boolean.')
        rerun = eigenmode_config['rerun']

    # perform all necessary checks
    processes = 1
    if 'processes' in eigenmode_config.keys():
        assert eigenmode_config['processes'] > 0, error('Number of proceses must be greater than zero.')
        assert isinstance(eigenmode_config['processes'], int), error('Number of proceses must be integer.')
    else:
        eigenmode_config['processes'] = processes

    freq_shifts = 0
    if 'f_shifts' in eigenmode_config.keys():
        freq_shifts = eigenmode_config['f_shifts']

    boundary_conds = 'mm'
    if 'boundary_conditions' in eigenmode_config.keys():
        eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[eigenmode_config['boundary_conditions']]
    else:
        eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[boundary_conds]

    # uq_config = None
    # if 'uq_config' in eigenmode_config.keys():
    #     uq_config = eigenmode_config['uq_config']
    #     if uq_config:
    #         assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
    #                                                                              "be equal to the number of "
    #                                                                              "variables.")
    # else:
    #     eigenmode_config['uq_config'] = uq_config

    def _run_ngsolve(cav, eigenmode_config):

        start_time = time.time()

        # write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
        #                     projectDir=projectDir, cell_type="None",
        #                     solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)
        # create cavity
        cav.create()
        ngsolve_mevp.solve(cav, eigenmode_config=eigenmode_config)

        #
        # if not shape['geo_file']:
        #     if shape['kind'] == 'elliptical cavity':
        #         n_cells = shape['n_cells']
        #         n_modules = 1
        #         bc = eigenmode_config['boundary_conditions']
        #         # create folders for all keys
        #         # print(projectDir)
        #         # ngsolve_mevp.createFolder(name, projectDir, subdir=subdir, opt=eigenmode_config['opt'])
        #
        #         if shape['CELL PARAMETERISATION'] == 'flattop':
        #             write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
        #                                 projectDir=projectDir, cell_type="None",
        #                                 solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)
        #
        #             ngsolve_mevp.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
        #                                         n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc,
        #                                         beampipes=shape['BP'], sim_folder=solver_save_dir,
        #                                         parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
        #                                         eigenmode_config=eigenmode_config)
        #
        #         elif shape['CELL PARAMETERISATION'] == 'multicell':
        #             write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
        #                                 projectDir=projectDir, cell_type="None",
        #                                 solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)
        #             ngsolve_mevp.cavity_multicell(n_cells, n_modules, shape_multi,
        #                                           n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc,
        #                                           beampipes=shape['BP'], sim_folder=solver_save_dir,
        #                                           parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
        #                                           eigenmode_config=eigenmode_config)
        #         else:
        #             write_cst_paramters(f"{name}", shape['IC'], shape['OC'], shape['OC_R'],
        #                                 projectDir=projectDir, cell_type="None",
        #                                 solver=eigenmode_config['solver_save_directory'], sub_dir=subdir)
        #
        #             ngsolve_mevp.cavity(n_cells, n_modules, shape,
        #                                 n_modes=n_cells, fid=f"{name}", f_shift=0, bc=bc, beampipes=shape['BP'],
        #                                 sim_folder=solver_save_dir,
        #                                 parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
        #                                 eigenmode_config=eigenmode_config)
        #     elif shape['kind'] == 'vhf gun':
        #         ngsolve_mevp.createFolder(name, projectDir, subdir=subdir, opt=eigenmode_config['opt'])
        #         ngsolve_mevp.vhf_gun(fid=f"{name}", shape=shape, sim_folder=solver_save_dir,
        #                              parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
        #                              eigenmode_config=eigenmode_config)
        #
        #     elif shape['kind'] == 'pillbox cavity':
        #         pass
        #     else:
        #         pass
        # else:
        #     bc = eigenmode_config['boundary_conditions']
        #     ngsolve_mevp.createFolder(name, projectDir, subdir=subdir, opt=eigenmode_config['opt'])
        #     ngsolve_mevp.cavity(1, 1, shape,
        #                         n_modes=1, fid=f"{name}", f_shift=0, bc=bc, beampipes='mm',
        #                         sim_folder=solver_save_dir,
        #                         parentDir=SOFTWARE_DIRECTORY, projectDir=projectDir, subdir=subdir,
        #                         eigenmode_config=eigenmode_config)
        #
        #
        # run UQ
        if 'uq_config' in eigenmode_config.keys():
            uq_config = eigenmode_config['uq_config']

            uq_cell_complexity = 'simplecell'
            if 'cell_complexity' in uq_config.keys():
                uq_cell_complexity = uq_config['cell_complexity']

            if uq_cell_complexity == 'multicell':
                # shape_space = {name: shape_multi}
                uq_parallel_multicell(cav, eigenmode_config, 'eigenmode')
            else:
                # shape_space = {name: shape}
                uq_parallel(cav, eigenmode_config, 'eigenmode')

        done(f'Done with Cavity {cav.name}. Time: {time.time() - start_time}')

    for i, (key, cav) in enumerate(list(cavs_dict.items())):

        if os.path.exists(os.path.join(cav.self_dir, key)):
            if rerun:
                # delete old results
                shutil.rmtree(os.path.join(cav.self_dir, "eigenmode", "monopole"))
                _run_ngsolve(cav, eigenmode_config)
            else:
                # check if eigenmode analysis results exist
                if os.path.exists(os.path.join(cav.self_dir, 'eigenmode', "monopole", "qois.json")):
                    pass
                else:
                    shutil.rmtree(os.path.join(cav.self_dir, key))
                    _run_ngsolve(cav, eigenmode_config)
        else:
            _run_ngsolve(cav, eigenmode_config)


def run_wakefield_s(cavs_dict, wakefield_config, subdir):
    rerun = True
    if 'rerun' in wakefield_config.keys():
        assert isinstance(wakefield_config['rerun'], bool), error('rerun must be boolean.')
        rerun = wakefield_config['rerun']

    # MROT = wakefield_config['polarisation']
    # MT = wakefield_config['MT']
    # NFS = wakefield_config['NFS']
    # UBT = wakefield_config['wake_config']['wakelength']
    # bunch_length = wakefield_config['beam_config']['bunch_length']
    # DDR_SIG = wakefield_config['mesh_config']['DDR_SIG']
    # DDZ_SIG = wakefield_config['mesh_config']['DDZ_SIG']

    operating_points = None
    if 'operating_points' in wakefield_config.keys():
        operating_points = wakefield_config['operating_points']

    uq_config = wakefield_config['uq_config']
    # if uq_config:
    #     assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
    #                                                                          "be equal to the number of "
    #                                                                          "variables.")

    WG_M = None

    def _run_abci(cav, wakefield_config):

        freq = 0
        R_Q = 0
        # run abci code
        start_time = time.time()
        # run both polarizations if MROT == 2

        # run abci code
        # run both polarizations if MROT == 2

        cav.geo_to_abc(wakefield_config)
        abci.solve(cav, wakefield_config)
        done(f'Cavity {cav.name}. Time: {time.time() - start_time}')

        if 'uq_config' in wakefield_config.keys():
            uq_config = wakefield_config['uq_config']

            uq_cell_complexity = 'simplecell'
            if 'cell_complexity' in uq_config.keys():
                uq_cell_complexity = uq_config['cell_complexity']

            if uq_cell_complexity == 'multicell':
                # shape_space = {name: shape_multi}
                uq_parallel_multicell(cav, wakefield_config, 'wakefield')
            else:
                # shape_space = {name: shape}
                uq_parallel(cav, wakefield_config, 'wakefield')

        if operating_points:
            try:
                # check folder for freq and R/Q
                folder = os.path.join(cav.eigenmode_dir, 'monopole', 'qois.json')
                if os.path.exists(folder):
                    try:
                        with open(folder, 'r') as json_file:
                            fm_results = json.load(json_file)
                        freq = fm_results['freq [MHz]']
                        R_Q = fm_results['R/Q [Ohm]']
                    except OSError:
                        info("To run analysis for working points, eigenmode simulation has to be run first"
                             "to obtain the cavity operating frequency and R/Q")

                if freq != 0 and R_Q != 0:
                    d = {}
                    # save qois
                    for key_op, vals in operating_points.items():
                        WP = key_op
                        I0 = float(vals['I0 [mA]'])
                        Nb = float(vals['Nb [1e11]'])
                        sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                        bl_diff = ['SR', 'BS']

                        # info("Running wakefield analysis for given operating points.")
                        for i, s in enumerate(sigma_z):
                            for ii in WG_M:
                                fid = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
                                OC_R = 'OC'
                                if 'OC_R' in shape.keys():
                                    OC_R = 'OC_R'
                                for m in range(2):
                                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                                     fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                     bunch_length=s,
                                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
                                                     projectDir=projectDir,
                                                     WG_M=ii, marker=ii, sub_dir=f"{name}")

                                dirc = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                                # try:
                                k_loss = abs(ABCIData(dirc, f'{fid}', 0).loss_factor['Longitudinal'])
                                k_kick = abs(ABCIData(dirc, f'{fid}', 1).loss_factor['Transverse'])
                                # except:
                                #     k_loss = 0
                                #     k_kick = 0

                                d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, n_cells)

                    # save qoi dictionary
                    run_save_directory = os.path.join(cav.wakefield_dir)
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The working point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()

    for i, (key, cav) in enumerate(cavs_dict.items()):
        if os.path.exists(os.path.join(cav.self_dir, key)):
            if rerun:
                # remove old simulation results
                shutil.rmtree(os.path.join(cav.self_dir, "wakefield", "longitudinal"))
                _run_abci(cav, wakefield_config)
            else:
                # check if wakefield analysis results exist
                if os.path.exists(os.path.join(cav.self_dir, 'wakefield', "longitudinal", "qois.json")):
                    pass
                else:
                    _run_abci(cav, wakefield_config)
        else:
            _run_abci(cav, wakefield_config)


# def uq_parallel(shape_space, objectives, solver_dict, solver_args_dict,
#                 solver):
#     """
#
#     Parameters
#     ----------
#     key: str | int
#         Cavity geomery identifier
#     shape: dict
#         Dictionary containing geometric dimensions of cavity geometry
#     qois: list
#         Quantities of interest considered in uncertainty quantification
#     n_cells: int
#         Number of cavity cells
#     n_modules: int
#         Number of modules
#     n_modes: int
#         Number of eigenmodes to be calculated
#     f_shift: float
#         Since the eigenmode solver uses the power method, a shift can be provided
#     bc: int
#         Boundary conditions {1:inner contour, 2:Electric wall Et = 0, 3:Magnetic Wall En = 0, 4:Axis, 5:metal}
#         bc=33 means `Magnetic Wall En = 0` boundary condition at both ends
#     pol: int {Monopole, Dipole}
#         Defines whether to calculate for monopole or dipole modes
#     parentDir: str | path
#         Parent directory
#     projectDir: str|path
#         Project directory
#
#     Returns
#     -------
#     :param select_solver:
#
#     """
#
#     if solver == 'eigenmode':
#         # parentDir = solver_args_dict['parentDir']
#         projectDir = solver_args_dict['projectDir']
#         uq_config = solver_args_dict['eigenmode']['uq_config']
#         # cell_type = uq_config['cell_type']
#         analysis_folder = solver_args_dict['analysis folder']
#         # opt = solver_args_dict['optimisation']
#         # delta = uq_config['delta']
#
#         method = ['stroud3']
#         if 'method' in uq_config.keys():
#             method = uq_config['method']
#
#         uq_vars = uq_config['variables']
#         # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')
#
#         for key, shape in shape_space.items():
#             # n_cells = shape['n_cells']
#             uq_path = projectDir / fr'SimulationData\{analysis_folder}\{key}'
#
#             result_dict_eigen = {}
#             result_dict_eigen_all_modes = {}
#             # eigen_obj_list = []
#
#             # for o in objectives:
#             #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
#             #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
#             #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#             #         eigen_obj_list.append(o)
#
#             rdim = len(uq_vars)
#             degree = 1
#
#             if isinstance(method, str):
#                 flag = method
#             else:
#                 flag = method[0]
#
#             if flag.lower() == 'stroud3':
#                 nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
#                 nodes_ = 2. * nodes_ - 1.
#                 # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this
#                 # gives a less accurate answer. the nodes are not the same as the custom function
#             elif flag.lower() == 'stroud5':
#                 nodes_, weights_ = cn_leg_05_2(rdim)
#             elif flag.lower() == 'cn_gauss':
#                 nodes_, weights_ = cn_gauss(rdim, 2)
#             elif flag.lower() == 'lhc':
#                 sampler = qmc.LatinHypercube(d=rdim)
#                 _ = sampler.reset()
#                 nsamp = 2500
#                 sample = sampler.random(n=nsamp)
#                 # ic(qmc.discrepancy(sample))
#                 l_bounds = [-1, -1, -1, -1, -1, -1]
#                 u_bounds = [1, 1, 1, 1, 1, 1]
#                 sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
#
#                 nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
#             else:
#                 # issue warning
#                 warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
#                 nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
#                 nodes_ = 2. * nodes_ - 1.
#
#             # save nodes and weights
#             # print('uqvars', uq_vars)
#             data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
#             data_table.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'nodes.csv'),
#                               index=False, sep='\t', float_format='%.32f')
#
#             data_table_w = pd.DataFrame(weights_, columns=['weights'])
#             data_table_w.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'weights.csv'),
#                                 index=False,
#                                 sep='\t', float_format='%.32f')
#
#             no_parm, no_sims = np.shape(nodes_)
#             # if delta is None:
#             #     delta = [0.05 for _ in range(len(uq_vars))]
#
#             sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run
#
#             proc_count = 1
#             if 'processes' in uq_config.keys():
#                 assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
#                 assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
#                 proc_count = uq_config['processes']
#             if proc_count > no_sims:
#                 proc_count = no_sims
#
#             share = int(round(no_sims / proc_count))
#             jobs = []
#
#             base_chunk_size = no_sims // proc_count
#             remainder = no_sims % proc_count
#
#             start_idx = 0
#             for p in range(proc_count):
#                 # Determine the size of the current chunk
#                 current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
#                 proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
#                 start_idx += current_chunk_size
#
#                 processor_nodes = nodes_[:, proc_keys_list]
#                 processor_weights = weights_[proc_keys_list]
#                 service = mp.Process(target=uq, args=(key, objectives, uq_config, uq_path,
#                                                       solver_args_dict, sub_dir,
#                                                       proc_keys_list, processor_nodes, p, shape, solver))
#
#                 service.start()
#                 jobs.append(service)
#
#             for job in jobs:
#                 job.join()
#
#             # combine results from processes
#             for i1 in range(proc_count):
#                 if i1 == 0:
#                     df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
#                     df_all_modes = pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')
#                 else:
#                     df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])
#                     df_all_modes = pd.concat(
#                         [df_all_modes, pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')])
#
#             df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
#             df.to_excel(uq_path / 'table.xlsx', index=False)
#
#             Ttab_val_f = df.to_numpy()
#             # print(Ttab_val_f.shape, weights_.shape)
#             mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)
#
#             # # append results to dict
#             # for i, o in enumerate(eigen_obj_list):
#             #     result_dict_eigen[o]['expe'].append(mean_obj[i])
#             #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
#             for i, o in enumerate(df.columns):
#                 result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                 result_dict_eigen[o]['expe'].append(mean_obj[i])
#                 result_dict_eigen[o]['stdDev'].append(std_obj[i])
#                 result_dict_eigen[o]['skew'].append(skew_obj[i])
#                 result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
#             with open(uq_path / fr'uq.json', 'w') as file:
#                 file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))
#
#             # for all modes
#             df_all_modes.to_csv(uq_path / 'table_all_modes.csv', index=False, sep='\t', float_format='%.32f')
#             df_all_modes.to_excel(uq_path / 'table_all_modes.xlsx', index=False)
#
#             Ttab_val_f_all_modes = df_all_modes.to_numpy()
#             # print(Ttab_val_f_all_modes.shape, weights_.shape)
#             # print()
#             mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
#                 Ttab_val_f_all_modes, weights_)
#             # print(mean_obj_all_modes)
#
#             # # append results to dict
#             # for i, o in enumerate(eigen_obj_list):
#             #     result_dict_eigen[o]['expe'].append(mean_obj[i])
#             #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
#             for i, o in enumerate(df_all_modes.columns):
#                 result_dict_eigen_all_modes[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                 result_dict_eigen_all_modes[o]['expe'].append(mean_obj_all_modes[i])
#                 result_dict_eigen_all_modes[o]['stdDev'].append(std_obj_all_modes[i])
#                 result_dict_eigen_all_modes[o]['skew'].append(skew_obj_all_modes[i])
#                 result_dict_eigen_all_modes[o]['kurtosis'].append(kurtosis_obj_all_modes[i])
#
#             with open(uq_path / fr'uq_all_modes.json', 'w') as file:
#                 file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))
#
#     elif solver == 'wakefield':
#         # parentDir = solver_args_dict['parentDir']
#         projectDir = solver_args_dict['projectDir']
#         solver_args = solver_args_dict['wakefield']
#         # n_cells = solver_args['n_cells']
#         uq_config = solver_args['uq_config']
#         # delta = uq_config['delta']
#
#         method = 'stroud3'
#         if 'method' in uq_config.keys():
#             method = uq_config['method']
#
#         uq_vars = uq_config['variables']
#         # cell_type = uq_config['cell_type']
#         analysis_folder = solver_args_dict['analysis folder']
#
#         # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')
#
#         for key, shape in shape_space.items():
#             # n_cells = shape['n_cells']
#             uq_path = projectDir / fr'SimulationData\ABCI\{key}'
#             result_dict_wakefield = {}
#             # wakefield_obj_list = []
#
#             # for o in objectives:
#             #     if isinstance(o, list):
#             #         if o[1].split(' ')[0] in ['ZL', 'ZT']:
#             #             result_dict_wakefield[o[1]] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#             #             wakefield_obj_list.append(o[1])
#             #     else:
#             #         if o in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
#             #             result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#             #             wakefield_obj_list.append(o)
#
#             rdim = len(uq_vars)
#             degree = 1
#
#             if isinstance(method, str):
#                 flag = method
#             else:
#                 flag = method[0]
#
#             if flag.lower() == 'stroud3':
#                 nodes_, weights_, bpoly_ = quad_stroud3(rdim, degree)
#                 nodes_ = 2. * nodes_ - 1.
#                 # nodes_, weights_ = cn_leg_03_1(rdim)  # <- for some reason unknown this
#                 # gives a less accurate answer. the nodes are not the same as the custom function
#             elif flag.lower() == 'stroud5':
#                 nodes_, weights_ = cn_leg_05_2(rdim)
#             elif flag.lower() == 'cn_gauss':
#                 nodes_, weights_ = cn_gauss(rdim, 2)
#             elif flag.lower() == 'lhc':
#                 sampler = qmc.LatinHypercube(d=rdim)
#                 _ = sampler.reset()
#                 nsamp = 2500
#                 sample = sampler.random(n=nsamp)
#                 # ic(qmc.discrepancy(sample))
#                 l_bounds = [-1, -1, -1, -1, -1, -1]
#                 u_bounds = [1, 1, 1, 1, 1, 1]
#                 sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
#
#                 nodes_, weights_ = sample_scaled.T, np.ones((nsamp, 1))
#             else:
#                 # issue warning
#                 warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
#                 nodes_, weights_, bpoly = quad_stroud3(rdim, degree)
#                 nodes_ = 2. * nodes_ - 1.
#
#             # save nodes
#             data_table = pd.DataFrame(nodes_.T, columns=uq_vars)
#             data_table.to_csv(os.path.join(projectDir, "SimulationData", analysis_folder, key, 'nodes.csv'),
#                               index=False, sep='\t', float_format='%.32f')
#
#             data_table_w = pd.DataFrame(weights_, columns=['weights'])
#             data_table_w.to_csv(os.path.join(projectDir, "SimulationData", analysis_folder, key, 'weights.csv'),
#                                 index=False,
#                                 sep='\t', float_format='%.32f')
#
#             no_parm, no_sims = np.shape(nodes_)
#             # if delta is None:
#             #     delta = [0.05 for _ in range(len(uq_vars))]
#
#             sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run
#
#             proc_count = 1
#             if 'processes' in uq_config.keys():
#                 assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
#                 assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
#                 proc_count = uq_config['processes']
#
#             jobs = []
#
#             # if proc_count > no_sims:
#             #     proc_count = no_sims
#             #
#             # share = int(round(no_sims / proc_count))
#             # for p in range(proc_count):
#             #     # try:
#             #     end_already = False
#             #     if p != proc_count - 1:
#             #         if (p + 1) * share < no_sims:
#             #             proc_keys_list = np.arange(p * share, p * share + share)
#             #         else:
#             #             proc_keys_list = np.arange(p * share, no_sims)
#             #             end_already = True
#             #
#             #     if p == proc_count - 1 and not end_already:
#             #         proc_keys_list = np.arange(p * share, no_sims)
#
#             base_chunk_size = no_sims // proc_count
#             remainder = no_sims % proc_count
#
#             start_idx = 0
#             for p in range(proc_count):
#                 # Determine the size of the current chunk
#                 current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
#                 proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
#                 start_idx += current_chunk_size
#
#                 processor_nodes = nodes_[:, proc_keys_list]
#                 processor_weights = weights_[proc_keys_list]
#
#                 service = mp.Process(target=uq, args=(key, objectives, uq_config, uq_path,
#                                                       solver_args_dict, sub_dir, proc_keys_list, processor_nodes,
#                                                       p, shape, solver))
#
#                 service.start()
#                 jobs.append(service)
#
#             for job in jobs:
#                 job.join()
#
#             # combine results from processes
#             # qois_result_dict = {}
#             # keys = []
#             # Ttab_val_f = []
#             for i1 in range(proc_count):
#                 if i1 == 0:
#                     df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
#                 else:
#                     df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])
#
#             df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
#             df.to_excel(uq_path / 'table.xlsx', index=False)
#             Ttab_val_f = df.to_numpy()
#             mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)
#             # # append results to dict
#             # for i, o in enumerate(wakefield_obj_list):
#             #     result_dict_wakefield[o]['expe'].append(mean_obj[i])
#             #     result_dict_wakefield[o]['stdDev'].append(std_obj[i])
#             #
#             for i, o in enumerate(df.columns):
#                 result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                 result_dict_wakefield[o]['expe'].append(mean_obj[i])
#                 result_dict_wakefield[o]['stdDev'].append(std_obj[i])
#                 result_dict_wakefield[o]['skew'].append(skew_obj[i])
#                 result_dict_wakefield[o]['kurtosis'].append(kurtosis_obj[i])
#
#             with open(uq_path / fr'uq.json', 'w') as f:
#                 f.write(json.dumps(result_dict_wakefield, indent=4, separators=(',', ': ')))
#
#     else:
#         pass
#
#
# def uq(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
#        proc_keys_list, processor_nodes, proc_num, shape, solver):
#     """
#
#     Parameters
#     ----------
#     shape_space: dict
#         Cavity geometry parameter space
#     objectives: list | ndarray
#         Array of objective functions
#     solver_dict: dict
#         Python dictionary of solver settings
#     solver_args_dict: dict
#         Python dictionary of solver arguments
#     uq_config:
#         Python dictionary of uncertainty quantification settings
#
#     Returns
#     -------
#     :param n_cells:
#
#     """
#
#     # print(processor_nodes)
#     if solver == 'eigenmode':
#         parentDir = solver_args_dict['parentDir']
#         projectDir = solver_args_dict['projectDir']
#         # cell_type = uq_config['cell_type']
#         analysis_folder = solver_args_dict['analysis folder']
#         opt = solver_args_dict['optimisation']
#
#         epsilon, delta = [], []
#         if 'epsilon' in uq_config.keys():
#             epsilon = uq_config['epsilon']
#         if 'delta' in uq_config.keys():
#             delta = uq_config['delta']
#         if 'perturbed_cell' in uq_config.keys():
#             perturbed_cell = uq_config['perturbed_cell']
#         else:
#             perturbed_cell = 'mid-cell'
#
#         # method = uq_config['method']
#         uq_vars = uq_config['variables']
#         cell_parameterisation = solver_args_dict['cell_parameterisation']
#         err = False
#         result_dict_eigen = {}
#         Ttab_val_f = []
#         Ttab_val_f_all_modes = []
#
#         # eigen_obj_list = objectives
#         eigen_obj_list = []
#
#         for o in objectives:
#             if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
#                      "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
#                 result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                 eigen_obj_list.append(o)
#
#         cell_node = shape['IC']
#         cell_node_left = shape['OC']
#         cell_node_right = shape['OC_R']
#
#         perturbed_cell_node = np.array(cell_node)
#         perturbed_cell_node_left = np.array(cell_node_left)
#         perturbed_cell_node_right = np.array(cell_node_right)
#
#         for i1, proc_key in enumerate(proc_keys_list):
#             skip = False
#             for j, uq_var in enumerate(uq_vars):
#                 uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
#                 if epsilon:
#                     if perturbed_cell.lower() == 'mid cell' or perturbed_cell.lower() == 'mid-cell' or perturbed_cell.lower() == 'mid_cell':
#                         perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
#                     elif perturbed_cell.lower() == 'end cell' or perturbed_cell.lower() == 'end-cell' or perturbed_cell.lower() == 'end_cell':
#                         perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * \
#                                                                 processor_nodes[
#                                                                     j, i1]
#                         perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
#                                                                  processor_nodes[j, i1]
#                     else:
#                         perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
#                         perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * \
#                                                                 processor_nodes[
#                                                                     j, i1]
#                         perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
#                                                                  processor_nodes[j, i1]
#
#                 else:
#                     if perturbed_cell.lower() == 'mid cell' or perturbed_cell.lower() == 'mid-cell' or perturbed_cell.lower() == 'mid_cell':
#                         perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#                     elif perturbed_cell.lower() == 'end cell' or perturbed_cell.lower() == 'end-cell' or perturbed_cell.lower() == 'end_cell':
#                         perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#                         perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#                     else:
#                         perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#                         perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#                         perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
#                                 1 + delta[j] * processor_nodes[j, i1])
#
#             # print('\tafter: ', perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)
#
#             enforce_Req_continuity(perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)
#
#             # perform checks on geometry
#             fid = fr'{key}_Q{proc_key}'
#
#             # check if folder exists and skip if it does
#             if os.path.exists(os.path.join(projectDir, 'SimulationData', analysis_folder, key, fid)):
#                 skip = True
#                 info(["Skipped: ", fid, os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)])
#
#             if not skip:
#                 ngsolve_mevp.createFolder(fid, projectDir, subdir=sub_dir, opt=opt)
#                 # it does not seem to make sense to perform uq on a multi cell by repeating the same perturbation
#                 # to all multi cells at once. For multicells, the uq_multicell option is more suitable as it creates
#                 # independent perturbations to all cells individually
#
#                 if 'tune_config' in uq_config.keys():
#                     tune_config = uq_config['tune_config']
#                     # tune first before running
#                     processor_shape_space = {fid:
#                                                  {'IC': perturbed_cell_node,
#                                                   'OC': perturbed_cell_node_left,
#                                                   'OC_R': perturbed_cell_node_right,
#                                                   'BP': 'both',
#                                                   "n_cells": 9,
#                                                   'CELL PARAMETERISATION': 'simplecell'
#                                                   },
#                                              }
#                     # save tune results to uq cavity folders
#                     sim_folder = os.path.join(analysis_folder, key)
#
#                     run_tune_s(processor_shape_space, tune_config['parameters'], tune_config['freqs'],
#                                tune_config['cell_types'], tune_config, projectDir, False, fid, sim_folder)
#
#                 else:
#                     if cell_parameterisation == 'simplecell':
#                         ngsolve_mevp.cavity(shape['n_cells'], 1, perturbed_cell_node,
#                                             perturbed_cell_node_left, perturbed_cell_node_right,
#                                             f_shift=0, bc=33, beampipes=shape['BP'],
#                                             fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
#                                             projectDir=projectDir,
#                                             subdir=sub_dir)
#                     if cell_parameterisation == 'flattop':
#                         ngsolve_mevp.cavity_flattop(shape['n_cells'], 1, perturbed_cell_node,
#                                                     perturbed_cell_node_left, perturbed_cell_node_right,
#                                                     f_shift=0, bc=33,
#                                                     beampipes=shape['BP'],
#                                                     fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
#                                                     projectDir=projectDir,
#                                                     subdir=sub_dir)
#
#             filename = uq_path / f'{fid}/monopole/qois.json'
#             filename_all_modes = uq_path / f'{fid}/monopole/qois_all_modes.json'
#             if os.path.exists(filename):
#                 qois_result_dict = dict()
#
#                 with open(filename) as json_file:
#                     qois_result_dict.update(json.load(json_file))
#                 qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)
#
#                 tab_val_f = qois_result
#                 Ttab_val_f.append(tab_val_f)
#             else:
#                 err = True
#
#             # for all modes
#             if os.path.exists(filename_all_modes):
#                 qois_result_dict_all_modes = dict()
#                 qois_result_all_modes = {}
#
#                 with open(filename_all_modes) as json_file:
#                     qois_result_dict_all_modes.update(json.load(json_file))
#
#                 tab_val_f_all_modes = []
#                 for kk, val in qois_result_dict_all_modes.items():
#                     qois_result_all_modes[kk] = get_qoi_value(val, eigen_obj_list)
#
#                     tab_val_f_all_modes.append(qois_result_all_modes[kk])
#
#                 tab_val_f_all_modes_flat = [item for sublist in tab_val_f_all_modes for item in sublist]
#                 Ttab_val_f_all_modes.append(tab_val_f_all_modes_flat)
#             else:
#                 err = True
#
#         data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
#         data_table.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')
#
#         # for all modes
#         keys = qois_result_dict_all_modes.keys()
#         eigen_obj_list_all_modes = [f"{name.split(' ')[0]}_{i} {name.split(' ', 1)[1]}" for i in keys for name in
#                                     eigen_obj_list]
#         data_table = pd.DataFrame(Ttab_val_f_all_modes, columns=eigen_obj_list_all_modes)
#         data_table.to_csv(uq_path / fr'table_{proc_num}_all_modes.csv', index=False, sep='\t', float_format='%.32f')
#
#     elif solver == 'wakefield':
#         Ttab_val_f = []
#         parentDir = solver_args_dict['parentDir']
#         projectDir = solver_args_dict['projectDir']
#         solver_args = solver_args_dict['wakefield']
#         # n_cells = solver_args_dict['n_cells']
#         # n_modules = solver_args_dict['n_modules']
#         MROT = solver_args['polarisation']
#         MT = solver_args['MT']
#         NFS = solver_args['NFS']
#         UBT = solver_args['wakelength']
#         bunch_length = solver_args['bunch_length']
#         DDR_SIG = solver_args['DDR_SIG']
#         DDZ_SIG = solver_args['DDZ_SIG']
#         # WG_M = solver_args['WG_M']
#
#         epsilon, delta = [], []
#         if 'epsilon' in uq_config.keys():
#             epsilon = uq_config['epsilon']
#         if 'delta' in uq_config.keys():
#             delta = uq_config['delta']
#
#         # method = uq_config['method']
#         cell_parameterisation = solver_args_dict['cell_parameterisation']
#         uq_vars = uq_config['variables']
#         # cell_type = uq_config['cell_type']
#         analysis_folder = solver_args_dict['analysis folder']
#         # marker = solver_args['marker']
#
#         result_dict_wakefield = {}
#         # wakefield_obj_list = objectives
#         wakefield_obj_list = []
#
#         for o in objectives:
#             if isinstance(o, list):
#                 if o[1].split(' ')[0] in ['ZL', 'ZT']:
#                     result_dict_wakefield[o[1]] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                     wakefield_obj_list.append(o[1])
#             else:
#                 if o in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
#                     result_dict_wakefield[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
#                     wakefield_obj_list.append(o)
#
#         # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
#         #     cell_node = shape['IC']
#         # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
#         #     cell_node = shape['OC']
#         # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
#         #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
#         #     cell_node = shape['OC']
#         # else:
#         #     cell_node = shape['OC']
#
#         cell_node = shape['IC']
#         cell_node_left = shape['OC']
#         cell_node_right = shape['OC_R']
#         perturbed_cell_node = np.array(cell_node)
#         perturbed_cell_node_left = np.array(cell_node_left)
#         perturbed_cell_node_right = np.array(cell_node_right)
#
#         uq_shape_space = {}
#         d_uq_op = {}
#         for i1, proc_key in enumerate(proc_keys_list):
#             skip = False
#             for j, uq_var in enumerate(uq_vars):
#                 uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
#                 if epsilon:
#                     perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] + epsilon[j] * processor_nodes[j, i1]
#                     perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] + epsilon[j] * processor_nodes[
#                         j, i1]
#                     perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] + epsilon[j] * \
#                                                              processor_nodes[j, i1]
#                 else:
#                     perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (1 + delta[j] * processor_nodes[j, i1])
#                     perturbed_cell_node_left[uq_var_indx] = cell_node_left[uq_var_indx] * (
#                             1 + delta[j] * processor_nodes[j, i1])
#                     perturbed_cell_node_right[uq_var_indx] = cell_node_right[uq_var_indx] * (
#                             1 + delta[j] * processor_nodes[j, i1])
#
#             # if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
#             #     # cell_node = shape['IC']
#             #     mid = perturbed_cell_node
#             #     left = shape['OC']
#             #     right = shape['OC_R']
#             # elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
#             #     mid = shape['IC']
#             #     left = perturbed_cell_node
#             #     right = perturbed_cell_node
#             # elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
#             #       or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
#             #     mid = perturbed_cell_node
#             #     left = perturbed_cell_node
#             #     right = perturbed_cell_node
#             # elif cell_type.lower() == 'end cell' or cell_type.lower() == 'end-cell' or cell_type.lower() == 'end_cell':
#             #     mid = shape['IC']
#             #     left = perturbed_cell_node
#             #     right = perturbed_cell_node
#             # else:
#             #     mid = perturbed_cell_node
#             #     left = perturbed_cell_node
#             #     right = perturbed_cell_node
#
#             enforce_Req_continuity(perturbed_cell_node, perturbed_cell_node_left, perturbed_cell_node_right)
#
#             # perform checks on geometry
#             fid = fr'{key}_Q{proc_key}'
#
#             # check if folder exists and skip if it does
#             if os.path.exists(os.path.join(projectDir, 'SimulationData', analysis_folder, key, fid)):
#                 skip = True
#                 info(["Skipped: ", fid, os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)])
#
#             if not skip:
#                 abci_geom.createFolder(fid, projectDir, subdir=sub_dir)
#                 for wi in range(MROT):
#                     if cell_parameterisation == 'simplecell':
#                         abci_geom.cavity(shape['n_cells'], 1, perturbed_cell_node,
#                                          perturbed_cell_node_left, perturbed_cell_node_right, fid=fid, MROT=wi,
#                                          DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, beampipes=shape['BP'],
#                                          bunch_length=bunch_length,
#                                          MT=MT, NFS=NFS, UBT=UBT,
#                                          parentDir=parentDir, projectDir=projectDir, WG_M='',
#                                          marker='', sub_dir=sub_dir
#                                          )
#                     if cell_parameterisation == 'flattop':
#                         abci_geom.cavity_flattop(shape['n_cells'], 1, perturbed_cell_node,
#                                                  perturbed_cell_node_left, perturbed_cell_node_right, fid=fid, MROT=wi,
#                                                  DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, beampipes=shape['BP'],
#                                                  bunch_length=bunch_length,
#                                                  MT=MT, NFS=NFS, UBT=UBT,
#                                                  parentDir=parentDir, projectDir=projectDir, WG_M='',
#                                                  marker='', sub_dir=sub_dir
#                                                  )
#
#             uq_shape = {'IC': perturbed_cell_node, 'OC': perturbed_cell_node_left,
#                         'OC_R': perturbed_cell_node_right, 'n_cells': shape['n_cells'], 'BP': shape['BP']}
#             uq_shape_space[fid] = uq_shape
#
#             # calculate uq for operating points
#             if 'operating_points' in uq_config.keys():
#                 operating_points = uq_config['operating_points']
#                 try:
#                     # check folder for freq and R/Q
#                     freq, R_Q = 0, 0
#                     folder = os.path.join(projectDir, 'SimulationData', 'NGSolveMEVP', key, fid, 'monopole',
#                                           'qois.json')
#                     if os.path.exists(folder):
#                         try:
#                             with open(folder, 'r') as json_file:
#                                 fm_results = json.load(json_file)
#                             freq = fm_results['freq [MHz]']
#                             R_Q = fm_results['R/Q [Ohm]']
#                         except OSError:
#                             info("To run analysis for working points, eigenmode simulation has to be run first"
#                                  "to obtain the cavity operating frequency and R/Q")
#
#                     if freq != 0 and R_Q != 0:
#                         d = {}
#                         # save qois
#                         for key_op, vals in operating_points.items():
#                             WP = key_op
#                             I0 = float(vals['I0 [mA]'])
#                             Nb = float(vals['Nb [1e11]'])
#                             sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
#                             bl_diff = ['SR', 'BS']
#
#                             # info("Running wakefield analysis for given operating points.")
#                             for i, s in enumerate(sigma_z):
#                                 for ii in ['']:
#                                     fid_op = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
#                                     OC_R = 'OC'
#                                     if 'OC_R' in uq_shape.keys():
#                                         OC_R = 'OC_R'
#                                     for m in range(2):
#                                         abci_geom.cavity(shape['n_cells'], 1, uq_shape['IC'], uq_shape['OC'],
#                                                          uq_shape[OC_R],
#                                                          fid=fid_op, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
#                                                          bunch_length=s,
#                                                          DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=SOFTWARE_DIRECTORY,
#                                                          projectDir=projectDir,
#                                                          WG_M=ii, marker=ii, sub_dir=fr"{key}\{fid}")
#
#                                     dirc = os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)
#                                     # try:
#                                     k_loss = abs(ABCIData(dirc, f'{fid_op}', 0).loss_factor['Longitudinal'])
#                                     k_kick = abs(ABCIData(dirc, f'{fid_op}', 1).loss_factor['Transverse'])
#                                     # except:
#                                     #     k_loss = 0
#                                     #     k_kick = 0
#
#                                     d[fid_op] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, shape['n_cells'])
#
#                         d_uq_op[fid] = d
#                         # save qoi dictionary
#                         run_save_directory = os.path.join(projectDir, 'SimulationData', 'ABCI', key, fid)
#                         with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
#                             json.dump(d, f, indent=4, separators=(',', ': '))
#
#                         done("Done with the secondary analysis for working points")
#                     else:
#                         info("To run analysis for working points, eigenmode simulation has to be run first"
#                              "to obtain the cavity operating frequency and R/Q")
#                 except KeyError:
#                     error('The working point entered is not valid. See below for the proper input structure.')
#                     show_valid_operating_point_structure()
#
#         df_uq_op = _data_uq_op(d_uq_op)
#
#         wakefield_folder = os.path.join(projectDir, 'SimulationData', 'ABCI', key)
#         data_table, _ = get_wakefield_objectives_value(uq_shape_space, uq_config['objectives_unprocessed'],
#                                                        wakefield_folder)
#
#         # merge with data table
#         if d_uq_op:
#             data_table_merged = pd.merge(df_uq_op, data_table, on='key', how='inner')
#         else:
#             data_table_merged = data_table
#         data_table_merged = data_table_merged.set_index('key')
#         data_table_merged.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')


def uq_parallel(cav, eigenmode_config, solver='eigenmode'):
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

    """
    if solver == 'eigenmode':

        if not os.path.exists(cav.uq_dir):
            os.mkdir(cav.uq_dir)

        result_dict_eigen = {}
        result_dict_eigen_all_modes = {}
        # eigen_obj_list = []

        # for o in objectives:
        #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
        #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
        #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
        #         eigen_obj_list.append(o)

        # print("multicell shape space", shape_space)
        # cav_var_list = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        # midcell_var_dict = dict()
        # for i1 in range(len(cav_var_list)):
        #     for i2 in range(n_cells):
        #         for i3 in range(2):
        #             midcell_var_dict[f'{cav_var_list[i1]}_{i2}_m{i3}'] = [i1, i2, i3]

        # # create random variables
        # multicell_mid_vars = shape['IC']
        # print(multicell_mid_vars)

        # if n_cells == 1:
        #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
        #     p_true = [np.array(shape['OC'])[:7], np.array(shape['OC_R'])[:7]]
        #     rdim = len(np.array(shape['OC'])[:7]) + len(
        #         np.array(shape['OC_R'])[:7])
        # else:
        #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
        #     p_true = [np.array(shape['OC'])[:7], multicell_mid_vars, np.array(shape['OC_R'])[:7]]
        #     rdim = len(np.array(shape['OC'])[:7]) + multicell_mid_vars.size + len(
        #         np.array(shape['OC_R'])[:7])
        #     # rdim = rdim - (n_cells*2 - 1)  # <- reduce dimension by making iris and equator radii to be equal
        #
        # ic(rdim, multicell_mid_vars.size)
        # rdim = n_cells*3  # How many variables will be considered as random in our case 5

        perturbed_cavities, weights_ = perturb_geometry(cav, eigenmode_config)

        nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
        # save nodes
        nodes_perturbed.to_csv(os.path.join(cav.uq_dir, 'nodes_before_continuity.csv'), index=False, sep='\t', float_format='%.32f')

        # enforce continuity
        # nodes_ = enforce_continuity_df(nodes_perturbed)
        nodes_ = nodes_perturbed

        # save nodes
        nodes_.to_csv(os.path.join(cav.uq_dir, 'nodes.csv'), index=False, sep='\t', float_format='%.32f')

        data_table_w = pd.DataFrame(weights_, columns=['weights'])
        data_table_w.to_csv(os.path.join(cav.uq_dir, 'weights.csv'),
                            index=False, sep='\t',
                            float_format='%.32f')

        #  mean value of geometrical parameters
        no_sims, no_parm = np.shape(nodes_)

        # sub_dir = fr'{cav.name}'  # the simulation runs at the quadrature points are saved to the key of mean value run

        proc_count = 1
        if 'processes' in eigenmode_config['uq_config'].keys():
            assert eigenmode_config['uq_config']['processes'] > 0, error('Number of processes must be greater than zero')
            assert isinstance(eigenmode_config['uq_config']['processes'], int), error('Number of processes must be integer')
            proc_count = eigenmode_config['uq_config']['processes']

        jobs = []

        base_chunk_size = no_sims // proc_count
        remainder = no_sims % proc_count

        keys = list(perturbed_cavities.keys())
        start_idx = 0
        for p in range(proc_count):
            # Determine the size of the current chunk
            current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
            # proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
            proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
            start_idx += current_chunk_size

            # processor_nodes = nodes_.iloc[proc_keys_list, :]
            # processor_weights = weights_[proc_keys_list]
            processor_cavs_dict = {key: perturbed_cavities[key] for key in proc_keys_list}

            service = mp.Process(target=uq, args=(processor_cavs_dict, eigenmode_config, p))

            service.start()
            jobs.append(service)

        for job in jobs:
            job.join()

            # combine results from processes
            # qois_result_dict = {}
            # Ttab_val_f = []
        # keys = []
        for i1 in range(proc_count):
            if i1 == 0:
                df = pd.read_csv(os.path.join(cav.uq_dir, fr'table_{i1}.csv'), sep='\t', engine='python')
                df_all_modes = pd.read_csv(os.path.join(cav.uq_dir, fr'table_{i1}_all_modes.csv'), sep='\t', engine='python')
            else:
                df = pd.concat([df, pd.read_csv(os.path.join(cav.uq_dir, fr'table_{i1}.csv'), sep='\t', engine='python')])
                df_all_modes = pd.concat(
                    [df_all_modes, pd.read_csv(os.path.join(cav.uq_dir, fr'table_{i1}_all_modes.csv'), sep='\t', engine='python')])

        df.to_csv(os.path.join(cav.uq_dir, 'table.csv'), index=False, sep='\t', float_format='%.32f')
        df.to_excel(os.path.join(cav.uq_dir, 'table.xlsx'), index=False)

        Ttab_val_f = df.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

        for i, o in enumerate(df.columns):
            result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            result_dict_eigen[o]['expe'].append(mean_obj[i])
            result_dict_eigen[o]['stdDev'].append(std_obj[i])
            result_dict_eigen[o]['skew'].append(skew_obj[i])
            result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
        with open(os.path.join(cav.uq_dir, fr'uq.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))

        # for all modes
        df_all_modes.to_csv(os.path.join(cav.uq_dir, 'table_all_modes.csv'), index=False, sep='\t', float_format='%.32f')
        df_all_modes.to_excel(os.path.join(cav.uq_dir, 'table_all_modes.xlsx'), index=False)

        Ttab_val_f_all_modes = df_all_modes.to_numpy()

        mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
            Ttab_val_f_all_modes, weights_)

        for i, o in enumerate(df_all_modes.columns):
            result_dict_eigen_all_modes[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            result_dict_eigen_all_modes[o]['expe'].append(mean_obj_all_modes[i])
            result_dict_eigen_all_modes[o]['stdDev'].append(std_obj_all_modes[i])
            result_dict_eigen_all_modes[o]['skew'].append(skew_obj_all_modes[i])
            result_dict_eigen_all_modes[o]['kurtosis'].append(kurtosis_obj_all_modes[i])

        with open(os.path.join(cav.uq_dir, 'uq_all_modes.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))


def uq(proc_cavs_dict, eigenmode_config, proc_num):
    result_dict_eigen = {}
    Ttab_val_f = []
    Ttab_val_f_all_modes = []

    qois_result_dict_all_modes = dict()
    qois_result_all_modes = {}

    # eigen_obj_list = objectives
    eigen_obj_list = []

    for o in eigenmode_config['uq_config']['objectives']:
        if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                 "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
            result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            eigen_obj_list.append(o)

    for name, cav in proc_cavs_dict.items():
        skip = False

        # skip analysis if folder already exists.
        if not skip:
            if 'tune_config' in eigenmode_config['uq_config'].keys():
                tune_config = eigenmode_config['uq_config']['tune_config']

                # save tune results to uq cavity folders
                run_tune_s({name: cav}, tune_config, proc_num)

            filename = os.path.join(cav.eigenmode_dir, 'monopole', 'qois.json')
            filename_all_modes = os.path.join(cav.eigenmode_dir, 'monopole', 'qois_all_modes.json')

            if os.path.exists(filename):
                qois_result_dict = dict()
                with open(filename) as json_file:
                    qois_result_dict.update(json.load(json_file))
                qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)

                tab_val_f = qois_result
                Ttab_val_f.append(tab_val_f)
            else:
                err = True

            # for all modes
            if os.path.exists(filename_all_modes):
                with open(filename_all_modes) as json_file:
                    qois_result_dict_all_modes.update(json.load(json_file))

                tab_val_f_all_modes = []
                for kk, val in qois_result_dict_all_modes.items():
                    qois_result_all_modes[kk] = get_qoi_value(val, eigen_obj_list)

                    tab_val_f_all_modes.append(qois_result_all_modes[kk])

                tab_val_f_all_modes_flat = [item for sublist in tab_val_f_all_modes for item in sublist]
                Ttab_val_f_all_modes.append(tab_val_f_all_modes_flat)
            else:
                err = True

    data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
    data_table.to_csv(os.path.join(cav.projectDir, fr'table_{proc_num}.csv'), index=False, sep='\t', float_format='%.32f')

    # for all modes
    keys = qois_result_dict_all_modes.keys()
    eigen_obj_list_all_modes = [f"{name.split(' ')[0]}_{i} {name.split(' ', 1)[1]}" for i in keys for name in
                                eigen_obj_list]
    data_table = pd.DataFrame(Ttab_val_f_all_modes, columns=eigen_obj_list_all_modes)
    data_table.to_csv(os.path.join(cav.projectDir, fr'table_{proc_num}_all_modes.csv'), index=False, sep='\t', float_format='%.32f')

def uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict, solver):
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

    """
    if solver == 'eigenmode':
        # parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        uq_config = solver_args_dict['eigenmode']['uq_config']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        uq_vars = uq_config['variables']
        which_cell = uq_config['cell']
        # opt = solver_args_dict['optimisation']
        # delta = uq_config['delta']

        method = uq_config['method']
        perturbation_mode = uq_config['perturbation_mode']
        if not isinstance(perturbation_mode[1], list):
            perturbation_mode[1] = [perturbation_mode[1]] * len(uq_vars)

        # assert len(uq_vars) == len(delta), error('Ensure number of variables equal number of deltas')

        # print('shape_space', shape_space)
        for key, shape in shape_space.items():
            n_cells = shape['n_cells']

            uq_path = projectDir / fr'SimulationData\{analysis_folder}\{key}'

            result_dict_eigen = {}
            result_dict_eigen_all_modes = {}
            # eigen_obj_list = []

            # for o in objectives:
            #     if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
            #              "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
            #         result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            #         eigen_obj_list.append(o)

            # print("multicell shape space", shape_space)
            # cav_var_list = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
            # midcell_var_dict = dict()
            # for i1 in range(len(cav_var_list)):
            #     for i2 in range(n_cells):
            #         for i3 in range(2):
            #             midcell_var_dict[f'{cav_var_list[i1]}_{i2}_m{i3}'] = [i1, i2, i3]

            # # create random variables
            # multicell_mid_vars = shape['IC']
            # print(multicell_mid_vars)

            # if n_cells == 1:
            #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            #     p_true = [np.array(shape['OC'])[:7], np.array(shape['OC_R'])[:7]]
            #     rdim = len(np.array(shape['OC'])[:7]) + len(
            #         np.array(shape['OC_R'])[:7])
            # else:
            #     # EXAMPLE: p_true = np.array([1, 2, 3, 4, 5]).T
            #     p_true = [np.array(shape['OC'])[:7], multicell_mid_vars, np.array(shape['OC_R'])[:7]]
            #     rdim = len(np.array(shape['OC'])[:7]) + multicell_mid_vars.size + len(
            #         np.array(shape['OC_R'])[:7])
            #     # rdim = rdim - (n_cells*2 - 1)  # <- reduce dimension by making iris and equator radii to be equal
            #
            # ic(rdim, multicell_mid_vars.size)
            # rdim = n_cells*3  # How many variables will be considered as random in our case 5

            perturbed_cavities, weights_ = generate_perturbed_shapes(shape,
                                                                     cells=which_cell,
                                                                     variables=uq_vars,
                                                                     mode=perturbation_mode,
                                                                     node_type=method)

            nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
            # save nodes
            nodes_perturbed.to_csv(uq_path / 'nodes_before_continuity.csv', index=False, sep='\t', float_format='%.32f')

            # enforce continuity
            nodes_ = enforce_continuity_df(nodes_perturbed)

            # save nodes
            nodes_.to_csv(uq_path / 'nodes.csv', index=False, sep='\t', float_format='%.32f')

            data_table_w = pd.DataFrame(weights_, columns=['weights'])
            data_table_w.to_csv(os.path.join(projectDir, 'SimulationData', analysis_folder, key, 'weights.csv'),
                                index=False, sep='\t',
                                float_format='%.32f')

            #  mean value of geometrical parameters
            no_sims, no_parm = np.shape(nodes_)

            sub_dir = fr'{key}'  # the simulation runs at the quadrature points are saved to the key of mean value run

            proc_count = 1
            if 'processes' in uq_config.keys():
                assert uq_config['processes'] > 0, error('Number of processes must be greater than zero')
                assert isinstance(uq_config['processes'], int), error('Number of processes must be integer')
                proc_count = uq_config['processes']

            jobs = []

            base_chunk_size = no_sims // proc_count
            remainder = no_sims % proc_count

            start_idx = 0
            for p in range(proc_count):
                # Determine the size of the current chunk
                current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
                proc_keys_list = np.arange(start_idx, start_idx + current_chunk_size)
                start_idx += current_chunk_size

                processor_nodes = nodes_.iloc[proc_keys_list, :]
                # processor_weights = weights_[proc_keys_list]

                service = mp.Process(target=uq_multicell_s, args=(key, objectives, uq_config, uq_path,
                                                                  solver_args_dict, sub_dir,
                                                                  proc_keys_list, processor_nodes, p, shape, solver))

                service.start()
                jobs.append(service)

            for job in jobs:
                job.join()

                # combine results from processes
                # qois_result_dict = {}
                # Ttab_val_f = []
            # keys = []
            for i1 in range(proc_count):
                if i1 == 0:
                    df = pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')
                    df_all_modes = pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')
                else:
                    df = pd.concat([df, pd.read_csv(uq_path / fr'table_{i1}.csv', sep='\t', engine='python')])
                    df_all_modes = pd.concat(
                        [df_all_modes, pd.read_csv(uq_path / fr'table_{i1}_all_modes.csv', sep='\t', engine='python')])

            df.to_csv(uq_path / 'table.csv', index=False, sep='\t', float_format='%.32f')
            df.to_excel(uq_path / 'table.xlsx', index=False)

            Ttab_val_f = df.to_numpy()
            # print(Ttab_val_f.shape, weights_.shape)
            mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

            # # append results to dict
            # for i, o in enumerate(eigen_obj_list):
            #     result_dict_eigen[o]['expe'].append(mean_obj[i])
            #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
            for i, o in enumerate(df.columns):
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                result_dict_eigen[o]['expe'].append(mean_obj[i])
                result_dict_eigen[o]['stdDev'].append(std_obj[i])
                result_dict_eigen[o]['skew'].append(skew_obj[i])
                result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
            with open(uq_path / fr'uq.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))

            # for all modes
            df_all_modes.to_csv(uq_path / 'table_all_modes.csv', index=False, sep='\t', float_format='%.32f')
            df_all_modes.to_excel(uq_path / 'table_all_modes.xlsx', index=False)

            Ttab_val_f_all_modes = df_all_modes.to_numpy()
            # print(Ttab_val_f_all_modes.shape, weights_.shape)
            # print()
            mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
                Ttab_val_f_all_modes, weights_)
            # print(mean_obj_all_modes)

            # # append results to dict
            # for i, o in enumerate(eigen_obj_list):
            #     result_dict_eigen[o]['expe'].append(mean_obj[i])
            #     result_dict_eigen[o]['stdDev'].append(std_obj[i])
            for i, o in enumerate(df_all_modes.columns):
                result_dict_eigen_all_modes[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                result_dict_eigen_all_modes[o]['expe'].append(mean_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['stdDev'].append(std_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['skew'].append(skew_obj_all_modes[i])
                result_dict_eigen_all_modes[o]['kurtosis'].append(kurtosis_obj_all_modes[i])

            with open(uq_path / fr'uq_all_modes.json', 'w') as file:
                file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))


def uq_multicell_s(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
                   proc_keys_list, processor_multicell_nodes, proc_num, shape, solver):
    if solver == 'eigenmode':
        parentDir = solver_args_dict['parentDir']
        projectDir = solver_args_dict['projectDir']
        # cell_type = uq_config['cell_type']
        analysis_folder = solver_args_dict['analysis folder']
        opt = solver_args_dict['optimisation']

        # method = uq_config['method']
        uq_vars = uq_config['variables']
        cell_parameterisation = solver_args_dict['cell_parameterisation']
        err = False
        result_dict_eigen = {}
        Ttab_val_f = []
        Ttab_val_f_all_modes = []

        # eigen_obj_list = objectives
        eigen_obj_list = []

        for o in objectives:
            if o in ["Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]",
                     "G [Ohm]", "Q []", 'kcc [%]', "ff [%]"]:
                result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
                eigen_obj_list.append(o)

        multicell_keys, multicell_values = processor_multicell_nodes.index.to_numpy(), processor_multicell_nodes.to_numpy()
        # print('multicell_keys', multicell_keys)
        for i1, shape in zip(multicell_keys, multicell_values):

            fid = fr'{key}_Q{i1}'

            # check if folder already exist (simulation already completed)

            skip = False
            if os.path.exists(uq_path / f'{fid}/monopole/qois.json'):
                skip = True
                info(f'processor {proc_num} skipped ', fid, 'Result already exists.')

            # skip analysis if folder already exists.
            if not skip:
                solver = ngsolve_mevp
                #  run model using SLANS or CST
                # # create folders for all keys

                if 'tune_config' in uq_config.keys():
                    tune_config = uq_config['tune_config']
                    # tune first before running
                    shape_space = {fid: shape}
                    # save tune results to uq cavity folders
                    sim_folder = os.path.join(analysis_folder, key)

                    tuned_shape_space = run_tune_s_multicell(shape_space, tune_config['parameters'],
                                                             tune_config, projectDir, False, fid, sim_folder)

                    shape = tuned_shape_space[fid]

                solver.createFolder(fid, projectDir, subdir=sub_dir)
                # print("multicell after tuningsdf", shape)
                n_cells = int(len(shape) / (2 * 8))

                solver.cavity_multicell(n_cells, 1, shape,
                                        n_modes=n_cells, fid=fid, f_shift=0,
                                        beampipes='both',
                                        parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

                filename = uq_path / f'{fid}/monopole/qois.json'
                filename_all_modes = uq_path / f'{fid}/monopole/qois_all_modes.json'
                if os.path.exists(filename):
                    qois_result_dict = dict()

                    with open(filename) as json_file:
                        qois_result_dict.update(json.load(json_file))
                    qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)

                    tab_val_f = qois_result
                    Ttab_val_f.append(tab_val_f)
                else:
                    err = True

                # for all modes
                if os.path.exists(filename_all_modes):
                    qois_result_dict_all_modes = dict()
                    qois_result_all_modes = {}

                    with open(filename_all_modes) as json_file:
                        qois_result_dict_all_modes.update(json.load(json_file))

                    tab_val_f_all_modes = []
                    for kk, val in qois_result_dict_all_modes.items():
                        qois_result_all_modes[kk] = get_qoi_value(val, eigen_obj_list)

                        tab_val_f_all_modes.append(qois_result_all_modes[kk])

                    tab_val_f_all_modes_flat = [item for sublist in tab_val_f_all_modes for item in sublist]
                    Ttab_val_f_all_modes.append(tab_val_f_all_modes_flat)
                else:
                    err = True

        data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
        data_table.to_csv(uq_path / fr'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')

        # for all modes
        keys = qois_result_dict_all_modes.keys()
        eigen_obj_list_all_modes = [f"{name.split(' ')[0]}_{i} {name.split(' ', 1)[1]}" for i in keys for name in
                                    eigen_obj_list]
        data_table = pd.DataFrame(Ttab_val_f_all_modes, columns=eigen_obj_list_all_modes)
        data_table.to_csv(uq_path / fr'table_{proc_num}_all_modes.csv', index=False, sep='\t', float_format='%.32f')


def sa_parallel():
    pass


def sa():
    pass


def get_wakefield_objectives_value(d, objectives_unprocessed, abci_data_dir):
    k_loss_array_transverse = []
    k_loss_array_longitudinal = []
    k_loss_M0 = []
    key_list = []

    # create list to hold Z
    Zmax_mon_list = []
    Zmax_dip_list = []
    xmax_mon_list = []
    xmax_dip_list = []
    processed_keys_mon = []
    processed_keys_dip = []

    def calc_k_loss():
        for key, value in d.items():
            abci_data_long = ABCIData(abci_data_dir, key, 0)
            abci_data_trans = ABCIData(abci_data_dir, key, 1)

            # trans
            x, y, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            # long
            x, y, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            abci_data_long.get_data('Loss Factor Spectrum Integrated up to F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            # append only after successful run
            k_loss_M0.append(k_M0)
            k_loss_array_longitudinal.append(k_loss_HOM)
            k_loss_array_transverse.append(k_loss_trans)

        return [k_loss_M0, k_loss_array_longitudinal, k_loss_array_transverse]

    def get_Zmax_L(mon_interval=None):
        if mon_interval is None:
            mon_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                abci_data_mon = ABCIData(abci_data_dir, f"{key}", 0)

                # get longitudinal and transverse impedance plot data
                xr_mon, yr_mon, _ = abci_data_mon.get_data('Real Part of Longitudinal Impedance')
                xi_mon, yi_mon, _ = abci_data_mon.get_data('Imaginary Part of Longitudinal Impedance')

                # Zmax
                if mon_interval is None:
                    mon_interval = [[0.0, 10]]

                # calculate magnitude
                ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]

                # get peaks
                peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
                xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

                for i, z_bound in enumerate(mon_interval):
                    # get mask
                    msk_mon = [(z_bound[0] < x < z_bound[1]) for x in xp_mon]

                    if len(yp_mon[msk_mon]) != 0:
                        Zmax_mon = max(yp_mon[msk_mon])

                        Zmax_mon_list[i].append(Zmax_mon)
                    elif len(yp_mon) != 0:
                        Zmax_mon_list[i].append(0)
                    else:
                        error("skipped, yp_mon = [], raise exception")
                        raise Exception()

                processed_keys_mon.append(key)
            except:
                info("skipped, yp_mon = []")
                # for i, z_bound in enumerate(mon_interval):
                #     Zmax_mon_list[i].append(-1)

        return Zmax_mon_list

    def get_Zmax_T(dip_interval=None):
        if dip_interval is None:
            dip_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                abci_data_dip = ABCIData(abci_data_dir, f"{key}", 1)

                xr_dip, yr_dip, _ = abci_data_dip.get_data('Real Part of Transverse Impedance')
                xi_dip, yi_dip, _ = abci_data_dip.get_data('Imaginary Part of Transverse Impedance')

                # Zmax
                if dip_interval is None:
                    dip_interval = [[0.0, 10]]

                # calculate magnitude
                ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

                # get peaks
                peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
                xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

                for i, z_bound in enumerate(dip_interval):
                    # get mask
                    msk_dip = [(z_bound[0] < x < z_bound[1]) for x in xp_dip]

                    if len(yp_dip[msk_dip]) != 0:
                        Zmax_dip = max(yp_dip[msk_dip])

                        Zmax_dip_list[i].append(Zmax_dip)
                    elif len(yp_dip) != 0:
                        Zmax_dip_list[i].append(0)
                    else:
                        error("skipped, yp_dip = [], raise exception")
                        raise Exception()

                processed_keys_dip.append(key)
            except:
                error("skipped, yp_dip = []")
                # for i, z_bound in enumerate(dip_interval):
                #     Zmax_dip_list[i].append(-1)

        return Zmax_dip_list

    def all(mon_interval, dip_interval):
        for key, value in d.items():
            abci_data_long = ABCIData(abci_data_dir, f"{key}_", 0)
            abci_data_trans = ABCIData(abci_data_dir, f"{key}_", 1)

            # get longitudinal and transverse impedance plot data
            xr_mon, yr_mon, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            xi_mon, yi_mon, _ = abci_data_long.get_data('Imaginary Part of Longitudinal Impedance')

            xr_dip, yr_dip, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            xi_dip, yi_dip, _ = abci_data_trans.get_data('Imaginary Part of Transverse Impedance')

            # loss factors
            # trans
            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            # long
            abci_data_long.get_data('Loss Factor Spectrum Integrated upto F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            # calculate magnitude
            ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]
            ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

            # get peaks
            peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
            xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

            peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
            xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

            for i, z_bound in enumerate(mon_interval):
                # get mask
                msk_mon = [(z_bound[0] < x < z_bound[1]) for x in xp_mon]

                if len(yp_mon[msk_mon]) != 0:
                    Zmax_mon = max(yp_mon[msk_mon])
                    xmax_mon = xp_mon[np.where(yp_mon == Zmax_mon)][0]

                    Zmax_mon_list[i].append(Zmax_mon)
                    xmax_mon_list[i].append(xmax_mon)
                elif len(yp_mon) != 0:
                    Zmax_mon_list[i].append(0.0)
                    xmax_mon_list[i].append(0.0)
                else:
                    continue

            for i, z_bound in enumerate(dip_interval):
                # get mask
                msk_dip = [(z_bound[0] < x < z_bound[1]) for x in xp_dip]

                if len(yp_dip[msk_dip]) != 0:
                    Zmax_dip = max(yp_dip[msk_dip])
                    xmax_dip = xp_dip[np.where(yp_dip == Zmax_dip)][0]

                    Zmax_dip_list[i].append(Zmax_dip)
                    xmax_dip_list[i].append(xmax_dip)
                elif len(yp_dip) != 0:
                    Zmax_dip_list[i].append(0.0)
                    xmax_dip_list[i].append(0.0)
                else:
                    continue

            # append only after successful run

            k_loss_M0.append(k_M0)
            k_loss_array_longitudinal.append(k_loss_HOM)
            k_loss_array_transverse.append(k_loss_trans)

    ZL, ZT = [], []
    df_ZL, df_ZT = pd.DataFrame(), pd.DataFrame()
    for obj in objectives_unprocessed:
        if "ZL" in obj[1]:
            freq_range = process_interval(obj[2])
            for i in range(len(freq_range)):
                Zmax_mon_list.append([])
                xmax_mon_list.append([])
                df_ZL[f"{obj[1]} [max({freq_range[i][0]}<f<{freq_range[i][1]})]"] = 0

            ZL = get_Zmax_L(freq_range)

        elif "ZT" in obj[1]:
            freq_range = process_interval(obj[2])

            for i in range(len(freq_range)):
                Zmax_dip_list.append([])
                xmax_dip_list.append([])
                # df_ZT[obj[1]] = 0
                df_ZT[f"{obj[1]} [max({freq_range[i][0]}<f<{freq_range[i][1]})]"] = 0

            ZT = get_Zmax_T(freq_range)

        elif obj[1] == "k_loss":
            pass
        elif obj[1] == "k_kick":
            pass

    # create dataframes from list
    df_ZL.loc[:, :] = np.array(ZL).T
    df_ZT.loc[:, :] = np.array(ZT).T
    df_ZL['key'] = processed_keys_mon
    df_ZT['key'] = processed_keys_dip

    processed_keys = list(set(processed_keys_mon) & set(processed_keys_dip))

    # ZL, ZT = np.array(ZL).T, np.array(ZT).T

    if len(ZL) != 0 and len(ZT) != 0:
        df_wake = df_ZL.merge(df_ZT, on='key', how='inner')
        # obj_result = np.hstack((ZL, ZT))
    elif len(ZL) != 0:
        df_wake = df_ZL
        # obj_result = ZL
    else:
        df_wake = df_ZT
        # obj_result = ZT

    return df_wake, processed_keys


def process_interval(interval_list):
    interval = []
    for i in range(len(interval_list) - 1):
        interval.append([interval_list[i], interval_list[i + 1]])

    return interval


def get_qois_value(f_fm, R_Q, k_loss, k_kick, sigma_z, I0, Nb, n_cell):
    c = 299792458
    w_fm = 2 * np.pi * f_fm * 1e6
    e = 1.602e-19

    k_fm = (w_fm / 4) * R_Q * np.exp(-(w_fm * sigma_z * 1e-3 / c) ** 2) * 1e-12
    k_hom = k_loss - k_fm
    p_hom = (k_hom * 1e12) * (I0 * 1e-3) * e * (Nb * 1e11)

    d = {
        "n cell": n_cell,
        # "freq [MHz]": f_fm,
        "R/Q [Ohm]": R_Q,
        "k_FM [V/pC]": k_fm,
        "I0 [mA]": I0,
        "sigma_z [mm]": sigma_z,
        "Nb [1e11]": Nb,
        "|k_loss| [V/pC]": k_loss,
        "|k_kick| [V/pC/m]": k_kick,
        "P_HOM [kW]": p_hom * 1e-3
    }
    return d


def process_objectives(objectives):
    processed_objectives = []
    weights = []
    for i, obj in enumerate(objectives):
        if obj[1] == "ZL" or obj[1] == "ZT":
            goal = obj[0]
            freq_ranges = process_interval(obj[2])
            for f in freq_ranges:
                processed_objectives.append([goal, f"{obj[1]} [max({f[0]}<f<{f[1]})]", f])
                weights.append(1)
        else:
            goal = obj[0]
            if goal == 'equal':
                processed_objectives.append(obj)
            else:
                processed_objectives.append(obj)
            weights.append(1)

    return processed_objectives, weights


def show_valid_operating_point_structure():
    dd = {
        '<wp1>': {
            'I0 [mA]': '<value>',
            'Nb [1e11]': '<value>',
            'sigma_z (SR/BS) [mm]': '<value>'
        },
        '<wp2>': {
            'I0 [mA]': '<value>',
            'Nb [1e11]': '<value>',
            'sigma_z (SR/BS) [mm]': '<value>'
        }
    }

    info(dd)


def get_surface_resistance(Eacc, b, m, freq, T):
    Rs_dict = {
        "Rs_NbCu_2K_400.79Mhz": 0.57 * (Eacc * 1e-6 * b) + 28.4,  # nOhm
        "Rs_NbCu_4.5K_400.79Mhz": 39.5 * np.exp(0.014 * (Eacc * 1e-6 * b)) + 27,  # nOhm
        "Rs_bulkNb_2K_400.79Mhz": (2.33 / 1000) * (Eacc * 1e-6 * b) ** 2 + 26.24,  # nOhm
        "Rs_bulkNb_4.5K_400.79Mhz": 0.0123 * (Eacc * 1e-6 * b) ** 2 + 62.53,  # nOhm

        "Rs_NbCu_2K_801.58Mhz": 1.45 * (Eacc * 1e-6 * b) + 92,  # nOhm
        "Rs_NbCu_4.5K_801.58Mhz": 50 * np.exp(0.033 * (Eacc * 1e-6 * b)) + 154,  # nOhm
        "Rs_bulkNb_2K_801.58Mhz": (16.4 + Eacc * 1e-6 * b * 0.092) * (800 / 704) ** 2,  # nOhm
        "Rs_bulkNb_4.5K_801.58Mhz": 4 * (62.7 + (Eacc * 1e-6 * b) ** 2 * 0.012)  # nOhm
    }
    if freq < 600:
        freq = 400.79

    if freq >= 600:
        freq = 801.58

    rs = Rs_dict[fr"Rs_{m}_{T}K_{freq}Mhz"]

    return rs


def axis_data_coords_sys_transform(axis_obj_in, xin, yin, inverse=False):
    """ inverse = False : Axis => Data
                    = True  : Data => Axis
        """
    if axis_obj_in.get_yscale() == 'log':
        xlim = axis_obj_in.get_xlim()
        ylim = axis_obj_in.get_ylim()

        x_delta = xlim[1] - xlim[0]

        if not inverse:
            x_out = xlim[0] + xin * x_delta
            y_out = ylim[0] ** (1 - yin) * ylim[1] ** yin
        else:
            x_delta2 = xin - xlim[0]
            x_out = x_delta2 / x_delta
            y_out = np.log(yin / ylim[0]) / np.log(ylim[1] / ylim[0])

    else:
        xlim = axis_obj_in.get_xlim()
        ylim = axis_obj_in.get_ylim()

        x_delta = xlim[1] - xlim[0]
        y_delta = ylim[1] - ylim[0]

        if not inverse:
            x_out = xlim[0] + xin * x_delta
            y_out = ylim[0] + yin * y_delta
        else:
            x_delta2 = xin - xlim[0]
            y_delta2 = yin - ylim[0]
            x_out = x_delta2 / x_delta
            y_out = y_delta2 / y_delta

    return x_out, y_out


def _get_nodes_and_weights(uq_config, rdim, degree):
    method = uq_config['method']
    uq_vars = uq_config['variables']

    if method[1].lower() == 'stroud3':
        nodes, weights, bpoly = quad_stroud3(rdim, degree)
        nodes = 2. * nodes - 1.
        # nodes, weights = cn_leg_03_1(rdim)
    elif method[1].lower() == 'stroud5':
        nodes, weights = cn_leg_05_2(rdim)
    elif method[1].lower() == 'gaussian':
        nodes, weights = cn_gauss(rdim, 2)
    elif method[1].lower() == 'lhs':
        sampler = qmc.LatinHypercube(d=rdim)
        _ = sampler.reset()
        nsamp = uq_config['integration'][2]
        sample = sampler.random(n=nsamp)

        l_bounds = [-1 for _ in range(len(uq_vars))]
        u_bounds = [1 for _ in range(len(uq_vars))]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)

        nodes, weights = sample_scaled.T, np.ones((nsamp, 1))
    elif method[0].lower() == 'from file':
        if len(method) == 2:
            nodes = pd.read_csv(method[1], sep='\\s+').iloc[:, method[1]]
        else:
            nodes = pd.read_csv(method[1], sep='\\s+')

        nodes = nodes.to_numpy().T
        weights = np.ones((nodes.shape[1], 1))
    else:
        # issue warning
        warning('Integration method not recognised. Defaulting to Stroud3 quadrature rule!')
        nodes, weights, bpoly = quad_stroud3(rdim, degree)
        nodes = 2. * nodes - 1.

    return nodes, weights


def add_text(ax, text, box, xy=(0.5, 0.5), xycoords='data', xytext=None, textcoords='data',
             size=14, rotation=0, arrowprops=None):
    """

    Parameters
    ----------
    ax
    text: str
        Matplotlib annotation text
    box
    xy: tuple
        Coordinates of annotation text
    xycoords: str {data, axis}
        Coordinate system reference
    xytext
    textcoords
    size
    rotation: float
        Annotation text rotation
    arrowprops

    Returns
    -------

    """
    if text.strip("") == "":
        return

    # add text
    if xytext:
        bbox_props = dict(boxstyle='{}'.format(box), fc='w', ec='k')
        annotext = ax.annotate(text, xy=xy, xycoords=xycoords,
                               xytext=xytext, textcoords=textcoords, bbox=bbox_props, fontsize=size,
                               rotation=rotation, arrowprops=arrowprops, zorder=500)
    else:
        if box == "None":
            annotext = ax.annotate(text, xy=xy, xycoords=xycoords, fontsize=size,
                                   rotation=rotation, arrowprops=arrowprops, zorder=500)
        else:
            bbox_props = dict(boxstyle='{}'.format(box), fc='w', ec='k')
            annotext = ax.annotate(text, xy=xy, xycoords=xycoords, bbox=bbox_props, fontsize=size,
                                   rotation=rotation, arrowprops=arrowprops, zorder=500)

    ax.get_figure().canvas.draw_idle()
    ax.get_figure().canvas.flush_events()
    return annotext


def _data_uq_op(d_uq_op):
    # Initialize an empty list to store the rows
    rows = []

    # Iterate over the dictionary
    for main_key, sub_dict in d_uq_op.items():
        row = {'key': main_key}
        for sub_key, metrics in sub_dict.items():
            for metric, value in metrics.items():
                # Check if the metric is one of the desired ones
                if metric in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]']:
                    # Create a new key combining the metric and the sub_key
                    new_key = f"{metric}_{sub_key}"
                    # Add the value to the row
                    row[new_key] = value
        # Add the row to the list
        rows.append(row)

    # Convert the list of rows into a DataFrame
    df = pd.DataFrame(rows)

    return df
