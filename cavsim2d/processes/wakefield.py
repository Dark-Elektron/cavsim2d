"""Parallel wakefield analysis process functions."""
import multiprocessing as mp
import math
import json
import os.path
import shutil
import time

import numpy as np
import pandas as pd
import scipy.signal as sps
from cavsim2d.analysis.wakefield.abci_geometry import ABCIGeometry
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.processes.uq import uq_parallel_multicell, uq_parallel
from cavsim2d.solvers.ABCI.abci import ABCI
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

abci = ABCI()
abci_geom = ABCIGeometry()


def run_wakefield_parallel(cavs_dict, solver_config, subdir=''):

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


def run_wakefield_s(cavs_dict, wakefield_config, subdir):
    rerun = wakefield_config.get('rerun', True)
    if 'rerun' in wakefield_config:
        assert isinstance(wakefield_config['rerun'], bool), error('rerun must be boolean.')

    operating_points = wakefield_config.get('operating_points')
    uq_config = wakefield_config['uq_config']

    def _run_abci(cav, wakefield_config):
        freq = 0
        R_Q = 0
        start_time = time.time()

        cav.geo_to_abc(wakefield_config)
        abci.solve(cav, wakefield_config)
        done(f'Cavity {cav.name}. Time: {time.time() - start_time}')

        if 'uq_config' in wakefield_config:
            uq_config = wakefield_config['uq_config']
            uq_cell_complexity = uq_config.get('cell_complexity', 'simplecell')

            if uq_cell_complexity == 'multicell':
                uq_parallel_multicell(cav, wakefield_config, 'wakefield')
            else:
                uq_parallel(cav, wakefield_config, 'wakefield')

        if operating_points:
            try:
                folder = os.path.join(cav.eigenmode_dir, 'qois.json')
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
                    for key_op, vals in operating_points.items():
                        WP = key_op
                        I0 = float(vals['I0 [mA]'])
                        Nb = float(vals['Nb [1e11]'])
                        sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                        bl_diff = ['SR', 'BS']

                        for i, s in enumerate(sigma_z):
                            fid = f"{WP}_{bl_diff[i]}_{s}mm"
                            wakefield_folder_structure = {fid: {'wakefield': {'longitudinal': None, 'transversal': None}}}
                            make_dirs_from_dict(wakefield_folder_structure, os.path.join(cav.self_dir, 'wakefield'))

                            cav.geo_to_abc(wakefield_config, os.path.join(cav.self_dir, 'wakefield', fid))
                            abci.solve(cav, wakefield_config, os.path.join(fid, 'wakefield'))

                            dirc = os.path.join(cav.self_dir, "wakefield")
                            k_loss = abs(ABCIData(dirc, os.path.join(fid, 'wakefield'), 0).loss_factor['Longitudinal'])
                            k_kick = abs(ABCIData(dirc, os.path.join(fid, 'wakefield'), 1).loss_factor['Transverse'])

                            d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, cav.n_cells)

                    print(d)
                    run_save_directory = os.path.join(cav.wakefield_dir)
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The operating point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()

    for i, (key, cav) in enumerate(cavs_dict.items()):
        if os.path.exists(cav.self_dir):
            if rerun:
                wake_long = os.path.join(cav.self_dir, "wakefield", "longitudinal")
                if os.path.exists(wake_long):
                    shutil.rmtree(wake_long)
                _run_abci(cav, wakefield_config)
            else:
                if os.path.exists(os.path.join(cav.self_dir, 'wakefield', "longitudinal", "qois.json")):
                    pass
                else:
                    _run_abci(cav, wakefield_config)
        else:
            _run_abci(cav, wakefield_config)


def get_wakefield_objectives_value(d, objectives_unprocessed, abci_data_dir, key_subdir=''):
    """Read ABCI results and compute wakefield objectives.

    ``key_subdir`` is inserted between <key> and 'wakefield' when building the
    ABCI fid path. Pass ``'tuned'`` when the wakefield ran on the tuned
    cavity (results live at <projectDir>/<key>/tuned/wakefield/).
    """
    k_loss_array_transverse = []
    k_loss_array_longitudinal = []
    k_loss_M0 = []

    Zmax_mon_list = []
    Zmax_dip_list = []
    xmax_mon_list = []
    xmax_dip_list = []
    processed_keys_mon = []
    processed_keys_dip = []

    def _fid(key):
        if key_subdir:
            return os.path.join(key, key_subdir, 'wakefield')
        return os.path.join(key, 'wakefield')

    def calc_k_loss():
        for key, value in d.items():
            fid = _fid(key)
            abci_data_long = ABCIData(abci_data_dir, fid, 0)
            abci_data_trans = ABCIData(abci_data_dir, fid, 1)

            x, y, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            x, y, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            abci_data_long.get_data('Loss Factor Spectrum Integrated up to F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            k_loss_M0.append(k_M0)
            k_loss_array_longitudinal.append(k_loss_HOM)
            k_loss_array_transverse.append(k_loss_trans)

        return [k_loss_M0, k_loss_array_longitudinal, k_loss_array_transverse]

    def get_Zmax_L(mon_interval=None):
        if mon_interval is None:
            mon_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                fid = _fid(key)
                abci_data_mon = ABCIData(abci_data_dir, fid, 0)

                xr_mon, yr_mon, _ = abci_data_mon.get_data('Real Part of Longitudinal Impedance')
                xi_mon, yi_mon, _ = abci_data_mon.get_data('Imaginary Part of Longitudinal Impedance')

                if mon_interval is None:
                    mon_interval = [[0.0, 10]]

                ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]

                peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
                xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

                for i, z_bound in enumerate(mon_interval):
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

        return Zmax_mon_list

    def get_Zmax_T(dip_interval=None):
        if dip_interval is None:
            dip_interval = [0.0, 2e10]

        for key, value in d.items():
            try:
                fid = _fid(key)
                abci_data_dip = ABCIData(abci_data_dir, fid, 1)

                xr_dip, yr_dip, _ = abci_data_dip.get_data('Real Part of Transverse Impedance')
                xi_dip, yi_dip, _ = abci_data_dip.get_data('Imaginary Part of Transverse Impedance')

                if dip_interval is None:
                    dip_interval = [[0.0, 10]]

                ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

                peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
                xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

                for i, z_bound in enumerate(dip_interval):
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

        return Zmax_dip_list

    def all(mon_interval, dip_interval):
        for key, value in d.items():
            fid = _fid(key)
            abci_data_long = ABCIData(abci_data_dir, fid, 0)
            abci_data_trans = ABCIData(abci_data_dir, fid, 1)

            xr_mon, yr_mon, _ = abci_data_long.get_data('Real Part of Longitudinal Impedance')
            xi_mon, yi_mon, _ = abci_data_long.get_data('Imaginary Part of Longitudinal Impedance')

            xr_dip, yr_dip, _ = abci_data_trans.get_data('Real Part of Transverse Impedance')
            xi_dip, yi_dip, _ = abci_data_trans.get_data('Imaginary Part of Transverse Impedance')

            k_loss_trans = abci_data_trans.loss_factor['Transverse']

            if math.isnan(k_loss_trans):
                error(f"Encountered an exception: Check shape {key}")
                continue

            abci_data_long.get_data('Loss Factor Spectrum Integrated upto F')

            k_M0 = abci_data_long.y_peaks[0]
            k_loss_long = abs(abci_data_long.loss_factor['Longitudinal'])
            k_loss_HOM = k_loss_long - k_M0

            ymag_mon = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_mon, yi_mon)]
            ymag_dip = [(a ** 2 + b ** 2) ** 0.5 for a, b in zip(yr_dip, yi_dip)]

            peaks_mon, _ = sps.find_peaks(ymag_mon, height=0)
            xp_mon, yp_mon = np.array(xr_mon)[peaks_mon], np.array(ymag_mon)[peaks_mon]

            peaks_dip, _ = sps.find_peaks(ymag_dip, height=0)
            xp_dip, yp_dip = np.array(xr_dip)[peaks_dip], np.array(ymag_dip)[peaks_dip]

            for i, z_bound in enumerate(mon_interval):
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
                df_ZT[f"{obj[1]} [max({freq_range[i][0]}<f<{freq_range[i][1]})]"] = 0

            ZT = get_Zmax_T(freq_range)

        elif obj[1] == "k_loss":
            pass
        elif obj[1] == "k_kick":
            pass

    df_ZL.loc[:, :] = np.array(ZL).T
    df_ZT.loc[:, :] = np.array(ZT).T
    df_ZL['key'] = processed_keys_mon
    df_ZT['key'] = processed_keys_dip

    processed_keys = list(set(processed_keys_mon) & set(processed_keys_dip))

    if len(ZL) != 0 and len(ZT) != 0:
        df_wake = df_ZL.merge(df_ZT, on='key', how='inner')
    elif len(ZL) != 0:
        df_wake = df_ZL
    else:
        df_wake = df_ZT

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
            processed_objectives.append(obj)
            weights.append(1)

    return processed_objectives, weights


def show_valid_operating_point_structure():
    dd = """{
        '<wp1>': {
            'I0 [mA]': <value>,
            'Nb [1e11]': <value>,
            'sigma_z (SR/BS) [mm]': <value>
        },
        '<wp2>': {
            'I0 [mA]': <value>,
            'Nb [1e11]': <value>,
            'sigma_z (SR/BS) [mm]': <value>
        }
    }"""
    info(dd)


def get_surface_resistance(Eacc, b, m, freq, T):
    Rs_dict = {
        "Rs_NbCu_2K_400.79Mhz": 0.57 * (Eacc * 1e-6 * b) + 28.4,
        "Rs_NbCu_4.5K_400.79Mhz": 39.5 * np.exp(0.014 * (Eacc * 1e-6 * b)) + 27,
        "Rs_bulkNb_2K_400.79Mhz": (2.33 / 1000) * (Eacc * 1e-6 * b) ** 2 + 26.24,
        "Rs_bulkNb_4.5K_400.79Mhz": 0.0123 * (Eacc * 1e-6 * b) ** 2 + 62.53,
        "Rs_NbCu_2K_801.58Mhz": 1.45 * (Eacc * 1e-6 * b) + 92,
        "Rs_NbCu_4.5K_801.58Mhz": 50 * np.exp(0.033 * (Eacc * 1e-6 * b)) + 154,
        "Rs_bulkNb_2K_801.58Mhz": (16.4 + Eacc * 1e-6 * b * 0.092) * (800 / 704) ** 2,
        "Rs_bulkNb_4.5K_801.58Mhz": 4 * (62.7 + (Eacc * 1e-6 * b) ** 2 * 0.012)
    }
    if freq < 600:
        freq = 400.79
    if freq >= 600:
        freq = 801.58

    return Rs_dict[fr"Rs_{m}_{T}K_{freq}Mhz"]


def axis_data_coords_sys_transform(axis_obj_in, xin, yin, inverse=False):
    """Transform between axis and data coordinate systems."""
    if axis_obj_in.get_yscale() == 'log':
        xlim = axis_obj_in.get_xlim()
        ylim = axis_obj_in.get_ylim()
        x_delta = xlim[1] - xlim[0]

        if not inverse:
            x_out = xlim[0] + xin * x_delta
            y_out = ylim[0] ** (1 - yin) * ylim[1] ** yin
        else:
            x_out = (xin - xlim[0]) / x_delta
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
            x_out = (xin - xlim[0]) / x_delta
            y_out = (yin - ylim[0]) / y_delta

    return x_out, y_out
