"""Parallel uncertainty quantification process functions."""
import json
import os.path
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.solvers.objectives import objective_polarisations, read_objective_values
from cavsim2d.solvers.NGSolve.eigen_ngsolve import parse_polarisations
from cavsim2d.solvers.eigenmode_result import pol_number
from cavsim2d.utils.shapes import (perturb_half_cells, half_cells_to_dataframe,
                                    perturb_half_cells_independent)
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

ngsolve_mevp = NGSolveMEVP()


def _parse_tune_mode(mode):
    """Parse a ``tune_config['mode']`` spec into ``(m, mode_index)``. Accepts
    ``'monopole:4'`` (polarisation name + 1-based mode number within the band),
    a bare polarisation name/number (fundamental of that band), or ``None`` —
    the default, which is the accelerating pi-mode (monopole fundamental of the
    quarter cell)."""
    if mode is None:
        return 0, 0
    if isinstance(mode, (list, tuple)):
        return pol_number(mode[0]), (int(mode[1]) - 1 if len(mode) > 1 else 0)
    s = str(mode)
    if ':' in s:
        pol, idx = s.split(':', 1)
        return pol_number(pol.strip()), max(int(idx) - 1, 0)
    return pol_number(s.strip()), 0


def _secant_tune_quarter_L(cell, bp, target_freq, save_dir, max_iter=15,
                           tol_mhz=0.02, m=0, mode_index=0, mesh_h=12, mesh_p=3):
    """Secant-tune a single half-cell's length ``L`` so its quarter-cell mode
    (PMC on the equator; a beampipe attached when ``bp``) equals *target_freq*.
    ``(m, mode_index)`` selects which quarter-cell mode is driven to the target
    (default the monopole fundamental — the assembled pi-mode). ``cell`` is the
    8-value ``[A,B,a,b,Ri,L,Req,l]`` array (mm); ``save_dir`` is the half-cell's
    own cavity folder (``cavity_quarter`` fills its ``geometry/`` + ``eigenmode/``).
    Returns ``(tuned_L, convergence, elapsed_s)`` where ``convergence`` is
    ``{'L [mm]': [...], 'freq [MHz]': [...]}`` — the secant iterates. ``tuned_L``
    is None only if the geometry never solved."""
    t0 = time.time()
    L0 = float(cell[5])
    L_list, f_list = [], []

    def freq_at(L):
        cell[5] = L
        f = ngsolve_mevp.cavity_quarter(cell, bp=bp, save_dir=save_dir,
                                        m=m, mode_index=mode_index,
                                        mesh_h=mesh_h, mesh_p=mesh_p)
        if f is not None:
            L_list.append(float(L))
            f_list.append(float(f))
        return f

    La, Lb = L0, L0 * 1.03
    fa, fb = freq_at(La), freq_at(Lb)
    conv = {'L [mm]': L_list, 'freq [MHz]': f_list}
    if fa is None or fb is None:
        return None, conv, time.time() - t0
    for _ in range(max_iter):
        if abs(fb - target_freq) < tol_mhz:
            break
        denom = fb - fa
        if abs(denom) < 1e-9:
            break
        Lc = Lb - (fb - target_freq) * (Lb - La) / denom
        Lc = float(max(0.5 * L0, min(2.0 * L0, Lc)))   # keep the geometry sane
        fc = freq_at(Lc)
        if fc is None:
            break
        La, fa, Lb, fb = Lb, fb, Lc, fc
    return Lb, conv, time.time() - t0


def _tune_half_cells_quarter(half_cells, target_freq, save_root=None,
                             beampipe='both', mode=None, mesh_h=12, mesh_p=3):
    """Per-half-cell quarter-cell tuning (Corno et al. / WEPB015): for every
    half-cell, vary its length ``L`` until its quarter cell resonates at
    *target_freq*. Returns the tuned ``(2*n_cells, 7)`` half-cell array so the
    assembled cavity's fundamental lands on the target with ~zero spread.

    ``mode`` (from ``tune_config['mode']``) selects which quarter-cell mode is
    driven to the target — e.g. ``'monopole:4'`` — defaulting to the accelerating
    pi-mode (monopole fundamental) when ``None``.

    **Beampipe convention**: an end cup carries a beampipe only where the cavity's
    ``beampipe`` actually is — ``'both'`` → the first and last half-cell,
    ``'left'`` → the first only, ``'right'`` → the last only, ``'none'`` → none;
    every other (mid) cup has no beampipe. This mirrors the real assembled cavity.

    Each half-cell is treated as a small cavity of its own: under
    ``save_root/halfcell_<i>/`` it gets ``geometry/`` (the tuned quarter .geo +
    mesh), a ``tune/`` folder (``tune_info.json`` — tuned L, iterations,
    convergence, timing, and ``tune_log.txt``), so its result is inspectable
    later. A ``tune_log.txt`` at ``save_root`` records the per-half-cell time and
    the total. A pure array-in/array-out function so a batch runs in a joblib
    worker (each worker owns its own gmsh)."""
    m_pol, mode_index = _parse_tune_mode(mode)
    hc = np.asarray(half_cells, dtype=float).copy()         # (2*n_cells, 7)
    n = hc.shape[0]
    if save_root is None:
        save_root = os.path.join(tempfile.mkdtemp(), 'multicell_tune')
    os.makedirs(save_root, exist_ok=True)
    bp = str(beampipe).lower()
    t_total = time.time()
    log = [f"multicell quarter-cell tuning - {n} half-cells, "
           f"target {target_freq} MHz, mode={mode or 'monopole (pi)'}, "
           f"cavity beampipe={bp!r}"]
    for row in range(n):
        has_bp = ((row == 0 and bp in ('both', 'left')) or
                  (row == n - 1 and bp in ('both', 'right')))
        cell = [float(v) for v in hc[row]] + [0.0]          # 8-value + l
        # Half-cell as its own cavity: cavity_quarter fills geometry/ + eigenmode/
        # under cell_dir; tune/ holds the tuning record.
        cell_dir = os.path.join(save_root, f'halfcell_{row}')
        tune_dir = os.path.join(cell_dir, 'tune')
        os.makedirs(tune_dir, exist_ok=True)
        tuned_L, conv, elapsed = _secant_tune_quarter_L(
            cell, has_bp, target_freq, cell_dir, m=m_pol, mode_index=mode_index,
            mesh_h=mesh_h, mesh_p=mesh_p)
        if tuned_L is not None and tuned_L > 0:
            hc[row, 5] = tuned_L
        final_f = conv['freq [MHz]'][-1] if conv['freq [MHz]'] else None
        with open(os.path.join(tune_dir, 'tune_info.json'), 'w') as f:
            json.dump({'has_beampipe': bool(has_bp),
                       'target_freq [MHz]': target_freq,
                       'mode': mode or 'monopole:1',
                       'tuned_L [mm]': tuned_L, 'final_freq [MHz]': final_f,
                       'iterations': len(conv['L [mm]']),
                       'time_s': round(elapsed, 3),
                       'convergence': conv}, f, indent=2)
        line = (f"halfcell_{row:<2d} bp={str(has_bp):<5} "
                f"iters={len(conv['L [mm]']):<3d} "
                f"L={tuned_L:.4f}mm  f={final_f:.4f}MHz  {elapsed:.1f}s"
                if final_f is not None else
                f"halfcell_{row:<2d} bp={str(has_bp):<5} FAILED  {elapsed:.1f}s")
        with open(os.path.join(tune_dir, 'tune_log.txt'), 'w') as f:
            f.write(line + '\n')
        log.append('  ' + line)
    log.append(f"  TOTAL: {time.time() - t_total:.1f}s")
    with open(os.path.join(save_root, 'tune_log.txt'), 'w') as f:
        f.write('\n'.join(log) + '\n')
    return hc


def _save_assembled_geometry(cav):
    """Persist the assembled (tuned) UQ-point cavity under ``<self_dir>/geometry/``:
    the boundary contour, the half-cell table, a ``parameters.json`` snapshot, and
    a copy of the solved mesh. A cavity whose half-cells were tuned independently is
    native-only (no ``.geo`` — the gmsh writer expresses uniform mid-cells only), so
    without this the assembled point that was actually solved leaves no inspectable
    geometry record."""
    self_dir = getattr(cav, 'self_dir', None)
    if not self_dir:
        return
    geo_dir = os.path.join(self_dir, 'geometry')
    os.makedirs(geo_dir, exist_ok=True)
    try:
        prof = cav.profile() if callable(getattr(cav, 'profile', None)) else None
        if prof is not None:
            pts = np.asarray(prof.points, dtype=float)
            pd.DataFrame(pts, columns=['z', 'r']).to_csv(
                os.path.join(geo_dir, 'contour.csv'), index=False, sep='\t')
    except Exception:
        pass                                  # contour is advisory
    try:
        hc = np.asarray(cav.half_cells(), dtype=float)
        cols = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        pd.DataFrame(hc, columns=cols[:hc.shape[1]]).to_csv(
            os.path.join(geo_dir, 'half_cells.csv'), index=True, sep='\t')
    except Exception:
        pass
    try:
        cav._write_geometry_snapshot()
    except Exception:
        pass
    eig_dir = getattr(cav, 'eigenmode_dir', None)
    if eig_dir:
        for cand in (os.path.join(eig_dir, 'monopole', 'mesh.pkl'),
                     os.path.join(eig_dir, 'mesh.pkl')):
            if os.path.exists(cand):
                shutil.copy(cand, os.path.join(geo_dir, 'mesh.pkl'))
                break


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

        uq_cfg = eigenmode_config['uq_config']
        multicell = uq_cfg.get('cell_complexity', 'simplecell') == 'multicell'

        if multicell:
            if uq_cfg.get('independent_half_cells'):
                # Paper-style (WEPB015): perturb every half-cell independently,
                # then weld — average — the shared seams back to continuity. Save
                # both so the before/after-continuity spread can be compared.
                perturbed_half_cells, before_hc, weights_ = \
                    perturb_half_cells_independent(cav, eigenmode_config)
                half_cells_to_dataframe(before_hc).to_csv(
                    os.path.join(cav.uq_dir, 'nodes_before_continuity.csv'),
                    index=False, sep='\t', float_format='%.32f')
            else:
                # Every half-cell is an independent random variable (subject to the
                # equator/iris continuity constraints), so UQ covers the whole cavity
                # rather than the mid-cell and end-cell groups separately.
                perturbed_half_cells, weights_ = perturb_half_cells(cav, eigenmode_config)
            nodes_ = half_cells_to_dataframe(perturbed_half_cells)
            nodes_.to_csv(os.path.join(cav.uq_dir, 'nodes.csv'), index=False,
                          sep='\t', float_format='%.32f')
            cavs_object = cav.spawn_half_cells(perturbed_half_cells,
                                               os.path.join(cav.self_dir, 'uq'))
        else:
            perturbed_cavities, weights_ = perturb_geometry(cav, eigenmode_config)

            nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
            # save nodes
            nodes_perturbed.to_csv(os.path.join(cav.uq_dir, 'nodes_before_continuity.csv'), index=False, sep='\t', float_format='%.32f')

            # enforce continuity
            # nodes_ = enforce_continuity_df(nodes_perturbed)
            nodes_ = nodes_perturbed
            # save nodes
            nodes_.to_csv(os.path.join(cav.uq_dir, 'nodes.csv'), index=False, sep='\t', float_format='%.32f')

            # spawn cavity objects
            cavs_object = cav.spawn(nodes_, os.path.join(cav.self_dir, 'uq'))
        
        # Save UQ config for traceability
        with open(os.path.join(cav.uq_dir, 'uq_config.json'), 'w') as f:
            json.dump(eigenmode_config, f, indent=4, default=str)

        objectives = uq_cfg['objectives']
        # Eigenmode config for the perturbed variants (strip the nested uq_config
        # so we don't recurse). The variants are independent, so they solve in
        # parallel across ``uq_config['processes']`` — NOT the outer run's
        # ``processes`` (which is 1 for the single nominal cavity). An objective
        # naming e.g. 'dipole:...' needs that polarisation solved.
        uq_processes = uq_cfg.get('processes')
        eig_cfg = {k: v for k, v in eigenmode_config.items() if k != 'uq_config'}
        if uq_processes:
            eig_cfg['processes'] = uq_processes
        eig_cfg['polarisation'] = sorted(
            set(parse_polarisations(eig_cfg.get('polarisation', 0)))
            | set(objective_polarisations(objectives)))

        if 'tune_config' in uq_cfg and uq_cfg['tune_config']:
            # Paper workflow: retune each perturbed variant to the target frequency
            # and THEN solve the eigenmode on the tuned geometry, so all objectives
            # — not just the tuning frequency — are measured on the tuned cavity.
            freqs = uq_cfg['tune_config']['freqs']
            target_freq = float(freqs[0] if isinstance(freqs, (list, tuple, np.ndarray))
                                else freqs)
            if multicell:
                # Tune EACH half-cell's L to the target via its quarter cell (end
                # cups with a beampipe, mid cups without) so the assembled
                # fundamental lands on the target with ~zero spread — what a
                # single-L full-cavity tune cannot do. Each cavity's tuned quarter
                # geometry + mesh are kept under its ``multicell_tune/`` dir. Then
                # solve the eigenmode on the tuned assemblies.
                #
                # The perturbed cavities are independent, so the tunes run in
                # parallel across ``uq_config['processes']`` — one cavity per task,
                # each worker owning its own gmsh (loky spawns a fresh process, so
                # no shared gmsh state). Results come back in task order.
                # Which quarter-cell mode to tune to the target (default pi-mode),
                # and the (coarser-than-eigenmode) tuning mesh — a quarter cell is
                # over-resolved at h=20, so h=12 is the same frequency ~30% faster.
                tune_mode = uq_cfg['tune_config'].get('mode')
                mesh_h = int(uq_cfg['tune_config'].get('mesh_h', 12))
                mesh_p = int(uq_cfg['tune_config'].get('mesh_p', 3))
                cavlist = cavs_object.cavities_list
                specs = [(np.asarray(scav.half_cells(), dtype=float),
                          os.path.join(scav.self_dir, 'multicell_tune'),
                          getattr(scav, 'beampipe', 'both'))
                         for scav in cavlist]
                nproc = min(int(uq_processes or 1), len(cavlist))
                if nproc > 1:
                    # Deferred: joblib is only pulled in for parallel runs.
                    from joblib import Parallel, delayed
                    tuned_hcs = Parallel(n_jobs=nproc, backend='loky')(
                        delayed(_tune_half_cells_quarter)(hc, target_freq, sr, bp,
                                                          tune_mode, mesh_h, mesh_p)
                        for hc, sr, bp in specs)
                else:
                    tuned_hcs = [_tune_half_cells_quarter(hc, target_freq, sr, bp,
                                                          tune_mode, mesh_h, mesh_p)
                                 for hc, sr, bp in specs]
                for scav, hc in zip(cavlist, tuned_hcs):
                    scav.set_half_cells(hc)
                cavs_object.run_eigenmode(eig_cfg)
                # Persist the assembled (tuned) geometry of each UQ point — it is
                # native-only, so run_eigenmode leaves no .geo behind.
                for scav in cavlist:
                    _save_assembled_geometry(scav)
                solved_object = cavs_object
            else:
                # Simplecell: retune the whole cavity via the standard tuner, then
                # solve on the tuned cavities (all-modes QOIs come from the solved
                # study).
                tune_cfg = dict(uq_cfg['tune_config'])
                if uq_processes and 'processes' not in tune_cfg:
                    tune_cfg['processes'] = uq_processes
                cavs_object.run_tune(tune_cfg)
                solved_object = cavs_object.tuned
                solved_object.run_eigenmode(eig_cfg)
        else:
            # Plain UQ: run eigenmode on each perturbed variant directly.
            cavs_object.run_eigenmode(eig_cfg)
            solved_object = cavs_object

        # The QOI table is built from the objectives, which are polarisation-
        # qualified: monopole and m-pole qois.json share key names, so an
        # unqualified 'R/Q [Ohm]' would be ambiguous. read_objective_values raises
        # on an unknown objective rather than silently dropping it.
        rows = {}
        for scav in cavs_object.cavities_list:
            tuned = getattr(scav, 'tuned', None)
            eig_dir = tuned.eigenmode_dir if tuned is not None else scav.eigenmode_dir
            rows[scav.name] = read_objective_values(eig_dir, objectives)
        uq_df = pd.DataFrame.from_dict(rows).T
        uq_df = uq_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')

        records = []
        for outer_key, mid_dict in solved_object.eigenmode_qois_all_modes.items():
            row = {}
            for mid_key, inner_dict in mid_dict.items():
                for k, v in inner_dict.items():
                    # create column name combining key numbers
                    m = re.search(r"([\w\s/\-()]+?)\s*(\[.*?\])?$", k)
                    if m:
                        name = m.group(1).strip()  # e.g. "GR/Q"
                        unit = m.group(2) or ""  # e.g. "[Ohm^2]" or "" if none
                        col_name = f"{name}_{mid_key} {unit}".strip()
                    else:
                        col_name = f"{k}_{mid_key}"
                    row[col_name] = v
            # include the outer key as index
            row['outer_index'] = outer_key
            records.append(row)

        # Create DataFrame
        uq_df_all = pd.DataFrame(records)
        uq_df_all.set_index('outer_index', inplace=True)

        data_table_w = pd.DataFrame(weights_, columns=['weights'])
        data_table_w.to_csv(os.path.join(cav.uq_dir, 'weights.csv'),
                            index=False, sep='\t',
                            float_format='%.32f')

        uq_df.to_csv(os.path.join(cav.uq_dir, 'table.csv'), index=False, sep='\t', float_format='%.32f')
        uq_df.to_excel(os.path.join(cav.uq_dir, 'table.xlsx'), index=False)

        Ttab_val_f = uq_df.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(Ttab_val_f, weights_)

        for i, o in enumerate(uq_df.columns):
            result_dict_eigen[o] = {'expe': [], 'stdDev': [], 'skew': [], 'kurtosis': []}
            result_dict_eigen[o]['expe'].append(mean_obj[i])
            result_dict_eigen[o]['stdDev'].append(std_obj[i])
            result_dict_eigen[o]['skew'].append(skew_obj[i])
            result_dict_eigen[o]['kurtosis'].append(kurtosis_obj[i])
        with open(os.path.join(cav.uq_dir, fr'uq.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen, indent=4, separators=(',', ': ')))

        # for all modes
        uq_df_all.to_csv(os.path.join(cav.uq_dir, 'table_all_modes.csv'), index=False, sep='\t', float_format='%.32f')
        uq_df_all.to_excel(os.path.join(cav.uq_dir, 'table_all_modes.xlsx'), index=False)

        # Numeric-only, as above (drop 'polarisation' and any other text metadata).
        uq_df_all = uq_df_all.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
        Ttab_val_f_all_modes = uq_df_all.to_numpy()

        mean_obj_all_modes, std_obj_all_modes, skew_obj_all_modes, kurtosis_obj_all_modes = weighted_mean_obj(
            Ttab_val_f_all_modes, weights_)

        for i, o in enumerate(uq_df_all.columns):
            # Extract the mode key and base property name from a column like
            # 'freq_0 [MHz]' or, with the unified all-modes schema, 'freq_0-1 [MHz]'
            # where the mode key is "<mode index>-<azimuthal m>".
            match = re.search(r'^(.*?)_(\d+-\d+|\d+)(\s*\[.*\])?$', o)
            if match:
                prop_base = match.group(1).strip()
                mode_idx = match.group(2)
                unit = match.group(3).strip() if match.group(3) else ""
                base_name = f"{prop_base} {unit}".strip()
            else:
                base_name = o
                mode_idx = "0"

            if mode_idx not in result_dict_eigen_all_modes:
                result_dict_eigen_all_modes[mode_idx] = {}

            result_dict_eigen_all_modes[mode_idx][base_name] = {
                'expe': [mean_obj_all_modes[i]],
                'stdDev': [std_obj_all_modes[i]],
                'skew': [skew_obj_all_modes[i]],
                'kurtosis': [kurtosis_obj_all_modes[i]]
            }

        with open(os.path.join(cav.uq_dir, 'uq_all_modes.json'), 'w') as file:
            file.write(json.dumps(result_dict_eigen_all_modes, indent=4, separators=(',', ': ')))

    elif solver == 'wakefield':
        # ``eigenmode_config`` is actually the wakefield_config here (callers
        # reuse the same parameter name across solvers). We alias it for
        # readability.
        wakefield_config = eigenmode_config
        uq_cfg = wakefield_config['uq_config']

        if not os.path.exists(cav.uq_dir):
            os.mkdir(cav.uq_dir)

        perturbed_cavities, weights_ = perturb_geometry(cav, wakefield_config)

        nodes_perturbed = shapes_to_dataframe(perturbed_cavities)
        nodes_perturbed.to_csv(os.path.join(cav.uq_dir, 'nodes_before_continuity.csv'),
                               index=False, sep='\t', float_format='%.32f')

        nodes_ = nodes_perturbed
        nodes_.to_csv(os.path.join(cav.uq_dir, 'nodes.csv'),
                      index=False, sep='\t', float_format='%.32f')

        cavs_object = cav.spawn(nodes_, os.path.join(cav.self_dir, 'uq'))

        with open(os.path.join(cav.uq_dir, 'uq_config.json'), 'w') as f:
            json.dump(wakefield_config, f, indent=4, default=str)

        # Run wakefield on the perturbed variants. Strip the nested uq_config
        # so we don't recurse into another UQ sweep.
        wk_cfg = {k: v for k, v in wakefield_config.items() if k != 'uq_config'}
        cavs_object.run_wakefield(wk_cfg)

        # Read ZL/ZT maxima per perturbed cavity via the same helper the
        # optimiser uses. Normalise objectives to the [goal, name, arg] form.
        # Deferred: breaks the uq <-> wakefield import cycle.
        from cavsim2d.processes.wakefield import get_wakefield_objectives_value

        objectives_unprocessed = uq_cfg.get('objectives_unprocessed',
                                            uq_cfg.get('objectives', []))
        norm_objs = []
        for obj in objectives_unprocessed:
            if isinstance(obj, list) and len(obj) == 2:
                norm_objs.append(['', obj[0], obj[1]])
            else:
                norm_objs.append(obj)

        spawn_names = [cav_.name for cav_ in cavs_object.cavities_list]
        d = {name: None for name in spawn_names}
        # Read objectives through the SAME backend the perturbed variants were
        # solved with (default 'abci'); otherwise a non-ABCI solver's frames are
        # unreadable and every objective comes back empty.
        df_wake, _ = get_wakefield_objectives_value(
            d, norm_objs, Path(cavs_object.projectDir),
            solver=wakefield_config.get('solver', 'abci'))

        if 'key' in df_wake.columns:
            df_wake = df_wake.set_index('key')

        # Preserve the perturbation order so rows align with ``weights_``.
        df_wake = df_wake.reindex(spawn_names)

        n_failed = int(df_wake.isna().any(axis=1).sum())
        if n_failed > 0:
            error(f"UQ wakefield: {n_failed} of {len(df_wake)} perturbed cavities "
                  f"failed to produce ABCI impedance output; skipping uq.json "
                  f"for {cav.name}.")
            return

        df_wake.to_csv(os.path.join(cav.uq_dir, 'table.csv'), sep='\t',
                       float_format='%.32f')
        df_wake.to_excel(os.path.join(cav.uq_dir, 'table.xlsx'))

        data_table_w = pd.DataFrame(weights_, columns=['weights'])
        data_table_w.to_csv(os.path.join(cav.uq_dir, 'weights.csv'),
                            index=False, sep='\t', float_format='%.32f')

        Ttab_val_f = df_wake.to_numpy()
        mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(
            Ttab_val_f, weights_)

        result_dict_wake = {}
        for i, o in enumerate(df_wake.columns):
            result_dict_wake[o] = {
                'expe': [mean_obj[i]],
                'stdDev': [std_obj[i]],
                'skew': [skew_obj[i]],
                'kurtosis': [kurtosis_obj[i]],
            }

        # Write to ``<cav>/uq/uq.json`` — the same file the eigenmode branch
        # writes and the exact file the optimiser reads back
        # (``filename_abci = uq_json`` -> ``<cav>/uq/uq.json`` in
        # ``Optimisation._evaluate_generation``) and the QOI-spread plots load.
        # ``cav.uq_dir`` is created above; ``cav.wakefield_dir`` (the old target)
        # need not exist here, which is why this used to raise FileNotFoundError.
        with open(os.path.join(cav.uq_dir, 'uq.json'), 'w') as f:
            json.dump(result_dict_wake, f, indent=4, separators=(',', ': '))


# `uq(proc_cavs_dict, ...)` was removed 2026-07-09: it had no callers, and it filtered
# objectives against a hardcoded whitelist, silently dropping anything not on it.
# Objectives are now polarisation-qualified and validated (cavsim2d.solvers.objectives).

def sa_parallel():
    pass


def sa():
    pass



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
