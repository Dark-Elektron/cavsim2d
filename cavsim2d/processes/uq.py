"""Parallel uncertainty quantification process functions."""
import json
import os.path
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.solvers.objectives import objective_polarisations, read_objective_values
from cavsim2d.solvers.NGSolve.eigen_ngsolve import parse_polarisations
from cavsim2d.utils.shapes import perturb_half_cells, half_cells_to_dataframe
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *

ngsolve_mevp = NGSolveMEVP()

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
        if 'tune_config' in uq_cfg and uq_cfg['tune_config']:
            # Refit each perturbed variant (e.g. retune Req) before measuring
            # QoIs, then pull QoIs off the tuned cavities.
            cavs_object.run_tune(uq_cfg['tune_config'])
        else:
            # Plain UQ: run eigenmode on each perturbed variant. Strip the
            # nested uq_config so we don't recurse into another UQ sweep.
            eig_cfg = {k: v for k, v in eigenmode_config.items() if k != 'uq_config'}
            # An objective naming e.g. 'dipole:...' needs that polarisation solved.
            eig_cfg['polarisation'] = sorted(
                set(parse_polarisations(eig_cfg.get('polarisation', 0)))
                | set(objective_polarisations(objectives)))
            cavs_object.run_eigenmode(eig_cfg)

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
        for outer_key, mid_dict in cavs_object.eigenmode_qois_all_modes.items():
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
