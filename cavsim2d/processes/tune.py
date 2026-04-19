"""Parallel tuning process functions."""
import json
import multiprocessing as mp
import os.path
import shutil
import time
from pathlib import Path

import numpy as np
from cavsim2d.analysis.tune.tuner import Tuner
from cavsim2d.constants import *
from cavsim2d.processes.eigenmode import run_eigenmode_parallel, run_eigenmode_s  # noqa: F401 — used by multicell path
from cavsim2d.utils.shared_functions import *
from cavsim2d.utils.printing import suppress_errors

tuner = Tuner()


# -- New-format canonicalisation ---------------------------------------------

_SUFFIX_FOR_CT = {
    'mid cell': '_m',
    'end cell': '_el',
    'end cell l': '_el',
    'end cell left': '_el',
    'end cell r': '_er',
    'end cell right': '_er',
    'single cell': '_el',
}

# Parameters that are physically shared across all cells of a multi-cell
# cavity — their value must stay identical on `_m`/`_el`/`_er` regardless
# of the pre-tune values. Tuning Req_m therefore *always* propagates into
# Req_el/Req_er (unless those are scheduled for a later tune stage), because
# later end-cell stages use a hybrid geometry [end-cell + adjacent mid-cell
# half] that requires Req_m == Req_el for a continuous equator.
_ALWAYS_PROPAGATE_BARE = {'Req'}

_VALID_BARE_VARS = {'A', 'B', 'a', 'b', 'Ri', 'L', 'Req', 'l'}
_VALID_SUFFIXES = ('_m', '_el', '_er')


def _normalize_ct_name(ct):
    return ct.lower().replace('-', ' ').replace('_', ' ')


def normalize_cell_type_config(tune_config):
    """Return ordered dict {cell_type: [tune_vars]} from ``tune_config``.

    Accepts the new form ``tune_config['cell_type'] = {ct: var or [vars]}``
    or the legacy form ``tune_config['parameters'] + tune_config['cell_types']``
    (emitting a DeprecationWarning).
    """
    if 'cell_type' in tune_config:
        raw = tune_config['cell_type']
        if not isinstance(raw, dict):
            raise TypeError(
                f"tune_config['cell_type'] must be a dict mapping "
                f"cell-type names to tune variables. Got {type(raw).__name__}.")
        out = {}
        for ct, vars_ in raw.items():
            if isinstance(vars_, str):
                vars_ = [vars_]
            out[ct] = list(vars_)
        return out

    # Legacy path: parameters + cell_types.
    if 'parameters' in tune_config and 'cell_types' in tune_config:
        warning(
            "tune_config: the 'parameters' and 'cell_types' keys are "
            "deprecated; please use the single 'cell_type' dict "
            "(e.g. {'mid-cell': 'Req'}).")
        params = tune_config['parameters']
        cts = tune_config['cell_types']
        # In the legacy API params/cts could be per-cavity lists OR scalars.
        # For the new-format the dict is shared across cavities. If lists are
        # passed, use the first entry (they had to be identical in practice
        # for a uniform tune) or fall back to zip mapping a single cavity.
        if isinstance(params, (list, tuple, np.ndarray)):
            params = list(params)
        else:
            params = [params]
        if isinstance(cts, (list, tuple, np.ndarray)):
            cts = list(cts)
        else:
            cts = [cts]
        # Collapse identical entries; otherwise emit a second warning and
        # keep only the first stage.
        if len(set(map(str, params))) > 1 or len(set(map(str, cts))) > 1:
            warning(
                "Legacy per-cavity heterogeneous tune config detected; "
                "only the first entry will be used. Split into multiple "
                "run_tune calls for per-cavity tuning.")
        return {cts[0]: [params[0]]}

    raise ValueError(
        "tune_config must contain 'cell_type' (new API) or both "
        "'parameters' and 'cell_types' (legacy API).")


def _resolve_suffixed_var(var, ct):
    """Map a bare tune variable and cell-type to its suffixed parameter name."""
    if var.endswith(_VALID_SUFFIXES):
        return var
    bare = var.split('_')[0] if '_' in var else var
    if bare not in _VALID_BARE_VARS:
        raise ValueError(f"Unknown tune variable {var!r}.")
    suffix = _SUFFIX_FOR_CT.get(_normalize_ct_name(ct), '_m')
    return f'{bare}{suffix}'


def last_stage_result(tune_res):
    """Return the consolidated tuned-cavity entry from a keyed ``tune_res.json``.

    Per-stage entries now only carry the parameters for the cell that
    was tuned in that stage (e.g. mid-cell stage stores only ``*_m``).
    This function merges parameters across all stages — so the returned
    entry has the full suffixed parameter dict needed to rebuild a
    complete cavity — while taking ``FREQ`` / ``TUNED VARIABLES`` from
    the final stage. Legacy flat dicts are passed through unchanged.
    """
    if not isinstance(tune_res, dict) or not tune_res:
        return None
    # Legacy flat dict carried ``FREQ`` / ``parameters`` at the top level.
    if 'FREQ' in tune_res and 'parameters' in tune_res:
        return tune_res
    stages = list(tune_res.values())
    if not stages:
        return None

    merged_params = {}
    for stage in stages:
        if isinstance(stage, dict):
            merged_params.update(stage.get('parameters', {}))

    last = stages[-1] if isinstance(stages[-1], dict) else {}
    return {
        'parameters': merged_params,
        'FREQ': last.get('FREQ'),
        'TUNED VARIABLES': last.get('TUNED VARIABLES', []),
    }


def _collect_scheduled_suffixed_vars(cell_type_config):
    """Return set of suffixed parameter names explicitly tuned by any stage."""
    scheduled = set()
    for ct, vars_ in cell_type_config.items():
        for v in vars_:
            scheduled.add(_resolve_suffixed_var(v, ct))
    return scheduled


def _filter_params_for_ct(params, ct_name):
    """Return only the parameters belonging to ``ct_name``'s cell.

    Used to keep per-stage ``tune_results`` snapshots focused on the
    parameters that were actually tuned in that stage, instead of
    repeating the full `_m`/`_el`/`_er` dict (which still carries the
    untouched values from other cells and is noisy to read).
    """
    suffix = _SUFFIX_FOR_CT.get(_normalize_ct_name(ct_name))
    if not suffix:
        return dict(params)
    return {k: v for k, v in params.items() if k.endswith(suffix)}


def run_tune_parallel(cavs_dict, tune_config, solver='NGSolveMEVP',
                      resume=False):
    tune_config_keys = tune_config.keys()
    if 'processes' in tune_config_keys:
        processes = tune_config['processes']
        assert processes > 0, error('Number of processes must be greater than zero.')
    else:
        processes = 1

    assert 'freqs' in tune_config_keys, error('Please enter the target tune "freqs" in tune_config.')

    # Canonicalise cell_type config (handles legacy parameters+cell_types).
    cell_type_config = normalize_cell_type_config(tune_config)
    tune_config = dict(tune_config)  # shallow copy so we don't mutate caller
    tune_config['cell_type'] = cell_type_config

    freqs = tune_config['freqs']
    if isinstance(freqs, (float, int)):
        freqs = np.array([freqs for _ in range(len(cavs_dict))], dtype=float)
    else:
        assert len(freqs) == len(cavs_dict), error(
            'Number of target frequencies must correspond to the number of cavities')
        freqs = np.asarray(freqs, dtype=float)
    tune_config['freqs'] = freqs

    # split shape_space for different processes/ MPI share process by rank
    keys = list(cavs_dict.keys())

    if processes > len(keys):
        processes = len(keys)

    shape_space_len = len(keys)
    jobs = []

    base_chunk_size = shape_space_len // processes
    remainder = shape_space_len % processes

    start_idx = 0
    for p in range(processes):
        current_chunk_size = base_chunk_size + (1 if p < remainder else 0)
        proc_keys_list = keys[start_idx:start_idx + current_chunk_size]
        proc_freqs = freqs[start_idx:start_idx + current_chunk_size]

        start_idx += current_chunk_size

        proc_tune_config = dict(tune_config)
        proc_tune_config['freqs'] = proc_freqs
        proc_tune_config['cell_type'] = cell_type_config

        processor_cavs_dict = {key: cavs_dict[key] for key in proc_keys_list}

        if processes == 1:
            # Run inline: avoids Windows spawn issues, keeps stdout in Jupyter,
            # and lets the caller retain direct references to cavity objects.
            run_tune_s(processor_cavs_dict, proc_tune_config, p)
        else:
            service = mp.Process(target=run_tune_s, args=(processor_cavs_dict, proc_tune_config, p))
            service.start()
            jobs.append(service)

    for job in jobs:
        job.join()


def run_tune_s(processor_cavs_dict, tune_config, p):
    # perform necessary checks
    if tune_config is None:
        tune_config = {}

    cell_type_config = tune_config.get('cell_type')
    if cell_type_config is None:
        # Late normalization in case caller invoked run_tune_s directly.
        cell_type_config = normalize_cell_type_config(tune_config)

    proc_freqs = tune_config['freqs']

    tune_config_keys = tune_config.keys()
    rerun = True
    if 'rerun' in tune_config_keys:
        if isinstance(tune_config['rerun'], bool):
            rerun = tune_config['rerun']

    scheduled_vars = _collect_scheduled_suffixed_vars(cell_type_config)

    def _propagate_after_stage(cav, tuned_suffixed_var, pre_tune_params):
        """Propagate a tuned parameter to sibling `_m/_el/_er` slots.

        Rules:
        - If the bare name is in ``_ALWAYS_PROPAGATE_BARE`` (e.g. ``Req``),
          the parameter is physically shared across all cells, so the
          new value is copied into every sibling that isn't itself
          scheduled for a later tune stage — regardless of pre-tune
          values.
        - After an end-cell-LEFT stage, if no end-cell-RIGHT stage is
          scheduled the tuned ``_el`` value is copied into ``_er`` so
          both ends of the cavity match by default.
        - Otherwise, siblings receive the new value only when their
          pre-tune value matched the tuned parameter's pre-tune value
          (preserves originally-identical relationships without
          overriding a deliberate mismatch).
        """
        if not tuned_suffixed_var:
            return
        bare = None
        src_suf = None
        for suf in _VALID_SUFFIXES:
            if tuned_suffixed_var.endswith(suf):
                bare = tuned_suffixed_var[:-len(suf)]
                src_suf = suf
                break
        if bare is None:
            return

        if src_suf == '_m':
            candidates = [f'{bare}_el', f'{bare}_er']
        elif src_suf == '_el':
            candidates = [f'{bare}_er']
        elif src_suf == '_er':
            candidates = [f'{bare}_el']
        else:
            candidates = []

        always = bare in _ALWAYS_PROPAGATE_BARE
        # Mirror tuned left end-cell to the right end-cell when the user
        # didn't schedule a dedicated end-cell-r stage.
        mirror_endcell = (src_suf == '_el'
                          and not any(v.endswith('_er') for v in scheduled_vars))

        new_val = cav.parameters[tuned_suffixed_var]
        for sibling in candidates:
            if sibling in scheduled_vars:
                continue
            if sibling not in pre_tune_params or tuned_suffixed_var not in pre_tune_params:
                continue
            if always:
                cav.parameters[sibling] = new_val
                continue
            if mirror_endcell and sibling.endswith('_er'):
                cav.parameters[sibling] = new_val
                continue
            if np.isclose(pre_tune_params[sibling], pre_tune_params[tuned_suffixed_var]):
                cav.parameters[sibling] = new_val

    def _run_tune(cav, key, target_freq):
        # Capture original parameters once — tuner mutates cav.parameters.
        pre_tune_params = dict(cav.parameters)
        tuned_self_dir = Path(cav.self_dir) / 'tuned'
        tune_info_dir = tuned_self_dir / 'tune_info'

        aggregated_tune_res = {}
        aggregated_conv = {}
        aggregated_abs_err = {}

        def _restore_and_bail():
            """Reset source cav to pre-tune state so a failed stage's
            garbage parameters don't leak out (e.g. into ``.tuned``)."""
            cav.parameters.update(pre_tune_params)
            if cav.geo_filepath:
                try:
                    with suppress_errors('Parameter set leads to degenerate geometry'):
                        cav.write_geometry(cav.parameters, cav.n_cells,
                                           cav.beampipe, write=cav.geo_filepath)
                except Exception:
                    pass

        for ct_name, tune_vars in cell_type_config.items():
            for tune_var in tune_vars:
                stage_tune_config = dict(tune_config)
                stage_tune_config.pop('cell_type', None)
                stage_tune_config['freqs'] = target_freq
                stage_tune_config['parameters'] = tune_var
                stage_tune_config['cell_types'] = ct_name

                # Align per-stage UQ with the stage's cell type. Without this
                # override, a uq_config pinned to 'mid-cell' would keep
                # perturbing mid-cell params during the end-cell stage —
                # making the UQ-averaged freq independent of the secant's
                # tune variable and producing a zero slope that blows up.
                if tune_config.get('uq_config'):
                    stage_uq = dict(tune_config['uq_config'])
                    stage_uq['cell_type'] = ct_name
                    stage_tune_config['uq_config'] = stage_uq

                try:
                    _tss, stage_res, conv_dict, abs_err_dict = tuner.tune_ngsolve(
                        {key: cav}, 33,
                        proc=p,
                        tune_variable=tune_var,
                        tune_config=stage_tune_config,
                    )
                except Exception as e:
                    error(f'Tuning failed for {key} at cell_type={ct_name} var={tune_var}: {e!r}')
                    _restore_and_bail()
                    return

                # ``tuner.tune_ngsolve`` now returns a dict keyed by cavity
                # name (``{key: d_tune_res}``); unwrap to the inner entry
                # for this cavity. Older callers that returned the inner
                # dict directly are still handled via the ``parameters``
                # sentinel check.
                if isinstance(stage_res, dict) and 'parameters' not in stage_res:
                    d_tune_res = stage_res.get(key, {}) or {}
                else:
                    d_tune_res = stage_res or {}

                if not d_tune_res:
                    error(f'Tune stage produced no result for {key}: cell_type={ct_name} var={tune_var}')
                    _restore_and_bail()
                    return

                resolved_var = d_tune_res.get('TUNED VARIABLE') or _resolve_suffixed_var(tune_var, ct_name)
                _propagate_after_stage(cav, resolved_var, pre_tune_params)

                # Per-stage snapshot records only the cell that was tuned
                # (e.g. mid-cell stage stores only ``*_m`` params). This
                # avoids the misleading repetition of untouched sibling
                # values from other cells.
                stage_params = _filter_params_for_ct(cav.parameters, ct_name)
                stage_freq = d_tune_res.get('FREQ', target_freq)

                ct_entry = aggregated_tune_res.setdefault(ct_name, {
                    'parameters': stage_params,
                    'TUNED VARIABLES': [],
                    'FREQ': stage_freq,
                })
                ct_entry['parameters'] = stage_params
                if resolved_var not in ct_entry['TUNED VARIABLES']:
                    ct_entry['TUNED VARIABLES'].append(resolved_var)
                ct_entry['FREQ'] = stage_freq

                aggregated_conv.setdefault(ct_name, {})[resolved_var] = conv_dict
                aggregated_abs_err.setdefault(ct_name, {})[resolved_var] = abs_err_dict

        if not aggregated_tune_res:
            _restore_and_bail()
            return

        tune_info_dir.mkdir(parents=True, exist_ok=True)

        # Build final tuned cavity from the cumulative tuned parameters
        # (cav.parameters was mutated in-place by the tuner).
        final_params = dict(cav.parameters)
        final_freq = next(reversed(aggregated_tune_res.values()))['FREQ']

        # Restore the *original* cavity's full geometry file — the tuner
        # overwrote it with a quarter/reduced geometry during stages. The
        # restore write reconstructs the untuned full multicell geometry
        # purely as a reference; its "Parameter set leads to degenerate
        # geometry" warnings (if any) are not user-actionable because the
        # tuned cavity lives in ``tuned_cav`` below.
        if cav.geo_filepath:
            cav.parameters.update(pre_tune_params)
            with suppress_errors('Parameter set leads to degenerate geometry'):
                cav.write_geometry(cav.parameters, cav.n_cells,
                                   cav.beampipe, write=cav.geo_filepath)

        tuned_cav = cav.clone_for_tuning(
            tuned_parameters=final_params,
            tuned_self_dir=str(tuned_self_dir),
            beampipe=cav.beampipe,
        )
        tuned_cav.tune_results = dict(aggregated_tune_res)
        tuned_cav.freq = final_freq

        with open(tune_info_dir / 'tune_res.json', 'w') as f:
            json.dump(aggregated_tune_res, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_convergence.json', 'w') as f:
            json.dump(aggregated_conv, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_absolute_error.json', 'w') as f:
            json.dump(aggregated_abs_err, f, indent=4, default=str)

        # Save UQ config if present
        if 'uq_config' in tune_config:
            with open(tune_info_dir / 'uq_config.json', 'w') as f:
                json.dump(tune_config['uq_config'], f, indent=4, default=str)

        # Aggregate per-stage UQ-tune results if the per-stage UQ writer
        # emitted ``<eigenmode>/uq.json`` — each stage overwrites the file,
        # so we keep whatever remains from the final stage and key it by
        # the final cell type.
        uq_tune_res_path = Path(cav.self_dir) / 'eigenmode' / 'uq.json'
        if uq_tune_res_path.exists():
            last_ct = next(reversed(aggregated_tune_res.keys()))
            with open(uq_tune_res_path) as f:
                last_uq = json.load(f)
            with open(tune_info_dir / 'uq_tune_results.json', 'w') as f:
                json.dump({last_ct: last_uq}, f, indent=4, default=str)

        # Mark original cavity as tuned in its own tune_results too.
        cav.tune_results = dict(aggregated_tune_res)

    for i, (key, cav) in enumerate(processor_cavs_dict.items()):
        target_freq = float(proc_freqs[i])
        cav.shape['FREQ'] = target_freq
        # Clear previous tuning / quarter-cell eigenmode scratch so we
        # start fresh on rerun.
        eigenmode_path = os.path.join(cav.self_dir, 'eigenmode')
        if os.path.exists(eigenmode_path) and rerun:
            shutil.rmtree(eigenmode_path)
            os.makedirs(eigenmode_path, exist_ok=True)
        tuned_path = os.path.join(cav.self_dir, 'tuned')
        if os.path.exists(tuned_path) and rerun:
            shutil.rmtree(tuned_path)
        _run_tune(cav, key, target_freq)


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

        return tuned_shape_space[key]

    processor_shape_space_tuned = {}
    for i, (key, shape) in enumerate(processor_shape_space.items()):
        if os.path.exists(os.path.join(projectDir, key, "eigenmode")):
            shutil.rmtree(os.path.join(projectDir, key, "eigenmode"))
            os.mkdir(os.path.join(projectDir, key, "eigenmode"))
            tuned_shape = _run_tune(key, shape)
        else:
            tuned_shape = _run_tune(key, shape)

        processor_shape_space_tuned[key] = tuned_shape

    return processor_shape_space_tuned
