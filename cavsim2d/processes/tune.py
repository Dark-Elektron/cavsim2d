"""Parallel tuning process functions."""
import json
import os.path
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from cavsim2d.analysis.tune.tuner import Tuner
from cavsim2d.analysis.tune.pyTuner import TUNE_DEFECTS
from cavsim2d.constants import *
from cavsim2d.processes.eigenmode import run_eigenmode_parallel, run_eigenmode_s  # noqa: F401 — used by multicell path
from cavsim2d.utils.shared_functions import *
from cavsim2d.utils.printing import suppress_errors
from cavsim2d.utils.config_validation import require

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


def _check_known(var, bare, cav):
    """Validate *bare* against the model's own parameters."""
    if cav is None:
        return                      # no model context: nothing to check against
    known = cav.tune_variables()
    if bare in known:
        return
    name = type(cav).__name__
    if not known:
        raise ValueError(
            f"{name} exposes no tunable parameters, so it cannot be tuned by name. "
            f"Geometries read from a mesh or CAD file are not parameterised — tune a "
            f"parameterised model, or give {name} a `parameters` dict.")
    raise ValueError(
        f"Unknown tune variable {var!r} for {name}. It accepts: "
        f"{', '.join(sorted(known))}.")


def _resolve_suffixed_var(var, ct, cav=None):
    """Map a tune variable and cell type to the model's parameter name.

    The variable is validated against the model's own parameters
    (:meth:`Cavity.tune_variables`) rather than a fixed elliptical name list, so
    any geometry that exposes scalar parameters — the gun's ``R6``, the spline's
    ``p2_r``, or a model not yet written — can name them.

    Only models whose parameters carry per-cell suffixes get one appended; for
    everything else the name is used as given. (The old code split on ``'_'`` to
    find the bare name, which silently turned the pillbox's ``L_bp`` into ``L``.)

    The alias suffix ``_e`` means "both end cells": ``L_e`` resolves to ``L_el``
    here (the tuner tunes the symmetric left end cell, and
    :func:`_propagate_after_stage` mirrors the result into ``L_er``), so a single
    handle keeps the two ends tied. No real parameter name ends in a bare ``_e``.
    """
    if cav is not None and not cav.uses_cell_suffixes:
        _check_known(var, var, cav)
        return var

    if (cav is None or cav.uses_cell_suffixes) and var.endswith('_e'):
        return _resolve_suffixed_var(f'{var[:-2]}_el', ct, cav)

    if var.endswith(_VALID_SUFFIXES):
        bare = next(var[:-len(s)] for s in _VALID_SUFFIXES if var.endswith(s))
        _check_known(var, bare, cav)
        return var

    _check_known(var, var, cav)
    suffix = _SUFFIX_FOR_CT.get(_normalize_ct_name(ct), '_m')
    return f'{var}{suffix}'


def validate_cell_type_config(cell_type_config, cavs):
    """Reject unknown tune variables before any process is spawned.

    Without this the failure surfaces deep inside a worker, per cavity, after
    the geometry has already been written.
    """
    for cav in cavs:
        for ct, tune_vars in cell_type_config.items():
            for var in tune_vars:
                _resolve_suffixed_var(var, ct, cav)


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


def _collect_scheduled_suffixed_vars(cell_type_config, cav=None):
    """Return set of suffixed parameter names explicitly tuned by any stage."""
    scheduled = set()
    for ct, vars_ in cell_type_config.items():
        for v in vars_:
            scheduled.add(_resolve_suffixed_var(v, ct, cav))
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
    filtered = {k: v for k, v in params.items() if k.endswith(suffix)}
    # Non-elliptical cavities (pillbox, ...) use unsuffixed params (Req, L, …);
    # if nothing matched the cell suffix, keep the full dict rather than drop it.
    return filtered if filtered else dict(params)


def _save_convergence(tune_info_dir, conv, abs_err, status, reason='', target=None):
    """Persist the tuning convergence history + a status stamp.

    Called on success AND on failure, so a tune that could not converge still leaves
    its iteration history (tune variable + frequency per stage) on disk for inspection
    via ``cav.tune.convergence`` — previously nothing was written when a stage failed.
    ``target`` (the requested frequency) is stored so ``plot_convergence`` can draw it.
    """
    try:
        Path(tune_info_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(tune_info_dir) / 'tune_convergence.json', 'w') as f:
            json.dump(conv, f, indent=4, default=str)
        with open(Path(tune_info_dir) / 'tune_absolute_error.json', 'w') as f:
            json.dump(abs_err, f, indent=4, default=str)
        with open(Path(tune_info_dir) / 'tune_status.json', 'w') as f:
            json.dump({'status': status, 'reason': reason, 'target_freq': target},
                      f, indent=4, default=str)
    except Exception:
        pass


def run_tune_parallel(cavs_dict, tune_config, solver='NGSolveMEVP',
                      resume=False):
    tune_config_keys = tune_config.keys()
    if 'processes' in tune_config_keys:
        processes = tune_config['processes']
        require(processes > 0, 'Number of processes must be greater than zero.')
    else:
        processes = 1

    require('freqs' in tune_config_keys, 'Please enter the target tune "freqs" in tune_config.')

    # Canonicalise cell_type config (handles legacy parameters+cell_types).
    cell_type_config = normalize_cell_type_config(tune_config)
    validate_cell_type_config(cell_type_config, cavs_dict.values())
    tune_config = dict(tune_config)  # shallow copy so we don't mutate caller
    tune_config['cell_type'] = cell_type_config

    freqs = tune_config['freqs']
    if isinstance(freqs, (float, int)):
        freqs = np.array([freqs for _ in range(len(cavs_dict))], dtype=float)
    else:
        require(len(freqs) == len(cavs_dict), 'Number of target frequencies must correspond to the number of cavities')
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

    specs = []
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
        specs.append((processor_cavs_dict, proc_tune_config, p))

    if processes == 1:
        # Run inline: no spawn overhead, keeps stdout in Jupyter, and lets the
        # caller retain direct references to the tuned cavity objects.
        run_tune_s(*specs[0])
    else:
        # Parallel across cavities via joblib's loky backend — notebook-safe on
        # Windows (unlike multiprocessing.Process spawn, which needs a __main__
        # guard). Each worker tunes its chunk and writes results to disk.
        # Deferred: joblib is only pulled in for parallel runs.
        from joblib import Parallel, delayed
        Parallel(n_jobs=processes, backend='loky')(
            delayed(run_tune_s)(*spec) for spec in specs)


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

    # A symmetric 2-cell is field-flat for any cell split at a fixed operating
    # frequency, so tuning it does not pin a single design -- it leaves a family.
    # By default we return that family (``n_family`` members, the split sweep);
    # ``equal_cell_freq=True`` instead pins the single each-cell-at-f design (the
    # standard two-stage tune). Only 2-cell elliptical-family cavities are affected.
    equal_cell_freq = bool(tune_config.get('equal_cell_freq', False))
    n_family = int(tune_config.get('n_family', 5))
    family_fraction = float(tune_config.get('family_fraction', 0.03))

    # Equality QoI targets (e.g. {'R/Q [Ohm]': 157}) turn the frequency tune into a
    # constrained multivariable solve: the cell_type variables are moved to hit freq +
    # these targets together (see _run_qoi_tune). Absent -> the ordinary frequency tune.
    # For n>2 with a total R/Q, a per-cup STAGED tune is used (see _run_staged_qoi_tune);
    # ``qoi_split`` optionally overrides the auto-derived per-section R/Q shares.
    qoi_targets = tune_config.get('qoi_targets') or None
    qoi_split = tune_config.get('qoi_split') or None

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
        scheduled_vars = _collect_scheduled_suffixed_vars(cell_type_config, cav)
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
                # 'both ends' alias: tune the left end cell (its reduced model is
                # symmetric with the right end's), and let _propagate_after_stage
                # mirror the tuned value into the right end so they stay tied.
                if getattr(cav, 'uses_cell_suffixes', False) and tune_var.endswith('_e'):
                    tune_var = f'{tune_var[:-2]}_el'
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
                except TUNE_DEFECTS:
                    # A bug or an unsupported cavity type, not a bad candidate
                    # geometry. Restore the pre-tune parameters, then let it out:
                    # swallowing these turned "this type cannot tune" into a silent
                    # no-op that left cav.tuned = None while run_tune returned
                    # normally.
                    _restore_and_bail()
                    raise
                except Exception as e:
                    # Genuine tuning failure (degenerate geometry, no convergence).
                    # Optimisation relies on these being tolerated per candidate.
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

                # Capture this stage's iteration history NOW — before any bail — so a
                # stage that fails to converge still leaves its convergence on disk.
                # tuner.tune_ngsolve wraps the history in a single-entry dict keyed by
                # an internal sentinel; unwrap it to a flat {param: [values...]} dict.
                stage_var = _resolve_suffixed_var(tune_var, ct_name, cav)
                conv_series = (next(iter(conv_dict.values()))
                              if isinstance(conv_dict, dict) and len(conv_dict) == 1
                              else conv_dict)
                abs_err_series = (next(iter(abs_err_dict.values()))
                                  if isinstance(abs_err_dict, dict) and len(abs_err_dict) == 1
                                  else abs_err_dict)
                aggregated_conv.setdefault(ct_name, {})[stage_var] = conv_series
                aggregated_abs_err.setdefault(ct_name, {})[stage_var] = abs_err_series

                if not d_tune_res:
                    error(f'Tune stage produced no result for {key}: cell_type={ct_name} var={tune_var}')
                    _save_convergence(tune_info_dir, aggregated_conv, aggregated_abs_err,
                                      'failed', reason=f'{ct_name} / {tune_var}: target not reachable',
                                      target=target_freq)
                    _restore_and_bail()
                    return

                resolved_var = d_tune_res.get('TUNED VARIABLE') or stage_var
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

        if not aggregated_tune_res:
            _save_convergence(tune_info_dir, aggregated_conv, aggregated_abs_err,
                              'failed', reason='no stage produced a tuned result',
                              target=target_freq)
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
        # The per-stage snapshots in tune_res.json are FILTERED to the tuned
        # cell (mid-cell stores only `*_m`) for readability — they are NOT a
        # complete, self-consistent parameter set. Reconstructing the tuned
        # cavity from them merges a tuned `Req_m` over an untuned `Req_el`,
        # and `_unify_equator_radius` (which, for a single cell, treats the
        # end-cell as canonical) then wipes the tuned value. Persist the
        # complete, sibling-propagated `final_params` so `.tuned` reloads the
        # exact geometry the tuner converged on.
        with open(tune_info_dir / 'tuned_parameters.json', 'w') as f:
            json.dump({k: (list(v) if isinstance(v, (list, tuple, np.ndarray))
                           else float(v) if isinstance(v, (int, float, np.floating))
                           else v)
                       for k, v in final_params.items()},
                      f, indent=4, default=str)
        with open(tune_info_dir / 'tune_convergence.json', 'w') as f:
            json.dump(aggregated_conv, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_absolute_error.json', 'w') as f:
            json.dump(aggregated_abs_err, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_status.json', 'w') as f:
            json.dump({'status': 'converged', 'reason': '', 'target_freq': target_freq},
                      f, indent=4, default=str)

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

    def _jsonable(params):
        return {k: (list(v) if isinstance(v, (list, tuple, np.ndarray))
                    else float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in params.items()}

    def _write_family_manifest(cav, fam, target_freq):
        """Persist a family Study (from ``family_scan``) as the tuned manifest:
        ``family.json`` plus the each-cell-at-f member as the representative tuned
        cavity, so ``cav.tuned`` stays valid and the optimiser can fan out over the
        members. Returns True on success."""
        tune_info_dir = Path(cav.self_dir) / 'tuned' / 'tune_info'
        designs = fam.designs
        # Keep only members that actually built + solved: a sweep can push the end
        # length into degenerate geometry (recorded as freq = NaN), and such a member
        # cannot be rebuilt or solved -- so it must not reach family.json.
        members, valid_cavs = [], []
        for c, (_, d) in zip(fam.cavities_list, designs.iterrows()):
            if not np.isfinite(d['freq [MHz]']):
                continue
            members.append({'parameters': _jsonable(c.parameters),
                            'Req': float(d['Req']), 'L_e': float(d['L_e']), 'L_m': float(d['L_m']),
                            'FREQ': float(d['freq [MHz]']),
                            'mid_freq': float(d['mid_freq']), 'end_freq': float(d['end_freq'])})
            valid_cavs.append(c)
        if not members:
            error(f'Family tune produced no valid members for {cav.name}')
            return False

        # representative tuned cavity = the each-cell-at-f member (mid closest to end)
        rep_idx = int(np.argmin([abs(m['mid_freq'] - m['end_freq']) for m in members]))
        rep_params = dict(valid_cavs[rep_idx].parameters)
        rep_freq = members[rep_idx]['FREQ']

        tune_info_dir.mkdir(parents=True, exist_ok=True)
        with open(tune_info_dir / 'family.json', 'w') as f:
            json.dump({'target_freq': target_freq, 'representative': rep_idx,
                       'members': members}, f, indent=4, default=str)
        with open(tune_info_dir / 'tuned_parameters.json', 'w') as f:
            json.dump(_jsonable(rep_params), f, indent=4, default=str)
        tune_res = {'family': {'parameters': _jsonable(rep_params),
                               'TUNED VARIABLES': ['Req_m', 'L_el'], 'FREQ': rep_freq}}
        with open(tune_info_dir / 'tune_res.json', 'w') as f:
            json.dump(tune_res, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_status.json', 'w') as f:
            json.dump({'status': 'converged', 'reason': f'family ({len(members)} members)',
                       'target_freq': target_freq}, f, indent=4, default=str)
        cav.parameters.update(rep_params)
        cav.tune_results = tune_res
        return True

    def _run_family_tune(cav, key, target_freq, n_family):
        """Tune a symmetric 2-cell to a FAMILY of designs (the split sweep) rather
        than a single geometry."""
        pre_tune_params = dict(cav.parameters)
        eig_cfg = tune_config.get('eigenmode_config') or {}
        try:
            fam = cav.tune.family_scan(freq=target_freq, n=n_family, fraction=family_fraction,
                                       vary='split', mesh_config=eig_cfg.get('mesh_config'))
        except Exception as e:
            error(f'Family tune failed for {key}: {e!r}')
            cav.parameters.update(pre_tune_params)
            return
        if not _write_family_manifest(cav, fam, target_freq):
            cav.parameters.update(pre_tune_params)

    def _run_qoi_tune(cav, key, target_freq, qoi_targets):
        """QoI-constrained tune. The ``cell_type`` variables are solved together for
        (frequency + ``qoi_targets``) on the assembled cavity via a damped multivariable
        Newton (see ``TuneSolver._solve_qoi_targets``). The variable/target count decides
        the outcome: over-constrained raises; equal counts -> a unique design; more
        variables than targets -> a family. Results are written in the same tuned/ layout
        as an ordinary tune, so ``cav.tuned`` / ``cav.tune.qois`` work unchanged."""
        pre_tune_params = dict(cav.parameters)
        tuned_self_dir = Path(cav.self_dir) / 'tuned'
        tune_info_dir = tuned_self_dir / 'tune_info'
        ncells = int(cav.n_cells or 1)

        def _qoi_name(ct, v):
            # Map a (cell_type, variable) to the Newton naming. Req is the shared
            # equator radius -> bare (every cell tied). Otherwise the CELL TYPE sets
            # the suffix at EVERY n (including 2): mid-cell vars get `_m`, end-cell
            # vars `_e` (both end cells, LEAVING THE MID CELL ALONE). A bare end-cell
            # name used to stay bare at n<=2, and ``_solve_qoi_targets.resolve`` then
            # tied it across EVERY cell -- so a bare 'L' silently moved L_m too. Suffix
            # it so the config's intent (the end cell) matches what the Newton moves.
            if v in _ALWAYS_PROPAGATE_BARE:
                return v
            # An explicit suffix is the caller being unambiguous, so never re-suffix
            # it: that turned 'L_e' into 'L_e_e' at n>2, which resolves to nothing.
            if v.endswith(_VALID_SUFFIXES) or v.endswith('_e'):
                return v
            n = _normalize_ct_name(ct)
            if 'mid' in n:
                return f'{v}_m'
            if 'end' in n:
                return f'{v}_e'
            return v

        var_names = list(dict.fromkeys(
            _qoi_name(ct, v) for ct, vars_ in cell_type_config.items() for v in vars_))
        targets = {'freq [MHz]': float(target_freq)}
        targets.update({k: float(v) for k, v in qoi_targets.items()})
        eig_cfg = tune_config.get('eigenmode_config') or {}
        try:
            res = cav.tune._solve_qoi_targets(var_names, targets,
                                              mesh_config=eig_cfg.get('mesh_config'),
                                              n_family=n_family, fraction=family_fraction,
                                              bc=eig_cfg.get('boundary_conditions'),
                                              tol=tune_config.get('qoi_tol'),
                                              maxit=int(tune_config.get('qoi_maxit', 20)))
        except ValueError:
            raise                             # over-constrained / unknown var: user error
        except Exception as e:
            error(f'QoI tune failed for {key}: {e!r}')
            cav.parameters.update(pre_tune_params)
            return

        if res['mode'] == 'family':
            if not _write_family_manifest(cav, res['family'], target_freq):
                cav.parameters.update(pre_tune_params)
            return

        final_params = res['parameters']
        ach = res['achieved']

        # FEASIBILITY -- the Newton returns its best point whether or not it reached
        # the targets, so without a gate a poorly converged candidate is written as a
        # successful tune and joins an optimisation front looking legitimate.
        #
        # OPT-IN, only when the caller sets ``qoi_tol``. Gating by default would
        # change long-standing behaviour and, worse, breaks callers that tune a cup
        # to an approximate share -- the staged path's own end-cell stage runs
        # through here and legitimately lands a fraction of a percent out.
        # ``qoi_tol`` is a per-target absolute dict, or a scalar relative tolerance.
        qoi_tol = tune_config.get('qoi_tol')

        def _tol_for(nm, tgt):
            if isinstance(qoi_tol, dict) and nm in qoi_tol:
                return abs(float(qoi_tol[nm]))
            if isinstance(qoi_tol, (int, float)) and not isinstance(qoi_tol, bool):
                return abs(float(qoi_tol) * tgt)
            return abs(1e-3 * tgt)

        missed = {} if qoi_tol is None else {
            nm: (ach.get(nm), tgt, _tol_for(nm, tgt))
            for nm, tgt in targets.items()
            if ach.get(nm) is None
            or abs(float(ach[nm]) - float(tgt)) > _tol_for(nm, tgt)}
        if missed:
            detail = '; '.join(
                f'{nm}: got {"None" if got is None else format(float(got), ".4f")} '
                f'vs target {tgt:g} (tol {tol:.4g})'
                for nm, (got, tgt, tol) in sorted(missed.items()))
            error(f'QoI tune for {key}: NO SOLUTION -- {detail}')
            cav.parameters.update(pre_tune_params)
            return

        final_freq = ach.get('freq [MHz]', target_freq)
        tune_info_dir.mkdir(parents=True, exist_ok=True)
        cav.parameters.update(final_params)
        tuned_cav = cav.clone_for_tuning(tuned_parameters=final_params,
                                         tuned_self_dir=str(tuned_self_dir),
                                         beampipe=cav.beampipe)
        # Record the model's SUFFIXED parameter names (Req_el, L_el, ...) -- the same
        # convention as the frequency/family stages -- so the optimiser's
        # get_tune_value(TUNED VARIABLES[-1]) resolves against `parameters`. (The Newton
        # itself uses bare names for shared/2-cell variables; those are only for the plot.)
        suffixed_vars = [_resolve_suffixed_var(v, ct, cav)
                         for ct, vars_ in cell_type_config.items() for v in vars_]
        # `suffixed_vars` is the config's intent re-resolved, NOT the scope the Newton
        # actually moved -- a bare name ties every cell, so 'L' records as 'L_el' while
        # L_m and L_er move too. Record the Newton's own variable names and the
        # parameters that actually changed, so the tune record cannot disagree with
        # the geometry it produced.
        moved = sorted(
            k for k, v in final_params.items()
            if k in pre_tune_params
            and isinstance(v, (int, float)) and not isinstance(v, bool)
            and isinstance(pre_tune_params[k], (int, float))
            and not np.isclose(float(v), float(pre_tune_params[k]), rtol=0, atol=1e-9))
        tune_res = {'qoi': {'parameters': _jsonable(final_params),
                            'TUNED VARIABLES': suffixed_vars,
                            'NEWTON VARIABLES': list(var_names),
                            'PARAMETERS CHANGED': moved,
                            'FREQ': final_freq, 'QOIS': ach}}
        tuned_cav.tune_results = tune_res
        tuned_cav.freq = final_freq
        cav.tune_results = tune_res
        # {stage: {tune_var: {param: [values]}}} so cav.tune.convergence parses it; record
        # BOTH the tune variables (h[0]) and the targets (h[1]) per Newton iteration, so
        # plot_convergence shows a panel for each variable as well as each target.
        newton = {vn: [h[0][j] for h in res['history']]
                  for j, vn in enumerate(res['variables'])}
        newton.update({t: [h[1][t] for h in res['history']] for t in targets})
        conv = {'qoi': {'newton': newton}}
        with open(tune_info_dir / 'tune_res.json', 'w') as f:
            json.dump(tune_res, f, indent=4, default=str)
        with open(tune_info_dir / 'tuned_parameters.json', 'w') as f:
            json.dump(_jsonable(final_params), f, indent=4, default=str)
        with open(tune_info_dir / 'tune_convergence.json', 'w') as f:
            json.dump(conv, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_status.json', 'w') as f:
            json.dump({'status': 'converged', 'reason': '', 'target_freq': target_freq},
                      f, indent=4, default=str)

    def _read_conv(subcav, label):
        """Read a sub-cup's tune_convergence.json and re-key its stages under ``label``
        so several staged sub-tunes aggregate onto one cav.tune.convergence."""
        p = Path(subcav.self_dir) / 'tuned' / 'tune_info' / 'tune_convergence.json'
        if not p.exists():
            return {}
        try:
            d = json.load(open(p))
        except Exception:
            return {}
        return {f'{label}:{k}': v for k, v in d.items()}

    def _run_staged_qoi_tune(cav, key, target_freq, qoi_targets, qoi_split):
        """Staged, per-cup R/Q tune for n>2, exploiting cup additivity
        (R/Q_total = (2N-2)*mid_cup + 2*end_cup). The mid cell is tuned FREQUENCY-ONLY
        (via Req, periodic) and its cup R/Q measured; the remaining cavity R/Q is then
        assigned to the end cups -- auto-derived (total - (2N-2)*mid_cup) or from
        ``qoi_split['end-cell']`` -- and the end cell is tuned to (freq, its share) with
        local (L, A) at the shared Req. If the end cannot reach its share, tuning stops
        with an informative NO SOLUTION message. Per-cup convergence is aggregated onto
        cav.tune.convergence; the achieved assembled (freq, ff, R/Q) is reported so the
        user can compare against the target and adjust."""
        from cavsim2d.models.elliptical import EllipticalCavity

        pre = dict(cav.parameters)
        tuned_self_dir = Path(cav.self_dir) / 'tuned'
        tune_info_dir = tuned_self_dir / 'tune_info'
        N = int(cav.n_cells or 1)
        total_rq = float(qoi_targets['R/Q [Ohm]'])
        mesh = (tune_config.get('eigenmode_config') or {}).get('mesh_config')
        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        mid7 = [float(x) for x in cav.mid_cell]
        end7 = [float(x) for x in cav.end_cell_left]
        n_mid_cups, n_end_cups = 2 * N - 2, 2
        conv = {}

        # Take the per-stage tune variables from the caller's cell_type rather
        # than hard-coding them. Both stages used to pin 'Req' and ['L', 'A']
        # regardless of what was configured, so a config naming anything else was
        # silently ignored -- the same class of defect as a bare 'L' tying every
        # cell: the record says one thing and the solve does another.
        def _vars_for(kind, default):
            for ct, vs in cell_type_config.items():
                if kind in _normalize_ct_name(ct):
                    return list(vs)
            return list(default)

        mid_vars = _vars_for('mid', ['Req'])
        end_vars = _vars_for('end', ['L', 'A'])
        # For the assembly solve the names go straight to the Newton, whose
        # ``resolve`` reads the suffix directly: a BARE name there means "every
        # cell", which would drag the mid cup along -- exactly how bare 'L' once
        # overwrote L_m. Under an 'end-cell' key the intent is the end cells, so
        # an unsuffixed name is promoted to '_e' to match what the cup branch
        # does. Say so rather than doing it silently: the whole class of bug here
        # was a config that read one way and solved another.
        end_vars_asm = []
        for v in end_vars:
            if v.endswith(('_m', '_el', '_er', '_e')):
                end_vars_asm.append(v)
            else:
                warning(f"Staged tune {key}: end-cell variable {v!r} is bare. A "
                        f"bare name means the same dimension on EVERY cell; "
                        f"under an 'end-cell' key it is taken as {v}_e (both end "
                        f"cells, mid cell untouched). Write {v}_e explicitly.")
                end_vars_asm.append(f'{v}_e')

        def _acc(c, bc=None):
            c.set_workspace(tempfile.mkdtemp())
            run = {'mesh_config': mesh, 'n_modes': (c.n_cells or 1) + 3}
            if bc:
                run['boundary_conditions'] = bc
            c.eigenmode.run(**run)
            # Deferred import: solver_objects <-> processes.tune cycle.
            from cavsim2d.solvers.solver_objects import _accelerating_mode_row
            return _accelerating_mode_row(c.eigenmode.qois_df, target_freq)

        # STAGE 1 -- mid cell frequency-only (periodic, mm): sets shared Req, measures mid_cup
        cm = EllipticalCavity(1, mid7, mid7, mid7, beampipe='none')
        cm.set_workspace(tempfile.mkdtemp())
        try:
            _mid_cfg = {'freqs': target_freq, 'cell_type': {'mid-cell': mid_vars},
                        'eigenmode_config': {'boundary_conditions': 'mm',
                                             'mesh_config': mesh}}
            if tune_config.get('tolerance') is not None:
                _mid_cfg['tolerance'] = tune_config['tolerance']
            cm.tune.run(_mid_cfg)
        except Exception as e:
            error(f'Staged tune {key}: mid-cell frequency tune failed: {e!r}')
            cav.parameters.update(pre); return
        if cm.tuned is None:
            error(f'Staged tune {key}: NO SOLUTION -- the mid cell cannot reach '
                  f'{target_freq} MHz.')
            cav.parameters.update(pre); return
        Req_shared = float(cm.tuned.parameters['Req_m'])
        am = _acc(cm.tuned, bc='mm')
        mid_cup = float(am['R/Q [Ohm]']) / 2
        mid7[6] = Req_shared
        conv.update(_read_conv(cm, 'mid freq'))
        info('Staged tune %s: mid cup f=%.3f (target %.3f, BC=mm), R/Q=%.3f (Req=%.3f)'
             % (key, float(am['freq [MHz]']), target_freq, mid_cup, Req_shared))

        # DERIVE the end-cup share from additivity (or the qoi_split override)
        if qoi_split and 'end-cell' in qoi_split:
            end_cup = float(qoi_split['end-cell']) / n_end_cups
        else:
            end_cup = (total_rq - n_mid_cups * mid_cup) / n_end_cups
        if end_cup <= 0:
            error('Staged tune %s: NO SOLUTION -- the %d mid cups already contribute '
                  'R/Q=%.2f, at/above the total target %.2f. Raise the total or lower the '
                  'mid contribution.' % (key, n_mid_cups, n_mid_cups * mid_cup, total_rq))
            _save_convergence(tune_info_dir, conv, {}, 'failed',
                              reason='mid contribution exceeds total', target=target_freq)
            cav.parameters.update(pre); return

        end7[6] = Req_shared

        # STAGE 2, 'assembly' variant -- tune the end cell against the ASSEMBLED
        # cavity instead of the isolated cup.
        #
        # Why this exists: two cells that each resonate at f0 under their own
        # boundary conditions do NOT assemble to f0. Measured on a 2-cell, the
        # assembly lands ~0.095 MHz high, and that offset is physical, not
        # numerical -- it survives mesh refinement (it converged to +0.0952 MHz
        # from h=20,p=3 through h=7,p=4). Field flatness is 99.999% either way,
        # so letting the end cell sit off-frequency costs nothing in flatness and
        # is the only way to hold the assembled frequency. The mid cell is still
        # tuned periodically in stage 1, so the design stays reusable at higher n.
        #
        # Costs assembled solves in the Newton, so it is opt-in.
        # For a 2-cell the two end cells ARE the whole cavity, so tuning the end on the
        # ASSEMBLED cavity is both the correct model (an end cell needs its mid neighbour)
        # and cheap -- and the isolated-cup path lands the assembly ~0.1 MHz off frequency
        # (verified: 801.676 vs 801.580). Default to 'assembly' for n <= 3 -- the assembled
        # Newton converges quickly there (~12-15 steps) and gives the best field flatness;
        # n > 3 uses the cup path, ITERATED (see the end-cell branch below) so the assembled
        # total converges to the target while each cup tune stays a cheap 1-cell solve.
        _stage2_default = 'assembly' if N <= 3 else 'end-cell'
        # The iterative cup branch assembles internally and stashes its achieved QoIs
        # here so the shared report below reuses them instead of re-solving; the
        # 'assembly' Newton branch leaves it None and the report assembles once.
        staged_ach = None
        if str(tune_config.get('qoi_stage2', _stage2_default)).lower().startswith('assembl'):
            # Build a fresh assembly from the stage-1 cells. Mutating
            # ``cav.parameters`` is not enough: ``_solve_qoi_targets`` reads
            # ``cav.mid_cell`` / ``cav.end_cell_left``, which do not refresh from a
            # parameters update, so the Newton would silently start from the
            # pre-stage-1 geometry (measured 792.7 MHz instead of 801.6).
            cav_asm = EllipticalCavity(N, mid7, end7, end7, beampipe=cav.beampipe)
            cav_asm.set_workspace(tempfile.mkdtemp())
            try:
                res2 = cav_asm.tune._solve_qoi_targets(
                    end_vars_asm,
                    # freq + EVERY qoi_target (not just R/Q) so extra QoIs are actually
                    # tuned -- and the Newton's counting then enforces one end variable
                    # per target (add a QoI => free another variable, else over-defined).
                    {'freq [MHz]': target_freq, **{k: float(v) for k, v in qoi_targets.items()}},
                    mesh_config=mesh,
                    bc=(tune_config.get('eigenmode_config') or {}).get(
                        'boundary_conditions'),
                    tol=tune_config.get('qoi_tol'),
                    # Assembly Newton default maxit 20 -- with the practical tolerances
                    # (freq 1e-2 MHz, R/Q 0.1 Ohm) it converges in ~12-15 steps for n<=3.
                    maxit=int(tune_config.get('qoi_maxit', 20)))
            except Exception as e:
                error(f'Staged tune {key}: assembly stage-2 failed: {e!r}')
                cav.parameters.update(pre)
                return
            if res2.get('mode') != 'unique':
                error(f'Staged tune {key}: assembly stage-2 returned '
                      f'{res2.get("mode")!r}, expected a unique design.')
                cav.parameters.update(pre)
                return
            p2 = res2['parameters']
            end7[0] = float(p2['A_el'])
            end7[5] = float(p2['L_el'])
            info('Staged tune %s: assembly stage 2 -> A_el=%.4f L_el=%.4f '
                 '(mid cup R/Q=%.3f, Req=%.3f)'
                 % (key, end7[0], end7[5], mid_cup, Req_shared))
        else:
            # STAGE 2 -- ITERATIVE per-cup tune. The additive split ``end_cup`` is only
            # the INITIAL GUESS for the per-cup share. Each pass tunes the end cell to
            # (freq, n_end_cups*share) via local (L, A) at the shared Req, ASSEMBLES the
            # full cavity, measures the achieved total R/Q, and corrects the share by
            # residual/n_end_cups -- d(assembled total)/d(share) ~ n_end_cups, so this
            # is a Newton step whose only error is the slowly-varying coupling offset.
            # Converges in ~3-4 steps to the exact total; cost stays flat in N (each
            # step is one 1-cell end tune + one N-cell assembly solve).
            #
            # KNOWN LIMITATION (geometry): the end cup is solved MIRRORED onto itself (a
            # symmetric beampipe='both' cell), not the physically-correct hybrid
            # [bp]-[end-cup]-[mid-cup] (end_right=mid7, beampipe='left', left-only vars).
            # The hybrid IS the right frequency model (its inner iris faces a mid neighbour
            # so its f0 matches the assembled frequency), BUT tuning R/Q with the mid half
            # pinned is Newton-FRAGILE: for a target below nominal the end cup must shrink
            # hard and the solve wanders into degenerate geometry (measured: R/Q=95 near
            # nominal tunes fine, R/Q=88=total/2 degenerates at A_el=40). So the mirrored
            # self-cup is kept -- it lands the assembled frequency ~0.05 MHz high but tunes
            # robustly; use qoi_stage2='assembly' for a frequency-exact result. See the
            # flattop/hybrid TODO in SHIPPING_ACTION_PLAN.md. Only affects n >= 3 (n=2
            # defaults to 'assembly').
            _qtol = tune_config.get('qoi_tol')
            if isinstance(_qtol, dict) and 'R/Q [Ohm]' in _qtol:
                rq_conv_tol = abs(float(_qtol['R/Q [Ohm]']))
            elif isinstance(_qtol, (int, float)) and not isinstance(_qtol, bool):
                rq_conv_tol = abs(float(_qtol) * total_rq)
            else:
                rq_conv_tol = max(0.2, 1e-4 * total_rq)   # user's target band ~ +-0.2 Ohm
            _maxit = max(1, int(tune_config.get('qoi_maxit', 20)))

            share = end_cup          # per-cup share, starts at the additive guess
            best_end7 = None         # last FEASIBLE (A_el, L_el) design
            for _it in range(_maxit):
                ce = EllipticalCavity(1, end7, end7, end7, beampipe='both')
                ce.set_workspace(tempfile.mkdtemp())
                try:
                    _end_cfg = {'freqs': target_freq,
                                'cell_type': {'end-cell': end_vars},
                                'qoi_targets': {'R/Q [Ohm]': n_end_cups * share},
                                'eigenmode_config': {'mesh_config': mesh}}
                    if tune_config.get('tolerance') is not None:
                        _end_cfg['tolerance'] = tune_config['tolerance']
                    ce.tune.run(_end_cfg)
                except Exception as e:
                    error(f'Staged tune {key}: end-cell tune failed: {e!r}')
                    cav.parameters.update(pre); return
                eq = (ce.tune.qois.get('qoi') or {}).get('QOIS', {})
                got_f, got_rq = eq.get('freq [MHz]'), eq.get('R/Q [Ohm]')
                conv.update(_read_conv(ce, 'end R/Q' if _it == 0 else f'end R/Q {_it + 1}'))
                # FEASIBILITY of the isolated cup at THIS share.
                f_tol = max(0.5, 1e-3 * target_freq)
                cup_tol = max(1.0, 0.03 * n_end_cups * share)
                cup_ok = (got_f is not None and got_rq is not None
                          and abs(got_f - target_freq) <= f_tol
                          and abs(got_rq - n_end_cups * share) <= cup_tol)
                if not cup_ok:
                    if best_end7 is not None:
                        # A residual correction pushed the share past what the cup can
                        # reach -- keep the best feasible design rather than failing.
                        warning('Staged tune %s: end-cup share %.2f/cup became unreachable '
                                'at iteration %d; keeping the last feasible design.'
                                % (key, share, _it + 1))
                        break
                    error('Staged tune %s: NO SOLUTION -- the end cell cannot achieve its '
                          'required share R/Q=%.2f per cup (=%.2f for the cell) at %.3f MHz '
                          'with the shared Req=%.2f. Closest reached: R/Q=%.2f at %.3f MHz. '
                          'Adjust the total R/Q, the qoi_split, or the end-cell iris/shape.'
                          % (key, share, n_end_cups * share, target_freq, Req_shared,
                             got_rq if got_rq is not None else float('nan'),
                             got_f if got_f is not None else float('nan')))
                    _save_convergence(tune_info_dir, conv, {}, 'failed',
                                      reason='end cell cannot reach derived R/Q', target=target_freq)
                    cav.parameters.update(pre); return
                # Candidate design -> ASSEMBLE and measure the achieved total.
                cand = end7[:]
                cand[0] = float(ce.tuned.parameters['A_el'])
                cand[5] = float(ce.tuned.parameters['L_el'])
                a_it = _acc(EllipticalCavity(N, mid7, cand, cand, beampipe=cav.beampipe))
                achieved_rq = float(a_it['R/Q [Ohm]'])
                best_end7 = cand
                staged_ach = {'freq [MHz]': float(a_it['freq [MHz]']),
                              'R/Q [Ohm]': achieved_rq, 'ff [%]': float(a_it['ff [%]'])}
                resid = total_rq - achieved_rq
                assembled_f = float(a_it['freq [MHz]'])
                info('Staged tune %s: cup iter %d -> share=%.3f/cup | end-cup f=%.3f '
                     '(target %.3f) | assembled f=%.3f R/Q=%.3f ff=%.1f%% '
                     '(target R/Q %.2f, resid %+.3f)'
                     % (key, _it + 1, share, got_f if got_f is not None else float('nan'),
                        target_freq, assembled_f, achieved_rq, float(a_it['ff [%]']),
                        total_rq, resid))
                if abs(resid) <= rq_conv_tol:
                    break
                share += resid / n_end_cups
                if share <= 0:
                    warning('Staged tune %s: residual correction drove the end-cup share '
                            'non-positive; keeping the last feasible design.' % key)
                    break
            # Did the R/Q loop reach tol, or run out of iterations? (With the hybrid
            # [bp]-[end-cup]-[mid-cup] geometry the assembled FREQUENCY should also land on
            # f0 -- the end cup is tuned in its true in-assembly context -- so report both.)
            _final_resid = total_rq - staged_ach['R/Q [Ohm]']
            if abs(_final_resid) <= rq_conv_tol:
                info('Staged tune %s: R/Q iteration converged in %d/%d step(s) '
                     '(resid %+.3f, tol %.3f).' % (key, _it + 1, _maxit, _final_resid, rq_conv_tol))
            else:
                warning('Staged tune %s: R/Q iteration hit maxit=%d without reaching tol '
                        '(resid %+.3f, tol %.3f); keeping the best design.'
                        % (key, _maxit, _final_resid, rq_conv_tol))
            end7[0], end7[5] = best_end7[0], best_end7[5]
            end_cup = share          # record the converged share for the report

        # REPORT the achieved total.
        #
        # METHOD NOTE (n>2 cup path). The additive split
        #     end_cup = (total - (2N-2)*mid_cup) / 2
        # is only the INITIAL GUESS. The 'end-cell' stage above ITERATES it: tune the
        # end cup to (freq, n_end_cups*share), ASSEMBLE, measure the achieved total,
        # correct share by residual/n_end_cups, and re-tune until the assembled total
        # meets the target (usually 3-4 steps). This converges to the EXACT total, and
        # its cost stays FLAT in N (each step = one 1-cell end tune + one N-cell
        # assembly solve). The full-assembly path (Method 2, the 'assembly' stage-2
        # above) instead runs a Newton on the whole cavity, whose cost GROWS with N;
        # measured, it is cheaper than the iterative cup for ~2-3 cells and the cup
        # wins beyond that -- which is why 2-cell defaults to 'assembly'. The iterative
        # cup assembled its final design in the loop and stashed the QoIs in
        # ``staged_ach``, so reuse them; the 'assembly' Newton branch left it None and
        # needs one assembly solve here to read ff/freq/R/Q.
        if staged_ach is None:
            cav_full = EllipticalCavity(N, mid7, end7, end7, beampipe=cav.beampipe)
            a = _acc(cav_full)
            ach = {'freq [MHz]': float(a['freq [MHz]']), 'R/Q [Ohm]': float(a['R/Q [Ohm]']),
                   'ff [%]': float(a['ff [%]'])}
        else:
            ach = staged_ach
        info('Staged tune %s: assembled R/Q=%.2f (target %.2f, %+.2f%%), ff=%.1f%%, f=%.3f'
             % (key, ach['R/Q [Ohm]'], total_rq, 100 * (ach['R/Q [Ohm]'] - total_rq) / total_rq,
                ach['ff [%]'], ach['freq [MHz]']))

        # FEASIBILITY on the ASSEMBLED cavity -- OPT-IN, only when the caller sets
        # ``qoi_tol``. The per-cup checks above confirm each cup reached its own
        # share, but the additive derivation is approximate (a few % on the
        # assembled total by design), so gating it by default would reject sound
        # designs. An optimisation that needs every candidate on the constraint
        # surface sets qoi_tol explicitly and gets the strict behaviour.
        _qtol = tune_config.get('qoi_tol')
        if _qtol is None:
            _asm_targets = {}
        else:
            _asm_targets = {'freq [MHz]': float(target_freq),
                            'R/Q [Ohm]': float(total_rq)}

        def _atol(nm, tgt):
            if isinstance(_qtol, dict) and nm in _qtol:
                return abs(float(_qtol[nm]))
            if isinstance(_qtol, (int, float)) and not isinstance(_qtol, bool):
                return abs(float(_qtol) * tgt)
            return abs(1e-3 * tgt)

        _missed = {nm: (ach.get(nm), tgt, _atol(nm, tgt))
                   for nm, tgt in _asm_targets.items()
                   if ach.get(nm) is None
                   or abs(float(ach[nm]) - tgt) > _atol(nm, tgt)}
        if _missed:
            detail = '; '.join(
                f'{nm}: assembled {"None" if got is None else format(float(got), ".4f")} '
                f'vs target {tgt:g} (tol {tol:.4g})'
                for nm, (got, tgt, tol) in sorted(_missed.items()))
            error(f'Staged tune {key}: NO SOLUTION -- {detail}')
            _save_convergence(tune_info_dir, conv, {}, 'failed',
                              reason='assembled targets missed', target=target_freq)
            cav.parameters.update(pre)
            return

        final_params = dict(cav.parameters)
        for j, nm in enumerate(names):
            final_params[f'{nm}_m'] = mid7[j]
            final_params[f'{nm}_el'] = end7[j]
            final_params[f'{nm}_er'] = end7[j]

        tune_info_dir.mkdir(parents=True, exist_ok=True)
        cav.parameters.update(final_params)
        tuned_cav = cav.clone_for_tuning(tuned_parameters=final_params,
                                         tuned_self_dir=str(tuned_self_dir),
                                         beampipe=cav.beampipe)
        tune_res = {'staged': {'parameters': _jsonable(final_params),
                               'TUNED VARIABLES': ['Req_m', 'L_el', 'A_el'],
                               'FREQ': ach['freq [MHz]'], 'QOIS': ach,
                               'mid_cup': mid_cup, 'end_cup': end_cup,
                               'R/Q target': total_rq}}
        tuned_cav.tune_results = tune_res
        tuned_cav.freq = ach['freq [MHz]']
        cav.tune_results = tune_res
        with open(tune_info_dir / 'tune_res.json', 'w') as f:
            json.dump(tune_res, f, indent=4, default=str)
        with open(tune_info_dir / 'tuned_parameters.json', 'w') as f:
            json.dump(_jsonable(final_params), f, indent=4, default=str)
        with open(tune_info_dir / 'tune_convergence.json', 'w') as f:
            json.dump(conv, f, indent=4, default=str)
        with open(tune_info_dir / 'tune_status.json', 'w') as f:
            json.dump({'status': 'converged',
                       'reason': 'staged: R/Q %.2f (target %.2f)' % (ach['R/Q [Ohm]'], total_rq),
                       'target_freq': target_freq}, f, indent=4, default=str)

    for i, (key, cav) in enumerate(processor_cavs_dict.items()):
        target_freq = float(proc_freqs[i])
        cav.shape['FREQ'] = target_freq
        # Warn if a cached tune is being served for a cavity whose
        # in-memory parameters no longer match the geometry on disk.
        cav._check_geometry_mismatch('tune')
        tuned_path = os.path.join(cav.self_dir, 'tuned')
        if os.path.exists(tuned_path) and rerun:
            # Scope: only the tuned cavity folder is wiped — sibling
            # analyses (root eigenmode/, wakefield/) are preserved.
            shutil.rmtree(tuned_path)
        # Staged is the standard recipe at every multi-cell count, n == 2 included.
        # It is the only one that leaves the mid cell resonant at the design
        # frequency on its own, which is what makes a design reusable at higher n:
        # the single-stage Newton solves the assembly, so the mid cell lands
        # wherever the assembly needs it (spreads of tens of MHz were measured).
        # The cup accounting already holds at n == 2 -- (2N-2) = 2 mid cups plus
        # 2 end cups covers the [end_l, mid, mid, end_r] half-cell sequence.
        _staged = (qoi_targets and 'R/Q [Ohm]' in qoi_targets and (cav.n_cells or 0) >= 2
                   and 'mid-cell' in cell_type_config and 'end-cell' in cell_type_config
                   and getattr(cav, 'uses_cell_suffixes', False))
        if _staged:
            _run_staged_qoi_tune(cav, key, target_freq, qoi_targets, qoi_split)
        elif qoi_targets:
            _run_qoi_tune(cav, key, target_freq, qoi_targets)
        elif (not equal_cell_freq and (cav.n_cells or 0) == 2
                and getattr(cav, 'uses_cell_suffixes', False)):
            _run_family_tune(cav, key, target_freq, n_family)
        else:
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
        # Only wipe the tuned/ folder on rerun — leave the cavity's
        # eigenmode/ and wakefield/ alone so a re-tune doesn't nuke
        # sibling analyses.
        tuned_path = os.path.join(projectDir, key, 'tuned')
        if rerun and os.path.exists(tuned_path):
            shutil.rmtree(tuned_path)
        tuned_shape = _run_tune(key, shape)

        processor_shape_space_tuned[key] = tuned_shape

    return processor_shape_space_tuned
