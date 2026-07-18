"""Lightweight, non-fatal validation of the user-facing config dictionaries.

Unknown keys are almost always typos — and because the run functions silently
ignore keys they don't recognise, a typo like ``'cell complexity'`` (instead of
``'cell_complexity'``) quietly changes behaviour. These validators emit a clear
warning (with a "did you mean" suggestion) but never raise, so existing configs
keep working.
"""
import difflib
import warnings

# Accepted keys per config type — supersets of the public keys plus the few
# keys the run functions inject internally, so validation flags typos without
# false positives.
EIGENMODE_KEYS = {
    'processes', 'rerun', 'boundary_conditions', 'polarisation', 'n_modes', 'nmodes',
    'mesh_config', 'uq_config', 'f_shift', 'direct_solver', 'n_cells',
    'conductivity', 'surface_resistance', 'normalization_length', 'pinvit_maxit',
    'mode_of_interest',
    'opt', 'target', 'solver_save_directory',
}
UQ_KEYS = {
    'variables', 'objectives', 'delta', 'epsilon', 'processes', 'distribution',
    'method', 'cell_type', 'cell_complexity', 'perturbation_mode', 'tune_config',
    'cell', 'operating_points', 'integration', 'objectives_unprocessed',
}
WAKEFIELD_KEYS = {
    'processes', 'rerun', 'MROT', 'polarisation', 'wakelength', 'bunch_length',
    'MT', 'NFS', 'DDR_SIG', 'DDZ_SIG', 'operating_points', 'uq_config',
    'beam_config', 'wake_config', 'mesh_config', 'target', 'LCPUTM',
    'save_fields', 'objectives', 'opt', 'solver_save_directory',
    'contour_ds', 'beampipe_length', 'solver',
}
TUNE_KEYS = {
    'freqs', 'freq', 'cell_type', 'cell_types', 'parameters', 'processes',
    'rerun', 'eigenmode_config', 'tolerance', 'tol', 'maxiter', 'uq_config',
}
OPTIMISATION_KEYS = {
    'initial_points', 'no_of_generation', 'no_of_generations', 'method',
    'bounds', 'objectives', 'tune_config', 'mutation_factor', 'crossover_factor',
    'elites_for_crossover', 'chaos_factor', 'weights', 'seed', 'mutation_sigma',
    'eta_sbx', 'resume', 'mid-cell',
}

# Keys people commonly mistype -> the key the code actually reads.
_DEPRECATED = {
    'cell complexity': 'cell_complexity',
    'cell_type ': 'cell_type',
}


def _warn(message):
    # Standard warnings so config typos are always visible (the project's
    # printing.warning is silent unless verbose mode is on). stacklevel=4
    # points the warning at the user's run_*(...) call site.
    warnings.warn(message, UserWarning, stacklevel=4)


def validate_config(config, known_keys, name='config'):
    """Warn about unrecognised keys in *config* (non-fatal)."""
    if not isinstance(config, dict):
        return
    for key in list(config):
        if key in _DEPRECATED:
            _warn(f"{name}: '{key}' is not read — did you mean "
                  f"'{_DEPRECATED[key]}'?")
            continue
        if key in known_keys:
            continue
        match = difflib.get_close_matches(str(key), list(known_keys), n=1)
        hint = f" Did you mean '{match[0]}'?" if match else ""
        _warn(f"{name}: unrecognised key '{key}'.{hint}")


def require(condition, message):
    """Raise ``ValueError(message)`` if *condition* is falsy.

    Use instead of ``assert cond, error(msg)`` for user-facing input checks:
    asserts are stripped under ``python -O`` and ``error()`` returns ``None``
    (so the AssertionError message would be ``None``)."""
    if not condition:
        raise ValueError(message)


def _check_processes(cfg, context):
    if 'processes' in cfg:
        p = cfg['processes']
        require(isinstance(p, int) and not isinstance(p, bool),
                f"{context}: 'processes' must be an integer.")
        require(p > 0, f"{context}: 'processes' must be greater than zero.")


def _check_objectives(cfg, context):
    """Eigenmode objectives must name a polarisation ('dipole:R/Q [Ohm]').

    Monopole and m-pole ``qois.json`` share QOI names, so a bare name is ambiguous.
    """
    objectives = cfg.get('objectives')
    if not isinstance(objectives, (list, tuple)):
        return
    # Deferred: breaks a real cycle — cavsim2d.solvers.objectives pulls in the
    # cavsim2d.solvers package, whose eigen_ngsolve imports require() from here.
    from cavsim2d.solvers.objectives import parse_objective

    for obj in objectives:
        # optimisation objectives are ['min', name] / ['equal', name, target] pairs;
        # UQ objectives are bare names. Wakefield names (ZL/ZT) carry no polarisation.
        name = obj[1] if isinstance(obj, (list, tuple)) and len(obj) >= 2 else obj
        if not isinstance(name, str) or name.startswith(('ZL', 'ZT')):
            continue
        try:
            parse_objective(name)
        except ValueError as exc:
            raise ValueError(f'{context}: {exc}') from None


def validate_uq_config(cfg, context='uq_config'):
    validate_config(cfg, UQ_KEYS, context)
    if not isinstance(cfg, dict):
        return
    _check_processes(cfg, context)
    _check_objectives(cfg, context)
    variables = cfg.get('variables')
    if isinstance(variables, (list, tuple)):
        for k in ('delta', 'epsilon'):
            if isinstance(cfg.get(k), (list, tuple)):
                require(len(cfg[k]) == len(variables),
                        f"{context}: '{k}' has {len(cfg[k])} entries but there are "
                        f"{len(variables)} 'variables' — they must match.")


def validate_eigenmode_config(cfg):
    validate_config(cfg, EIGENMODE_KEYS, 'eigenmode_config')
    if isinstance(cfg, dict):
        _check_processes(cfg, 'eigenmode_config')
        for key in ('n_modes', 'nmodes'):
            # Explicit None means unset (complete configs carry every key,
            # with None for run-time-resolved values), so only typed values
            # are checked.
            if cfg.get(key) is not None:
                require(isinstance(cfg[key], int) and not isinstance(cfg[key], bool),
                        f"eigenmode_config: '{key}' must be an integer.")
                require(cfg[key] > 0,
                        f"eigenmode_config: '{key}' must be greater than zero.")
        _check_mode_of_interest(cfg)
        if isinstance(cfg.get('uq_config'), dict):
            validate_uq_config(cfg['uq_config'])


def _check_mode_of_interest(cfg):
    """``mode_of_interest`` is a 1-based mode index, a list of them (a polarisation
    may have several modes of interest), or a dict of either keyed by polarisation
    ('monopole', 'dipole', ... or the azimuthal number)."""
    # Explicit None means unset (the complete-config convention).
    if cfg.get('mode_of_interest') is None:
        return
    value = cfg['mode_of_interest']
    entries = list(value.items()) if isinstance(value, dict) else [(None, value)]

    flat = []           # (pol, mode)
    for pol, modes in entries:
        where = f" for polarisation {pol!r}" if pol is not None else ''
        if not isinstance(modes, (list, tuple)):
            modes = [modes]
        require(len(modes) > 0,
                f"eigenmode_config: 'mode_of_interest'{where} is empty; give at least "
                f"one 1-based mode index.")
        for mode in modes:
            require(isinstance(mode, int) and not isinstance(mode, bool),
                    f"eigenmode_config: 'mode_of_interest'{where} must be an integer "
                    f"(1-based), or a list of them, got {mode!r}.")
            require(mode > 0,
                    f"eigenmode_config: 'mode_of_interest'{where} is 1-based — mode 1 is "
                    f"the lowest of the passband. Got {mode}.")
            flat.append((pol, mode))

    n_modes = cfg.get('n_modes', cfg.get('nmodes'))
    if isinstance(n_modes, int) and not isinstance(n_modes, bool):
        for pol, mode in flat:
            where = f" for polarisation {pol!r}" if pol is not None else ''
            require(mode <= n_modes,
                    f"eigenmode_config: 'mode_of_interest'={mode}{where} exceeds "
                    f"'n_modes'={n_modes}. Raise n_modes to at least {mode}.")


def validate_wakefield_config(cfg):
    validate_config(cfg, WAKEFIELD_KEYS, 'wakefield_config')
    if isinstance(cfg, dict):
        _check_processes(cfg, 'wakefield_config')
        for k in ('MT', 'NFS'):
            if k in cfg:
                require(isinstance(cfg[k], int) and not isinstance(cfg[k], bool),
                        f"wakefield_config: '{k}' must be an integer.")
        if isinstance(cfg.get('uq_config'), dict):
            validate_uq_config(cfg['uq_config'])


def validate_tune_config(cfg):
    validate_config(cfg, TUNE_KEYS, 'tune_config')
    if isinstance(cfg, dict):
        _check_processes(cfg, 'tune_config')
        if isinstance(cfg.get('eigenmode_config'), dict):
            validate_eigenmode_config(cfg['eigenmode_config'])
        if isinstance(cfg.get('uq_config'), dict):
            validate_uq_config(cfg['uq_config'])


def validate_optimisation_config(cfg):
    validate_config(cfg, OPTIMISATION_KEYS, 'optimisation_config')
    if isinstance(cfg, dict):
        _check_processes(cfg, 'optimisation_config')
        _check_objectives(cfg, 'optimisation_config')
        if isinstance(cfg.get('tune_config'), dict):
            validate_tune_config(cfg['tune_config'])
