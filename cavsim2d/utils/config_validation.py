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


def validate_uq_config(cfg, context='uq_config'):
    validate_config(cfg, UQ_KEYS, context)
    if not isinstance(cfg, dict):
        return
    _check_processes(cfg, context)
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
            if key in cfg:
                require(isinstance(cfg[key], int) and not isinstance(cfg[key], bool),
                        f"eigenmode_config: '{key}' must be an integer.")
                require(cfg[key] > 0,
                        f"eigenmode_config: '{key}' must be greater than zero.")
        if isinstance(cfg.get('uq_config'), dict):
            validate_uq_config(cfg['uq_config'])


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
        if isinstance(cfg.get('tune_config'), dict):
            validate_tune_config(cfg['tune_config'])
