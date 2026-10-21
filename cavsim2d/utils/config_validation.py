"""Lightweight, non-fatal validation of the user-facing config dictionaries.

Unknown keys are almost always typos — and because the run functions silently
ignore keys they don't recognise, a typo like ``'cell complexity'`` (instead of
``'cell_complexity'``) quietly changes behaviour. These validators emit a clear
warning (with a "did you mean" suggestion) but never raise, so existing configs
keep working.
"""
import difflib
import warnings
from cavsim2d.utils.printing import CavsimWarning

#: Keys a config accepts besides those of its default config: second spellings
#: (see CONFIG_ALIASES), per-call switches, settings without a default and keys
#: the run functions inject internally. The default configs live with the
#: solvers, which add their keys with register_defaults(), so a new default is
#: accepted without being listed a second time here.
_EXTRA_KEYS = {
    'eigenmode': {
        'force', 'nmodes', 'n_cells', 'normalization_length', 'arnoldi_vectors',
        'freq_only', 'opt', 'target', 'solver_save_directory',
    },
    'wakefield': {
        'force', 'polarisation', 'objectives', 'mesh_config', 'beam_config',
        'wake_config', 'LCPUTM', 'contour_ds', 'cutoff', 'opt', 'target',
        'solver_save_directory',
    },
    'tune': {
        'force', 'freq', 'tolerance', 'parameters', 'cell_types', 'n_family',
        'family_fraction', 'qoi_split', 'qoi_tol', 'qoi_stage2', 'qoi_maxit',
    },
    'multipacting': {'force'},
    'uq': {
        'epsilon', 'cell', 'operating_points', 'perturbation_mode',
        'objectives_unprocessed',
    },
    'optimisation': {
        'force', 'no_of_generations', 'seed', 'weights', 'mutation_sigma', 'eta_sbx',
        'resume', 'mid-cell', 'beampipe', 'eigenmode_config', 'wakefield_config',
        'uq_config', 'constraints', 'hv_tol', 'hv_consecutive', 'archive_size',
        'eps_tol', 'eps_consecutive',
    },
}

#: Accepted keys per config: its extra keys plus the keys of its default config.
KNOWN_KEYS = {kind: set(keys) for kind, keys in _EXTRA_KEYS.items()}
EIGENMODE_KEYS = KNOWN_KEYS['eigenmode']
UQ_KEYS = KNOWN_KEYS['uq']
WAKEFIELD_KEYS = KNOWN_KEYS['wakefield']
TUNE_KEYS = KNOWN_KEYS['tune']
OPTIMISATION_KEYS = KNOWN_KEYS['optimisation']
MULTIPACTING_KEYS = KNOWN_KEYS['multipacting']


def register_defaults(kind, defaults):
    """Accept every key of *defaults*, the default config of a *kind* run."""
    KNOWN_KEYS[kind].update(defaults)


# Keys people commonly mistype -> the key the code actually reads.
_DEPRECATED = {
    'cell complexity': 'cell_complexity',
    'cell_type ': 'cell_type',
}

#: Second spellings accepted for a key, per config, mapped to the key the code
#: reads. canonicalise() renames them before a run merges its defaults in: left
#: as they were, a default for the real key shadowed them, so
#: ``cav.tune.run(freq=1400)`` tuned to the default frequency and
#: ``no_of_generations=20`` ran the default 100 generations.
CONFIG_ALIASES = {
    'eigenmode': {'nmodes': 'n_modes', 'overwrite': 'force'},
    'wakefield': {'overwrite': 'force'},
    'tune': {'freq': 'freqs', 'tolerance': 'tol', 'overwrite': 'force'},
    'multipacting': {'overwrite': 'force'},
    'optimisation': {'no_of_generations': 'no_of_generation', 'overwrite': 'force'},
}

#: Configs nested inside another one, canonicalised with it.
_NESTED = {
    'tune': {'eigenmode_config': 'eigenmode'},
    'multipacting': {'eigenmode_config': 'eigenmode'},
    'optimisation': {'tune_config': 'tune', 'eigenmode_config': 'eigenmode',
                     'wakefield_config': 'wakefield'},
}


def canonicalise(config, kind):
    """A copy of *config* with every alias renamed to the key the code reads.

    For a tune config the legacy ``parameters`` + ``cell_types`` pair becomes the
    ``cell_type`` dict. Nested configs (a tune's ``eigenmode_config``, an
    optimisation's ``tune_config``, ...) are canonicalised too. An alias given
    alongside its key with a different value raises: there is no telling which
    was meant.
    """
    out = dict(config or {})
    for alias, key in CONFIG_ALIASES.get(kind, {}).items():
        if alias not in out:
            continue
        value = out.pop(alias)
        if key in out and out[key] != value:
            raise ValueError(
                f"{kind} config: {alias!r} and {key!r} are the same setting but were "
                f"given different values ({value!r} and {out[key]!r}). Give just one.")
        out[key] = value
    if kind == 'tune' and ('parameters' in out or 'cell_types' in out):
        params, cell_types = out.pop('parameters', None), out.pop('cell_types', None)
        if out.get('cell_type'):
            raise ValueError("tune config: give 'cell_type' or the older 'parameters' + "
                             "'cell_types' pair, not both.")
        require(params and cell_types,
                "tune config: the older 'parameters' + 'cell_types' form needs both, e.g. "
                "parameters='Req', cell_types='mid-cell'. Or use cell_type={'mid-cell': 'Req'}.")
        # In that form a list held one entry per cavity, and tuning only ever
        # used the first; keep that reading, and say so when that drops a stage.
        lists = [v for v in (params, cell_types) if isinstance(v, (list, tuple))]
        if any(len(set(map(str, v))) > 1 for v in lists):
            warnings.warn(
                f"tune config: parameters={params!r} / cell_types={cell_types!r} hold "
                f"several entries, but this older form tunes only the first. Use "
                f"cell_type={{'mid-cell': 'Req', 'end-cell': 'L'}} to tune several "
                f"cell types in turn.", CavsimWarning, stacklevel=3)
        if isinstance(params, (list, tuple)):
            params = params[0]
        if isinstance(cell_types, (list, tuple)):
            cell_types = cell_types[0]
        out['cell_type'] = {cell_types: [params]}
    for key, sub_kind in _NESTED.get(kind, {}).items():
        if isinstance(out.get(key), dict):
            out[key] = canonicalise(out[key], sub_kind)
    return out


def _warn(message):
    # Standard warnings so config typos are always visible (the project's
    # printing.warning is silent unless verbose mode is on). stacklevel=4
    # points the warning at the user's run_*(...) call site.
    warnings.warn(message, CavsimWarning, stacklevel=4)


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
        # A tighter cutoff than difflib's 0.6, which offered 'force' for 'freq'.
        match = difflib.get_close_matches(str(key), list(known_keys), n=1, cutoff=0.75)
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
        # Explicit None means "unset" under the complete-config convention, so only
        # a typed value is checked. A misspelled loss model must not fall through to
        # the default and silently return a perturbative Q for a lossy material.
        if cfg.get('loss_model') is not None:
            require(str(cfg['loss_model']).lower() in
                    ('auto', 'lossless', 'perturbation', 'lossy'),
                    f"eigenmode_config: 'loss_model'={cfg['loss_model']!r} must be one of "
                    "'auto' (default), 'lossless', 'perturbation' or 'lossy'.")
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
        for k in ('MT',):
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
        if isinstance(cfg.get('uq_config'), dict):
            validate_uq_config(cfg['uq_config'])


def validate_multipacting_config(cfg):
    validate_config(cfg, MULTIPACTING_KEYS, 'multipacting config')
    if isinstance(cfg, dict) and isinstance(cfg.get('eigenmode_config'), dict):
        validate_eigenmode_config(cfg['eigenmode_config'])
