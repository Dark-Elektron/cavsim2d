"""Default configs of every analysis, the samples users copy, and the merge that
completes a run's config."""
import copy
import difflib
import pprint
import warnings
from cavsim2d.utils.printing import CavsimWarning

from cavsim2d.utils.config_validation import (canonicalise, register_defaults,
                                              validate_eigenmode_config,
                                              validate_multipacting_config,
                                              validate_optimisation_config,
                                              validate_tune_config,
                                              validate_wakefield_config)
from cavsim2d.solvers.NGSolve.eigen_ngsolve import (DEFAULT_PINVIT_MAXIT, DEFAULT_PINVIT_TOL,
                                                    default_direct_solver)


def _documented(spec):
    """Split ``{key: (default, note)}`` into a default config and its notes. The
    notes, units included, are what a printed config sample shows beside each key."""
    return ({key: value for key, (value, _) in spec.items()},
            {key: note for key, (_, note) in spec.items()})


DEFAULT_TUNE_CONFIG, TUNE_NOTES = _documented({
    'freqs': (None, 'target [MHz]; None -> c/(4L) of the half cell, with a warning'),
    'cell_type': (None, "None -> {'mid-cell': 'Req'}, plus 'end-cell': 'L' for multicell"),
    'tol': (1e-4, 'stop once the frequency is this close to the target [MHz]'),
    'maxiter': (10, 'secant iterations per stage'),
    'eigenmode_config': (None, "mesh for the tuning solves, e.g. {'mesh_config': {'h': 6, 'p': 2}}"),
    'solve_tuned': (False, 'then run the eigenmode analysis on cav.tuned with eigenmode_config'),
    'qoi_targets': (None, "equality targets besides frequency, e.g. {'R/Q [Ohm]': 157}"),
    'equal_cell_freq': (False, '2-cell: one each-cell-at-f design instead of a family'),
    'uq_config': (None, "tune the mean over a UQ sweep (simplecell); see config_sample('uq')"),
    'processes': (1, 'worker processes'),
    'rerun': (True, 'False -> keep a finished tune of this geometry'),
})


# Every analysis has a complete default config. run() merges the user's input
# over it and saves the MERGED dict, so a saved config.json always records every
# setting a run used — not just the keys the user happened to pass. The values
# below mirror the solvers' effective defaults (None = resolved at run time,
# e.g. n_modes -> n_cells + 2).
DEFAULT_EIGENMODE_CONFIG, EIGENMODE_NOTES = _documented({
    'polarisation': ('monopole', "'monopole', 'dipole', ..., the azimuthal order m, or a list"),
    'n_modes': (None, 'None -> n_cells + 2'),
    'mode_of_interest': (None, 'None -> the accelerating pi-mode'),
    'boundary_conditions': ('mm', 'per pipe end: m magnetic, e electric, o open (PML), p port'),
    'f_shift': (0, 'spectral shift of the eigensolver'),
    'processes': (1, 'worker processes'),
    'rerun': (True, 'False -> reuse results computed for this geometry'),
    'conductivity': (5.96e7, 'wall conductivity [S/m]; copper'),
    'surface_resistance': (None, 'fixed surface resistance [Ohm] instead (e.g. SRF)'),
    'beta': (1.0, 'particle velocity v/c in the voltage (R/Q, Eacc, kick); 1 = ultra-relativistic'),
    'materials': (None, "{'quartz': {'eps_r': 3.8, 'tan_delta': 1e-4}} overrides a region "
                        "declared with cav.add_dielectric()"),
    'loss_model': (None, "None/'auto' -> from tan_delta; 'lossless', 'perturbation' or "
                         "'lossy' to force one"),
    'uq_config': (None, "a UQ sweep around this geometry; see config_sample('uq')"),
    'beampipe_length': (None, "[m]; None -> the model's own default (2*L_m)"),
    'pml_length': (None, 'PML block length [m]; None -> 3x the pipe radius'),
    'pml_alpha': (1j, "PML complex stretch (only with boundary_conditions='oo')"),
    'n_port_modes': (3, "TM_0n pipe modes per port (boundary_conditions='port')"),
    'mesh_config': ({'h': 20, 'p': 3, 'adaptive': None},
                    'h: largest element [mm]; p: polynomial order; adaptive: refinement'),
    'direct_solver': (default_direct_solver(), 'every factorisation of the eigensolve'),
    'preconditioner': ('direct', "PINVIT preconditioner: 'direct' or 'bddc'"),
    'pinvit_tol': (DEFAULT_PINVIT_TOL, "stop once every checked mode's residual is below"),
    'pinvit_converge_modes': (None, 'modes that must converge; None -> all n_modes'),
    'pinvit_padding': (None, 'extra vectors iterated on; None -> 2 x n_modes, at most 30'),
    'pinvit_maxit': (DEFAULT_PINVIT_MAXIT, 'iteration cap'),
})


DEFAULT_WAKEFIELD_CONFIG, WAKEFIELD_NOTES = _documented({
    'solver': ('abci', 'the wakefield backend'),
    'MROT': (2, '0 longitudinal, 1 transverse, 2 both'),
    'MT': (10, 'ABCI time steps per mesh cell (at most 19)'),
    'wakelength': (50, 'wake length [m]'),
    'bunch_length': (25, 'RMS bunch length [mm]'),
    'DDR_SIG': (0.1, 'radial mesh step as a fraction of the bunch length'),
    'DDZ_SIG': (0.1, 'axial mesh step as a fraction of the bunch length'),
    'beampipe_length': (None, '[m]; None -> 3x the axial length'),
    'save_fields': (None, "True or {'nshot': N} -> E-field-line snapshots"),
    'operating_points': (None, 'machine operating points for the loss and HOM power tables'),
    'uq_config': (None, "a UQ sweep around this geometry; see config_sample('uq')"),
    'processes': (1, 'worker processes'),
    'rerun': (True, 'False -> reuse results computed for this geometry'),
})


DEFAULT_MULTIPACTING_CONFIG, MULTIPACTING_NOTES = _documented({
    'mode': (0, 'which computed mode to track (0 = first)'),
    'polarisation': (None, "None -> monopole; 'dipole' etc. for the own field"),
    'n_modes': (None, 'how many modes to solve for the own field'),
    'xrange': (None, 'REQUIRED: emission-site z-interval [m]'),
    'n_points': (None, 'sites interpolated along the wall in-band'),
    'epks': (None, 'peak fields [V/m]; None -> 0-80 MV/m in 192 steps'),
    'phis': (None, 'launch phases [rad]; None -> 72 over [0, 2*pi]'),
    'v_init': (2, 'emission energy [eV]'),
    'step': (None, 'tracking time step'),
    'proc_count': (None, 'None -> automatic worker count'),
    'loss_model': ('field', 'how escaped particles are handled'),
    't_max': (1000e-10, 'track duration [s]'),
    'pec_maxh': (None, '[mm]; set -> multipacting solves its own field on this mesh'),
    'mesh_config': (None, "{'h' [mm], 'p'} of the own field mesh (with pec_maxh)"),
    'eigenmode_config': (None, 'eigenmode settings of the own field'),
    'progress': (True, 'live progress bar over the field-level sweep'),
})


def merge_config(defaults, *overrides):
    """Complete config = defaults, overridden left-to-right.

    One level of nested dicts (e.g. ``mesh_config``) is merged rather than
    replaced, so a user passing ``{'mesh_config': {'h': 25}}`` still gets the
    default ``p`` recorded — the saved config stays complete.
    """
    merged = dict(defaults)
    for ov in overrides:
        for k, v in (ov or {}).items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k] = {**merged[k], **v}
            else:
                merged[k] = v
    return merged


# A nested uq_config and an optimisation config have no defaults to merge over:
# their variables, bounds and objectives are the user's. These samples are
# complete, valid templates to copy and replace those in; every other value is
# what the code uses when the key is left out.
DEFAULT_UQ_CONFIG, UQ_NOTES = _documented({
    'variables': (['A', 'B', 'a', 'b'], "the model's own parameter names"),
    'delta': ([0.05, 0.05, 0.05, 0.05], 'spread per variable, added [mm]'),
    'distribution': ('gaussian', "or 'uniform' (delta = half-width)"),
    'method': (['Quadrature', 'Stroud3'], "or 'Stroud5', ['normal', N], ['lhs', N]"),
    'objectives': (['monopole:freq [MHz]', 'monopole:R/Q [Ohm]',
                    'monopole:Epk/Eacc []', 'monopole:Bpk/Eacc [mT/MV/m]'],
                   "'polarisation:QoI' of every quantity gathered"),
    'cell_type': ('mid-cell', 'simplecell: the cell block perturbed'),
    'cell_complexity': ('simplecell', "or 'multicell': every half-cell varies"),
    'independent_half_cells': (False, 'multicell: perturb, then weld the seams'),
    'tune_config': (None, 'retune every variant before measuring'),
    'processes': (1, 'worker processes'),
})


DEFAULT_OPTIMISATION_CONFIG, OPTIMISATION_NOTES = _documented({
    'bounds': ({'A': [20.0, 80.0], 'B': [20.0, 80.0], 'a': [10.0, 60.0],
                'b': [10.0, 60.0], 'Ri': [60.0, 85.0],
                'L': [93.5, 93.5], 'Req': [170.0, 170.0]},
               '[mm]; equal bounds fix a variable'),
    'objectives': ([['min', 'monopole:Epk/Eacc []'],
                    ['min', 'monopole:Bpk/Eacc [mT/MV/m]'],
                    ['max', 'monopole:R/Q [Ohm]']],
                   "[direction, 'polarisation:QoI'] per objective"),
    'tune_config': ({'freqs': 801.58, 'cell_type': {'mid-cell': 'Req'}},
                    'puts every candidate on frequency'),
    'initial_points': (5, 'candidates before the first generation'),
    'no_of_generation': (100, 'generations'),
    'method': ({'LHS': {'seed': 5}}, 'initial-point sampler'),
    'crossover_factor': (5, 'offspring from crossover per generation'),
    'mutation_factor': (5, 'offspring from mutation per generation'),
    'chaos_factor': (5, 'random newcomers per generation'),
    'elites_for_crossover': (2, 'best candidates kept as parents'),
    'processes': (1, 'worker processes'),
})


class ConfigSample(dict):
    """A default config: an ordinary dict, freshly copied, to edit and pass to
    ``run()``. Calling it returns another fresh copy, so
    ``cav.tune.config_sample()`` works as well as ``cav.tune.config_sample``.

    Displayed, it notes what each setting means and its unit beside it, and still
    reads as a Python dict: paste it into a script and edit it there."""

    def __init__(self, values=(), notes=None):
        super().__init__(values)
        self.notes = dict(notes or {})

    def __call__(self):
        return ConfigSample(copy.deepcopy(dict(self)), self.notes)

    def __repr__(self):
        if not self:
            return '{}'
        rows = []
        for key, value in self.items():
            head = f'    {key!r}: '
            text = pprint.pformat(value, width=max(40, 79 - len(head)), sort_dicts=False)
            lines = text.splitlines()
            lines = [head + lines[0]] + [' ' * len(head) + line for line in lines[1:]]
            lines[-1] += ','
            rows.append((lines, self.notes.get(key)))
        # Notes line up, except beside a long first line, which would push
        # them all off to the right.
        width = max((len(lines[0]) for lines, note in rows if note and len(lines[0]) <= 48),
                    default=0)
        out = []
        for lines, note in rows:
            if note:
                lines = [f'{lines[0]:<{width}}  # {note}'] + lines[1:]
            out.extend(lines)
        return '{\n' + '\n'.join(out) + '\n}'


class _ConfigSampleMixin:
    """``config_sample`` on a solver namespace, under every spelling people reach
    for: ``cfg_sample``, ``sample_config`` and ``sample_cfg`` are the same."""

    #: The kind of config ``run()`` takes (a key of :data:`CONFIG_SAMPLES`).
    _config_kind = None

    @property
    def config_sample(self):
        """A fresh copy of this solver's default config: every key with its
        default, ready to edit and pass to ``run()``. Unlike ``config`` (the saved
        config of a run that has happened), it needs no run."""
        return config_sample(self._config_kind)

    cfg_sample = sample_config = sample_cfg = config_sample


#: The default config of every kind of run, by the name ``config_sample`` takes.
CONFIG_SAMPLES = {
    'eigenmode': DEFAULT_EIGENMODE_CONFIG,
    'wakefield': DEFAULT_WAKEFIELD_CONFIG,
    'tune': DEFAULT_TUNE_CONFIG,
    'multipacting': DEFAULT_MULTIPACTING_CONFIG,
    'uq': DEFAULT_UQ_CONFIG,
    'optimisation': DEFAULT_OPTIMISATION_CONFIG,
}


#: What each key of a default config means, units included.
CONFIG_NOTES = {
    'eigenmode': EIGENMODE_NOTES,
    'wakefield': WAKEFIELD_NOTES,
    'tune': TUNE_NOTES,
    'multipacting': MULTIPACTING_NOTES,
    'uq': UQ_NOTES,
    'optimisation': OPTIMISATION_NOTES,
}


# Every key of a default config is one the validator accepts.
for _kind, _defaults in CONFIG_SAMPLES.items():
    register_defaults(_kind, _defaults)


def config_sample(kind, show=False):
    """A fresh copy of the default config for *kind* (one of
    :data:`CONFIG_SAMPLES`), printed with what each key means when *show*."""
    key = str(kind).strip().lower()
    key = {'optimization': 'optimisation', 'tuning': 'tune'}.get(key, key)
    if key not in CONFIG_SAMPLES:
        close = difflib.get_close_matches(key, list(CONFIG_SAMPLES), n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ''
        raise ValueError(f"No config sample for {kind!r}. Choose one of: "
                         f"{', '.join(CONFIG_SAMPLES)}.{hint}")
    sample = ConfigSample(copy.deepcopy(CONFIG_SAMPLES[key]), CONFIG_NOTES[key])
    if show:
        print(f"# default {key} config - every available option, with its default\n"
              f"{sample!r}")
    return sample


_VALIDATORS = {'eigenmode': validate_eigenmode_config,
               'wakefield': validate_wakefield_config,
               'tune': validate_tune_config,
               'multipacting': validate_multipacting_config,
               'optimisation': validate_optimisation_config}


def _user_config(kind, config, kwargs):
    """The settings a caller gave a *kind* run: *config* overridden by *kwargs*,
    aliases renamed to the keys the code reads, unknown keys warned about."""
    config = canonicalise(config, kind)
    kwargs = canonicalise(kwargs, kind)
    clash = sorted(k for k in set(config) & set(kwargs) if config[k] != kwargs[k])
    if clash:
        warnings.warn(f"{kind} setting(s) {clash} given both in the config and as "
                      f"keyword arguments; the keyword value(s) are used: "
                      f"{ {k: kwargs[k] for k in clash} }", CavsimWarning, stacklevel=3)
    user = {**config, **kwargs}
    _VALIDATORS[kind](user)
    return user
