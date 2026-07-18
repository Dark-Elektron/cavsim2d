"""Solver-as-object wrappers for cavity simulation workflows.

Each solver class manages its own folder, configuration, and results.
Solvers are attached to Cavity or Cavities instances as lazy properties.
"""
import json
import os
import pickle
import shutil
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.analysis.impedance import (NATIVE_Z_UNIT, convert_impedance_frame,
                                         frame_unit, impedance_frame,
                                         impedance_unit, prefix_factor,
                                         reconstruct_impedance)
from cavsim2d.analysis.multipacting.sey import SEY
from cavsim2d.analysis.multipacting import metrics as mp_metrics
from cavsim2d.solvers.eigenmode_result import (EigenmodeResult, MPOLE_NAMES, pol_name,
                                               pol_number, monopole_dir)
from cavsim2d.utils.printing import done, error, info, suppress_errors
from cavsim2d.utils.style import house_style, WARM
from cavsim2d.processes.eigenmode import run_eigenmode_parallel, run_eigenmode_s
from cavsim2d.processes.wakefield import run_wakefield_parallel, run_wakefield_s
from cavsim2d.solvers.wakefield import get_backend
from itertools import combinations
from cavsim2d.utils.printing import warning

DEFAULT_TUNE_CONFIG = {
    'freqs': 1300,
    'parameters': 'A',
    'cell_types': 'mid-cell',
    'processes': 1,
    'rerun': True,
}

# Every analysis has a complete default config. run() merges the user's input
# over it and saves the MERGED dict, so a saved config.json always records every
# setting a run used — not just the keys the user happened to pass. The values
# below mirror the solvers' effective defaults (None = resolved at run time,
# e.g. n_modes -> n_cells + 2).
DEFAULT_EIGENMODE_CONFIG = {
    'polarisation': 'monopole',
    'n_modes': None,                 # None -> n_cells + 2
    'mode_of_interest': None,        # None -> the accelerating pi-mode
    'boundary_conditions': 'mm',
    'f_shift': 0,
    'processes': 1,
    'rerun': True,
    'conductivity': 5.96e7,          # copper [S/m]
    'surface_resistance': None,      # fixed Rs override (e.g. SRF)
    'uq_config': None,
    'mesh_config': {'h': 20, 'p': 3, 'adaptive': None},
}

DEFAULT_WAKEFIELD_CONFIG = {
    'solver': 'abci',
    'MROT': 2,                       # 0 longitudinal, 1 transverse, 2 both
    'MT': 10,
    'NFS': 10000,
    'wakelength': 50,                # [m]
    'bunch_length': 25,              # [mm]
    'DDR_SIG': 0.1,
    'DDZ_SIG': 0.1,
    'beampipe_length': None,         # [m]; None -> 3x the axial length
    'operating_points': None,
    'uq_config': None,
    'processes': 1,
    'rerun': True,
}

DEFAULT_MULTIPACTING_CONFIG = {
    'mode': 0,                       # fundamental (accelerating) mode
    'xrange': None,                  # None -> a band at the equator
    'epks': None,                    # None -> 0-80 MV/m in 192 steps [V/m]
    'phis': None,                    # None -> 72 phases over [0, 2*pi]
    'v_init': 2,                     # emission energy [eV]
    'step': None,
    'proc_count': None,              # None -> auto worker count
    'loss_model': 'field',
    't_max': 1000e-10,               # track duration [s]
    'pec_maxh': None,                # [mm]; set -> multipacting-owned field
    'mesh_config': None,             # {'h','p'} for the OWN field mesh (with pec_maxh)
    'eigenmode_config': None,
    'progress': True,                # live tqdm bar over the field-level sweep
}


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

def _z_axis_label(unit, transverse=False):
    """Mathtext y-label for an impedance axis, e.g. r'$|Z_\\parallel|$ [k$\\Omega$]'."""
    _, prefix = prefix_factor(unit)
    if transverse:
        return rf'$|Z_\perp|$ [{prefix}$\Omega$/m]'
    return rf'$|Z_\parallel|$ [{prefix}$\Omega$]'


def _maybe_show(show):
    """Display the current figure when *show* is True.

    Every ``plot_*`` method defaults to ``show=True`` so a plot appears without
    a manual ``plt.show()`` (the common interactive case, and notebooks with the
    non-interactive ``inline`` backend only render on show). Pass ``show=False``
    to keep the axes live for overlaying/compositing or to reuse the returned
    ``ax``; the study-level overlays pass it through for exactly that reason."""
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# TuneSolver
# ---------------------------------------------------------------------------

class TuneSolver:
    """Manages tuning artefacts under ``<cavity>/tuned/tune_info/``.

    After a successful run, ``cavity.tuned`` becomes the fresh
    :class:`Cavity` instance living in ``<cavity>/tuned/`` — subsequent
    ``cavity.tuned.run_eigenmode()`` / ``cavity.tuned.run_wakefield()``
    calls write into the tuned cavity's own folders.

    Access
    ------
    - ``cav.tune.config``         — tuning configuration dict
    - ``cav.tune.qois``           — ``tune_res.json`` contents
    - ``cav.tune.convergence``    — iteration‐history DataFrame
    - ``cav.tune.plot_convergence()``
    """

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._qois = None
        self._convergence = None

    @property
    def folder(self):
        """Directory holding tune artefacts (``<cav>/tuned/tune_info/``)."""
        return Path(self.cavity.self_dir) / 'tuned' / 'tune_info'

    # -- Lazy‐loaded results ------------------------------------------------

    @property
    def config(self):
        if self._config is None:
            cfg_path = self.folder / 'config.json'
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    @property
    def qois(self):
        """``tune_res.json`` contents: the tuned parameters and reached
        frequency, keyed by cell type."""
        if self._qois is None:
            res_path = self.folder / 'tune_res.json'
            if res_path.exists():
                with open(res_path, 'r') as f:
                    self._qois = json.load(f)
            else:
                self._qois = {}
        return self._qois

    @property
    def convergence(self):
        """Tidy long-form convergence history: one row per (stage, iteration,
        parameter), where stage is 'cell_type:tune_var'. Long-form rather than
        wide because different tune stages (e.g. mid-cell then end-cell) can
        have different iteration counts, which a wide table can't hold cleanly."""
        if self._convergence is None:
            conv_path = self.folder / 'tune_convergence.json'
            if conv_path.exists():
                with open(conv_path, 'r') as f:
                    data = json.load(f)
                rows = []
                for ct_name, tune_vars in data.items():
                    for tune_var, history in tune_vars.items():
                        if not isinstance(history, dict):
                            continue
                        n_iter = max((len(v) for v in history.values()
                                     if isinstance(v, list)), default=0)
                        for it in range(n_iter):
                            for param, values in history.items():
                                if it < len(values):
                                    rows.append({
                                        'stage': f'{ct_name}:{tune_var}',
                                        'iteration': it,
                                        'parameter': param,
                                        'value': values[it],
                                    })
                self._convergence = pd.DataFrame(rows)
            else:
                self._convergence = pd.DataFrame()
        return self._convergence

    # -- Actions ------------------------------------------------------------

    def run(self, tune_config=None, **kwargs):
        """Run frequency tuning. Delegates to processes/tune.py.

        Parameters
        ----------
        tune_config : dict, optional
            Full tune configuration dict. Merged over DEFAULT_TUNE_CONFIG.
        **kwargs
            Individual tune arguments (e.g. freqs=1300, parameters='A').
            Merged in last, so they take final precedence. Keys present in
            both tune_config and kwargs trigger a warning.
        """
        # Deferred: breaks the solver_objects <-> processes.tune import cycle.
        from cavsim2d.processes.tune import run_tune_parallel

        tune_config = dict(tune_config) if tune_config else {}

        overlap = set(tune_config) & set(kwargs)
        if overlap:
            warnings.warn(
                f"Tune argument(s) {sorted(overlap)} passed in both tune_config "
                f"and as keyword arguments; the keyword argument value(s) will "
                f"be used: { {k: kwargs[k] for k in overlap} }",
                stacklevel=2,
            )

        merged_config = {**DEFAULT_TUNE_CONFIG, **tune_config, **kwargs}

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed
        self.folder.mkdir(parents=True, exist_ok=True)

        # Save config first so it's available even if tuning crashes
        with open(self.folder / 'config.json', 'w') as f:
            json.dump(merged_config, f, indent=4, default=str)

        self._config = merged_config
        self._invalidate()

        cavs_dict = {self.cavity.name: self.cavity}
        # Degenerate-geometry errors are expected noise from the secant
        # tuner's probe iterations — silence them here so users only see
        # genuine failures.
        with suppress_errors('Parameter set leads to degenerate geometry'):
            run_tune_parallel(cavs_dict, merged_config)

        # Load the tuned cavity from the newly-written folder
        self.cavity._tuned_cavity = None  # force reload via property
        _ = self.cavity.tuned

        self._invalidate()

    def _invalidate(self):
        """Clear cached results so they are re‐read from disk."""
        self._qois = None
        self._convergence = None

    # -- Visualisation ------------------------------------------------------

    def plot_convergence(self, show=True):
        """One subplot per tuned parameter (e.g. Req_m, freq [MHz]), with one
        line per tune stage. Returns (fig, axes)."""
        df = self.convergence
        if df.empty:
            info("No convergence data available.")
            return None, None

        params = list(df['parameter'].unique())
        fig, axes = plt.subplots(len(params), 1, figsize=(6, 3 * len(params)),
                                 squeeze=False)
        axes = axes[:, 0]

        for ax, param in zip(axes, params):
            for stage, sub in df[df['parameter'] == param].groupby('stage'):
                ax.plot(sub['iteration'], sub['value'], marker='o', label=stage)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param)
            ax.legend()

        fig.suptitle(f'Tuning Convergence — {self.cavity.name}')
        plt.tight_layout()
        _maybe_show(show)
        return fig, axes


# ---------------------------------------------------------------------------
# EigenmodeSolver
# ---------------------------------------------------------------------------

class EigenmodeSolver:
    """Manages ``<cavity>/eigenmode/`` — eigenmode analysis.

    Access
    ------
    - ``cav.eigenmode.config``         — solver configuration
    - ``cav.eigenmode.modes``          — list of all EigenmodeResult objects
    - ``cav.eigenmode[i]``             — i‑th mode
    - ``cav.eigenmode[i].frequency``
    - ``cav.eigenmode[i].m``
    - ``cav.eigenmode[i].qois``
    - ``cav.eigenmode.plot_spectrum()``
    """

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._modes = None
        self._qois = None

    @property
    def folder(self):
        return Path(self.cavity.self_dir) / 'eigenmode'

    # -- Lazy‐loaded results ------------------------------------------------

    @property
    def config(self):
        if self._config is None:
            cfg_path = self.folder / 'config.json'
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    @property
    def qois(self):
        """Return QOIs for the fundamental (pi‐mode) — backward compat."""
        if self._qois is None:
            qoi_path = self.pol_folder(0) / 'qois.json'
            if qoi_path.exists():
                with open(qoi_path, 'r') as f:
                    self._qois = json.load(f)
            else:
                self._qois = {}
        return self._qois

    @property
    def modes(self):
        """All computed eigenmodes as a list of EigenmodeResult."""
        if self._modes is None:
            modes_path = self.pol_folder(0) / 'modes.json'
            if modes_path.exists():
                with open(modes_path, 'r') as f:
                    data = json.load(f)
                self._modes = [EigenmodeResult.from_dict(d) for d in data]
            else:
                # Fallback: build from qois_all_modes.json (legacy format)
                all_path = self.pol_folder(0) / 'qois_all_modes.json'
                if all_path.exists():
                    with open(all_path, 'r') as f:
                        data = json.load(f)
                    self._modes = []
                    for idx_str, qois in data.items():
                        self._modes.append(EigenmodeResult(
                            index=int(idx_str),
                            frequency=qois.get('freq [MHz]', 0),
                            m=0,  # monopole by default for legacy data
                            qois=qois,
                        ))
                else:
                    self._modes = []
        return self._modes

    def __getitem__(self, i):
        return self.modes[i]

    def __len__(self):
        return len(self.modes)

    # -- m-pole (dipole, quadrupole, ...) results ----------------------------

    def pol_folder(self, pol):
        """Folder holding results for a polarisation ('monopole', 'dipole',
        2, ...): ``eigenmode/<pol name>/``. Monopole falls back to the
        legacy flat ``eigenmode/`` layout when no subfolder exists."""
        m = pol_number(pol)
        if m == 0:
            return Path(monopole_dir(self.folder))
        return self.folder / pol_name(m)

    def mpole_qois(self, pol, all_modes=True):
        """QOIs of an m-pole solve, keyed by mode index (``all_modes=True``)
        or just the fundamental mode's dict."""
        fname = 'qois_all_modes.json' if all_modes else 'qois.json'
        path = self.pol_folder(pol) / fname
        if not path.exists():
            info(f"No {pol_name(pol_number(pol))} eigenmode results found at {path}. "
                 f"Run eigenmode analysis with 'polarisation' set in the config.")
            return {}
        with open(path, 'r') as f:
            return json.load(f)

    def mpole_modes(self, pol):
        """All computed m-pole eigenmodes as a list of EigenmodeResult."""
        m = pol_number(pol)
        data = self.mpole_qois(pol, all_modes=True)
        return [EigenmodeResult(index=int(idx), frequency=q.get('freq [MHz]', 0),
                                m=m, qois=q)
                for idx, q in data.items()]

    @property
    def qois_df(self):
        """Every QOI, every mode, every polarisation — as one DataFrame.

        The dict-of-dicts results (``qois``, ``mpole_qois``) are convenient to
        look up but awkward to filter and plot across polarisations. This is the
        same data in the layout ``convergence_df_data`` uses, so it slices the
        same way: one row per (polarisation, mode), every QOI as a column, plus

        - ``m``            azimuthal mode number (0 monopole, 1 dipole, ...)
        - ``polarisation`` its name
        - ``mode``         mode index within that polarisation
        - ``mode_index``   ``'<m>-<mode>'`` — '0-0' monopole fundamental,
                           '1-0' dipole fundamental

        Only polarisations that have actually been solved appear. Empty if the
        cavity has no eigenmode results yet.

        >>> df = cav.eigenmode.qois_df
        >>> df[df.polarisation == 'dipole'].plot('freq [MHz]', 'R/Q [Ohm]')
        """
        rows = []
        for m in sorted(MPOLE_NAMES):
            path = self.pol_folder(m) / 'qois_all_modes.json'
            if not path.exists():
                continue
            with open(path, 'r') as f:
                data = json.load(f)
            for idx, q in sorted(data.items(), key=lambda kv: int(kv[0])):
                row = dict(q)
                row['m'] = row.get('m', m)
                row['polarisation'] = row.get('polarisation', pol_name(m))
                row['mode'] = int(idx)
                row['mode_index'] = f"{m}-{int(idx)}"
                rows.append(row)
        return pd.DataFrame(rows)

    # -- Derived quantities --------------------------------------------------

    def impedance(self, kind='longitudinal', span=None, n_points=8001, Q=None,
                  unit='k'):
        """Impedance spectrum reconstructed from the computed modes.

        Each mode is treated as a parallel RLC resonator with shunt impedance
        ``R = 1/2 Q (R/Q)``, and their contributions are summed. The result uses
        the **same columns as** ``cav.wakefield.wake_z`` / ``wake_t``
        (``f [MHz]``, ``|Z|``, ``Re(Z)``, ``Im(Z)``), so it overlays a wakefield
        solve — or a beam spectrum — directly.

        This is a reconstruction, not a simulation: it contains exactly the modes
        that were solved and nothing else. There is no broadband or
        resistive-wall contribution, and nothing above the highest computed mode.
        Ask for enough modes to cover the band you care about.

        Parameters
        ----------
        kind : {'longitudinal', 'transverse'}
            Longitudinal [Ohm] from the monopole modes, or transverse [Ohm/m]
            from the dipole modes (the cavity is axisymmetric, so the two
            transverse planes are degenerate and one R/Q describes both).
        span : (float, float), optional
            Frequency range **in MHz**. Defaults to ``(0, highest computed mode)``.
        n_points : int
            Size of the *uniform* part of the frequency grid. Each resonance is
            additionally sampled across its own linewidth and merged in, so the
            peaks reach their true height ``R = 1/2 Q (R/Q)`` regardless of this
            number — a uniform grid alone would step over them (a Q of 1e4 gives
            a peak only ~f/1e4 wide) and understate them badly. The returned
            grid is therefore not evenly spaced.
        Q : float or sequence, optional
            Override the quality factor: a scalar applies to every mode, a
            sequence gives one per mode. **Defaults to the solver's unloaded Q0**
            (set by the wall material). If the beam sees a loaded/external Q,
            pass it — the peak heights scale with it.
        unit : {'k', '', 'M', 'G'}
            SI prefix for the impedance: ``'k'`` for kOhm (the default, which
            matches what the wakefield solver reports), ``''`` for Ohm, ``'M'``
            for MOhm.

        Returns
        -------
        pandas.DataFrame
            Columns ``f [MHz]``, ``|Z| [u]``, ``Re(Z) [u]``, ``Im(Z) [u]`` with
            ``u`` = kOhm (longitudinal) or kOhm/m (transverse) by default.

        >>> z = cav.eigenmode.impedance()                       # 0 .. highest mode
        >>> zt = cav.eigenmode.impedance('transverse', span=(400, 1200), Q=1e4)
        >>> z = cav.eigenmode.impedance(unit='')                # in Ohm instead
        """
        m = 0 if str(kind).lower().startswith('long') else 1
        transverse = m == 1

        df = self.qois_df
        modes = df[df['m'] == m] if not df.empty else df
        if modes.empty:
            what = 'dipole' if transverse else 'monopole'
            error(f"No {what} eigenmode results to build a {kind} impedance from. "
                  f"Run cav.eigenmode.run({{'polarisation': '{what}'}}) first.")
            return pd.DataFrame()

        f0 = modes['freq [MHz]'].to_numpy(dtype=float) * 1e6        # -> Hz
        roq = modes['R/Q [Ohm]'].to_numpy(dtype=float)
        if Q is None:
            q = modes['Q []'].to_numpy(dtype=float)
        else:
            q = np.broadcast_to(np.asarray(Q, dtype=float), f0.shape).copy()

        if span is None:
            span = (0.0, float(modes['freq [MHz]'].max()))
        f_lo, f_hi = (float(s) * 1e6 for s in span)                  # -> Hz
        if f_hi <= f_lo:
            raise ValueError(f"span must be increasing; got {span}.")

        # A uniform grid steps straight over the resonances: a mode of Q = 1e4 at
        # 1 GHz is only ~100 kHz wide, so an evenly spaced grid can land either
        # side of the peak and understate it by any factor you like. Sample each
        # mode's line shape explicitly and merge that into the uniform grid, so
        # the peaks come out at their true height whatever n_points is.
        grids = [np.linspace(f_lo, f_hi, int(n_points))]
        for fi, qi in zip(f0, q):
            if not (f_lo <= fi <= f_hi):
                continue
            half_width = fi / max(abs(qi), 1.0)
            grids.append(np.linspace(max(fi - 10 * half_width, f_lo),
                                     min(fi + 10 * half_width, f_hi), 101))
        f_span = np.unique(np.concatenate(grids))

        z = reconstruct_impedance(f0, roq, q, f_span, transverse=transverse)
        return impedance_frame(f_span * 1e-6, z, unit=unit, transverse=transverse)

    def plot_impedance(self, kind='longitudinal', ax=None, span=None,
                       n_points=8001, Q=None, unit='k', show=True, **kwargs):
        """Plot the reconstructed impedance (see :meth:`impedance`).

        Returns the axes, so a wakefield result can be overlaid on the same one —
        both default to kOhm, so the two land on the same scale (pass
        ``show=False`` on the first call to keep the axes live)::

            ax = cav.eigenmode.plot_impedance(show=False)
            cav.wakefield.plot_impedance(ax=ax)
        """
        df = self.impedance(kind=kind, span=span, n_points=n_points, Q=Q, unit=unit)
        if df.empty:
            return ax
        transverse = str(kind).lower().startswith('trans')
        u = impedance_unit(unit, transverse)
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('label', f'{self.cavity.name} (eigenmode)')
            ax.plot(df['f [MHz]'], df[f'|Z| [{u}]'], **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(_z_axis_label(unit, transverse))
            ax.set_yscale('log')
        _maybe_show(show)
        return ax

    # -- Actions ------------------------------------------------------------

    def run(self, eigenmode_config=None, **kwargs):
        """Run eigenmode analysis. Delegates to processes/eigenmode.py.

        The user's config is merged over :data:`DEFAULT_EIGENMODE_CONFIG` and
        the **merged** dict is what runs and what is saved to ``config.json`` —
        the saved config always records every setting the run used. Any config
        key can also be passed as a keyword argument
        (``run(mesh_config={'h': 10})``); kwargs override the config dict.
        """
        merged_config = merge_config(DEFAULT_EIGENMODE_CONFIG, eigenmode_config, kwargs)

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed
        self.folder.mkdir(parents=True, exist_ok=True)

        # Save config first so it's available even if the solve crashes
        with open(self.folder / 'config.json', 'w') as f:
            json.dump(merged_config, f, indent=4, default=str)

        self._config = merged_config

        eigenmode_config_copy = merged_config.copy()
        eigenmode_config_copy['target'] = run_eigenmode_s

        cavs_dict = {self.cavity.name: self.cavity}
        run_eigenmode_parallel(cavs_dict, eigenmode_config_copy)

        self._invalidate()

    def _invalidate(self):
        self._modes = None
        self._qois = None

    # -- Visualisation ------------------------------------------------------

    def plot_spectrum(self, ax=None, show=True):
        """Plot frequency spectrum of all modes."""
        modes = self.modes
        if not modes:
            info("No eigenmode data available.")
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        freqs = [m.frequency for m in modes if m.frequency > 0]
        indices = list(range(len(freqs)))
        ax.stem(indices, freqs)
        ax.set_xlabel('Mode index')
        ax.set_ylabel('Frequency [MHz]')
        ax.set_title(f'Eigenmode spectrum — {self.cavity.name}')
        _maybe_show(show)
        return ax


# ---------------------------------------------------------------------------
# WakefieldSolver
# ---------------------------------------------------------------------------

class WakefieldSolver:
    """Manages ``<cavity>/wakefield/`` — wakefield / impedance analysis.

    Access
    ------
    - ``cav.wakefield.config``
    - ``cav.wakefield.wake_z``   — longitudinal wake DataFrame
    - ``cav.wakefield.wake_t``   — transverse wake DataFrame
    - ``cav.wakefield.impedance(kind, unit=...)`` — impedance frame
    - ``cav.wakefield.plot_impedance()`` / ``plot_wake()``
    - ``cav.wakefield.plot_k_loss()`` / ``plot_k_kick()``
    """

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._result = None
        self._qois = None

    @property
    def folder(self):
        return Path(self.cavity.self_dir) / 'wakefield'

    @property
    def config(self):
        if self._config is None:
            self._config = {}
            if self.cavity.self_dir:
                cfg_path = self.folder / 'config.json'
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        self._config = json.load(f)
        return self._config

    @property
    def backend(self):
        """The wakefield backend for this cavity (from ``config['solver']``,
        default ``'abci'``). All reads go through it, so the result schema is the
        same whatever solver produced it."""
        return get_backend(self.config.get('solver', 'abci'))

    @property
    def result(self):
        """The normalised :class:`WakefieldResult` (frames + qois), read via the
        backend and cached."""
        if self._result is None:
            self._result = self.backend.read(self.cavity)
        return self._result

    @property
    def wake_z(self):
        """Longitudinal impedance spectrum + wake potential as a DataFrame."""
        return self.result.wake_z

    @property
    def wake_t(self):
        """Transverse impedance spectrum + wake potential as a DataFrame."""
        return self.result.wake_t

    @property
    def qois(self):
        """The main run's normalised scalar QOIs — always reported: loss/kick
        factors ``|k_loss| [V/pC]``, ``k_FM [V/pC]``, ``k_loss_HOM [V/pC]``,
        ``|k_kick| [V/pC/m]``."""
        if self._qois is None:
            self._qois = self.backend.read_qois(self.cavity)
        return self._qois

    @property
    def qois_op(self):
        """The operating-point QOIs (per operating point / bunch length: P_HOM,
        loss/kick, …), produced only when ``operating_points`` were given. Empty
        dict otherwise. Distinct from :attr:`qois`, the main-run factors."""
        return self.backend.read_qois_op(self.cavity)

    # -- Actions ------------------------------------------------------------

    def run(self, wakefield_config=None, **kwargs):
        """Run wakefield analysis. Delegates to processes/wakefield.py.

        The user's config is merged over :data:`DEFAULT_WAKEFIELD_CONFIG` and the
        **merged** dict is what runs and what is saved to ``config.json`` — the
        saved config always records every setting the run used. Behaviour is
        unchanged: the defaults mirror the effective ones. (Known quirk, not
        changed here: the live ABCI deck writer reads ``MT`` from
        ``wake_config['MT']`` — top-level ``MT``/``NFS`` are recorded but only
        the legacy writer consumed them.) Any config key can also be passed as
        a keyword argument (``run(wakelength=80)``); kwargs override the
        config dict.
        """
        merged_config = merge_config(DEFAULT_WAKEFIELD_CONFIG, wakefield_config, kwargs)

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed
        self.folder.mkdir(parents=True, exist_ok=True)

        with open(self.folder / 'config.json', 'w') as f:
            json.dump(merged_config, f, indent=4, default=str)

        self._config = merged_config

        wakefield_config_copy = merged_config.copy()
        wakefield_config_copy['target'] = run_wakefield_s

        cavs_dict = {self.cavity.name: self.cavity}
        run_wakefield_parallel(cavs_dict, wakefield_config_copy)

        self._invalidate()

    def _invalidate(self):
        self._result = None
        self._qois = None

    # -- Visualisation ------------------------------------------------------

    def impedance(self, kind='longitudinal', unit='k'):
        """The wakefield impedance spectrum, in the requested unit.

        The mirror of :meth:`EigenmodeSolver.impedance`, so the simulated and the
        mode-reconstructed spectra come back in the same columns and the same
        unit and can be compared directly.

        Parameters
        ----------
        kind : {'longitudinal', 'transverse'}
        unit : {'k', '', 'M', 'G'}
            SI prefix: ``'k'`` for kOhm (default — ABCI's native unit), ``''``
            for Ohm, ``'M'`` for MOhm.

        Returns
        -------
        pandas.DataFrame
            ``f [MHz]``, ``|Z| [u]``, ``Re(Z) [u]``, ``Im(Z) [u]``.
        """
        transverse = str(kind).lower().startswith('trans')
        df = self._frame(kind)
        if df is None or df.empty:
            info("No wakefield data available — run cav.wakefield.run(...) first.")
            return pd.DataFrame()
        cols = ['f [MHz]'] + [c for c in df.columns
                              if c.startswith(('|Z|', 'Re(Z)', 'Im(Z)'))]
        # take the source unit from what the frame declares, not from a hardcoded
        # assumption — backends may report in Ohm or kOhm
        out = convert_impedance_frame(df[cols].dropna(), frame_unit(df), unit,
                                      transverse=transverse)
        return out.reset_index(drop=True)

    def plot_impedance(self, kind='longitudinal', ax=None, unit='k', show=True,
                       **kwargs):
        """Plot the wakefield impedance: |Z| vs f.

        The same call as :meth:`EigenmodeSolver.plot_impedance`, in the same
        default unit (kOhm), so the two overlay without a scale factor (pass
        ``show=False`` on the first call to keep the axes live)::

            ax = cav.eigenmode.plot_impedance(show=False)  # from the modes
            cav.wakefield.plot_impedance(ax=ax)            # from the wake solve
        """
        transverse = str(kind).lower().startswith('trans')
        df = self.impedance(kind, unit=unit)
        if df.empty:
            return ax
        u = impedance_unit(unit, transverse)
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('label', f'{self.cavity.name} (wakefield)')
            ax.plot(df['f [MHz]'], df[f'|Z| [{u}]'], **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(_z_axis_label(unit, transverse))
            ax.margins(x=0)
        _maybe_show(show)
        return ax

    def plot_wake(self, kind='longitudinal', ax=None, show=True, **kwargs):
        """Plot the wake potential: W vs s (distance behind the bunch head)."""
        return self._plot(self._frame(kind), ax, 'wake', show=show,
                          **self._labels(kind), **kwargs)

    def plot_k_loss(self, ax=None, show=True, **kwargs):
        """Plot the cumulative loss-factor spectrum :math:`k_{\\mathrm{loss}}(F)`
        vs frequency — ABCI's 'Loss Factor Spectrum Integrated upto F', the
        running loss factor accumulated up to each frequency.

        This is the frequency-resolved spectrum, not the single cumulative
        number ``cav.wakefield.qois['|k_loss| [V/pC]']`` (that scalar is what the
        study comparison bars, e.g. ``plot_compare_hom_bar``, show)."""
        return self._plot_kspectrum('longitudinal', ax, show=show, **kwargs)

    def plot_k_kick(self, ax=None, show=True, **kwargs):
        """Plot the cumulative kick-factor spectrum :math:`k_{\\mathrm{kick}}(F)`
        vs frequency (the transverse analogue of :meth:`plot_k_loss`)."""
        return self._plot_kspectrum('transverse', ax, show=show, **kwargs)

    def _plot_kspectrum(self, kind, ax, show=True, **kwargs):
        transverse = str(kind).lower().startswith('trans')
        stem = 'k_kick(f)' if transverse else 'k_loss(f)'
        df = self._frame(kind)
        kcol = next((c for c in (df.columns if df is not None else [])
                     if c.startswith(stem)), None)
        if df is None or df.empty or kcol is None or 'fk [MHz]' not in df.columns:
            info("No loss-factor spectrum available — run cav.wakefield.run(...) "
                 "first (the backend must provide 'Loss Factor Spectrum Integrated "
                 "upto F').")
            return ax
        sub = df[['fk [MHz]', kcol]].dropna().sort_values('fk [MHz]')
        label = (r'$k_{\mathrm{kick}}(F)$ [V/pC/m]' if transverse
                 else r'$k_{\mathrm{loss}}(F)$ [V/pC]')
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('label', self.cavity.name)
            kwargs.setdefault('color', getattr(self.cavity, 'color', None) or WARM[0])
            ax.plot(sub['fk [MHz]'], sub[kcol], **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(label)
            ax.margins(x=0)
        _maybe_show(show)
        return ax

    def _frame(self, kind):
        return self.wake_t if str(kind).lower().startswith('trans') else self.wake_z

    def _labels(self, kind):
        """Plot labels + the frame's own |Z| column, resolved from what the frame
        declares rather than a hardcoded unit (which is how the [Ohm]/kOhm
        mislabel survived unnoticed)."""
        transverse = str(kind).lower().startswith('trans')
        u = frame_unit(self._frame(kind))
        zcol = f'|Z| [{impedance_unit(u, transverse)}]'
        if transverse:
            return dict(impedance_cols=('f [MHz]', zcol),
                        wake_cols=('s [m]', 'W [V/pC/m]'),
                        ylabels=(_z_axis_label(u, True), r'$W_\perp$ [V/pC/m]'),
                        title='Transverse')
        return dict(impedance_cols=('f [MHz]', zcol),
                    wake_cols=('s [m]', 'W [V/pC]'),
                    ylabels=(_z_axis_label(u, False), r'$W_\parallel$ [V/pC]'),
                    title='Longitudinal')

    def _plot(self, df, ax, quantity, impedance_cols, wake_cols, ylabels, title,
              show=True, **kwargs):
        if df is None or df.empty:
            info("No wakefield data available — run cav.wakefield.run(...) first.")
            return ax
        if quantity == 'wake':
            xcol, ycol, ylabel, xlabel = wake_cols[0], wake_cols[1], ylabels[1], 'S [m]'
        else:
            xcol, ycol, ylabel, xlabel = impedance_cols[0], impedance_cols[1], ylabels[0], 'f [MHz]'
        sub = df[[xcol, ycol]].dropna()

        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('label', self.cavity.name)
            ax.plot(sub[xcol], sub[ycol], **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{title} {quantity}')
            ax.margins(x=0)
        _maybe_show(show)
        return ax


# ---------------------------------------------------------------------------
# MultipactingSolver
# ---------------------------------------------------------------------------

class MultipactingSolver:
    """Manages ``<cavity>/multipacting/`` — multipacting analysis.

    Tracks emitted electrons in the cavity's eigenmode field (with secondary
    emission) over a peak-field sweep; the counter function ``c20/c0`` shows the
    field levels where multipacting resonates. The EM field is read from the
    cavity's eigenmode result — ``run()`` solves the (monopole) eigenmode first
    if it has not been run.

    Access
    ------
    - ``cav.multipacting.run(config)``   — run the peak-field sweep
    - ``cav.multipacting.set_sey(path)`` / ``cav.multipacting.sey``
    - ``cav.multipacting.counter`` / ``.epk`` / ``.final_energy`` / ``.results``
    - ``cav.multipacting.plot_counter()`` / ``plot_final_energy()`` /
      ``plot_enhanced_counter()`` / ``plot_distance_map(i)`` /
      ``plot_trajectories()`` / ``animate_trajectories()``
    """

    #: Eigenmode config for the auto-run: monopole, magnetic end-planes, and a
    #: near-wall-adequate mesh (~6 mm, matching the validated PyMultipact setup).
    DEFAULT_EIGENMODE = {'polarisation': 'monopole', 'boundary_conditions': 'mm',
                         'mesh_config': {'h': 6, 'p': 3}}

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._results = None
        self._sey = None
        self._staged = {}          # settings staged via set_* before run()

    @property
    def folder(self):
        return Path(self.cavity.self_dir) / 'multipacting'

    # -- Stepwise setup -------------------------------------------------------
    # Everything can be bundled into run(config), but each step can also be set
    # (and inspected) individually beforehand: stage the mesh, look at it, stage
    # the emission band, look at the emission points, THEN run. run() merges the
    # staged values under its config — and warns when a config entry overwrites
    # a previously staged one.

    def set_mesh_parameters(self, h=None, p=None, pec_maxh=None):
        """Stage the multipacting field mesh: global size ``h`` [mm], order
        ``p``, and the PEC surface refinement ``pec_maxh`` [mm]. Setting any of
        these makes multipacting solve its OWN field (see :meth:`run`); inspect
        the result with :meth:`show_mesh` before running. Returns ``self``."""
        mesh = dict(self._staged.get('mesh_config') or {})
        if h is not None:
            mesh['h'] = h
        if p is not None:
            mesh['p'] = p
        if mesh:
            self._staged['mesh_config'] = mesh
        if pec_maxh is not None:
            self._staged['pec_maxh'] = pec_maxh
        return self

    def set_xrange(self, xrange):
        """Stage the emission-site z-interval [m] (inspect with
        :meth:`show_emission_points`). Returns ``self``."""
        self._staged['xrange'] = list(xrange)
        return self

    def set_epks(self, epks):
        """Stage the peak-field sweep values [V/m]. Returns ``self``."""
        self._staged['epks'] = list(epks)
        return self

    def set_phis(self, phis):
        """Stage the RF launch phases [rad]. Returns ``self``."""
        self._staged['phis'] = list(phis)
        return self

    def _field_mesh_spec(self, cfg=None):
        """The effective own-field mesh spec {h, p, pec_maxh}, or None when no
        multipacting-specific mesh was requested (reuse the eigenmode field)."""
        cfg = cfg if cfg is not None else merge_config({}, self._staged)
        mesh_config = cfg.get('mesh_config')
        pec_maxh = cfg.get('pec_maxh')
        if not mesh_config and not pec_maxh:
            return None
        spec = dict(self.DEFAULT_EIGENMODE['mesh_config'])
        spec.update(mesh_config or {})
        if pec_maxh:
            spec['pec_maxh'] = pec_maxh
        return spec

    def _preview_mesh(self):
        """The mesh multipacting would use right now: the staged own-field mesh
        if mesh parameters / pec_maxh were set, otherwise the saved eigenmode
        mesh (or a default-spec preview if none exists yet)."""
        # Deferred: ngsolve is an optional heavy dependency.
        from cavsim2d.analysis.multipacting.fields import load_eigenmode_fields

        spec = self._field_mesh_spec()
        if spec is None:
            pol_dir = Path(monopole_dir(self.cavity.eigenmode.folder))
            if (pol_dir / 'mesh.pkl').exists():
                mesh, _, _ = load_eigenmode_fields(str(pol_dir))
                return mesh
            spec = dict(self.DEFAULT_EIGENMODE['mesh_config'])
        # Deferred: same ngsolve-heavy path.
        from cavsim2d.solvers.NGSolve.eigen_ngsolve import mesh_h_metres
        profile = self.cavity.profile()
        edge = {'PEC': float(spec['pec_maxh']) * 1e-3} if spec.get('pec_maxh') else None
        # order=1 (straight): the own-field mesh is deliberately uncurved so
        # the tracker's collision polyline coincides with the element edges —
        # see solve_multipacting_field. The preview shows the mesh actually used.
        return profile.mesh(maxh=mesh_h_metres(spec, default=6),
                            order=1, edge_maxh=edge)

    def show_mesh(self, plotter='ngsolve'):
        """Preview the mesh multipacting will use — the staged own-field mesh
        when :meth:`set_mesh_parameters`/``pec_maxh`` were given, otherwise the
        eigenmode mesh. No field is solved."""
        mesh = self._preview_mesh()
        if plotter == 'matplotlib':
            # Deferred: ngsolve is an optional heavy dependency.
            from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
            NGSolveMEVP()._plot_mesh_matplotlib(mesh)
            return None
        # Deferred: webgui is notebook-oriented.
        from ngsolve.webgui import Draw
        return Draw(mesh)

    def show_emission_points(self, xrange=None, step=None, ax=None, show=True):
        """Plot the wall with the emission sites the staged (or given) ``xrange``
        selects — the same selection the tracker will make — so the launch
        region can be checked and adjusted before running."""
        # Deferred: ngsolve is an optional heavy dependency.
        from cavsim2d.analysis.multipacting.driver import (_surface_points,
                                                           default_xrange)
        mesh = self._preview_mesh()
        xsurf = _surface_points(mesh)
        if xrange is None:
            xrange = self._staged.get('xrange') or default_xrange(xsurf)
        pts = xsurf[(xsurf[:, 0] > xrange[0]) & (xsurf[:, 0] < xrange[1])]
        if step:
            keep, last = [], pts[0][0] - step
            for p_ in pts:
                if p_[0] >= last + step:
                    keep.append(p_)
                    last = p_[0]
            pts = np.array(keep)
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(xsurf[:, 0] * 1e3, xsurf[:, 1] * 1e3, '.', ms=2, color='0.6',
                    label='wall points')
            ax.scatter(pts[:, 0] * 1e3, pts[:, 1] * 1e3, fc='none',
                       ec=WARM[1], s=60, label=f'emission sites ({len(pts)})')
            ax.set_xlabel('z [mm]')
            ax.set_ylabel('r [mm]')
            ax.set_aspect('equal', 'box')
            ax.legend()
        info(f"{len(pts)} emission site(s) in xrange = "
             f"[{xrange[0]:.4g}, {xrange[1]:.4g}] m.")
        _maybe_show(show)
        return ax

    # -- SEY ----------------------------------------------------------------

    @property
    def sey(self):
        """The secondary-emission-yield curve (bundled copper-like default until
        :meth:`set_sey` overrides it)."""
        if self._sey is None:
            self._sey = SEY()
        return self._sey

    def set_sey(self, sey_filepath):
        """Use a custom SEY table (two columns: impact energy [eV], yield)."""
        self._sey = SEY(sey_filepath)
        return self

    # -- Lazy-loaded results ------------------------------------------------

    @property
    def config(self):
        if self._config is None:
            path = self.folder / 'config.json'
            self._config = json.load(open(path)) if path.exists() else {}
        return self._config

    @property
    def results(self):
        """The saved sweep result dict (``multipacting/mresults.pkl``), or {}."""
        if self._results is None:
            path = self.folder / 'mresults.pkl'
            if path.exists():
                with open(path, 'rb') as f:
                    self._results = pickle.load(f)
            else:
                self._results = {}
        return self._results

    @property
    def epk(self):
        """Peak-field sweep axis [MV/m]."""
        r = self.results
        if not r:
            return np.array([])
        return np.asarray(r['epks']) * r['Epk'] * 1e-6

    @property
    def counter(self):
        """Counter function c20/c0 vs peak field."""
        return np.asarray(self.results.get('cn/c0', []), dtype=float)

    @property
    def final_energy(self):
        """Mean final impact energy [eV] vs peak field."""
        po = self.results.get('particles_objects')
        return mp_metrics.final_energy(po) if po else []

    @property
    def particles(self):
        """Per-field-level tracked-particle objects."""
        return self.results.get('particles_objects', [])

    def _invalidate(self):
        self._results = None

    # -- Actions ------------------------------------------------------------

    def run(self, config=None, **kwargs):
        """Run the multipacting peak-field sweep.

        Reads the cavity's monopole eigenmode field, auto-running the eigenmode
        solve (``DEFAULT_EIGENMODE`` merged with ``config['eigenmode_config']``)
        if no field is on disk. Config keys: ``mode`` (0-based eigenmode index,
        default the fundamental), ``xrange`` (emission-site z-interval [m]),
        ``epks`` (peak fields [V/m]), ``phis`` (launch phases), ``v_init``
        (emission energy [eV]), ``step``, ``proc_count`` (parallel workers;
        default auto, 1 = in-process), ``loss_model`` ('field'/'wait'/'always'),
        ``t_max`` (track duration [s], default 1000e-10), ``eigenmode_config``,
        ``pec_maxh`` (see below).

        ``pec_maxh`` (mm) requests a **finer PEC surface mesh just for
        multipacting** — the surviving electrons live in a sub-millimetre layer
        at the wall, so the near-wall field needs more resolution than the
        eigenmode QOIs do. With ``pec_maxh`` set, multipacting solves the
        monopole eigenmode on its OWN surface-refined mesh, stored under
        ``multipacting/field/`` — ``cav.eigenmode``'s results (and everything
        downstream of them) are left untouched. Without it, the existing
        eigenmode field is reused as before.

        Values staged with :meth:`set_mesh_parameters` / :meth:`set_xrange` /
        :meth:`set_epks` / :meth:`set_phis` are merged in under the config; a key
        appearing in **both** warns that the config entry overwrites the staged
        value. The **complete** merged config (defaults included) is what runs
        and what ``config.json`` records.

        The sweep parallelises over field levels; each worker is a plain
        subprocess (`python -m ...driver`), so it works identically from
        notebooks, scripts (no ``if __name__ == '__main__':`` guard needed)
        and tests, on every platform. ``proc_count=1`` runs in-process. Worker
        output goes to ``multipacting/mworker_{p}.log``; a crashed worker's
        traceback is included in the raised error.
        """
        # Deferred: the driver pulls in ngsolve (optional heavy dependency); keeps
        # `import cavsim2d` soft on it, mirroring the eigenmode/tune run paths.
        from cavsim2d.analysis.multipacting.driver import run_sweep
        # Deferred: same ngsolve-heavy path as the driver.
        from cavsim2d.analysis.multipacting.fields import solve_multipacting_field

        user_cfg = merge_config({}, config, kwargs)
        overlap = sorted(set(self._staged) & set(user_cfg))
        if overlap:
            # warnings.warn (not the verbosity-gated warning()): the user must
            # see that their staged values are being overridden.
            warnings.warn(
                f"Multipacting config entr{'ies' if len(overlap) > 1 else 'y'} "
                f"{overlap} overwrite value(s) previously staged with set_*(); "
                f"the run() config takes precedence.", UserWarning, stacklevel=2)
        cfg = merge_config(DEFAULT_MULTIPACTING_CONFIG, self._staged, user_cfg)

        self.cavity._ensure_workspace()
        self.folder.mkdir(parents=True, exist_ok=True)

        eigenmode_config = cfg.get('eigenmode_config')
        field_spec = self._field_mesh_spec(cfg)
        if field_spec is not None:
            # Multipacting-owned field: solve the monopole eigenmode on a
            # (PEC-refined) multipacting-specific mesh into multipacting/field/.
            # The shared cav.eigenmode results are deliberately NOT recomputed
            # or replaced.
            info(f"Multipacting: own field mesh {field_spec} — solving the "
                 f"monopole eigenmode into {self.folder / 'field'}. "
                 f"cav.eigenmode results are left untouched.")
            fields_dir = self.folder / 'field'
            freqs = solve_multipacting_field(self.cavity, str(fields_dir),
                                             mesh_config=field_spec,
                                             n_modes=cfg.get('n_modes'))
        else:
            fields_dir = self._ensure_eigenmode(eigenmode_config)
            freqs = [m.frequency for m in self.cavity.eigenmode.modes]
        if not freqs:
            error("No monopole eigenmodes available for multipacting. "
                  "Run cav.eigenmode.run({'polarisation':'monopole'}) first.")
            return self

        with open(self.folder / 'config.json', 'w') as f:
            json.dump(cfg, f, indent=4, default=str)
        self._config = cfg

        self._results = run_sweep(str(self.folder), str(fields_dir), freqs, self.sey,
                                  mode=cfg.get('mode', 0),
                                  xrange=cfg.get('xrange'),
                                  epks=cfg.get('epks'),
                                  phis=cfg.get('phis'),
                                  v_init=cfg.get('v_init', 2),
                                  step=cfg.get('step'),
                                  proc_count=cfg.get('proc_count'),
                                  loss_model=cfg.get('loss_model', 'field'),
                                  t_max=cfg.get('t_max', 1000e-10),
                                  progress=cfg.get('progress', True))
        return self

    def _ensure_eigenmode(self, eigenmode_config):
        """Path to the monopole eigenmode field folder, running the eigenmode
        solve first if its fields are not on disk."""
        def pol_dir():
            return Path(monopole_dir(self.cavity.eigenmode.folder))
        have = (pol_dir() / 'gfu_EH.pkl').exists() and (pol_dir() / 'mesh.pkl').exists()
        if not have:
            cfg = dict(self.DEFAULT_EIGENMODE)
            if eigenmode_config:
                cfg.update(eigenmode_config)
            info("Multipacting: no eigenmode field found — running the monopole "
                 "eigenmode solve first.")
            self.cavity.eigenmode.run(cfg)
            self.cavity.eigenmode._modes = None      # force reload of the new modes
        # recompute after the run: the 'monopole/' subdir now exists
        return pol_dir()

    # -- Visualisation ------------------------------------------------------

    def plot_counter(self, ax=None, launchable_norm=False, show=True, **kwargs):
        """Counter function c20/c0 vs peak field (the multipacting-resonance
        signature). ``launchable_norm=True`` divides by the launchable fraction
        (~0.5) to match MultiPac's normalisation."""
        r = self.results
        if not r:
            info("No multipacting results — run cav.multipacting.run(...) first.")
            return ax
        cf = self.counter
        label = r'$c_{20}/c_0$'
        if launchable_norm:
            # Deferred: driver/fields pull in ngsolve (optional heavy dependency).
            from cavsim2d.analysis.multipacting.driver import load_eigenmode_fields
            from cavsim2d.analysis.multipacting.fields import build_emfield
            # the field the sweep actually used — the multipacting-owned one when
            # pec_maxh was set, the shared eigenmode one otherwise
            fields_dir = r.get('fields_dir') or str(Path(monopole_dir(
                self.cavity.eigenmode.folder)))
            mesh, gfu_E, _ = load_eigenmode_fields(fields_dir)
            em = build_emfield(gfu_E, r.get('mode', 0), r['freq [MHz]'])
            frac = mp_metrics.launchable_fraction((self.particles or [None])[0], em, mesh)
            cf = cf / frac
            label += fr'  (launchable /{frac:.2f})'
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('color', getattr(self.cavity, 'color', None) or WARM[0])
            kwargs.setdefault('label', self.cavity.name)
            ax.plot(self.epk, cf, **kwargs)
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r'$E_{\mathrm{pk}}$ [MV/m]')
            ax.set_ylabel(label)
        _maybe_show(show)
        return ax

    def plot_final_energy(self, ax=None, show=True, **kwargs):
        """Mean final impact energy of the 20-hit electrons vs peak field, with
        the SEY = 1 crossover energies marked."""
        if not self.results:
            info("No multipacting results — run cav.multipacting.run(...) first.")
            return ax
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('color', getattr(self.cavity, 'color', None) or WARM[0])
            kwargs.setdefault('label', self.cavity.name)
            ax.plot(self.epk, self.final_energy, **kwargs)
            sey_E = np.asarray(self.sey.data['E'], dtype=float)
            sey_v = np.asarray(self.sey.data['sey'], dtype=float)
            above = sey_v > 1
            for i in np.nonzero(np.diff(above.astype(int)) != 0)[0]:
                cross = sey_E[i] + (1 - sey_v[i]) * (sey_E[i + 1] - sey_E[i]) \
                    / (sey_v[i + 1] - sey_v[i])
                ax.axhline(cross, color=WARM[3], lw=1)
            ax.set_yscale('log')
            ax.set_xlabel(r'$E_{\mathrm{pk}}$ [MV/m]')
            ax.set_ylabel(r'$E_{\mathrm{f,20}}$ [eV]')
        _maybe_show(show)
        return ax

    def plot_enhanced_counter(self, ax=None, show=True, **kwargs):
        """Enhanced counter e20/c0 (secondary-yield weighted) vs peak field."""
        r = self.results
        if not r:
            info("No multipacting results — run cav.multipacting.run(...) first.")
            return ax
        e20 = mp_metrics.enhanced_counter(self.particles, r['n_init_particles'])
        if len(e20) == 0:
            info("No secondaries recorded to build the enhanced counter.")
            return ax
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            kwargs.setdefault('color', getattr(self.cavity, 'color', None) or WARM[0])
            kwargs.setdefault('label', self.cavity.name)
            ax.plot(self.epk, e20, **kwargs)
            ax.axhline(1, color=WARM[1], ls='--', lw=1)
            ax.set_yscale('log')
            ax.set_ylim(bottom=1e-3)
            ax.set_xlabel(r'$E_{\mathrm{pk}}$ [MV/m]')
            ax.set_ylabel(r'$e_{20}/c_0$')
        _maybe_show(show)
        return ax

    def plot_sey(self, ax=None, show=True, **kwargs):
        """The secondary-emission-yield curve delta(E)."""
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 4))
            d = self.sey.data
            kwargs.setdefault('color', WARM[0])
            ax.plot(d['E'][:-1], d['sey'][:-1], **kwargs)
            ax.axhline(1, color=WARM[1], ls='--', lw=1)
            ax.set_xlabel('impact energy [eV]')
            ax.set_ylabel(r'secondary yield $\delta$')
        _maybe_show(show)
        return ax

    def plot_distance_map(self, epk_i, metric='d20', vmax=None, show=True):
        """MultiPac-style distance map over (emission site, launch phase) for the
        ``epk_i``-th field level; minima locate resonant fixed points. Delegates
        to the ported implementation."""
        # Deferred: plots reads the cavity profile / (for the viewer) ipywidgets.
        from cavsim2d.analysis.multipacting.plots import distance_map
        return distance_map(self, epk_i, metric=metric, vmax=vmax, show=show)

    def plot_trajectories(self, color_by=None):
        """Interactive (ipywidgets) viewer of the surviving trajectories.

        ``color_by``: ``None`` (plain line, default), ``'velocity'`` or
        ``'energy'`` — gradient-colours the path along the trajectory."""
        # Deferred: the viewer needs ipywidgets (notebook-only [jupyter] extra).
        from cavsim2d.analysis.multipacting.plots import trajectory_viewer
        return trajectory_viewer(self, color_by=color_by)

    def animate_trajectories(self, epk_i=None, phi_i=None, traj=None,
                             color_by='energy', trail=40, step=1, fps=30,
                             save=None, dpi=120, zoom='auto', progress=True,
                             embed='auto'):
        """Animate the surviving trajectories: moving heads with a short fading
        trace, gradient-coloured by ``'energy'`` (default) or ``'velocity'``.

        Select a subset with ``epk_i`` (field-level index/list), ``phi_i``
        (launch-phase index/list) and/or ``traj`` (trajectory index/list);
        ``None`` selects all (the default). ``save='mp.gif'`` / ``'mp.mp4'``
        writes the animation to disk.

        ``progress=True`` shows a per-frame progress bar with timing (a
        many-particle animation renders slowly). In a notebook the animation
        also plays **inline** automatically (``embed='auto'``; suppressed when
        ``save`` is given to avoid a second render). The
        :class:`~matplotlib.animation.FuncAnimation` is always returned, so
        ``anim = cav.multipacting.animate_trajectories(); anim.save(...)``
        works too."""
        # Deferred: pulls in matplotlib.animation (only needed here).
        from cavsim2d.analysis.multipacting.plots import trajectory_animation
        return trajectory_animation(self, epk_i=epk_i, phi_i=phi_i, traj=traj,
                                    color_by=color_by, trail=trail, step=step,
                                    fps=fps, save=save, dpi=dpi, zoom=zoom,
                                    progress=progress, embed=embed)


# ---------------------------------------------------------------------------
# OptimisationSolver
# ---------------------------------------------------------------------------

class OptimisationSolver:
    """Manages ``<project>/optimisation/`` — multi-objective cavity optimisation.

    Access
    ------
    - ``cavs.optimisation.config``
    - ``cavs.optimisation.history``           — DataFrame of all evaluated candidates
    - ``cavs.optimisation.pareto``            — Pareto-optimal solutions
    - ``cavs.optimisation.pareto_history``    — Pareto fronts from every generation
    - ``cavs.optimisation.objective_vars``    — list of objective column names

    Visualisation (kind='scatter'|'pcp'|'radar'|'heatmap', normalise=True|False)
    - ``cavs.optimisation.plot_pareto(kind, normalise)``
    - ``cavs.optimisation.plot_history(kind, normalise)``
    - ``cavs.optimisation.plot_pareto_history(normalise, color_by_gen)``
    - ``cavs.optimisation.plot_convergence()``
    """

    def __init__(self, cavities_parent):
        self._parent = cavities_parent
        self._config = None
        self._history = None
        self._pareto = None
        self._cavities = None

    @property
    def folder(self):
        return Path(self._parent.projectDir) / 'optimisation'

    @property
    def candidates_folder(self):
        return self.folder / 'candidates'

    @property
    def config(self):
        if self._config is None:
            cfg_path = self.folder / 'config.json'
            if cfg_path.exists():
                with open(cfg_path, 'r') as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    @property
    def history(self):
        if self._history is None:
            hist_path = self.folder / 'history.csv'
            if hist_path.exists():
                self._history = pd.read_csv(hist_path)
            else:
                self._history = pd.DataFrame()
        return self._history

    @property
    def pareto(self):
        if self._pareto is None:
            p_path = self.folder / 'pareto_front.csv'
            if p_path.exists():
                self._pareto = pd.read_csv(p_path)
            else:
                self._pareto = pd.DataFrame()
        return self._pareto

    @property
    def cavities(self):
        """Returns a Cavities object of the optimised candidates."""
        if self._cavities is None:
            # Deferred: breaks the solver_objects <-> study import cycle.
            from cavsim2d.study import Cavities
            cands = self.candidates_folder
            if cands.exists():
                # Load existing candidate cavities
                # This will be populated after optimisation runs
                pass
            self._cavities = None
        return self._cavities

    # -- Actions ------------------------------------------------------------

    def run(self, config, resume=False):
        """Run the full optimisation loop.

        This method:
        1. Uses the cavity added to the study as the optimisation template (its
           type drives tune-variable resolution / spawn / rebuild, so every
           model optimises); falls back to a bounds-derived elliptical template
           only when no cavity was added.
        2. Runs the evolutionary algorithm
        3. Saves results to ``<project>/optimisation/``

        Parameters
        ----------
        config : dict
            Optimisation config. A ``seed`` field makes the run reproducible.
        resume : bool, default False
            If True, reuses any existing ``<project>/optimisation/`` artefacts
            (generation tables, candidate simulation results) instead of
            starting from scratch. The stored config is compared against the
            new one — mismatched fields trigger a warning but the run still
            proceeds using the caller-supplied config.
        """
        # Deferred: breaks the solver_objects <-> optimisation import cycle.
        from cavsim2d.analysis.optimisation import Optimisation

        self.folder.mkdir(parents=True, exist_ok=True)
        self.candidates_folder.mkdir(parents=True, exist_ok=True)

        cfg_path = self.folder / 'config.json'
        if resume and cfg_path.exists():
            try:
                with open(cfg_path, 'r') as f:
                    prev_cfg = json.load(f)
                # Compare core reproducibility fields only — leave things like
                # processes/verbose flexible.
                key_fields = ('bounds', 'method', 'initial_points', 'seed',
                              'mutation_factor', 'crossover_factor',
                              'chaos_factor', 'elites_for_crossover',
                              'mutation_sigma', 'eta_sbx')
                diffs = [k for k in key_fields
                         if prev_cfg.get(k) != config.get(k)]
                if diffs:
                    warning(f'Resume: config differs from saved run in {diffs}. '
                            f'Continuing with new config — results may diverge.')
            except Exception:
                pass

        # Persist effective config so future resumes can compare. ``resume``
        # itself is not a reproducibility field, so don't save it.
        cfg_to_save = {k: v for k, v in config.items() if k != 'resume'}
        with open(cfg_path, 'w') as f:
            json.dump(cfg_to_save, f, indent=4, default=str)
        self._config = config

        # Thread resume through to the Optimisation class.
        config = dict(config)
        config.setdefault('resume', resume)

        # The optimisation template is the cavity the user added to the study —
        # THIS is what makes optimisation work for every model, not just
        # elliptical: its type drives tune-variable resolution, spawn and
        # rebuild, and its parameters supply defaults for the non-swept ones.
        # (A rebuilt copy, so the user's own cavity is not mutated.) Only when
        # no cavity was added does it fall back to fabricating an elliptical
        # from the bounds — the bare ``Cavities(dir).run_optimisation(cfg)`` path.
        bounds = config['bounds']
        added = list(getattr(self._parent, 'cavities_list', []) or [])
        if added:
            if len(added) > 1:
                info(f"Optimisation uses one template cavity; using the first of "
                     f"{len(added)} ({added[0].name}).")
            template_cav = added[0].rebuild(added[0].parameters)
        else:
            # Midpoint of each bound for the swept params, defaults otherwise.
            # Order must match EllipticalCavity constructor: A, B, a, b, Ri, L, Req
            mid_cell_params = config.get('mid-cell', None)
            if mid_cell_params is None:
                param_names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
                defaults = {'A': 42, 'B': 42, 'a': 12, 'b': 19,
                            'Ri': 35, 'L': 57.7, 'Req': 103.4}
                mid_cell_params = []
                for name in param_names:
                    if name in bounds:
                        low, high = bounds[name]
                        mid_cell_params.append((low + high) / 2)
                    else:
                        mid_cell_params.append(defaults.get(name, 50))
            # Deferred: breaks the solver_objects <-> models.elliptical import cycle.
            from cavsim2d.models.elliptical import EllipticalCavity
            tune_config = config.get('tune_config', {})
            eigenmode_config = tune_config.get('eigenmode_config', {})
            n_cells = eigenmode_config.get('n_cells', 1)
            template_cav = EllipticalCavity(n_cells=n_cells, mid_cell=mid_cell_params)

        template_cav.projectDir = self._parent.projectDir
        template_cav.self_dir = str(self.candidates_folder / '_template')
        os.makedirs(template_cav.self_dir, exist_ok=True)

        # Create the optimiser and run
        opt = Optimisation()
        opt.run(template_cav, config, opt_solver=self)

        self._invalidate()

    def _invalidate(self):
        self._history = None
        self._pareto = None
        self._cavities = None

    # -- Data helpers ---------------------------------------------------------

    @property
    def pareto_history(self):
        """DataFrame of Pareto fronts from all generations (with 'generation' column)."""
        p = self.folder / 'pareto_history.csv'
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    @property
    def objective_vars(self):
        """List of objective column names stored during the run."""
        meta = self._load_objective_meta()
        return meta.get('objective_vars', [])

    @property
    def objectives(self):
        """List of objective specs [[sense, name, ...], ...]."""
        meta = self._load_objective_meta()
        return meta.get('objectives', [])

    def _load_objective_meta(self):
        p = self.folder / 'objective_meta.json'
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f)
        return {}

    def _obj_data(self, df, obj_cols):
        """Extract objective columns from *df*, returning (values, labels)."""
        present = [c for c in obj_cols if c in df.columns]
        if not present:
            return None, []
        return df[present].values, present

    @staticmethod
    def _normalise(vals):
        """Min-max normalise each column to [0, 1]."""
        mn = vals.min(axis=0)
        mx = vals.max(axis=0)
        rng = mx - mn
        rng[rng == 0] = 1.0
        return (vals - mn) / rng

    # -- Visualisation ------------------------------------------------------

    def plot_pareto(self, kind='scatter', normalise=True, show=True, ax=None, **kwargs):
        """Plot the current Pareto front.

        Parameters
        ----------
        kind : str
            'scatter', 'pcp', 'radar', or 'heatmap'.
        normalise : bool
            Min-max normalise objectives to [0, 1].
        show : bool
            Call ``plt.show()`` at the end.
        ax : matplotlib Axes, optional
            Single-axis plots ('pcp', 'heatmap') reuse this axes.
        **kwargs
            Forwarded to the underlying matplotlib call (e.g. marker, s, lw, alpha).

        Returns
        -------
        (fig, axes)
        """
        df = self.pareto
        if df.empty:
            info("No Pareto front data available.")
            return None, None
        obj_cols = self.objective_vars or self._default_obj_cols(df)
        return self._plot_objectives(df, obj_cols, kind, normalise, show, ax,
                                     title_prefix='Pareto front', **kwargs)

    def plot_history(self, kind='scatter', normalise=True, color_by_gen=False,
                     show=True, ax=None, **kwargs):
        """Plot the full history of evaluated candidates.

        Parameters
        ----------
        kind : str
            'scatter', 'pcp', 'radar', or 'heatmap'.
        normalise : bool
            Min-max normalise objectives to [0, 1].
        color_by_gen : bool
            Colour points by generation.
        show : bool
            Call ``plt.show()`` at the end.
        ax : matplotlib Axes, optional
        **kwargs
            Forwarded to the underlying matplotlib call.
        """
        df = self.history
        if df.empty:
            info("No history data available.")
            return None, None
        obj_cols = self.objective_vars or self._default_obj_cols(df)

        generations = None
        if color_by_gen:
            if 'generation' not in df.columns and 'key' in df.columns:
                df = df.copy()
                df['generation'] = df['key'].str.extract(r'G(\d+)_').astype(float)
            generations = df.get('generation')

        return self._plot_objectives(df, obj_cols, kind, normalise, show, ax,
                                     title_prefix='History',
                                     generations=generations, **kwargs)

    def plot_pareto_history(self, normalise=True, color_by_gen=True, show=True, **kwargs):
        """Plot the Pareto front from every generation using a subplot mosaic.

        Each unique pair of objectives gets its own subplot. Points are coloured
        by generation so you can see how the front evolved.

        Parameters
        ----------
        normalise : bool
            Min-max normalise objectives to [0, 1].
        color_by_gen : bool
            Colour points by generation (True) or use a single colour (False).
        show : bool
            Call ``plt.show()`` at the end.
        **kwargs
            Forwarded to ``ax.scatter`` (e.g. s, marker, alpha, edgecolors).

        Returns
        -------
        (fig, axes_dict) — axes_dict maps ``'label_i vs label_j'`` to axes.
        """

        df = self.pareto_history
        if df.empty or 'generation' not in df.columns:
            info("No Pareto history data available.")
            return None, None

        obj_cols = self.objective_vars or self._default_obj_cols(df)
        vals, labels = self._obj_data(df, obj_cols)
        if vals is None or len(labels) < 2:
            info("Need at least 2 objective columns in Pareto history.")
            return None, None

        if normalise:
            vals = self._normalise(vals)

        generations = df['generation'].values
        gen_unique = np.unique(generations)
        cmap = plt.cm.viridis
        norm_gen = plt.Normalize(gen_unique.min(), gen_unique.max())

        # Build mosaic layout: one subplot per objective pair
        pairs = list(combinations(range(len(labels)), 2))
        n_pairs = len(pairs)
        ncols = min(n_pairs, 3)
        nrows = int(np.ceil(n_pairs / ncols))

        mosaic_labels = []
        pair_map = {}
        for k, (i, j) in enumerate(pairs):
            key = f'{labels[i]} vs {labels[j]}'
            mosaic_labels.append(key)
            pair_map[key] = (i, j)

        # Pad to fill the grid
        while len(mosaic_labels) < nrows * ncols:
            mosaic_labels.append('.')

        mosaic = [mosaic_labels[r * ncols:(r + 1) * ncols] for r in range(nrows)]

        fig, axes_dict = plt.subplot_mosaic(mosaic,
                                            figsize=(5 * ncols, 4 * nrows))

        scatter_kw = {'s': 20, 'alpha': 0.6, 'edgecolors': 'none'}
        scatter_kw.update(kwargs)

        sc = None
        for key, (i, j) in pair_map.items():
            ax = axes_dict[key]
            x, y = vals[:, i], vals[:, j]

            if color_by_gen:
                kw = {k: v for k, v in scatter_kw.items() if k not in ('c', 'color')}
                sc = ax.scatter(x, y, c=generations, cmap=cmap, norm=norm_gen, **kw)
                # Highlight final generation
                final = generations == gen_unique[-1]
                ax.scatter(x[final], y[final], facecolors='red', s=scatter_kw.get('s', 20) * 2,
                           edgecolors='k', linewidths=0.5, zorder=5)
            else:
                ax.scatter(x, y, **scatter_kw)

            ax.set_xlabel(labels[i])
            ax.set_ylabel(labels[j])

        if color_by_gen and sc is not None:
            fig.colorbar(sc, ax=list(axes_dict.values()), label='Generation', shrink=0.6)

        suffix = ' (normalised)' if normalise else ''
        fig.suptitle(f'Pareto history{suffix}', fontsize=14)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, axes_dict

    def plot_convergence(self, show=True, **kwargs):
        """Plot hypervolume convergence from saved HV history.

        Returns (fig, (ax1, ax2)).
        """
        hv_path = self.folder / 'hv_history.json'
        if not hv_path.exists():
            info("No HV convergence data saved. Run optimisation first.")
            return None, None
        with open(hv_path, 'r') as f:
            hv_data = json.load(f)
        hv = np.array(hv_data['hv_history'])
        tol = hv_data.get('hv_tol', 1e-9)
        if len(hv) < 2:
            info("Not enough generations for convergence plot.")
            return None, None

        plot_kw = {'marker': 'o'}
        plot_kw.update(kwargs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(hv, **plot_kw)
        ax1.set_xlabel('Generation $n$')
        ax1.set_ylabel('Hypervolume')
        ax1.set_title('Hypervolume indicator')

        rel_change = np.abs(np.diff(hv)) / np.maximum(hv[1:], 1e-30)
        ax2.plot(range(1, len(hv)), rel_change, **{**plot_kw, 'marker': 's'})
        ax2.axhline(y=tol, color='r', linestyle='--', label=f'tol = {tol:.0e}')
        ax2.set_yscale('log')
        ax2.set_xlabel('Generation $n$')
        ax2.set_ylabel(r'$|\Delta \mathrm{HV}| / \mathrm{HV}_n$')
        ax2.set_title('Relative HV change')
        ax2.legend()

        plt.tight_layout()
        if show:
            plt.show()
        return fig, (ax1, ax2)

    # -- Internal plot helpers ------------------------------------------------

    @staticmethod
    def _default_obj_cols(df):
        return [c for c in df.columns if c not in ('key', 'total_rank', 'generation')]

    def _plot_objectives(self, df, obj_cols, kind, normalise, show, ax,
                         title_prefix, generations=None, **kwargs):
        vals, labels = self._obj_data(df, obj_cols)
        if vals is None:
            info(f"Objective columns {obj_cols} not found in data.")
            return None, None

        if normalise:
            vals = self._normalise(vals)

        dispatch = {
            'scatter': self._plot_scatter_matrix,
            'pcp': self._plot_pcp,
            'radar': self._plot_radar,
            'heatmap': self._plot_heatmap,
        }
        fn = dispatch.get(kind)
        if fn is None:
            info(f"Unknown plot kind '{kind}'. Choose from: {list(dispatch.keys())}")
            return None, None
        fig, axes = fn(vals, labels, normalise, ax, title_prefix,
                       generations=generations, **kwargs)
        if show and fig is not None:
            plt.show()
        return fig, axes

    def _plot_scatter_matrix(self, vals, labels, normalise, _ax, title_prefix,
                             generations=None, **kwargs):
        """Pairwise scatter matrix of objectives."""
        d = len(labels)
        if d < 2:
            info("Need at least 2 objectives for scatter matrix.")
            return None, None

        fig, axes = plt.subplots(d - 1, d - 1, figsize=(4 * (d - 1), 4 * (d - 1)),
                                 squeeze=False)

        scatter_kw = {'s': 15, 'alpha': 0.6}
        scatter_kw.update(kwargs)

        has_gen = generations is not None and len(generations) == len(vals)
        if has_gen:
            gen = np.asarray(generations)
            gen_unique = np.unique(gen[~np.isnan(gen)])
            cmap = plt.cm.viridis
            norm_gen = plt.Normalize(gen_unique.min(), gen_unique.max())
            scatter_kw.pop('color', None)
            scatter_kw.pop('c', None)

        for row in range(d - 1):
            for col in range(d - 1):
                ax = axes[row][col]
                if col > row:
                    ax.set_visible(False)
                    continue
                j, i = col, row + 1
                if has_gen:
                    sc = ax.scatter(vals[:, j], vals[:, i], c=gen,
                                    cmap=cmap, norm=norm_gen, **scatter_kw)
                else:
                    ax.scatter(vals[:, j], vals[:, i], **scatter_kw)
                if row == d - 2:
                    ax.set_xlabel(labels[j])
                if col == 0:
                    ax.set_ylabel(labels[i])

        if has_gen:
            fig.colorbar(sc, ax=axes, label='Generation', shrink=0.6)

        suffix = ' (normalised)' if normalise else ''
        fig.suptitle(f'{title_prefix} — pairwise scatter{suffix}', fontsize=14)
        plt.tight_layout()
        return fig, axes

    def _plot_pcp(self, vals, labels, normalise, ax, title_prefix,
                  generations=None, **kwargs):
        """Parallel coordinate plot."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 5))
        else:
            fig = ax.figure

        plot_kw = {'alpha': 0.4, 'linewidth': 0.8}
        plot_kw.update(kwargs)

        x = np.arange(len(labels))
        has_gen = generations is not None and len(generations) == len(vals)
        if has_gen:
            gen = np.asarray(generations)
            gen_unique = np.unique(gen[~np.isnan(gen)])
            cmap = plt.cm.viridis
            norm_gen = plt.Normalize(gen_unique.min(), gen_unique.max())
            plot_kw.pop('color', None)

        for idx, row in enumerate(vals):
            kw = dict(plot_kw)
            if has_gen:
                kw['color'] = cmap(norm_gen(gen[idx]))
            ax.plot(x, row, **kw)

        if has_gen:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_gen)
            fig.colorbar(sm, ax=ax, label='Generation')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        suffix = ' (normalised)' if normalise else ''
        ax.set_title(f'{title_prefix} — parallel coordinates{suffix}')
        if normalise:
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Normalised value')
        plt.tight_layout()
        return fig, ax

    def _plot_radar(self, vals, labels, normalise, _ax, title_prefix,
                    generations=None, **kwargs):
        """Radar (spider) plot — best for comparing a small number of solutions."""
        n_solutions = min(len(vals), 12)
        d = len(labels)
        angles = np.linspace(0, 2 * np.pi, d, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})

        plot_kw = {'linewidth': 1.2, 'alpha': 0.7}
        plot_kw.update(kwargs)

        cmap = plt.cm.tab10
        for i in range(n_solutions):
            row = vals[i].tolist() + [vals[i][0]]
            color = cmap(i % 10)
            ax.plot(angles, row, color=color, **plot_kw)
            ax.fill(angles, row, alpha=0.05, color=color)

        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        suffix = ' (normalised)' if normalise else ''
        ax.set_title(f'{title_prefix} — radar{suffix}', y=1.1)
        if normalise:
            ax.set_ylim(0, 1.05)
        plt.tight_layout()
        return fig, ax

    def _plot_heatmap(self, vals, labels, normalise, ax, title_prefix,
                      generations=None, **kwargs):
        """Heatmap of objective values across solutions."""
        if ax is None:
            h = max(4, 0.3 * len(vals))
            fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(labels)), min(h, 20)))
        else:
            fig = ax.figure

        imshow_kw = {'aspect': 'auto', 'cmap': 'RdYlGn_r'}
        imshow_kw.update(kwargs)

        im = ax.imshow(vals, **imshow_kw)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Solution index')
        fig.colorbar(im, ax=ax, label='Normalised value' if normalise else 'Value')

        suffix = ' (normalised)' if normalise else ''
        ax.set_title(f'{title_prefix} — heatmap{suffix}')
        plt.tight_layout()
        return fig, ax


# ---------------------------------------------------------------------------
# Study-level solver namespaces
# ---------------------------------------------------------------------------
# A cavity exposes cav.eigenmode / cav.wakefield; a Study exposes the same two
# names, so a plot is reached the same way whether you have one cavity or many:
#
#     cav.eigenmode.plot_impedance()        study.eigenmode.plot_impedance()
#     cav.wakefield.plot_impedance()        study.wakefield.plot_impedance()
#
# The study-level versions overlay every cavity, one warm colour each.

class _StudyNamespace:
    """Shared plumbing for the study-level solver namespaces."""

    def __init__(self, study):
        self.study = study

    @property
    def cavities(self):
        return self.study.cavities_list

    def _colors(self):
        return self.study._compare_colors()

    def _overlay_curves(self, method, ax, empty_msg, show=True, **kwargs):
        """Overlay one per-cavity curve method (e.g. 'plot_k_loss') on one axis,
        one warm colour each."""
        drew = False
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4.5))
            for cav, color in zip(self.cavities, self._colors()):
                before = len(ax.lines)
                # show=False: draw onto the shared overlay axis, show once below
                getattr(cav.wakefield, method)(ax=ax, color=color,
                                               label=cav.name, show=False, **kwargs)
                drew = drew or len(ax.lines) > before
            if not drew:
                info(empty_msg)
                return ax
            ax.legend()
        _maybe_show(show)
        return ax


class StudyEigenmode(_StudyNamespace):
    """``study.eigenmode`` — eigenmode results across every cavity in the study."""

    def run(self, eigenmode_config=None, **kwargs):
        """Run the eigenmode solve for every cavity (same as ``study.run_eigenmode``)."""
        return self.study.run_eigenmode(eigenmode_config, **kwargs)

    @property
    def qois_df(self):
        """Every mode of every polarisation of every cavity, as one DataFrame.

        The single-cavity frame (:attr:`EigenmodeSolver.qois_df`) plus a
        ``cavity`` column, so a study filters exactly like one cavity does."""
        frames = []
        for cav in self.cavities:
            df = cav.eigenmode.qois_df
            if df.empty:
                continue
            df = df.copy()
            df.insert(0, 'cavity', cav.name)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def plot_impedance(self, kind='longitudinal', ax=None, span=None,
                       n_points=8001, Q=None, unit='k', show=True, **kwargs):
        """Overlay each cavity's impedance, reconstructed from its eigenmodes.

        See :meth:`EigenmodeSolver.impedance`. The span defaults to
        ``(0, highest mode across ALL cavities)`` so the curves share one axis
        and are genuinely comparable.

        >>> ax = study.eigenmode.plot_impedance(show=False)
        >>> study.wakefield.plot_impedance(ax=ax)      # compare with the wake solve
        """
        transverse = str(kind).lower().startswith('trans')
        m = 1 if transverse else 0

        if span is None:
            highest = [df[df['m'] == m]['freq [MHz]'].max()
                       for df in (c.eigenmode.qois_df for c in self.cavities)
                       if not df.empty and (df['m'] == m).any()]
            if not highest:
                error(f"No {'dipole' if transverse else 'monopole'} eigenmode results "
                      f"in this study — run study.eigenmode.run(...) first.")
                return ax
            span = (0.0, float(max(highest)))

        u = impedance_unit(unit, transverse)
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4.5))
            for cav, color in zip(self.cavities, self._colors()):
                df = cav.eigenmode.impedance(kind=kind, span=span,
                                             n_points=n_points, Q=Q, unit=unit)
                if df.empty:
                    continue
                ax.plot(df['f [MHz]'], df[f'|Z| [{u}]'], color=color,
                        label=cav.name, **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(_z_axis_label(unit, transverse))
            ax.set_yscale('log')
            ax.legend()
        _maybe_show(show)
        return ax


class StudyWakefield(_StudyNamespace):
    """``study.wakefield`` — wakefield results across every cavity in the study."""

    def run(self, wakefield_config=None, **kwargs):
        """Run the wakefield solve for every cavity (same as ``study.run_wakefield``)."""
        return self.study.run_wakefield(wakefield_config, **kwargs)

    def plot_impedance(self, kind='longitudinal', ax=None, unit='k', show=True,
                       **kwargs):
        """Overlay each cavity's wakefield impedance (|Z| vs f)."""
        drew = False
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4.5))
            for cav, color in zip(self.cavities, self._colors()):
                if cav.wakefield.impedance(kind, unit=unit).empty:
                    continue
                # show=False: overlay onto the shared axis, show once below
                cav.wakefield.plot_impedance(kind, ax=ax, unit=unit, color=color,
                                             label=cav.name, show=False, **kwargs)
                drew = True
            if not drew:
                info("No wakefield results in this study — run study.wakefield.run(...).")
                return ax
            ax.set_yscale('log')
            ax.legend()
        _maybe_show(show)
        return ax

    def plot_wake(self, kind='longitudinal', ax=None, show=True, **kwargs):
        """Overlay each cavity's wake potential (W vs s)."""
        return self._overlay(kind, 'wake', ax, show=show, **kwargs)

    def plot_k_loss(self, ax=None, show=True, **kwargs):
        """Overlay each cavity's cumulative loss-factor spectrum k_loss(F) vs f.

        (The single cumulative number per cavity is what the study comparison
        bars, e.g. ``plot_compare_hom_bar``, show instead.)"""
        return self._overlay_curves(
            'plot_k_loss', ax,
            "No loss-factor spectra in this study — run study.wakefield.run(...).",
            show=show, **kwargs)

    def plot_k_kick(self, ax=None, show=True, **kwargs):
        """Overlay each cavity's cumulative kick-factor spectrum k_kick(F) vs f."""
        return self._overlay_curves(
            'plot_k_kick', ax,
            "No kick-factor spectra in this study — run study.wakefield.run(...).",
            show=show, **kwargs)

    def _overlay(self, kind, quantity, ax, show=True, **kwargs):
        transverse = str(kind).lower().startswith('trans')
        drew = False
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4.5))
            for cav, color in zip(self.cavities, self._colors()):
                wf = cav.wakefield
                df = wf.wake_t if transverse else wf.wake_z
                if df is None or df.empty:
                    continue
                # show=False: overlay onto the shared axis, show once below
                wf._plot(df, ax, quantity, **wf._labels(kind),
                         color=color, label=cav.name, show=False, **kwargs)
                drew = True
            if not drew:
                info("No wakefield results in this study — run study.wakefield.run(...).")
                return ax
            ax.legend()
        _maybe_show(show)
        return ax
