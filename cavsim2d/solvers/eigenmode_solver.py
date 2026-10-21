"""The eigenmode namespace, ``cav.eigenmode``."""
import json
import os
from collections import defaultdict
from pathlib import Path
import warnings
from cavsim2d.utils.printing import CavsimWarning, error, info, warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from cavsim2d.utils.integrity import guard_runs, notice_tuned_versions, warn_if_stale
from cavsim2d.analysis.impedance import (impedance_frame, impedance_unit, classify_modes,
                                         pml_stable_modes, reconstruct_impedance,
                                         reconstruct_impedance_qnm)
from cavsim2d.solvers.eigenmode_result import (EigenmodeResult, MPOLE_NAMES, pol_name,
                                               pol_number, monopole_dir)
from cavsim2d.utils.style import house_style, WARM, polarisation_color, shades
from cavsim2d.utils.slope_triangles import add_slope_triangles
from cavsim2d.processes.eigenmode import run_eigenmode_parallel, run_eigenmode_s
from cavsim2d.solvers.NGSolve.eigen_ngsolve import (NGSolveMEVP, direct_solver_available,
                                                    parse_polarisations)
from itertools import product
from cavsim2d.solvers._shared import (_gaussian_sum, _has_mesh, _maybe_show,
                                      _MODE_FILTER_CAVEAT, _reference_series,
                                      _solved_with_open_boundary, _z_axis_label)
from cavsim2d.solvers.configs import (_ConfigSampleMixin, _user_config,
                                      DEFAULT_EIGENMODE_CONFIG, merge_config)


# ---------------------------------------------------------------------------
# EigenmodeSolver
# ---------------------------------------------------------------------------

class EigenmodeSolver(_ConfigSampleMixin):
    """Manages ``<cavity>/eigenmode/`` — eigenmode analysis.

    Access
    ------
    - ``cav.eigenmode.config``         — solver configuration
    - ``cav.eigenmode.config_sample``  — the default config, to edit and pass to run()
    - ``cav.eigenmode.qois``           — QOIs of the accelerating (pi) mode
    - ``cav.eigenmode.modes``          — list of all EigenmodeResult objects
    - ``cav.eigenmode[i]``             — i‑th mode
    - ``cav.eigenmode[i].frequency``
    - ``cav.eigenmode[i].m``
    - ``cav.eigenmode[i].qois``
    - ``cav.eigenmode.plot_spectrum()``
    """

    _config_kind = 'eigenmode'

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._modes = None
        self._qois = None
        self.benchmark_df = None

    @property
    def folder(self):
        return Path(self.cavity.self_dir) / 'eigenmode'

    def _results_folder(self, pol=0):
        """The folder of polarisation *pol* when the cavity has a workspace, else
        None; warns when its results belong to another geometry."""
        if not self.cavity.self_dir:
            return None
        folder = self.pol_folder(pol)
        warn_if_stale(self.cavity, folder, 'The eigenmode results')
        return folder

    # -- Lazy‐loaded results ------------------------------------------------

    @property
    def config(self):
        if not self.cavity.self_dir:
            return {}
        if self._config is None:
            self._config = {}
            # 'eigenmode_config.json' is what Study.run_eigenmode wrote before the
            # names were unified; results produced by an older run still carry it.
            for name in ('config.json', 'eigenmode_config.json'):
                cfg_path = self.folder / name
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        self._config = json.load(f)
                    break
        return self._config

    @property
    def qois(self):
        """QOIs of the monopole mode of interest (the accelerating pi-mode by
        default); empty before the eigenmode analysis has run."""
        folder = self._results_folder(0)
        if folder is None:
            return {}
        if self._qois is None:
            qoi_path = folder / 'qois.json'
            if qoi_path.exists():
                with open(qoi_path, 'r') as f:
                    self._qois = json.load(f)
            else:
                self._qois = {}
        return self._qois

    @property
    def modes(self):
        """All computed eigenmodes as a list of EigenmodeResult."""
        if self._results_folder(0) is None:
            return []
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

    def plot_mesh(self, pol='monopole', ax=None, mirror=True, show=True, **kwargs):
        """Draw the mesh actually used for the solve as a matplotlib triangulation.

        When the run was adaptive (``mesh_config['adaptive']``) this is the
        *refined* mesh, so it reveals where refinement concentrated elements —
        typically sharp corners and high-field regions. ``mirror`` reflects the
        upper-half analysis domain about the axis for the full cross-section.
        """
        # Deferred: pulls in the heavy ngsolve mesh backend only when plotting.
        from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
        folder = self.pol_folder(pol)
        if not _has_mesh(folder):
            warning(f"No saved mesh for {pol_name(pol_number(pol))} at {folder}. "
                 "Run the eigenmode analysis first.")
            return ax
        mesh = NGSolveMEVP.load_mesh(str(folder))
        pts, tris = NGSolveMEVP._mesh_points_and_triangles(mesh)
        pts = np.asarray(pts, dtype=float) * 1e3        # metres -> mm
        tris = np.asarray(tris)
        z, r = pts[:, 0] - pts[:, 0].min(), pts[:, 1]
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(7, 4))
            color = kwargs.pop('color', WARM[7])
            lw = kwargs.pop('lw', kwargs.pop('linewidth', 0.3))
            ax.triplot(z, r, tris, color=color, lw=lw, **kwargs)
            if mirror:
                ax.triplot(z, -r, tris, color=color, lw=lw, **kwargs)
            ax.set_aspect('equal')
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel('$r$ [mm]')
            ax.set_title(f'{self.cavity.name}: {len(tris)} elements')
        _maybe_show(show)
        return ax

    def adaptive_history(self, pol='monopole'):
        """Adaptive-refinement history for *pol* as a list of per-refinement
        dicts (``No of DOFs``, ``No of Mesh Elements``, ``freq [MHz]`` and the
        per-mode recovery ``max_err``), or ``[]`` when the solve was not
        adaptive (``mesh_config={'adaptive': True}``)."""
        path = self.pol_folder(pol) / 'adaptive_history.json'
        if not path.exists():
            return []
        with open(path) as f:
            return json.load(f)

    def plot_convergence(self, pol='monopole', show=True, slopes=False):
        """Adaptive-refinement convergence for *pol* — the AMR namespace's
        "unique property". Two panels versus the number of DOFs along the
        refinement path: the recovery **error** (log-log, driving the marking)
        and each mode's **frequency** (it should flatten as the mesh resolves).
        ``info`` + ``None`` when the run was not adaptive.

        *slopes* marks the rate at which each mode's error falls with a slope
        triangle: ``True`` for every mode, or a list of 1-based mode numbers. A mode
        whose error does not follow a power law gets none (see
        :func:`~cavsim2d.utils.slope_triangles.fit_convergence_rate`)."""
        history = self.adaptive_history(pol)
        if not history:
            info(f"No adaptive history for {pol_name(pol_number(pol))} — run the "
                 "eigenmode analysis with mesh_config={'adaptive': True}.")
            return None
        dofs = [h['No of DOFs'] for h in history]
        errs = np.array([h['max_err'] for h in history])      # (levels, modes)
        freqs = np.array([h['freq [MHz]'] for h in history])  # (levels, modes)
        n_modes = errs.shape[1] if errs.ndim == 2 else 1
        with house_style():
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for j in range(n_modes):
                c = WARM[j % len(WARM)]
                axes[0].loglog(dofs, errs[:, j], marker='o', color=c,
                               label=f'mode {j + 1}')
                axes[1].semilogx(dofs, freqs[:, j], marker='o', color=c,
                                 label=f'mode {j + 1}')
            if slopes:
                chosen = range(n_modes) if slopes is True else [int(j) - 1 for j in slopes]
                chosen = [j for j in chosen if 0 <= j < n_modes]
                add_slope_triangles(axes[0], [(dofs, errs[:, j]) for j in chosen],
                                    [WARM[j % len(WARM)] for j in chosen])
            axes[0].set_xlabel('Number of DOFs')
            axes[0].set_ylabel('Recovery error')
            axes[0].set_title('Error vs DOFs')
            axes[1].set_xlabel('Number of DOFs')
            axes[1].set_ylabel('Frequency [MHz]')
            axes[1].set_title('Frequency vs DOFs')
            if n_modes > 1:
                axes[0].legend(fontsize='small')
            fig.suptitle(f'Adaptive convergence — {self.cavity.name} '
                         f'({pol_name(pol_number(pol))})')
            plt.tight_layout()
        _maybe_show(show)
        return fig, axes

    # -- Solver benchmark ---------------------------------------------------

    BENCHMARK_ERRORS = ('freq (mode of interest)', 'freq (passband)', 'freq (all modes)',
                        'R/Q', 'Epk/Eacc', 'Bpk/Eacc', 'G', 'ff')

    def benchmark(self, variants=None, sweep=None, n_repeats=5, reference=None,
                  eigenmode_config=None, **kwargs):
        """Time and score eigenmode solver settings on this cavity.

        Each variant is a set of eigenmode-config overrides. With *sweep* (config
        key -> list of values), every variant runs at every combination of the
        swept values. Each point is simulated *n_repeats* times (mesh, solve and
        figures of merit, timed stage by stage) and scored against a reference
        solve on the same mesh, converged far beyond any setting being compared.
        Nothing is written to the cavity's results.

        >>> cav.eigenmode.benchmark(
        ...     variants={'direct': {}, 'bddc': {'preconditioner': 'bddc'}},
        ...     sweep={'pinvit_tol': [1e-4, 1e-6, 1e-8]})
        >>> cav.eigenmode.benchmark_table()
        >>> cav.eigenmode.plot_benchmark()

        Parameters
        ----------
        variants : dict, optional
            Label -> eigenmode-config overrides. Default: the defaults alone. A
            variant asking for a ``direct_solver`` this NGSolve build does not
            provide is skipped with a warning, so one benchmark runs on every
            platform.
        sweep : dict, optional
            Config key -> list of values, applied on top of every variant.
        n_repeats : int
            Runs per point (default 5). The eigensolver starts from random vectors,
            so times and errors are reported as a mean and a spread.
        reference : dict, optional
            Overrides for the reference solve. Default: the direct preconditioner
            converged to ``pinvit_tol=1e-10``.
        eigenmode_config : dict, optional
            The config every variant is merged over (with ``**kwargs``), itself
            merged over the defaults. One polarisation at a time.

        Returns
        -------
        pandas.DataFrame
            One row per run: ``variant``, the swept keys, ``repeat``,
            ``No of DOFs``, ``PINVIT iterations``, ``converged``, ``stalled``
            (stopped at the residual's round-off floor, above ``pinvit_tol``), the stage times
            ``mesh [s]``, ``solve [s]`` and ``QOIs [s]``, the time per simulation
            ``total [s]``, the worst relative error against the reference of
            ``freq (mode of interest)``, ``freq (passband)`` (the lowest ``n_cells``
            modes), ``freq (all modes)``, ``R/Q``, ``Epk/Eacc``, ``Bpk/Eacc``, ``G``
            and ``ff``, and ``freq per mode``. The voltage-derived quantities are
            scored on the modes that carry a voltage. Also kept as
            :attr:`benchmark_df`.
        """
        cfg = merge_config(DEFAULT_EIGENMODE_CONFIG, eigenmode_config, kwargs)
        keys = list(sweep or {})
        runs = []
        for label, over in (variants or {'default': {}}).items():
            backend = (over or {}).get('direct_solver')
            if backend and not direct_solver_available(backend):
                warnings.warn(f"benchmark: skipping {label!r}, this NGSolve build has no "
                              f"{backend!r} direct solver.", CavsimWarning, stacklevel=2)
                continue
            for combo in (product(*(sweep[k] for k in keys)) if keys else [()]):
                tags = dict(zip(keys, combo))
                runs.append((label, {**(over or {}), **tags}, tags))
        self.cavity._ensure_workspace()
        self.cavity.create()
        rows, references = NGSolveMEVP().solve_benchmark(self.cavity, cfg, runs,
                                                         n_repeats, reference)
        df = pd.DataFrame(rows)
        df.attrs['sweep'] = keys
        df.attrs['references'] = references
        self.benchmark_df = df
        return df

    def _benchmark_stats(self, df=None):
        """Per (variant, swept value): time mean and std, and the mean and std of
        log10(error) for every error column that applies."""
        df = self.benchmark_df if df is None else df
        if df is None or df.empty:
            raise ValueError("no benchmark results: run cav.eigenmode.benchmark() first.")
        keys = ['variant'] + list(df.attrs.get('sweep', []))
        by = [df[k] for k in keys]
        stats = pd.DataFrame({
            'time mean': df.groupby(by, sort=False)['total [s]'].mean(),
            'time std': df.groupby(by, sort=False)['total [s]'].std(),
            'solve mean': df.groupby(by, sort=False)['solve [s]'].mean(),
            'iterations': pd.to_numeric(df['PINVIT iterations'], errors='coerce')
                            .groupby(by, sort=False).mean()})
        errors = [e for e in self.BENCHMARK_ERRORS if e in df and df[e].notna().any()]
        for e in errors:
            lg = np.log10(pd.to_numeric(df[e], errors='coerce').clip(lower=1e-16))
            stats[f'{e} log mean'] = lg.groupby(by, sort=False).mean()
            stats[f'{e} log std'] = lg.groupby(by, sort=False).std()
        return stats.reset_index(), keys, errors

    def benchmark_table(self, df=None):
        """The benchmark summarised per (variant, swept value).

        Time per simulation is the mean and standard deviation of ``total [s]``
        over the repeats; ``solve [s]`` and ``PINVIT iterations`` are means. The
        errors vary over orders of magnitude between runs, so each is the
        geometric mean (the mean of log10) of its worst relative error. Error
        columns that do not apply to the cavity (ff of a single cell) are left out.
        """
        stats, keys, errors = self._benchmark_stats(df)
        out = stats[keys].copy()
        out['time per simulation [s]'] = stats['time mean']
        out['std [s]'] = stats['time std']
        out['solve [s]'] = stats['solve mean']
        out['PINVIT iterations'] = stats['iterations']
        for e in errors:
            out[e] = 10 ** stats[f'{e} log mean']
        return out.set_index(keys)

    def plot_benchmark(self, df=None, kind='accuracy', errors=None, show=True):
        """Plot a benchmark from :meth:`benchmark`.

        ``kind='accuracy'`` (default): one panel per error quantity, the error
        against the time per simulation, one line per variant through its swept
        values (labelled). ``kind='time'``: the time per simulation against the
        swept value. ``kind='modes'``: the frequency error of every mode.

        Every point is a mean over the repeats with its spread as error bars: the
        standard deviation of the time, and ``10**(mu +- sigma)`` of the error,
        whose mean and standard deviation are taken of log10.
        """
        stats, keys, available = self._benchmark_stats(df)
        df = self.benchmark_df if df is None else df
        sweep = keys[1:]
        variants = list(dict.fromkeys(stats['variant']))

        def fmt(v):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return f'{v:.0e}' if v and (abs(v) < 1e-3 or abs(v) >= 1e4) else f'{v:g}'
            return str(v)

        def point_label(row):
            return ', '.join(fmt(row[k]) for k in sweep)

        with house_style():
            if kind == 'accuracy':
                errors = [e for e in (errors or available) if e in available]
                ncols = min(4, len(errors))
                nrows = int(np.ceil(len(errors) / ncols))
                fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows),
                                         squeeze=False, layout='constrained')
                for ax, e in zip(axes.flat, errors):
                    for i, v in enumerate(variants):
                        s = stats[stats['variant'] == v]
                        mu, sd = s[f'{e} log mean'].values, s[f'{e} log std'].fillna(0).values
                        y = 10 ** mu
                        ax.errorbar(s['time mean'], y, xerr=s['time std'].fillna(0),
                                    yerr=[y - 10 ** (mu - sd), 10 ** (mu + sd) - y],
                                    fmt='o-', ms=4, lw=0.8, capsize=2,
                                    color=WARM[i % len(WARM)], label=v)
                        if sweep:
                            for (_, row), xx, yy in zip(s.iterrows(), s['time mean'], y):
                                ax.annotate(point_label(row), (xx, yy), fontsize=7,
                                            textcoords='offset points', xytext=(3, 3),
                                            color=WARM[i % len(WARM)])
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_title(e, fontsize=10)
                    ax.set_xlabel('time per simulation [s]')
                for row in axes:
                    row[0].set_ylabel('worst relative error')
                for ax in axes.flat[len(errors):]:
                    ax.set_visible(False)
                axes.flat[0].legend(fontsize=8)
            elif kind == 'time':
                fig, ax = plt.subplots(figsize=(7, 4.2), layout='constrained')
                numeric = (len(sweep) == 1
                           and pd.api.types.is_numeric_dtype(stats[sweep[0]]))
                for i, v in enumerate(variants):
                    s = stats[stats['variant'] == v]
                    x = s[sweep[0]] if numeric else np.full(len(s), i)
                    ax.errorbar(x, s['time mean'], yerr=s['time std'].fillna(0), fmt='o-',
                                ms=5, capsize=3, color=WARM[i % len(WARM)], label=v)
                if numeric:
                    vals = stats[sweep[0]].astype(float)
                    if (vals > 0).all() and vals.max() / vals.min() >= 100:
                        ax.set_xscale('log')
                    ax.set_xlabel(sweep[0])
                else:
                    ax.set_xticks(range(len(variants)), variants)
                ax.set_yscale('log')
                ax.set_ylabel('time per simulation [s]')
                ax.legend(fontsize=8)
                axes = ax
            elif kind == 'modes':
                fig, ax = plt.subplots(figsize=(8, 4.4), layout='constrained')
                groups = df.groupby(keys, sort=False)
                for i, (key, g) in enumerate(groups):
                    per = np.log10(np.clip(np.vstack(g['freq per mode'].values), 1e-16, None))
                    mu, sd = per.mean(axis=0), per.std(axis=0, ddof=1) if len(per) > 1 else 0 * per[0]
                    idx = np.arange(1, per.shape[1] + 1)
                    y = 10 ** mu
                    label = ', '.join(f'{k}' if not isinstance(k, float) else f'{k:g}'
                                      for k in (key if isinstance(key, tuple) else (key,)))
                    ax.errorbar(idx, y, yerr=[y - 10 ** (mu - sd), 10 ** (mu + sd) - y],
                                fmt='o-', ms=4, lw=0.8, capsize=2, color=WARM[i % len(WARM)],
                                label=label)
                    ax.set_xticks(idx)
                ax.set_yscale('log')
                ax.set_xlabel('mode number')
                ax.set_ylabel('relative frequency error')
                ax.legend(fontsize=7)
                axes = ax
            else:
                raise ValueError(f"kind must be 'accuracy', 'time' or 'modes', not {kind!r}")
            fig.suptitle(f'{self.cavity.name}: solver benchmark')
        _maybe_show(show)
        return fig, axes

    def mpole_qois(self, pol, all_modes=True):
        """QOIs of an m-pole solve, keyed by mode index (``all_modes=True``)
        or just the fundamental mode's dict."""
        fname = 'qois_all_modes.json' if all_modes else 'qois.json'
        folder = self._results_folder(pol)
        if folder is None:
            return {}
        path = folder / fname
        if not path.exists():
            warning(f"No {pol_name(pol_number(pol))} eigenmode results found at {path}. "
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
        if not self.cavity.self_dir:
            return pd.DataFrame()
        for m in sorted(MPOLE_NAMES):
            path = self.pol_folder(m) / 'qois_all_modes.json'
            if not path.exists():
                continue
            warn_if_stale(self.cavity, self.pol_folder(m), 'The eigenmode results')
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
                  unit='k', model='rlc', modes=None):
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
            Longitudinal [Ohm] or transverse [Ohm/m]. Every computed mode is
            summed, each through its type-appropriate R/Q: the longitudinal
            spectrum is built from the monopole R/Q (higher multipoles are zero on
            axis, so they add nothing but still set the span), the transverse from
            every ``m>=1`` multipole's Panofsky-Wenzel R/Q (dipole + quadrupole +
            …), not the dipole alone. The cavity is axisymmetric, so the two
            transverse planes are degenerate and one R/Q describes both.
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
        modes : pandas.DataFrame, optional
            Reconstruct from this mode table instead of the full ``qois_df`` —
            typically a filtered one, e.g. the output of :meth:`stable_modes`.
            Sampling still adapts to each mode's linewidth, which a hand-built
            uniform grid cannot do: a Q of 3e4 at 1.3 GHz is only 44 kHz wide and
            an evenly spaced grid steps straight over it.
        model : {'rlc', 'qnm'}
            How the modes are summed.

            ``'rlc'`` (default) is the sum of parallel resonators above. It is
            exact for **isolated** resonances and is what accelerator tables
            assume, but it gives every mode a positive-real shunt impedance, so
            modes that overlap add *in phase*. Below a beam-pipe cutoff, where
            linewidths are far narrower than the mode spacing, that never
            matters.

            ``'qnm'`` sums the complex quasi-normal-mode residues instead
            (``'Re(R/Q) [Ohm]'`` / ``'Im(R/Q) [Ohm]'``), so overlapping poles
            interfere rather than pile up — the effect the RLC form cannot
            represent. It is **experimental and warns**: the residue magnitudes
            are exact but their phases are not yet physical, and the result can
            violate passivity above the cutoff. For a trustworthy above-cutoff
            impedance, filter the modes with :meth:`stable_modes` and keep
            ``'rlc'``.

            Neither model contains anything the mode set does not: no broadband
            or resistive-wall term, and nothing above the highest computed mode.

        Returns
        -------
        pandas.DataFrame
            Columns ``f [MHz]``, ``|Z| [u]``, ``Re(Z) [u]``, ``Im(Z) [u]`` with
            ``u`` = kOhm (longitudinal) or kOhm/m (transverse) by default.

        Warns
        -----
        UserWarning
            Every reconstruction warns that the mode set is only as good as the
            beam-pipe/PML mode filtering behind it, and that no filter is perfect
            (see :meth:`stable_modes`).

        >>> z = cav.eigenmode.impedance()                       # 0 .. highest mode
        >>> zt = cav.eigenmode.impedance('transverse', span=(400, 1200), Q=1e4)
        >>> z = cav.eigenmode.impedance(unit='')                # in Ohm instead
        >>> z = cav.eigenmode.impedance(model='qnm')            # interfering poles
        """
        df = self._impedance_frame(kind, span, n_points, Q, unit, model, modes)
        if not df.empty:
            self._warn_mode_filtering(modes)
        return df

    def _warn_mode_filtering(self, modes, plotted_unfiltered=False):
        """The caveat every eigenmode impedance carries: the mode set is filtered
        imperfectly, or not at all.

        An open (PML) solve returns genuine cavity modes mixed with modes of the beam
        pipe and of the absorbing block, and above the pipe cutoff the artefacts are
        the broad, high-R/Q ones that dominate ``|Z|``. The best discriminant found
        so far is frequency invariance between two solves that differ only in pipe
        (or PML) length, and even that misclassifies a few per cent of modes in
        both directions: it drops real modes whose Q has not converged with pipe
        length, or that the other solve missed, and it keeps artefacts whose
        frequency repeats within the tolerance by chance. The error does not have a
        fixed sign either. The same filter under-counted cavity modes on a 3-cell
        pillbox and over-counted them on a 9-cell one.
        """
        total = len(self.qois_df)
        if modes is None:
            head = (f"This impedance sums all {total} computed modes, with no "
                    "beam-pipe/PML mode filter. Above the beam-pipe cutoff it includes "
                    "modes of the pipe and the absorbing layer, which are artefacts "
                    "of the truncated domain and can dominate |Z|. Filter the set with "
                    "cav.eigenmode.stable_modes(<a second solve with a different "
                    "beampipe_length or pml_length>) and pass modes=. ")
        else:
            head = (f"This impedance sums {len(modes)} of {total} computed modes, "
                    "after beam-pipe/PML mode filtering. ")
        tail = _MODE_FILTER_CAVEAT
        if plotted_unfiltered:
            tail += (" The reconstruction from every mode is drawn on the same axes "
                     "(dashed) to show how much the filter changes.")
        warnings.warn(head + tail, CavsimWarning, stacklevel=3)

    def _impedance_frame(self, kind='longitudinal', span=None, n_points=8001, Q=None,
                         unit='k', model='rlc', modes=None):
        """Build the reconstructed spectrum for :meth:`impedance`, without the
        mode-filtering warning (callers that draw several spectra warn once)."""
        transverse = not str(kind).lower().startswith('long')

        df = self.qois_df if modes is None else modes
        modes = df.sort_values('freq [MHz]') if not df.empty else df
        if modes.empty:
            error(f"No eigenmode results to build a {kind} impedance from. "
                  f"Run cav.eigenmode.run(...) first.")
            return pd.DataFrame()

        # Reconstruct from EVERY computed mode. Each contributes through the R/Q
        # that matches the requested impedance type: the monopole R/Q is
        # longitudinal, the m>=1 R/Q is the transverse (Panofsky-Wenzel) R/Q. A
        # mode of the *other* type has zero R/Q of this type, so it enters the sum
        # contributing nothing — longitudinal stays a monopole sum (higher m are
        # exactly zero on axis), while transverse spans every m>=1 multipole
        # (dipole + quadrupole + ...), not just the dipole.
        m_arr = modes['m'].to_numpy()
        applicable = (m_arr >= 1) if transverse else (m_arr == 0)
        if not applicable.any():
            what = 'm>=1 (dipole/quadrupole/…)' if transverse else 'monopole'
            error(f"No {what} eigenmode results to build a {kind} impedance from. "
                  f"Run cav.eigenmode.run(...) with that polarisation first.")
            return pd.DataFrame()

        model = str(model).lower()
        if model not in ('rlc', 'qnm'):
            raise ValueError(f"model must be 'rlc' or 'qnm'; got {model!r}.")

        f0 = modes['freq [MHz]'].to_numpy(dtype=float) * 1e6        # -> Hz
        roq = np.where(applicable, modes['R/Q [Ohm]'].to_numpy(dtype=float), 0.0)

        roq_c = None
        if model == 'qnm':
            if not {'Re(R/Q) [Ohm]', 'Im(R/Q) [Ohm]'} <= set(modes.columns):
                error("model='qnm' needs the complex mode residues, which only a run "
                      "with a complex eigenvalue produces (boundary_conditions='oo', "
                      "or a lossy dielectric). Re-run, or use the default model='rlc'.")
                return pd.DataFrame()
            warning(
                "model='qnm' is EXPERIMENTAL and not validated on radiating modes: "
                "the residue phases can come out unphysical (the spectrum may show "
                "Re(Z) < 0). For an above-cutoff impedance, filter the mode set with "
                "cav.eigenmode.stable_modes() and use the default model='rlc'.")
            roq_c = np.where(applicable,
                             modes['Re(R/Q) [Ohm]'].to_numpy(dtype=float)
                             + 1j * modes['Im(R/Q) [Ohm]'].to_numpy(dtype=float),
                             0.0)
        if Q is None:
            # BLOCKED on closed boundaries. With CLOSED (PEC/PMC) beam-pipe ends a mode
            # ABOVE the pipe cutoff -- which physically radiates out of the pipe and so
            # has a low, radiation-limited Q -- is instead REFLECTED and reported with
            # its high ohmic Q0. The sum-of-resonators |Z| built from that Q0
            # OVERESTIMATES every propagating mode, often by 100-1000x, and turns a
            # smooth above-cutoff continuum into spurious sharp lines. That is not a
            # result worth returning, so it is refused rather than warned about: solve
            # with at least one open (PML) end, use the wakefield, or state the Q
            # yourself with Q=.
            if not _solved_with_open_boundary(self.config, modes):
                error(
                    "refusing to reconstruct impedance from a CLOSED-boundary eigenmode "
                    "solve. The pipe ends reflect, so every mode above the beam-pipe "
                    "cutoff keeps its ohmic Q0 instead of its radiation Q and |Z| comes "
                    "out ~100-1000x too high. Re-run with boundary_conditions='oo' (an "
                    "open/PML end -- see the open_boundary_impedance example), use "
                    "cav.wakefield for the broadband answer, or pass an explicit Q= if "
                    "you know the loaded Q and want the reconstruction anyway.")
                return pd.DataFrame()
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

        if model == 'qnm':
            z = reconstruct_impedance_qnm(f0, roq_c, q, f_span, transverse=transverse)
        else:
            z = reconstruct_impedance(f0, roq, q, f_span, transverse=transverse)
        return impedance_frame(f_span * 1e-6, z, unit=unit, transverse=transverse)

    def stable_modes(self, other, **kwargs):
        """The cavity modes of this open solve, judged against a second solve of the
        same cavity with a different ``beampipe_length`` or ``pml_length``.

        A cavity mode keeps its frequency when the pipe changes; a mode of the pipe
        or of the absorbing layer moves. Pass the second run's namespace or its
        ``qois_df``. Returns the subset of this solve's ``qois_df`` judged to be
        cavity modes; :meth:`classify_modes` returns the evidence for every mode.
        The rule is the best discriminant found so far, and it is not perfect:
        see :func:`~cavsim2d.analysis.impedance.classify_modes`.

            trusted = cav_a.eigenmode.stable_modes(cav_b.eigenmode)
        """
        df = self.qois_df
        other_df = getattr(other, 'qois_df', other)
        if df.empty or other_df is None or len(other_df) == 0:
            return df.iloc[0:0]
        return df[pml_stable_modes(df, other_df, **kwargs)]

    def classify_modes(self, other, **kwargs):
        """Every mode of this solve with its verdict (``'cavity mode'``) and the
        evidence behind it, against a second solve with a different pipe or PML
        length. See :func:`~cavsim2d.analysis.impedance.classify_modes`."""
        df = self.qois_df
        other_df = getattr(other, 'qois_df', other)
        return df.join(classify_modes(df, other_df, **kwargs))

    #: External-Q column by solve type, first present wins: a port solve reports
    #: Q_ext directly, an open (PML) or lossy solve through the complex eigenvalue,
    #: and a perturbative dielectric run through its loss integral.
    Q_EXT_COLUMNS = ('Q_ext []', 'Q_eig []', 'Q_diel []')

    def plot_q_ext(self, ax=None, keep=None, label=None, color=None, marker='o',
                   polarisation=None, cutoff=None, wall_q=False, show=True, **kwargs):
        """External Q against frequency, one marker per computed mode.

        The column is whichever this solve produced (see :data:`Q_EXT_COLUMNS`):
        ``'Q_ext []'`` for waveguide ports, ``'Q_eig []'`` for a PML or a lossy
        material, ``'Q_diel []'`` otherwise. Call it once per solve on the same
        axes to compare methods.

        Parameters
        ----------
        keep : array of bool or pandas.DataFrame, optional
            Modes to draw filled; the rest are drawn hollow and labelled as
            artefacts. A boolean mask aligned with ``qois_df``, or a filtered frame
            (e.g. :meth:`stable_modes`), whose index is used.
        polarisation : str or int, optional
            Only this azimuthal order.
        cutoff : float or sequence of float, optional
            Beam-pipe cutoff frequencies [MHz], drawn as dotted vertical lines.
        wall_q : bool
            Also draw the median wall Q0 as a dashed horizontal line: an external
            Q far above it no longer matters for the total.
        """
        df = self.qois_df
        if df is None or df.empty:
            info("No eigenmode data available.")
            return ax
        if polarisation is not None:
            df = df[df['m'] == pol_number(polarisation)]
        col = next((c for c in self.Q_EXT_COLUMNS if c in df.columns), None)
        if col is None:
            error("This solve has no external-Q column: it needs an open ('oo'), port "
                  "('port') or lossy boundary. See Q_EXT_COLUMNS.")
            return ax
        if keep is None:
            real = np.ones(len(df), dtype=bool)
        elif isinstance(keep, pd.DataFrame):
            real = df.index.isin(keep.index)
        else:
            real = np.asarray(pd.Series(keep, index=self.qois_df.index).loc[df.index],
                              dtype=bool)
        label = label or self.cavity.name
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(9, 4.5))
            if color is None:
                # one house-palette colour per call, so methods overlaid on the
                # same axes stay apart
                calls = getattr(ax, '_q_ext_calls', 0)
                ax._q_ext_calls = calls + 1
                color = WARM[(2 * calls) % len(WARM)]
            f, q = df['freq [MHz]'].to_numpy(), df[col].to_numpy(dtype=float)
            ax.scatter(f[real], q[real], marker=marker, s=46, color=color,
                       edgecolor='k', linewidth=0.5, zorder=3, label=label, **kwargs)
            if (~real).any():
                ax.scatter(f[~real], q[~real], marker=marker, s=46, facecolor='none',
                           edgecolor=color, linewidth=1.2, zorder=2,
                           label=f'{label}, artefacts')
            for fc in np.atleast_1d(cutoff if cutoff is not None else []):
                ax.axvline(float(fc), ls=':', color='k', lw=1.1, zorder=1)
            if wall_q and 'Q_wall []' in df.columns:
                q0 = float(np.nanmedian(df['Q_wall []']))
                ax.axhline(q0, ls='--', color='0.55', lw=1.0, zorder=1,
                           label=f'wall $Q_0$ ({q0:.1e})')
            ax.set_yscale('log')
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r'$Q_\mathrm{ext}$')
        _maybe_show(show)
        return ax

    def plot_impedance(self, kind='longitudinal', ax=None, span=None,
                       n_points=8001, Q=None, unit='k', model='rlc', modes=None,
                       show_unfiltered=True, show=True, **kwargs):
        """Plot the reconstructed impedance (see :meth:`impedance`).

        With a filtered mode table (``modes=``, e.g. from :meth:`stable_modes`),
        the reconstruction from every computed mode is drawn on the same axes
        as a dashed grey line, so the effect of the filter is visible. No mode
        filter is perfect, so that reference stays on the plot. Pass
        ``show_unfiltered=False`` to draw the filtered curve alone.

        Returns the axes, so a wakefield result can be overlaid on the same one —
        both default to kOhm, so the two land on the same scale (pass
        ``show=False`` on the first call to keep the axes live)::

            ax = cav.eigenmode.plot_impedance(show=False)
            cav.wakefield.plot_impedance(ax=ax)

            trusted = cav.eigenmode.stable_modes(cav_longer_pipe.eigenmode)
            cav.eigenmode.plot_impedance(modes=trusted)   # filtered + every mode
        """
        overlay = modes is not None and show_unfiltered
        if overlay and span is None:
            # One span for both curves, so they are comparable: the filtered set
            # alone would stop at its own highest mode.
            span = (0.0, float(self.qois_df['freq [MHz]'].max()))
        df = self._impedance_frame(kind=kind, span=span, n_points=n_points, Q=Q,
                                   unit=unit, model=model, modes=modes)
        if df.empty:
            return ax
        df_all = pd.DataFrame()
        if overlay:
            df_all = self._impedance_frame(kind=kind, span=span, n_points=n_points,
                                           Q=Q, unit=unit, model=model, modes=None)
        self._warn_mode_filtering(modes, plotted_unfiltered=not df_all.empty)

        transverse = str(kind).lower().startswith('trans')
        u = impedance_unit(unit, transverse)
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 4))
            if not df_all.empty:
                ax.plot(df_all['f [MHz]'], df_all[f'|Z| [{u}]'], ls='--', lw=1.0,
                        color='0.6', zorder=1,
                        label=f'{self.cavity.name}, every mode ({len(self.qois_df)}, '
                              f'unfiltered)')
            if modes is not None:
                kwargs.setdefault('label', f'{self.cavity.name}, filtered '
                                           f'({len(modes)} of {len(self.qois_df)} modes)')
            kwargs.setdefault('label', f'{self.cavity.name} (eigenmode)')
            kwargs.setdefault('zorder', 2)
            ax.plot(df['f [MHz]'], df[f'|Z| [{u}]'], **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(_z_axis_label(unit, transverse))
            ax.set_yscale('log')
            if not df_all.empty:
                ax.legend()
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

        Before overwriting existing results, or anything in the cavity's folder
        computed for another geometry, it asks; ``force=True`` (or
        ``overwrite=True``) answers yes. See
        :func:`cavsim2d.utils.integrity.guard_runs`.

        Returns this namespace, so calls chain: ``cav.eigenmode.run(cfg).qois``.
        """
        merged_config = merge_config(DEFAULT_EIGENMODE_CONFIG,
                                     _user_config('eigenmode', eigenmode_config, kwargs))

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed

        pols = {pol_name(m) for m in parse_polarisations(merged_config.get('polarisation', 0))}
        notice_tuned_versions([self.cavity], 'eigenmode')
        if not guard_runs([self.cavity], 'eigenmode', merged_config, pol_labels=pols,
                          title='Eigenmode'):
            return self
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
        # The cavity's own summary attributes (cav.freq, cav.R_Q, ...) follow the
        # results whichever way the run was started.
        try:
            self.cavity.get_eigenmode_qois(merged_config)
        except FileNotFoundError:
            pass                                 # the solve reported its own failure
        return self

    def _invalidate(self):
        self._modes = None
        self._qois = None

    # -- Visualisation ------------------------------------------------------

    def plot_spectrum(self, ax=None, bands=False, show=True):
        """Spectral comb of every computed eigenfrequency, coloured by polarisation.

        Every eigenmode is a vertical line at its frequency, one colour per
        polarisation (monopole, dipole, …) — so all solved polarisations appear,
        not just the monopole. A downward triangle marks the **mean frequency of
        each passband** (a cluster of ``n_cells`` consecutive modes), shaded per
        band within the polarisation's colour family. With ``bands=True`` each
        passband's frequency span is shaded, giving a quick read of where the
        passbands sit and how wide they are.

        Solve extra polarisations (``eigenmode_config['polarisation']``) to see
        them here; with only the monopole solved it shows the monopole comb.
        """
        df = self.qois_df
        if df is None or df.empty or 'freq [MHz]' not in df:
            info("No eigenmode data available.")
            return ax

        n = int(getattr(self.cavity, 'n_cells', 1) or 1)
        with house_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(11, 3.2))

            for m, sub in df.groupby('m'):
                freqs = np.sort(sub['freq [MHz]'].to_numpy())
                freqs = freqs[freqs > 0]
                if len(freqs) == 0:
                    continue
                hue = polarisation_color(int(m))
                pretty = pol_name(int(m))
                # one vertical line per mode
                ax.vlines(freqs, 0.0, 1.0, color=hue, lw=1.2, alpha=0.9)
                # passbands = consecutive groups of n_cells modes; a triangle at
                # each passband's mean frequency, shaded by band within the hue.
                n_pass = int(np.ceil(len(freqs) / n))
                band_cols = shades(hue, n_pass)
                labelled = False
                for b in range(n_pass):
                    seg = freqs[b * n:(b + 1) * n]
                    if len(seg) == 0:
                        continue
                    ax.plot([seg.mean()], [1.05], marker='v', ms=9,
                            color=band_cols[b], clip_on=False,
                            label=(pretty if not labelled else None))
                    labelled = True
                    if bands:
                        ax.axvspan(seg.min(), seg.max(), color=hue, alpha=0.08,
                                   zorder=0)

            ax.set_ylim(0.0, 1.12)
            ax.set_yticks([])
            ax.set_xlabel('frequency [MHz]')
            ax.set_title(f'Eigenfrequency spectrum — {self.cavity.name}')
            # one legend entry per polarisation
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), loc='upper right',
                          ncol=len(uniq))
        _maybe_show(show)
        return ax

    def _gather_modes(self, var='freq [MHz]', uq=True):
        """Every eigenmode as ``(m, mode_index, mean, std|None)`` for *var*.

        Reads the per-mode UQ distribution (``<cav>/uq/uq_all_modes.json``, keyed
        ``'<mode index>-<m>'``) when ``uq`` and it exists — so each mode carries
        the mean and standard deviation of *var* over the perturbed geometries —
        else the deterministic modes from :attr:`qois_df` (``std=None``)."""
        out = []
        if uq:
            path = os.path.join(self.cavity.uq_dir, 'uq_all_modes.json')
            if os.path.exists(path):
                with open(path) as f:
                    allm = json.load(f)
                for key, qois in allm.items():
                    fd = qois.get(var)
                    if fd is None:
                        continue
                    parts = str(key).split('-')
                    mode_i = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 0
                    out.append((m, mode_i, float(fd['expe'][0]),
                                float(fd['stdDev'][0])))
                if out:
                    return out
            info("No per-mode UQ results (uq_all_modes.json) — falling back to "
                 "the deterministic modes.")
        df = self.qois_df
        if df is not None and not df.empty and var in df:
            for _, r in df.sort_values('freq [MHz]').iterrows():
                out.append((int(r['m']), int(r['mode']) if 'mode' in r else 0,
                            float(r[var]), None))
        return out

    def plot_spectra(self, var='freq [MHz]', modes=None, freq_range=None, pol=None,
                     reference=None, density_cap=None, std_cap=None, kde=True,
                     uq=True, break_axis=False, ax=None, show=True, **kwargs):
        r"""Mode-spectrum plot — the general view that supersedes
        ``plot_tm010_spectrum``.

        For every selected eigenmode it draws the distribution of *var* across the
        UQ samples — an area-normalised Gaussian of the mode's mean/std
        (``kde=True``), a line for a deterministic / zero-spread mode, or
        (``kde=False``) the value vs mode index with error bars. Modes are
        **coloured by polarisation** (monopole, dipole, …), so it covers
        higher-order modes, not just the TM\ :sub:`010` monopole passband.

        Selection (dynamic):

        - ``pol`` — a polarisation name/number or list ('monopole', 'dipole', 1, …);
          default every solved polarisation.
        - ``modes`` — a mode index or list of indices to keep (per polarisation).
        - ``freq_range`` — one ``(lo, hi)`` window, or a **list** of windows
          ``[(lo1, hi1), (lo2, hi2), ...]`` in *var* units; keep only modes whose
          mean lands inside any window. With ``break_axis=True`` and several
          windows, each window is drawn as its own independent panel (one axis per
          range, in ascending order) — the caller controls the split directly.
        - ``density_cap`` — cap the plotted **probability density** (the y-axis).
          A sharply-tuned mode has a near-zero spread and therefore a very tall,
          narrow peak (area-normalised, height ~ ``1/std``) that dwarfs the
          broader modes; capping the density clips that spike and keeps the rest
          visible. (``std_cap`` is accepted as an alias.) NB this caps the
          *height*, not the width — capping the std instead would make a narrow
          mode taller, the opposite of what you want.

        ``reference`` overlays external labelled spectra (mean/std) — e.g. the
        Corno et al. RI/EZ European-XFEL measurements. Per-mode UQ data is read
        from ``<cav>/uq/uq_all_modes.json`` when ``uq`` and present, else the
        deterministic modes.
        """
        density_cap = density_cap if density_cap is not None else std_cap
        entries = self._gather_modes(var=var, uq=uq)
        if not entries:
            info("No eigenmode data available.")
            return ax

        want_m = None
        if pol is not None:
            pl = [pol] if isinstance(pol, (str, int)) else list(pol)
            want_m = {pol_number(p) for p in pl}
        want_modes = None
        if modes is not None:
            want_modes = {modes} if isinstance(modes, int) else set(modes)

        # ``freq_range`` is one ``(lo, hi)`` window or a list of them. With several
        # windows a mode is kept if it falls in ANY, and (when break_axis) each
        # window becomes its own panel — so the caller controls the split directly
        # instead of relying on the automatic gap detection.
        ranges = None
        if freq_range is not None:
            fr = list(freq_range)
            if len(fr) == 2 and all(isinstance(v, (int, float)) for v in fr):
                ranges = [(float(fr[0]), float(fr[1]))]
            else:
                ranges = [(float(lo), float(hi)) for lo, hi in fr]
            ranges.sort()

        sel = []
        for m, mi, mean, std in entries:
            if want_m is not None and m not in want_m:
                continue
            if want_modes is not None and mi not in want_modes:
                continue
            if ranges is not None and not any(lo <= mean <= hi for lo, hi in ranges):
                continue
            sel.append((m, mi, mean, std))
        if not sel:
            info("No modes match the selection.")
            return ax
        sel.sort(key=lambda e: e[2])

        refs = _reference_series(reference)
        with house_style():
            if kde:
                spreads = [s for _, _, _, s in sel if s]
                widest = (max(spreads) if spreads
                          else max((sel[-1][2] - sel[0][2]) * 0.01, 1e-6))

                def _draw_kde(a, window=None):
                    # ``window`` = (lo, hi): a per-band panel. Every mode is still
                    # drawn (lines that spill across the x-limit stay continuous),
                    # but the y-axis is scaled to the peak of the modes *inside* the
                    # window, so a broad low HOM band isn't crushed by the tall
                    # tuned fundamental sitting on another panel.
                    def _in(v):
                        return window is None or (window[0] <= v <= window[1])
                    seen = set()
                    peak = 0.0
                    for m, mi, mean, std in sel:
                        color = polarisation_color(m)
                        label = pol_name(m) if m not in seen else None
                        seen.add(m)
                        if std and std > 0:
                            x = np.linspace(mean - 4 * std, mean + 4 * std, 400)
                            yv = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                            a.plot(x, yv, color=color, label=label, **kwargs)
                            if _in(mean):
                                peak = max(peak, float(yv.max()))
                        else:
                            a.axvline(mean, color=color, ls='--', label=label)
                    for i, (rlabel, rmean, rstd) in enumerate(refs):
                        color = WARM[(i + 3) % len(WARM)]
                        std_r = rstd if rstd is not None else [widest] * len(rmean)
                        grid = np.linspace(float(min(rmean)) - 4 * widest,
                                           float(max(rmean)) + 4 * widest, 2000)
                        gsum = _gaussian_sum(rmean, std_r, grid)
                        a.plot(grid, gsum, color=color, label=rlabel)
                        msk = np.ones_like(grid, bool) if window is None else \
                            (grid >= window[0]) & (grid <= window[1])
                        if msk.any():
                            peak = max(peak, float(gsum[msk].max()))
                    # Cap the probability DENSITY (not the width): a sharply-tuned
                    # mode is a very tall, narrow peak that would flatten the broad
                    # modes. Scale each panel to its own visible peak, then clip to
                    # density_cap if the user set one.
                    top = 1.08 * peak if peak > 0 else None
                    if density_cap:
                        top = density_cap if top is None else min(density_cap, top)
                    a.set_ylim(0.0, top)

                # Explicit ``freq_range`` windows become the panels directly;
                # otherwise auto-detect the bands from the gaps between modes.
                if break_axis and ax is None and ranges is not None and len(ranges) > 1:
                    clusters, explicit = ranges, True
                else:
                    centers = np.asarray(sorted([e[2] for e in sel]
                                                + [v for _, rm, _ in refs for v in rm]), float)
                    clusters = (self.cavity._frequency_clusters(centers)
                                if (break_axis and ax is None and centers.size > 1) else None)
                    explicit = False
                if clusters and len(clusters) > 1:
                    # One independent panel per frequency band (subplot_mosaic, NOT
                    # a shared-y broken axis): the bands can differ in amplitude by
                    # orders of magnitude — a broad low HOM would vanish under the
                    # tall tuned fundamental on a common y-scale — so each gets its
                    # own y-axis and autoscales to its own data. Panels are drawn at
                    # equal width (no width_ratios) regardless of each band's span.
                    labels = [f'band{i}' for i in range(len(clusters))]
                    fig, axd = plt.subplot_mosaic(
                        [labels], figsize=(11, 4),
                        gridspec_kw={'wspace': 0.16})
                    axes = [axd[k] for k in labels]
                    for a, (lo, hi) in zip(axes, clusters):
                        _draw_kde(a, window=(lo, hi))
                        if explicit:
                            a.set_xlim(lo, hi)          # honour the caller's window
                        else:
                            pad = 4 * widest + (hi - lo) * 0.05 + 1e-6
                            a.set_xlim(lo - pad, hi + pad)
                    fig.supxlabel(var)
                    fig.supylabel('probability density')
                    ax = axes[-1]
                else:
                    if ax is None:
                        fig, ax = plt.subplots(figsize=(11, 4))
                    axes = [ax]
                    _draw_kde(ax)
                    ax.set_xlabel(var)
                    ax.set_ylabel('probability density')
            else:
                if ax is None:
                    fig, ax = plt.subplots(figsize=(11, 4))
                by_m = defaultdict(list)
                for m, mi, mean, std in sel:
                    by_m[m].append((mi, mean, std))
                for m, lst in sorted(by_m.items()):
                    lst.sort()
                    ax.errorbar([t[0] + 1 for t in lst], [t[1] for t in lst],
                                yerr=[t[2] or 0 for t in lst], marker='o', ms=5,
                                capsize=3, color=polarisation_color(m),
                                label=pol_name(m), **kwargs)
                for i, (rlabel, rmean, rstd) in enumerate(refs):
                    ax.errorbar(np.arange(1, len(rmean) + 1), rmean, yerr=rstd,
                                marker='s', ms=5, capsize=3, ls=':',
                                color=WARM[(i + 3) % len(WARM)], label=rlabel)
                ax.set_xlabel('mode index')
                ax.set_ylabel(var)
                axes = [ax]

            # Gather legend handles from every panel (a band-split spectrum may
            # have monopole on one panel and a HOM on another).
            uniq = {}
            for a in axes:
                h, l = a.get_legend_handles_labels()
                for li, hi in zip(l, h):
                    uniq.setdefault(li, hi)
            if uniq:
                axes[-1].legend(uniq.values(), uniq.keys())
            ax.figure.suptitle(f'Spectrum — {var} — {self.cavity.name}')
        _maybe_show(show)
        return ax

    def plot_tm010_spectrum(self, reference=None, kde=True, uq=False, ax=None,
                            show=True, **kwargs):
        r"""The fundamental (TM\ :sub:`010`) monopole passband — a thin wrapper over
        :meth:`plot_spectra` restricted to the monopole modes (the first
        ``n_cells`` modes). Prefer :meth:`plot_spectra` directly for higher-order
        modes or custom mode/frequency selection."""
        n = int(getattr(self.cavity, 'n_cells', 1) or 1)
        return self.plot_spectra(var='freq [MHz]', pol='monopole',
                                 modes=list(range(n)), reference=reference,
                                 kde=kde, uq=uq, ax=ax, show=show, **kwargs)

    # -- UQ sample distributions (read <cav>/uq/) ----------------------------
    def plot_fm_distribution(self, qois=None, bins='auto', show=True):
        """Histogram of each figure of merit over the UQ samples — reads
        ``<cav>/uq/table.csv`` (one row per perturbed geometry).

        Reproduces WEPB015 Fig. 3 (the FM distributions). ``qois`` selects/orders
        columns (bare names like ``'freq [MHz]'`` match a polarisation-prefixed
        column); default is every numeric column. Needs a Monte-Carlo UQ sweep
        (``uq_config`` with ``method=['normal', N]``) for a real distribution."""
        path = os.path.join(self.cavity.uq_dir, 'table.csv')
        if not os.path.exists(path):
            warning(f"No UQ table at {path} — run a UQ sweep (uq_config) first.")
            return None
        df = pd.read_csv(path, sep='\t').apply(pd.to_numeric, errors='coerce')
        df = df.dropna(axis=1, how='all')

        def _bare(c):
            return c.split(':')[-1].strip()
        if qois is None:
            cols = list(df.columns)
        else:
            if isinstance(qois, str):
                qois = [qois]
            cols = [c for q in qois for c in df.columns if _bare(c) == q or c == q]
        if not cols:
            info("None of the requested qois are in the UQ table.")
            return None

        with house_style():
            fig, axd = plt.subplot_mosaic([cols], figsize=(3 * len(cols), 3),
                                          layout='constrained')
            for c in cols:
                axd[c].hist(df[c].dropna(), bins=bins, density=True,
                            color=WARM[0], edgecolor='k')
                axd[c].set_title(_bare(c))
            axd[cols[0]].set_ylabel('Density')
        _maybe_show(show)
        return axd

    def plot_node_distribution(self, variables, compare_before=True, bins='auto',
                               max_cells=None, show=True):
        """Histogram of the perturbed half-cell variables over the UQ samples —
        reads ``<cav>/uq/nodes.csv``. With ``compare_before=True`` it overlays the
        **before**-continuity spread (``nodes_before_continuity.csv``, written when
        the sweep used ``uq_config['independent_half_cells']=True``), reproducing
        WEPB015 Fig. 4: welding the independently-perturbed half-cells narrows the
        distribution.

        ``variables`` is a list of half-cell variable stems (e.g. ``['Ri', 'Req']``);
        each expands to its per-half-cell columns (``Ri1, Ri2, …``)."""
        after_path = os.path.join(self.cavity.uq_dir, 'nodes.csv')
        if not os.path.exists(after_path):
            warning(f"No UQ nodes at {after_path} — run a multicell UQ sweep first.")
            return None
        after = pd.read_csv(after_path, sep='\t')
        before = None
        bpath = os.path.join(self.cavity.uq_dir, 'nodes_before_continuity.csv')
        if compare_before and os.path.exists(bpath):
            before = pd.read_csv(bpath, sep='\t')

        if isinstance(variables, str):
            variables = [variables]
        col_lists = {}
        for var in variables:
            cols = sorted([c for c in after.columns
                           if c.startswith(var) and c[len(var):].isdigit()
                           and (max_cells is None or int(c[len(var):]) <= max_cells)],
                          key=lambda c: int(c[len(var):]))
            if cols:
                col_lists[var] = cols
        if not col_lists:
            warning(f"No node columns found for variables {variables}.")
            return None

        mosaic = [col_lists[v] for v in col_lists]
        width = max(len(r) for r in mosaic)
        mosaic = [r + ['.'] * (width - len(r)) for r in mosaic]

        with house_style():
            fig, axd = plt.subplot_mosaic(mosaic, figsize=(3 * width, 3 * len(mosaic)),
                                          layout='constrained', sharey=True)
            for var, cols in col_lists.items():
                for col in cols:
                    ax = axd[col]
                    if before is not None:
                        ax.hist(before[col].dropna(), bins=bins, density=True,
                                color=WARM[3 % len(WARM)], alpha=0.55,
                                edgecolor='k', label='before continuity')
                    ax.hist(after[col].dropna(), bins=bins, density=True,
                            color=WARM[0], alpha=0.9, edgecolor='k',
                            label='after continuity')
                    ax.set_xlabel(col)
                axd[cols[0]].set_ylabel('Density')
            if before is not None:
                h, l = axd[col_lists[list(col_lists)[0]][0]].get_legend_handles_labels()
                fig.legend(h, l, loc='outside upper center', ncol=2)
        _maybe_show(show)
        return axd

    # -- Variance-based (Sobol') sensitivity analysis ------------------------
    def run_sensitivity(self, objectives=None, N=1024, degree=2, save=True):
        r"""Sobol' sensitivity analysis of the figures of merit on this cavity's
        UQ samples. Fits a degree-``degree`` polynomial surrogate to
        ``uq/nodes.csv`` (inputs) -> ``uq/table.csv`` (FMs), then evaluates Sobol'
        **main** (``S1``) and **total** (``ST``) indices over a Saltelli QMC
        design of base size ``N``. Reproduces the WEPB015 / Corno et al. (2020)
        method (PCE surrogate + QMC).

        Returns ``{objective: {input_name: {'S1':…, 'ST':…}}}`` and, with
        ``save=True``, writes ``uq/sobol.json`` (indices) and ``uq/surrogate.json``
        (the surrogate goodness-of-fit — inspect with :meth:`plot_surrogate_quality`).
        Run a multicell UQ sweep first (a Monte-Carlo design,
        ``uq_config['method']=['normal', N]``)."""
        # Deferred: SALib + scikit-learn are heavy, sensitivity-only dependencies.
        from cavsim2d.analysis.uq.sobol_sa import analyse
        uqd = self.cavity.uq_dir
        npath = os.path.join(uqd, 'nodes.csv')
        tpath = os.path.join(uqd, 'table.csv')
        if not (os.path.exists(npath) and os.path.exists(tpath)):
            # Always-visible (not the verbosity-gated info): a user calling this
            # directly must be told why it produced nothing.
            warning("run_sensitivity: needs a multicell UQ sweep first "
                    f"(missing {'nodes.csv' if not os.path.exists(npath) else 'table.csv'} "
                    f"in {uqd}).")
            return None
        X = pd.read_csv(npath, sep='\t')
        Y = pd.read_csv(tpath, sep='\t')
        results, surrogate = analyse(X, Y, objectives=objectives, N=N, degree=degree)
        if save:
            with open(os.path.join(uqd, 'sobol.json'), 'w') as f:
                json.dump(results, f, indent=2)
            with open(os.path.join(uqd, 'surrogate.json'), 'w') as f:
                json.dump(surrogate, f, indent=2)
        return results

    def surrogate_quality(self):
        """The SA surrogate's goodness-of-fit per figure of merit as a DataFrame,
        read from ``uq/surrogate.json`` (written by :meth:`run_sensitivity`).

        Columns: ``r2`` and ``cv_r2`` (in-sample and cross-validated
        coefficient of determination), ``rmse`` and ``max_abs_err`` in the
        FM’s own units, ``y_std`` and ``near_constant`` (a FM with no spread
        has nothing to fit), ``n_samples``, ``n_terms``, their ratio
        ``samples_per_term``, the ridge penalty ``alpha`` the fit settled on,
        and ``undersampled`` (fewer than three samples per free coefficient).
        Read ``cv_r2`` first: it is the honest out-of-sample number, and a FM
        whose ``cv_r2`` sits near zero has no usable Sobol indices however
        good its ``r2`` looks."""
        path = os.path.join(self.cavity.uq_dir, 'surrogate.json')
        if not os.path.exists(path):
            warning(f"surrogate_quality: no surrogate.json in {self.cavity.uq_dir} - "
                    "run run_sensitivity() first (a UQ rerun clears it).")
            return None
        with open(path) as f:
            surro = json.load(f)
        cols = ['r2', 'cv_r2', 'rmse', 'max_abs_err', 'y_std', 'near_constant',
                'n_samples', 'n_terms', 'samples_per_term', 'alpha', 'undersampled']
        return pd.DataFrame(
            {obj: {c: surro[obj].get(c) for c in cols} for obj in surro}).T

    def plot_surrogate_quality(self, objectives=None, ncols=3, show=True):
        r"""Parity plots — the surrogate's **cross-validated** prediction vs the
        true figure of merit, one panel per FM, annotated with $R^2$, the
        cross-validated $R^2$ and the RMSE / max error (in the FM's own units).
        Points on the diagonal mean the polynomial surrogate faithfully represents
        the data; scatter off it flags where a higher degree or more samples are
        needed. Reads ``uq/surrogate.json`` (written by :meth:`run_sensitivity`)."""
        path = os.path.join(self.cavity.uq_dir, 'surrogate.json')
        if not os.path.exists(path):
            warning(f"plot_surrogate_quality: no surrogate.json in {self.cavity.uq_dir} - "
                    "run run_sensitivity() first (a UQ rerun clears it).")
            return None
        with open(path) as f:
            surro = json.load(f)
        fms = list(surro) if objectives is None else \
            [o for o in surro if o in objectives or o.split(':')[-1].strip() in objectives]
        if not fms:
            warning("plot_surrogate_quality: no matching objectives in the surrogate "
                    f"results (have {list(surro)}).")
            return None

        def _bare(c):
            return c.split(':')[-1].strip()
        n = len(fms)
        ncols = min(ncols, n)
        nrows = int(np.ceil(n / ncols))
        with house_style():
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows),
                                     squeeze=False, layout='constrained')
            flat = axes.ravel()
            for ax, fm in zip(flat, fms):
                a = np.asarray(surro[fm]['actual'], float)
                p = np.asarray(surro[fm]['predicted'], float)
                lo, hi = float(min(a.min(), p.min())), float(max(a.max(), p.max()))
                ax.plot([lo, hi], [lo, hi], color='#888888', lw=1, ls='--', zorder=1)
                ax.scatter(a, p, s=18, color=WARM[0], edgecolors='k', lw=0.4, zorder=2)
                # One range on both axes: this is a parity plot, so the dashed
                # line must read as 45 degrees, and letting the axes scale
                # independently gives a near-constant FM two different offsets
                # ("+1.2999e3" against "+1.29997e3") that cannot be compared.
                pad = 0.05 * (hi - lo) or 1e-12
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
                # Few ticks, and room above the axes: a near-constant FM is drawn
                # with a shared offset ("+1.2999e3") that matplotlib parks where
                # the title would otherwise sit, and its full-precision labels
                # run into each other at the default tick density.
                # The SAME tick count on both: the axes already share a range, and
                # matplotlib derives its shared offset from the tick values, so two
                # different counts give a near-constant FM two offsets that look
                # like two different scales.
                ax.xaxis.set_major_locator(MaxNLocator(4))
                ax.yaxis.set_major_locator(MaxNLocator(4))
                ax.set_title(_bare(fm), fontsize=10, pad=14)
                ax.set_xlabel('actual')
                ax.set_ylabel('surrogate (CV)')
                txt = (f"$R^2$={surro[fm]['r2']:.4f}\nCV $R^2$={surro[fm]['cv_r2']:.4f}\n"
                       f"RMSE={surro[fm]['rmse']:.3g}\nmax err={surro[fm]['max_abs_err']:.3g}")
                # A near-constant FM (e.g. a tuned frequency) has no signal to fit —
                # a low/negative R^2 is expected, not a bad surrogate. Say so.
                if surro[fm].get('near_constant'):
                    txt += (f"\nstd={surro[fm].get('y_std', float('nan')):.2g}"
                            "\n(~const: no SA signal)")
                # Only worth saying when the fit is actually poor: a QOI the
                # polynomial represents exactly fits fine on few samples, and the
                # note would be noise on its panel.
                spt = surro[fm].get('samples_per_term')
                if (surro[fm].get('undersampled') and spt
                        and surro[fm]['cv_r2'] < 0.95
                        and not surro[fm].get('near_constant')):
                    txt += f"\n({spt:.1f} samples/term: undersampled)"
                ax.text(0.04, 0.96, txt,
                        transform=ax.transAxes, va='top', ha='left', fontsize=8,
                        bbox=dict(boxstyle='round', fc='white', ec='#333333', alpha=0.9))
            for ax in flat[n:]:
                ax.set_visible(False)
        _maybe_show(show)
        # Return the axes, not the Figure: a bare Figure as a cell's last
        # expression does not render under the ipympl (%matplotlib widget)
        # backend. Every other plot_* returns axes — match them.
        return flat[:n]

    def plot_sobol_indices(self, results=None, objectives=None, sort_by=None,
                           group=None, show=True):
        r"""Grouped bars of Sobol' main (``S_1``) and total (``S_T``) indices per
        input variable, one panel per figure of merit (WEPB015 Fig. 6). Pass the
        ``results`` from :meth:`run_sensitivity`, or omit to read ``uq/sobol.json``.
        The wide gap between a variable's main and total index flags its
        higher-order interactions.

        ``sort_by`` reorders the x-axis:

        - ``None`` (default) — the order the inputs were sampled in, which for a
          multicell interleaves the variables along the cavity
          (``Ri1, Req1, Ri2, …``).
        - ``'variable'`` — by variable stem, so every ``Req`` (``Req1, Req3,
          …``) is drawn before every ``Ri``, as in the paper's Fig. 6. Use this to
          read one variable's profile along the cavity at a glance.
        - ``'S1'`` / ``'ST'`` — descending by that index, which puts the
          influential inputs first.

        ``group=True`` is the old spelling of ``sort_by='variable'``.
        """
        if group is not None and sort_by is None:
            sort_by = 'variable' if group else None
        if results is None:
            path = os.path.join(self.cavity.uq_dir, 'sobol.json')
            if not os.path.exists(path):
                warning(f"plot_sobol_indices: no sobol.json in {self.cavity.uq_dir} - "
                        "run run_sensitivity() first (a UQ rerun clears it).")
                return None
            with open(path) as f:
                results = json.load(f)
        fms = list(results) if objectives is None else \
            [o for o in results if o in objectives or o.split(':')[-1].strip() in objectives]
        if not fms:
            warning("plot_sobol_indices: no matching objectives in the Sobol "
                    f"results (have {list(results)}).")
            return None

        def _bare(c):
            return c.split(':')[-1].strip()

        def _stem_key(v):
            # split a trailing integer: 'Req12' -> ('Req', 12); 'Ri' -> ('Ri', 0)
            i = len(v)
            while i > 0 and v[i - 1].isdigit():
                i -= 1
            return (v[:i], int(v[i:]) if i < len(v) else 0)

        with house_style():
            n = len(fms)
            n_vars = max(len(results[fm]) for fm in fms)
            fig, axes = plt.subplots(n, 1, figsize=(max(6, 0.45 * n_vars + 2),
                                                    2.4 * n), squeeze=False,
                                     layout='constrained')
            for ax, fm in zip(axes[:, 0], fms):
                variables = list(results[fm])
                if sort_by == 'variable':
                    variables = sorted(variables, key=_stem_key)
                elif sort_by in ('S1', 'ST'):
                    variables = sorted(variables,
                                       key=lambda v: -results[fm][v][sort_by])
                elif sort_by is not None:
                    raise ValueError(
                        f"plot_sobol_indices: sort_by={sort_by!r} is not one of "
                        "None, 'variable', 'S1', 'ST'.")
                x = np.arange(len(variables))
                w = 0.4
                s1 = [results[fm][v]['S1'] for v in variables]
                st = [results[fm][v]['ST'] for v in variables]
                ax.bar(x - w / 2, s1, w, color=WARM[0], edgecolor='k',
                       lw=0.4, label=r'main $S_1$')
                ax.bar(x + w / 2, st, w, color=WARM[3 % len(WARM)], edgecolor='k',
                       lw=0.4, label=r'total $S_T$')
                ax.set_xticks(x)
                ax.set_xticklabels(variables, rotation=90, fontsize=7)
                ax.set_ylabel(_bare(fm))
                # Sorting by variable puts each stem's cells in one run; a rule at
                # every change of stem is what makes that run readable as a block.
                if sort_by == 'variable':
                    stems = [_stem_key(v)[0] for v in variables]
                    for j in range(1, len(stems)):
                        if stems[j] != stems[j - 1]:
                            ax.axvline(j - 0.5, color='#888888', lw=0.8, ls=':')
            axes[0, 0].legend(ncol=2, loc='upper right')
        _maybe_show(show)
        # Return the axes, not the Figure: a cell whose last expression is a bare
        # Figure fails to render under the ipympl (%matplotlib widget) backend
        # (it double-claims the canvas plt.show() already emitted). Every other
        # plot_* returns axes, so match them and stay backend-robust.
        return axes[:, 0]

    # -- Fundamental-mode figure-of-merit plots for THIS cavity ---------------
    # The same plots the Study offers (study.eigenmode.plot_fm_*), applied to one
    # cavity: with uq=True this is the single-cavity UQ mean±std + design point.
    def _single_study(self, uq=False):
        """A transient one-cavity Study so the fundamental-mode comparison plots
        (which iterate a study's cavities) work for a single cavity."""
        # Deferred: breaks the solver_objects <-> study import cycle.
        from cavsim2d.study import Study
        cav = self.cavity
        # The cavity's OWN folder, never '.': the study this builds is handed to
        # the comparison plots, which save into <projectDir>/PostProcessingData/,
        # and a '.' here writes that tree into whatever directory the interpreter
        # was started in.
        s = Study(cav._ensure_workspace(), _skip_project_init=True)
        s.cavities_list = [cav]
        s.cavities_dict = {cav.name: cav}
        s.eigenmode_qois = {cav.name: getattr(cav, 'eigenmode_qois', {}) or {}}
        if uq:
            cav.get_uq_fm_results(cav.uq_dir)
            s.uq_fm_results = {cav.name: cav.uq_fm_results}
        return s

    def plot_fm_scatter(self, ncols=3, uq=False, qois=None, show=True):
        """Scatter this cavity's fundamental-mode QOIs. With ``uq=True`` each QOI
        is a mean with a ±1σ error bar and the nominal design point overlaid."""
        return self._single_study(uq=uq).eigenmode.plot_fm_scatter(
            ncols=ncols, uq=uq, qois=qois, show=show)

    def plot_fm_bar(self, ncols=3, uq=False, qois=None, show=True):
        """Bar chart of this cavity's fundamental-mode QOIs (optionally UQ)."""
        return self._single_study(uq=uq).eigenmode.plot_fm_bar(
            ncols=ncols, uq=uq, qois=qois, show=show)
