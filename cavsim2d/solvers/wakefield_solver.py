"""The wakefield namespace, ``cav.wakefield``."""
import json
from pathlib import Path
from cavsim2d.utils.printing import info, warning
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.utils.integrity import guard_runs, notice_tuned_versions, warn_if_stale
from cavsim2d.analysis.impedance import convert_impedance_frame, frame_unit, impedance_unit
from cavsim2d.utils.style import house_style, WARM
from cavsim2d.processes.wakefield import run_wakefield_parallel, run_wakefield_s
from cavsim2d.solvers.wakefield import WakefieldBackend, WakefieldResult, get_backend
from cavsim2d.solvers._shared import _maybe_show, _require_vacuum, _z_axis_label
from cavsim2d.solvers.configs import (_ConfigSampleMixin, _user_config,
                                      DEFAULT_WAKEFIELD_CONFIG, merge_config)


# ---------------------------------------------------------------------------
# WakefieldSolver
# ---------------------------------------------------------------------------

class WakefieldSolver(_ConfigSampleMixin):
    """Manages ``<cavity>/wakefield/`` — wakefield / impedance analysis.

    Access
    ------
    - ``cav.wakefield.config``
    - ``cav.wakefield.config_sample`` — the default config, to edit and pass to run()
    - ``cav.wakefield.qois``     — loss and kick factors of the main run
    - ``cav.wakefield.wake_z``   — longitudinal wake DataFrame
    - ``cav.wakefield.wake_t``   — transverse wake DataFrame
    - ``cav.wakefield.impedance(kind, unit=...)`` — impedance frame
    - ``cav.wakefield.plot_impedance()`` / ``plot_wake()``
    - ``cav.wakefield.plot_k_loss()`` / ``plot_k_kick()``
    """

    _config_kind = 'wakefield'

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._result = None
        self._qois = None

    @property
    def folder(self):
        return Path(self.cavity.self_dir) / 'wakefield'

    def _has_results_folder(self):
        """True when the cavity has a workspace; warns when its wakefield results
        belong to another geometry."""
        if not self.cavity.self_dir:
            return False
        warn_if_stale(self.cavity, self.folder, 'The wakefield results')
        return True

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
        if not self._has_results_folder():
            return WakefieldResult()
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
        if not self._has_results_folder():
            return {}
        if self._qois is None:
            self._qois = self.backend.read_qois(self.cavity)
        return self._qois

    @property
    def qois_op(self):
        """The operating-point QOIs (per operating point / bunch length: P_HOM,
        loss/kick, …), produced only when ``operating_points`` were given. Empty
        dict otherwise. Distinct from :attr:`qois`, the main-run factors."""
        if not self._has_results_folder():
            return {}
        # The process layer writes this table, whatever backend ran, so the base
        # reader serves every backend (a custom one need not implement it).
        return WakefieldBackend.read_qois_op(self.cavity)

    # -- Actions ------------------------------------------------------------

    def run(self, wakefield_config=None, **kwargs):
        """Run wakefield analysis. Delegates to processes/wakefield.py.

        The user's config is merged over :data:`DEFAULT_WAKEFIELD_CONFIG` and the
        **merged** dict is what runs and what is saved to ``config.json`` — the
        saved config always records every setting the run used. Behaviour is
        unchanged: the defaults mirror the effective ones. (Known quirk, not
        changed here: the live ABCI deck writer reads ``MT`` from
        ``wake_config['MT']`` — a top-level ``MT`` is recorded but only
        the legacy writer consumed them.) Any config key can also be passed as
        a keyword argument (``run(wakelength=80)``); kwargs override the
        config dict.

        Returns this namespace, so calls chain: ``cav.wakefield.run(cfg).qois``.
        """
        _require_vacuum(self.cavity, 'wakefield')
        merged_config = merge_config(DEFAULT_WAKEFIELD_CONFIG,
                                     _user_config('wakefield', wakefield_config, kwargs))

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed

        notice_tuned_versions([self.cavity], 'wakefield')
        if not guard_runs([self.cavity], 'wakefield', merged_config, title='Wakefield'):
            return self
        # The run writes config.json beside the results it produces; with
        # rerun=False and results of this geometry already there, it keeps them
        # and the config that made them.
        self._config = None

        wakefield_config_copy = merged_config.copy()
        wakefield_config_copy['target'] = run_wakefield_s

        cavs_dict = {self.cavity.name: self.cavity}
        run_wakefield_parallel(cavs_dict, wakefield_config_copy)

        self._invalidate()
        # Keep the cavity's operating-point attributes (cav.k_loss, ...) in step.
        try:
            self.cavity.get_wakefield_qois(merged_config)
        except FileNotFoundError:
            pass
        return self

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
            warning("No wakefield data available — run cav.wakefield.run(...) first.")
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
        """Plot the wakefield impedance: ``|Z|`` vs f.

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
        study comparison bars, e.g. ``study.wakefield.plot_hom_bar``, show)."""
        return self._plot_kspectrum('longitudinal', ax, show=show, **kwargs)

    def plot_k_kick(self, ax=None, show=True, **kwargs):
        """Plot the cumulative kick-factor spectrum :math:`k_{\\mathrm{kick}}(F)`
        vs frequency (the transverse analogue of :meth:`plot_k_loss`)."""
        return self._plot_kspectrum('transverse', ax, show=show, **kwargs)

    def animate_fields(self, save=None, fps=4, embed='auto', inline_format='auto',
                       progress=True):
        """Animate ABCI's electric field-line snapshots through the cavity.

        Needs the wakefield to have been run with ``save_fields=True`` (or
        ``{'nshot': N}``), which turns on ABCI's field-line plotting (``LPLE``) so
        each snapshot is written to ``wakefield/longitudinal/cavity.top``::

            cav.wakefield.run({'MROT': 0, 'wakelength': 20, 'save_fields': True})
            cav.wakefield.animate_fields()

        Like :meth:`MultipactingSolver.animate_trajectories`, the inline player is a
        compact H.264 ``<video>`` when ffmpeg is available (``inline_format``), so
        the whole sequence embeds; ``save='fields.mp4'``/``'.gif'`` writes a file.
        Returns the :class:`~matplotlib.animation.FuncAnimation`."""
        # Deferred: field-line parser + animation writers only needed here.
        from cavsim2d.utils.data_utils import get_wakefield_data
        from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
        top = self.folder / 'longitudinal' / 'cavity.top'
        if not top.exists():
            warning("No wakefield output yet — run cav.wakefield.run(...) first.")
            return None
        frames = get_wakefield_data(str(top))
        if not frames:
            warning("No electric-field-line snapshots in cavity.top. Re-run the "
                    "wakefield with save_fields=True, e.g. "
                    "cav.wakefield.run({..., 'save_fields': True}).")
            return None
        keys = sorted(frames, key=lambda k: int(k.split('_')[1]))
        with house_style():
            fig, ax = plt.subplots(figsize=(9, 4))

            def draw(k):
                ax.clear()
                for _kind, df in frames[k]:
                    ax.plot(df['X'], df['Y'], '-', lw=0.8, color=WARM[0])
                    ax.plot(df['X'], -df['Y'], '-', lw=0.8, color=WARM[0])
                ax.set_xlabel('z [m]')
                ax.set_ylabel('r [m]')
                ax.set_aspect('equal', 'box')
                ax.set_title(f'{self.cavity.name} — E-field lines ({k})')

            draw(keys[0])
            anim = FuncAnimation(fig, draw, frames=keys, interval=1000 / fps,
                                 blit=False)
        n = len(keys)
        if save:
            writer = (PillowWriter(fps=fps) if str(save).lower().endswith('.gif')
                      else FFMpegWriter(fps=fps))
            anim.save(save, writer=writer, dpi=120)
            info(f"Field-line animation saved to {save} ({n} frames).")

        # Deferred: notebook/animation APIs (also reuses the multipacting helpers).
        from cavsim2d.analysis.multipacting.plots import (_is_notebook,
                                                          _to_html5_video_with_progress)
        want_embed = (_is_notebook() and not save) if embed == 'auto' else bool(embed)
        if want_embed:
            # Deferred: IPython display is a notebook-only API.
            from IPython.display import HTML, display
            use_video = (FFMpegWriter.isAvailable() if inline_format == 'auto'
                         else inline_format == 'video')
            html = (_to_html5_video_with_progress(anim, fps, n, 120, progress)
                    if use_video else anim.to_jshtml(fps=fps, embed_frames=True))
            plt.close(fig)
            display(HTML(html))
        return anim

    def _plot_kspectrum(self, kind, ax, show=True, **kwargs):
        transverse = str(kind).lower().startswith('trans')
        stem = 'k_kick(f)' if transverse else 'k_loss(f)'
        df = self._frame(kind)
        kcol = next((c for c in (df.columns if df is not None else [])
                     if c.startswith(stem)), None)
        if df is None or df.empty or kcol is None or 'fk [MHz]' not in df.columns:
            warning("No loss-factor spectrum available — run cav.wakefield.run(...) "
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
            warning("No wakefield data available — run cav.wakefield.run(...) first.")
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
