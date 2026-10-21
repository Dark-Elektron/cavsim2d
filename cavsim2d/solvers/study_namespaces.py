"""The study-level namespaces, ``study.eigenmode`` and ``study.wakefield``."""
import warnings
from cavsim2d.utils.printing import CavsimWarning, error, info
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.analysis.impedance import impedance_unit
from cavsim2d.utils.style import house_style
from cavsim2d.solvers._shared import _maybe_show, _MODE_FILTER_CAVEAT, _z_axis_label
from cavsim2d.solvers.configs import _ConfigSampleMixin


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


class StudyEigenmode(_ConfigSampleMixin, _StudyNamespace):
    """``study.eigenmode`` — eigenmode results across every cavity in the study."""

    _config_kind = 'eigenmode'

    def run(self, eigenmode_config=None, **kwargs):
        """Run the eigenmode solve for every cavity (same as ``study.run_eigenmode``);
        returns this namespace, so ``study.eigenmode.run(cfg).qois_df`` works."""
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
            drawn = 0
            for cav, color in zip(self.cavities, self._colors()):
                df = cav.eigenmode._impedance_frame(kind=kind, span=span,
                                                    n_points=n_points, Q=Q, unit=unit)
                if df.empty:
                    continue
                drawn += 1
                ax.plot(df['f [MHz]'], df[f'|Z| [{u}]'], color=color,
                        label=cav.name, **kwargs)
            if drawn:
                # Once for the study, not once per cavity.
                warnings.warn(
                    "Each curve sums every computed mode of its cavity, with no "
                    "beam-pipe/PML mode filter, so above the beam-pipe cutoff it "
                    "includes modes of the pipe and the absorbing layer. Filter a "
                    "cavity with cav.eigenmode.stable_modes() and plot it with "
                    "cav.eigenmode.plot_impedance(modes=...). " + _MODE_FILTER_CAVEAT,
                    CavsimWarning, stacklevel=2)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(_z_axis_label(unit, transverse))
            ax.set_yscale('log')
            ax.legend()
        _maybe_show(show)
        return ax

    # -- Fundamental-mode comparison plots (were Study.plot_compare_fm_*) -----
    def plot_fm_scatter(self, ncols=3, uq=False, qois=None, show=True):
        """Scatter each cavity's fundamental-mode QOIs (optionally UQ mean±std).

        With ``uq=True`` each QOI is a mean with a ±1σ error bar and the nominal
        design point overlaid — the built-in UQ figure-of-merit comparison.
        """
        axd = self.study._fm_scatter_impl(ncols=ncols, uq=uq, qois=qois)
        _maybe_show(show)
        return axd

    def plot_fm_bar(self, ncols=3, uq=False, qois=None, show=True):
        """Bar chart of each cavity's fundamental-mode QOIs (optionally UQ)."""
        axd = self.study._fm_bar_impl(ncols=ncols, uq=uq, qois=qois)
        _maybe_show(show)
        return axd

    def plot_compare(self, kind='scatter', uq=False, ncols=3, qois=None, show=True):
        """Dispatch to :meth:`plot_fm_scatter` (``kind='scatter'``) or
        :meth:`plot_fm_bar` (``kind='bar'``)."""
        r = self.study._eigenmode_compare_impl(kind=kind, uq=uq, ncols=ncols, qois=qois)
        _maybe_show(show)
        return r


class StudyWakefield(_ConfigSampleMixin, _StudyNamespace):
    """``study.wakefield`` — wakefield results across every cavity in the study."""

    _config_kind = 'wakefield'

    def run(self, wakefield_config=None, **kwargs):
        """Run the wakefield solve for every cavity (same as ``study.run_wakefield``);
        returns this namespace."""
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
        bars, e.g. ``study.wakefield.plot_hom_bar``, show instead.)"""
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

    # -- HOM / power comparison plots (were Study.plot_compare_{hom,power}_*) --
    def plot_hom_scatter(self, op_points_list=None, ncols=3, uq=False, figsize=(12, 3),
                         qois=None, show=True):
        """Scatter each cavity's higher-order-mode QOIs (optionally UQ mean±std).

        ``op_points_list`` names the operating points for the per-operating-point
        QOIs (``k_loss``, ``k_kick``, ``p_hom``). Omit it with ``uq=True`` to plot
        a wakefield UQ run whose objectives are impedance windows instead
        (``ZL``/``ZT`` peaks): one panel per window, each cavity's mean with its
        ±1σ error bar."""
        axd = self.study._hom_scatter_impl(op_points_list, ncols=ncols, uq=uq,
                                           figsize=figsize, qois=qois)
        _maybe_show(show)
        return axd

    def plot_hom_bar(self, op_points_list=None, ncols=3, uq=False, figsize=(12, 3),
                     qois=None, show=True):
        """Bar chart of each cavity's higher-order-mode QOIs (optionally UQ).

        ``op_points_list`` may be omitted with ``uq=True`` for impedance-window
        objectives — see :meth:`plot_hom_scatter`."""
        axd = self.study._hom_bar_impl(op_points_list, ncols=ncols, uq=uq,
                                       figsize=figsize, qois=qois)
        _maybe_show(show)
        return axd

    def plot_power_scatter(self, rf_config, op_points_list, ncols=3, uq=False,
                           figsize=(12, 3), qois=None, show=True):
        """Scatter each cavity's RF-power QOIs across operating points."""
        axd = self.study._power_scatter_impl(rf_config, op_points_list, ncols=ncols,
                                             uq=uq, figsize=figsize, qois=qois)
        _maybe_show(show)
        return axd

    def plot_power_bar(self, show=True):
        """Bar chart of each cavity's normalised RF-power QOIs."""
        r = self.study._power_bar_impl()
        _maybe_show(show)
        return r

    def plot_compare(self, opt, kind='scatter', uq=False, ncols=3, figsize=(12, 3),
                     qois=None, show=True):
        """Dispatch to :meth:`plot_hom_scatter` (``kind='scatter'``) or
        :meth:`plot_hom_bar` (``kind='bar'``)."""
        r = self.study._wakefield_compare_impl(opt, kind=kind, uq=uq, ncols=ncols,
                                               figsize=figsize, qois=qois)
        _maybe_show(show)
        return r
