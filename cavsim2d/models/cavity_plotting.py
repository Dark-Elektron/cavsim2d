"""Plots and interactive views of a cavity: geometry, mesh, fields, dispersion,
spectra, convergence.

:class:`~cavsim2d.models.base.Cavity` inherits these from :class:`CavityPlotting`.
"""
from cavsim2d.constants import c0
from cavsim2d.geometry.plotting import write_cavity_geometry_cli_wo_gmsh
from cavsim2d.geometry.writers.gmsh import (write_cavity_geometry_cli,
                                            write_cavity_geometry_cli_flattop)
from cavsim2d.processes.uq import ngsolve_mevp
from cavsim2d.utils.data_utils import get_wakefield_data
from cavsim2d.utils.printing import error, info, warning
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from cavsim2d.solvers.eigenmode_result import pol_name, pol_number
from cavsim2d.solvers.solver_objects import _maybe_show
from cavsim2d.utils.integrity import warn_if_stale
from cavsim2d.utils.style import house_style, polarisation_color, shades, WARM
from cavsim2d.utils.slope_triangles import add_slope_triangles
from fractions import Fraction


class CavityPlotting:
    """Plots and interactive views of a cavity and its results."""

    def plot_mesh_convergence(self, qoi='freq [MHz]', mode_index='0-0', reference=None,
                              slopes=False, floor=5e-14, ax=None, show=True):
        """The relative error of *qoi* against the number of DOFs, one curve per polynomial
        order, from :meth:`study_mesh_convergence`.

        *mode_index* is ``'<m>-<mode>'`` (``'0-0'``: the monopole fundamental). *reference*
        is the exact value; ``None`` compares against the solve with the most DOFs, which
        is then left out of the plot.

        ``slopes=True`` marks each order's convergence rate with a slope triangle. A
        triangle is drawn only where the curve follows a power law (see
        :func:`~cavsim2d.utils.slope_triangles.fit_convergence_rate`), so a figure of merit
        whose error changes sign or size irregularly from mesh to mesh gets none rather
        than a meaningless slope. Errors at or below *floor* are round-off: they stay in
        the plot and are left out of the rates. Returns ``(fig, ax)``.
        """
        df = self.convergence_df_data
        if df is None or len(df) == 0:
            raise ValueError("no mesh-convergence results: run study_mesh_convergence() first.")
        sub = df[df['mode_index'] == mode_index]
        if sub.empty:
            raise ValueError(f"mode_index {mode_index!r} is not in the results; available: "
                             f"{sorted(df['mode_index'].unique())}")
        if qoi not in sub:
            raise ValueError(f"{qoi!r} is not in the results; available: "
                             f"{[c for c in sub.columns if '[' in c]}")
        if reference is None:
            finest = sub.loc[sub['No of DOFs'].idxmax()]
            ref, against = float(finest[qoi]), 'the finest solve'
            sub = sub.drop(index=finest.name)
        else:
            ref, against = float(reference), 'the reference'
        sub = sub.assign(error=((sub[qoi] - ref).abs() / abs(ref)).clip(lower=1e-16))

        with house_style():
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 5.2))
            else:
                fig = ax.figure
            curves, colors = [], []
            for i, (p, g) in enumerate(sub.groupby('p')):
                g = g.sort_values('No of DOFs')
                color = WARM[i % len(WARM)]
                ax.plot(g['No of DOFs'], g['error'], 'o-', ms=5, color=color, label=f'p = {p}')
                curves.append((g['No of DOFs'].values, g['error'].values))
                colors.append(color)
            ax.set_xscale('log')
            ax.set_yscale('log')
            if (sub['error'] <= 10 * floor).any():
                ax.axhline(floor, color='0.5', lw=0.8, ls=':')
                ax.text(1.0, floor * 1.4, 'round-off ', transform=ax.get_yaxis_transform(),
                        ha='right', va='bottom', fontsize=8, color='0.4')
            if slopes:
                add_slope_triangles(ax, curves, colors, floor=floor)
            ax.set_xlabel('degrees of freedom')
            ax.set_ylabel(f'relative error in {qoi}')
            ax.set_title(f'{self.name}: mode {mode_index} against {against}')
            ax.legend()
        _maybe_show(show)
        return fig, ax

    @staticmethod
    def _phase_advance_label(j, n):
        """Tick label for a phase advance of ``j*pi/n`` as a reduced fraction of
        pi, rendered with mathtext (no LaTeX install needed): ``0``, ``pi/2``,
        ``2pi/3``, ``pi``.
        """
        if j == 0:
            return r'$0$'
        f = Fraction(j, n)
        if f == 1:
            return r'$\pi$'
        num = '' if f.numerator == 1 else str(f.numerator)
        return rf'$\dfrac{{{num}\pi}}{{{f.denominator}}}$'

    @staticmethod
    def _phase_advances(n):
        """The per-cell phase advances of an ``n``-cell fundamental passband:
        ``mu_q = q*pi/(n-1)`` for ``q = 0..n-1`` — the standing-wave modes from the
        0-mode (``mu = 0``) to the pi-mode (``mu = pi``). A single cell has only
        the pi-mode. Returns ``(mu array, tick-label list)``.

        This is the textbook convention for a well-ordered passband; it assigns
        each mode by frequency order (the accelerating pi-mode at the high end).
        The fully general value would come from each mode's on-axis field node
        count, which is monopole-specific.
        """
        if n <= 1:
            return np.array([np.pi]), [r'$\pi$']
        mu = np.linspace(0.0, np.pi, n)
        labels = [CavityPlotting._phase_advance_label(q, n - 1) for q in range(n)]
        return mu, labels

    @staticmethod
    def _resolve_bands(bands, pols):
        """Normalise the ``bands`` argument to one 1-based list per polarisation.

        - ``None`` -> ``None`` for every polarisation (all computed passbands).
        - a flat list like ``[1, 2]`` -> the same selection for every polarisation.
        - a nested list like ``[[1, 2], [1]]`` -> one selection per polarisation,
          and its length must match ``pols``.
        """
        if bands is None:
            return [None] * len(pols)
        if not isinstance(bands, (list, tuple)):
            raise TypeError("'bands' must be a list of passband numbers, or a "
                            "list of such lists (one per polarisation).")
        nested = [b for b in bands if isinstance(b, (list, tuple))]
        if nested:
            if len(nested) != len(bands):
                raise ValueError("'bands' mixes plain numbers and per-polarisation "
                                 "lists; use either [1, 2] or [[1, 2], [1]].")
            if len(bands) != len(pols):
                raise ValueError(f"'bands' has {len(bands)} per-polarisation lists "
                                 f"but there are {len(pols)} polarisation(s).")
            return [list(b) for b in bands]
        return [list(bands)] * len(pols)          # one flat list, shared

    def _cell_length_m(self):
        """Axial length of one (mid-)cell in metres, for the dispersion light
        line. Returns None when the period is not defined for this geometry, in
        which case the light line is skipped. Overridden by the cavity types
        that have a periodic cell."""
        return None

    def _dispersion_series(self, pol=None, bands=None, color=None, label=None):
        """Collect the dispersion curves as a list of ``dict(x, y, color, label)``,
        one per (polarisation, passband). Shared by the single-cavity and the
        ``Study`` overlay plotters so both render identically."""
        if pol is None:
            pols = self._available_polarisations() or ['monopole']
        elif isinstance(pol, (str, int)):
            pols = [pol]
        else:
            pols = list(pol)

        band_selection = self._resolve_bands(bands, pols)
        n = self.n_cells
        series = []

        for pol_i, want_bands in zip(pols, band_selection):
            all_modes = self.eigenmode.mpole_qois(pol_i)   # {idx: qois}
            df = pd.DataFrame.from_dict(all_modes, orient='index')
            if df.empty or 'freq [MHz]' not in df:
                info(f"No {pol_name(pol_number(pol_i))} passband data — run "
                     f"eigenmode with that polarisation first.")
                continue
            df = df.loc[sorted(df.index, key=lambda s: int(s))]
            freqs = df['freq [MHz]'].to_numpy()

            n_passbands = int(np.ceil(len(freqs) / n))
            hue = color or polarisation_color(pol_i)
            # colour by absolute band index, so passband 2 keeps its shade
            # whether or not passband 1 is also drawn
            band_colors = shades(hue, n_passbands)
            pretty = pol_name(pol_number(pol_i))
            chosen = range(1, n_passbands + 1) if want_bands is None else want_bands

            for b in chosen:
                if not isinstance(b, int) or isinstance(b, bool) or b < 1:
                    raise ValueError(f"passband numbers are 1-based integers; got {b!r}.")
                if b > n_passbands:
                    warning(f"{pretty}: passband {b} was not computed "
                         f"(only {n_passbands} available).")
                    continue
                seg = freqs[(b - 1) * n:b * n]
                k = self._phase_advances(n)[0][:len(seg)]
                if label is not None:
                    lbl = label
                elif n_passbands == 1:
                    lbl = f'{self.name} ({pretty})'
                else:
                    lbl = f'{self.name} ({pretty}, passband {b})'
                series.append({'x': k, 'y': seg, 'color': band_colors[b - 1], 'label': lbl})
        return series, n

    @staticmethod
    def _frequency_clusters(values, gap_factor=4.0, min_rel_gap=0.12):
        """Group *values* into clusters split at large vertical gaps.

        A gap becomes a break only when it is both several times the typical
        within-cluster spacing (``gap_factor``) and a real fraction of the total
        span (``min_rel_gap``) — so a broken axis appears only when passbands are
        genuinely far apart, not for ordinary spread.
        Returns a list of ``(low, high)`` ranges, ascending.
        """
        vals = np.unique(np.asarray(values, dtype=float))
        if len(vals) < 2:
            v = float(vals[0]) if len(vals) else 0.0
            return [(v, v)]
        diffs = np.diff(vals)
        span = vals[-1] - vals[0]
        typical = float(np.median(diffs)) or float(diffs.min())
        clusters, start, prev = [], vals[0], vals[0]
        for v, d in zip(vals[1:], diffs):
            if d > gap_factor * typical and d > min_rel_gap * span:
                clusters.append((float(start), float(prev)))
                start = v
            prev = v
        clusters.append((float(start), float(prev)))
        return clusters

    @staticmethod
    def _clusters_from_breaks(values, breaks):
        """Cluster *values* at explicit gap ranges. Each break is a ``(low, high)``
        frequency gap to cut, e.g. ``[(1350, 1600), (1900, 2350)]``; the split
        falls between the two data points that straddle the gap's midpoint, so the
        gap bounds only need to be approximate. Returns ``(low, high)`` ranges."""
        vals = np.unique(np.asarray(values, dtype=float))
        if len(vals) < 2:
            v = float(vals[0]) if len(vals) else 0.0
            return [(v, v)]
        mids = []
        for b in breaks:
            if not (isinstance(b, (list, tuple)) and len(b) == 2):
                raise ValueError("each entry in 'breaks' must be a (low, high) "
                                 f"frequency gap, e.g. (1350, 1600); got {b!r}.")
            lo, hi = sorted(float(x) for x in b)
            mids.append(0.5 * (lo + hi))
        mids.sort()
        clusters, start, prev = [], vals[0], vals[0]
        for v in vals[1:]:
            if any(prev < m <= v for m in mids):
                clusters.append((float(start), float(prev)))
                start = v
            prev = v
        clusters.append((float(start), float(prev)))
        return clusters

    @staticmethod
    def _render_dispersion(series, n, ax=None, break_axis=True, breaks=None,
                           light_lines=None, **plot_kw):
        """Draw collected dispersion *series*, optionally on a broken y-axis.

        Breaks are chosen automatically from the frequency grouping unless
        *breaks* gives explicit ``(low, high)`` gap ranges. *light_lines* is a
        list of ``dict(d=cell_length_m, label=...)`` speed-of-light reference
        lines. Returns the (top) axis; when broken, the stacked sub-axes are
        reachable via ``ax.figure.axes``.
        """
        with house_style():
            if not series:
                return ax
            yvals = np.concatenate([s['y'] for s in series])
            ticks, ticklabels = CavityPlotting._phase_advances(n)

            def draw(a):
                for s in series:
                    a.plot(s['x'], s['y'], marker='o', mec='k', mew=0.6,
                           color=s['color'], label=s['label'], **plot_kw)

            def draw_light(a, add_label):
                # Light line f = c*phi / (2*pi*d) (phase velocity = c). The diagram
                # is a reduced (folded) zone, so every band is folded into
                # mu in [0, pi] and the light line becomes a triangle wave: for
                # band m the extended-zone phase is (m-1)*pi + mu (odd m, rising)
                # or m*pi - mu (even m, reflected). One segment per band covers the
                # plotted frequencies; drawn on every panel and clipped by y-limits.
                mu = np.linspace(ticks[0], np.pi, 80)
                for ll in (light_lines or []):
                    d = ll['d']
                    step = c0 / (2 * d) / 1e6                 # MHz gained per band
                    n_bands = max(1, int(np.ceil(yvals.max() / step)))
                    labelled = False
                    for m in range(1, n_bands + 1):
                        phi = (m - 1) * np.pi + mu if m % 2 else m * np.pi - mu
                        f = c0 * phi / (2 * np.pi * d) / 1e6
                        lab = ll['label'] if (add_label and not labelled) else '_nolegend_'
                        a.plot(mu, f, ls='--', lw=1.3, color='0.35', zorder=1, label=lab)
                        labelled = True

            def draw_single(a):
                draw(a)
                # clip to the data band before the light line, so its full
                # triangle wave does not stretch the y-axis
                pad = 0.05 * (float(yvals.max() - yvals.min()) or 1.0)
                a.set_ylim(yvals.min() - pad, yvals.max() + pad)
                draw_light(a, add_label=True)
                a.set_xticks(ticks)
                a.set_xticklabels(ticklabels)
                a.set_xlabel('phase advance per cell')
                a.set_ylabel(r'$f$ [MHz]')
                a.legend()
                return a

            if not break_axis:
                clusters = [(float(yvals.min()), float(yvals.max()))]
            elif breaks is not None:
                clusters = CavityPlotting._clusters_from_breaks(yvals, breaks)
            else:
                clusters = CavityPlotting._frequency_clusters(yvals)

            # A single cluster needs no break.
            if len(clusters) <= 1:
                if ax is None:
                    _, ax = plt.subplots()
                return draw_single(ax)

            # Broken axis: highest-frequency cluster on top. Each sub-axis gets
            # height proportional to its own data span, so the empty gaps vanish.
            clusters_desc = clusters[::-1]
            spans = [hi - lo for lo, hi in clusters_desc]
            max_span = max(spans) or 1.0
            ratios = [max(s, 0.25 * max_span) for s in spans]
            global_range = float(yvals.max() - yvals.min()) or 1.0
            k = len(clusters_desc)

            # Build the stack of sub-axes. When the caller passed an axis we
            # subdivide *its* grid slot in place (so it works inside a mosaic or
            # a subplot grid); a free-floating axis with no slot can't be split,
            # so fall back to drawing it on one axis.
            owns_figure = ax is None
            if owns_figure:
                fig = plt.figure(figsize=(7, 6))
                axes = list(np.atleast_1d(fig.subplots(
                    k, 1, sharex=True,
                    gridspec_kw={'height_ratios': ratios, 'hspace': 0.07})))
            else:
                spec = ax.get_subplotspec()
                if spec is None:
                    return draw_single(ax)
                fig = ax.figure
                ax.remove()
                subspec = spec.subgridspec(k, 1, height_ratios=ratios, hspace=0.07)
                axes = [fig.add_subplot(subspec[i]) for i in range(k)]
                for a in axes[1:]:
                    a.sharex(axes[0])

            for a in axes:
                draw(a)
                a.grid(True)
            # light line on every panel (clipped to each band); label it once
            for i, a in enumerate(axes):
                draw_light(a, add_label=(i == 0))
            for a, (lo, hi) in zip(axes, clusters_desc):
                pad = max(0.12 * (hi - lo), 0.02 * global_range)
                a.set_ylim(lo - pad, hi + pad)

            # diagonal break marks; hide the spines that face the gap
            mark = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=9, linestyle='none',
                        color='#333333', mec='#333333', mew=1, clip_on=False)
            for upper, lower in zip(axes[:-1], axes[1:]):
                upper.spines['bottom'].set_visible(False)
                lower.spines['top'].set_visible(False)
                upper.tick_params(labelbottom=False, bottom=False)
                upper.plot([0, 1], [0, 0], transform=upper.transAxes, **mark)
                lower.plot([0, 1], [1, 1], transform=lower.transAxes, **mark)

            axes[-1].set_xticks(ticks)
            axes[-1].set_xticklabels(ticklabels)
            axes[-1].set_xlabel('phase advance per cell')
            axes[0].legend()
            # y-label: supylabel centres it on the whole figure (fine when we own
            # it); inside someone's mosaic, label the middle panel instead so we
            # don't relabel their other axes.
            if owns_figure:
                fig.supylabel(r'$f$ [MHz]')
            else:
                axes[k // 2].set_ylabel(r'$f$ [MHz]')
            return axes[0]

    def plot_dispersion(self, ax=None, pol=None, bands=None, break_axis=True,
                        breaks=None, light_line=True, **kwargs):
        """Plot the Brillouin (dispersion) diagram: frequency vs per-cell phase
        advance.

        ``pol`` selects the polarisation(s):

        - ``None`` (default) overlays every polarisation that has results
          (monopole, dipole, …);
        - a name or azimuthal number ('dipole', 1) plots just that one;
        - a list plots the given subset.

        ``bands`` selects which passbands to show (1-based; passband 1 is the
        fundamental). By default every computed passband is drawn.

        - a flat list, e.g. ``bands=[1, 2]``, applies to every polarisation;
        - a nested list, e.g. ``bands=[[1, 2], [1]]`` with
          ``pol=['monopole', 'dipole']``, selects the first two passbands for the
          monopole and only the first for the dipole (one inner list per
          polarisation, in order).

        ``break_axis`` (default ``True``) splits the y-axis where passbands are
        far apart, so the empty space between them is not wasted. The breaks are
        found automatically from the frequency grouping; pass ``breaks`` to place
        them by hand as ``(low, high)`` gap ranges, e.g.
        ``breaks=[(1350, 1600), (1900, 2350)]`` (the bounds are approximate — the
        split falls between the data points straddling each gap). Set
        ``break_axis=False`` for a single axis. Breaking needs this call to own
        the figure (``ax=None``); pass an ``ax`` to draw on one existing axis.

        ``light_line`` (default ``True``) overlays the speed-of-light line
        ``f = c*mu / (2*pi*d)`` (``d`` the cell length), the synchronism line the
        accelerating pi-mode sits on. It is skipped for geometries whose cell
        period is undefined.

        Each polarisation gets a hue from the house palette; its passbands share
        the hue and shade, so a whole polarisation reads as one colour family.
        """
        color = kwargs.pop('color', None)
        label = kwargs.pop('label', None)
        series, n = self._dispersion_series(pol=pol, bands=bands, color=color, label=label)
        light_lines = None
        if light_line:
            d = self._cell_length_m()
            if d:
                light_lines = [{'d': d, 'label': 'light line'}]
            else:
                info(f"No light line: cell period is undefined for "
                     f"{type(self).__name__}.")
        return self._render_dispersion(series, n, ax=ax, break_axis=break_axis,
                                       breaks=breaks, light_lines=light_lines, **kwargs)

    def axis_field_profile(self, mode, pol='monopole', n=801, normalise=True):
        """Signed on-axis accelerating field of one mode: ``(z_mm, Ez)``.

        The cached ``Ez_0_abs.csv`` holds ``|Ez|`` for the *mode of interest*
        only, which cannot show whether two parts of a structure are in phase.
        This samples the saved field of any solved mode, keeping the sign, so a
        multi-cavity line can be read directly: where the lobes sit tells you
        which cavity a mode lives in, and their relative sign tells you how the
        cavities are phased.

        *mode* is the 0-based index within the polarisation, matching
        ``eigenmode.qois_df``. ``normalise`` scales to a peak of 1 so modes can
        be overlaid.
        """
        gfu_E, _ = self.get_fields(mode, pol)
        field = gfu_E[mode].components[0]          # in-plane (E_z, E_r)
        mesh = gfu_E[mode].space.mesh
        xs = np.array([v.point[0] for v in mesh.vertices], dtype=float)
        # Just off the axis: r = 0 is the domain boundary, and a point exactly on
        # it can fall outside the element the evaluator picks.
        r_probe = 1e-4
        z = np.linspace(xs.min(), xs.max(), int(n))
        # Complex: a lossy solve (a beam line absorber, say) returns complex
        # fields, and a float buffer would reject them.
        vals = np.full(z.shape, np.nan, dtype=complex)
        for i, zi in enumerate(z):
            try:
                point = mesh(float(zi), r_probe)
            except Exception:
                continue                            # outside the domain
            vals[i] = complex(field(point)[0])

        good = np.isfinite(vals)
        if not good.any():
            raise RuntimeError(
                f"could not sample the axis field of mode {mode} for "
                f"{self.name!r}: every probe point fell outside the meshed "
                "domain. The saved mesh spans "
                f"z = {xs.min() * 1e3:.1f} to {xs.max() * 1e3:.1f} mm.")
        # A complex mode carries an arbitrary global phase, so the raw real part
        # can come out near zero everywhere. Rotate the largest sample onto the
        # real axis first; the remaining sign is then the physical one.
        k = int(np.argmax(np.where(good, np.abs(vals), -1.0)))
        if abs(vals[k]) > 0:
            vals = vals * (np.conj(vals[k]) / abs(vals[k]))
        ez = np.where(good, vals.real, np.nan)

        if normalise:
            peak = np.nanmax(np.abs(ez))
            if peak > 0:
                ez = ez / peak
        return z * 1e3, ez

    def plot_axis_field(self, mode=None, show_min_max=True, show=True, ax=None,
                        pol='monopole', normalise=True, label=None, **kwargs):
        """Plot the on-axis accelerating field.

        With *mode* given, the **signed** field of that mode is drawn (see
        :meth:`axis_field_profile`), which is what shows how a multi-cavity line
        is phased. With ``mode=None`` (default) the cached ``|Ez|`` of the mode
        of interest is drawn, annotated with the field flatness.
        """
        if mode is not None:
            z, ez = self.axis_field_profile(mode, pol=pol, normalise=normalise)
            if ax is None:
                _, ax = plt.subplots(figsize=(12, 3))
            ax.plot(z, ez, label=label if label is not None else f'mode {mode}',
                    **kwargs)
            ax.axhline(0.0, color='0.6', lw=0.8, ls=(0, (6, 4)), zorder=0)
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel('$E_z(0,0)$' + (' [normalised]' if normalise else ''))
            if show:
                plt.tight_layout()
                plt.show()
            return ax

        # Load the on-axis field if it isn't already in memory.
        if len(self.Ez_0_abs['z(0, 0)']) == 0:
            csv = os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')
            if not os.path.exists(csv):
                error('Axis field plot data not found.')
                return None
            with open(csv) as csv_file:
                self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        z = np.asarray(self.Ez_0_abs['z(0, 0)'])
        e = np.asarray(self.Ez_0_abs['|Ez(0, 0)|'])

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(z, e, label='$|E_z(0,0)|$')

        # Field flatness from the on-axis cell peaks — computed here from the
        # same peaks the min/max lines below draw, so the printed eta always
        # matches the plot. (self.ff is only populated after get_eigenmode_qois;
        # reading it here printed 0.00% on a fresh cavity.) Uses the same peak
        # detection as evaluate_qois so the value equals the QOI 'ff [%]'.
        peaks, _ = find_peaks(e, distance=max(1, len(e) // 100), width=100)
        ff = (min(e[peaks]) / max(e[peaks]) * 100) if len(peaks) else 100.0
        ax.text(0.95, 0.05, '$' + r'\eta=' + fr'{ff:.2f}\%' + '$', fontsize=12,
                ha='right', va='bottom', transform=ax.transAxes)
        ax.legend(loc="upper right")
        if show_min_max and len(peaks):
            ax.plot(z[peaks], e[peaks], marker='o', ls='')
            ax.axhline(min(e[peaks]), c='r', ls='--')
            ax.axhline(max(e[peaks]), c='k')
        _maybe_show(show)
        return ax

    def plot_spectra(self, var='freq [MHz]', ax=None, show=True, **kwargs):
        """Mode-spectrum plot — delegates to :meth:`EigenmodeSolver.plot_spectra`
        (the canonical home). Draws the per-mode distribution of *var* over the UQ
        samples, coloured by polarisation; see that method for mode/frequency
        selection, ``std_cap`` and reference overlays."""
        return self.eigenmode.plot_spectra(var=var, ax=ax, show=show, **kwargs)

    def plot_animate_wakefield(self, save=False):
        # Deferred: IPython is a notebook-only optional dependency ([jupyter] extra).
        from IPython.display import HTML, display_html

        def plot_contour_for_frame(data, frame_key, ax):
            ax.clear()

            colors = {'DOTS': 'k', 'SOLID': 'k'}

            if frame_key in data:
                for plot_type, df in data[frame_key]:
                    ax.plot(df['X'], df['Y'], linestyle='-', color=colors[plot_type])

            ax.set_xlabel('X-axis (m)')
            ax.set_ylabel('Y-axis (m)')
            ax.set_title(f'Contour Plot for {frame_key}')

        def animate_frames(data):
            fig, ax = plt.subplots(figsize=(18, 4))
            frame_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]))  # Sort frames by their order

            def update(frame_key):
                plot_contour_for_frame(data, frame_key, ax)

            ani = FuncAnimation(fig, update, frames=frame_keys, repeat=True, interval=1000)
            plt.close(fig)

            return ani

        top_folder = self.wakefield_dir
        efield_contour = get_wakefield_data(os.path.join(top_folder, 'longitudinal', 'cavity.top'))
        ani = animate_frames(efield_contour)
        display_html(HTML(animate_frames(efield_contour).to_jshtml()))

        if save:
            # Save the animation as an MP4 file
            ani.save(os.path.join(top_folder, f'{self.name}_e_field_animation.mp4'), writer='ffmpeg', dpi=150)

    def _plot_profile(self, profile, ax=None, mirror=False, fill=False,
                      center=True, control_points=None, regions=True, **kwargs):
        """Draw a cavity's meridian outline straight from its :class:`Profile`.

        Geometry-independent: works for every model, since each builds a Profile.
        The wall is drawn in mm. Only the upper half (``r >= 0``) is the analysed,
        axisymmetric domain, so that is what is drawn by default; pass
        ``mirror=True`` to reflect it about the axis for the full cross-section.

        ``center`` (default ``True``) centres each cavity on ``z = 0`` so cavities of
        different lengths line up about their middle when overlaid for comparison;
        ``center=False`` instead starts every cavity at ``z = 0`` (left-aligned).
        ``fill`` shades the interior (default ``False``).

        ``control_points`` overlays a spline cavity's control points and the
        polygon connecting them: ``None`` (default) shows them automatically for
        cavities that expose ``control_polygons()`` (i.e. splines), ``True`` forces
        it, ``False`` hides it.
        """
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 3))
            pts = np.asarray(profile.contour_points(3e-4, skip=('AXI',)), dtype=float)
            z, r = pts[:, 0] * 1e3, pts[:, 1] * 1e3          # metres -> mm
            # centre on z = 0 (overlaid cavities align at their middle) or, with
            # center=False, start every cavity at z = 0 (left-aligned).
            z_shift = 0.5 * (z.min() + z.max()) if center else z.min()
            z = z - z_shift
            color = kwargs.pop('color', None) or getattr(self, 'color', None) or WARM[1]
            lw = kwargs.pop('lw', kwargs.pop('linewidth', 1.8))
            label = kwargs.pop('label', self.name)
            if fill:
                poly_z = np.concatenate([z, z[::-1]])
                poly_r = np.concatenate([r, -r[::-1] if mirror else np.zeros_like(r)])
                ax.fill(poly_z, poly_r, color=color, alpha=0.12, lw=0)
            ax.plot(z, r, color=color, lw=lw, label=label, **kwargs)
            if mirror:
                ax.plot(z, -r, color=color, lw=lw, **kwargs)
            self._overlay_control_points(ax, z_shift, mirror, control_points)
            self._overlay_regions(ax, z_shift, mirror, regions)
            ax.set_aspect('equal')
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel(r'$r$ [mm]')
            return ax

    def _overlay_regions(self, ax, z_shift_mm, mirror, regions=True):
        """Shade this cavity's material regions on a geometry plot.

        A beam line absorber, a ceramic window or a dielectric liner is part of
        the geometry, so a drawing that omits it is misleading — most obviously
        for a :class:`~cavsim2d.models.bla.BLA`, whose whole point is the ring
        in its wall. Drawn in the same shifted-mm frame as the wall, and skipped
        with ``regions=False``.
        """
        if not regions:
            return
        for i, d in enumerate(self.dielectrics or ()):
            fc = d.get('color') or WARM[1 + i % (len(WARM) - 1)]
            if d.get('points'):
                poly = np.asarray(d['points'], dtype=float)
            elif d.get('z') is not None:
                (z0, z1), (r0, r1) = d['z'], d['r']
                poly = np.array([[z0, r0], [z1, r0], [z1, r1], [z0, r1]], dtype=float)
            else:
                continue
            pz = poly[:, 0] - z_shift_mm            # region extents are already mm
            pr = poly[:, 1]
            ax.fill(pz, pr, color=fc, alpha=0.45, lw=0, zorder=0,
                    label=d['material'])
            if mirror:
                ax.fill(pz, -pr, color=fc, alpha=0.45, lw=0, zorder=0,
                        label='_nolegend_')
        return ax

    def _overlay_control_points(self, ax, z_shift_mm, mirror, control_points):
        """Overlay spline control points and their connecting control polygon on a
        geometry plot, in the same shifted-mm frame as the wall.

        Drawn when *control_points* is ``True``, or when it is ``None`` (default)
        and this cavity exposes ``control_polygons()`` (i.e. a spline). A silent
        no-op for models without control points; ``control_points=False`` always
        suppresses it. Dashed polygon + open markers, so the interpolated wall
        still reads clearly underneath.
        """
        if control_points is False:
            return
        maker = getattr(self, 'control_polygons', None)
        polys = maker() if callable(maker) else None
        if not polys:
            if control_points:                    # explicitly asked but none exist
                info(f"{type(self).__name__} '{self.name}' has no control points "
                     "to overlay.")
            return
        cp_color = WARM[4]
        for i, poly in enumerate(polys):
            poly = np.asarray(poly, dtype=float)
            pz = poly[:, 0] * 1e3 - z_shift_mm
            pr = poly[:, 1] * 1e3
            # label only the first cell's polygon, so a legend shows one entry.
            ax.plot(pz, pr, ls='--', lw=1.0, marker='o', ms=5, color=cp_color,
                    mfc='white', mec=cp_color, zorder=6,
                    label='control points' if i == 0 else '_nolegend_')
            if mirror:
                ax.plot(pz, -pr, ls='--', lw=1.0, marker='o', ms=5, color=cp_color,
                        mfc='white', mec=cp_color, zorder=6, label='_nolegend_')
        return ax

    def plot(self, what, ax=None, scale_x=1, show=True, **kwargs):
        """Plot a static view of this cavity (``what='geometry'`` or
        ``'convergence'``). Like the solver-object ``plot_*`` methods this shows
        the figure by default; pass ``show=False`` to compose onto the returned
        axis (e.g. add a title, or overlay several cavities on one ``ax``)."""
        result = self._plot_dispatch(what, ax=ax, scale_x=scale_x, **kwargs)
        _maybe_show(show)
        return result

    def _plot_dispatch(self, what, ax=None, scale_x=1, **kwargs):
        if what.lower() == 'geometry':
            # Opt-in: `tuned=True` routes the plot to self.tuned when it
            # exists, so callers can overlay before/after on the same ax.
            tuned_flag = kwargs.pop('tuned', False)
            target = self
            if tuned_flag:
                if self.tuned is None:
                    warning('tuned=True requested but cavity has no tuned version yet — '
                            'plotting original geometry instead.')
                else:
                    target = self.tuned

            # Use the pure-matplotlib geometry function. The gmsh-oriented
            # `write_cavity_geometry_cli` only populates `geo` under
            # `write=...`, so it cannot draw a visible plot on its own.
            # ``ignore_degenerate=True`` keeps the drawing going on
            # borderline input instead of bailing on the tangent check —
            # the caller wants to *see* the geometry, even if it would
            # fail a strict mesh check.
            # Single-cell elliptical views (mid_cell / end_cell_*) keep the
            # cell-list drawer, which only the elliptical family carries.
            cell_kwargs = ('mid_cell', 'end_cell_left', 'end_cell_right')
            if any(k in kwargs for k in cell_kwargs) and hasattr(target, 'mid_cell'):
                common_kwargs = dict(scale=1, ax=ax, plot=True, ignore_degenerate=True)
                if 'mid_cell' in kwargs:
                    kw = {k: v for k, v in kwargs.items() if k != 'mid_cell'}
                    ax = write_cavity_geometry_cli_wo_gmsh(target.mid_cell, target.mid_cell,
                                                           target.mid_cell, 'none', 1,
                                                           **common_kwargs, **kw)
                elif 'end_cell_left' in kwargs:
                    ax = write_cavity_geometry_cli_wo_gmsh(target.end_cell_left, target.end_cell_left,
                                                           target.end_cell_left, 'left', 1,
                                                           **common_kwargs, **kwargs)
                else:
                    ax = write_cavity_geometry_cli_wo_gmsh(target.end_cell_right, target.end_cell_right,
                                                           target.end_cell_right, 'right', 1,
                                                           **common_kwargs, **kwargs)
                if ax is None:
                    return None
                ax.set_xlabel('$z$ [mm]')
                ax.set_ylabel(r"$r$ [mm]")
                return ax

            # General path: every model builds a Profile, so draw that directly.
            # This is geometry-independent — the spline, gun and pillbox all work,
            # not just the elliptical family (whose mid_cell/end_cell lists the
            # old drawer required).
            maker = getattr(target, 'profile', None)
            profile = maker() if callable(maker) else None
            if profile is not None:
                return target._plot_profile(profile, ax=ax, **kwargs)

            # Fallback for a cavity imported from a .geo/CAD file (no profile):
            # the elliptical cell-list drawer, if this type carries the lists.
            if hasattr(target, 'mid_cell'):
                ax = write_cavity_geometry_cli_wo_gmsh(
                    target.mid_cell, target.end_cell_left, target.end_cell_right,
                    target.beampipe, target.n_cells,
                    scale=1, ax=ax, plot=True, ignore_degenerate=True, **kwargs)
                if ax is None:
                    return None
                ax.set_xlabel('$z$ [mm]')
                ax.set_ylabel(r"$r$ [mm]")
                return ax

            error(f"{type(target).__name__} has no profile() and no cell lists to "
                  f"plot its geometry.")
            return None

        # All wakefield plotting now lives on the wakefield namespace:
        #   cav.wakefield.plot_impedance() / plot_wake() / plot_k_loss() / plot_k_kick()
        # cav.plot(...) keeps only the geometry and convergence views.
        if what.lower() in ('zl', 'zt', 'wpl', 'wpt'):
            error(f"cav.plot('{what}') was removed. Use the wakefield namespace: "
                  f"cav.wakefield.plot_impedance() / plot_wake() "
                  f"(kind='longitudinal'|'transverse').")
            return ax

        if what.lower() == 'convergence':
            try:
                if ax:
                    self._plot_convergence(ax)
                else:
                    fig, ax = plt.subplot_mosaic([['conv', 'abs_err']], layout='constrained', figsize=(12, 4))
                    self._plot_convergence(ax)
                return ax
            except ValueError:
                warning("Convergence data not available.")

    def show_geometry(self, plotter='ngsolve', maxh=20e-3, order=1, ax=None, show=True):
        """Interactive NGSolve (webgui) view of the analysed geometry.

        Meshes the geometry on the fly, so it needs no prior eigenmode run.
        For the flat matplotlib meridian outline use :meth:`plot` instead::

            cav.plot('geometry')     # matplotlib, half cross-section (r >= 0)

        Draws whatever :attr:`beampipe_length` the cavity carries, so it always
        shows the geometry a solve would mesh.

        With ``plotter='matplotlib'``, *ax* and *show* work as they do on the
        ``plot_*`` methods: draw into a given axes and pass ``show=False`` to
        keep the figure open for further panels. Both are ignored by the
        webgui plotter.

        The webgui view renders only in a Jupyter notebook. Anywhere else, or
        when the viewer's widget package is missing, this draws the matplotlib
        view instead and warns why.
        """
        return ngsolve_mevp.show_geometry(self, maxh=maxh, order=order, plotter=plotter,
                                          ax=ax, show=show)

    def show_mesh(self, plotter='ngsolve', pol='monopole', ax=None, show=True):
        """Interactive NGSolve (webgui) view of the mesh used in the analysis.

        *ax* / *show* apply to ``plotter='matplotlib'`` — see :meth:`show_geometry`."""
        mesh_path = self._eigenmode_pol_dir(pol)
        if any(os.path.exists(os.path.join(mesh_path, f)) for f in ('mesh.vol', 'mesh.pkl')):
            warn_if_stale(self, mesh_path, 'The eigenmode mesh')
            return ngsolve_mevp.show_mesh(mesh_path, plotter=plotter, ax=ax, show=show)
        self._eigenmode_artifact_error(pol, 'mesh.vol', 'mesh')

    def show_fields(self, mode=0, which='E', plotter='ngsolve', pol='monopole',
                    ax=None, show=True):
        """Interactive NGSolve (webgui) view of the eigenmode fields.

        For m-pole results pass ``pol`` ('dipole', 'quadrupole', ... or the mode
        number m); *which* then accepts 'E'/'H' (in-plane envelopes) as well as
        'Ephi'/'Hphi' (azimuthal envelopes).

        *ax* / *show* apply to ``plotter='matplotlib'`` — see :meth:`show_geometry`."""
        field_path = self._eigenmode_pol_dir(pol)
        if any(os.path.exists(os.path.join(field_path, f)) for f in ('field_meta.json', 'gfu_EH.pkl')):
            warn_if_stale(self, field_path, 'The eigenmode fields')
            return ngsolve_mevp.show_fields(field_path, mode, which, plotter,
                                            ax=ax, show=show)
        self._eigenmode_artifact_error(pol, 'field_meta.json', 'field data')

    def _plot_convergence(self, ax):
        keys = list(ax.keys())
        # plot convergence
        conv_filepath = os.path.join(self.self_dir, 'eigenmode', "tune_convergence.json")
        if os.path.exists(conv_filepath):
            with open(conv_filepath, 'r') as f:
                convergence_dict = json.load(f)
            if len(convergence_dict) > 0:
                x, y = convergence_dict[list(convergence_dict.keys())[0]], convergence_dict['freq [MHz]']
                ax[keys[0]].scatter(x, y, ec='k', label=self.name)

                # plot directions
                for i in range(len(x) - 1):
                    dx = x[i + 1] - x[i]
                    dy = y[i + 1] - y[i]
                    ax[keys[0]].quiver(x[i], y[i], dx, dy, ls='--', angles='xy',
                                       scale_units='xy', scale=1, color='red',
                                       width=0.005, units='width', headwidth=3, headlength=5,
                                       headaxislength=4)

        # plot absolute error
        abs_err_filepath = os.path.join(self.self_dir, 'eigenmode', "tune_absolute_error.json")
        abs_err_dict = {}
        if os.path.exists(conv_filepath):
            with open(abs_err_filepath, 'r') as f:
                abs_err_dict = json.load(f)
        if len(abs_err_dict) > 0:
            ax[keys[1]].plot(abs_err_dict['abs_err'], marker='o', mec='k', label=self.name)

        ax[keys[0]].set_xlabel('Parameter')
        ax[keys[0]].set_ylabel(r"Value")
        ax[keys[1]].set_xlabel('Iteration')
        ax[keys[1]].set_ylabel(r"Absolute error")
        ax[keys[1]].set_yscale('log')

    def inspect(self, cell_type='mid-cell', variation=0.2):
        # Deferred: IPython and ipywidgets are notebook-only optional dependencies
        # ([jupyter] extra).
        from IPython.display import display
        from ipywidgets import HBox, VBox, Label
        import ipywidgets as widgets

        if cell_type == 'mid-cell':
            cell = self.shape['IC']
        elif cell_type == 'end-cell-left':
            cell = self.shape['OC']
        elif cell_type == 'end-cell-right':
            cell = self.shape['OC_R']
        else:
            cell = self.shape['IC']

        if self.cell_parameterisation == 'flattop':
            A_, B_, a_, b_, Ri_, L_, Req_, l_ = cell[:8]

            # Define the function that plots the graph
            def plot_cavity_geometry_flattop(A, B, a, b, Ri, L, Req, l):
                cell = np.array([A, B, a, b, Ri, L, Req, l])
                write_cavity_geometry_cli_flattop(cell, cell, cell, BP='none',
                                                  n_cell=1, tangent_check=True, lw=1,
                                                  plot=True,
                                                  ignore_degenerate=True)

                # Update the sum display
                sum_label.value = f'Sum of A + a + l: {A + a + l:.2f}, L: {L}, delta: {A + a + l - L}'

            # Create sliders for each variable
            A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
                                           description='A')
            B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
                                           description='B')
            a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
                                           description='a')
            b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
                                           description='b')
            Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
                                            description='Ri')
            L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
                                           description='L')
            Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
                                             value=Req_,
                                             description='Req')
            l_slider = widgets.FloatSlider(min=(1 - variation) * l_, max=(1 + variation) * l_, step=0.1, value=l_,
                                           description='l')
            # Create a label to display the sum of A + a
            sum_label = Label()

            # Arrange the sliders in a 3x3 layout
            ui = VBox([
                HBox([A_slider, B_slider, a_slider]),
                HBox([b_slider, Ri_slider, L_slider]),
                HBox([Req_slider, l_slider]),
                sum_label  # Add the sum label to the layout
            ])

            # Create an interactive widget to update the plot
            out = widgets.interactive_output(plot_cavity_geometry_flattop,
                                             {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
                                              'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider, 'l': l_slider})

        else:
            A_, B_, a_, b_, Ri_, L_, Req_ = cell[:7]

            # Define the function that plots the graph
            def plot_cavity_geometry(A, B, a, b, Ri, L, Req):
                cell = np.array([A, B, a, b, Ri, L, Req])
                write_cavity_geometry_cli(cell, cell, cell, BP='none', n_cell=1,
                                          tangent_check=True, lw=1,
                                          plot=True,
                                          ignore_degenerate=True)

                # Update the sum display
                sum_label.value = f'Sum of A + a: {A + a:.2f}, L: {L}, delta: {A + a - L}'

            # Create sliders for each variable
            A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
                                           description='A')
            B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
                                           description='B')
            a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
                                           description='a')
            b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
                                           description='b')
            Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
                                            description='Ri')
            L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
                                           description='L')
            Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
                                             value=Req_,
                                             description='Req')
            # Create a label to display the sum of A + a
            sum_label = Label()

            # Arrange the sliders in a 3x3 layout
            ui = VBox([
                HBox([A_slider, B_slider, a_slider]),
                HBox([b_slider, Ri_slider, L_slider]),
                HBox([Req_slider]),
                sum_label  # Add the sum label to the layout
            ])

            # Create an interactive widget to update the plot
            out = widgets.interactive_output(plot_cavity_geometry,
                                             {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
                                              'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider})

        # Display the layout
        display(out, ui)
