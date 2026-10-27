"""Comparison plots and summary tables across the cavities of a study.

:class:`~cavsim2d.study.Study` inherits these from :class:`StudyPlotting`.
"""
from cavsim2d.models.base import Cavity
from cavsim2d.constants import c0, LABELS, q0
from cavsim2d.geometry.writers.gmsh import (write_cavity_geometry_cli,
                                            write_cavity_geometry_cli_flattop)
from cavsim2d.processes.wakefield import axis_data_coords_sys_transform
from cavsim2d.utils.data_utils import reorder_legend
from cavsim2d.utils.printing import error, info
from scipy.special import jn_zeros, jnp_zeros
import fnmatch
import json
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.interpolate as sci
import scipy.io as spio
from cavsim2d.utils.config_validation import require
import re
from cavsim2d.solvers.solver_objects import _maybe_show
from cavsim2d.utils.style import house_style, WARM


def _primary(by_label):
    """The entry of an operating point's primary (first) bunch length, from a
    result nested ``{label: ...}``. The primary used to be hard-wired as 'SR'."""
    return by_label[next(iter(by_label))]


def _qoi(cav, key):
    """*cav*'s eigenmode result *key*, 0 when it has none (for the summary tables)."""
    return (getattr(cav, 'eigenmode_qois', None) or {}).get(key, 0)


class StudyPlotting:
    """Comparison plots and summary tables of a :class:`~cavsim2d.study.Study`."""

    def save_plot_as_json(self, ax):
        ax_children = ax.get_children()

        ax_objects_dict = {
            'ax props': None,
            'fig props': {},
            'lines': {
                # 'line1': line_properties,
            },
            'axvline': {},
            'axhline': {},
            'scatter': {},
            'patches': {
                'text': {},
                'rectangle': {}
            },
        }

        # get axis properties
        axis_properties = self._get_axis_properties(ax)
        ax_objects_dict['ax props'] = axis_properties

        # get figure properties
        fig_props = self._get_figure_properties(ax.get_figure())
        ax_objects_dict['fig props'] = fig_props

        # List of known properties for Line2D
        line_properties = [
            'xdata', 'ydata', 'alpha', 'animated', 'antialiased', 'clip_on', 'clip_path',
            'color', 'dash_capstyle', 'dash_joinstyle', 'drawstyle',
            'label', 'linestyle', 'linewidth', 'marker', 'markeredgecolor',
            'markeredgewidth', 'markerfacecolor', 'markersize', 'path_effects', 'rasterized', 'sketch_params',
            'snap',
            'solid_capstyle',
            'solid_joinstyle', 'url', 'visible', 'zorder',
            # 'transform',
            # 'clip_box',
            # 'figure', 'picker', 'pickradius'
        ]

        for mpl_obj in ax_children:
            if isinstance(mpl_obj, matplotlib.lines.Line2D):
                # Extract properties into a dictionary
                line_properties_data = {prop: getattr(mpl_obj, f'get_{prop}')() for prop in line_properties}
                ax_objects_dict['lines'][fr'{id(mpl_obj)}'] = line_properties_data

        return ax_objects_dict

    def load_plot_from_json(self, filepath, ax=None):
        with open(filepath, 'r') as f:
            plot_config = json.load(f)

        if ax is not None:
            fig, ax = plt.subplot_mosaic([[0]])

        for key, value in plot_config.items():
            if key == 'lines':
                line = ax.plot([], [])
                line.set()
            if key == 'scatter':
                ax.scatter(value['x'], value['y'], value['kwargs'])
            if key == 'patches':
                if value['type'] == 'text':
                    ax.add_text(value['x'], value['y'], value['kwargs'])

    def plot_from_json(self, plot_config, ax=None):
        if ax is None:
            fig, ax = plt.subplot_mosaic([[0]])

        for key, value in plot_config.items():
            if key == 'lines':
                for line_keys, line_values in value.items():
                    line, = ax[0].plot([], [])
                    line.set(**line_values)
            if key == 'scatter':
                pass
                # ax[0].scatter(value['x'], value['y'], value['kwargs'])
            if key == 'patches':
                pass
                # if value['type'] == 'text':
                #     ax[0].add_text(value['x'], value['y'], value['kwargs'])
            if key == 'fig props':
                ax[0].get_figure().set(**value)

            if key == 'ax props':
                ax[0].set(**value)

        if isinstance(ax, dict):
            ax[0].relim()
            ax[0].get_figure().canvas.draw()
            ax[0].get_figure().canvas.flush_events()
            ax[0].get_figure().canvas.draw_idle()
        else:
            ax.relim()
            ax.get_figure().canvas.draw()
            ax.get_figure().canvas.flush_events()
            ax.get_figure().canvas.draw_idle()

        # plot legend wthout duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    def _get_axis_properties(self, ax):
        # List of common properties for an Axis
        axis_properties = [
            'label', 'ticks', 'scale', 'margin', 'bound', 'aspect'
        ]

        def get_axis_properties(axis, prefix):
            properties = {}
            for prop in axis_properties:
                try:
                    # Use get_* methods dynamically if available
                    getter = getattr(axis, f'get_{prop}', None)
                    if callable(getter):
                        if prop == 'label':
                            properties[fr'{prefix}{prop}'] = getter().get_text()
                        else:
                            properties[fr'{prefix}{prop}'] = getter()
                    else:
                        # Fallback to axis.get_property()
                        properties[fr'{prefix}{prop}'] = axis.get_property(prop)
                except Exception:
                    pass
            return properties

        axis_properties = get_axis_properties(ax.xaxis, 'x')
        axis_properties.update(get_axis_properties(ax.yaxis, 'y'))

        return axis_properties

    def _get_figure_properties(self, fig):
        # List of properties to retrieve
        figure_properties = [
            'figwidth', 'figheight', 'dpi', 'tight_layout', 'constrained_layout'
        ]

        def get_figure_properties(fig):
            properties = {}
            for prop in figure_properties:
                getter = getattr(fig, f'get_{prop}', None)
                if callable(getter):
                    properties[prop] = getter()

                else:
                    properties[prop] = 'N/A'
            return properties

        # Get properties of the figure
        fig_size = []
        fig_properties = get_figure_properties(fig)

        return fig_properties

    def plot_dispersion(self, pol=None, bands=None, ax=None, break_axis=True,
                        breaks=None, light_line=True, tuned='auto', show=True, **kwargs):
        """Overlay each cavity's dispersion diagram. ``pol`` selects the
        polarisation(s): ``None`` (default) shows every computed polarisation,
        or name one ('monopole', 'dipole', ...) or a list. ``bands`` selects
        passbands; ``break_axis`` splits the y-axis where passbands are far apart,
        ``breaks`` places those splits by hand, and ``light_line`` overlays the
        speed-of-light line (see :meth:`Cavity.plot_dispersion`). A cavity's
        current tuned version is drawn in its place when it has eigenmode
        results; ``tuned=False`` draws every cavity untuned. ``show=False`` leaves
        the figure open to compose on."""
        view = self._comparison_view('eigenmode', tuned)
        if view is not self:
            return view.plot_dispersion(pol=pol, bands=bands, ax=ax,
                                        break_axis=break_axis, breaks=breaks,
                                        light_line=light_line, tuned=False, show=show,
                                        **kwargs)
        if not self.cavities_list:
            return ax
        # Gather every cavity's series first, then render once, so the broken
        # axis is computed over all the data at once.
        series, n, light_lines = [], None, []
        for cav in self.cavities_list:
            cav_series, cav_n = cav._dispersion_series(pol=pol, bands=bands)
            series.extend(cav_series)
            n = n or cav_n
            d = cav._cell_length_m() if light_line else None
            if d and not any(abs(d - ll['d']) < 1e-9 for ll in light_lines):
                lbl = 'light line' if len(self.cavities_list) == 1 else f'light line ({cav.name})'
                light_lines.append({'d': d, 'label': lbl})
        ax = Cavity._render_dispersion(series, n, ax=ax, break_axis=break_axis,
                                       breaks=breaks, light_lines=light_lines or None,
                                       **kwargs)
        _maybe_show(show)
        return ax

    def plot(self, what, ax=None, scale_x=None, **kwargs):
        # All wakefield plotting is on the wakefield namespace now
        # (study.wakefield.plot_impedance() / plot_wake() / plot_k_loss()); this
        # dispatch keeps only the geometry and convergence views.
        if what.lower() in ('zl', 'zt', 'wpl', 'wpt'):
            error(f"study.plot('{what}') was removed. Use the wakefield namespace: "
                  f"study.wakefield.plot_impedance() / plot_wake().")
            return ax
        for ii, cav in enumerate(self.cavities_list):
            if what.lower() == 'geometry':
                ax = cav.plot('geometry', ax, show=False, **kwargs)

            if what.lower() == 'convergence':
                ax = cav.plot('convergence', ax, show=False)
        if 'logy' in kwargs:
            if kwargs['logy']:
                ax.set_yscale('log')

        plt.legend()
        return ax

    def qois_hom(self, opt):
        """
        Retrieves the higher-order modes quantities of interest

        Returns
        -------
        Dictionary containing higher-order modes quantities of interest (normed optional).
        """

        results = []
        for cavity in self.cavities_list:
            cavity.get_wakefield_qois(self.wakefield_config)

            results.append({
                r"|k_loss| [V/pC]": cavity.k_loss[opt],
                r"|k_kick| [V/pC/m]": cavity.k_kick[opt],
                r"P_HOM [kW]": cavity.phom[opt]
            })

        return results

    def qois_all(self, opt):
        """
        Retrieves the fundamental mode quantities of interest

        Returns
        -------
        Dictionary containing fundamental mode quantities of interest (normed optional).
        """
        results = []
        for cav in self.cavities_list:
            results.append({
                r"$E_\mathrm{pk}/E_\mathrm{acc} ~[\cdot]$": _qoi(cav, 'Epk/Eacc []'),
                r"$B_\mathrm{pk}/E_\mathrm{acc} ~\mathrm{[mT/MV/m]}$": _qoi(cav, 'Bpk/Eacc [mT/MV/m]'),
                r"$k_\mathrm{cc}$": _qoi(cav, 'kcc [%]'),
                r"$R/Q~ \mathrm{[\Omega]}$": _qoi(cav, 'R/Q [Ohm]'),
                r"$G ~\mathrm{[\Omega]}$": _qoi(cav, 'G [Ohm]'),
                r"$G\cdot R/Q ~\mathrm{[10^4\Omega^2]}$": _qoi(cav, 'GR/Q [Ohm^2]') * 1e-4,
                r"$|k_\parallel| ~\mathrm{[V/pC]}$": cav.k_loss[opt],
                r"$|k_\perp| ~\mathrm{[V/pC/m]}$": cav.k_kick[opt],
                r"$P_\mathrm{HOM}/cav~ \mathrm{[kW]}$": cav.phom[opt]
            })

        results_norm_units = []
        for cav in self.cavities_list:
            results_norm_units.append({
                r"$e_\mathrm{pk}/e_\mathrm{acc}$": _qoi(cav, 'Epk/Eacc []'),
                r"$b_\mathrm{pk}/e_\mathrm{acc}$": _qoi(cav, 'Bpk/Eacc [mT/MV/m]'),
                r"$k_\mathrm{cc}$": _qoi(cav, 'kcc [%]'),
                r"$r/q$": _qoi(cav, 'R/Q [Ohm]'),
                r"$g$": _qoi(cav, 'G [Ohm]'),
                # r"$g\cdot r/q $": cav.GR_Q,
                # r"$|k_\mathrm{FM}|$": cav.k_fm,
                r"$|k_\parallel|$": cav.k_loss[opt],
                r"$k_\perp$": cav.k_kick[opt],
                r"$p_\mathrm{HOM}/cav$": cav.phom[opt]
            })

        return results

    @staticmethod
    def find_folders_with_tag(directory, tag):
        matching_folders = []

        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for dir_name in dirs:
                # Check if the folder name contains the tag
                if fnmatch.fnmatch(dir_name, f'*{tag}*'):
                    matching_folders.append(os.path.join(root, dir_name))

        return matching_folders

    def _resolve_qoi_keys(self, qois, df_columns):
        def normalize(s):
            s = s.lower()
            s = s.replace(r'\mathrm', '').replace(r'\epk', 'epk').replace(r'\bpk', 'bpk')
            s = s.replace(r'\parallel', 'loss').replace(r'\perp', 'kick')
            s = s.replace('kick', 'kick').replace('loss', 'loss').replace('perp', 'kick').replace('parallel', 'loss')
            s = re.sub(r'[^a-z0-9]', '', s)
            return s

        resolved = []
        if isinstance(qois, str):
            qois = [qois]

        normalized_cols = {normalize(col): col for col in df_columns}

        for qoi in qois:
            norm_qoi = normalize(qoi)
            if norm_qoi in normalized_cols:
                resolved.append(normalized_cols[norm_qoi])
            else:
                found = False
                for norm_col, col in normalized_cols.items():
                    if norm_qoi in norm_col or norm_col in norm_qoi:
                        resolved.append(col)
                        found = True
                        break
                if not found:
                    resolved.append(qoi)
        return [r for r in resolved if r in df_columns]

    def get_power_qois(self, cav, rf_config, op_points_list, uq=False):
        """

        Parameters
        ----------
        cavity: object
            Cavity object
        op_field: float
            Cavity operating field

        Returns
        -------
        Dictionary containing quantities of interest (normed optional).
        """

        if uq:
            qois = cav.get_power_uq(rf_config, op_points_list)
        else:
            qois = cav.get_power(rf_config, op_points_list)

        return qois

    def _power_scatter_impl(self, rf_config, op_points_list, ncols=3, uq=False, figsize=(12, 3), qois=None):
        """
        Plots bar chart of power quantities of interest

        Returns
        -------

        """
        if qois is None:
            qois = ['Pwp/cav', 'Pin/cav', 'PHOM/cav', 'k_loss', 'k_kick', 'p_hom']

        self.rf_config = rf_config

        require(len(rf_config['Q0 []']) == len(self.cavities_list),
                'Lengh of Q0 [] must equal number of Cavity objects.')
        require(len(rf_config['inv_eta []']) == len(self.cavities_list),
                'Lengh of inv_eta [] must equal number of Cavity objects.')
        if 'Eacc [MV/m]' in rf_config.keys():
            require(len(rf_config['Eacc [MV/m]']) == len(self.cavities_list),
                    'length of accelerating field Eacc [MV/m] must equal number of Cavity objects in the Study.')
        if 'V [GV]' in rf_config.keys():
            require(len(rf_config['V [GV]']) == len(op_points_list),
                    'length of RF Voltage V [GV] must equal number of operating points.')

        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        for ii, cav in enumerate(self.cavities_list):
            # set cavity intrinsic quality factor and inv_eta
            cav.Q0 = rf_config['Q0 []'][ii]
            cav.inv_eta = rf_config['inv_eta []'][ii]
            if 'Eacc [MV/m]' in rf_config.keys():
                cav.Eacc_rf_config = rf_config['Eacc [MV/m]'][ii]  # change this later, not effective way

            self.power_qois[cav.name] = self.get_power_qois(cav, rf_config, op_points_list)
            # Only with uq=True: the UQ variant reads the UQ sample table
            # (cav.neighbours), which a cavity without a UQ run does not have, so
            # computing it unconditionally made every plain power plot raise.
            if uq:
                self.power_qois_uq[cav.name] = self.get_power_qois(cav, rf_config, op_points_list,
                                                                   uq=True)

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            selected_qois = self._resolve_qoi_keys(qois, df.columns)
            if not selected_qois:
                raise ValueError("None of the specified qois are present in the results.")
            df = df[selected_qois]
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained', figsize=(3 * len(df.columns), 3))

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index[i], df[key].iloc[i], color=colors[i % len(colors)], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:
            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for op_pt in op_points_list:
                # get nominal qois
                dd_nominal = {}
                for cav, metrics in self.power_qois.items():
                    # for metric, values in _primary(metrics[op_pt]).items():
                    dd_nominal[cav] = _primary(metrics[op_pt])

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.power_qois_uq.items():
                    for metric, values in _primary(metrics[op_pt]).items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            selected_qois = self._resolve_qoi_keys(qois, metrics)
            metrics = [q for q in selected_qois if q in metrics]
            if not metrics:
                raise ValueError("None of the specified qois are present in the UQ results.")
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(3 * len(metrics), 3))

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                    marker=opt_format[ii],
                                                    fc='none', ec=colors[i], lw=2, zorder=100,
                                                    label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=scatter_points.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_scatter_{op_points_list}.png")

    def _eigenmode_compare_impl(self, kind='scatter', uq=False, ncols=3, qois=None):
        if kind == 'scatter' or kind == 's':
            self._fm_scatter_impl(uq=uq, ncols=ncols, qois=qois)
        if kind == 'bar' or kind == 'b':
            self._fm_bar_impl(uq=uq, ncols=ncols, qois=qois)

    def _wakefield_compare_impl(self, opt, kind='scatter', uq=False, ncols=3, figsize=(12, 3), qois=None):
        if kind == 'scatter' or kind == 's':
            self._hom_scatter_impl(opt, uq=uq, ncols=ncols, figsize=figsize, qois=qois)
        if kind == 'bar' or kind == 'b':
            self._hom_bar_impl(opt, uq=uq, ncols=ncols, qois=qois)

    def _hom_window_uq_impl(self, kind='scatter', ncols=3, figsize=(12, 3),
                            qois=None):
        """Mean and standard deviation of each impedance-window UQ objective.

        The companion to :meth:`_hom_scatter_impl` for a wakefield UQ run whose
        objectives are ``ZL``/``ZT`` peaks over frequency windows rather than
        operating-point quantities. One panel per window, every cavity in the
        study on it, each drawn as its mean with a +/-1 sigma error bar.
        """
        rows = []
        for cavity, metrics in self.uq_hom_results.items():
            for metric, values in metrics.items():
                if not (isinstance(values, dict) and 'expe' in values):
                    continue        # an operating-point nest, not a flat window
                rows.append({'cavity': cavity, 'metric': metric,
                             'mean': float(np.atleast_1d(values['expe'])[0]),
                             'std': float(np.atleast_1d(values['stdDev'])[0])})
        if not rows:
            info('No impedance-window UQ results in this study. Run a wakefield '
                 'with a uq_config whose objectives are ZL/ZT windows, or pass '
                 'op_points_list for the operating-point QOIs.')
            return None

        df = pd.DataFrame(rows)
        metrics = list(dict.fromkeys(df['metric']))
        if qois:
            wanted = self._resolve_qoi_keys(qois, metrics) or [
                m for m in metrics if any(str(q) in m for q in qois)]
            metrics = [m for m in metrics if m in wanted] or metrics

        labels = list(dict.fromkeys(df['cavity']))
        # Honour an explicitly-set cavity colour; fall back to the house palette
        # when it is still the default 'k', so several cavities stay tellable
        # apart instead of all coming out black.
        colors = {cav.name: (cav.color if cav.color not in (None, 'k') else None)
                  for cav in self.cavities_list}
        with house_style():
            fig, axd = plt.subplot_mosaic([metrics], layout='constrained',
                                          figsize=(3 * len(metrics), figsize[1]))
            for metric, ax in axd.items():
                sub = df[df['metric'] == metric]
                for i, label in enumerate(labels):
                    r = sub[sub['cavity'] == label]
                    if r.empty:
                        continue
                    colour = colors.get(label) or WARM[i % len(WARM)]
                    if kind == 'bar':
                        ax.bar(i, r['mean'].iloc[0], yerr=r['std'].iloc[0],
                               capsize=6, color=colour, edgecolor='k', label=label)
                    else:
                        ax.errorbar(i, r['mean'].iloc[0], yerr=r['std'].iloc[0],
                                    fmt='o', ms=8, capsize=6, lw=1.6, mfc='none',
                                    mew=2, color=colour, label=label)
                ax.set_xticks([])
                ax.set_xlim(-0.6, len(labels) - 0.4)
                ax.margins(y=0.25)
                # LABELS has no entry for an expanded window name, and indexing it
                # used to raise; fall back to the objective's own name.
                ax.set_ylabel(LABELS.get(metric, metric))
            h, l = axd[metrics[0]].get_legend_handles_labels()
            if not ncols:
                ncols = min(4, len(labels))
            fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center',
                       borderaxespad=0, ncol=ncols)

        fname = '_'.join(cav.name for cav in self.cavities_list)
        self.save_all_plots(f"{fname}_hom_window_uq_{kind}.png")
        return axd

    def _hom_bar_impl(self, op_points_list=None, ncols=3, uq=False, figsize=(12, 3), qois=None):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.
        qois : list of str, optional
            List of quantities of interest (QoIs) to plot.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        # Impedance-window objectives. Without 'operating_points' in the uq_config,
        # uq_hom_results is flat -- {objective: {'expe', 'stdDev'}} keyed by the
        # expanded window name ('ZL [max(0.2<f<2.0)]') -- with no operating point and
        # no 'SR' level for the branch below to index. That is what a wakefield UQ
        # run over ZL/ZT windows produces, so it gets its own renderer.
        if uq and not op_points_list:
            return self._hom_window_uq_impl(kind='bar', ncols=ncols,
                                            figsize=figsize, qois=qois)

        if qois is None:
            qois = ['k_loss', 'k_kick', 'p_hom']

        if not op_points_list:
            raise ValueError(
                'plot_hom_bar: op_points_list is required for the operating-point '
                'QOIs (k_loss, k_kick, p_hom). Omit it only with uq=True, to plot '
                'impedance-window UQ objectives.')

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            selected_qois = self._resolve_qoi_keys(qois, df.columns)
            if not selected_qois:
                raise ValueError("None of the specified qois are present in the HOM results.")
            df = df[selected_qois]
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained', figsize=(3 * len(df.columns), 3))

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.bar(df.index[i], df[key].iloc[i], color=colors[i % len(colors)], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:

            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for opt in op_points_list:

                # get nominal qois
                dd_nominal = {}
                for cav, ops_id in self.wakefield_qois_op.items():
                    for kk, vv in ops_id.items():
                        if kk == self._primary_tag(opt):
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in _primary(metrics[opt]).items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            selected_qois = self._resolve_qoi_keys(qois, metrics)
            metrics = [q for q in selected_qois if q in metrics]
            if not metrics:
                raise ValueError("None of the specified qois are present in the UQ HOM results.")
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(3 * len(metrics), 3))

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        bar = ax.bar(sub_df['cavity'], sub_df['mean'], color=colors[i],
                                     fc='none', ec=colors[i], lw=2, zorder=100,
                                     label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=bar.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_hom_scatter_{op_points_list}.png")

        return axd

    def _hom_scatter_impl(self, op_points_list=None, ncols=3, uq=False, figsize=(12, 3), qois=None):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.
        qois : list of str, optional
            List of quantities of interest (QoIs) to plot.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        # Impedance-window objectives. Without 'operating_points' in the uq_config,
        # uq_hom_results is flat -- {objective: {'expe', 'stdDev'}} keyed by the
        # expanded window name ('ZL [max(0.2<f<2.0)]') -- with no operating point and
        # no 'SR' level for the branch below to index. That is what a wakefield UQ
        # run over ZL/ZT windows produces, so it gets its own renderer.
        if uq and not op_points_list:
            return self._hom_window_uq_impl(kind='scatter', ncols=ncols,
                                            figsize=figsize, qois=qois)

        if qois is None:
            qois = ['k_loss', 'k_kick', 'p_hom']

        if not op_points_list:
            raise ValueError(
                'plot_hom_scatter: op_points_list is required for the operating-point '
                'QOIs (k_loss, k_kick, p_hom). Omit it only with uq=True, to plot '
                'impedance-window UQ objectives.')

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            selected_qois = self._resolve_qoi_keys(qois, df.columns)
            if not selected_qois:
                raise ValueError("None of the specified qois are present in the HOM results.")
            df = df[selected_qois]
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained', figsize=(3 * len(df.columns), 3))

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index[i], df[key].iloc[i], color=colors[i % len(colors)], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:

            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for opt in op_points_list:

                # get nominal qois
                dd_nominal = {}
                for cav, ops_id in self.wakefield_qois_op.items():
                    for kk, vv in ops_id.items():
                        if kk == self._primary_tag(opt):
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in _primary(metrics[opt]).items():
                        rows.append({
                            'cavity': cavity,
                            'metric': metric,
                            'mean': values['expe'][0],
                            'std': values['stdDev'][0]
                        })

                df = pd.DataFrame(rows)
                df_list.append(df)

            # Step 2: Create a Mosaic Plot
            metrics = df_list[0]['metric'].unique()
            selected_qois = self._resolve_qoi_keys(qois, metrics)
            metrics = [q for q in selected_qois if q in metrics]
            if not metrics:
                raise ValueError("None of the specified qois are present in the UQ HOM results.")
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(3 * len(metrics), 3))

            # Plotting labels and colors
            labels = df_list[0]['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]
            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    for ii, (opt, df, df_nominal) in enumerate(zip(op_points_list, df_list, df_nominal_list)):
                        sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                        scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                    marker=opt_format[ii],
                                                    fc='none', ec=colors[i], lw=2, zorder=100,
                                                    label=fr'{label} ({LABELS[opt]})')
                        ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt=opt_format[ii],
                                    capsize=10, lw=2, mfc='none',
                                    color=scatter_points.get_edgecolor()[0])

                        # plot nominal
                        ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', ec='k',
                                   lw=1, marker=opt_format[ii],
                                   zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_hom_scatter_{op_points_list}.png")

        return axd

    def _fm_bar_impl(self, ncols=3, uq=False, qois=None):
        """
        Plot bar chart of fundamental mode quantities of interest

        Returns
        -------

        """
        if qois is None:
            qois = ['freq', 'Epk/Eacc', 'Bpk/Eacc', 'R/Q']

        if not uq:
            self.fm_results = self.qois_fm(tuned=False)

            df = pd.DataFrame.from_dict(self.fm_results)
            selected_qois = self._resolve_qoi_keys(qois, df.columns)
            if not selected_qois:
                raise ValueError("None of the specified qois are present in the fundamental mode results.")
            df = df[selected_qois]
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained', figsize=(3 * len(df.columns), 4))

            labels = [cav.plot_label for cav in self.cavities_list]
            # Plot each column in a separate subplot
            for key, ax in axd.items():
                ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
                       edgecolor='k',
                       width=1)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS[key])
                h, l = ax.get_legend_handles_labels()
        else:
            self.fm_results = self.uq_fm_results

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cav, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cav,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            labels = [cav.plot_label for cav in self.cavities_list]

            # Step 2: Create a Mosaic Plot. UQ statistics are computed for every
            # QOI, but only the recognised figures of merit (those with a plot
            # label) are worth a bar — bookkeeping keys like
            # "Normalization Length [mm]" or "N Cells" are skipped.
            # UQ metric keys carry a polarisation prefix now
            # ('monopole:freq [MHz]'); match/label on the bare QOI name.
            def _bare(k):
                return k.split(':')[-1].strip()
            metrics = [m for m in df['metric'].unique() if _bare(m) in LABELS]
            selected_qois = self._resolve_qoi_keys(qois, metrics)
            metrics = [q for q in selected_qois if q in metrics]
            if not metrics:
                info("No plottable figures of merit in the UQ results.")
                return None

            layout = [list(metrics)]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(3 * len(metrics), 4))

            for metric, ax in axd.items():
                sub_df = df[df['metric'] == metric]
                ax.bar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], label=labels, capsize=5,
                       color=matplotlib.colormaps['Set2'].colors[:len(df)], edgecolor='k',
                       width=1)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS.get(_bare(metric), metric))
                h, l = ax.get_legend_handles_labels()

        # fig.set_tight_layout(True)
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_fm_bar.png")

        return axd

    def _compare_colors(self):
        """One distinct warm colour per cavity for the comparison plots — the
        cavity's own ``color`` when set to something other than the default black,
        otherwise the house WARM palette by position."""
        out = []
        for i, cav in enumerate(self.cavities_list):
            c = getattr(cav, 'color', None)
            out.append(c if c and c not in ('k', 'black', '#000000') else WARM[i % len(WARM)])
        return out

    def _fm_scatter_impl(self, ncols=3, uq=False, qois=None):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.
        qois : list of str, optional
            List of quantities of interest (QoIs) to plot. Default includes:
            frequency, Epk/Eacc, Bpk/Eacc, R/Q.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        QOI_MAP = {
            'frequency': 'freq [MHz]',
            'freq': 'freq [MHz]',
            'freq [mhz]': 'freq [MHz]',
            'epk/eacc': 'Epk/Eacc []',
            'epk/eacc []': 'Epk/Eacc []',
            'epk': 'Epk/Eacc []',
            'bpk/eacc': 'Bpk/Eacc [mT/MV/m]',
            'bpk/eacc [mt/mv/m]': 'Bpk/Eacc [mT/MV/m]',
            'bpk': 'Bpk/Eacc [mT/MV/m]',
            'r/q': 'R/Q [Ohm]',
            'r/q [ohm]': 'R/Q [Ohm]',
            'g': 'G [Ohm]',
            'g [ohm]': 'G [Ohm]',
            'gr/q': 'GR/Q [Ohm^2]',
            'gr/q [ohm^2]': 'GR/Q [Ohm^2]',
            'kcc': 'kcc [%]',
            'kcc [%]': 'kcc [%]',
            'ff': 'ff [%]',
            'ff [%]': 'ff [%]',
            'q': 'Q []',
            'q []': 'Q []'
        }

        if qois is None:
            # The paper's Fig. 5 set (G, kcc, ff were missing before). Absent
            # columns are filtered out below, so this degrades gracefully to
            # whatever the run actually recorded.
            selected_qois = ['freq [MHz]', 'Epk/Eacc []', 'Bpk/Eacc [mT/MV/m]',
                             'R/Q [Ohm]', 'G [Ohm]', 'kcc [%]', 'ff [%]']
        else:
            if isinstance(qois, str):
                qois = [qois]
            selected_qois = []
            for qoi in qois:
                qoi_lower = qoi.strip().lower()
                if qoi_lower in QOI_MAP:
                    selected_qois.append(QOI_MAP[qoi_lower])
                else:
                    selected_qois.append(qoi)

        if not uq:
            self.fm_results = self.qois_fm(tuned=False)
            df = pd.DataFrame.from_dict(self.fm_results)
            selected_qois = [q for q in selected_qois if q in df.columns]
            if not selected_qois:
                raise ValueError("None of the specified qois are present in the fundamental mode results.")
            df = df[selected_qois]

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = self._compare_colors()

            with house_style():
                fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained',
                                              figsize=(3 * len(df.columns), 3))
                # Each cavity is one point per subplot; plot only *its own* row so
                # the marker colour matches its legend entry. (The old code drew
                # the whole column once per cavity, so every point took the last
                # cavity's colour while the legend kept per-cavity colours.)
                for key, ax in axd.items():
                    for i, label in enumerate(labels):
                        ax.scatter(df.index[i], df[key].iloc[i], color=colors[i],
                                   edgecolors='k', linewidths=0.8, label=label, zorder=3)
                    ax.set_xticklabels([])
                    ax.set_xticks([])
                    ax.set_ylabel(LABELS[key])

                h, l = ax.get_legend_handles_labels()
        else:
            df_nominal = pd.DataFrame.from_dict(self.eigenmode_qois).T

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cav, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cav,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = self._compare_colors()

            # Step 2: Create a Mosaic Plot. UQ metric keys carry a polarisation
            # prefix now ('monopole:freq [MHz]'), so match/label on the bare QOI.
            def _bare(k):
                return k.split(':')[-1].strip()
            available = list(df['metric'].unique())
            # Accept a polarisation-qualified request ('monopole:R/Q [Ohm]') as well
            # as a bare one ('R/Q [Ohm]'): compare on the bare name from BOTH sides,
            # preferring an exact match. Comparing a stripped available name against
            # an unstripped requested one meant a qualified `qois=` never matched.
            metrics = []
            for q in selected_qois:
                hits = ([m for m in available if m == q]
                        or [m for m in available if _bare(m) == _bare(q)])
                for m in hits:
                    if m not in metrics:
                        metrics.append(m)
            if not metrics:
                raise ValueError(
                    f"None of the specified qois are present in the UQ fundamental "
                    f"mode results. Asked for {list(selected_qois)}; available: "
                    f"{available}.")

            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(3 * len(metrics), 3))

            # Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                    scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                fc='none', ec=colors[i], label='Mean', lw=3, zorder=100)
                    ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], capsize=10, lw=3,
                                color=scatter_points.get_edgecolor()[0])

                    # Overlay the nominal design point when it is available. A cavity
                    # reconstructed from a workspace has no `eigenmode_qois` until its
                    # nominal solve is reloaded, and the UQ mean and spread are still
                    # worth plotting without it -- so skip the overlay rather than
                    # raising KeyError on the missing column.
                    if not df_nominal.empty and _bare(metric) in df_nominal.columns:
                        ax.scatter(df_nominal.index, df_nominal[_bare(metric)],
                                   facecolor='none', label='Design Point', ec='k',
                                   lw=1, s=75, zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_xlabel(LABELS.get(_bare(metric), _bare(metric)))

            h, l = ax.get_legend_handles_labels()

        # Set legend
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_fm_scatter.png")

        return axd

    def _all_scatter_impl(self, opt, ncols=3, uq=False):
        """
        Plot scatter chart of fundamental mode quantities of interest.

        Parameters
        ----------
        ncols : int, optional
            Number of columns for the legend. The default is 3.
        uq : bool, optional
            If True, plots with uncertainty quantification. The default is False.

        Returns
        -------
        axd : dict
            A dictionary of axes from the scatter plot.
        """
        plt.rcParams["figure.figsize"] = (16, 3)

        if not uq:
            self.hom_results = self.qois_hom(opt)

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index[i], df[key].iloc[i], color=colors[i % len(colors)], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:
            # get nominal qois
            dd_nominal = {}
            for cav, ops_id in self.wakefield_qois_op.items():
                for kk, vv in ops_id.items():
                    if kk == self._primary_tag(opt):
                        dd_nominal[cav] = vv

            dict_all_nominal = {key: {**self.eigenmode_qois.get(key, {}), **dd_nominal.get(key, {})} for key in
                                set(self.eigenmode_config) | set(dd_nominal)}
            df_nominal = pd.DataFrame.from_dict(dict_all_nominal).T

            dict_all = self.uq_fm_results | self.uq_hom_results

            # Step 1: Flatten the dictionary into a DataFrame
            rows = []
            for cavity, metrics in self.uq_fm_results.items():
                for metric, values in metrics.items():
                    rows.append({
                        'cavity': cavity,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            for cavity, metrics in self.uq_hom_results.items():
                for metric, values in _primary(metrics[opt]).items():
                    rows.append({
                        'cavity': cavity,
                        'metric': metric,
                        'mean': values['expe'][0],
                        'std': values['stdDev'][0]
                    })

            df = pd.DataFrame(rows)

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            # Plotting labels and colors
            labels = df['cavity'].unique()
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]

            # Step 3: Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                    scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                ec='k', zorder=100, label=label)
                    ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], fmt='o', capsize=10,
                                color=scatter_points.get_facecolor()[0])

                    # plot nominal
                    ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none',
                               ec='k', lw=2,
                               zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_ylabel(LABELS[metric])

            # Step 4: Set legend
            h, l = ax.get_legend_handles_labels()

        # by_label = dict(zip(l, h))
        # if 'nominal' in by_label.keys():
        #     nominal_handle = by_label.pop('nominal')
        #     # Reinsert 'nominal' as the last entry
        #     by_label['nominal'] = nominal_handle

        if not ncols:
            ncols = min(4, len(labels))

        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_all_scatter_{opt}.png")

        return axd

    def _all_bar_impl(self, opt, ncols=3):
        """
        Plot bar chart of fundamental mode quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (15, 3)
        # plot barchart
        self.all_results = self.qois_all(opt)

        df = pd.DataFrame.from_dict(self.all_results)
        fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

        labels = [cav.plot_label for cav in self.cavities_list]
        # Plot each column in a separate subplot
        for key, ax in axd.items():
            ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
                   edgecolor='k', width=1)
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_ylabel(key)
            h, l = ax.get_legend_handles_labels()

        # fig.set_tight_layout(True)
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(h, l, loc='outside upper center', borderaxespad=0, ncol=ncols)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_all_bar_{opt}.png")

        return axd

    def plot_cavities_contour(self, opt='mid', figsize=None):
        """Plot geometric contour of Cavity objects

        Parameters
        ----------
        opt: {"mid", "end", "all"}
            Either plot contour for only mid cells or end cells or the entire cavity

        Returns
        -------

        """
        min_x, max_x, min_y, max_y = [], [], [], []

        if figsize:
            fig, axs = plt.subplot_mosaic([[0]], figsize=figsize,
                                          layout='constrained')
        else:
            fig, axs = plt.subplot_mosaic([[0]], figsize=(12, 4),
                                          layout='constrained')
        ax = axs[0]
        ax.set_aspect('equal', adjustable='box')
        ax.margins(x=0)

        for i, cav in enumerate(self.cavities_list):
            if opt.lower() == 'mid':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['IC'])
                beampipe = 'none'
            elif opt.lower() == 'end':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'right'
            else:
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['OC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'both'

            if cav.freq:
                scale = (cav.freq * 1e6) / c0
            else:
                scale = 1
            if cav.cell_parameterisation == 'flattop':
                write_cavity_geometry_cli_flattop(mid_cell, end_cell_left, end_cell_right,
                                                  BP=beampipe, n_cell=1, ax=ax, scale=scale, plot=True, contour=True,
                                                  lw=4, c=cav.color)
            else:
                write_cavity_geometry_cli(mid_cell, end_cell_left, end_cell_right,
                                          BP=beampipe, n_cell=1, ax=ax, scale=scale, plot=True, contour=True,
                                          lw=4, c=cav.color)
            ax.lines[-1].set_label(cav.name)
            ax.axvline(0, c='k', ls='--')
            ax.legend(loc='upper right')

            x_label = r"$z/\lambda$"
            y_label = r"$r/\lambda$"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            min_x.append(min(ax.lines[-1].get_xdata()))
            min_y.append(min(ax.lines[-1].get_ydata()))
            max_x.append(max(ax.lines[-1].get_xdata()))
            max_y.append(max(ax.lines[-1].get_ydata()))

        ax.set_ylim(0)

        # plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_contour_{opt}.png")

        # fig.show()

    # -- Cross-domain comparison plots (span eigenmode + wakefield) ----------
    # Renamed from plot_compare_all_* — these overlay both fundamental-mode and
    # HOM/power QOIs, so they live at the Study level rather than under a single
    # result namespace. Per-domain comparisons live under study.eigenmode.* /
    # study.wakefield.*.
    def plot_all_scatter(self, opt, ncols=3, uq=False, show=True, tuned='auto'):
        """Scatter of fundamental-mode + HOM/power QOIs across every cavity
        (``uq=True``: UQ mean with a +/-1 sigma error bar). A cavity's current
        tuned version stands in for it when it has the results; ``tuned=False``
        compares every cavity untuned."""
        view = self._comparison_view('all', tuned, uq=uq)
        r = view._all_scatter_impl(opt, ncols=ncols, uq=uq)
        _maybe_show(show)
        return r

    def plot_all_bar(self, opt, ncols=3, show=True, tuned='auto'):
        """Bar chart of fundamental-mode + HOM/power QOIs across every cavity; a
        cavity's current tuned version stands in for it when it has the results
        (``tuned=False``: every cavity untuned)."""
        view = self._comparison_view('all', tuned)
        r = view._all_bar_impl(opt, ncols=ncols)
        _maybe_show(show)
        return r

    def plot_cavities_contour_dimension(self, opt='mid', figsize=None):
        """Plot geometric contour of Cavity objects

        Parameters
        ----------
        opt: {"mid", "end", "all"}
            Either plot contour for only mid cells or end cells or the entire cavity

        Returns
        -------

        """
        min_x, max_x, min_y, max_y = [], [], [], []

        if figsize:
            fig, axs = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], figsize=figsize,
                                          layout='constrained')
        else:
            fig, axs = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], figsize=(12, 4),
                                          layout='constrained')

        for i, (cav, key) in enumerate(zip(self.cavities_list, axs)):
            ax = axs[key]
            ax.set_aspect('equal')
            if opt.lower() == 'mid':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['IC'])
                beampipe = 'none'
            elif opt.lower() == 'end':
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['IC'])
                end_cell_right = np.array(cav.shape['OC'])
                beampipe = 'right'
            else:
                mid_cell = np.array(cav.shape['IC'])
                end_cell_left = np.array(cav.shape['OC'])
                end_cell_right = np.array(cav.shape['OC_R'])
                beampipe = 'both'

            if cav.cell_parameterisation == 'flattop':
                write_cavity_geometry_cli_flattop(mid_cell * 1e3, end_cell_left * 1e3, end_cell_right * 1e3,
                                                  BP=beampipe, n_cell=1, ax=ax, scale=1, plot=True,
                                                  dimension=True, lw=3, c='k')
            else:
                write_cavity_geometry_cli(mid_cell * 1e3, end_cell_left * 1e3, end_cell_right * 1e3,
                                          BP=beampipe, n_cell=1, ax=ax, scale=1, plot=True,
                                          dimension=True, lw=3, c='k')
            ax.lines[-1].set_label(cav.name)
            ax.legend(loc='upper right')

            x_label = r"$z$ [mm]"
            y_label = r"$r$ [mm]"
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            min_x.append(min(ax.lines[-1].get_xdata()))
            min_y.append(min(ax.lines[-1].get_ydata()))
            max_x.append(max(ax.lines[-1].get_xdata()))
            max_y.append(max(ax.lines[-1].get_ydata()))

            ax.set_ylim(0)

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_dimension.png")

        # fig.show()

    def plot_axis_fields(self):
        """
        Plot axis fields of cavities

        Returns
        -------

        """
        for cav in self.cavities_list:
            # normalize fields
            e_axis = np.abs(cav.Ez_0_abs['|Ez(0, 0)|'])
            e_axis_norm = e_axis / e_axis.max()

            # shift to mid
            z = cav.Ez_0_abs['z(0, 0)']
            z_shift = z - z.max() / 2
            plt.plot(z_shift, e_axis_norm, label=cav.name)

        plt.xlabel('$z$ [mm]')
        plt.ylabel(r'$|E_\mathrm{0, z}|/|E_\mathrm{0, z}|_\mathrm{max}$')
        plt.axhline(1.02, c='k')
        plt.ylim(-0.01, 1.5)
        plt.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_axis_fields.png")

        plt.show()

    def plot_multipac_triplot(self, folders, kind='triplot'):
        """
        Plot Multipac triplot

        Parameters
        ----------
        folders: list, array like
            List of folder to read multipacting results from
        kind

        Notes
        -----
        This will be changed later so that the multipac results will be in the same location as the SLANS and ABCI
        results

        Returns
        -------

        """

        if kind == 'triplot':
            # create figure
            fig = plt.figure()
            gs = fig.add_gridspec(3, 1)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[2, 0])
            axs = [ax1, ax2, ax3]

        else:
            fig, axs = plt.subplots(1, 1)
            axs = [axs]

        mpl.rcParams['figure.figsize'] = [6, 10]

        Eacc_list = [cav.op_field * 1e-6 for cav in self.cavities_list]
        Epk_Eacc_list = [_qoi(cav, 'Epk/Eacc []') for cav in self.cavities_list]
        labels = [cav.name for cav in self.cavities_list]
        for Eacc, Epk_Eacc, folder, label in zip(Eacc_list, Epk_Eacc_list, folders, labels):
            # load_output_data
            # files
            fnames = ["Ccounter.mat", "Acounter.mat", "Atcounter.mat", "Efcounter.mat", "param",
                      "geodata.n", "secy1", "counter_flevels.mat", "counter_initials.mat"]
            data = {}
            # files_folder = "D:\Dropbox\multipacting\MPGUI21"
            for f in fnames:
                if ".mat" in f:
                    data[f] = spio.loadmat(fr"{folder}\\{f}")
                else:
                    data[f] = pd.read_csv(fr"{folder}\\{f}", sep='\\s+', header=None)

            A = data["Acounter.mat"]["A"]
            At = data["Atcounter.mat"]["At"]
            C = data["Ccounter.mat"]["C"]
            Ef = data["Efcounter.mat"]["Ef"]
            flevel = data["counter_flevels.mat"]["flevel"]
            initials = data["counter_initials.mat"]["initials"]
            secy1 = data["secy1"].to_numpy()
            Pow = flevel
            n = len(initials[:, 0]) / 2  # number of initials in the bright set
            N = int(data["param"].to_numpy()[4])  # number of impacts
            U = flevel
            Efl = flevel
            q = q0
            Efq = Ef / q

            e1 = np.min(np.where(secy1[:, 1] >= 1))  # lower threshold
            e2 = np.max(np.where(secy1[:, 1] >= 1))  # upper threshold
            val, e3 = np.max(secy1[:, 1]), np.argmax(secy1[:, 1])  # maximum secondary yield

            cl = 0
            ok, ok1, ok2 = 1, 1, 1
            if ok > 0:
                if n == 0:
                    error('Unable to plot the counters. No initial points.')
                    return

                if ok1 * ok2 == 0:
                    cl = error('Counter functions or impact energy missing.')
                else:
                    # if ss > 0:
                    #     cl = error(np.array(['Plotting the triplot (counter, enhanced ', 'counter and impact energy).']))

                    if kind == 'counter function' or kind == 'triplot':
                        # fig, axs = plt.subplots(3)
                        axs[0].plot(Efl / 1e6, C / n, lw=2, label=label)
                        axs[0].set_ylabel("$c_" + "{" + f"{N}" + "}/ c_0 $")
                        axs[0].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[0].set_title(r'$\mathbf{MultiPac 2.1~~~~~Counter function~~~~}$')
                        axs[0].set_xlim(np.amin(Efl) / 1e6, np.amax(Efl) / 1e6)
                        axs[0].set_ylim(0, np.max([0.1, axs[0].get_ylim()[1]]))

                        # plot peak operating field
                        axs[0].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[0].text(np.round(Eacc * Epk_Eacc, 2) - 1.5, 0.1,
                                    f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                    size=12, rotation=90,
                                    transform=axs[0].get_xaxis_transform())

                        axs[0].minorticks_on()
                        axs[0].legend(loc='upper left')

                    if kind == 'final impact energy' or kind == 'triplot':
                        s = 0
                        if kind == 'final impact energy':
                            s = 1
                        axs[1 - s].semilogy(Efl / 1e6, Efq, lw=2, label=label)

                        # axs[1-s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e1, 0], secy1[e1, 0]], '-r')
                        e0 = sci.interp1d(secy1[0:e1 + 1, 1], secy1[0:e1 + 1, 0])(1)
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [e0, e0], '-r')
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e2, 0], secy1[e2, 0]], '-r')
                        axs[1 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [secy1[e3, 0], secy1[e3, 0]], '--r')

                        axs[1 - s].set_ylabel("$Ef_" + "{" + f"{N}" + "}$")
                        axs[1 - s].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[1-s].set_title('$\mathbf{Final~Impact~Energy~in~eV}$')
                        axs[1 - s].set_xlim(np.min(Efl) / 1e6, np.max(Efl) / 1e6)
                        axs[1 - s].set_ylim(0, axs[1 - s].get_ylim()[1])

                        axs[1 - s].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[1 - s].text(np.round(Eacc * Epk_Eacc, 2) - 1.5, 0.1,
                                        f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                        size=12, rotation=90,
                                        transform=axs[1 - s].get_xaxis_transform())

                        axs[1 - s].minorticks_on()
                        axs[1 - s].legend(loc='upper left')
                    if kind == 'enhanced counter function' or kind == 'triplot':
                        s = 0
                        if kind == 'enhanced counter function':
                            s = 2
                        axs[2 - s].semilogy(Efl / 1e6, (A + 1) / n, lw=2, label=label)
                        axs[2 - s].set_xlabel('$V$ [MV]')
                        axs[2 - s].plot([np.min(Efl) / 1e6, np.max(Efl) / 1e6], [1, 1], '-r')
                        axs[2 - s].set_xlim(np.min(Efl) / 1e6, np.max(Efl) / 1e6)
                        axs[2 - s].set_ylim(np.min((A + 1) / n), axs[2 - s].get_ylim()[1])
                        axs[2 - s].set_ylabel("$e_" + "{" + f"{N}" + "}" + "/ c_0$")
                        axs[2 - s].set_xlabel(r'$E_\mathrm{pk}$ [MV/m]')
                        # axs[2-s].set_title('$\mathbf{Enhanced~counter~function}$')

                        axs[2 - s].axvline(Eacc * Epk_Eacc, c='k', ls='--', lw=2)
                        axs[2 - s].text(np.round(Eacc * Epk_Eacc, 2) - 1, 0.1,
                                        f"{label[0]}: {np.round(Eacc * Epk_Eacc, 2)} MV/m",
                                        size=12, rotation=90,
                                        transform=axs[2 - s].get_xaxis_transform())

                        axs[2 - s].minorticks_on()
                        axs[2 - s].legend(loc='upper left')

        fig.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_{kind.replace(' ', '_')}.png")

        plt.show()

        # plt.plot(px, py)

    def make_latex_summary_tables(self, which='geometry', op_pts_list=None):
        fname = [cav.name for cav in self.cavities_list]
        # create new sub directory
        if not os.path.exists(fr"{self.projectDir}\PostprocessingData\Data\{'_'.join(fname)}"):
            os.mkdir(fr"{self.projectDir}\PostprocessingData\Data\{'_'.join(fname)}")
        try:
            if which == 'geometry':
                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters of " + fr"{', '.join(fname)}" + " cavity geometries.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{''.join(['c' for i in self.cavities_list])}" + "|}"
                toprule = r"\toprule"
                header = r" ".join([fr"& {cav.name} " for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"
                A = r"$A$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][0], 2)}/{round(cav.shape['OC'][0], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                B = r"$B$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][1], 2)}/{round(cav.shape['OC'][1], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                a = r"$a$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][2], 2)}/{round(cav.shape['OC'][2], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                b = r"$b$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][3], 2)}/{round(cav.shape['OC'][3], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Ri = r"$R_\mathrm{i}$ " + "".join(
                    [fr"& {round(cav.shape['IC'][4], 2)}/{round(cav.shape['OC'][4], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                L = r"$L$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][5], 2)}/{round(cav.shape['OC'][5], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Req = r"$R_\mathrm{eq}$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][6], 2)}/{round(cav.shape['OC'][6], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                alpha = r"$ \alpha [^\circ]$" + "".join(
                    [fr"& {round(cav.shape['IC'][7], 2)}/{round(cav.shape['OC'][7], 2)} " for cav in
                     self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} geometric properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header,
                                     hhline,
                                     A, B, a, b, Ri, L, Req, alpha,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_geom_latex.tex", 'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'fm':
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{FM QoIs of " + fr"{', '.join(fname)}" + " cavity geometries.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{''.join(['c' for i in self.cavities_list])}" + "|}"
                toprule = r"\toprule"
                header = r" ".join([fr"& {cav.name} " for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. [MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for cav in self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~[\Omega$] " + "".join([fr"& {round(_qoi(cav, 'R/Q [Ohm]'), 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(_qoi(cav, 'G [Ohm]'), 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'GR/Q [Ohm^2]') * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'kcc [%]'), 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(_qoi(cav, 'Epk/Eacc []'), 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'Bpk/Eacc [mT/MV/m]'), 2)} " for cav in self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} fm properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header,
                                     hhline,
                                     rf_freq, R_Q, G, GR_Q, kcc, epk, bpk,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)
                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_fm_latex.tex", 'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'hom':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())
                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{HOM QoIs of " + fr"{', '.join(fname)}" + " cavity geometries for the " + fr"{', '.join([LABELS[op_pt] for op_pt in op_pts_list])}" + " operating points.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c|}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                kfm = r"$|k_\mathrm{FM}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} hom properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_hom_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'qois':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters and QoIs of cavities.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. ~[MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for cav in self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~\mathrm{[\Omega$]} " + "".join(
                    [fr"& {round(_qoi(cav, 'R/Q [Ohm]'), 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(_qoi(cav, 'G [Ohm]'), 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'GR/Q [Ohm^2]') * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'kcc [%]'), 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(_qoi(cav, 'Epk/Eacc []'), 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'Bpk/Eacc [mT/MV/m]'), 2)} " for cav in self.cavities_list]) + r" \\"

                kfm = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}~[]$ " + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ ~[kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * _primary(cav.rf_performance_qois[op_pt])['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: selected shape}"
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     rf_freq, R_Q, G, GR_Q, kcc, epk, bpk,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hhline, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom, PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_qois_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            elif which == 'power':
                if op_pts_list is None:
                    op_pts_list = list(self.operating_points.keys())

                fname = [cav.name for cav in self.cavities_list]
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Static, dynamic and HOM power loss and input power per cavity for " + fr"{', '.join(fname)}" + " cavity geometries for the " + fr"{', '.join([LABELS[op_pt] for op_pt in op_pts_list])}" + " operating points.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c|}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"

                rf_freq = r"RF Freq. [MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"

                if 'V [GV]' in self.rf_config.keys():
                    Vrf = r"$V_\mathrm{RF}$ [GV] " + "".join(
                        [fr"& {round(self.rf_config['V [GV]'][ii], 2):.2f} " for ii in range(len(op_pts_list)) for _ in
                         self.cavities_list]) + r" \\"
                else:
                    Vrf = r"$V_\mathrm{RF}$ [GV] " + "".join(
                        [fr"& {round(self.operating_points[op_pt]['V [GV]'], 2):.2f} " for op_pt in op_pts_list for _ in
                         self.cavities_list]) + r" \\"

                Eacc = r"$E_\mathrm{acc}$ [MV/m] " + "".join(
                    [fr"& {round(cav.Eacc_rf_config, 2)} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$~[]" + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} ~[\mathrm{kW}]$ " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * _primary(cav.rf_performance_qois[op_pt])['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                fname = '_'.join(fname)
                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: " + fr"{fname} power properties" + '}'
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline, rf_freq, Vrf, Eacc, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom,
                                     # PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_power_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')
            else:
                l1 = r"\begin{table}[htb!]"
                l2 = r"\centering"
                l3 = r"\caption{Geometric parameters and QoIs of cavities.}"
                l4 = r"\resizebox{\ifdim\width>\columnwidth \columnwidth \else \width \fi}{!}{\begin{tabular}{|l|" + f"{'|'.join([''.join(['c' for i in self.cavities_list]) for _ in range(len(op_pts_list))])}" + "|}"
                toprule = r"\toprule"
                header1 = r"&\multicolumn{" + r" &\multicolumn{".join(
                    [fr'{len(self.cavities_list)}' + r'}{c}{' + fr"{LABELS[op_pt]}" + '}' for op_pt in
                     op_pts_list]) + r" \\"
                header2 = r" ".join([fr"& {cav.name} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                hline = r"\hline"
                hhline = r"\hline \hline"
                A = r"$A$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][0], 2)}/{round(cav.shape['OC'][0], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                B = r"$B$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][1], 2)}/{round(cav.shape['OC'][1], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                a = r"$a$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][2], 2)}/{round(cav.shape['OC'][2], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                b = r"$b$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][3], 2)}/{round(cav.shape['OC'][3], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Ri = r"$R_\mathrm{i}$ " + "".join(
                    [fr"& {round(cav.shape['IC'][4], 2)}/{round(cav.shape['OC'][4], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                L = r"$L$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][5], 2)}/{round(cav.shape['OC'][5], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                Req = r"$R_\mathrm{eq}$ [mm] " + "".join(
                    [fr"& {round(cav.shape['IC'][6], 2)}/{round(cav.shape['OC'][6], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                alpha = r"$ \alpha [^\circ]$" + "".join(
                    [fr"& {round(cav.shape['IC'][7], 2)}/{round(cav.shape['OC'][7], 2)} " for cav in
                     self.cavities_list]) + r" \\"
                rf_freq = r"RF Freq. ~[MHz] " + "".join(
                    [fr"& {round(cav.freq, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                Vrf = r"$V_\mathrm{RF}$ ~[GV] " + "".join(
                    [fr"& {round(self.operating_points[op_pt]['V [GV]'], 2):.2f} " for op_pt in op_pts_list for _ in
                     self.cavities_list]) + r" \\"

                Eacc = r"$E_\mathrm{acc}$ [MV/m] " + "".join(
                    [fr"& {round(self.operating_points[op_pt]['Eacc [MV/m]'], 2)} " for op_pt in op_pts_list for _ in
                     self.cavities_list]) + r" \\"
                R_Q = r"$R/Q ~[\Omega$] " + "".join(
                    [fr"& {round(_qoi(cav, 'R/Q [Ohm]'), 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                G = r"$G$ [$\Omega$] " + "".join(
                    [fr"& {round(_qoi(cav, 'G [Ohm]'), 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'GR/Q [Ohm^2]') * 1e-4, 2)} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ [] " + "".join(
                    [fr"& {round(_qoi(cav, 'Epk/Eacc []'), 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} \left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(_qoi(cav, 'Bpk/Eacc [mT/MV/m]'), 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"

                kfm = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_fm, 4)) for k_fm in [vv for kk, vv in cav.k_fm.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in [vv for kk, vv in cav.k_loss.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                    [
                        fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in [vv for kk, vv in cav.k_kick.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # kloss = r"$|k_\mathrm{\parallel}|$ [V/pC]" + "".join(
                #     [fr"& {'/'.join([str(round(k_loss, 4)) for k_loss in cav.k_loss])} " for cav in
                #      self.cavities_list]) + r" \\"
                #
                # kkick = r"$k_\mathrm{\perp}$ [V/pC/m]" + "".join(
                #     [fr"& {'/'.join([str(round(k_kick, 4)) for k_kick in cav.k_kick])} " for cav in
                #      self.cavities_list]) + r" \\"

                Ncav = r"$N_\mathrm{cav}$ " + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$ " + "".join(
                    [fr"& {_primary(cav.rf_performance_qois[op_pt])['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(_primary(cav.rf_performance_qois[op_pt])['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                #     [fr"& {'/'.join([str(round(phom, 4)) for phom in cav.phom])} " for cav in self.cavities_list]) + r" \\"

                phom_values = np.array([[phom for phom in cav.phom] for cav in self.cavities_list])

                # if len(phom_values) % 2 == 0:
                #     result_array = phom_values[1::2] - phom_values[::2]
                #
                #     Phom_diff = r"$\Delta P_\mathrm{HOM}$/cav [W] & \multicolumn{2}{c|}{" + " & \multicolumn{2}{c|}{".join(
                #         [fr"{'/'.join([str(round(val * 1e3, 2)) for val in arr])} " + '}' for arr in result_array]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * _primary(cav.rf_performance_qois[op_pt])['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                # PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                #     [fr"& {'/'.join([str(round(phom * cav.n_cav_op_field, 2)) for phom in cav.phom])} " for cav in
                #      self.cavities_list]) + r" \\"

                bottomrule = r"\bottomrule"
                l34 = r"\end{tabular}}"
                l35 = r"\label{tab: selected shape}"
                l36 = r"\end{table}"

                self.latex_output = (l1, l2, l3, l4,
                                     toprule, hline,
                                     header1,
                                     hhline,
                                     header2,
                                     hhline,
                                     A, B, a, b, Ri, L, Req, alpha,
                                     hhline,
                                     rf_freq, Vrf, Eacc, R_Q, G, GR_Q, epk, bpk,
                                     hhline,
                                     kfm, kloss, kkick, Phom,
                                     hhline, Ncav, Q0, Pin, Pstat, Pdyn, Pwp, Phom, PHOM,
                                     hline,
                                     bottomrule,
                                     l34, l35, l36)

                # save plots
                fname = [cav.name for cav in self.cavities_list]
                fname = '_'.join(fname)
                with open(fr"{self.projectDir}\PostprocessingData\Data\{fname}\{fname}_all_{op_pts_list}_latex.tex",
                          'w') as f:
                    for ll in self.latex_output:
                        f.write(ll + '\n')

        except KeyError as e:
            error(f"A cavity has no eigenmode or wakefield result for {e}. Run "
                  f"study.run_eigenmode(...) and study.run_wakefield(...) first.")

    def save_all_plots(self, plot_name):
        """Save the current figure into this study's project folder, under
        ``<projectDir>/PostProcessingData/Plots/<cavity names>/``.

        Returns the path written, or ``None`` when there is no project folder to
        write into.

        A comparison plot is a result, so it belongs beside the simulation it came
        from. It must therefore never be written to the *working* directory: a
        cavity that has no workspace used to resolve ``projectDir`` to ``'.'``, and
        plotting it dropped a ``PostProcessingData/`` tree wherever the interpreter
        happened to be started -- including, once, into the repository itself.
        """
        if not self.projectDir:
            return None
        project = os.path.abspath(str(self.projectDir))
        if project == os.path.abspath(os.getcwd()):
            info('save_all_plots: no project folder for this study (it resolves to '
                 'the working directory), so the figure was not saved. Give the '
                 'cavity a workspace with cav.set_workspace(...) to keep it.')
            return None

        fname = '_'.join(cav.name for cav in self.cavities_list)
        # makedirs, not mkdir: the old code created the leaf only when the whole
        # tree was absent, so a half-existing tree raised. Casing is consistent
        # now too -- it used to mix 'PostProcessingData' and 'PostprocessingData',
        # which are the same folder on Windows and two different ones elsewhere.
        save_folder = os.path.join(project, 'PostProcessingData', 'Plots', fname)
        os.makedirs(save_folder, exist_ok=True)
        path = os.path.join(save_folder, plot_name)
        plt.savefig(path, dpi=300)
        return path

    def calc_limits(self, which, selection):
        if self.operating_points is not None:
            cols = selection
            E0 = [self.operating_points[col]["E [GeV]"] for col in cols]
            nu_s = [self.operating_points[col]["nu_s []"] for col in cols]
            I0 = [self.operating_points[col]["I0 [mA]"] for col in cols]
            alpha_c = [self.operating_points[col]["alpha_p [1e-5]"] for col in cols]
            tau_z = [self.operating_points[col]["tau_z [ms]"] for col in cols]
            tau_xy = [self.operating_points[col]["tau_xy [ms]"] for col in cols]
            f_rev = [self.operating_points[col]["f_rev [kHz]"] for col in cols]
            beta_xy = [self.operating_points[col]["beta_xy [m]"] for col in cols]
            n_cav = [self.operating_points[col]["N_c []"] for col in cols]

            unit = {'MHz': 1e6,
                    'GHz': 1e9}

            Z_list, ZT_le = [], []
            if which == 'longitudinal':
                f_list = np.linspace(0, 10000, num=1000) * 1e6
                Z_le = []
                try:
                    for i, n in enumerate(n_cav):
                        Z = [(2 * E0[i] * 1e9 * nu_s[i])
                             / (n * I0[i] * 1e-3 * alpha_c[i] * 1e-5 * tau_z[i] * 1e-3 * f)
                             if f > 1e-8 else 1e5 for f in f_list]

                        Z_le.append(np.round((2 * E0[i] * 1e9 * nu_s[i])
                                             / (n * I0[i] * 1e-3 * alpha_c[i] * 1e-5
                                                * tau_z[i] * 1e-3) * 1e-9 * 1e-3, 2))

                        Z_list.append(np.array(Z) * 1e-3)  # convert to kOhm

                except ZeroDivisionError:
                    error("ZeroDivisionError, check input")
                return f_list * 1e-6, Z_list, cols

            elif which == 'transversal':

                f_list = np.linspace(0, 10000, num=1000) * 1e6
                try:
                    # print(E0, beta_xy, tau_xy, n_cav, f_rev, I0)
                    for i, n in enumerate(n_cav):
                        # print(E0[i], beta_xy[i], tau_xy[i], n_cav[i], f_rev[i], I0[i])
                        ZT = (2 * E0[i]) * 1e9 / (n * I0[i] * 1e-3 * beta_xy[i] * tau_xy[i] * 1e-3 * f_rev[i] * 1e3)
                        ZT_le.append(np.round(ZT * 1e-3, 2))
                        Z_list.append(np.array(ZT) * 1e-3)  # convert to kOhm/m

                except Exception as e:
                    error("ZeroDivisionError, check input", e)

                return f_list * 1e-6, Z_list, cols
        else:
            error("Please load a valid operating point(s) file.")

    def plot_thresholds(self, which, selection, ax=None, ncav_mod=1):
        labels_info = []

        def update_text_positions(event):
            """Callback function to update the text positions based on current limits."""
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for info_ in labels_info:
                # Check if the label is out of bounds
                if not (xlim[0] <= info_['x'] <= xlim[1]) or not (ylim[0] <= info_['y'] <= ylim[1]):
                    # Reposition the label to the center of the current view
                    axes_coord = (0.01, 0.5)
                    # Transform axes coordinate to display coordinate
                    display_coord = ax.transAxes.transform(axes_coord)
                    # Transform display coordinate to data coordinate
                    data_coord = ax.transData.inverted().transform(display_coord)
                    new_x = data_coord[0]
                    new_y = info_['line'].get_ydata()[np.abs(info_['line'].get_xdata() - new_x).argmin()]
                    info_['text_obj'].set_position((new_x, new_y))

        self.threshold_texts_objects = []
        if ax is None:
            fig, ax = plt.subplots()
            ax.margins(x=0)

        if which.lower() in 'longitudinal':
            # calculate limits
            f_list, Z_list, labels = self.calc_limits('longitudinal', selection)

            # plot baselines
            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                aa, = ax.plot(f_list, ncav_mod * z, ls='--', c='k')
                labels_info.append({'line': aa})

            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                pos = axis_data_coords_sys_transform(ax, 0.01, 0.5)
                indx = np.argmin(abs(f_list - pos[0]))
                # x, y = axis_data_coords_sys_transform(ax, f_list[indx], z[indx], True)
                txt = LABELS[lab]

                ab = ax.text(f_list[indx], ncav_mod * z[indx], txt, color='k', fontsize=12, ha='left',
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                labels_info[i] = labels_info[i] | {'x': f_list[indx], 'y': ncav_mod * z[indx], 'label': txt,
                                                   'text_obj': ab}
                # ab = add_text(ax, txt, box="Square", xy=(x, y), xycoords='axes fraction', size=12)
                self.threshold_texts_objects.append(ab)

        else:
            # calculate limits
            f_list, Z_list, labels = self.calc_limits('transversal', selection)

            # plot baselines
            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                aa = ax.axhline(ncav_mod * z, ls='--', c='k')
                labels_info.append({'line': aa})

            for i, (z, lab) in enumerate(zip(Z_list, labels)):
                pos = axis_data_coords_sys_transform(ax, 0.01, 0.5)
                indx = np.argmin(abs(f_list - pos[0]))
                # x, y = axis_data_coords_sys_transform(ax, f_list[indx], z, True)

                txt = LABELS[lab]

                ab = ax.text(f_list[indx], ncav_mod * z, txt, color='k', fontsize=12, ha='left',
                             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
                labels_info[i] = labels_info[i] | {'x': f_list[indx], 'y': ncav_mod * z, 'label': txt, 'text_obj': ab}
                self.threshold_texts_objects.append(ab)

        # Attach the callback to limit changes
        ax.callbacks.connect('xlim_changed', update_text_positions)
        ax.callbacks.connect('ylim_changed', update_text_positions)

    def _adjust_texts(self, texts, ax, separation=0.1, iterations=20):
        renderer = ax.figure.canvas.get_renderer()

        def get_text_bbox(text):
            bbox = text.get_window_extent(renderer=renderer)
            bbox_axes_coords = bbox.transformed(ax.transAxes.inverted())
            return bbox_axes_coords

        def calculate_adjustment(bbox1, bbox2):
            center1 = np.array([bbox1.x0 + bbox1.width / 2, bbox1.y0 + bbox1.height / 2])
            center2 = np.array([bbox2.x0 + bbox2.width / 2, bbox2.y0 + bbox2.height / 2])

            pos1 = np.array(text1.get_position())
            pos2 = np.array(text2.get_position())

            if bbox1.overlaps(bbox2):
                overlap_x = max(0, min(bbox1.x1, bbox2.x1) - max(bbox1.x0, bbox2.x0))
                overlap_y = max(0, min(bbox1.y1, bbox2.y1) - max(bbox1.y0, bbox2.y0))

                if overlap_x > 0:
                    midpoint_x = (center1[0] + center2[0]) / 2
                    if bbox1.x1 > bbox2.x1:
                        pos1[0] = midpoint_x + bbox1.width / 2 + separation
                        pos2[0] = midpoint_x - bbox2.width / 2 - separation
                    else:
                        pos1[0] = midpoint_x - bbox1.width / 2 - separation
                        pos2[0] = midpoint_x + bbox2.width / 2 + separation

                # if overlap_y > 0:
                #     midpoint_y = (center1[1] + center2[1]) / 2
                #     if bbox1.y1 > bbox2.y1:
                #         pos1[1] = midpoint_y + bbox1.height / 2 + separation
                #         pos2[1] = midpoint_y - bbox2.height / 2 - separation
                #     else:
                #         pos1[1] = midpoint_y - bbox1.height / 2 - separation
                #         pos2[1] = midpoint_y + bbox2.height / 2 + separation

            return pos1, pos2

        # Run the adjustment algorithm
        for _ in range(iterations):
            for i, text1 in enumerate(texts):
                bbox1 = get_text_bbox(text1)
                for j, text2 in enumerate(texts):
                    if i == j:
                        continue
                    bbox2 = get_text_bbox(text2)

                    if bbox1.overlaps(bbox2):
                        pos1, pos2 = calculate_adjustment(bbox1, bbox2)
                        text1.set_position((pos1[0], pos1[1]))
                        text2.set_position((pos2[0], pos2[1]))

        # Ensure the texts are within the plot limits
        for text in texts:
            x, y = text.get_position()
            x = np.clip(x, ax.get_xlim()[0], ax.get_xlim()[1])
            y = np.clip(y, ax.get_ylim()[0], ax.get_ylim()[1])
            text.set_position((x, y))

    def plot_cutoff(self, Ri_list, which, ax=None):
        labels_info = []

        def update_text_positions(event):
            """Callback function to update the text positions based on current limits."""
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            for info_ in labels_info:
                # Check if the label is out of bounds
                if not (xlim[0] <= info_['x'] <= xlim[1]) or not (ylim[0] <= info_['y'] <= ylim[1]):
                    # Reposition the label to the center of the current view
                    axes_coord = (0.01, 0.05)
                    # Transform axes coordinate to display coordinate
                    display_coord = ax.transAxes.transform(axes_coord)
                    # Transform display coordinate to data coordinate
                    data_coord = ax.transData.inverted().transform(display_coord)
                    new_y = data_coord[1]
                    new_x = info_['line'].get_xdata()[np.abs(info_['line'].get_ydata() - new_y).argmin()]
                    info_['text_obj'].set_position((new_x, new_y))

        if ax is None:
            fig, ax = plt.subplots()
            ax.margins(x=0)

        f_list = self.calculate_beampipe_cutoff(Ri_list, which)

        i = 0
        for Ri in Ri_list:
            for mode_type, freq in zip(which, f_list[f'{Ri}']):
                vl = ax.axvline(freq, ls='--', c='k', zorder=900)  # label=f"{sc[0]} cutoff (Ri={sc[1]})",
                labels_info.append({'line': vl})

                # get y text position from axis position. Position x is not used
                pos = axis_data_coords_sys_transform(ax, freq, 0.05, inverse=False)

                txt = r"$f_\mathrm{c," + f"{mode_type}" + r"} (R_\mathrm{i} = " + f"{Ri}" + r" ~\mathrm{mm}) $"
                ab = ax.text(freq, pos[1], txt, color='k', fontsize=16, ha='left', rotation=90, zorder=1000)
                labels_info[i] = labels_info[i] | {'x': freq, 'y': pos[1], 'label': txt, 'text_obj': ab}
                i += 1

                # ab = add_text(ax, r"$f_\mathrm{c," + f"{mode_type}" + r"} (R_\mathrm{i} = "
                #               + f"{Ri}" + r" ~\mathrm{mm}) $",
                #               box="None", xy=(freq, pos[1]),
                #               xycoords='data', size=14, rotation=90)

        # Attach the callback to limit changes
        ax.callbacks.connect('xlim_changed', update_text_positions)
        ax.callbacks.connect('ylim_changed', update_text_positions)

    @staticmethod
    def calculate_beampipe_cutoff(Ri_list, which):
        c = c0
        mode_list = {}
        f_list = {}

        mode_dict = {}
        for mode in which:
            mode_type = mode[:2]
            m = int(mode[2:3])
            n = int(mode[3:])
            mode_dict[mode] = {'type': mode_type, 'm': m, 'n': n}

        for Ri in Ri_list:
            f_list[f'{Ri}'] = []
            for mode, indices in mode_dict.items():
                m, n = indices['m'], indices['n']
                # get jacobian
                if mode_dict[mode]['type'].lower() == 'tm':
                    j = jn_zeros(m, n)[n - 1]
                else:
                    j = jnp_zeros(m, n)[n - 1]

                f = c / (2 * np.pi) * (j / (Ri * 1e-3))

                # append to f list
                f_list[f'{Ri}'].append(f * 1e-6)

        return f_list

    @staticmethod
    def calc_cutoff(Ri, mode):
        # calculate frequency from Ri
        p_TM01, p_TE11 = 2.405, 1.841
        c = c0

        if mode == 'TM01':
            freq = (c * p_TM01) / (2 * np.pi * Ri * 1e9) * 1e3
        else:
            freq = (c * p_TE11) / (2 * np.pi * Ri * 1e9) * 1e3

        return freq
