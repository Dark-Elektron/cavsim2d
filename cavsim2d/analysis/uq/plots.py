"""Comparison plots for uncertainty-quantification results.

The solver namespaces already plot the UQ of a *cavity* (``cav.eigenmode.plot_fm_scatter
(uq=True)`` and friends, which scatter one point per cavity in a study). What they cannot
express is the other axis of comparison: several UQ runs of the **same** cavity that
differ in how the randomness was posed - a per-cell-type tolerance against a per-half-cell
one, a dumb-bell model against a multicup one, a cubature rule against Monte Carlo, or one
tolerance amplitude against another.

:func:`plot_uq_comparison` takes any number of labelled UQ result sets and draws, for each
figure of merit, the mean with a +/-1 sigma error bar. Inputs may be ``uq.json`` paths, the
directories holding them, or the already-loaded dicts.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from cavsim2d.constants import LABELS
from cavsim2d.utils.style import house_style, WARM

__all__ = ['plot_uq_comparison', 'uq_comparison_table']


def _as_uq_dict(src):
    """A UQ result mapping from a dict, a ``uq.json`` path, or its directory."""
    if isinstance(src, dict):
        return src
    p = Path(src)
    if p.is_dir():
        p = p / 'uq.json'
    if not p.exists():
        raise FileNotFoundError(f'no UQ results at {p}')
    return json.loads(p.read_text())


def _bare(key):
    """``'monopole:R/Q [Ohm]'`` -> ``'R/Q [Ohm]'``."""
    return str(key).split(':')[-1].strip()


def _moment(entry, name):
    """``expe``/``stdDev`` are stored as one-element lists; tolerate bare scalars."""
    v = entry.get(name)
    if isinstance(v, (list, tuple, np.ndarray)):
        v = v[0] if len(v) else None
    # 'se_stdDev' is null for a cubature design (its nodes are deterministic, so
    # there is nothing to bootstrap) - that is a legitimate absence, not an error.
    if v is None:
        return np.nan
    return float(v)


def _plottable(results, qois):
    """QOI keys to draw, in a stable order.

    Defaults to those present in EVERY result set and carrying a plot label, so
    bookkeeping entries ('N Cells', 'Normalization Length [mm]') are skipped.
    """
    common = None
    for d in results.values():
        keys = set(d)
        common = keys if common is None else (common & keys)
    common = common or set()
    if qois:
        want = {_bare(q) for q in qois}
        sel = [k for k in common if _bare(k) in want]
        missing = want - {_bare(k) for k in sel}
        if missing:
            raise ValueError(f'not in every UQ result set: {sorted(missing)}')
    else:
        sel = [k for k in common if _bare(k) in LABELS]
    # order by the first result set, so panels follow the solver's own ordering
    order = list(next(iter(results.values())))
    return sorted(sel, key=lambda k: order.index(k) if k in order else 1e9)


def uq_comparison_table(results, qois=None):
    """Tidy ``DataFrame`` of mean and standard deviation per (label, QOI).

    Same inputs as :func:`plot_uq_comparison`. Useful when the numbers are wanted
    alongside - or instead of - the figure.
    """
    import pandas as pd
    results = {k: _as_uq_dict(v) for k, v in results.items()}
    rows = []
    for label, d in results.items():
        for key in _plottable(results, qois):
            rows.append({'label': label, 'quantity': _bare(key),
                         'mean': _moment(d[key], 'expe'),
                         'std': _moment(d[key], 'stdDev'),
                         'se_std': _moment(d[key], 'se_stdDev')})
    return pd.DataFrame(rows)


def plot_uq_comparison(results, qois=None, nominal=None, ncols=4, relative=False,
                       kind='mean', reference=None, figsize=None, capsize=4, show=True):
    """Scatter mean +/- 1 sigma for several labelled UQ result sets, one panel per QOI.

    Parameters
    ----------
    results : dict
        ``{label: source}``, where *source* is a ``uq.json`` path, the directory
        containing one, or an already-loaded dict. Insertion order sets the order
        along each panel's x-axis.
    kind : {'mean', 'sd'}, default 'mean'
        ``'mean'`` plots the ensemble mean with a +/-1 sigma error bar - the usual
        view. ``'sd'`` plots the **standard deviation itself** as the quantity, with
        its own uncertainty (``se_stdDev``) as the bar where the results carry one.
        Use it when the question is how the *spreads* compare: a 20% difference in
        sigma is nearly invisible as a difference in error-bar length, but obvious
        when sigma is what the axis measures.
    reference : str, optional
        With ``kind='sd'``, the label to divide by, turning each panel into a ratio
        against that result set (its own entry becomes 1). Handy for "rule vs
        reference" comparisons.
    qois : list of str, optional
        Figures of merit to draw, qualified (``'monopole:R/Q [Ohm]'``) or bare
        (``'R/Q [Ohm]'``). Default: every labelled QOI present in *all* result sets.
    nominal : dict, optional
        ``{qoi: value}`` (bare or qualified) drawn as a dashed reference line - the
        unperturbed design point.
    relative : bool, default False
        Plot each quantity as a percentage of its *nominal* value rather than in its
        own units. Requires *nominal*, and puts every panel on one comparable scale.
    ncols : int, default 4
        Panels per row.
    capsize : float, default 4
        Error-bar cap width, in points.
    show : bool, default True
        Call ``plt.show()`` before returning.

    Returns
    -------
    numpy.ndarray of matplotlib axes, one per QOI drawn.

    Notes
    -----
    The error bar is the **spread** of the perturbed ensemble (1 sigma), not the
    uncertainty of the estimate. For a sampled design the latter is available as
    ``se_stdDev`` in the UQ results and is typically far smaller; do not read the bar
    as a confidence interval on the mean.
    """
    if not results:
        raise ValueError('results is empty - pass at least one labelled UQ result set')
    if kind not in ('mean', 'sd'):
        raise ValueError("kind must be 'mean' or 'sd'")
    if relative and kind == 'mean' and not nominal:
        raise ValueError("relative=True needs 'nominal' to normalise against")
    if reference is not None and reference not in results:
        raise ValueError(f'reference {reference!r} is not one of {list(results)}')

    results = {k: _as_uq_dict(v) for k, v in results.items()}
    keys = _plottable(results, qois)
    if not keys:
        raise ValueError('no labelled figures of merit common to every result set')

    # A solver's qois.json carries non-numeric bookkeeping too ('polarisation':
    # 'monopole'), so take only what can be a reference line and ignore the rest.
    nom = {}
    for k, v in (nominal or {}).items():
        try:
            nom[_bare(k)] = float(v)
        except (TypeError, ValueError):
            continue

    labels = list(results)
    x = np.arange(len(labels))
    nrows = int(np.ceil(len(keys) / ncols))
    ncols_eff = min(ncols, len(keys))
    figsize = figsize or (3.4 * ncols_eff, 3.1 * nrows)

    with house_style():
        fig, axes = plt.subplots(nrows, ncols_eff, figsize=figsize, squeeze=False)
        flat = axes.ravel()
        for ax, key in zip(flat, keys):
            name = _bare(key)
            mean = np.array([_moment(results[l][key], 'expe') for l in labels])
            std = np.array([_moment(results[l][key], 'stdDev') for l in labels])
            ref = nom.get(name)

            if kind == 'sd':
                # sigma IS the quantity; its own uncertainty is the bar.
                y = std.copy()
                err = np.array([_moment(results[l][key], 'se_stdDev') for l in labels])
                denom = 1.0
                if reference is not None:
                    denom = std[labels.index(reference)]
                    y, err = y / denom, err / denom
                for i, l in enumerate(labels):
                    ax.errorbar(x[i], y[i], yerr=None if not np.isfinite(err[i]) else err[i],
                                fmt='o', ms=7, color=WARM[i % len(WARM)],
                                ecolor=WARM[i % len(WARM)], capsize=capsize,
                                capthick=1.4, elinewidth=1.4,
                                label=l if ax is flat[0] else None)
                if reference is not None:
                    ax.axhline(1.0, color='0.35', ls='--', lw=1, zorder=0)
                    ax.set_ylabel(f'sd / sd({reference})')
                else:
                    ax.set_ylabel(f'sd  {LABELS.get(name, name)}')
            else:
                if relative:
                    mean, std = 100 * mean / ref, 100 * std / ref
                    ref = 100.0
                for i, l in enumerate(labels):
                    ax.errorbar(x[i], mean[i], yerr=std[i], fmt='o', ms=7,
                                color=WARM[i % len(WARM)], ecolor=WARM[i % len(WARM)],
                                capsize=capsize, capthick=1.4, elinewidth=1.4,
                                label=l if ax is flat[0] else None)
                if ref is not None and np.isfinite(ref):
                    ax.axhline(ref, color='0.35', ls='--', lw=1, zorder=0)
                ax.set_ylabel(f'{LABELS.get(name, name)}  [% of nominal]' if relative
                              else LABELS.get(name, name))
            ax.set_title(name, fontsize=10) if kind == 'sd' else None
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha='right')
            ax.set_xlim(-0.6, len(labels) - 0.4)
            ax.grid(alpha=0.3, axis='y')
        for ax in flat[len(keys):]:
            ax.set_visible(False)
        if len(labels) > 1:
            handles, lab = flat[0].get_legend_handles_labels()
            fig.legend(handles, lab, loc='upper center', ncol=min(len(labels), 6),
                       frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout()
        if show:
            plt.show()
    return axes.ravel()[:len(keys)]
