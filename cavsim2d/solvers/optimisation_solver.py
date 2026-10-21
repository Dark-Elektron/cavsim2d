"""The optimisation namespace, ``study.optimisation``."""
import json
import os
from pathlib import Path
from cavsim2d.utils.printing import info, warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.utils.config_validation import canonicalise
from cavsim2d.utils.integrity import confirm
from itertools import combinations
from cavsim2d.solvers._shared import _maybe_show
from cavsim2d.solvers.configs import _ConfigSampleMixin


# ---------------------------------------------------------------------------
# OptimisationSolver
# ---------------------------------------------------------------------------

class OptimisationSolver(_ConfigSampleMixin):
    """Manages ``<project>/optimisation/`` — multi-objective cavity optimisation.

    Access
    ------
    - ``cavs.optimisation.config``
    - ``cavs.optimisation.config_sample``   — a valid template config to edit
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

    _config_kind = 'optimisation'

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
        """The evaluated candidates as a Study, read from disk with their results
        (None before a run). It used to be None always."""
        if self._cavities is None:
            cands = self.candidates_folder
            if cands.is_dir() and any(cands.iterdir()):
                # Deferred: breaks the solver_objects <-> study import cycle.
                from cavsim2d.study import Study
                self._cavities = Study.load(cands)
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

        config = canonicalise(config, 'optimisation')
        # A fresh run clears the previous one's generations and candidates, which
        # can be hours of work: ask first, as every other analysis does.
        previous = self.folder / 'history.csv'
        if not resume and previous.exists() and not confirm(
                f"Optimisation: replace the previous run in {self.folder}? "
                f"(resume=True continues it instead)", force=config.get('force')):
            print("Optimisation: nothing was run; the previous run is kept.")
            return self

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
        # from the bounds — the bare ``Study(dir).run_optimisation(cfg)`` path.
        bounds = config['bounds']
        added = list(getattr(self._parent, 'cavities_list', []) or [])
        if added:
            if len(added) > 1:
                warning(f"Optimisation uses one template cavity; using the first of "
                     f"{len(added)} ({added[0].name}).")
            # Carry any dielectric regions across: rebuild() works from the
            # parameter dict, which does not contain them, so the template would
            # otherwise be a vacuum copy of a loaded cavity. (Note the tuning
            # step inside optimisation is still vacuum-only.)
            template_cav = added[0]._carry_layout_to(added[0]._carry_dielectrics_to(
                added[0].rebuild(added[0].parameters)))
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
            # The beampipe is part of the cavity being optimised, not a detail:
            # a multi-cell assembly without beampipes resonates several MHz away
            # from the same geometry with them (14.4 MHz measured on a 2-cell at
            # 800 MHz), so a silent 'none' default optimises a different object
            # than the one being designed -- and end cells exist precisely to
            # match into the beampipes. Take it from the config; default to
            # 'both' for a multi-cell, which is what a real cavity has.
            beampipe = config.get('beampipe',
                                  'both' if (n_cells or 1) > 1 else 'none')
            template_cav = EllipticalCavity(n_cells=n_cells,
                                            mid_cell=mid_cell_params,
                                            beampipe=beampipe)
            info(f"Optimisation template: {n_cells}-cell, beampipe={beampipe!r} "
                 f"(set config['beampipe'] to override).")

        template_cav.projectDir = self._parent.projectDir
        template_cav.self_dir = str(self.candidates_folder / '_template')
        os.makedirs(template_cav.self_dir, exist_ok=True)

        # Create the optimiser and run
        opt = Optimisation()
        opt.run(template_cav, config, opt_solver=self)

        self._invalidate()
        return self

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
        _maybe_show(show)
        return fig, axes_dict

    def plot_convergence(self, show=True, **kwargs):
        """Plot hypervolume convergence from saved HV history.

        Returns (fig, (ax1, ax2)).
        """
        hv_path = self.folder / 'hv_history.json'
        if not hv_path.exists():
            warning("No HV convergence data saved. Run optimisation first.")
            return None, None
        with open(hv_path, 'r') as f:
            hv_data = json.load(f)
        hv = np.array(hv_data['hv_history'])
        tol = hv_data.get('hv_tol', 1e-9)
        if len(hv) < 2:
            warning("Not enough generations for convergence plot.")
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
        _maybe_show(show)
        return fig, (ax1, ax2)

    # -- Internal plot helpers ------------------------------------------------

    @staticmethod
    def _default_obj_cols(df):
        return [c for c in df.columns if c not in ('key', 'total_rank', 'generation')]

    def _plot_objectives(self, df, obj_cols, kind, normalise, show, ax,
                         title_prefix, generations=None, **kwargs):
        vals, labels = self._obj_data(df, obj_cols)
        if vals is None:
            warning(f"Objective columns {obj_cols} not found in data.")
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
