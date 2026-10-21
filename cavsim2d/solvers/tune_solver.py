"""The tuning namespace, ``cav.tune``."""
import json
import os
import tempfile
from pathlib import Path
from cavsim2d.utils.printing import suppress_errors, warning
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cavsim2d.utils.integrity import guard_runs, stamp_matches, warn_if_stale
from cavsim2d.utils.scratch import scratch_folders
from cavsim2d.solvers._shared import _accelerating_mode_row, _maybe_show
from cavsim2d.solvers.configs import (_ConfigSampleMixin, _user_config, DEFAULT_TUNE_CONFIG,
                                      merge_config)


# ---------------------------------------------------------------------------
# TuneSolver
# ---------------------------------------------------------------------------

class TuneSolver(_ConfigSampleMixin):
    """Manages tuning artefacts under ``<cavity>/tuned/tune_info/``.

    After a successful run, ``cavity.tuned`` becomes the fresh
    :class:`Cavity` instance living in ``<cavity>/tuned/`` — subsequent
    ``cavity.tuned.run_eigenmode()`` / ``cavity.tuned.run_wakefield()``
    calls write into the tuned cavity's own folders.

    Access
    ------
    - ``cav.tune.config``         — tuning configuration dict
    - ``cav.tune.config_sample``  — the default config, to edit and pass to run()
    - ``cav.tune.qois``           — ``tune_res.json`` contents
    - ``cav.tune.convergence``    — iteration‐history DataFrame
    - ``cav.tune.plot_convergence()``
    """

    _config_kind = 'tune'

    def __init__(self, cavity):
        self.cavity = cavity
        self._config = None
        self._qois = None
        self._convergence = None

    @property
    def folder(self):
        """Directory holding tune artefacts (``<cav>/tuned/tune_info/``)."""
        return Path(self.cavity.self_dir) / 'tuned' / 'tune_info'

    def _results_folder(self):
        """:attr:`folder` when the cavity has a workspace, else None; warns when
        the tune there started from a geometry other than the cavity's now."""
        if not self.cavity.self_dir:
            return None
        warn_if_stale(self.cavity, self.folder, 'The tune results')
        return self.folder

    # -- Lazy‐loaded results ------------------------------------------------

    @property
    def config(self):
        if not self.cavity.self_dir:
            return {}
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
        folder = self._results_folder()
        if folder is None:
            return {}
        if self._qois is None:
            res_path = folder / 'tune_res.json'
            if res_path.exists():
                with open(res_path, 'r') as f:
                    self._qois = json.load(f)
            else:
                self._qois = {}
        return self._qois

    @property
    def family(self):
        """The family of tuned designs from a 2-cell family tune (``family.json``),
        as a list of cavities -- each a full geometry at the target frequency, ordered
        by the mid/end split. Empty for a single-design tune (``equal_cell_freq=True``
        or a non-2-cell cavity). The optimiser fans its objectives out over these."""
        folder = self._results_folder()
        if folder is None or not (folder / 'family.json').exists():
            return []
        fam_path = folder / 'family.json'
        with open(fam_path, 'r') as f:
            data = json.load(f)
        cavs = []
        for i, m in enumerate(data.get('members', [])):
            c = self.cavity.rebuild(m['parameters'])
            c.name = f'{self.cavity.name}_family_{i}'
            cavs.append(c)
        return cavs

    @property
    def convergence(self):
        """Tidy long-form convergence history: one row per (stage, iteration,
        parameter), where stage is 'cell_type:tune_var'. Long-form rather than
        wide because different tune stages (e.g. mid-cell then end-cell) can
        have different iteration counts, which a wide table can't hold cleanly."""
        if not self.cavity.self_dir:
            return pd.DataFrame()
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
            Full tune configuration dict, merged over ``DEFAULT_TUNE_CONFIG``
            (see :attr:`config_sample`).
        **kwargs
            Individual tune arguments (e.g. ``freq=1300``,
            ``cell_type={'mid-cell': 'Req'}``). Merged in last, so they take final
            precedence; a key given in both with different values warns.

        The cavity itself is never changed: the tuned geometry becomes
        ``cav.tuned``, in ``<cav>/tuned/``. Before replacing an existing tune, or
        anything in the cavity's folder computed for another geometry, it asks
        (see :func:`cavsim2d.utils.integrity.guard_runs`).

        Returns this namespace, so calls chain: ``cav.tune.run(cfg).qois``.
        """
        # Deferred: breaks the solver_objects <-> processes.tune import cycle.
        from cavsim2d.processes.tune import resolve_tune_defaults, run_tune_parallel

        merged_config = merge_config(DEFAULT_TUNE_CONFIG,
                                     _user_config('tune', tune_config, kwargs))
        resolve_tune_defaults(merged_config, [self.cavity])

        self.cavity._ensure_workspace()      # standalone: provision ./<name>/ if needed

        if not guard_runs([self.cavity], 'tune', merged_config, title='Tune'):
            return self
        # rerun=False keeps a finished tune of this geometry, and its config.
        reused = (not merged_config['rerun'] and (self.folder / 'tune_res.json').exists()
                  and stamp_matches(self.folder, self.cavity) is not False)

        self._invalidate()

        cavs_dict = {self.cavity.name: self.cavity}
        # Degenerate-geometry errors are expected noise from the secant
        # tuner's probe iterations — silence them here so users only see
        # genuine failures.
        with suppress_errors('Parameter set leads to degenerate geometry'):
            run_tune_parallel(cavs_dict, merged_config)

        # Written after the run: the run starts by clearing tuned/, which took a
        # config saved beforehand with it, so cav.tune.config was always empty.
        if self.folder.is_dir() and not reused:
            with open(self.folder / 'config.json', 'w') as f:
                json.dump(merged_config, f, indent=4, default=str)
        self._config = None                     # re-read from what is on disk

        # Load the tuned cavity from the newly-written folder
        self.cavity._tuned_cavity = None  # force reload via property
        tuned = self.cavity.tuned
        # Opt-in, and the same as study.run_tune: eigenmode_config alone only sets
        # the mesh of the tuning solves (the study used to solve the tuned
        # cavities whenever it was given, this method never did).
        if merged_config.get('solve_tuned') and tuned is not None:
            ec = dict(merged_config.get('eigenmode_config') or {})
            if 'n_cells' in ec:
                tuned.set_n_cells(ec['n_cells'])
            tuned.eigenmode.run(ec)

        self._invalidate()
        return self

    def _invalidate(self):
        """Clear cached results so they are re‐read from disk."""
        self._qois = None
        self._convergence = None

    # -- Visualisation ------------------------------------------------------

    def plot_convergence(self, show=True, target=None):
        """One subplot per tuned parameter (e.g. Req_m, freq [MHz]), with one
        line per tune stage. The target frequency is drawn as a horizontal line
        on the ``freq [MHz]`` panel — read from the saved tune status unless passed
        explicitly. Returns (fig, axes).

        A tune that resolved to a design *family* has no iteration history to draw:
        a family is a parametric sweep over the split, not a root-find. Look at
        ``cav.tune.family`` instead."""
        df = self.convergence
        if df.empty:
            # Always-visible (not the verbosity-gated info): someone calling this
            # directly and getting a blank needs to be told why, and "a family has
            # no convergence history" is a different answer from "nothing ran".
            if (self.folder / 'family.json').exists():
                warning("plot_convergence: this tune resolved to a design family, "
                        "which is a sweep rather than an iterative search, so it has "
                        "no convergence history. Use cav.tune.family to inspect the "
                        "members.")
            else:
                warning(f"plot_convergence: no convergence history in {self.folder} - "
                        "run cav.tune.run(...) first.")
            return None, None

        if target is None:                         # target frequency for the reference line
            status_path = self.folder / 'tune_status.json'
            if status_path.exists():
                try:
                    with open(status_path) as f:
                        target = json.load(f).get('target_freq')
                except Exception:
                    target = None

        params = list(df['parameter'].unique())
        fig, axes = plt.subplots(len(params), 1, figsize=(6, 3 * len(params)),
                                 squeeze=False)
        axes = axes[:, 0]

        for ax, param in zip(axes, params):
            for stage, sub in df[df['parameter'] == param].groupby('stage'):
                ax.plot(sub['iteration'], sub['value'], marker='o', label=stage)
            if param == 'freq [MHz]' and target is not None:
                ax.axhline(float(target), ls='--', color='k',
                           label=f'target {float(target):.3f} MHz')
            ax.set_xlabel('Iteration')
            ax.set_ylabel(param)
            ax.legend()

        fig.suptitle(f'Tuning Convergence — {self.cavity.name}')
        plt.tight_layout()
        _maybe_show(show)
        return fig, axes

    def family_scan(self, freq=None, n=5, fraction=0.01, mesh_config=None, vary='shape'):
        """Iso-frequency design family for a symmetric elliptical cavity.

        Every member holds the accelerating (operating) frequency ``f`` fixed; what
        varies is the *shape freedom* left over once ``f`` is pinned. Two modes:

        ``vary='shape'`` (default) -- trace the constant-frequency locus in the
        ``(Req, L)`` plane for a **uniform** cell (equator radius vs cell length),
        returning *n* designs spaced equidistantly along it. The locus is close to a
        straight diagonal, so the members are equidistant in both ``Req`` and ``L``,
        spanning ``+/- fraction`` of the nominal ``Req``. ``L`` is the free shape knob
        once ``Req`` fixes the frequency.

        ``vary='split'`` (2+ cells) -- hold the **mid** cell fixed and sweep the
        **end**-cell length ``L_e``, retuning the shared ``Req`` per member so the
        assembly stays at ``f``. This traces the mid/end frequency *split* at constant
        operating frequency; the member where the mid-cell and end-cell (quarter)
        frequencies coincide is the classical *each-cell-at-f* design.

        For a symmetric **2-cell** every member is field-flat by mirror symmetry, so
        these are *tuning-equivalent* designs -- same frequency, same flatness -- yet
        with genuinely different ``R/Q``, peak-field ratios and wakes. (For ``n >= 3``
        the ``shape`` members are not all field-flat -- use the two-stage tune.)

        Parameters
        ----------
        freq : float, optional
            Target accelerating frequency [MHz]; default is the nominal cavity's own.
        n : int
            Number of family members (default 5).
        fraction : float
            Sweep half-width as a fraction of the nominal ``Req`` (``shape``) or of the
            nominal end length ``L_e`` (``split``); default 0.01. Too large a value can
            drive a cell into degenerate geometry (recorded as ``freq = NaN``).
        mesh_config : dict, optional
            Eigenmode mesh for the internal trace solves (default ``{'h': 15, 'p': 3}``).
        vary : {'shape', 'split'}
            Which residual shape freedom to trace (see above).

        Returns
        -------
        Study
            Container of the *n* cavities in ``.cavities_list`` (each ready for
            ``.eigenmode.run()`` / ``.wakefield.run()``), with the design table
            attached as ``.designs``.
        """
        # The trace solves run in scratch folders removed afterwards; the members
        # returned live in a folder of their own, ready to be solved.
        with scratch_folders() as scratch:
            return self._family_scan(scratch, freq, n, fraction, mesh_config, vary)

    def _family_scan(self, scratch, freq, n, fraction, mesh_config, vary):
        # Deferred: models and study both import solver_objects (cycle).
        from cavsim2d.models.elliptical import EllipticalCavity
        from cavsim2d.study import Study

        cav = self.cavity
        if not getattr(cav, 'uses_cell_suffixes', False):
            raise ValueError('family_scan is defined for elliptical-family cavities only.')
        A, B, aa, bb, Ri, L0, Req0 = (float(x) for x in np.asarray(cav.half_cells()[0], float))
        ncells = int(cav.n_cells or 1)
        bp = getattr(cav, 'beampipe', 'both')
        mesh = mesh_config or {'h': 20, 'p': 3}

        if vary == 'split':
            # Hold the mid cell fixed; sweep the end-cell length and retune the shared
            # Req per member so the assembly stays at f (the mid/end split at fixed
            # operating frequency). The member where mid-freq == end-freq is each-cell-at-f.
            mid = [float(x) for x in cav.mid_cell]
            end = [float(x) for x in cav.end_cell_left]
            Le0, Req0 = end[5], mid[6]

            def asmf(Req, Le):
                m = mid[:]; m[6] = Req
                e = end[:]; e[5] = Le; e[6] = Req
                c = EllipticalCavity(ncells, m, e, e, beampipe=bp)
                if c.profile() is None:
                    return np.nan
                c.set_workspace(scratch())
                try:
                    c.eigenmode.run(mesh_config=mesh, n_modes=ncells + 2)
                    return float(_accelerating_mode_row(c.eigenmode.qois_df, freq)['freq [MHz]'])
                except Exception:
                    return np.nan

            def _fund(params, bpx):
                # fundamental of a single cell: the mid cell's periodic frequency
                # (bp='none') or the end cup's beampipe-loaded quarter frequency
                # (bp='both' = the symmetric two-end-cup cell the quarter halves).
                c = EllipticalCavity(1, params, params, params, beampipe=bpx)
                if c.profile() is None:
                    return np.nan
                c.set_workspace(scratch())
                try:
                    c.eigenmode.run(mesh_config=mesh, n_modes=3)
                    dfm = c.eigenmode.qois_df.query('m == 0')
                    dfm = dfm[dfm['freq [MHz]'] > 1.0]        # drop spurious ~0 MHz modes
                    return float(dfm.sort_values('freq [MHz]')['freq [MHz]'].iloc[0])
                except Exception:
                    return np.nan

            f0 = asmf(Req0, Le0)
            target = float(freq) if freq is not None else f0
            dfdR = (asmf(Req0 + 0.01 * Req0, Le0) - f0) / (0.01 * Req0)
            if not np.isfinite(dfdR) or abs(dfdR) < 1e-12:
                raise ValueError('assembly frequency is insensitive to Req; cannot retune the split.')
            rows = []
            for Le in np.linspace(Le0 * (1 - fraction), Le0 * (1 + 3 * fraction), int(n)):
                Req = Req0
                f = asmf(Req, Le)
                for _ in range(5):                        # Newton-retune Req onto the target f
                    if not np.isfinite(f) or abs(f - target) < 1e-3:
                        break
                    Req = Req - (f - target) / dfdR
                    f = asmf(Req, Le)
                mm = mid[:]; mm[6] = Req
                ee = end[:]; ee[5] = Le; ee[6] = Req
                rows.append({'Req': float(Req), 'L_e': float(Le), 'L_m': float(mid[5]),
                             'freq [MHz]': f, 'mid_freq': _fund(mm, 'none'),
                             'end_freq': _fund(ee, 'both')})
            folder = tempfile.mkdtemp(prefix='cavsim2d_family_')
            family = Study(folder, _skip_project_init=True)
            for i, rr in enumerate(rows):
                m = mid[:]; m[6] = rr['Req']
                e = end[:]; e[5] = rr['L_e']; e[6] = rr['Req']
                c = EllipticalCavity(ncells, m, e, e, beampipe=bp, name='family_%d' % i)
                c.set_workspace(os.path.join(folder, c.name))
                family.cavities_list.append(c)
                family.cavities_dict[c.name] = c
            family.designs = pd.DataFrame(rows)
            return family

        def fcav(Req, L):
            c = EllipticalCavity(ncells, [A, B, aa, bb, Ri, L, Req],
                                 [A, B, aa, bb, Ri, L, Req],
                                 [A, B, aa, bb, Ri, L, Req], beampipe=bp)
            if c.profile() is None:
                return np.nan
            c.set_workspace(scratch())
            try:
                c.eigenmode.run(mesh_config=mesh, n_modes=ncells + 2)
                return float(_accelerating_mode_row(c.eigenmode.qois_df, freq)['freq [MHz]'])  # pi-mode
            except Exception:
                return np.nan

        f0 = fcav(Req0, L0)
        target = float(freq) if freq is not None else f0
        # local frequency gradient -> tangent of the constant-f family (dL/dReq)
        fR = (fcav(Req0 + 0.01 * Req0, L0) - f0) / (0.01 * Req0)
        fL = (fcav(Req0, L0 + 0.01 * L0) - f0) / (0.01 * L0)
        if not np.isfinite(fL) or abs(fL) < 1e-12:
            raise ValueError('accelerating frequency is insensitive to L; cannot trace the family.')
        slope = -fR / fL

        rows = []
        for Req in np.linspace(Req0 * (1 - fraction), Req0 * (1 + fraction), int(n)):
            L = L0 + slope * (Req - Req0)                 # predictor along the diagonal
            f = fcav(Req, L)
            for _ in range(4):                            # Newton-correct L onto the target f
                if not np.isfinite(f) or abs(f - target) < 1e-3:
                    break
                L = L - (f - target) / fL
                f = fcav(Req, L)
            rows.append({'Req': float(Req), 'L': float(L), 'freq [MHz]': f})

        folder = tempfile.mkdtemp(prefix='cavsim2d_family_')
        family = Study(folder, _skip_project_init=True)
        for i, rr in enumerate(rows):
            c = EllipticalCavity(ncells, [A, B, aa, bb, Ri, rr['L'], rr['Req']],
                                 [A, B, aa, bb, Ri, rr['L'], rr['Req']],
                                 [A, B, aa, bb, Ri, rr['L'], rr['Req']],
                                 beampipe=bp, name='family_%d' % i)
            c.set_workspace(os.path.join(folder, c.name))
            family.cavities_list.append(c)
            family.cavities_dict[c.name] = c
        family.designs = pd.DataFrame(rows)
        return family

    def _solve_qoi_targets(self, var_names, targets, mesh_config=None, maxit=20,
                           n_family=5, fraction=0.03, bc=None, tol=None):
        """Backend for QoI-constrained tuning -- **not a public entry point**.

        Invoked by the tune process when the config carries ``qoi_targets``; users
        drive it through the ordinary config API::

            cav.tune.run({'freqs': 801.58,
                          'cell_type': {'mid-cell': ['Req', 'L']},
                          'qoi_targets': {'R/Q [Ohm]': 157}})

        An equality target (e.g. ``R/Q = 157``) is a constraint, not an objective:
        handed to the optimiser as ``|R/Q - 157|`` it is merely traded off against the
        other objectives. Pinning it here puts every candidate *on* the target surface.

        Moves the resolved shape variables ``var_names`` until ``targets`` (frequency
        plus any QoI columns of the accelerating monopole mode) are met, by a damped
        multivariate Newton on the assembled cavity with a finite-difference Jacobian
        and a degenerate-geometry guard -- exact, making no use of the approximate
        per-cell R/Q additivity. In one dimension it is the secant the frequency tuner
        already uses.

        Counting rule: ``#targets > #vars`` -> ``ValueError`` (over-constrained);
        ``==`` -> a unique design; ``<`` -> a family. Returns a result dict:

        - unique: ``{'mode': 'unique', 'parameters': <full suffixed dict>,
          'achieved': {col: val}, 'variables': [...], 'history': [...]}``
        - family: ``{'mode': 'family', 'family': <Study>}``
        """
        # Every evaluation solves in a scratch folder removed afterwards.
        with scratch_folders() as scratch:
            return self._solve_qoi_targets_in(scratch, var_names, targets, mesh_config,
                                              maxit, n_family, fraction, bc, tol)

    def _solve_qoi_targets_in(self, scratch, var_names, targets, mesh_config, maxit,
                              n_family, fraction, bc, tol):
        # Deferred: models imports solver_objects (cycle).
        from cavsim2d.models.elliptical import EllipticalCavity

        cav = self.cavity
        if not getattr(cav, 'uses_cell_suffixes', False):
            raise ValueError('QoI-constrained tuning is defined for elliptical-family '
                             'cavities only.')
        ncells = int(cav.n_cells or 1)
        bp = getattr(cav, 'beampipe', 'both')
        mesh = mesh_config or {'h': 20, 'p': 3}

        cells = {'m': [float(x) for x in cav.mid_cell],
                 'el': [float(x) for x in cav.end_cell_left],
                 'er': [float(x) for x in getattr(cav, 'end_cell_right', cav.end_cell_left)]}
        pidx = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6}
        bare_of = {v: k for k, v in pidx.items()}
        suf = {'m': '_m', 'el': '_el', 'er': '_er'}

        def resolve(name):
            for sx, whichs in (('_m', ['m']), ('_el', ['el']), ('_er', ['er']),
                               ('_e', ['el', 'er'])):
                if name.endswith(sx) and name[:-len(sx)] in pidx:
                    return [(w, pidx[name[:-len(sx)]]) for w in whichs]
            if name in pidx:
                return [(w, pidx[name]) for w in ('m', 'el', 'er')]
            return None

        var_names = list(var_names)
        loc = {}
        for n in var_names:
            r = resolve(n)
            if not r:
                raise ValueError("Unknown tune variable '%s' (expected one of %s, optionally "
                                 "suffixed _m/_el/_er/_e)." % (n, sorted(pidx)))
            loc[n] = r

        tgt_names = list(targets)
        tgt = np.array([float(targets[t]) for t in tgt_names], float)
        nv, nt = len(var_names), len(tgt_names)
        if nt > nv:
            raise ValueError('Over-constrained: %d target(s) but %d tune variable(s). Give at '
                             'least as many variables as targets.' % (nt, nv))

        def build(x):
            for k, n in enumerate(var_names):
                for w, i in loc[n]:
                    cells[w][i] = float(x[k])
            return EllipticalCavity(ncells, cells['m'][:], cells['el'][:], cells['er'][:],
                                    beampipe=bp)

        def evaluate(x):
            c = build(x)
            if c.profile() is None:
                return None
            c.set_workspace(scratch())
            try:
                # Use the DEFAULT solver (no BDDC) so the R/Q the Newton optimises MATCHES
                # the R/Q the rest of the code reports (the re-assembled gate, eigenmode,
                # wakefield). BDDC was ~1.5-1.9x faster but converges the eigenVECTOR
                # slightly differently, shifting R/Q by ~0.07 Ohm on a 3-cell -- enough to
                # make a design that met the Newton's tolerance FAIL the re-assembled gate
                # (measured: Newton 343.003 vs re-assembled 343.073). Consistency > speed.
                # ``bc`` lets a periodic mid cup run with 'mm' boundaries (staged cup tuning).
                run_kw = dict(mesh_config=mesh, n_modes=ncells + 2)
                if bc:
                    run_kw['boundary_conditions'] = bc
                c.eigenmode.run(**run_kw)
                acc = _accelerating_mode_row(c.eigenmode.qois_df, targets.get('freq [MHz]'))
                return np.array([float(acc[t]) for t in tgt_names])
            except Exception:
                return None

        if nt < nv:
            # Under-constrained: more shape freedom than pinned targets -> a family. The
            # frequency is held across all members (family_scan); any further QoI target
            # then selects a member rather than pinning the whole set.
            if 'freq [MHz]' in tgt_names:
                fam = self.family_scan(freq=float(targets['freq [MHz]']), n=n_family,
                                       fraction=fraction, mesh_config=mesh,
                                       vary=('split' if ncells >= 2 else 'shape'))
                return {'mode': 'family', 'family': fam}
            raise ValueError('Under-constrained (%d variables, %d targets) with no frequency '
                             'target: cannot form a design family.' % (nv, nt))

        # Square case: a Newton solve (the secant, in 1-D). Two things keep it cheap:
        # a SMALL finite-difference probe (1% of the value, not 20%) so the Jacobian is
        # local and convergence is fast; and a BROYDEN rank-1 update so each iteration
        # costs ONE eigensolve rather than re-probing the whole Jacobian -- the full FD
        # Jacobian is (re)built only at the start and whenever a step stops reducing the
        # residual. A short backtrack steps off degenerate geometry.
        x = np.array([float(cells[loc[n][0][0]][loc[n][0][1]]) for n in var_names], float)
        fd = np.maximum(0.3, 0.01 * np.abs(x))
        # Per-target ABSOLUTE convergence tolerance, in each target's own units:
        # frequency 1e-2 MHz and R/Q 0.1 Ohm by default -- the practical acceptance band
        # for a cavity design. Both are comfortably reachable, and crucially R/Q's 0.1 Ohm
        # is WIDER than the ~0.05-0.1 Ohm parameter round-trip, so the assembled-target
        # gate does not reject a valid design over a reconstruction discrepancy. Pinning
        # the joint solve far tighter than this is unproductive (the Broyden solve wanders
        # when freq and R/Q are held below their reproducibility). ``tol`` overrides per
        # target as an absolute tolerance -- e.g. {'freq [MHz]': 1e-2, 'R/Q [Ohm]': 0.1}.
        _user_tol = tol if isinstance(tol, dict) else {}
        tol = np.array([_user_tol[t] if t in _user_tol
                        else (0.1 if t == 'R/Q [Ohm]' else 1e-2)
                        for t, v in zip(tgt_names, tgt)], float)
        history = []

        def _fd_jacobian(x0, f0):
            jac = np.zeros((nt, nv))
            for k in range(nv):
                xp = x0.copy(); xp[k] += fd[k]; fp = evaluate(xp)
                if fp is None:                              # forward degenerate -> backward
                    xp[k] -= 2 * fd[k]; fp = evaluate(xp)
                    if fp is None:
                        raise RuntimeError('QoI tune: degenerate geometry probing %s'
                                           % var_names[k])
                    jac[:, k] = (f0 - fp) / fd[k]
                else:
                    jac[:, k] = (fp - f0) / fd[k]
            return jac

        f = evaluate(x)
        if f is None:
            raise RuntimeError('QoI tune: degenerate start geometry.')
        jac = _fd_jacobian(x, f)
        prev = np.inf
        for _ in range(maxit):
            history.append((x.tolist(), dict(zip(tgt_names, f.tolist()))))
            r = f - tgt
            if np.all(np.abs(r) < tol):
                break
            dx = np.clip(np.linalg.solve(jac, -r), -5.0, 5.0)
            fn, scale = None, 1.0
            for _bt in range(4):                            # backtrack off degenerate geometry
                fn = evaluate(x + scale * dx)
                if fn is not None:
                    break
                scale *= 0.5
            if fn is None:
                raise RuntimeError('QoI tune: degenerate geometry at %s'
                                   % dict(zip(var_names, np.round(x + dx, 4).tolist())))
            s = scale * dx
            jac = jac + np.outer((fn - f) - jac @ s, s) / (s @ s)   # Broyden rank-1 update
            x, f = x + s, fn
            cur = float(np.linalg.norm((f - tgt) / np.maximum(np.abs(tgt), 1e-9)))
            if cur > prev:                                  # stalled -> refresh the Jacobian
                jac = _fd_jacobian(x, f)
            prev = cur

        ach = f if f is not None else evaluate(x)
        build(x)                                     # leave `cells` at the solution
        params = dict(cav.parameters)
        for k, n in enumerate(var_names):
            for w, i in loc[n]:
                params['%s%s' % (bare_of[i], suf[w])] = float(x[k])
        return {'mode': 'unique',
                'parameters': params,
                'achieved': dict(zip(tgt_names, ach.tolist())) if ach is not None else {},
                'variables': list(var_names),
                'history': history}
