"""The multipacting namespace, ``cav.multipacting``."""
import json
import pickle
import shutil
from pathlib import Path
import warnings
from cavsim2d.utils.printing import CavsimWarning, error, info, warning
import numpy as np
import matplotlib.pyplot as plt

from cavsim2d.utils.config_validation import canonicalise
from cavsim2d.utils.integrity import (guard_runs, notice_tuned_versions, stamp_matches,
                                      warn_if_stale, write_stamp)
from cavsim2d.analysis.multipacting.sey import SEY
from cavsim2d.analysis.multipacting import metrics as mp_metrics
from cavsim2d.solvers.eigenmode_result import monopole_dir
from cavsim2d.utils.style import house_style, WARM
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP, choose_viewer, webgui_problem
from cavsim2d.solvers._shared import _has_field, _has_mesh, _maybe_show, _require_vacuum
from cavsim2d.solvers.configs import (_ConfigSampleMixin, _user_config,
                                      DEFAULT_MULTIPACTING_CONFIG, merge_config)


# ---------------------------------------------------------------------------
# MultipactingSolver
# ---------------------------------------------------------------------------

class MultipactingSolver(_ConfigSampleMixin):
    """Manages ``<cavity>/multipacting/`` — multipacting analysis.

    Tracks emitted electrons in the cavity's eigenmode field (with secondary
    emission) over a peak-field sweep; the counter function ``c20/c0`` shows the
    field levels where multipacting resonates. The EM field is read from the
    cavity's eigenmode result — ``run()`` solves the (monopole) eigenmode first
    if it has not been run.

    Access
    ------
    Set up (each returns ``self``, so they chain), then inspect, then run:

    - ``set_mesh_parameters(h, p, pec_maxh)`` -> ``show_mesh()``
    - ``compute_field()`` -> ``show_field('E'|'H', mode=i)`` / ``field_frequencies``
    - ``set_emission_points(xrange, step)`` -> ``show_emission_points()``
    - ``set_sey(path)`` -> ``plot_sey()`` / ``sey``
    - ``set_epks(...)`` / ``set_phis(...)``
    - ``run(config)`` — the peak-field sweep. Anything staged above can equally be
      given in the config; a key in both warns that the config wins.
    - Results: ``counter`` / ``epk`` / ``final_energy`` / ``results``;
      ``plot_counter()`` / ``plot_final_energy()`` / ``plot_enhanced_counter()`` /
      ``plot_distance_map(i)`` / ``plot_trajectories()`` /
      ``animate_trajectories()`` / ``animate_phase_space(x, y)``
    """

    #: Eigenmode config for the auto-run: monopole, magnetic end-planes, and a
    #: near-wall-adequate mesh (~6 mm, matching the validated PyMultipact setup).
    _config_kind = 'multipacting'

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

    def set_emission_points(self, xrange, step=None, n_points=None):
        """Stage where electrons are launched from: the emission-site z-interval
        ``xrange`` [m], an optional ``step`` [m] that thins the sites to at most
        one per ``step`` of z, and an optional ``n_points`` that **adds** sites.

        Emission sites are wall-polyline vertices, so a coarse mesh offers only a
        few in a narrow band. ``n_points`` resamples the wall *inside the band* to
        that many equally spaced (by arclength) sites, interpolating along the
        existing straight wall segments — the polyline, collision geometry and
        surface normals are unchanged, so this buys launch sites **without**
        refining the mesh (much cheaper and more local than shrinking
        ``pec_maxh``). It is a no-op if the band already has that many.

        This is the single place emission sites are defined — :meth:`show_emission_points`
        and :meth:`run` both use what is staged here (``run`` may override it from
        its config, and says so). Inspect the selection with
        :meth:`show_emission_points` before running. Returns ``self``."""
        xrange = list(xrange)
        # An interval, NOT a list of sites: only [0] and [1] are ever used, so a
        # passed-in point array (e.g. np.arange(z0, z1, dz)) would silently select
        # a band one dz wide. Say so instead.
        if len(xrange) != 2:
            raise ValueError(
                f"set_emission_points expects a 2-element [z_min, z_max] interval "
                f"in metres; got {len(xrange)} values. To emit from a subset of the "
                f"sites in that interval pass step=<spacing in m> — e.g. "
                f"set_emission_points([{xrange[0]:.4g}, {xrange[-1]:.4g}], step=5e-4).")
        if xrange[1] <= xrange[0]:
            raise ValueError(f"set_emission_points needs z_min < z_max; got {xrange}.")
        self._staged['xrange'] = xrange
        if step is not None:
            self._staged['step'] = step
        if n_points is not None:
            self._staged['n_points'] = int(n_points)
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
            if _has_mesh(pol_dir):
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

    def show_mesh(self, plotter='ngsolve', ax=None, show=True):
        """Preview the mesh multipacting will use — the staged own-field mesh
        when :meth:`set_mesh_parameters`/``pec_maxh`` were given, otherwise the
        eigenmode mesh. No field is solved.

        *ax* / *show* apply to ``plotter='matplotlib'`` and behave as on the
        ``plot_*`` methods; the webgui plotter ignores them."""
        mesh = self._preview_mesh()
        if choose_viewer(plotter) == 'matplotlib':
            return NGSolveMEVP()._plot_mesh_matplotlib(mesh, ax=ax, show=show)
        # Deferred: webgui is notebook-oriented.
        from ngsolve.webgui import Draw
        return Draw(mesh)

    # -- Field -----------------------------------------------------------------

    @property
    def field_dir(self):
        """Folder holding the field multipacting will track in: its OWN field
        (``multipacting/field/``) once :meth:`compute_field` (or a ``pec_maxh`` /
        ``mesh_config`` run) has solved one, else the shared monopole eigenmode
        folder. ``None`` if neither exists yet."""
        own = self.folder / 'field'
        if _has_field(own):
            return own
        pol = Path(monopole_dir(self.cavity.eigenmode.folder))
        return pol if _has_field(pol) else None

    @property
    def has_field(self):
        """True when a field multipacting can track in already exists on disk."""
        return self.field_dir is not None

    @property
    def field_frequencies(self):
        """Frequencies [MHz] of the computed modes, in solve order — index 0 is
        the mode multipacting uses by default (override with ``config['mode']``)."""
        own = self.folder / 'field'
        if (own / 'freqs.json').exists():
            with open(own / 'freqs.json') as f:
                return list(json.load(f))
        return [m.frequency for m in self.cavity.eigenmode.modes]

    def compute_field(self, eigenmode_config=None, **kwargs):
        """Solve the multipacting field **now**, independently of :meth:`run`.

        Takes the same keys as an **eigenmode config** — ``polarisation``
        ('monopole', 'dipole', …), ``n_modes``, and ``mesh_config`` ``{'h', 'p'}``
        — plus multipacting's own ``pec_maxh`` [mm] surface refinement. So the
        tracking can be driven by the second monopole mode, or by a dipole mode::

            cav.multipacting.compute_field(polarisation='monopole', n_modes=4,
                                           mesh_config={'h': 6, 'p': 3}, pec_maxh=1)
            cav.multipacting.show_field('E', mode=1)      # the 2nd mode
            cav.multipacting.run({'mode': 1, ...})

        Solving here (rather than inside :meth:`run`) lets the mesh and field be
        inspected first with :meth:`show_mesh` / :meth:`show_field`; ``run()`` then
        **reuses** this field rather than re-solving. ``cav.eigenmode``'s own
        results are left untouched. Everything passed is staged, so the sweep sees
        the same settings. Returns ``self``.

        NB the tracker is 2D axisymmetric: for a dipole (m>=1) mode it follows the
        in-plane E in the phi=0 plane, where the azimuthal component vanishes.
        """
        # Deferred: ngsolve is an optional heavy dependency.
        from cavsim2d.analysis.multipacting.fields import solve_multipacting_field
        cfg = canonicalise(merge_config({}, eigenmode_config, kwargs), 'multipacting')
        mesh_config = cfg.get('mesh_config') or {}
        self.set_mesh_parameters(**{k: v for k, v in
                                    (('h', mesh_config.get('h')),
                                     ('p', mesh_config.get('p')),
                                     ('pec_maxh', cfg.get('pec_maxh'))) if v is not None})
        for key in ('polarisation', 'n_modes'):
            if cfg.get(key) is not None:
                self._staged[key] = cfg[key]

        self.cavity._ensure_workspace()
        if not self._guard(cfg, sweep=False):
            return self
        self.folder.mkdir(parents=True, exist_ok=True)
        spec = self._field_mesh_spec() or dict(self.DEFAULT_EIGENMODE['mesh_config'])
        pol = self._staged.get('polarisation',
                               self.DEFAULT_EIGENMODE['polarisation'])
        n_modes = self._staged.get('n_modes')
        fields_dir = self.folder / 'field'
        info(f"Multipacting: solving its own {pol} field on mesh {spec} into "
             f"{fields_dir}. cav.eigenmode results are left untouched.")
        freqs = solve_multipacting_field(self.cavity, str(fields_dir),
                                         mesh_config=spec, n_modes=n_modes,
                                         polarisation=pol)
        with open(fields_dir / 'freqs.json', 'w') as f:
            json.dump(list(freqs), f)
        write_stamp(self.folder, self.cavity)
        info(f"Computed {len(freqs)} {pol} mode(s): "
             f"{', '.join(f'{i}: {v:.3f} MHz' for i, v in enumerate(freqs))}. "
             f"Mode 0 is tracked unless config['mode'] says otherwise.")
        return self

    def _guard(self, cfg, sweep):
        """Ask before this run replaces anything (see
        :func:`~cavsim2d.utils.integrity.guard_runs`); False when declined.

        A sweep overwrites a finished sweep; solving the field alone does not.
        Unlike the other analyses multipacting reuses its folder's field instead
        of rebuilding it, so once the answer is yes a folder computed for another
        geometry is cleared here rather than by the run.
        """
        def replaces(unit):
            return sweep and (unit.path / 'mresults.pkl').exists()
        notice_tuned_versions([self.cavity], 'multipacting', stacklevel=4)
        if not guard_runs([self.cavity], 'multipacting', cfg, title='Multipacting',
                          replaces=replaces):
            return False
        if self.folder.exists() and stamp_matches(self.folder, self.cavity) is False:
            shutil.rmtree(self.folder)
            self._invalidate()
        return True

    def show_field(self, which='E', mode=None):
        """Draw the field multipacting tracks in — ``which='E'`` (electric) or
        ``'H'`` (magnetic) — for the selected ``mode``.

        ``mode`` is the 0-based index into the computed modes (default: the one
        the sweep will use, i.e. ``config['mode']`` or 0). Several modes are often
        solved, so pass ``mode=i`` to inspect any of them; see
        :attr:`field_frequencies` for what was computed. Requires a field: run
        :meth:`compute_field` (or :meth:`run`) first."""
        # Deferred: ngsolve is an optional heavy dependency.
        from cavsim2d.analysis.multipacting.fields import (load_eigenmode_fields,
                                                           build_emfield)
        fields_dir = self.field_dir
        if fields_dir is None:
            raise RuntimeError(
                "No multipacting field yet. Call cav.multipacting.compute_field() "
                "(or run()) first — or run the monopole eigenmode.")
        which = str(which).upper()
        if which not in ('E', 'H'):
            raise ValueError(f"which must be 'E' or 'H'; got {which!r}.")
        if mode is None:
            mode = (self._staged.get('mode')
                    or (self._config or {}).get('mode') or 0)
        mesh, gfu_E, _ = load_eigenmode_fields(str(fields_dir))
        freqs = self.field_frequencies
        if mode >= len(gfu_E):
            raise IndexError(f"mode {mode} out of range — {len(gfu_E)} mode(s) computed "
                             f"({[round(float(v), 3) for v in freqs]}).")
        # Draw exactly what the tracker sees: the in-plane E (the product space's
        # HCurl block) and the reconstructed azimuthal H. The raw product-space
        # GridFunction cannot be drawn directly. Magnitude, since H carries the
        # physical 1j phase and webgui will not render a complex field.
        em = build_emfield(gfu_E, mode, float(freqs[mode]))
        # Deferred: ngsolve is an optional heavy dependency.
        from ngsolve import Norm
        cf = Norm(em.e if which == 'E' else em.h)
        label = f"|{which}| mode {mode} ({float(freqs[mode]):.3f} MHz)"
        info(f"Multipacting: showing {label} from {fields_dir}.")
        problem = webgui_problem()
        if problem:
            warnings.warn(f"{problem}; nothing will be drawn.", CavsimWarning, stacklevel=2)
        # Deferred: webgui is notebook-oriented.
        from ngsolve.webgui import Draw
        return Draw(cf, mesh, label)

    def show_emission_points(self, ax=None, show=True):
        """Plot the wall with the emission sites staged by
        :meth:`set_emission_points` — exactly the selection the tracker will make,
        so the launch region can be checked before running.

        Raises ``RuntimeError`` if no emission points have been set: the launch
        region is a physical choice, so it is set once (and deliberately) with
        ``set_emission_points(xrange, step=...)`` rather than passed here."""
        # Deferred: ngsolve is an optional heavy dependency.
        from cavsim2d.analysis.multipacting.driver import (_surface_points,
                                                           densify_wall)
        xrange = self._staged.get('xrange')
        if xrange is None:
            raise RuntimeError(
                "No emission points set. Call "
                "cav.multipacting.set_emission_points(xrange, step=...) first "
                "(xrange is the launch-site z-interval in metres).")
        step = self._staged.get('step')
        mesh = self._preview_mesh()
        # Preview the wall the tracker will actually launch from, including any
        # in-band densification requested via set_emission_points(n_points=...).
        xsurf = densify_wall(_surface_points(mesh), xrange,
                             self._staged.get('n_points'))
        pts = xsurf[(xsurf[:, 0] > xrange[0]) & (xsurf[:, 0] < xrange[1])]
        if len(pts) == 0:
            # Two distinct causes: the band misses the wall, or the mesh is too
            # coarse for the band to contain any node (emission sites ARE mesh
            # surface nodes, so a 5 mm band on a 6 mm mesh can be empty).
            gaps = np.diff(np.sort(xsurf[:, 0]))
            typical = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
            raise RuntimeError(
                f"No wall points fall in xrange = [{xrange[0]:.4g}, {xrange[1]:.4g}] m "
                f"(width {xrange[1] - xrange[0]:.4g} m). The wall spans "
                f"z = [{xsurf[:, 0].min():.4g}, {xsurf[:, 0].max():.4g}] m with a typical "
                f"surface-node spacing of {typical:.2g} m. Emission sites are wall "
                f"vertices, so a band narrower than the node spacing selects nothing: "
                f"widen the interval, pass set_emission_points(..., n_points=N) to "
                f"interpolate N sites along the wall inside the band (no mesh change), "
                f"or refine the wall with set_mesh_parameters(pec_maxh=<mm>).")
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
        if not self.cavity.self_dir:
            return {}
        if self._config is None:
            path = self.folder / 'config.json'
            if path.exists():
                with open(path) as f:
                    self._config = json.load(f)
            else:
                self._config = {}
        return self._config

    @property
    def results(self):
        """The saved sweep result dict (``multipacting/mresults.pkl``), or {}."""
        if not self.cavity.self_dir:
            return {}
        warn_if_stale(self.cavity, self.folder, 'The multipacting results')
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

        Emission points are **required**: stage them with
        :meth:`set_emission_points` (recommended — then check them with
        :meth:`show_emission_points`) or pass ``xrange``/``step`` in the config.
        Without either, this raises.

        If a field was already solved by :meth:`compute_field`, it is **reused**
        (and says so) rather than re-solved, so the mesh you inspected is the one
        that is tracked. ``config['mode']`` picks which computed mode drives the
        tracking (default the first); inspect any of them with :meth:`show_field`.

        Values staged with :meth:`set_mesh_parameters` / :meth:`set_emission_points`
        / :meth:`set_epks` / :meth:`set_phis` are merged in under the config; a key
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

        _require_vacuum(self.cavity, 'multipacting')
        user_cfg = _user_config('multipacting', config, kwargs)
        overlap = sorted(set(self._staged) & set(user_cfg))
        if overlap:
            # warnings.warn (not the verbosity-gated warning()): the user must
            # see that their staged values are being overridden.
            warnings.warn(
                f"Multipacting config entr{'ies' if len(overlap) > 1 else 'y'} "
                f"{overlap} overwrite value(s) previously staged with set_*(); "
                f"the run() config takes precedence.", CavsimWarning, stacklevel=2)
        cfg = merge_config(DEFAULT_MULTIPACTING_CONFIG, self._staged, user_cfg)

        if cfg.get('xrange') is None:
            raise RuntimeError(
                "No emission points set. Either stage them with "
                "cav.multipacting.set_emission_points(xrange, step=...) — then "
                "check them with show_emission_points() — or pass 'xrange' (and "
                "optionally 'step') in the run() config.")

        self.cavity._ensure_workspace()
        if not self._guard(cfg, sweep=True):
            return self
        self.folder.mkdir(parents=True, exist_ok=True)

        eigenmode_config = cfg.get('eigenmode_config')
        field_spec = self._field_mesh_spec(cfg)
        # A field solved earlier by compute_field() is reused as-is: re-solving it
        # would silently discard the mesh the user inspected.
        own_field = self.folder / 'field'
        if _has_field(own_field):
            info(f"Multipacting: a field has already been computed in {own_field} "
                 f"— using it (call compute_field() again to replace it).")
            fields_dir = own_field
            freqs = self.field_frequencies
        elif field_spec is not None:
            # Multipacting-owned field: solve the monopole eigenmode on a
            # (PEC-refined) multipacting-specific mesh into multipacting/field/.
            # The shared cav.eigenmode results are deliberately NOT recomputed
            # or replaced.
            pol = cfg.get('polarisation') or self.DEFAULT_EIGENMODE['polarisation']
            info(f"Multipacting: own field mesh {field_spec} — solving the "
                 f"{pol} eigenmode into {self.folder / 'field'}. "
                 f"cav.eigenmode results are left untouched.")
            fields_dir = self.folder / 'field'
            freqs = solve_multipacting_field(self.cavity, str(fields_dir),
                                             mesh_config=field_spec,
                                             n_modes=cfg.get('n_modes'),
                                             polarisation=pol)
            with open(fields_dir / 'freqs.json', 'w') as f:
                json.dump(list(freqs), f)
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

        # Which of the computed modes drives the tracking. Several are usually
        # solved; the first is the default. Inspect any of them with
        # show_field(which='E'|'H', mode=i).
        mode_i = cfg.get('mode', 0) or 0
        if len(freqs) > 1:
            info(f"Multipacting: {len(freqs)} mode(s) available "
                 f"{[round(float(v), 3) for v in freqs]} MHz; tracking mode "
                 f"{mode_i} ({float(freqs[mode_i]):.3f} MHz). "
                 f"Set config['mode'] to choose another.")

        self._results = run_sweep(str(self.folder), str(fields_dir), freqs, self.sey,
                                  mode=mode_i,
                                  xrange=cfg.get('xrange'),
                                  epks=cfg.get('epks'),
                                  phis=cfg.get('phis'),
                                  v_init=cfg.get('v_init', 2),
                                  step=cfg.get('step'),
                                  proc_count=cfg.get('proc_count'),
                                  loss_model=cfg.get('loss_model', 'field'),
                                  t_max=cfg.get('t_max', 1000e-10),
                                  progress=cfg.get('progress', True),
                                  n_points=cfg.get('n_points'))
        write_stamp(self.folder, self.cavity)
        return self

    def _ensure_eigenmode(self, eigenmode_config):
        """Path to the monopole eigenmode field folder, running the eigenmode
        solve first if its fields are not on disk."""
        def pol_dir():
            return Path(monopole_dir(self.cavity.eigenmode.folder))
        have = _has_field(pol_dir()) and _has_mesh(pol_dir())
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
            warning("No multipacting results — run cav.multipacting.run(...) first.")
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
            warning("No multipacting results — run cav.multipacting.run(...) first.")
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
            warning("No multipacting results — run cav.multipacting.run(...) first.")
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
                             embed='auto', zoom_smooth=0.12, zoom_min=0.5,
                             inline_format='auto'):
        """Animate the surviving trajectories: moving heads with a short fading
        trace, gradient-coloured by ``'energy'`` (default) or ``'velocity'``.

        Select a subset with ``epk_i`` (field-level index/list), ``phi_i``
        (launch-phase index/list) and/or ``traj`` (trajectory index/list);
        ``None`` selects all (the default). ``save='mp.gif'`` / ``'mp.mp4'``
        writes the animation to disk.

        ``zoom`` frames the view: ``'auto'`` (default) one fixed box around all
        the selected trajectories, ``'follow'`` a **moving camera** that re-frames
        onto the trajectories still alive — so an opening burst across the domain
        is shown wide and the view closes in as the multipacting localises —
        and ``None`` the whole cavity. ``zoom_smooth`` (0-1) sets how quickly the
        camera glides (smaller = smoother/slower) and ``zoom_min`` [mm] floors how
        far it will zoom in.

        ``inline_format='auto'`` embeds the inline player as a compact H.264
        ``<video>`` when ffmpeg is available (so the *whole* animation shows,
        unlike the default JS frame player which is truncated by
        ``animation.embed_limit`` on a long run); ``'video'``/``'jshtml'`` force it.

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
                                    progress=progress, embed=embed,
                                    zoom_smooth=zoom_smooth, zoom_min=zoom_min,
                                    inline_format=inline_format)

    def animate_phase_space(self, x='z', y='r', epk_i=None, phi_i=None, traj=None,
                            color_by='energy', trail=40, step=1, fps=30, save=None,
                            dpi=120, progress=True, embed='auto',
                            inline_format='auto', zoom='auto', zoom_smooth=0.15):
        """Animate the surviving trajectories in a **phase-space projection** —
        the companion to :meth:`animate_trajectories`, with the user free to
        choose what each axis plots.

        ``x``, ``y`` and ``color_by`` each name a per-point quantity: ``'z'`` /
        ``'r'`` [mm], ``'vz'`` / ``'vr'`` / ``'speed'`` [m/s], ``'energy'`` [eV],
        ``'phase'`` [deg] or ``'time'`` [ns] — any combination (default
        ``x='z', y='r'``, the real-space projection). Velocity and energy are
        finite-differenced from the recorded positions (the same derivation the
        trajectory colouring uses), so velocity axes look jagged near impacts.

        Selection (``epk_i`` / ``phi_i`` / ``traj``), ``trail``, ``step``,
        ``fps``, ``save`` (``.gif`` / ``.mp4``), ``embed`` and ``inline_format``
        behave as in :meth:`animate_trajectories`. ``zoom='auto'`` (default) uses
        one fixed box around all selected points; ``zoom='follow'`` is an
        **adaptive camera** that re-frames each axis onto the particles still
        alive every frame (smoothed by ``zoom_smooth``). Returns the
        :class:`~matplotlib.animation.FuncAnimation`."""
        # Deferred: pulls in matplotlib.animation (only needed here).
        from cavsim2d.analysis.multipacting.plots import phase_space_animation
        return phase_space_animation(self, x=x, y=y, epk_i=epk_i, phi_i=phi_i,
                                     traj=traj, color_by=color_by, trail=trail,
                                     step=step, fps=fps, save=save, dpi=dpi,
                                     progress=progress, embed=embed,
                                     inline_format=inline_format, zoom=zoom,
                                     zoom_smooth=zoom_smooth)
