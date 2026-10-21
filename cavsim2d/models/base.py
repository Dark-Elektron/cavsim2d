import re
from abc import ABC, abstractmethod
from pathlib import Path
from cavsim2d.geometry.tangency import update_alpha
from cavsim2d.processes.uq import ngsolve_mevp
from cavsim2d.utils.printing import CavsimWarning, error, info, warning
from cavsim2d.utils.quadrature import weighted_mean_obj
import json
import numpy as np
import os
import pandas as pd
import shutil
import warnings
from cavsim2d.processes.tune import last_stage_result
from cavsim2d.solvers.eigenmode_result import pol_name, pol_number, monopole_dir
from cavsim2d.solvers.solver_objects import (TuneSolver, EigenmodeSolver, WakefieldSolver,
                                             MultipactingSolver, config_sample)
from cavsim2d.utils.integrity import (describe_change, geometry_fingerprint,
                                      is_cavity_workspace, stale_units, warn_if_stale)
from cavsim2d.data_module.operating_points import bunch_lengths, bunch_tag
from cavsim2d.models.abci_deck import AbciDeck
from cavsim2d.models.abci_deck import (ABCI_DEFAULT_STEP, abci_bunch_length,  # noqa: F401
                                       abci_mesh_steps)  # (re-exported: they lived here)
from cavsim2d.models.cavity_plotting import CavityPlotting


def _result_alias(attr, key):
    """A deprecated, read-only cavity attribute that reads ``cav.eigenmode.qois[key]``.

    ``cav.R_Q``, ``cav.ff`` and the like were copies of the eigenmode results that
    only some run paths refreshed, so they went stale; the Bpk/Eacc copy,
    ``cav.b``, also overwrote the semi-axis ``b`` of an elliptical cell.
    """
    def read(self):
        warnings.warn(f"cav.{attr} is deprecated; read cav.eigenmode.qois[{key!r}].",
                      DeprecationWarning, stacklevel=2)
        return (self.eigenmode_qois or {}).get(key, 0)
    return property(read, doc=f"Deprecated: ``cav.eigenmode.qois[{key!r}]``.")


class Cavity(CavityPlotting, AbciDeck, ABC):
    """
    Command Line Interface module for running analysis.

    .. note::

       Still under development so some functions might not work properly
    """

    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k',
                 plot_label=None, geo_filepath=None, chain=1,
                 spacing=None):
        """
        Initialise cavity object. You can either specify geometry by dimensions
        (n_cells, mid_cell, end_cell_left, end_cell_right, etc.) *or* by providing
        a path to a geometry file (`geo_filepath`). If `geo_filepath` is not None,
        we load that file and skip the dimension‐based setup.

        Parameters
        ----------
        n_cells: int
            Number of cells (ignored if geo_filepath is provided)
        mid_cell: list or ndarray
            Mid‐cell geometric parameters (ignored if geo_filepath is provided)
        end_cell_left: list or ndarray
            Left end‐cell geometric parameters (ignored if geo_filepath is provided)
        end_cell_right: list or ndarray
            Right end‐cell geometric parameters (ignored if geo_filepath is provided)
        beampipe: {'none', 'both', 'left', 'right'}
            Beampipe options (ignored if geo_filepath is provided)
        name: str
            Name of the cavity
        cell_parameterisation: {'simplecell', 'flattop', ...}
            Parameterisation approach (ignored if geo_filepath is provided)
        color: str
            Colour for plotting
        plot_label: str or None
            Label for plotting; defaults to `name` if None
        geo_filepath: str or None
            If given, load geometry from this file instead of using dimensions.
        chain: int or dict
            Number of copies of this base cavity chained end-to-end into a
            **module** (default 1 = a single cavity). ``profile()`` repeats the
            wall this many times; adjacent beam pipes merge into the inter-cavity
            drift, so the module is one connected structure. Works for every model
            that builds a native ``profile()``. May also be a dict that bundles the
            count with the drift, ``chain={'count': 4, 'spacing': 60}`` — equivalent
            to ``chain=4, spacing=60``.
        spacing: float or sequence of float or None
            Inter-cavity straight-drift length(s) **in mm**, used only when
            ``chain > 1`` (may instead be given inside the ``chain`` dict). Controls
            the gap between adjacent cavities directly (iris-to-iris): the internal
            beam-pipe stubs are disregarded and the gap is set to this value, so the
            drift can be tuned/optimised independently of the base cavity's own
            beam-pipe length. A single value sets every gap; a list sets each of the
            ``chain - 1`` gaps (its length must then equal ``chain - 1``). ``None``
            (default) keeps the base cavity's beam pipes, so the drift is their
            natural ``2 * L_bp``.
        """

        # ───────────────────────────────────────────────────────────────────────────────────────────
        #  1) Set all “always‐present” attributes first (so nothing else breaks).
        # ───────────────────────────────────────────────────────────────────────────────────────────

        self.self_dir = None
        self.geo_filepath = None
        self.uq_dir = None
        # `chain` is either a copy count (int) or a dict bundling the count with the
        # inter-cavity drift, e.g. chain={'count': 4, 'spacing': 60}. The dict form is
        # unpacked here; `spacing` may still be given as its own kwarg (the dict wins
        # when it sets the gap).
        if isinstance(chain, dict):
            spec = dict(chain)
            count = spec.pop('count', 1)
            gap = spec.pop('spacing', None)
            if spec:
                raise ValueError(
                    "chain dict accepts only 'count' and 'spacing'; "
                    "unknown key(s): %s" % list(spec))
            if gap is not None:
                spacing = gap
            chain = count
        # Number of copies of this base cavity chained into a module (1 = single
        # cavity). profile() repeats the wall this many times; see Profile.chained.
        self.chain = max(1, int(chain or 1))
        # Inter-cavity drift length(s) in mm for chaining (None = keep beam pipes).
        self.spacing = spacing

        # Solver-as-object: lazy properties (see bottom of class)
        self._tune_solver = None
        self._eigenmode_solver = None
        self._wakefield_solver = None
        self._multipacting_solver = None
        self._tuned_cavity = None

        self.Ez_0_abs = {'z(0, 0)': [], '|Ez(0, 0)|': []}
        self.uq_nodes = None
        self.convergence_df_data = None
        self.convergence_df = None
        self.uq_weights = None
        self.V_rf_config = 0
        self.Eacc_rf_config = 0
        self.rf_performance_qois_uq = {}
        self.rf_performance_qois = {}
        self.uq_hom_results = None
        self.uq_fm_results = None
        self.mesh = None
        self.shape = None
        self.shape_multicell = None

        # plot_label: default to `name` if not provided
        self.name = name
        self.plot_label = plot_label if plot_label is not None else name

        self.projectDir = None
        self.bc = 33
        self.custom_eig_qois = {}
        self.wake_op_points = {}
        self.convergence_list = []
        self.operating_points = None
        self.Q0 = None
        self.inv_eta = None
        self.neighbours = {}
        self.wall_material = None

        # wakefield results placeholders
        (self.k_fm,
         self.k_loss,
         self.k_kick,
         self.phom,
         self.sigma,
         self.I0) = [{} for _ in range(6)]

        self.geo_filepath = geo_filepath

        # ───────────────────────────────────────────────────────────────────────────────────────────
        #  2) Choose initialisation path: “from file” or “from dimensions”
        # ───────────────────────────────────────────────────────────────────────────────────────────
        self.parameters = {}

    # ─── Results folder ───────────────────────────────────────────────────
    # A tuned cavity lives in <parent>/tuned/ but keeps its parent's name and
    # project, and every model's create() rebuilds self_dir as
    # <projectDir>/<name>: the parent's folder. An eigenmode run on cav.tuned then
    # wrote the tuned geometry's results over the parent's (elliptical cavities
    # with a .geo were spared only by create()'s early return). A pinned folder
    # survives that assignment, in custom models' create() too; moving the cavity
    # on purpose (set_workspace, save, load, Study.add_cavity) unpins it.

    @property
    def self_dir(self):
        """This cavity's results folder (None until it has one)."""
        return self.__dict__.get('_self_dir')

    @self_dir.setter
    def self_dir(self, value):
        pinned = self.__dict__.get('_pinned_self_dir')
        self.__dict__['_self_dir'] = pinned if (pinned and value is not None) else value

    def _pin_folder(self, folder):
        """Keep this cavity in *folder* whatever create() computes."""
        self.__dict__['_pinned_self_dir'] = str(folder)
        self.self_dir = str(folder)

    def _unpin_folder(self):
        self.__dict__['_pinned_self_dir'] = None

    # ─── Composition ──────────────────────────────────────────────────────

    #: Does this model contribute *accelerating cells* to a beam line?
    #: An :class:`~cavsim2d.models.assembly.Assembly` sums ``n_cells`` over the
    #: elements that say yes, and that sum is what picks the monopole mode of
    #: interest (the pi-mode of an n-cell structure). Passive elements — a beam
    #: pipe, a bellows, an absorber — carry ``n_cells = 1`` for the rest of the
    #: machinery but must not shift that mode index, so they set this False.
    contributes_cells = True

    #: Set by :mod:`cavsim2d.models.assembly` when it is imported. Registering it
    #: rather than importing it here keeps the dependency one-way: the assembly
    #: knows about the base class, not the other way round.
    _assembly_class = None

    def __add__(self, other):
        """``a + b`` concatenates two devices into an
        :class:`~cavsim2d.models.assembly.Assembly`.

        The result is itself a :class:`Cavity`, so it meshes, solves, tunes and
        optimises like any single geometry, and ``a + b + c`` flattens into one
        three-element line rather than nesting. Use ``chain=`` instead to repeat
        *one* cavity into a module — that path is unchanged.
        """
        if not isinstance(other, Cavity):
            return NotImplemented
        if Cavity._assembly_class is None:
            raise RuntimeError(
                'Assembly support is not loaded; import cavsim2d.models (or the '
                'top-level cavsim2d package) before concatenating devices.')
        return Cavity._assembly_class([self, other])

    # ─── Chaining helpers ─────────────────────────────────────────────────

    def _spacing_m(self):
        """Inter-cavity drift length(s) in **metres** for chaining, or ``None``.

        Reads ``self.spacing`` (mm) and converts to metres so it can be handed
        straight to :meth:`Profile.chained`. Returns ``None`` when unset, so chaining
        keeps the base cavity's own beam pipes.
        """
        g = getattr(self, 'spacing', None)
        if g is None:
            return None
        if isinstance(g, (list, tuple, np.ndarray)):
            return [float(x) * 1e-3 for x in g]
        return float(g) * 1e-3

    def _chained(self, prof):
        """Apply this cavity's ``chain`` / ``spacing`` to a base *prof*.

        Every model's ``profile()`` returns ``self._chained(base_profile)`` so the
        module-building logic (and its unit handling) lives in one place.
        """
        if prof is None:
            return None
        return prof.chained(getattr(self, 'chain', 1), spacing=self._spacing_m())

    # ─── Solver-as-object properties ──────────────────────────────────────

    @property
    def tune(self):
        """TuneSolver object for this cavity."""
        if self._tune_solver is None:
            self._tune_solver = TuneSolver(self)
        return self._tune_solver

    @property
    def eigenmode(self):
        """EigenmodeSolver object for this cavity."""
        if self._eigenmode_solver is None:
            self._eigenmode_solver = EigenmodeSolver(self)
        return self._eigenmode_solver

    @property
    def wakefield(self):
        """WakefieldSolver object for this cavity."""
        if self._wakefield_solver is None:
            self._wakefield_solver = WakefieldSolver(self)
        return self._wakefield_solver

    @property
    def multipacting(self):
        """MultipactingSolver object for this cavity."""
        if self._multipacting_solver is None:
            self._multipacting_solver = MultipactingSolver(self)
        return self._multipacting_solver

    # ─── Results, read through the solver namespaces ──────────────────────
    # These were plain attributes that only run_eigenmode()/run_wakefield() and
    # the Study filled in, so after cav.eigenmode.run() cav.eigenmode_qois stayed
    # empty. Reading the namespaces makes every run path agree.

    @property
    def eigenmode_qois(self):
        """QOIs of the monopole mode of interest: the same dict as
        ``cav.eigenmode.qois``."""
        return self.eigenmode.qois

    @property
    def eigenmode_qois_all_modes(self):
        """Every mode of every solved polarisation, keyed ``'<mode>-<m>'``."""
        all_modes = {}
        if not self.self_dir:
            return all_modes
        for pol in self._available_polarisations():
            for idx, qois in (self.eigenmode.mpole_qois(pol) or {}).items():
                all_modes[f"{idx}-{qois.get('m', pol_number(pol))}"] = qois
        return all_modes

    @property
    def wakefield_qois(self):
        """Loss and kick factors of the main wakefield run: the same dict as
        ``cav.wakefield.qois`` (and as ``study.wakefield_qois[name]``)."""
        return self.wakefield.qois

    @property
    def wakefield_qois_op(self):
        """Operating-point wakefield QOIs (P_HOM, loss and kick factors per
        operating point and bunch length): ``cav.wakefield.qois_op``."""
        return self.wakefield.qois_op

    @property
    def tune_results(self):
        """The tune record (tuned parameters and frequency per stage): the same
        dict as ``cav.tune.qois``."""
        return self.tune.qois

    @property
    def freq(self):
        """The cavity's frequency [MHz]: the eigenmode result when there is one,
        otherwise the frequency a tune set (0 when neither has)."""
        value = (self.eigenmode_qois or {}).get('freq [MHz]')
        return self.__dict__.get('_freq', 0) if value is None else value

    @freq.setter
    def freq(self, value):
        self.__dict__['_freq'] = value

    R_Q = _result_alias('R_Q', 'R/Q [Ohm]')
    GR_Q = _result_alias('GR_Q', 'GR/Q [Ohm^2]')
    G = _result_alias('G', 'G [Ohm]')
    Q = _result_alias('Q', 'Q []')
    ff = _result_alias('ff', 'ff [%]')
    k_cc = _result_alias('k_cc', 'kcc [%]')
    e = _result_alias('e', 'Epk/Eacc []')
    Epk_Eacc = _result_alias('Epk_Eacc', 'Epk/Eacc []')
    Bpk_Eacc = _result_alias('Bpk_Eacc', 'Bpk/Eacc [mT/MV/m]')

    @property
    def tuned(self):
        """Separate Cavity object living in ``<self_dir>/tuned/``.

        Lazy: if the tuned folder exists on disk we instantiate the tuned
        cavity from it on first access. Returns None when the cavity has
        not been tuned yet. Warns when the tune started from a geometry other
        than the one this cavity has now: that tuned version belongs to the
        earlier geometry, so ``study.tuned`` leaves it out and the next run on
        this cavity offers to delete it.
        """
        if self.self_dir is not None:
            warn_if_stale(self, Path(self.self_dir) / 'tuned' / 'tune_info',
                          'The tuned cavity and its results')
        if self._tuned_cavity is not None:
            return self._tuned_cavity

        if self.self_dir is None:
            return None

        tuned_dir = Path(self.self_dir) / 'tuned'
        # Require proof of a completed tune — directory alone may be a
        # leftover from a failed stage.
        if not (tuned_dir / 'tune_info' / 'tune_res.json').exists():
            return None

        self._tuned_cavity = self._load_tuned_from_disk(tuned_dir)
        return self._tuned_cavity

    @tuned.setter
    def tuned(self, value):
        self._tuned_cavity = value

    @property
    def tuned_dir(self):
        """Path to ``<self_dir>/tuned/`` (folder for the tuned cavity)."""
        if self.self_dir is None:
            return None
        return str(Path(self.self_dir) / 'tuned')

    @property
    def tune_info_dir(self):
        """Path to ``<self_dir>/tuned/tune_info/`` (tuning convergence data)."""
        if self.self_dir is None:
            return None
        return str(Path(self.self_dir) / 'tuned' / 'tune_info')

    @property
    def eigenmode_dir(self):
        """Backward-compat: eigenmode folder path."""
        if self.self_dir is None:
            return None
        return str(Path(self.self_dir) / 'eigenmode')

    @property
    def wakefield_dir(self):
        """Backward-compat: wakefield folder path."""
        if self.self_dir is None:
            return None
        return str(Path(self.self_dir) / 'wakefield')

    #: Whether ``self.parameters`` keys carry per-cell suffixes (``Req_m``,
    #: ``Req_el``, ``Req_er``). The tuner uses this to decide whether to expand a
    #: bare tune variable like ``'Req'``. True for the elliptical families; the
    #: pillbox, gun, waveguide and spline use unsuffixed parameter names.
    uses_cell_suffixes = False

    #: Suffixes appended to per-cell parameter names when ``uses_cell_suffixes``.
    CELL_SUFFIXES = ('_m', '_el', '_er')

    def tune_variables(self):
        """The names this model accepts as tune variables.

        A model declares its tunable handles simply by exposing them in
        ``self.parameters`` — there is no central list to update when a new
        geometry is added. Only *scalar* parameters qualify, because tuning
        drives a single number with a secant iteration.

        When ``uses_cell_suffixes`` the bare name is reported (``Req_m`` ->
        ``Req``), since that is what a ``tune_config['cell_type']`` mapping
        names; the cell type supplies the suffix.

        Returns an empty set for a geometry that has no parameters at all
        (one imported from a mesh or CAD file), which is what lets the tuner
        say so instead of blaming the variable name.
        """
        names = set()
        for key, value in (self.parameters or {}).items():
            if not isinstance(key, str):
                continue
            if isinstance(value, bool) or not np.isscalar(value):
                continue
            if self.uses_cell_suffixes:
                for suffix in self.CELL_SUFFIXES:
                    if key.endswith(suffix):
                        names.add(key[:-len(suffix)])
                        break
                else:
                    names.add(key)
            else:
                names.add(key)
        return names

    def expand_variable(self, name):
        """Every parameter slot the variable *name* refers to.

        A bare name on a multi-cell elliptical cavity means "the same quantity in
        every cell", so ``'Req'`` expands to ``['Req_m', 'Req_el', 'Req_er']``.
        Everywhere else a name refers to exactly one slot.

        UQ used to expand by substring-matching the name against parameter keys,
        which found nothing for a spline's ``'p3_r'`` (silently yielding zero
        random variables) and quietly pulled the pillbox's ``'L_bp'`` in
        alongside ``'L'``.
        """
        if self.uses_cell_suffixes and not name.endswith(self.CELL_SUFFIXES):
            expanded = [f'{name}{s}' for s in self.CELL_SUFFIXES
                        if f'{name}{s}' in self.parameters]
            if expanded:
                return expanded
        if name in self.parameters or name in self.tune_variables():
            return [name]
        # Dielectric regions declare their own handles, '<material>:<field>'.
        if self._dielectric_slot(name) is not None:
            return [name]
        known = ', '.join(sorted(self.tune_variables() | self.dielectric_variables()))
        if not known:
            raise ValueError(
                f"{type(self).__name__} is not parameterised, so {name!r} cannot be "
                f"perturbed or tuned.")
        raise ValueError(
            f"Unknown variable {name!r} for {type(self).__name__}. "
            f"It accepts: {known}.")

    def get_tune_value(self, name):
        """Read the variable *name* — a geometry parameter, or a
        ``<material>:<field>`` dielectric variable. See :meth:`tune_variables`
        and :meth:`dielectric_variables`."""
        if self._dielectric_slot(name) is not None:
            return self.get_dielectric_value(name)
        return self.parameters[name]

    def set_tune_value(self, name, value):
        """Write the variable *name*. See :meth:`get_tune_value`."""
        if self._dielectric_slot(name) is not None:
            return self.set_dielectric_value(name, value)
        self.parameters[name] = value

    def half_cells(self):
        """The per-half-cell parameter array — the canonical multicell representation.

        Only the elliptical families carry independently varying cells; other
        geometries (pillbox, gun, waveguide, spline) have no half-cell decomposition.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no half-cell representation; "
            "only elliptical cavities decompose into per-cell parameters.")

    def set_half_cells(self, half_cells):
        """Install an explicit per-half-cell parameter array. See :meth:`half_cells`."""
        raise NotImplementedError(
            f"{type(self).__name__} has no half-cell representation; "
            "only elliptical cavities decompose into per-cell parameters.")

    def rebuild(self, parameters, beampipe=None):
        """Return a fresh, bare cavity of this type built from *parameters*.

        This is the one hook every concrete model owes the framework. It creates
        the object only — no directories, no geometry file, no result caches. The
        generic machinery that needs to reconstruct a cavity from a parameter dict
        (:meth:`clone_for_tuning` for tuning, :meth:`spawn` for UQ and
        optimisation) is built on top of it, so a model that implements ``rebuild``
        gets tuning, UQ and optimisation for free.

        ``parameters`` uses the same keys as ``self.parameters``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement rebuild(parameters), so it "
            f"cannot be reconstructed from a parameter dict. Tuning, UQ and "
            f"optimisation all need this. Implement "
            f"{type(self).__name__}.rebuild(parameters, beampipe=None) returning a "
            f"fresh instance built from those parameters."
        )

    #: Geometry held outside ``parameters``, which ``rebuild()`` cannot see.
    _LAYOUT_ATTRS = ('chain', 'spacing', 'beampipe_length')

    def _carry_layout_to(self, clone):
        """Copy chaining and the beam-pipe length onto a rebuilt *clone*.

        ``rebuild()`` works from the parameter dict, which holds neither, so the
        tuned cavity of a chained module, its UQ and sweep variants and a module
        reloaded by ``Study.load`` all came back as a single cavity.
        """
        for attr in self._LAYOUT_ATTRS:
            if hasattr(self, attr):
                setattr(clone, attr, getattr(self, attr))
        return clone

    def _carry_dielectrics_to(self, clone):
        """Copy this cavity's dielectric regions onto a rebuilt *clone*.

        ``rebuild()`` reconstructs a cavity from its **parameter dict**, and the
        regions are not in it — so every generic clone path (tuning, UQ, sweeps)
        would otherwise hand back a quietly *vacuum* copy of a dielectric-loaded
        cavity. Carrying them here keeps a sweep or a UQ study over a loaded
        geometry loaded.
        """
        for d in self.dielectrics:
            if not any(e['material'] == d['material'] for e in clone.dielectrics):
                clone.dielectrics.append(dict(d))
        return clone

    def clone_for_tuning(self, tuned_parameters, tuned_self_dir, beampipe=None):
        """A cavity of this type carrying ``tuned_parameters``, living in
        ``tuned_self_dir`` (typically ``<self_dir>/tuned/``).

        Generic: the only type-specific step is :meth:`rebuild`. The clone gets its
        own ``geometry/`` folder with a freshly written geometry, and empty result
        caches so later eigenmode/wakefield runs write into the clone's folders.
        """
        clone = self._carry_layout_to(self._carry_dielectrics_to(
            self.rebuild(tuned_parameters,
                         beampipe=self.beampipe if beampipe is None else beampipe)))
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone._pin_folder(tuned_self_dir)

        geo_dir = os.path.join(clone.self_dir, 'geometry')
        os.makedirs(geo_dir, exist_ok=True)
        clone.uq_dir = os.path.join(clone.self_dir, 'uq')
        # Native (profile()-based) models don't emit a gmsh .geo — the solver
        # meshes their profile() directly. Only write a .geo when the model
        # actually has a writer, so a custom geometry that provides just
        # profile() + rebuild() (no write_geometry) still tunes.
        writer = getattr(clone, 'write_geometry', None)
        if callable(writer):
            clone.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            writer(clone.parameters, clone.n_cells, clone.beampipe,
                   write=clone.geo_filepath)
        else:
            clone.geo_filepath = None
        clone._write_geometry_snapshot()
        return clone

    def _load_tuned_from_disk(self, tuned_dir):
        """Rebuild the tuned cavity from a persisted ``tuned/`` folder.

        Prefers the complete, self-consistent ``tune_info/tuned_parameters.json``
        (the full sibling-propagated parameter set the tuner converged on). Falls
        back to merging the FILTERED per-stage snapshots in ``tune_res.json`` only
        for older tuned folders written before that file existed — that merge is
        lossy (a mid-cell stage stores only ``*_m``, so a tuned ``Req_m`` lands on
        top of an untuned ``Req_el`` and ``_unify_equator_radius`` can wipe it),
        which is exactly why the complete dump is preferred.
        """
        params = dict(self.parameters)
        full_path = os.path.join(str(tuned_dir), 'tune_info', 'tuned_parameters.json')
        tune_res_path = os.path.join(str(tuned_dir), 'tune_info', 'tune_res.json')
        if os.path.exists(full_path):
            with open(full_path) as fh:
                tuned_params = json.load(fh)
            for k, v in tuned_params.items():
                if isinstance(v, (list, tuple)):
                    params[k] = list(v)
                    continue
                try:
                    params[k] = float(v)
                except (TypeError, ValueError):
                    params[k] = v
        elif os.path.exists(tune_res_path):
            with open(tune_res_path) as fh:
                tune_res = json.load(fh)
            last = last_stage_result(tune_res)
            if last and last.get('parameters'):
                for k, v in last['parameters'].items():
                    # Not every model is parameterised by scalars: a spline's
                    # control point is [z, r]. float() raises on those, and
                    # swallowing it silently dropped the tuned value.
                    if isinstance(v, (list, tuple)):
                        params[k] = list(v)
                        continue
                    try:
                        params[k] = float(v)
                    except (TypeError, ValueError):
                        pass

        clone = self._carry_layout_to(
            self._carry_dielectrics_to(self.rebuild(params, beampipe=self.beampipe)))
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone._pin_folder(tuned_dir)
        clone.uq_dir = os.path.join(clone.self_dir, 'uq')
        geo_dir = os.path.join(clone.self_dir, 'geometry')
        if os.path.exists(os.path.join(geo_dir, 'geodata.geo')):
            clone.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
        return clone


    @abstractmethod
    def create(self):
        # If a geometry file is provided, skip dimension‐based setup:
        if self.geo_filepath:
            # make directory paths — flat structure, no Cavities/ subfolder
            self.self_dir = str(Path(self.projectDir) / self.name)
            geo_dir = Path(self.self_dir) / 'geometry'
            geo_dir.mkdir(parents=True, exist_ok=True)

            self._init_from_geo(self.geo_filepath, str(Path(self.self_dir) / 'geometry'))
            self.get_geometric_parameters()

    def _init_from_geo(self, filepath, output_filepath, kind='geo'):
        """
        Load geometry from a file (e.g. a .geo)
        """
        self.step_geo, self.mesh, self.bcs = ngsolve_mevp.load_geo(filepath, output_filepath)

    def get_geometric_parameters(self):
        with open(self.geo_filepath, "r") as file:
            for line in file:
                match = re.match(r'\s*(\w+)\s*=\s*DefineNumber\[\s*([\d.eE+-]+)\s*,\s*Name\s*"Parameters/[^"]*"\s*\];',
                                 line)
                if match:
                    var_name, var_value = match.groups()
                    self.parameters[var_name] = float(var_value)

    # ─── Material regions ─────────────────────────────────────────────────

    @property
    def dielectrics(self):
        """The dielectric regions declared on this cavity (see
        :meth:`add_dielectric`). Lazily created so models that build instances
        without going through ``__init__`` still work."""
        if getattr(self, '_dielectrics', None) is None:
            self._dielectrics = []
        return self._dielectrics

    def add_dielectric(self, material, eps_r, *, tan_delta=0.0, z=None, r=None,
                       points=None, maxh=None, color=(1.0, 1.0, 0.0)):
        """Fill part of the cavity with a dielectric.

        Two shapes. ``z``/``r`` give an axisymmetric **rectangular ring** in the
        (z, r) meridian plane — an annular cylinder in 3D — the shape dielectric
        loading usually takes: a ceramic window, a beam-pipe liner, an absorber
        ring. ``points`` gives an arbitrary closed **polygon**, revolved into a
        solid of revolution, so a triangle becomes a **cone** — the shape a
        beam-tube HOM absorber actually takes, and a far better match to the pipe
        above cutoff than a flat ring.

        Parameters
        ----------
        material : str
            Name of the region. It becomes the mesh material name and the key in
            ``eigenmode_config['materials']``.
        eps_r : float
            Real relative permittivity, eps'. Magnetic materials (``mu_r``) are not
            modelled yet and are rejected rather than silently ignored.
        tan_delta : float, optional
            Loss tangent, ``eps''/eps'`` — default 0 (lossless). With the
            ``exp(+j w t)`` convention the permittivity is ``eps'(1 - j tan_delta)``.
            How it reaches Q is chosen by ``eigenmode_config['loss_model']``: by
            default the solver treats it perturbatively up to ``tan_delta = 1e-2``
            and switches to the full complex eigenproblem above that. See
            :func:`~cavsim2d.solvers.NGSolve.eigen_ngsolve.resolve_loss_model`.
        z, r : (lo, hi) pairs, **millimetres**
            Extent of a rectangular region, in the same units as every other
            cavity dimension. Clipped to the cavity, so a generous span is fine —
            e.g. ``z=(-1e4, 1e4)`` for "the full length". Mutually exclusive with
            *points*.
        points : sequence of (z, r), **millimetres**
            Vertices of an arbitrary closed polygon instead, in the meridian
            plane; the outline is closed automatically and its winding fixed. A
            triangle revolves into a cone. Mutually exclusive with *z*/*r*.
        maxh : float, optional
            Local mesh size **in millimetres** inside the region. A thin shell
            needs this: a 0.5 mm tube wall in a cavity meshed at the default
            ``h = 20`` would otherwise be crossed by a single element.
        color : (r, g, b), optional
            Face colour for the region, default **yellow** ``(1, 1, 0)``. Carried
            onto the meshed face so the dielectric shows up when the geometry or
            mesh is drawn (e.g. ``ngsolve.Draw``). Purely cosmetic — it does not
            affect the solve.

        Examples
        --------
        A 0.5 mm-thick quartz tube lining a 5 mm-diameter aperture, running the
        full length of the cavity::

            cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0), maxh=0.25)

        The same tube in a lossy ceramic::

            cav.add_dielectric('ceramic', 9.8, tan_delta=2e-4,
                               z=(-1e4, 1e4), r=(2.5, 3.0), maxh=0.25)

        A conical absorber pointing up the beam tube from the pipe end, on a pipe
        of radius 39 mm — the Marhauser HOM-absorber geometry::

            cav.add_dielectric('absorber', 10.0, tan_delta=0.3,
                               points=[(300, 0), (420, 39), (300, 39)], maxh=3)

        Notes
        -----
        Dielectrics are supported by the **eigenmode** solver only, and only on
        cavities with a native ``profile()``. Wakefield and multipacting raise on
        a cavity that has them rather than quietly solving the vacuum problem.
        """
        eps_r = float(eps_r)
        tan_delta = float(tan_delta or 0.0)
        if eps_r <= 0:
            raise ValueError(f"eps_r must be positive, got {eps_r!r}.")
        if tan_delta < 0:
            raise ValueError(
                f"tan_delta must be >= 0 (a negative loss tangent is gain, not "
                f"loss), got {tan_delta!r}.")
        if points is not None:
            if z is not None or r is not None:
                raise ValueError(
                    f"dielectric {material!r}: give either 'points' or 'z'/'r', "
                    "not both.")
        else:
            if z is None or r is None:
                raise ValueError(
                    f"dielectric {material!r}: give 'z' and 'r' (a rectangular "
                    "ring), or 'points' (an arbitrary polygon).")
            if len(z) != 2 or len(r) != 2:
                raise ValueError("'z' and 'r' must each be a (lo, hi) pair in mm, "
                                 f"got z={z!r}, r={r!r}.")
            if float(min(z)) == float(max(z)) or float(min(r)) == float(max(r)):
                raise ValueError(f"dielectric {material!r} has zero extent "
                                 f"(z={tuple(z)}, r={tuple(r)}); both spans must be "
                                 "non-degenerate, in mm.")
        if any(d['material'] == material for d in self.dielectrics):
            raise ValueError(f"cavity {self.name!r} already has a dielectric region "
                             f"named {material!r}; use clear_dielectrics() to reset.")
        entry = {
            'material': str(material),
            'eps_r': eps_r,
            'tan_delta': tan_delta,
            'maxh': None if maxh is None else float(maxh),
            'color': tuple(float(c) for c in color),
        }
        if points is not None:
            # Validated (>=3 vertices, r >= 0, non-zero area, winding) by
            # MaterialRegion when the profile is built.
            entry['points'] = [(float(a), float(b)) for a, b in points]
            entry['z'] = entry['r'] = None
        else:
            entry['points'] = None
            entry['z'] = (float(min(z)), float(max(z)))
            entry['r'] = (float(min(r)), float(max(r)))
        self.dielectrics.append(entry)
        return self

    def clear_dielectrics(self):
        """Remove every dielectric region, returning the cavity to all-vacuum."""
        self.dielectrics.clear()
        return self

    #: Fields of a dielectric region that UQ / sensitivity can perturb, written
    #: ``<material>:<field>``. ``wall`` is derived (``r_out - r_in``) and moves the
    #: inner radius, which is the physical case: a tube sits in the bore, so its
    #: outer diameter is fixed by the aperture and the wall thickness varies.
    DIELECTRIC_FIELDS = ('eps_r', 'tan_delta', 'r_in', 'r_out', 'wall', 'z_lo', 'z_hi')

    def dielectric_variables(self):
        """UQ variable names for the dielectric regions, ``<material>:<field>``.

        A cavity declares these the same way it declares geometry handles — by
        having the regions — so nothing central needs updating when a region is
        added. Empty for an all-vacuum cavity.
        """
        out = set()
        for d in self.dielectrics:
            for f in self.DIELECTRIC_FIELDS:
                # A polygon region has no (lo, hi) extent, so its geometric
                # handles do not exist; only the material fields are tunable.
                if d.get('points') and f not in ('eps_r', 'tan_delta'):
                    continue
                out.add('%s:%s' % (d['material'], f))
        return out

    def _dielectric_slot(self, name):
        """``(region dict, field)`` for a ``<material>:<field>`` name, or None."""
        if not isinstance(name, str) or ':' not in name:
            return None
        material, _, field = name.partition(':')
        if field not in self.DIELECTRIC_FIELDS:
            return None
        for d in self.dielectrics:
            if d['material'] == material:
                if d.get('points') and field not in ('eps_r', 'tan_delta'):
                    return None          # no rectangle handles on a polygon
                return d, field
        return None

    def get_dielectric_value(self, name):
        """Read a ``<material>:<field>`` dielectric variable."""
        slot = self._dielectric_slot(name)
        if slot is None:
            raise ValueError(
                f"{name!r} is not a dielectric variable of cavity {self.name!r} "
                f"(has {sorted(self.dielectric_variables()) or 'none'}).")
        d, field = slot
        if field in ('eps_r', 'tan_delta'):
            return d.get(field, 0.0)
        if field == 'wall':
            return d['r'][1] - d['r'][0]
        return {'r_in': d['r'][0], 'r_out': d['r'][1],
                'z_lo': d['z'][0], 'z_hi': d['z'][1]}[field]

    def set_dielectric_value(self, name, value):
        """Write a ``<material>:<field>`` dielectric variable."""
        slot = self._dielectric_slot(name)
        if slot is None:
            raise ValueError(
                f"{name!r} is not a dielectric variable of cavity {self.name!r} "
                f"(has {sorted(self.dielectric_variables()) or 'none'}).")
        d, field = slot
        value = float(value)
        if field == 'eps_r':
            if value <= 0:
                raise ValueError(f"{name} must stay positive, got {value!r}.")
            d['eps_r'] = value
        elif field == 'tan_delta':
            # A UQ/optimisation step that walks tan_delta negative would come back
            # as a negative Q, which reads as a solver bug rather than an out-of-
            # range input.
            if value < 0:
                raise ValueError(
                    f"{name} must stay >= 0 (a negative loss tangent is gain, not "
                    f"loss), got {value!r}.")
            d['tan_delta'] = value
        elif field == 'wall':
            # Hold the outer radius (set by the bore) and move the inner one.
            if not 0 < value < d['r'][1]:
                raise ValueError(
                    f"{name} must satisfy 0 < wall < r_out ({d['r'][1]:g} mm), "
                    f"got {value!r}.")
            d['r'] = (d['r'][1] - value, d['r'][1])
        elif field in ('r_in', 'r_out'):
            r = list(d['r'])
            r[0 if field == 'r_in' else 1] = value
            if not 0 <= r[0] < r[1]:
                raise ValueError(
                    f"{name}={value!r} gives a degenerate region (r={tuple(r)}); "
                    "need 0 <= r_in < r_out.")
            d['r'] = (r[0], r[1])
        else:
            z = list(d['z'])
            z[0 if field == 'z_lo' else 1] = value
            if z[0] >= z[1]:
                raise ValueError(
                    f"{name}={value!r} gives a degenerate region (z={tuple(z)}).")
            d['z'] = (z[0], z[1])
        return self

    def dielectric_values(self):
        """``{variable: value}`` for every dielectric variable — the row UQ records."""
        return {n: self.get_dielectric_value(n) for n in sorted(self.dielectric_variables())}

    def _dielectric_signature(self):
        """Flat numeric keys describing the dielectric regions.

        Folded into the geometry snapshot so that changing a permittivity — or
        adding/removing a region — invalidates cached results exactly the way a
        changed dimension does. Without this, editing ``eps_r`` and rerunning with
        ``rerun=False`` would silently serve the previous run's numbers.
        """
        sig = {}
        for d in self.dielectrics:
            m = d['material']
            sig[f'_dielectric:{m}:eps_r'] = d['eps_r']
            sig[f'_dielectric:{m}:tan_delta'] = d.get('tan_delta', 0.0)
            if d.get('points'):
                # A polygon has no (lo, hi) pair; flatten its vertices instead so
                # that moving one of them still invalidates cached results.
                for i, (zv, rv) in enumerate(d['points']):
                    sig[f'_dielectric:{m}:z{i}'] = zv
                    sig[f'_dielectric:{m}:r{i}'] = rv
            else:
                sig[f'_dielectric:{m}:z0'], sig[f'_dielectric:{m}:z1'] = d['z']
                sig[f'_dielectric:{m}:r0'], sig[f'_dielectric:{m}:r1'] = d['r']
        return sig

    def _write_geometry_snapshot(self):
        """Persist current ``self.parameters`` next to the geometry file.

        Written as ``<self_dir>/geometry/parameters.json`` so subsequent
        analyses can detect when the in-memory cavity no longer matches
        the cached simulation results. A companion ``model.json`` records
        enough to *reconstruct* the cavity from disk (:meth:`Study.load`).

        Skipped while the folder holds results computed for a different
        geometry: the snapshot keeps describing those until a run replaces them
        (after asking), or ``Study.load`` would rebuild a cavity the results do
        not belong to.
        """
        if self.self_dir is None:
            return
        if stale_units(self):
            return
        geo_dir = Path(self.self_dir) / 'geometry'
        try:
            geo_dir.mkdir(parents=True, exist_ok=True)
            with open(geo_dir / 'parameters.json', 'w') as f:
                json.dump({**dict(self.parameters), **self._dielectric_signature()},
                          f, indent=2, default=str)
        except Exception:
            # Snapshot is advisory — never block geometry writes on IO.
            pass
        try:
            with open(geo_dir / 'model.json', 'w') as f:
                json.dump(self._reconstruct_state(), f, indent=2, default=str)
        except Exception as e:
            warning(f"Could not write {geo_dir / 'model.json'} ({e!r}); Study.load "
                    f"will not be able to rebuild cavity {self.name!r}.")

    def _reconstruct_state(self):
        """The state :meth:`Study.load` needs to rebuild this cavity from disk:
        the model class + the attributes :meth:`rebuild` reads, plus parameters."""
        return {
            'model': type(self).__name__,
            'module': type(self).__module__,
            'n_cells': int(getattr(self, 'n_cells', 1) or 1),
            'beampipe': getattr(self, 'beampipe', 'none'),
            'cell_parameterisation': getattr(self, 'cell_parameterisation', 'simplecell'),
            'name': self.name,
            'parameters': dict(self.parameters),
            'dielectrics': [dict(d) for d in self.dielectrics],
            **{attr: getattr(self, attr) for attr in self._LAYOUT_ATTRS if hasattr(self, attr)},
        }

    @classmethod
    def _reconstruct_from_state(cls, state):
        """Rebuild a bare cavity of this type from a :meth:`_reconstruct_state`
        dict. Built on :meth:`rebuild` (the one hook every model implements), so
        it is generic — a stub carrying just the attributes ``rebuild`` reads is
        enough to produce a fresh, fully-typed instance."""
        stub = cls.__new__(cls)
        stub.n_cells = state.get('n_cells', 1)
        stub.beampipe = state.get('beampipe', 'none')
        stub.cell_parameterisation = state.get('cell_parameterisation', 'simplecell')
        stub.name = state.get('name', 'cavity')
        stub.color = state.get('color', 'k')
        stub.plot_label = state.get('plot_label', None)
        cav = stub.rebuild(dict(state['parameters']), beampipe=stub.beampipe)
        cav.name = state.get('name', cav.name)
        for attr in cls._LAYOUT_ATTRS:
            if attr in state:
                setattr(cav, attr, state[attr])
        for d in state.get('dielectrics', ()) or ():
            if d.get('points'):
                cav.add_dielectric(d['material'], d['eps_r'],
                                   tan_delta=d.get('tan_delta', 0.0),
                                   points=d['points'], maxh=d.get('maxh'),
                                   color=d.get('color', (1.0, 1.0, 0.0)))
            else:
                cav.add_dielectric(d['material'], d['eps_r'],
                                   tan_delta=d.get('tan_delta', 0.0),
                                   z=d['z'], r=d['r'], maxh=d.get('maxh'),
                                   color=d.get('color', (1.0, 1.0, 0.0)))
        return cav

    def set_name(self, name):
        """
        Set cavity name

        Parameters
        ----------
        name: str
            Name of cavity

        Returns
        -------

        """
        self.name = name

    def set_color(self, color):
        """
        Set cavity name

        Parameters
        ----------
        color: str
            color of cavity

        Returns
        -------

        """
        self.color = color

    def set_parameterisation(self, cell_parameterisation):
        """
        Set cavity name

        Parameters
        ----------
        name: str
            Name of cavity

        Returns
        -------

        """
        self.cell_parameterisation = cell_parameterisation
        self._set_shape_key('CELL PARAMETERISATION', cell_parameterisation)

    def set_plot_label(self, plot_label):
        """
        Set cavity plot label

        Parameters
        ----------
        plot_label: str
            Cavity plot label

        Returns
        -------

        """

        if plot_label is None:
            self.plot_label = self.name
        else:
            self.plot_label = plot_label

    def set_n_cells(self, n_cells):
        """
        Sets number of cells of cavity

        Parameters
        ----------
        n_cells: int
            Number of cavity cells

        Returns
        -------

        """
        self.n_cells = int(n_cells)
        self._set_shape_key('n_cells', self.n_cells)

    def set_mid_cell(self, cell):
        """
        Set mid cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.mid_cell = cell
        self.shape['IC'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_end_cell_left(self, cell):
        """
        Set left end cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.end_cell_left = cell
        self.shape['OC'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_end_cell_right(self, cell):
        """
        Set right end cell geometric parameters of cavity

        Parameters
        ----------
        cell: list, array like
            Geometric parameters of cells

        Returns
        -------

        """
        self.end_cell_right = cell
        self.shape['OC_R'] = update_alpha(cell, self.cell_parameterisation)
        self.to_multicell()

    def set_boundary_conditions(self, bc):
        """
        Sets boundary conditions for the beampipes of cavity

        Parameters
        ----------
        bc: int
            Boundary condition of left and right cell/beampipe ends

        Returns
        -------

        """
        self.bc = int(bc)

    def set_beampipe(self, bp):
        """
        Set beampipe option of cavity

        Parameters
        ----------
        bp: str
            Beampipe option of cell

        Returns
        -------

        """
        self.beampipe = bp
        self._set_shape_key('BP', bp)

    # ─── Standalone workspace ────────────────────────────────────────────
    # A cavity can be analysed on its own, without a Study: the first analysis
    # provisions a workspace folder (named after the cavity, in the CWD) and the
    # per-cavity solvers (cav.eigenmode/tune/wakefield) write there. save()/load()
    # relocate or reopen it.

    def set_workspace(self, folder):
        """Use *folder* as this cavity's results directory and write its geometry.

        This is what ``Study.add_cavity`` does under the hood; call it directly to
        analyse a cavity standalone (``cav.set_workspace('runs/tesla')``). Warns
        when *folder* already holds results computed for a different geometry."""
        self._attach(folder)
        self._notice_stale_results()
        return self

    def _attach(self, folder):
        folder = str(folder)
        self._unpin_folder()
        self.projectDir = str(Path(folder).parent)
        self.name = Path(folder).name
        self.self_dir = folder
        Path(folder).mkdir(parents=True, exist_ok=True)
        self.create()

    def _notice_stale_results(self):
        """Warn when this cavity's folder already holds results computed for a
        different geometry: a name reused for another cavity, or this cavity
        changed since they were run."""
        stale = stale_units(self)
        if not stale:
            return
        current = geometry_fingerprint(self)
        listing = '; '.join(f"{unit.label} ({describe_change(stamp.get('geometry'), current)})"
                            for unit, stamp in stale)
        warnings.warn(
            f"{self.self_dir} already holds results computed for a different geometry "
            f"than cavity {self.name!r}: {listing}. They are kept for now; the next "
            f"analysis run on this cavity asks before deleting them. To keep both, "
            f"give this cavity another name.", CavsimWarning, stacklevel=3)

    def _ensure_workspace(self):
        """Provision a default workspace (``./<name>/``) if none is set, so a
        standalone cavity can be analysed without a Study. No-op once ``self_dir``
        exists (the Study path sets it first). Results already there for another
        geometry are left to the run's own check, which asks about them."""
        if not self.self_dir:
            self._attach(os.path.join(os.getcwd(), self.name))
        return self.self_dir

    def save(self, folder, overwrite=False):
        """Copy this cavity's workspace (geometry + all results) to *folder* and
        continue writing there. Persists a standalone run to a chosen location.

        ``overwrite=True`` replaces *folder* only if it is empty or a cavsim2d
        cavity folder; anything else raises instead of being deleted."""
        self._ensure_workspace()
        dst = str(folder)
        if os.path.abspath(dst) == os.path.abspath(self.self_dir):
            return self
        if os.path.exists(dst):
            if not overwrite:
                error(f"{dst} already exists — pass overwrite=True to replace it.")
                return self
            if not is_cavity_workspace(dst):
                raise ValueError(
                    f"{dst} is not a cavsim2d cavity folder, so overwrite=True will not "
                    f"delete it. Choose another folder, or empty it first.")
            shutil.rmtree(dst)
        shutil.copytree(self.self_dir, dst)
        self.set_workspace(dst)
        return self

    def load(self, folder):
        """Reopen a previously-saved cavity workspace: point this cavity at
        *folder* so ``cav.eigenmode.qois`` etc. read the results already there.
        Warns when those results were computed for a different geometry than
        this cavity object has."""
        folder = str(folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"No cavity workspace at {folder}.")
        self._unpin_folder()
        self.projectDir = str(Path(folder).parent)
        self.name = Path(folder).name
        self.self_dir = folder
        self._notice_stale_results()
        return self

    def study_mesh_convergence(self, eigenmode_config=None, h=10, p=2, p_passes=3, p_step=1,
                               n_modes=10, polarisation=('monopole', 'dipole'),
                               tol=1e-12, max_refinements=8, max_ndof=100000, theta=None):
        """Mesh-convergence sweep: adaptive h-refinement at each polynomial order.

        For every polynomial order ``p`` (``p_passes`` of them, stepping by
        ``p_step``) the cavity is solved on an **adaptively refined** mesh that
        starts from ``maxh = h``. Rather than a fixed geometric h-sequence, each
        refinement is error-driven (monopole error; see the solver's
        ``mesh_config['adaptive']``), and the DOF count is recorded at every
        refinement level so error-vs-DOF converges along the optimal path.

        The result keeps the full per-mode layout: ``convergence_df_data`` has
        one row per (order, refinement level, polarisation, mode) with all the
        QOIs (``freq [MHz]``, ``Q []``, ``R/Q [Ohm]``, …) plus each mode's own
        ``max_err``. ``h_pass`` is the refinement level (0-based), ``No of DOFs``
        the monopole DOF count there, and ``mode_index`` is ``'<m>-<mode>'``.

        Parameters
        ----------
        h : float
            Starting maximum element size (mm) for the h-refinement.
        p, p_passes, p_step : int
            Polynomial order to start at, how many orders to sweep, and the step.
        n_modes : int
            How many modes to solve for and report per polarisation (default 10).
        polarisation : sequence of str
            Which polarisations to evaluate on each level's mesh — any of
            'monopole', 'dipole', 'quadrupole', … (default monopole + dipole).
        tol : float
            Refinement stops once every solved mode's ``max_err`` is below this
            (default 1e-12).
        max_refinements, max_ndof : int
            Hard caps: stop once either is reached even if ``tol`` is not met.
        theta : float, optional
            Fraction of each mode's largest element error above which an element is
            refined (default 0.25). ``theta=0`` refines every element: uniform,
            nested refinement, each level a subdivision of the one before.
        """
        self._ensure_workspace()      # standalone: provision ./<name>/ if needed
        self.create()                 # write the geometry the mesher reads

        all_rows = []
        for ip in range(p_passes):
            p_now = p + ip * p_step
            cfg = dict(eigenmode_config) if eigenmode_config else {}
            cfg.setdefault('boundary_conditions', 'mm')
            cfg.setdefault('polarisation', list(polarisation))
            cfg.setdefault('n_modes', n_modes)
            cfg['mesh_config'] = {'h': h, 'p': p_now,
                                  'adaptive': {'tol': tol,
                                               'max_refinements': max_refinements,
                                               'max_ndof': max_ndof}}
            if theta is not None:
                cfg['mesh_config']['adaptive']['theta'] = float(theta)

            rows = ngsolve_mevp.solve_convergence(self, cfg)
            for row in rows:
                row['h'] = h
                row['p'] = p_now
                row['p_pass'] = ip
            all_rows.extend(rows)

        self.convergence_df_data = pd.DataFrame(all_rows)
        try:
            self.convergence_df = self.calculate_rel_errors(self.convergence_df_data)
        except Exception:
            self.convergence_df = self.convergence_df_data.copy()

    def calculate_rel_errors(self, df):
        # Level-to-level relative error of the numeric QOI columns. Metadata
        # (mesh size/order, the refinement/pass indices, timings, mode/pol
        # labels) is carried through unchanged, and text columns are skipped so
        # a 'polarisation' string never breaks the subtraction.
        metadata = ['h', 'p', 'h_pass', 'p_pass', 'refinement', 'mode_index',
                    'No of Mesh Elements', 'No of DOFs', 'time [s]', 'max_err']
        numeric = df.select_dtypes(include=[np.number])
        compute_cols = [c for c in numeric.columns if c not in metadata]
        df_to_compute = numeric[compute_cols]
        df_prev = df_to_compute.shift(1)
        relative_errors = (df_to_compute - df_prev).abs() / df_prev.abs()
        relative_errors.columns = [f'rel_error_{col}' for col in compute_cols]
        carried = [c for c in df.columns if c not in compute_cols]
        return pd.concat([relative_errors, df[carried]], axis=1)

    def run_eigenmode(self, solver='ngsolve', freq_shift=0, boundary_cond=None, subdir='', uq_config=None):
        """
        Run eigenmode analysis on cavity

        Parameters ---------- solver: {'SLANS', 'NGSolve'} Solver to be used. Native solver is still under
        development. Results are not as accurate as that of SLANS. freq_shift: Frequency shift. Eigenmode solver
        searches for eigenfrequencies around this value boundary_cond: int Boundary condition of left and right
        cell/beampipe ends subdir: str Sub directory to save results to uq_config: None | dict Provides inputs
        required for uncertainty quantification. Default is None and disables uncertainty quantification.

        Returns
        -------

        """

        # Delegate to the modern solver-object pipeline (works for every
        # cavity type via cav.create() + the NGSolve solve). ``config`` may be
        # passed positionally as the first argument (``solver``) as an
        # eigenmode_config dict, or built here from the legacy keyword args.
        config = solver if isinstance(solver, dict) else {}
        if boundary_cond:
            config.setdefault('boundary_conditions', boundary_cond)
        if freq_shift:
            config.setdefault('f_shift', freq_shift)
        if uq_config:
            config.setdefault('uq_config', uq_config)
        config.setdefault('processes', 1)
        config.setdefault('rerun', True)

        if 'polarisation' not in config.keys():
            config["polarisation"] = ['monopole']

        self.eigenmode.run(config)

        try:
            self.get_eigenmode_qois(config)
            if config.get('uq_config'):
                # UQ results are written to <cav>/uq/ (uq.json, uq_all_modes.json);
                # get_uq_fm_results takes that folder, not a file path.
                self.get_uq_fm_results(os.path.join(self.self_dir, 'uq'))
        except FileNotFoundError as e:
            error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")
        # The namespace, like every run: true when modes were computed.
        return self.eigenmode

    def run_wakefield(self, MROT=2, MT=10, wakelength=50, bunch_length=25,
                      DDR_SIG=0.1, DDZ_SIG=0.1, WG_M=None, marker='', operating_points=None, solver='ABCI'):
        """
        Run wakefield analysis on cavity

        Parameters
        ----------
        MROT: {0, 1}
            Polarisation 0 for longitudinal polarization and 1 for transversal polarization
        MT: int
            Number of time steps it takes for a beam to move from one mesh cell to the other
        wakelength:
            Wakelength to be analysed
        bunch_length: float
            Length of the bunch
        DDR_SIG: float
            Mesh to bunch length ration in the r axis
        DDZ_SIG: float
            Mesh to bunch length ration in the z axis
        WG_M:
            For module simulation. Specifies the length of the beampipe between two cavities.
        marker: str
            Marker for the cavities. Adds this to the cavity name specified in a shape space json file
        wp_dict: dict
            Python dictionary containing relevant parameters for the wakefield analysis for a specific operating point
        solver: {'ABCI'}
            Only one solver is currently available

        Returns
        -------
        :param MROT:
        :param wakelength:
        :param WG_M:
        :param marker:
        :param solver:
        :param operating_points:

        """

        # Delegate to the modern solver-object pipeline. ``MROT`` may be passed
        # positionally as a full wakefield_config dict, or the config is built
        # here from the legacy keyword args.
        config = MROT if isinstance(MROT, dict) else {
            'MROT': MROT, 'MT': MT, 'wakelength': wakelength,
            'bunch_length': bunch_length,
            'mesh_config': {'DDR_SIG': DDR_SIG, 'DDZ_SIG': DDZ_SIG},
        }
        if operating_points is not None:
            config.setdefault('operating_points', operating_points)
        config.setdefault('processes', 1)
        config.setdefault('rerun', True)

        self.wakefield.run(config)

        try:
            self.get_wakefield_qois(config)
        except FileNotFoundError:
            error("Could not find the wakefield results. Please rerun wakefield analysis.")
        return self.wakefield

    def run_tune(self, tune_config=None, **kwargs):
        """Tune this cavity: the same as ``cav.tune.run(...)``, which documents the
        settings. The tuned geometry becomes ``cav.tuned``; this cavity is not
        changed."""
        return self.tune.run(tune_config, **kwargs)

    def active_length(self):
        """Accelerating length [mm] that ``Eacc`` is normalised to.

        One definition shared by the eigenmode figures of merit (``Eacc``,
        ``Epk/Eacc``, ``Bpk/Eacc``) and the RF power budget (:attr:`l_active`), so
        the two can no longer disagree. ``None`` for a model with no cell structure
        to define it; the eigenmode solver then uses the on-axis field extent, or
        ``eigenmode_config['normalization_length']`` when given.
        """
        return None

    @property
    def l_active(self):
        """Active length [m] for the RF power budget: :meth:`active_length`."""
        length = self.active_length()
        return None if length is None else float(length) * 1e-3

    # Cavity._run_ngsolve (a dead pre-refactor staticmethod) was removed 2026-07-09:
    # it was never called (the live one is processes.eigenmode._run_ngsolve), it
    # referenced an undefined bare `OC_R`, and it drove the removed
    # uq_parallel_multicell / cavity_multicell pair.

    def set_wall_material(self, wm):
        self.wall_material = wm

    def optimise(self, optimisation_config, optimiser):
       optimiser(self, optimisation_config)

    def get_ngsolve_tune_res(self):
        """Load tune results from ``<self_dir>/tuned/tune_info/tune_res.json``.

        After the tune refactor, tune artefacts live alongside the tuned
        cavity — not inside the original cavity's ``eigenmode/`` folder.
        A legacy fallback is kept for older projects.

        The file is keyed by cell type (e.g. ``{'mid-cell': {...}}``); when
        multiple cell types were tuned the final stage's ``FREQ`` is used
        as the cavity's target frequency.
        """

        new_path = Path(self.self_dir) / 'tuned' / 'tune_info' / 'tune_res.json'
        legacy_path = Path(self.self_dir) / 'eigenmode' / 'tune_res.json'
        tune_res_path = new_path if new_path.exists() else legacy_path

        if tune_res_path.exists():
            with open(tune_res_path, 'r') as json_file:
                tune_res = json.load(json_file)
            last = last_stage_result(tune_res)
            if last is not None and 'FREQ' in last:
                self.freq = last['FREQ']
        else:
            error("Tune results not found. Please tune the cavity")

    def get_eigenmode_qois(self, config=None):
        """Reload the saved eigenmode results.

        The results are ``cav.eigenmode.qois`` (also ``cav.eigenmode_qois``) and
        ``cav.eigenmode_qois_all_modes``, read from disk whenever they are asked
        for. Raises ``FileNotFoundError`` when there is no eigenmode result at all.
        """
        # Which polarisations to check: those named in the config (a bare
        # 'dipole'/1 or a list are all accepted), or, when no config is given
        # (e.g. the Study-level caller), whatever was solved.
        raw = (config or {}).get('polarisation') if isinstance(config, dict) else None
        if raw is not None:
            if not isinstance(raw, (list, tuple, set)):
                raw = [raw]
            poles = [pol_name(pol_number(p)) for p in raw]
        else:
            poles = self._available_polarisations() or ['monopole']

        for pole in poles:
            qois_path = os.path.join(self._eigenmode_pol_dir(pole), 'qois.json')
            if not os.path.exists(qois_path):
                # No result for this polarisation. If others were solved (an
                # m-pole-only run), the monopole summary attributes simply aren't
                # defined; read those with cav.eigenmode.mpole_qois(). Only error
                # when there is no eigenmode result at all.
                available = self._available_polarisations()
                if available:
                    info(f"No {pole} eigenmode result for {self.name}; solved "
                         f"polarisations: {available}. Read them with "
                         f"cav.eigenmode.mpole_qois('<pol>').")
                    return
                raise FileNotFoundError(
                    f"No eigenmode result for {self.name} at {qois_path}. "
                    f"Run cav.eigenmode.run() first.")

        self.eigenmode._invalidate()

    def get_uq_fm_results(self, folder):
        # load uq result
        with open(os.path.join(folder, 'uq.json'), 'r') as json_file:
            self.uq_fm_results = json.load(json_file)

        if os.path.exists(os.path.join(folder, 'uq_all_modes.json')):
            with open(os.path.join(folder, 'uq_all_modes.json'), 'r') as json_file:
                self.uq_fm_results_all_modes = json.load(json_file)

        # get neighbours and all qois
        neighbours = {}
        cavities_dir = folder
        for dirr in os.listdir(cavities_dir):
            dirr_path = os.path.join(cavities_dir, dirr)
            if os.path.isdir(dirr_path) and 'Q' in dirr.split('_')[-1]:
                qois_path = os.path.join(monopole_dir(os.path.join(dirr_path, 'eigenmode')), 'qois.json')
                if os.path.exists(qois_path):
                    with open(qois_path, 'r') as json_file:
                        neighbour_uq_fm_results = json.load(json_file)
                    neighbours[dirr] = neighbour_uq_fm_results

        self.neighbours = pd.DataFrame.from_dict(neighbours, orient='index')

        # nodes, weights = cn_leg_05_2(7)
        # write weights

        # data_table = pd.DataFrame(weights, columns=['weights'])
        # data_table.to_csv(fr'{folder}\weights.csv', index=False, sep='\t', float_format='%.32f')

        # get nodes
        self.uq_nodes = pd.read_csv(os.path.join(folder, 'nodes.csv'), sep='\t')
        # get weights
        self.uq_weights = pd.read_csv(os.path.join(folder, 'weights.csv'))

    def get_uq_hom_results(self, folder):
        with open(folder, 'r') as json_file:
            self.uq_hom_results = json.load(json_file)

    def get_wakefield_qois(self, wakefield_config):
        """Fill the per-operating-point attributes (``k_fm``, ``k_loss``,
        ``k_kick``, ``phom``, ``I0``) from ``wakefield/qois_op.json``.

        The main run's loss/kick factors are ``cav.wakefield_qois`` (the same as
        ``cav.wakefield.qois``, always written); the operating-point table,
        ``cav.wakefield_qois_op``, exists only when ``operating_points`` were
        given, because it needs the machine parameters and a rerun per bunch
        length.
        """
        all_op = self.wakefield_qois_op
        # Truthy check (not just presence): the complete saved config carries
        # 'operating_points': None when unset. Matched on the exact sub-run ids,
        # not a substring: 'Z' is a substring of 'Z_b_2024_SR_4.32mm', so two
        # operating points sharing a prefix used to pull in each other's results.
        if not wakefield_config.get('operating_points'):
            return
        for op_pt, op_vals in wakefield_config['operating_points'].items():
            for label, sigma in bunch_lengths(op_vals).items():
                key = bunch_tag(op_pt, label, sigma)
                val = all_op.get(key)
                if val is None:
                    continue
                self.k_fm[key] = val['k_FM [V/pC]']
                self.k_loss[key] = val['|k_loss| [V/pC]']
                self.k_kick[key] = val['|k_kick| [V/pC/m]']
                self.phom[key] = val['P_HOM [kW]']
                self.I0[key] = val['I0 [mA]']

    def get_power_uq(self, rf_config, op_points_list):
        stat_moms = ['expe', 'stdDev', 'skew', 'kurtosis']
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois_uq[op_pt] = {}
            val = self.operating_points[op_pt]
            for sig_id, vv in bunch_lengths(val).items():
                if 'Eacc [MV/m]' in rf_config.keys():
                    op_field = {'expe': [self.Eacc_rf_config * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                else:
                    # All four moments: the loop below reads every one of them,
                    # and a bare {'expe', 'stdDev'} raised KeyError on 'skew'.
                    op_field = {'expe': [val['Eacc [MV/m]'] * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    self.Eacc_rf_config = val['Eacc [MV/m]']

                if 'V [GV]' in rf_config.keys():
                    v_rf = {'expe': [rf_config['V [GV]'][ii] * 1e9], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                else:
                    v_rf = {'expe': [val['V [GV]'] * 1e9], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}

                Q0 = {'expe': [self.Q0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                inv_eta = {'expe': [self.inv_eta], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                p_sr = {'expe': [rf_config['SR per turn [MW]'] * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                n_cav = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                p_in = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                p_cryo = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                pdyn = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                pstat = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                p_wp = {'expe': [0], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}

                # test
                n_cav['expe'][0] = int(np.ceil(v_rf['expe'][0] / (self.Eacc_rf_config * 1e6 * self.l_active)))

                # p_in = rf_config['SR per turn [MW]'] * 1e6 / n_cav * 1e-3  # maximum synchrotron radiation per beam

                p_cryo_n = 8 / (np.sqrt(self.neighbours['freq [MHz]'] / 500))  # W/m

                pdyn_n = v_rf['expe'][0] * (self.Eacc_rf_config * 1e6 * self.l_active) / (
                        self.neighbours['R/Q [Ohm]'] * self.Q0 * n_cav['expe'][0])  # per cavity
                pdyn_expe, pdyn_std, pdyn_skew, pdyn_kurtosis = weighted_mean_obj(
                    np.atleast_2d(pdyn_n.to_numpy()).T, self.uq_weights)
                pdyn = {'expe': pdyn_expe, 'stdDev': pdyn_std, 'skew': pdyn_skew, 'kurtosis': pdyn_kurtosis}

                pstat_n = (self.l_cavity * v_rf['expe'][0] / (
                        self.l_active * self.Eacc_rf_config * 1e6 * n_cav['expe'][0])) * p_cryo_n
                pstat_expe, pstat_std, pstat_skew, pstat_kurtosis = weighted_mean_obj(
                    np.atleast_2d(pstat_n.to_numpy()).T, self.uq_weights)
                pstat = {'expe': pstat_expe, 'stdDev': pstat_std, 'skew': pstat_skew, 'kurtosis': pstat_kurtosis}

                p_wp_n = self.inv_eta * (pdyn_n + pstat_n) * 1e-3  # per cavity
                p_wp_expe, p_wp_std, p_wp_skew, p_wp_kurtosis = weighted_mean_obj(
                    np.atleast_2d(p_wp_n.to_numpy()).T, self.uq_weights)
                p_wp = {'expe': p_wp_expe, 'stdDev': p_wp_std, 'skew': p_wp_skew, 'kurtosis': p_wp_kurtosis}

                for stat_mom in stat_moms:
                    if op_field[stat_mom][0] != 0:
                        # n_cav[stat_mom] = [
                        #     int(np.ceil(v_rf[stat_mom][0] / (op_field[stat_mom][0] * self.l_active)))]

                        p_in[stat_mom] = [
                            p_sr[stat_mom][0] / n_cav[stat_mom][0] * 1e-3]  # maximum synchrotron radiation per beam

                    # if self.uq_fm_results['freq [MHz]'][stat_mom][0] != 0:
                    #     p_cryo[stat_mom] = [
                    #         8 / (np.sqrt(self.uq_fm_results['freq [MHz]'][stat_mom][0] / 500))]  # W/m
                    #
                    # if self.uq_fm_results['R/Q [Ohm]'][stat_mom][0] * Q0[stat_mom][0] * n_cav[stat_mom][0] != 0:
                    #     pdyn[stat_mom] = [v_rf[stat_mom][0] * (op_field[stat_mom][0] * self.l_active) / (
                    #             self.uq_fm_results['R/Q [Ohm]'][stat_mom][0] * Q0[stat_mom][0] *
                    #             n_cav[stat_mom][0])]  # per cavity
                    #
                    # if op_field[stat_mom][0] != 0 and n_cav[stat_mom][0] != 0:
                    #     pstat[stat_mom] = [(self.l_cavity * v_rf[stat_mom][0] / (
                    #             self.l_active * op_field[stat_mom][0] * n_cav[stat_mom][0])) * p_cryo[stat_mom][
                    #                            0]]
                    #
                    # if inv_eta[stat_mom][0] != 0:
                    #     p_wp[stat_mom] = [
                    #         (inv_eta[stat_mom][0]) * (pdyn[stat_mom][0] + pstat[stat_mom][0]) * 1e-3]  # per cavity

                self.rf_performance_qois_uq[op_pt][sig_id] = {
                    r"Ncav": n_cav,
                    r"Q0 []": Q0,
                    r"Pstat/cav [W]": pstat,
                    r"Pdyn/cav [W]": pdyn,
                    r"Pwp/cav [kW]": p_wp,
                    r"Pin/cav [kW]": p_in,
                    r"PHOM/cav [kW]": self.uq_hom_results[f'P_HOM [kW]_{bunch_tag(op_pt, sig_id, vv)}']
                }
        return self.rf_performance_qois_uq

    def get_power(self, rf_config, op_points_list):
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois[op_pt] = {}
            val = self.operating_points[op_pt]
            for sig_id, vv in bunch_lengths(val).items():
                if 'Eacc [MV/m]' in rf_config.keys():
                    op_field = self.Eacc_rf_config * 1e6
                else:
                    op_field = val['Eacc [MV/m]'] * 1e6
                    # MV/m, like every other writer of this attribute. Storing
                    # op_field here (V/m) made the next reader scale it by 1e6
                    # a second time.
                    self.Eacc_rf_config = val['Eacc [MV/m]']

                if 'V [GV]' in rf_config.keys():
                    v_rf = rf_config['V [GV]'][ii] * 1e9
                else:
                    v_rf = val['V [GV]'] * 1e9

                Q0 = self.Q0
                inv_eta = self.inv_eta
                p_sr = rf_config['SR per turn [MW]'] * 1e6

                n_cav = int(np.ceil(v_rf / (op_field * self.l_active)))
                p_in = p_sr / n_cav  # maximum synchrotron radiation per beam

                p_cryo = 8 / (np.sqrt(self.freq / 500))

                pdyn = v_rf * (op_field * self.l_active) / (self.R_Q * Q0 * n_cav)

                pstat = (self.l_cavity * v_rf / (self.l_active * op_field * n_cav)) * p_cryo  # per cavity
                p_wp = (inv_eta) * (pdyn + pstat)  # per cavity

                self.rf_performance_qois[op_pt][sig_id] = {
                    r"Ncav": n_cav,
                    r"Q0 []": Q0,
                    r"Pstat/cav [W]": pstat,
                    r"Pdyn/cav [W]": pdyn,
                    r"Pwp/cav [kW]": p_wp * 1e-3,
                    r"Pin/cav [kW]": p_in * 1e-3,
                    r"PHOM/cav [kW]": self.phom[bunch_tag(op_pt, sig_id, vv)]
                }
        return self.rf_performance_qois

    def _eigenmode_pol_dir(self, pol='monopole'):
        """Folder holding eigenmode results for a polarisation:
        ``eigenmode/<pol name>/`` (monopole, dipole, ...). Monopole results
        written during the flat-layout refactor era live directly in
        ``eigenmode/``, so fall back to that when no monopole subfolder
        exists."""
        m = pol_number(pol)
        if m == 0:
            return monopole_dir(os.path.join(self.self_dir, 'eigenmode'))
        return os.path.join(self.self_dir, 'eigenmode', pol_name(m))

    def get_fields(self, mode=1, pol='monopole'):
        gfu_E, gfu_H = ngsolve_mevp.load_fields(self._eigenmode_pol_dir(pol), mode)
        return gfu_E, gfu_H

    def _available_polarisations(self):
        """Polarisation result folders that actually exist for this cavity
        (a 'monopole' entry covers the legacy flat layout too)."""
        eig = os.path.join(self.self_dir, 'eigenmode')
        avail = []
        if os.path.exists(os.path.join(self._eigenmode_pol_dir('monopole'), 'qois.json')):
            avail.append('monopole')
        if os.path.isdir(eig):
            for name in sorted(os.listdir(eig)):
                if name != 'monopole' and os.path.exists(os.path.join(eig, name, 'qois.json')):
                    avail.append(name)
        return avail

    def _eigenmode_artifact_error(self, pol, required_file, what):
        avail = self._available_polarisations()
        avail_msg = (f"Available: {avail}." if avail
                     else "No eigenmode results found — run run_eigenmode() first.")
        hint = ("" if avail == ['monopole'] else
                f" For higher-order modes pass pol='dipole'/'quadrupole'/… "
                f"and make sure that polarisation was solved "
                f"(eigenmode_config['polarisation']).")
        error(f"No {what} for polarisation '{pol}' "
              f"(looked for {required_file}). {avail_msg}{hint}")

    def define_operating_points(self, op):
        self.operating_points = op

    def config_sample(self, kind, show=True):
        """Show every configurable entry of a config, with its default.

        Prints the complete default config for *kind* — all available options with
        their default values — and returns it as a dict you can copy, edit and pass
        to the matching ``run(...)``. *kind* is one of ``'eigenmode'``,
        ``'wakefield'``, ``'tune'``, ``'multipacting'``, ``'uq'`` or
        ``'optimisation'``; an unknown one raises ``ValueError``.

        The defaults are the ones the solvers merge over, so this never drifts
        from what a run uses. Each solver namespace has the same without the
        printout: ``cav.tune.config_sample`` (or ``cfg_sample``,
        ``sample_config``, ``sample_cfg``). ``cfg_sample``, ``sample_config`` and
        ``sample_cfg`` are also other names for this method.
        """
        return config_sample(kind, show=show)

    cfg_sample = sample_config = sample_cfg = config_sample

    def _set_shape_key(self, key, value):
        """Record *key* in ``shape`` and rebuild ``shape_multicell`` from it."""
        if isinstance(self.shape, dict):
            self.shape[key] = value
        self.to_multicell()

    def to_multicell(self):
        """Rebuild ``shape_multicell`` from the elliptical cell dict in ``shape``.

        Only a model described by mid and end cells (``IC``, ``OC``, ``OC_R``) has
        one; for the others this does nothing. It used to raise ``KeyError``, so
        ``set_n_cells`` and ``set_beampipe`` failed on a pillbox, a spline or a gun.
        """
        if not (isinstance(self.shape, dict)
                and all(k in self.shape for k in ('IC', 'OC', 'OC_R', 'BP', 'n_cells'))):
            return
        mid_cell = self.shape['IC']

        self.shape_multicell = {}
        mid_cell_multi = np.array([[[a, a] for _ in range(self.n_cells - 1)] for a in mid_cell])

        self.shape_multicell['OC'] = self.shape['OC']
        self.shape_multicell['OC_R'] = self.shape['OC_R']
        self.shape_multicell['IC'] = mid_cell_multi
        self.shape_multicell['BP'] = self.shape['BP']
        self.shape_multicell['n_cells'] = self.shape['n_cells']
        self.shape_multicell['CELL PARAMETERISATION'] = 'multicell'
        self.shape_multicell['kind'] = self.kind
        self.shape_multicell['geo_file'] = None

    def _create_project(self, overwrite):
        project_name = self.name
        project_dir = self.projectDir

        if project_name != '':

            # check if folder already exist
            e = self._check_if_path_exists(project_dir, project_name, overwrite)

            if e:
                # Create project directory (flat — cavity objects go directly inside)
                proposed_path = Path(project_dir) / project_name
                try:
                    os.makedirs(str(proposed_path), exist_ok=True)
                    self.projectDir = str(proposed_path)
                    return True
                except Exception as e:
                    self.projectDir = str(proposed_path)
                    error("An exception occurred in created project: ", e)
                    return False
            else:
                self.projectDir = str(Path(project_dir) / project_name)
                return True
        else:
            info('\tPlease enter a valid project name')
            self.projectDir = str(Path(project_dir) / project_name)
            return False

    @staticmethod
    def _check_if_path_exists(directory, folder, overwrite=False):
        path = f"{directory}/{folder}"
        if os.path.exists(path):
            if overwrite:
                x = 'y'
            else:
                x = 'n'

            if x == 'y':
                try:
                    directory_list = os.listdir(path)

                    # Flat structure: just check it's a project folder
                    if len(directory_list) < 20:
                        shutil.rmtree(path)
                        return True
                    else:
                        info('\tIt seems that the folder specified is not a cavity project folder. Please check folder'
                             'again to avoid deleting important files.')
                        return False

                except Exception as e:
                    error("Exception occurred: ", e)
                    return False
            else:
                return False
        else:
            return True

    def __repr__(self):
        bits = [f"name={self.name!r}"]
        n_cells = getattr(self, 'n_cells', None)
        if n_cells is not None:
            bits.append(f"n_cells={n_cells}")
        beampipe = getattr(self, 'beampipe', None)
        if beampipe is not None:
            bits.append(f"beampipe={beampipe!r}")
        # a compact "what has run" status
        done = []
        if getattr(self, 'eigenmode_qois', None):
            done.append('eigenmode')
        if getattr(self, 'wakefield_qois', None):
            done.append('wakefield')
        if getattr(self, 'tune_results', None):
            done.append('tuned')
        if done:
            bits.append(f"results={'+'.join(done)}")
        return f"{type(self).__name__}({', '.join(bits)})"

    def __str__(self):
        p = dict()
        p[self.name] = {
            'tune': self.tune_results,
            'fm': self.eigenmode_qois,
            'hom': self.wakefield_qois,
            'uq': {
                'tune': 0,
                'fm': self.uq_fm_results,
                'hom': self.uq_hom_results
            }
        }
        return fr"{json.dumps(p, indent=4)}"

    # @abstractmethod
    def _row_slots(self, key):
        """Parameter slot(s) a spawn / UQ / optimisation row column writes to.

        A column is normally a parameter name, and writes to itself. The hook
        exists so a model whose variables need resolving — an
        :class:`~cavsim2d.models.assembly.Assembly`, where ``'cav:A'`` stands for
        three per-cell slots — can expand it. A column that resolves to nothing
        is ignored, which is what keeps unrelated bookkeeping columns out of the
        parameter dict.
        """
        return [key] if key in (self.parameters or {}) else []

    def spawn(self, difference, folder):
        """A container of perturbed cavities, one per row of *difference*.

        Generic, built on :meth:`rebuild`: each row is a partial parameter dict
        overlaid on this cavity's own parameters. Used by UQ and optimisation.

        (This used to call ``type(self)(name=..., geo_filepath=...)``, which no
        model's constructor accepts — every non-elliptical UQ died with
        ``TypeError: Pillbox.__init__() got an unexpected keyword argument 'name'``.)
        """
        # Deferred: breaks the base <-> cavities import cycle.
        from cavsim2d.study import Study
        spawn = Study(folder, _skip_project_init=True)
        os.makedirs(folder, exist_ok=True)

        for key, row in difference.iterrows():
            params = dict(self.parameters)
            for k, v in row.items():
                for slot in self._row_slots(k):
                    params[slot] = v

            scav = self._carry_layout_to(self._carry_dielectrics_to(self.rebuild(params)))
            # Dielectric variables are not in `parameters`, so rebuild() cannot
            # carry them; overlay them onto the clone's own regions.
            for k, v in row.items():
                if scav._dielectric_slot(k) is not None:
                    scav.set_dielectric_value(k, v)
            scav.name = str(key)
            # The clone inherits the template's plot_label, so a sweep of six
            # points used to draw six legend entries all reading "cavity".
            # The row key is what distinguishes them, so label by it unless the
            # template carries a label of its own.
            if self.plot_label in (None, self.name):
                scav.plot_label = str(key)
            scav.projectDir = folder
            scav.self_dir = os.path.join(folder, str(key))
            scav.uq_dir = os.path.join(scav.self_dir, 'uq')
            geo_dir = os.path.join(scav.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)

            # Native (profile()-based) models have no gmsh writer; the solver
            # meshes profile() directly. Only write a .geo when a writer exists,
            # so a custom geometry (profile() + rebuild() only) also spawns.
            writer = getattr(scav, 'write_geometry', None)
            if callable(writer):
                scav.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
                writer(scav.parameters, scav.n_cells, scav.beampipe,
                       write=scav.geo_filepath)
            else:
                scav.geo_filepath = None
            scav._write_geometry_snapshot()

            spawn.cavities_list.append(scav)
            spawn.cavities_dict[scav.name] = scav
            spawn.shape_space[scav.name] = getattr(scav, 'shape', None)
            spawn.shape_space_multicell[scav.name] = getattr(scav, 'shape_multicell', None)

        return spawn

    def update_geo_parameters(self, input_geo, output_folder, update_params):
        with open(input_geo, "r") as f:
            content = f.read()

        for name, new_value in update_params.items():
            # Pattern matches DefineNumber[...] form, capturing the part before and after the number
            pattern = rf"(\b{name}\s*=\s*DefineNumber\[)\s*[^,\]]+"
            replacement = rf"\g<1>{new_value}"
            content, n = re.subn(pattern, replacement, content)

            if n == 0:
                warning(f"Parameter '{name}' not found in {os.path.basename(input_geo)}.")
                info(fr"Here are available parameters: {(self.parameters.keys())}")

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(input_geo))

        with open(output_path, "w") as f:
            f.write(content)

        # print(f"Updated .geo file written to: {output_path}")
        return output_path


