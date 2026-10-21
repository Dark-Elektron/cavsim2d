from pathlib import Path
from cavsim2d.models.base import Cavity
from cavsim2d.processes.eigenmode import run_eigenmode_parallel, run_eigenmode_s
from cavsim2d.processes.tune import (last_stage_result, normalize_cell_type_config,
                                     resolve_tune_defaults, run_tune_parallel)
from cavsim2d.processes.wakefield import (process_objectives, run_wakefield_parallel,
                                          run_wakefield_s)
from cavsim2d.utils.printing import CavsimWarning, done, error, info, warning
import importlib
import itertools
import json
import numpy as np
import os
import shutil
import warnings
import pandas as pd
from cavsim2d.utils.config_validation import (canonicalise, require, validate_eigenmode_config,
                                              validate_optimisation_config,
                                              validate_tune_config, validate_wakefield_config)
from cavsim2d.solvers.ABCI.abci import resolve_mrot
from cavsim2d.solvers.NGSolve.eigen_ngsolve import parse_polarisations
from cavsim2d.solvers.eigenmode_result import pol_name
from cavsim2d.solvers.solver_objects import (OptimisationSolver, StudyEigenmode,
                                             StudyWakefield, merge_config, config_sample,
                                             DEFAULT_EIGENMODE_CONFIG, DEFAULT_TUNE_CONFIG,
                                             DEFAULT_WAKEFIELD_CONFIG)
from cavsim2d.utils.integrity import (current_tuned, guard_runs, notice_tuned_versions,
                                      project_contents_ok, stamp_matches, warn_if_stale)
from cavsim2d.data_module.operating_points import bunch_lengths, bunch_tag
from cavsim2d.study_plotting import StudyPlotting


def _sweep_point_name(keys, combo):
    """A filesystem-safe cavity name for one sweep point, e.g. ``Req103p35_Ri20``."""
    parts = []
    for k, v in zip(keys, combo):
        s = f'{v:g}' if isinstance(v, (int, float)) else str(v)
        parts.append(f'{k}{s}'.replace('.', 'p').replace('-', 'm').replace(' ', ''))
    return '_'.join(parts)


class Study(StudyPlotting):
    """A study: a manager over several RF-device simulations (cavities, guns, …).

    Groups devices under one project folder to run and compare analyses across
    them. A single device can also be analysed on its own without a ``Study`` —
    see :meth:`Cavity.set_workspace` / :meth:`Cavity.save`.

    The study keeps its cavities in a folder named after it inside *folder*:
    ``Study('proj')`` uses ``proj/cavities/``, ``Study('proj', name='run1')`` uses
    ``proj/run1/``, so several studies can share one project folder. A *folder*
    already named after the study is used as it is. ``study.projectDir`` is the
    folder in use.
    """

    def __init__(self, folder, name=None, cavities_list=None, names_list=None, overwrite=False,
                 _skip_project_init=False):
        """Constructs all the necessary attributes of the Cavity object

        Parameters
        ----------
        cavities_list: list, array like
            List containing Cavity objects.
        _skip_project_init: bool
            Internal flag. When True, skips project directory creation
            (used by spawn() for flat candidate folders).

        """

        self._optimisation_solver = None
        self._eigenmode_ns = None
        self._wakefield_ns = None

        # Legacy-API shim: historically the first positional arg was a list of
        # cavities. The refactor promoted ``folder`` to that slot, so older
        # notebooks calling ``Study([])`` or ``Study([cav, ...])`` would
        # feed a list into ``Path(...)`` and crash in ``_create_project``.
        # Route lists to ``cavities_list`` and defer folder creation until
        # ``save()`` is called explicitly.
        if isinstance(folder, (list, tuple)):
            if cavities_list is None:
                cavities_list = folder
            folder = None

        self.projectDir = folder
        self.cavities_list = []
        self.cavities_dict = {}

        self.name = 'cavities'
        if name:
            if not isinstance(name, str):
                raise ValueError('Study name must be a string.')
            self.name = name

        self.shape_space = {}
        self.shape_space_multicell = {}

        # The project folder first, then the cavities: added the other way round
        # they were created outside the study folder it then made.
        if not _skip_project_init and folder is not None:
            self.save(folder, overwrite)
        if cavities_list:
            self.add_cavity(cavities_list, names_list)

        self.uq_nodes = {}
        self.uq_fm_results_all_modes = {}
        self.rf_config = None
        self.power_qois_uq = {}
        self.eigenmode_qois = {}
        self.eigenmode_qois_all_modes = {}
        self.wakefield_qois = {}       # main run: per-cavity loss/kick factors
        self.wakefield_qois_op = {}    # per-cavity operating-point QOIs (P_HOM, ...)
        self.tune_results = {}

        self.uq_fm_results = {}
        self.uq_hom_results = {}

        self.power_qois = {}
        self.fm_results = None
        self.hom_results = None

        self.operating_points = None
        self.operating_points_threshold = {}

        self.returned_results = None
        self.ls = ['solid', 'dashed', 'dashdot', 'dotted',
                   'solid', 'dashed', 'dashdot', 'dotted',
                   'solid', 'dashed', 'dashdot', 'dotted']

        # Not pushed to the cavities here: set_cavities_field() calls a
        # cav.set_Eacc() no model has, so a Study built with cavities raised.
        self.E_acc = np.linspace(0.5, 30, 100) * 1e6  # V/m

    @property
    def tuned(self):
        """Return a Study view containing each cavity's tuned version.

        Each element is `cav.tuned` — a full cavity object with its own
        folder (geometry, eigenmode, etc.), so you can call `.plot`,
        `.eigenmode.run`, etc. on it. Cavities without a tuned version are
        omitted, and so are cavities tuned before their geometry changed: that
        tuned version belongs to the earlier geometry.
        """
        tuned_cavs = []
        tuned_names = []
        left_out = []
        for cav in self.cavities_list:
            t = current_tuned(cav)
            if t is not None:
                tuned_cavs.append(t)
                tuned_names.append(t.name or f'{cav.name}_tuned')
            elif cav.self_dir and warn_if_stale(cav, Path(cav.self_dir) / 'tuned' / 'tune_info',
                                                'The tuned cavity and its results'):
                left_out.append(cav.name)
        if left_out:
            warning(f"Left out of study.tuned, as they were tuned before their geometry "
                    f"changed: {', '.join(map(repr, left_out))}. Tune them again with "
                    f"run_tune(...).")
        if not tuned_cavs:
            warning('No cavities in this Study have a tuned version of their current '
                    'geometry. Run run_tune(...) first.')
            return None
        view = Study(self.projectDir, name=f'{self.name}_tuned',
                     _skip_project_init=True)
        # Not add_cavity: that re-homes a cavity to <projectDir>/<name>, which for
        # a tuned cavity is its parent's folder, so an eigenmode run on the view
        # wrote the tuned geometry's results over the parent's. Each tuned cavity
        # keeps the <parent>/tuned/ folder it was built in.
        for tc, nm in zip(tuned_cavs, tuned_names):
            view.cavities_list.append(tc)
            view.cavities_dict[nm] = tc
            view.shape_space[nm] = getattr(tc, 'shape', None)
            view.shape_space_multicell[nm] = getattr(tc, 'shape_multicell', None)
        return view

    def add_cavity(self, cavs, names=None, plot_labels=None):
        """
        Adds cavity to cavities
        Parameters
        ----------
        plot_labels: list, str
            Plot labels
        cavs: Cavity, list
            Cavity object or list of cavity objects
        names: list, str
            Cavity name or list of cavity names

        Returns
        -------

        """

        if isinstance(cavs, Cavity):
            cavs, names = [cavs], [names or f'cav_{len(self.cavities_list)}']
            plot_labels = [plot_labels[0] if isinstance(plot_labels, list) else plot_labels]
        else:
            if names is not None:
                require(len(cavs) == len(names), 'Number of cavities does not correspond to number of names.')
            else:
                start = len(self.cavities_list)
                names = [f'cav_{start + ii}' for ii in range(len(cavs))]

            if plot_labels is not None:
                require(len(cavs) == len(plot_labels), 'Number of cavities does not correspond to number of labels.')
            else:
                plot_labels = names

        # One name, one folder: a second cavity under a name already in the study
        # would share the first one's folder and mix its results in.
        taken = set(self.cavities_dict)
        for name in names:
            if name in taken:
                raise ValueError(
                    f"A cavity named {name!r} is already in this study, and the two "
                    f"would share the folder {os.path.join(str(self.projectDir), name)}. "
                    f"Give each cavity its own name.")
            taken.add(name)

        for cav, name, label in zip(cavs, names, plot_labels):
            cav._unpin_folder()              # adding a cavity moves it here
            cav.projectDir = self.projectDir
            cav.set_name(name)
            cav.set_plot_label(label)
            cav.create()
            cav._notice_stale_results()

            self.cavities_list.append(cav)
            self.cavities_dict[cav.name] = cav
            self.shape_space[cav.name] = cav.shape
            self.shape_space_multicell[cav.name] = cav.shape_multicell

    # -- Parameter sweep ----------------------------------------------------

    def sweep(self, template, parameters, mode='tensor', names=None, folder=None):
        """Build a Study of cavities spanning a grid of parameter values.

        A sweep is Study-initiated because every point is a distinct cavity with
        its own folder and simulations.

        Parameters
        ----------
        template : Cavity
            The base geometry; each point is ``template.rebuild(...)`` with the
            swept variables overridden. Works for any model (uses the model's own
            ``rebuild`` / ``expand_variable``), so a multicell ``'Req'`` correctly
            sets every cell.
        parameters : dict ``{var: [values]}``
            The variables to sweep and the values for each.
        mode : {'tensor', 'hadamard'}
            ``'tensor'`` sweeps **every combination** (cartesian product);
            ``'hadamard'`` sweeps them **element-wise** (zipped) and requires every
            value list to be the same length (raises ``ValueError`` otherwise).
        names : list of str, optional
            Explicit per-point cavity names (default: derived from the values).

        Returns
        -------
        Study
            A study of the swept cavities. Run any analysis on it
            (``sw.run_eigenmode(...)``) and compare via ``sw.eigenmode.qois_df``,
            ``sw.eigenmode.plot_compare()`` or :meth:`results` (which joins the
            swept values with the QOIs). The swept values are on ``sw.sweep_table``.
        """
        if not isinstance(template, Cavity):
            raise TypeError('sweep(template, ...): template must be a Cavity.')
        if mode not in ('tensor', 'hadamard'):
            raise ValueError(f"sweep mode must be 'tensor' or 'hadamard', got {mode!r}.")
        if not parameters:
            raise ValueError('sweep: `parameters` must be a non-empty {var: [values]} dict.')

        keys = list(parameters)
        vals = [list(parameters[k]) for k in keys]
        for k in keys:                       # reject unknown vars up front, with the model's own names
            template.expand_variable(k)
        if mode == 'hadamard':
            lengths = {k: len(v) for k, v in zip(keys, vals)}
            if len(set(lengths.values())) != 1:
                raise ValueError(
                    'hadamard sweep requires every parameter value list to be the '
                    f'same length; got {lengths}. Use mode="tensor" for a full grid.')
            combos = list(zip(*vals))
        else:
            combos = list(itertools.product(*vals))

        rows, table_rows = {}, {}
        for i, combo in enumerate(combos):
            nm = names[i] if names else _sweep_point_name(keys, combo)
            slot_row = {}
            for k, v in zip(keys, combo):
                for slot in template.expand_variable(k):
                    slot_row[slot] = v
            rows[nm] = slot_row
            table_rows[nm] = dict(zip(keys, combo))

        diff = pd.DataFrame.from_dict(rows, orient='index')
        if folder is None:
            # A unique subfolder per sweep call, so successive sweeps on the same
            # study don't accumulate into one directory (which would then reload
            # mixed together via Study.load).
            n = getattr(self, '_sweep_counter', 0)
            self._sweep_counter = n + 1
            base = self.projectDir or os.getcwd()
            folder = os.path.join(str(base), 'sweep' if n == 0 else f'sweep_{n}')
        sw = template.spawn(diff, folder)
        sw.name = 'sweep'
        sw.sweep_parameters = dict(parameters)
        sw.sweep_mode = mode
        sw.sweep_table = pd.DataFrame.from_dict(table_rows, orient='index')
        return sw

    def results(self, analysis='eigenmode'):
        """One table joining the swept parameter values with an analysis' QOIs,
        one row per cavity (index = cavity name).

        Reads each cavity's primary result dict (``cav.eigenmode.qois`` /
        ``cav.wakefield.qois``) so the table has exactly one row per point; only
        scalar QOIs are kept. Requires the analysis to have been run on this
        (sweep) study; without a ``sweep_table`` it returns the QOIs alone."""
        rows = {}
        for cav in self.cavities_list:
            ns = getattr(cav, analysis, None)
            q = getattr(ns, 'qois', None)
            if isinstance(q, dict) and q:
                rows[cav.name] = {k: v for k, v in q.items()
                                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        qdf = pd.DataFrame.from_dict(rows, orient='index')
        table = getattr(self, 'sweep_table', None)
        if table is not None:
            return table.join(qdf, how='left')
        return qdf

    # -- Persistence --------------------------------------------------------

    @classmethod
    def load(cls, project_dir, models=None, name='cavities'):
        """Reconstruct a Study from a saved project directory.

        Scans ``project_dir`` for cavity folders (each carries a
        ``geometry/model.json`` snapshot written on creation) and rebuilds every
        cavity with its ``self_dir`` pointing at the existing folder, so all its
        cached results — ``cav.eigenmode.qois``, ``cav.wakefield.qois``,
        ``cav.tuned`` — read straight from disk with **no re-simulation**. The
        returned Study behaves like any other, so two loaded studies can be
        compared directly (``study.eigenmode.plot_compare()``, ``qois_df``).

        ``models`` optionally maps a model's class name to the class, for custom
        (in-notebook) geometries whose module is not importable by name.
        """
        project_dir = str(project_dir)
        if not os.path.isdir(project_dir):
            raise FileNotFoundError(f'No such project directory: {project_dir}')

        # Study(project_dir) keeps its cavities in <project_dir>/<name>/ (name
        # defaults to 'cavities'). Given that same project folder, load from there,
        # as the constructor would; it used to find nothing and return empty.
        def holds_cavities(folder):
            return any(os.path.isfile(os.path.join(folder, e, 'geometry', 'model.json'))
                       for e in os.listdir(folder))
        nested = os.path.join(project_dir, name)
        if (os.path.basename(os.path.normpath(project_dir)) != name
                and os.path.isdir(nested) and not holds_cavities(project_dir)):
            project_dir = nested

        study = cls(project_dir, _skip_project_init=True, name=name)
        models = models or {}
        loaded = 0
        for entry in sorted(os.listdir(project_dir)):
            sub = os.path.join(project_dir, entry)
            model_json = os.path.join(sub, 'geometry', 'model.json')
            if not os.path.isfile(model_json):
                continue
            try:
                with open(model_json) as f:
                    state = json.load(f)
            except Exception as e:
                warning(f'load: could not read {model_json}: {e!r}')
                continue
            klass = models.get(state.get('model'))
            if klass is None:
                try:
                    klass = getattr(importlib.import_module(state.get('module', '')),
                                    state['model'])
                except Exception:
                    warning(
                        f"load: cannot import model {state.get('model')!r} from "
                        f"{state.get('module')!r} for '{entry}'. Pass "
                        f"models={{'{state.get('model')}': <class>}} to load a custom "
                        f'geometry. Skipping.')
                    continue
            try:
                cav = klass._reconstruct_from_state(state)
            except Exception as e:
                warning(f"load: could not reconstruct '{entry}': {e!r}")
                continue
            cav.name = entry
            cav.projectDir = project_dir
            cav.self_dir = sub
            cav.uq_dir = os.path.join(sub, 'uq')
            geo = os.path.join(sub, 'geometry', 'geodata.geo')
            cav.geo_filepath = geo if os.path.isfile(geo) else None
            study.cavities_list.append(cav)
            study.cavities_dict[cav.name] = cav
            study.shape_space[cav.name] = getattr(cav, 'shape', None)
            study.shape_space_multicell[cav.name] = getattr(cav, 'shape_multicell', None)
            loaded += 1

        if loaded == 0:
            warnings.warn(f'Study.load: no reconstructable cavities found under '
                          f'{project_dir}. (A cavity folder needs a geometry/model.json, '
                          f'written when the cavity is created.)', CavsimWarning, stacklevel=2)
        return study

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

    def save(self, project_folder, overwrite=False):
        """
        Set folder to save cavity analysis results

        Parameters
        ----------
        files_path: str
            Save project directory

        Returns
        -------

        """

        if project_folder is None:
            raise ValueError('Give a folder to write the simulation results to.')
        # No catch-all: it printed the error and carried on, so a study whose
        # folder could not be created looked fine until its first run failed.
        self.projectDir = project_folder
        if self._create_project(overwrite):
            done(f"Project {self.projectDir} created.")
        else:
            done(f"Project {self.projectDir} already exists; its results are kept. "
                 f"Pass overwrite=True to start afresh.")

    def _create_project(self, overwrite):
        """Make the study folder, ``<projectDir>/<name>`` (or *projectDir* itself
        when it already carries the name); True if it was created, False if it
        already existed. Raises when it cannot be made."""
        if not self.name:
            raise ValueError('A study needs a non-empty name.')
        proposed_path = Path(self.projectDir)
        if proposed_path.name != self.name:
            proposed_path = proposed_path / self.name
        self.projectDir = str(proposed_path)

        # Check (and, with overwrite, clear) the folder actually used. This
        # used to look at <project_dir>/<name> even when the folder already
        # carried the name, so it checked one folder and used another.
        if self._check_if_path_exists(proposed_path, overwrite):
            return False
        os.makedirs(proposed_path, exist_ok=True)
        return True

    @staticmethod
    def _check_if_path_exists(path, overwrite):
        """True if *path* exists (and is kept). With *overwrite*, a folder holding
        nothing but cavsim2d's own output is deleted (False is returned); anything
        else raises rather than being deleted.

        The old test deleted any folder with fewer than 20 entries."""
        if not os.path.exists(path):
            return False
        if not overwrite:
            return True
        ok, unexpected = project_contents_ok(path)
        if not ok:
            raise ValueError(
                f"{path} is not a cavsim2d project folder (it holds "
                f"{', '.join(sorted(unexpected)[:5])}), so overwrite=True will not "
                f"delete it. Choose another folder or study name.")
        shutil.rmtree(path)
        return False

    def run_tune(self, tune_config=None, **kwargs):
        """

        Parameters
        ----------
        tune_config: dict

            .. code-block:: python

                tune_config = {
                            'freqs': 801.58,
                            'cell_type': {
                                'mid-cell': 'Req',
                                'end-cell': 'Req',
                            },
                            'processes': 1,
                            'rerun': True
                        }

            Each key of ``cell_type`` is a cell type (``'mid-cell'``,
            ``'end-cell'``, ``'single-cell'``) and each value is the tune
            variable(s) for that cell type — either a single string
            (``'Req'``) or a list (``['Req', 'L']``). Multiple cell types
            are tuned consecutively in the order they appear; each stage's
            tuned dimensions feed the next stage.

            The older ``'parameters'``/``'cell_types'`` pair is still accepted
            and converted to ``cell_type``.

        Recognised keys
        ---------------
        freqs (alias ``freq``) : float | list
            Target frequency in MHz (a scalar for all cavities, or one per cavity).
            Unset, each elliptical cavity is tuned to the pi-mode estimate c/(4L)
            of its half-cell length, with a warning.
        cell_type : dict
            ``{cell type: tune variable(s)}`` as above. A tune variable is any
            name the model exposes via ``cav.tune_variables()`` (e.g. ``'Req'``,
            ``'L'`` for elliptical, ``'R6'`` for the gun, ``'p3_r'`` for a spline).
            Unset: the mid-cell ``Req``, plus the end-cell ``L`` on a multi-cell
            elliptical cavity.
        processes : int
            Number of parallel worker processes (default 1).
        rerun : bool
            Recompute even if a tuned result exists (default True). ``False``
            keeps the finished tunes of unchanged cavities and tunes the rest.
        force (alias ``overwrite``) : bool
            Answer yes to the question asked before existing results are replaced
            or results computed for another geometry are deleted.
        tol (alias ``tolerance``) : float
            Frequency tolerance [MHz]: the secant stops once the frequency is this
            close to the target (default 1e-4).
        maxiter : int
            Maximum secant iterations per stage.
        eigenmode_config : dict
            Eigenmode sub-config used for each tuning solve (see
            :meth:`run_eigenmode` keys).
        solve_tuned : bool
            After tuning, run the eigenmode analysis on every tuned cavity with
            ``eigenmode_config`` (default False).
        uq_config : dict
            Optional UQ-averaged tuning (see :meth:`run_uq` keys).

        Returns
        -------

        """

        # Config keys may also be passed as kwargs; kwargs override the dict.
        # Aliases ('freq', 'tolerance', the parameters + cell_types pair) become the
        # keys the tuner reads before the defaults go in, or a default would win.
        tune_config = {**canonicalise(tune_config, 'tune'), **canonicalise(kwargs, 'tune')}
        validate_tune_config(tune_config)
        # The same complete defaults as cav.tune.run; freqs and cell_type left
        # unset are resolved per cavity (c/(4L), and Req, plus L at the end cells
        # of a multicell cavity).
        tune_config = merge_config(DEFAULT_TUNE_CONFIG, tune_config)
        resolve_tune_defaults(tune_config, self.cavities_list)

        # Validate by normalising (raises if the dict shape is wrong).
        normalize_cell_type_config(tune_config)

        # Truthy checks (not just presence): complete saved configs carry the
        # keys with explicit None when disabled.
        if tune_config.get('uq_config'):
            uq_config = tune_config['uq_config']

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

        if tune_config.get('eigenmode_config'):
            eigenmode_config = tune_config['eigenmode_config']

            if eigenmode_config.get('uq_config'):
                uq_config_eig = eigenmode_config['uq_config']

                if 'epsilon' in uq_config_eig.keys() and 'uq_config' in uq_config_eig.keys():
                    info('epsilon and delta are both entered. Epsilon is preferred.')

        if not isinstance(tune_config['rerun'], bool):
            raise ValueError("tune_config['rerun'] must be a boolean.")

        if not guard_runs(self.cavities_list, 'tune', tune_config, title='Tune'):
            return self
        # With rerun=False the tuner keeps every finished tune of an unchanged
        # cavity (and its config) and tunes only the rest.
        reused = set()
        if not tune_config['rerun']:
            for cav in self.cavities_list:
                info_dir = Path(cav.self_dir) / 'tuned' / 'tune_info'
                if (info_dir / 'tune_res.json').exists() and stamp_matches(info_dir, cav) is not False:
                    reused.add(cav.name)
        run_tune_parallel(self.cavities_dict, tune_config, solver='NGSolveMEVP', resume=False)
        # Saved after the run, which starts by clearing each tuned/ folder.
        for cav in self.cavities_list:
            info_dir = Path(cav.self_dir) / 'tuned' / 'tune_info'
            if info_dir.is_dir() and cav.name not in reused:
                with open(info_dir / 'config.json', 'w') as f:
                    json.dump(tune_config, f, indent=4, default=str)
            cav.tune._config = None

        self.tune_config = tune_config
        # get tune results (also attaches cav.tuned lazily from disk)
        self.get_tune_res()

        # Solve the tuned cavities too, when asked (solve_tuned). eigenmode_config
        # alone only sets the mesh of the tuning solves: it used to trigger this as
        # well, here but not in cav.tune.run.
        if tune_config.get('solve_tuned'):
            tuned_view = self.tuned
            if tuned_view:
                info(f"Running the eigenmode analysis on the tuned cavities...")
                ec = dict(tune_config.get('eigenmode_config') or {})

                # Adjust n_cells if specified in config
                if 'n_cells' in ec:
                    for cav in tuned_view.cavities_list:
                        cav.set_n_cells(ec['n_cells'])

                tuned_view.run_eigenmode(ec)
        return self

    def get_tune_res(self):
        """Reload tune results from disk and attach tuned cavities.

        After ``run_tune`` the subprocesses have written
        ``<cav>/tuned/tune_info/tune_res.json`` and the tuned geometry.
        The parent‐process cavity objects still have no ``_tuned_cavity``,
        so we force the ``cav.tuned`` lazy loader to rebuild them from disk.
        """
        for cav in self.cavities_dict.values():
            cav._tuned_cavity = None  # force reload via property
            cav.tune._invalidate()
            if cav.tuned is not None:
                # The tune record is the parent's cav.tune.qois. This used to copy
                # the reloaded tuned cavity's own tune_results and freq, which are
                # always empty and 0, over the parent's.
                last = last_stage_result(cav.tune_results)
                if last and last.get('FREQ') is not None:
                    cav.freq = last['FREQ']
            else:
                # Fallback: legacy path or failed tune
                cav.get_ngsolve_tune_res()
            self.tune_results[cav.name] = cav.tune_results

    def run_eigenmode(self, eigenmode_config=None, **kwargs):
        """
        Runs the eigenmode analysis with the given configuration.

        Parameters
        ----------
        eigenmode_config : dict
            Configuration for running the eigenmode analysis. Example structure:

            .. code-block:: python

                eigenmode_config = {
                    'processes': 3,
                    'rerun': True,
                    'boundary_conditions': 'mm',
                    'uq_config': {
                        'variables': ['A', 'B', 'a', 'b'],
                        # 'objectives': ["freq [MHz]", "R/Q [Ohm]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "G [Ohm]", "kcc [%]", "ff [%]"],
                        'objectives': ["Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]"],
                        # 'objectives': ["ZL"],
                        'delta': [0.05, 0.05, 0.05, 0.05],
                        'processes': 4,
                        'distribution': 'gaussian',
                        # 'method': ['QMC', 'LHS', 1000],
                        # 'method': ['QMC', 'Sobol', 1000],
                        # 'method': ['Qudrature', 'Gaussian', 1000],
                        'method': ['Quadrature', 'Stroud3'],
                        # 'method': ['Quadrature', 'Stroud5'],
                        # 'gaussian': ['Quadrature', 'Gaussian'],
                        # 'from file': ['<file path>', columns],
                        'cell_type': 'mid-cell',
                        'cell_complexity': 'simplecell'
                    }
                }

        Recognised keys
        ---------------
        processes : int
            Number of parallel worker processes (default 1).
        rerun : bool
            Recompute even if results already exist (default True). ``False``
            keeps the results of unchanged cavities and computes the rest.
        force (alias ``overwrite``) : bool
            Answer yes to the question asked before existing results are replaced
            or results computed for another geometry are deleted.
        boundary_conditions : str
            Two-character wall condition, e.g. ``'mm'`` (magnetic-magnetic),
            ``'ee'``, ``'me'``.
        polarisation : str | int | list
            Azimuthal mode(s) to solve: ``'monopole'``/0, ``'dipole'``/1,
            ``'quadrupole'``/2, … or a list of them.
        n_modes (alias ``nmodes``) : int
            Number of eigenmodes to compute.
        mode_of_interest : int | list | dict
            1-based index (or list, or per-polarisation dict) of the mode(s) the
            reported QOIs refer to. Defaults to the accelerating pi-mode.
        f_shift : float
            Frequency shift for the shift-invert eigensolver.
        direct_solver : str
            Linear solver: ``'pardiso'``/``'umfpack'``/``'sparsecholesky'``
            (defaults per platform).
        mesh_config : dict
            Mesh controls, e.g. ``{'h': <element size mm>}``.
        conductivity, surface_resistance, normalization_length : float
            Material/normalisation parameters for the loss-derived QOIs.
        uq_config : dict
            Uncertainty-quantification sub-config (see :meth:`run_uq` keys).

        Returns
        -------
        None

        """

        # Config keys may also be passed as kwargs; kwargs override the dict.
        # Aliases (nmodes, overwrite) become the keys the solver reads first.
        eigenmode_config = {**canonicalise(eigenmode_config, 'eigenmode'),
                            **canonicalise(kwargs, 'eigenmode')}

        validate_eigenmode_config(eigenmode_config)

        # Merge over the complete defaults so the saved config records every
        # setting the run used (same pattern as EigenmodeSolver.run).
        eigenmode_config = merge_config(DEFAULT_EIGENMODE_CONFIG, eigenmode_config)

        # One question for every cavity before anything is replaced.
        pols = {pol_name(m) for m in parse_polarisations(eigenmode_config.get('polarisation', 0))}
        notice_tuned_versions(self.cavities_list, 'eigenmode', study=True)
        if not guard_runs(self.cavities_list, 'eigenmode', eigenmode_config,
                          pol_labels=pols, title='Eigenmode'):
            return self.eigenmode

        eigenmode_config['target'] = run_eigenmode_s

        rerun = True
        if 'rerun' in eigenmode_config.keys():
            if isinstance(eigenmode_config['rerun'], bool):
                rerun = eigenmode_config['rerun']

        uq_config = {}
        if 'uq_config' in eigenmode_config.keys():
            uq_config = eigenmode_config['uq_config']
            if uq_config:


                if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                    info('Epsilon and delta are both entered. Epsilon is preferred.')

        # add save directory field to eigenmode_config
        eigenmode_config['solver_save_directory'] = 'NGSolveMEVP'
        eigenmode_config['opt'] = False

        # run_eigenmode_s honours 'rerun' per polarisation: rerun=True clears
        # and re-solves only the requested polarisations; rerun=False solves
        # only the ones whose results are missing.
        run_eigenmode_parallel(self.cavities_dict, eigenmode_config,
                               self.projectDir)

        self.eigenmode_config = eigenmode_config
        self.get_eigenmode_qois(uq_config)

        # Save the eigenmode config for traceability, under the name every reader
        # expects: 'config.json' is what EigenmodeSolver.config (and the wakefield,
        # tune and multipacting namespaces) look for. This used to be written as
        # 'eigenmode_config.json', which nothing read -- so a study-driven run left
        # cav.eigenmode.config empty and anything keying off it (e.g. the open-boundary
        # check behind the impedance reconstruction) saw a run with no settings at all.
        saved_cfg = {k: v for k, v in eigenmode_config.items() if k != 'target'}
        for cav in self.cavities_list:
            ec_path = Path(cav.self_dir) / 'eigenmode' / 'config.json'
            # Ensure folder exists (might be empty if analysis failed but we still want the log)
            os.makedirs(ec_path.parent, exist_ok=True)
            with open(ec_path, 'w') as f:
                json.dump(saved_cfg, f, indent=4, default=str)
        return self.eigenmode

    def get_eigenmode_qois(self, uq_config):
        # get results
        for key, cav in self.cavities_dict.items():
            # try:
            cav.get_eigenmode_qois()
            self.eigenmode_qois[cav.name] = cav.eigenmode_qois
            self.eigenmode_qois_all_modes[cav.name] = cav.eigenmode_qois_all_modes
            if uq_config:
                cav.get_uq_fm_results(cav.uq_dir)
                self.uq_fm_results[cav.name] = cav.uq_fm_results
                self.uq_nodes[cav.name] = cav.uq_nodes
                self.uq_fm_results_all_modes[cav.name] = cav.uq_fm_results_all_modes

    def run_wakefield(self, wakefield_config=None, **kwargs):
        """

        Parameters
        ----------
        wakefield_config:

            .. code-block:: python

                op_points = {
                            "Z": {
                                "freq [MHz]": 400.79,  # Operating frequency
                                "E [GeV]": 45.6,  # <- Beam energy
                                "I0 [mA]": 1280,  # <- Beam current
                                "V [GV]": 0.12,  # <- Total voltage
                                "Eacc [MV/m]": 5.72,  # <- Accelerating field
                                "nu_s []": 0.0370,  # <- Synchrotron oscillation tune
                                "alpha_p [1e-5]": 2.85,  # <- Momentum compaction factor
                                "tau_z [ms]": 354.91,  # <- Longitudinal damping time
                                "tau_xy [ms]": 709.82,  # <- Transverse damping time
                                "f_rev [kHz]": 3.07,  # <- Revolution frequency
                                "beta_xy [m]": 56,  # <- Beta function
                                "N_c []": 56,  # <- Number of cavities
                                "T [K]": 4.5,  # <- Operating tempereature
                                # RMS bunch length: one value, or several under
                                # labels of your choice (here with synchrotron
                                # radiation only, and with beamstrahlung too).
                                # The first is the primary one the plots show.
                                "sigma [mm]": {"SR": 4.32, "BS": 15.2},
                                "Nb [1e11]": 2.76  # <- Bunch population
                            }
                }
                wakefield_config = {
                    'bunch_length': 25,
                    'wakelength': 50,
                    'processes': 2,
                    'rerun': True,
                    'operating_points': op_points,
                }

        Recognised keys
        ---------------
        MROT : {0, 1, 2} or str
            Beam mode: 0/``'monopole'``/``'longitudinal'``,
            1/``'dipole'``/``'transverse'``, 2/``'both'``. (``'polarisation'`` is
            a deprecated alias.)
        wakelength : float
            Wake length to compute, in metres.
        bunch_length : float
            RMS bunch length, in mm.
        processes : int
            Number of parallel worker processes (default 1).
        rerun : bool
            Recompute even if results exist (default True). ``False`` keeps the
            results of unchanged cavities and computes the rest.
        force (alias ``overwrite``) : bool
            Answer yes to the question asked before existing results are replaced
            or results computed for another geometry are deleted.
        MT : int
            ABCI time steps per mesh cell (clamped to 19, ABCI's limit).
        DDR_SIG, DDZ_SIG : float
            Radial / axial mesh step as a fraction of the bunch length (default
            0.1), capped at 1.25 mm. ``mesh_config['DDR'/'DDZ']`` overrides them.
        mesh_config : dict
            Mesh controls, e.g. ``{'DDR': ..., 'DDZ': ...}`` (metres).
        contour_ds : float
            Wall sampling spacing for the geometry deck (metres).
        beampipe_length : float
            Override the auto beam-pipe length (metres). ABCI requires a pipe at
            each end; one is added (3x the device length) where missing.
        operating_points : dict
            Machine/beam parameters for the derived power/HOM quantities
            (see the ``op_points`` example above).
        uq_config : dict
            Uncertainty-quantification sub-config (see :meth:`run_uq` keys).

        Returns
        -------

        """

        # Config keys may also be passed as kwargs; kwargs override the dict.
        wakefield_config = {**canonicalise(wakefield_config, 'wakefield'),
                            **canonicalise(kwargs, 'wakefield')}

        validate_wakefield_config(wakefield_config)

        # Merge over the complete defaults so the saved config records every
        # setting the run used (same pattern as WakefieldSolver.run).
        wakefield_config = merge_config(DEFAULT_WAKEFIELD_CONFIG, wakefield_config)

        # One question for every cavity before anything is replaced.
        notice_tuned_versions(self.cavities_list, 'wakefield', study=True)
        if not guard_runs(self.cavities_list, 'wakefield', wakefield_config,
                          title='Wakefield'):
            return self.wakefield

        wakefield_config['target'] = run_wakefield_s

        wakefield_config_keys = wakefield_config.keys()

        if 'operating_points' in wakefield_config_keys:
            self.operating_points = wakefield_config['operating_points']
            for cav in self.cavities_list:
                cav.operating_points = self.operating_points

        uq_config = {}
        # Truthy check (not just presence): the complete merged config carries
        # 'uq_config' with explicit None when UQ is disabled.
        if wakefield_config.get('uq_config'):
            # A copy: the objectives are rewritten below, and the caller's dict
            # must come back as it went in.
            wakefield_config['uq_config'] = dict(wakefield_config['uq_config'])
            uq_config = wakefield_config['uq_config']

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

            require('objectives' in uq_config.keys(), 'Please enter objectives in uq_config.')

            # Keep the FULL objective form (goal, name, [args]). process_objectives
            # and the UQ wakefield reader both key on obj[1]=name and obj[2]=args
            # (e.g. a ZL/ZT objective's frequency windows), exactly as the
            # optimiser's objectives_unprocessed does. The old code truncated to
            # ['', goal, name] — dropping the frequency range and shifting the QOI
            # name into the goal slot, so every ZL/ZT objective came back empty.
            objectives_unprocessed = list(uq_config['objectives'])
            objectives, weights = process_objectives(objectives_unprocessed)
            uq_config['objectives_unprocessed'] = objectives_unprocessed
            uq_config['objectives'] = objectives
        else:
            wakefield_config['uq_config'] = uq_config

        # MROT is the canonical key (0 longitudinal, 1 transverse, 2 both);
        # names and the deprecated 'polarisation' alias are also accepted.
        # Note: for wakefield this is the ABCI beam mode, not the eigenmode
        # azimuthal order that 'polarisation' means in eigenmode_config.
        if 'polarisation' in wakefield_config_keys and 'MROT' not in wakefield_config_keys:
            warning("wakefield_config['polarisation'] is deprecated; use 'MROT' "
                 "(0=longitudinal, 1=transverse, 2=both).")
        resolved_mrot = resolve_mrot(wakefield_config, default=2)
        wakefield_config['MROT'] = resolved_mrot
        wakefield_config['polarisation'] = resolved_mrot  # back-compat for downstream reads

        # The bunch length, wake length and mesh ratios are read from the top
        # level, where the defaults and the caller put them. This used to nest
        # hard-coded copies under beam_config / wake_config / mesh_config, which
        # the deck writer reads first, so every study-level run used a 25 mm bunch
        # and a 50 m wake whatever it was given.
        #
        # With rerun=False the solver keeps the finished results of unchanged
        # cavities and runs the rest; it used to skip the whole study.
        run_wakefield_parallel(self.cavities_dict, wakefield_config)

        self.wakefield_config = wakefield_config
        self.get_wakefield_qois(uq_config)
        return self.wakefield

    def get_wakefield_qois(self, uq_config):
        for key, cav in self.cavities_dict.items():
            # try:
            # main run loss/kick factors — always reported now
            self.wakefield_qois[cav.name] = cav.wakefield_qois
            # operating-point QOIs (P_HOM, ...) — only when op-points were given
            cav.get_wakefield_qois(self.wakefield_config)
            self.wakefield_qois_op[cav.name] = cav.wakefield_qois_op
            if uq_config:
                # Wakefield UQ lives in <cav>/wakefield/uq/, apart from the
                # eigenmode UQ in <cav>/uq/.
                cav.get_uq_hom_results(os.path.join(cav.self_dir, 'wakefield', 'uq', 'uq.json'))
                cav_uq_hom_results = cav.uq_hom_results
                if 'operating_points' in uq_config:
                    # separate into opearating points
                    op_points = uq_config['operating_points']
                    uq_hom_results_op = {}
                    for op, val in op_points.items():
                        uq_hom_results_op[op] = {}
                        for sig_id, v in bunch_lengths(val).items():
                            ident = f'_{bunch_tag(op, sig_id, v)}'
                            uq_hom_results_op[op][sig_id] = {kk.replace(ident, ''): vv for (kk, vv) in
                                                             cav_uq_hom_results.items() if ident in kk}
                else:
                    uq_hom_results_op = cav.uq_hom_results

                self.uq_hom_results[cav.name] = uq_hom_results_op
            # except FileNotFoundError:
            #     error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    @property
    def eigenmode(self):
        """Eigenmode results across the whole study — the same namespace a single
        cavity has, so ``study.eigenmode.plot_impedance()`` mirrors
        ``cav.eigenmode.plot_impedance()`` and overlays every cavity."""
        if self._eigenmode_ns is None:
            self._eigenmode_ns = StudyEigenmode(self)
        return self._eigenmode_ns

    @property
    def wakefield(self):
        """Wakefield results across the whole study — mirrors
        ``cav.wakefield.*`` (``plot_impedance``, ``plot_wake``, ``plot_k_loss``,
        ``plot_k_kick``) and overlays every cavity."""
        if self._wakefield_ns is None:
            self._wakefield_ns = StudyWakefield(self)
        return self._wakefield_ns

    @property
    def optimisation(self):
        """OptimisationSolver object for this cavity collection."""
        if self._optimisation_solver is None:
            self._optimisation_solver = OptimisationSolver(self)
        return self._optimisation_solver

    def run_optimisation(self, optimisation_config, resume=False):
        """Run a multi-objective shape optimisation.

        Pass ``resume=True`` to continue an interrupted run: the solver
        picks up from the last completed generation table under
        ``<project>/optimisation/generations/`` and reuses any candidate
        simulation results already written to disk.

        Recognised keys of ``optimisation_config``
        ------------------------------------------
        bounds : dict
            ``{variable: [min, max]}`` search box for each free variable (mm).
        objectives : list
            Each objective is ``['min'|'max', name]`` or ``['equal', name,
            target]``. Names are polarisation-qualified, e.g.
            ``'monopole:R/Q [Ohm]'``, or wakefield peaks ``'ZL'``/``'ZT'`` with a
            ``[low, ..., high]`` frequency window as a third element.
        initial_points : int
            Size of the initial population.
        no_of_generation (alias ``no_of_generations``) : int
            Number of generations to evolve.
        method : str
            Optimiser, e.g. ``'DE'`` (differential evolution), ``'NSGA2'``.
        mutation_factor, crossover_factor, elites_for_crossover, chaos_factor,
        mutation_sigma, eta_sbx : float
            Evolutionary-operator controls.
        weights : list
            Per-objective weights.
        seed : int
            RNG seed for reproducibility.
        tune_config : dict
            Optional per-candidate tuning (see :meth:`run_tune` keys).
        resume : bool
            Same as the ``resume`` argument.
        """
        validate_optimisation_config(optimisation_config)
        return self.optimisation.run(optimisation_config, resume=resume)

    def config_sample(self, kind, show=True):
        """Print and return the default config for *kind* (``'eigenmode'``,
        ``'wakefield'``, ``'tune'``, ``'multipacting'``, ``'uq'`` or
        ``'optimisation'``); see :meth:`Cavity.config_sample`. Needs no cavity in
        the study. ``cfg_sample``, ``sample_config`` and ``sample_cfg`` are other
        names for it."""
        return config_sample(kind, show=show)

    cfg_sample = sample_config = sample_cfg = config_sample

    def qois_fm(self):
        """
        Retrieves the fundamental mode quantities of interest

        Returns
        -------
        Dictionary containing fundamental mode quantities of interest (normed optional).
        """
        # Read from the results themselves (cav.eigenmode_qois): the summary
        # attributes were never filled on a loaded study, whose summary was all
        # zeros. They remain the fallback for objects that carry only those.
        attrs = {'freq [MHz]': 'freq', 'Epk/Eacc []': 'e', 'Bpk/Eacc [mT/MV/m]': 'b',
                 'kcc [%]': 'k_cc', 'R/Q [Ohm]': 'R_Q', 'G [Ohm]': 'G',
                 'GR/Q [Ohm^2]': 'GR_Q'}
        results = []
        for cav in self.cavities_list:
            qois = getattr(cav, 'eigenmode_qois', None)
            if qois is None:
                results.append({k: getattr(cav, a, None) for k, a in attrs.items()})
            else:
                # Never the attribute: an elliptical cavity's b is its semi-axis.
                results.append({k: qois.get(k) for k in attrs})

        results_norm_units = []
        for row in results:
            results_norm_units.append({
                r"$e_\mathrm{pk}$": row['Epk/Eacc []'],
                r"$b_\mathrm{pk}$": row['Bpk/Eacc [mT/MV/m]'],
                r"$k_\mathrm{cc}$": row['kcc [%]'],
                r"$r/q$": row['R/Q [Ohm]'],
                r"$g$": row['G [Ohm]'],
                r"$g\cdot r/q $": row['GR/Q [Ohm^2]']
            })

        return results

    def summary(self):
        """Fundamental-mode QOIs for all cavities as a DataFrame, indexed by
        cavity name. In Jupyter it renders as an HTML table; elsewhere use
        ``.to_string()`` / ``.to_markdown()``. Requires eigenmode results
        (run ``run_eigenmode`` first)."""
        names = [cav.name for cav in self.cavities_list]
        return pd.DataFrame(self.qois_fm(), index=names)

    def define_operating_points(self, ops):
        self.operating_points = ops

    def remove_cavity(self, cav):
        """
        Removes cavity from cavity list
        Parameters
        ----------
        cav: object
            Cavity object

        Returns
        -------

        """
        self.cavities_list.remove(cav)

    def _primary_tag(self, opt):
        """Sub-run id of operating point *opt*'s primary (first) bunch length, or
        ``None`` if the study does not know that operating point."""
        ops = (self.operating_points
               or (getattr(self, 'wakefield_config', None) or {}).get('operating_points')
               or {})
        if opt not in ops:
            return None
        label, sigma = next(iter(bunch_lengths(ops[opt]).items()))
        return bunch_tag(opt, label, sigma)

    def __repr__(self):
        names = list(self.cavities_dict.keys())
        shown = names if len(names) <= 6 else names[:6] + ['...']
        return (f"Study(name={getattr(self, 'name', None)!r}, "
                f"folder={str(self.projectDir) if self.projectDir else None!r}, "
                f"{len(names)} cavities: {shown})")

    def __str__(self):
        p = dict()
        for name, cav in self.cavities_dict.items():
            p[name] = {
                'tune': cav.tune_results,
                'fm': cav.eigenmode_qois,
                'hom': cav.wakefield_qois,
                'uq': {
                    'tune': 0,
                    'fm': cav.uq_fm_results,
                    'hom': cav.uq_hom_results
                }
            }
        return fr"{json.dumps(p, indent=4)}"

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.cavities_list[key]
        elif isinstance(key, str):
            if key in self.cavities_dict.keys():
                return self.cavities_dict[key]
            else:
                error('Invalid key. Study does not contain a Cavity named {key}.')
        else:
            raise TypeError("Invalid argument type. Must be int or str.")


