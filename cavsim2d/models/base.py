import re
from IPython.core.display import HTML, display_html, Math
from IPython.core.display_functions import display
from abc import ABC, abstractmethod
from pathlib import Path
from cavsim2d.constants import *
from cavsim2d.processes import *
from cavsim2d.utils.shared_functions import *
from distutils import dir_util
from ipywidgets import HBox, VBox, Label
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
import ast
import copy
import ipywidgets as widgets
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator as op
import os
import pandas as pd
import time
from cavsim2d.processes.tune import last_stage_result
from cavsim2d.geometry.beampipes import abci_shape
from cavsim2d.solvers.eigenmode_result import pol_name, pol_number, monopole_dir
from cavsim2d.solvers.ABCI.abci import resolve_mrot
from cavsim2d.solvers.solver_objects import TuneSolver, EigenmodeSolver, WakefieldSolver
from cavsim2d.constants import SOFTWARE_DIRECTORY
from cavsim2d.utils.style import house_style, polarisation_color, shades, WARM
from fractions import Fraction

# Safe arithmetic evaluator for simple expressions
_ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

class Cavity(ABC):
    """
    Command Line Interface module for running analysis.

    .. note::

       Still under development so some functions might not work properly
    """

    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k',
                 plot_label=None, geo_filepath=None):
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
        """

        # ───────────────────────────────────────────────────────────────────────────────────────────
        #  1) Set all “always‐present” attributes first (so nothing else breaks).
        # ───────────────────────────────────────────────────────────────────────────────────────────

        self.self_dir = None
        self.geo_filepath = None
        self.uq_dir = None

        # Solver-as-object: lazy properties (see bottom of class)
        self._tune_solver = None
        self._eigenmode_solver = None
        self._wakefield_solver = None
        self._tuned_cavity = None

        self.eigenmode_qois_all_modes = {}
        self.Epk_Eacc = None
        self.Bpk_Eacc = None
        self.Q = None
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
        self.sweep_results = {}
        self.sweep_results_uq = {}
        self.uq_fm_results = None
        self.mesh = None
        self.shape = None
        self.shape_multicell = None

        # plot_label: default to `name` if not provided
        self.name = name
        self.plot_label = plot_label if plot_label is not None else name

        self.projectDir = None
        self.bc = 33
        self.eigenmode_qois = {}
        self.custom_eig_qois = {}
        self.wakefield_qois = {}
        self.wake_op_points = {}
        self.convergence_list = []
        self.tune_results = {}
        self.operating_points = None
        self.Q0 = None
        self.inv_eta = None
        self.neighbours = {}
        self.wall_material = None

        # eigenmode results placeholders
        (self.R_Q,
         self.k_fm,
         self.GR_Q,
         self.freq,
         self.e,
         self.b,
         self.G,
         self.ff,
         self.k_cc,
         self.axis_field,
         self.surface_field) = [0] * 11

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
    def tuned(self):
        """Separate Cavity object living in ``<self_dir>/tuned/``.

        Lazy: if the tuned folder exists on disk we instantiate the tuned
        cavity from it on first access. Returns None when the cavity has
        not been tuned yet.
        """
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
        known = ', '.join(sorted(self.tune_variables()))
        if not known:
            raise ValueError(
                f"{type(self).__name__} is not parameterised, so {name!r} cannot be "
                f"perturbed or tuned.")
        raise ValueError(
            f"Unknown variable {name!r} for {type(self).__name__}. "
            f"It accepts: {known}.")

    def get_tune_value(self, name):
        """Read the tune variable *name*. See :meth:`tune_variables`."""
        return self.parameters[name]

    def set_tune_value(self, name, value):
        """Write the tune variable *name*. See :meth:`tune_variables`."""
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

    def clone_for_tuning(self, tuned_parameters, tuned_self_dir, beampipe=None):
        """A cavity of this type carrying ``tuned_parameters``, living in
        ``tuned_self_dir`` (typically ``<self_dir>/tuned/``).

        Generic: the only type-specific step is :meth:`rebuild`. The clone gets its
        own ``geometry/`` folder with a freshly written geometry, and empty result
        caches so later eigenmode/wakefield runs write into the clone's folders.
        """
        clone = self.rebuild(tuned_parameters,
                             beampipe=self.beampipe if beampipe is None else beampipe)
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_self_dir)

        geo_dir = os.path.join(clone.self_dir, 'geometry')
        os.makedirs(geo_dir, exist_ok=True)
        clone.uq_dir = os.path.join(clone.self_dir, 'uq')
        clone.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
        clone.write_geometry(clone.parameters, clone.n_cells, clone.beampipe,
                             write=clone.geo_filepath)
        clone._write_geometry_snapshot()
        return clone

    def _load_tuned_from_disk(self, tuned_dir):
        """Rebuild the tuned cavity from a persisted ``tuned/`` folder.

        Generic: reads the merged tuned parameters from ``tune_info/tune_res.json``
        and reconstructs via :meth:`rebuild`.
        """
        params = dict(self.parameters)
        tune_res_path = os.path.join(str(tuned_dir), 'tune_info', 'tune_res.json')
        if os.path.exists(tune_res_path):
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

        clone = self.rebuild(params, beampipe=self.beampipe)
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_dir)
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

    def _write_geometry_snapshot(self):
        """Persist current ``self.parameters`` next to the geometry file.

        Written as ``<self_dir>/geometry/parameters.json`` so subsequent
        analyses can detect when the in-memory cavity no longer matches
        the cached simulation results.
        """
        if self.self_dir is None:
            return
        geo_dir = Path(self.self_dir) / 'geometry'
        try:
            geo_dir.mkdir(parents=True, exist_ok=True)
            with open(geo_dir / 'parameters.json', 'w') as f:
                json.dump(dict(self.parameters), f, indent=2, default=str)
        except Exception:
            # Snapshot is advisory — never block geometry writes on IO.
            pass

    def _check_geometry_mismatch(self, analysis):
        """Warn if the on-disk geometry snapshot disagrees with
        ``self.parameters``. Returns True when they match (or no snapshot
        exists yet); False when a mismatch was detected and warned.

        ``analysis`` is a short tag (``'eigenmode'``, ``'wakefield'``,
        ``'tune'``) used to label the warning.
        """
        if self.self_dir is None:
            return True
        snap = Path(self.self_dir) / 'geometry' / 'parameters.json'
        if not snap.exists():
            return True
        try:
            with open(snap, 'r') as f:
                saved = json.load(f)
        except Exception:
            return True
        cur = dict(self.parameters)
        changed = []
        for k in set(cur.keys()) | set(saved.keys()):
            vc, vs = cur.get(k), saved.get(k)
            try:
                if vc is None or vs is None or not np.isclose(
                        float(vc), float(vs), rtol=1e-9, atol=1e-9):
                    changed.append(k)
            except (TypeError, ValueError):
                if vc != vs:
                    changed.append(k)
        if not changed:
            return True
        preview = ', '.join(sorted(changed)[:5])
        more = '...' if len(changed) > 5 else ''
        warning(
            f"[{analysis}] geometry mismatch for cavity '{self.name}': "
            f"{len(changed)} parameter(s) differ from the geometry snapshot "
            f"on disk ({preview}{more}). Existing results may not correspond "
            f"to the current in-memory parameters — pass rerun=True to "
            f"regenerate, or delete {self.self_dir} to start fresh."
        )
        return False

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
        self.shape['CELL PARAMETERISATION'] = cell_parameterisation
        self.to_multicell()

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
        self.shape['n_cells'] = n_cells
        self.to_multicell()

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
        self.shape['BP'] = bp
        self.to_multicell()

    def load(self):
        """
        Load existing cavity project folder

        Parameters
        ----------

        Returns
        -------

        """
        pass

    def load_shape_space(self, filepath):
        """
        Get cavity geometric parameters from shape space

        Parameters
        ----------
        filepath: str
            Shape space directory

        Returns
        -------

        """
        pass

    def save_shape_space(self, filepath=None):
        """
        Save current geometric parameters as shape space

        Parameters
        ----------
        filepath: str
            Directory to save shape space to. If no input is given, it is saved to the Cavities directory

        Returns
        -------

        """
        pass

    def sweep(self, sweep_config, which='eigenmode', how='independent', uq_config=None):
        """Parameter sweep — NOT YET PORTED to the current geometry pipeline.

        The old implementation mutated ``self.shape['IC']`` and re-ran the
        legacy eigenmode path. After the refactor, geometry is regenerated
        from ``self.parameters`` (not ``self.shape``), and ``create()`` is a
        no-op once the geometry file exists — so mutating ``shape`` no longer
        changes what is simulated. Porting requires updating ``parameters``,
        invalidating the geometry snapshot, and re-running via
        ``self.eigenmode.run()`` for each sweep point.

        Workaround until then: create one cavity per sweep value (varying the
        parameter in the constructor) and run eigenmode on each.
        """
        raise NotImplementedError(
            "Cavity.sweep is not available in this release (pending a port to "
            "the current geometry pipeline). Workaround: build one cavity per "
            "sweep value and call cav.eigenmode.run() on each.")

    def study_mesh_convergence_fm(self, h=10, h_passes=3, h_step=1, p=2, p_passes=3, p_step=1):
        """

        Parameters
        ----------
        h_passes
        h
        passes
        solver
        type: str
            h or p refinement

        Returns
        -------

        """

        convergence_df_ph = None
        convergence_df_data_ph = None
        # define start value, refinement (adaptive? fixed step?)
        hs = h
        for ip in range(p_passes):
            convergence_dict_h = {}
            for ih in range(h_passes):
                eigenmode_config = {'boundary_conditions': 'mm',
                                    'mesh_config': {
                                        'h': hs,
                                        'p': p
                                    }
                                    }
                if 'polarisation' not in eigenmode_config.keys():
                    eigenmode_config["polarisation"] = ['monopole']

                t0 = time.perf_counter()
                run_eigenmode_s({self.name: self}, eigenmode_config, '')
                elapsed = time.perf_counter() - t0
                # read results
                self.get_eigenmode_qois(eigenmode_config)

                qois_copy = dict(self.eigenmode_qois)
                qois_copy['time [s]'] = elapsed
                convergence_dict_h[ih] = qois_copy
                convergence_dict_h[ih]['h'] = hs
                convergence_dict_h[ih]['p'] = p
                convergence_dict_h[ih]['No of Mesh Elements'] = self.eigenmode_qois['No of Mesh Elements']
                convergence_dict_h[ih]['No of DOFs'] = self.eigenmode_qois.get('No of DOFs', 0)

                hs /= h_step
            convergence_df_data_h = pd.DataFrame.from_dict(convergence_dict_h, orient='index')
            convergence_df_h = self.calculate_rel_errors(convergence_df_data_h)

            if convergence_df_ph is None:
                convergence_df_data_ph = convergence_df_data_h
                convergence_df_ph = convergence_df_h
            else:
                convergence_df_data_ph = pd.concat([convergence_df_data_ph,
                                                    convergence_df_data_h], ignore_index=True)
                convergence_df_ph = pd.concat([convergence_df_ph,
                                               convergence_df_h], ignore_index=True)
            hs = h
            p += p_step

        # convert to dataframe
        self.convergence_df_data = convergence_df_data_ph
        self.convergence_df = convergence_df_ph

    def study_mesh_convergence(self, eigenmode_config=None, h=10, h_passes=3, h_step=1, p=2, p_passes=3, p_step=1):
        """
        Runs a mesh convergence sweep (h- and p-refinement) tracking
        all extracted eigenmodes instead of a single scalar mode.
        """

        # We will collect flat row dictionaries across the entire sweep
        all_rows = []

        hs = h
        for ip in range(p_passes):
            for ih in range(h_passes):
                if eigenmode_config is None:
                    eigenmode_config = {
                        'boundary_conditions': 'mm',
                        'mesh_config': {
                            'h': hs,
                            'p': p
                        }
                    }

                eigenmode_config['mesh_config']['h'] = hs
                eigenmode_config['mesh_config']['p'] = p

                if 'polarisation' not in eigenmode_config.keys():
                    eigenmode_config["polarisation"] = ['monopole']

                t0 = time.perf_counter()
                # Run the multimode solver
                run_eigenmode_s({self.name: self}, eigenmode_config, '')
                self.get_eigenmode_qois(eigenmode_config)
                # print(self.eigenmode_qois_all_modes)
                elapsed = time.perf_counter() - t0

                # Fetch the dictionary containing all modes
                # Expected structure: {'0': {...}, '1': {...}, ...}
                all_modes_data = getattr(self, 'eigenmode_qois_all_modes', {})

                # Fallback if all_modes isn't populated but the single mode is
                if not all_modes_data and hasattr(self, 'eigenmode_qois'):
                    all_modes_data = {'0': self.eigenmode_qois}

                # Loop through every tracked mode in this simulation step
                for mode_str, qois in all_modes_data.items():
                    row = dict(qois)  # Copy all quantities (freq, Q, R/Q, etc.)

                    # Append iteration tracking metadata
                    row['mode_index'] = mode_str
                    row['h'] = hs
                    row['p'] = p
                    row['h_pass'] = ih
                    row['p_pass'] = ip
                    row['time [s]'] = elapsed

                    # Safe checking for element/DOF tracking
                    row['No of Mesh Elements'] = qois.get('No of Mesh Elements',
                                                          self.eigenmode_qois.get('No of Mesh Elements', 0))
                    row['No of DOFs'] = qois.get('No of DOFs', self.eigenmode_qois.get('No of DOFs', 0))

                    all_rows.append(row)

                hs /= h_step
            hs = h
            p += p_step

        # Build the comprehensive raw data dataframe
        convergence_df_data_ph = pd.DataFrame(all_rows)

        # Compute relative errors
        # Note: Ensure your self.calculate_rel_errors handles multi-mode data frames,
        # or apply it grouped by mode index:
        try:
            convergence_df_ph = self.calculate_rel_errors(convergence_df_data_ph)
        except Exception:
            # Fallback if calculate_rel_errors expects single-mode schemas
            convergence_df_ph = convergence_df_data_ph.copy()

            # Assign data structures back to the object instance
        self.convergence_df_data = convergence_df_data_ph
        self.convergence_df = convergence_df_ph

    def calculate_rel_errors(self, df):
        # Specify the columns to exclude
        columns_to_exclude = ['h', 'p', 'No of Mesh Elements', 'No of DOFs', 'time [s]']
        
        # Filter columns to exclude to only those present in df
        actual_exclude = [col for col in columns_to_exclude if col in df.columns]
        df_to_compute = df.drop(columns=actual_exclude)
        df_prev = df_to_compute.shift(1)
        relative_errors = (df_to_compute - df_prev).abs() / df_prev.abs()
        relative_errors.columns = [f'rel_error_{col}' for col in df_to_compute.columns]
        excluded_columns = df[actual_exclude]
        rel_errors_df = pd.concat([relative_errors, excluded_columns], axis=1)

        return rel_errors_df

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
            return True
        except FileNotFoundError as e:
            error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")
            return False

    def run_wakefield(self, MROT=2, MT=10, NFS=10000, wakelength=50, bunch_length=25,
                      DDR_SIG=0.1, DDZ_SIG=0.1, WG_M=None, marker='', operating_points=None, solver='ABCI'):
        """
        Run wakefield analysis on cavity

        Parameters
        ----------
        MROT: {0, 1}
            Polarisation 0 for longitudinal polarization and 1 for transversal polarization
        MT: int
            Number of time steps it takes for a beam to move from one mesh cell to the other
        NFS: int
            Number of frequency samples
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
            'MROT': MROT, 'MT': MT, 'NFS': NFS, 'wakelength': wakelength,
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

    def calc_op_freq(self):
        """
        Calculates operating frequency. The operating frequency is used for tuning when a frequency is not given.
        It is advisable to always include the desired tune frequency. Example

        .. py:function:: cav.run_tune('Req', freq=1300)

        Returns
        -------

        """
        if not self.freq:
            self.freq = (c0 / 4 * self.L)

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
                self.tune_results = json.load(json_file)
            last = last_stage_result(self.tune_results)
            if last is not None and 'FREQ' in last:
                self.freq = last['FREQ']
        else:
            error("Tune results not found. Please tune the cavity")

    def get_eigenmode_qois(self, config=None):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        qois = 'qois.json'

        # Which polarisations to read: those named in the config (normalised to
        # names — a bare 'dipole'/1 or a list are all accepted), or, when no
        # config is given (e.g. the Cavities-level caller), whatever was solved.
        raw = (config or {}).get('polarisation') if isinstance(config, dict) else None
        if raw is not None:
            if not isinstance(raw, (list, tuple, set)):
                raw = [raw]
            poles = [pol_name(pol_number(p)) for p in raw]
        else:
            poles = self._available_polarisations() or ['monopole']

        # Initialize the all-modes dictionary so we can append to it across polarisations
        self.eigenmode_qois_all_modes = {}

        for pole in poles:
            mono_dir = self._eigenmode_pol_dir(pole)
            qois_path = os.path.join(mono_dir, qois)

            if not os.path.exists(qois_path):
                # No monopole result. If higher-order (m-pole) results exist, this
                # was an m-pole-only run — the monopole-derived summary attributes
                # (freq, R/Q, …) simply aren't available; use cav.eigenmode.mpole_qois()
                # for those. Only error when there is no eigenmode result at all.
                available = self._available_polarisations()
                if available:
                    info(f"No monopole eigenmode result for {self.name}; solved "
                         f"polarisations: {available}. Read them with "
                         f"cav.eigenmode.mpole_qois('<pol>').")
                    return
                raise FileNotFoundError(
                    f"No eigenmode result for {self.name} at {qois_path}. "
                    f"Run run_eigenmode() first.")

            # Keep tracking the baseline summary QOIs if monopole is processed
            if pole == 'monopole':
                with open(qois_path) as json_file:
                    self.eigenmode_qois = json.load(json_file)

            # Process the all-modes JSON and dynamically append/remap keys
            all_modes_path = os.path.join(mono_dir, 'qois_all_modes.json')
            if os.path.exists(all_modes_path):
                with open(all_modes_path) as json_file:
                    data = json.load(json_file)

                    # Loop through each mode profile and modify the keys
                    for mode_idx, mode_data in data.items():
                        m_value = mode_data.get('m', 'unknown')
                        new_key = f"{mode_idx}-{m_value}"

                        # Store inside our master dictionary
                        self.eigenmode_qois_all_modes[new_key] = mode_data

        # with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
        #     self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        # Monopole-derived summary attributes. Only set them when a monopole
        # result was actually read — an m-pole-only run leaves eigenmode_qois
        # empty, and those scalars aren't defined without the monopole.
        if self.eigenmode_qois:
            self.freq = self.eigenmode_qois['freq [MHz]']
            self.k_cc = self.eigenmode_qois['kcc [%]']
            self.ff = self.eigenmode_qois['ff [%]']
            self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
            self.GR_Q = self.eigenmode_qois['GR/Q [Ohm^2]']
            self.G = self.GR_Q / self.R_Q
            self.Q = self.eigenmode_qois['Q []']
            self.e = self.eigenmode_qois['Epk/Eacc []']
            self.b = self.eigenmode_qois['Bpk/Eacc [mT/MV/m]']
            self.Epk_Eacc = self.e
            self.Bpk_Eacc = self.b

    @staticmethod
    def _phase_advance_label(j, n):
        """Tick label for a phase advance of ``j*pi/n`` as a reduced fraction of
        pi, rendered with mathtext (no LaTeX install needed): ``pi/2``, ``2pi/3``,
        ``pi``.
        """
        f = Fraction(j, n)
        if f == 1:
            return r'$\pi$'
        num = '' if f.numerator == 1 else str(f.numerator)
        return rf'$\dfrac{{{num}\pi}}{{{f.denominator}}}$'

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
        ``Cavities`` overlay plotters so both render identically."""
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
                    info(f"{pretty}: passband {b} was not computed "
                         f"(only {n_passbands} available).")
                    continue
                seg = freqs[(b - 1) * n:b * n]
                k = np.linspace(np.pi / n, np.pi, n, endpoint=True)[:len(seg)]
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
            ticks = np.linspace(np.pi / n, np.pi, n, endpoint=True)
            ticklabels = [Cavity._phase_advance_label(j + 1, n) for j in range(n)]

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
                clusters = Cavity._clusters_from_breaks(yvals, breaks)
            else:
                clusters = Cavity._frequency_clusters(yvals)

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

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
            ax.text(
                0.95, 0.05,  # Position (normalized coordinates)
                r'$\eta=' + fr'{self.ff:.2f}\%' + '$',  # Text content
                fontsize=12,  # Font size
                ha='right',  # Horizontal alignment
                va='bottom',  # Vertical alignment
                transform=plt.gca().transAxes  # Use axes-relative positioning
            )
            ax.legend(loc="upper right")
            if show_min_max:
                minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 50, width=100)
                Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                ax.plot(self.Ez_0_abs['z(0, 0)'][peaks], Ez_0_abs_peaks, marker='o', ls='')
                ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                ax.axhline(max(Ez_0_abs_peaks), c='k')
        else:
            if os.path.exists(os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')):
                with open(os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
                ax.text(
                    0.95, 0.05,  # Position (normalized coordinates)
                    r'$\eta=$' + fr'{self.ff:.2f}\%' + '$',  # Text content
                    fontsize=12,  # Font size
                    ha='right',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                    transform=plt.gca().transAxes  # Use axes-relative positioning
                )
                ax.legend(loc="upper right")
                if show_min_max:
                    minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                    peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 100,
                                          width=100)
                    Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                    ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                    ax.axhline(max(Ez_0_abs_peaks), c='k')
            else:
                error('Axis field plot data not found.')

    def plot_spectra(self, var, ax=None):
        if len(self.uq_fm_results_all_modes) != 0:
            results = self.uq_fm_results_all_modes

            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 4))

            # Generate KDE data for each mode found in the nested results
            # Sort modes numerically if possible
            try:
                sorted_modes = sorted(results.keys(), key=lambda x: int(x))
            except ValueError:
                sorted_modes = sorted(results.keys())

            for mode_idx in sorted_modes:
                qois = results[mode_idx]
                if var in qois:
                    stats = qois[var]
                    mean = stats["expe"][0]
                    std = stats["stdDev"][0]

                    if std == 0:
                        # Avoid division by zero, plot a vertical line for zero-variance results
                        ax.axvline(mean, ls='--', alpha=0.5, label=f"{var}_{mode_idx} (std=0)")
                        continue

                    # Dynamic range based on standard deviation
                    x_values = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
                    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
                    
                    ax.plot(x_values, y_values, label=f"{var}_{mode_idx}")

            # Plot settings
            ax.set_title(f"KDE Plots for {var}")
            ax.set_xlabel(var)
            ax.set_ylabel("Density")
            ax.legend()

        else:
            error("uq_fm_results_all_modes not found")

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
        """
        Get the quantities of interest written by the ABCI code

        Parameters
        ----------
        opt: {'SR', 'BS'}
            SR - Synchrotron radiation bunch length
            BS - Bremsstrahlung

        Returns
        -------

        """

        if os.path.exists(os.path.join(self.self_dir, 'wakefield', 'qois.json')):
            with open(os.path.join(self.self_dir, 'wakefield', 'qois.json')) as json_file:
                all_wakefield_qois = json.load(json_file)

        # get only keys in op_points
        if 'operating_points' in wakefield_config:
            for op_pt in wakefield_config['operating_points'].keys():
                for key, val in all_wakefield_qois.items():
                    if op_pt in key:
                        self.wakefield_qois[key] = val

            for key, val in self.wakefield_qois.items():
                self.k_fm[key] = val['k_FM [V/pC]']
                self.k_loss[key] = val['|k_loss| [V/pC]']
                self.k_kick[key] = val['|k_kick| [V/pC/m]']
                self.phom[key] = val['P_HOM [kW]']
                self.I0[key] = val['I0 [mA]']

    def plot_animate_wakefield(self, save=False):
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

    def get_power_uq(self, rf_config, op_points_list):
        stat_moms = ['expe', 'stdDev', 'skew', 'kurtosis']
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois_uq[op_pt] = {}
            val = self.operating_points[op_pt]
            for kk, vv in val.items():
                if 'sigma' in kk:
                    sig_id = kk.split('_')[-1].split(' ')[0]

                    if 'Eacc [MV/m]' in rf_config.keys():
                        op_field = {'expe': [self.Eacc_rf_config * 1e6], 'stdDev': [0], 'skew': [0], 'kurtosis': [0]}
                    else:
                        op_field = {'expe': [val['Eacc [MV/m]'] * 1e6], 'stdDev': [0]}
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
                        r"PHOM/cav [kW]": self.uq_hom_results[fr'P_HOM [kW]_{op_pt}_{sig_id}_{vv}mm']
                    }
        return self.rf_performance_qois_uq

    def get_power(self, rf_config, op_points_list):
        for ii, op_pt in enumerate(op_points_list):
            self.rf_performance_qois[op_pt] = {}
            val = self.operating_points[op_pt]
            for kk, vv in val.items():
                if 'sigma' in kk:
                    sig_id = kk.split('_')[-1].split(' ')[0]

                    if 'Eacc [MV/m]' in rf_config.keys():
                        op_field = self.Eacc_rf_config * 1e6
                    else:
                        op_field = val['Eacc [MV/m]'] * 1e6
                        self.Eacc_rf_config = op_field

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
                        r"PHOM/cav [kW]": self.phom[fr'{op_pt}_{sig_id}_{vv}mm']
                    }
        return self.rf_performance_qois

    def get_uq_post(self, qoi):
        pass

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

    def _plot_profile(self, profile, ax=None, mirror=True, fill=True, **kwargs):
        """Draw a cavity's meridian outline straight from its :class:`Profile`.

        Geometry-independent: works for every model, since each builds a Profile.
        The wall is drawn in mm; ``mirror`` reflects it about the axis for the full
        cross-section and ``fill`` shades the interior.
        """
        with house_style():
            if ax is None:
                _, ax = plt.subplots(figsize=(8, 3))
            pts = np.asarray(profile.contour_points(3e-4, skip=('AXI',)), dtype=float)
            z, r = pts[:, 0] * 1e3, pts[:, 1] * 1e3          # metres -> mm
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
            ax.set_aspect('equal')
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel(r'$r$ [mm]')
            return ax

    def _wake_xy(self, pol, kind):
        """``(x, y)`` for a wake plot, read from the normalised wakefield frame
        (solver-agnostic; ``pol`` is ``'z'``/``'t'``, ``kind`` is ``'impedance'``
        or ``'wake'``). Empty arrays when there is no result yet."""
        df = self.wakefield.wake_z if pol == 'z' else self.wakefield.wake_t
        if df is None or df.empty:
            return np.array([]), np.array([])
        if kind == 'impedance':
            cols = ['f [MHz]', '|Z| [Ohm]' if pol == 'z' else '|Z| [Ohm/m]']
        else:
            cols = ['s [m]', 'W [V/pC]' if pol == 'z' else 'W [V/pC/m]']
        sub = df[cols].dropna()
        return sub[cols[0]].to_numpy(), sub[cols[1]].to_numpy()

    def plot(self, what, ax=None, scale_x=1, **kwargs):
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

        wake_specs = {
            'zl': ('z', 'impedance', scale_x, 'Wake. Long.', 'f [MHz]',
                   r"$Z_{\parallel} ~[\mathrm{k\Omega}]$", 1.8),
            'zt': ('t', 'impedance', scale_x, 'Wake. Trans.', 'f [MHz]',
                   r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$", 1.8),
            'wpl': ('z', 'wake', 1.0, 'Longitudinal wake potentials',
                    'Distance from Bunch Head S [m]',
                    r"Scaled Wake Potentials $W (S)$ [V/pC]", 1.8),
            'wpt': ('t', 'wake', 1.0, 'Transversal wake potentials',
                    'Distance from Bunch Head S [m]',
                    r"Scaled Wake Potentials $W (S)$ [V/pC/m]", 3),
        }
        if what.lower() in wake_specs:
            pol, kind, sx, tag, xlabel, ylabel, lw = wake_specs[what.lower()]
            x, y = self._wake_xy(pol, kind)
            if len(x) == 0:
                info("No wakefield data — run cav.wakefield.run(...) first.")
                return ax
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
            kwargs.setdefault('c', self.color)
            kwargs.setdefault('lw', lw)
            ax.plot(x * sx, y, label=fr'{self.name} ({tag})', **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
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
                info("Convergence data not available.")

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

    def plot_mesh(self, plotter='ngsolve', pol='monopole'):
        mesh_path = self._eigenmode_pol_dir(pol)
        if os.path.exists(os.path.join(mesh_path, 'mesh.pkl')):
            ngsolve_mevp.plot_mesh(mesh_path, plotter=plotter)
        else:
            self._eigenmode_artifact_error(pol, 'mesh.pkl', 'mesh')

    def plot_fields(self, mode=1, which='E', plotter='ngsolve', pol='monopole'):
        """Plot eigenmode fields. For m-pole results pass ``pol`` ('dipole',
        'quadrupole', ... or the mode number m); *which* then accepts 'E'/'H'
        (in-plane envelopes) as well as 'Ephi'/'Hphi' (azimuthal envelopes)."""
        field_path = self._eigenmode_pol_dir(pol)
        if os.path.exists(os.path.join(field_path, 'gfu_EH.pkl')):
            ngsolve_mevp.plot_fields(field_path, mode, which, plotter)
        else:
            self._eigenmode_artifact_error(pol, 'gfu_EH.pkl', 'field data')

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

    def define_operating_points(self, op):
        self.operating_points = op

    def inspect(self, cell_type='mid-cell', variation=0.2):

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

    def config_sample(self, kind):
        if kind == 'eigenmode':
            return EIGENMODE_CONFIG

        if kind == 'wakefield':
            return WAKEFIELD_CONFIG

        if kind == 'tune':
            return TUNE_CONFIG

        if kind == 'uq':
            return UQ_CONFIG

        if kind == 'optimisation':
            pass

        if kind == 'sa':
            return
    # def inspect(self, cell_type='mid-cell', variation=0.2, tangent_check=False):
    #
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.set_aspect('equal')
    #
    #     if cell_type == 'mid-cell':
    #         cell = self.shape['IC']
    #     elif cell_type == 'end-cell-left':
    #         cell = self.shape['OC']
    #     elif cell_type == 'end-cell-right':
    #         cell = self.shape['OC_R']
    #     else:
    #         cell = self.shape['IC']
    #
    #     if self.cell_parameterisation == 'flattop':
    #         A_, B_, a_, b_, Ri_, L_, Req_, l_ = cell[:8]
    #
    #         # Define the function that plots the graph
    #         def plot_cavity_geometry_flattop(A, B, a, b, Ri, L, Req, l):
    #             cell = np.array([A, B, a, b, Ri, L, Req, l])
    #             write_cavity_geometry_cli_flattop(cell, cell, cell, BP='none',
    #                                               n_cell=1, tangent_check=True, lw=1,
    #                                               plot=True,
    #                                               ignore_degenerate=True)
    #
    #             # Update the sum display
    #             sum_label.value = f'Sum of A + a + l: {A + a + l:.2f}, L: {L}, delta: {A + a + l - L}'
    #
    #         # Create sliders for each variable
    #         A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
    #                                        description='A')
    #         B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
    #                                        description='B')
    #         a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
    #                                        description='a')
    #         b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
    #                                        description='b')
    #         Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
    #                                         description='Ri')
    #         L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
    #                                        description='L')
    #         Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
    #                                          value=Req_,
    #                                          description='Req')
    #         l_slider = widgets.FloatSlider(min=(1 - variation) * l_, max=(1 + variation) * l_, step=0.1, value=l_,
    #                                        description='l')
    #         # Create a label to display the sum of A + a
    #         sum_label = Label()
    #
    #         # Arrange the sliders in a 3x3 layout
    #         ui = VBox([
    #             HBox([A_slider, B_slider, a_slider]),
    #             HBox([b_slider, Ri_slider, L_slider]),
    #             HBox([Req_slider, l_slider]),
    #             sum_label  # Add the sum label to the layout
    #         ])
    #
    #         # Create an interactive widget to update the plot
    #         out = widgets.interactive_output(plot_cavity_geometry_flattop,
    #                                          {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
    #                                           'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider, 'l': l_slider})
    #
    #     else:
    #         A_, B_, a_, b_, Ri_, L_, Req_ = cell[:7]
    #
    #         # Define the function that plots the graph
    #         def plot_cavity_geometry(A, B, a, b, Ri, L, Req):
    #             cell = np.array([A, B, a, b, Ri, L, Req])
    #             # ax.clear()
    #             write_cavity_geometry_cli(cell, cell, cell, BP='none', n_cell=1,
    #                                       tangent_check=tangent_check, lw=3,
    #                                       plot=True, ax=ax,
    #                                       ignore_degenerate=True)
    #             # Update the sum display
    #             sum_label.value = f'Sum of A + a: {A + a:.2f}, L: {L}, delta: {A + a - L}'
    #
    #         def run_eigenmode(b):
    #             eigenmode_config = {'processes': 1}
    #             boundary_conds = 'mm'
    #             eigenmode_config['boundary_conditions'] = BOUNDARY_CONDITIONS_DICT[boundary_conds]
    #
    #             shape_space = {}
    #
    #             # run_eigenmode_s({self.name: self.shape}, {self.name: self.shape_multicell}, self.projectDir, eigenmode_config)
    #
    #         # Create sliders for each variable
    #         A_slider = widgets.FloatSlider(min=(1 - variation) * A_, max=(1 + variation) * A_, step=0.1, value=A_,
    #                                        description='A')
    #         B_slider = widgets.FloatSlider(min=(1 - variation) * B_, max=(1 + variation) * B_, step=0.1, value=B_,
    #                                        description='B')
    #         a_slider = widgets.FloatSlider(min=(1 - variation) * a_, max=(1 + variation) * a_, step=0.1, value=a_,
    #                                        description='a')
    #         b_slider = widgets.FloatSlider(min=(1 - variation) * b_, max=(1 + variation) * b_, step=0.1, value=b_,
    #                                        description='b')
    #         Ri_slider = widgets.FloatSlider(min=(1 - variation) * Ri_, max=(1 + variation) * Ri_, step=0.1, value=Ri_,
    #                                         description='Ri')
    #         L_slider = widgets.FloatSlider(min=(1 - variation) * L_, max=(1 + variation) * L_, step=0.1, value=L_,
    #                                        description='L')
    #         Req_slider = widgets.FloatSlider(min=(1 - variation) * Req_, max=(1 + variation) * Req_, step=0.1,
    #                                          value=Req_,
    #                                          description='Req')
    #         # Create a label to display the sum of A + a
    #         sum_label = Label()
    #
    #         # create run tune and run eigenmode button
    #         button_tune = widgets.Button(description="Tune")
    #         button_eigenmode = widgets.Button(description="Eigenmode")
    #         button_eigenmode.on_click(run_eigenmode)
    #
    #         # Arrange the sliders in a 3x3 layout
    #         ui = VBox([
    #             HBox([A_slider, B_slider, a_slider]),
    #             HBox([b_slider, Ri_slider, L_slider]),
    #             HBox([Req_slider]),
    #             sum_label,  # Add the sum label to the layout
    #             # HBox([button_tune, button_eigenmode])
    #         ])
    #
    #         # Create an interactive widget to update the plot
    #         out = widgets.interactive_output(plot_cavity_geometry,
    #                                          {'A': A_slider, 'B': B_slider, 'a': a_slider, 'b': b_slider,
    #                                           'Ri': Ri_slider, 'L': L_slider, 'Req': Req_slider})
    #
    #     # Display the layout
    #     display(out, ui)

    def eval_expr(self, expr, symbols):
        """Evaluate numeric expression with symbols."""
        node = ast.parse(expr, mode='eval').body

        def _eval(n):
            if isinstance(n, ast.Num): return n.n
            if isinstance(n, ast.Name): return symbols[n.id]
            if isinstance(n, ast.BinOp): return _ops[type(n.op)](_eval(n.left), _eval(n.right))
            if isinstance(n, ast.UnaryOp): return _ops[type(n.op)](_eval(n.operand))
            raise ValueError(f"Unsupported expression: {expr}")

        return _eval(node)

    # Main converter
    def geo_to_abc(self, wakefield_config=None, folder=None, **kwargs):
        """Write this cavity's ABCI input deck(s).

        The contour comes from the unified :class:`~cavsim2d.geometry.Profile`
        (densified to the mesh spacing, exact for ellipse / circle / spline
        walls), with beam pipes guaranteed at both ends because ABCI refuses to
        run without them. Cavities that expose no ``profile()`` — an imported
        ``.geo``/CAD — fall back to parsing the ``.geo`` text.
        """
        self.create()

        cfg = wakefield_config or {}
        mesh_cfg = cfg.get('mesh_config') or {}
        ddr = float(mesh_cfg.get('DDR', 0.00125))
        ddz = float(mesh_cfg.get('DDZ', 0.00125))
        ds = float(cfg.get('contour_ds', min(ddr, ddz)))
        # ABCI wants at least 5 mesh lengths of pipe; ask for a little margin.
        min_pipe = 6.0 * ddz
        pipe_length = cfg.get('beampipe_length')      # metres; None -> 3x axial length

        maker = getattr(self, 'profile', None)
        profile = maker() if callable(maker) else None

        if profile is not None:
            # ABCI's own arc primitive is kept: replacing an arc that meets a beam
            # pipe tangentially (an elliptical cavity's iris) with a dense polyline
            # makes ABCI's mesher emit NaN wake potentials.
            items, added_l, added_r = abci_shape(profile, ds, min_pipe, pipe_length)
            if added_l or added_r:
                info(f"{self.name}: ABCI needs a beam pipe at each end; added "
                     f"{added_l * 1e3:.1f} mm on the left and {added_r * 1e3:.1f} mm on "
                     f"the right.")
            zmin = min(it[1] if it[0] == 'point' else it[2][0] for it in items)
            segments = []
            for it in items:
                if it[0] == 'point':
                    segments.append(('point', it[2], it[1] - zmin))
                else:
                    (zc, rc), (ze, re) = it[1], it[2]
                    segments.append(('arc', rc, zc - zmin, re, ze - zmin))
            points = {}
        else:
            points, segments, zmin = self._contour_from_geo()

        wakefield_folder_structure = {
            'wakefield': {
                'longitudinal': None,
                'transversal': None
            }
        }
        if folder:
            make_dirs_from_dict(wakefield_folder_structure, folder)
        else:
            make_dirs_from_dict(wakefield_folder_structure, self.self_dir)
            folder = self.self_dir

        MROT = resolve_mrot(wakefield_config)
        if MROT == 2:
            for m in range(2):
                self._write_abc(points, segments, zmin, {}, m, wakefield_config, folder, **kwargs)
        else:
            self._write_abc(points, segments, zmin, {}, MROT, wakefield_config, folder, **kwargs)

    def _contour_from_geo(self):
        """Legacy contour extraction: regex-parse the gmsh ``.geo`` text.

        Only understands ``Point(...)`` and ``Ellipse(...)``, and encodes each
        ellipse as an ABCI ``-3.`` *circular* arc about the ellipse centre — which
        is only faithful when the ellipse is a circle. Kept for cavities with an
        imported ``.geo`` and no ``profile()``.
        """
        with open(self.geo_filepath) as f:
            lines = [l.split('%', 1)[0].strip() for l in f]

        symbols = {}
        define_re = re.compile(r"(\w+)\s*=\s*DefineNumber\[\s*([^,\]]+)")
        for line in lines:
            m = define_re.search(line)
            if m:
                symbols[m.group(1)] = m.group(2).strip()
        changed = True
        while changed:
            changed = False
            for k, v in list(symbols.items()):
                if isinstance(v, str):
                    try:
                        symbols[k] = self.eval_expr(v, symbols)
                        changed = True
                    except Exception:
                        pass

        point_re = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\}")
        ellipse_re = re.compile(r"Ellipse\(\d+\)\s*=\s*\{([^}]+)\}")

        points = {}
        segments = []
        zvals = []
        for line in lines:
            m = point_re.match(line)
            if m:
                pid = int(m.group(1))
                coords = [p.strip() for p in m.group(2).split(',')]
                z = self.eval_expr(coords[0], symbols)
                r = self.eval_expr(coords[1], symbols)
                points[pid] = (r, z)
                zvals.append(z)

        zmin = min(zvals) if zvals else 0.0

        for line in lines:
            if line.startswith('Point('):
                m = point_re.match(line)
                pid = int(m.group(1))
                r, z = points[pid]
                segments.append(('point', r, z - zmin))
            elif line.startswith('Line('):
                continue
            elif line.startswith('Ellipse('):
                m = ellipse_re.match(line)
                ids = [int(x.strip()) for x in m.group(1).split(',')]
                if len(segments) >= 3:
                    segments = segments[:-3]
                segments.append(('ellipse', ids[1], ids[3]))

        return points, segments, zmin

    def _write_abc(self, points, segments, zmin, symbols, MROT, wakefield_config, folder, **kwargs):
        # defaults
        RDRIVE, ISIG = 5e-3, 5
        LCRBW = 'F'  # counter-rotating beam
        ZSEP = 0.0
        BSEP = 0
        NBUNCH = 1
        BETA = 1
        LMATPR = '.F.'
        LPRW, LPPW, LSVW, LSVWA, LSVWT, LSVWL, LSVF = '.T.', '.T.', '.T.', '.F.', '.T.', '.T.', '.F.'
        LSAV, LCPUTM = 'F', 'F'
        LCBACK = '.T.'
        LPLE = '.F.'
        NSHOT = 0
        UBT = 50
        SIG = 25e-3
        MT = 4
        mesh_DDR = 0.00125
        mesh_DDZ = 0.00125

        # unpack kwargs
        for key, value in kwargs.items():
            if key == 'RADIAL BEAM OFFSET AT (RDRIVE)':
                RDRIVE = value
            if key == 'NUMBER OF WAKE POTENTIAL POINTS (NW)':
                NW = value
            if key == 'WAKE FOR A COUNTER-ROTATING BEAM (LCRBW)':
                LCRBW = value
            if key == 'VELOCITY OF THE BUNCH / C (BETA)':
                BETA = value
            if key == 'PRINTOUT OF CAVITY SHAPE USED (LMATPR)':
                LMATPR = value
            if key == 'PRINTOUT OF WAKE POTENTIALS (LPRW)':
                LPRW = value
            if key == 'LINE-PRINTER PLOT OF WAKE POT. (LPPW)':
                LPPW = value
            if key == 'SAVE WAKE POTENTIALS IN A FILE (LSVW)':
                LSVW = value
            if key == 'SAVE AZIMUTHAL WAKE IN A FILE (LSVWA)':
                LSVWA = value
            if key == 'SAVE TRANSVERSE WAKE IN A FILE (LSVWT)':
                LSVWT = value
            if key == 'SAVE LONGITUDINAL WAKE IN A FILE (LSVWL)':
                LSVWL = value
            if key == 'SAVE FFT RESULTS IN A FILE (LSVF)':
                LSVF = value
            if key == 'SAVE FIELDS INTO FILE (LSAV)':
                LSAV = value
            if key == 'CPUTIME MONITOR ACTIVE (LCPUTM)':
                LCPUTM = value

        if 'save_fields' in wakefield_config.keys():
            LPLE, LCBACK = 'T', 'F'
            if isinstance(wakefield_config['save_fields'], dict):
                if 'nshot' in wakefield_config['save_fields'].keys():
                    NSHOT = wakefield_config['save_fields']['nshot']

        if 'wake_config' in wakefield_config.keys():
            wake_config = wakefield_config['wake_config']
            if 'MT' in wake_config.keys():
                MT = wake_config['MT']
            if 'counter_rotating'in wakefield_config['wake_config'].keys():
                LCRBW = 'T'
                if 'separation' in wakefield_config['wake_config']['counter_rotating'].keys():
                    ZSEP = wakefield_config['wake_config']['counter_rotating']['separation']

        if 'beam_config' in wakefield_config.keys():
            if 'beam_offset' in wakefield_config['beam_config'].keys():
                RDRIVE = wakefield_config['beam_config']['beam_offset']
            if 'nbunch' in wakefield_config['beam_config'].keys():
                NBUNCH = wakefield_config['beam_config']['nbunch']
            if 'separation' in wakefield_config['beam_config'].keys():
                BSEP = wakefield_config['beam_config']['separation']

        with open(os.path.join(folder, 'wakefield', MROT_DICT[MROT], 'cavity.abc'), 'w') as out:
            # Header
            out.write(f' &FILE LSAV = .{LSAV}., ITEST = 0, LREC = .F., LCPUTM = .{LCPUTM}. &END \n')
            out.write(' SAMPLE INPUT #1 A SIMPLE CAVITY STRUCTURE \n')
            out.write(' &BOUN  IZL = 3, IZR = 3  &END \n')
            out.write(f' &MESH DDR = {mesh_DDR}, DDZ = {mesh_DDZ} &END \n')
            out.write(" #CAVITYSHAPE\n0.\n")

            # Body
            for seg in segments:
                if seg[0] == 'point':
                    _, r, z = seg
                    out.write(f"{r:.16f} {z:.16f}\n")
                elif seg[0] == 'arc':
                    # '-3.' introduces an arc given by its centre, then its end point.
                    # Emitted by the Profile path; already z-shifted.
                    _, r_c, z_c, r_e, z_e = seg
                    out.write("-3., 0.000\n")
                    out.write(f"{r_c:.16f} {z_c:.16f}\n")
                    out.write(f"{r_e:.16f} {z_e:.16f}\n")
                elif seg[0] == 'ellipse':
                    # legacy .geo path: point ids into `points`
                    _, pid_m, pid_e = seg
                    r_m, z_m = points[pid_m]
                    r_e, z_e = points[pid_e]
                    out.write("-3., 0.000\n")
                    out.write(f"{r_m:.16f} {z_m - zmin:.16f}\n")
                    out.write(f"{r_e:.16f} {z_e - zmin:.16f}\n")

            # Closing
            out.write("0.000 0.000\n9999. 9999.\n")

            # wakefield simulation paramters
            out.write(f' &BEAM  SIG = {SIG}, ISIG = {ISIG}, RDRIVE = {RDRIVE}, MROT = {MROT}, NBUNCH = {NBUNCH}, BSEP = {BSEP} &END \n')
            # out.write(' &BEAM  SIG = {}, MROT = {}, RDRIVE = {}  &END \n'.format(SIG, MROT, 0.005))
            out.write(f' &TIME  MT = {int(MT)} &END \n')
            out.write(
                f' &WAKE  UBT = {int(UBT)}, LCRBW = .{LCRBW}., LCBACK = {LCBACK}, LCRBW = .{LCRBW}., ZSEP = {ZSEP} &END \n')  # , NFS = {NFS}
            # f.write(' &WAKE  UBT = {}, LCHIN = F, LNAPOLY = F, LNONAP = F &END \n'.format(UBT, wake_offset))
            # f.write(' &WAKE R  = {}   &END \n'.format(wake_offset))
            out.write(f' &PLOT  LCAVIN = .T., LCAVUS = .F., LPLW = .T., LFFT = .T., LSPEC = .T., '
                    f'LINTZ = .F., LPATH = .T., LPLE = {LPLE}, LPLC= .F. &END \n')
            out.write(f' &PRIN  LMATPR = {LMATPR}, LPRW = {LPRW}, LPPW = {LPPW}, LSVW = {LSVW}, '
                    f'LSVWA = {LSVWA}, LSVWT = {LSVWT}, LSVWL = {LSVWL},  LSVF = {LSVF}   &END\n')
            out.write('\nSTOP\n')

    def to_multicell(self):
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

    def _overwriteFolder(self, invar, projectDir, name):
        path = os.path.join(self.self_dir, 'eigenmode', f'_process_{invar}')
        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}

        os.makedirs(path)

    @staticmethod
    def _copyFiles(invar, parentDir, projectDir, name):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, name, 'eigenmode', f'_process_{invar}', 'SLANS_exe')

        dir_util.copy_tree(src, dst)

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
    def spawn(self, difference, folder):
        """A container of perturbed cavities, one per row of *difference*.

        Generic, built on :meth:`rebuild`: each row is a partial parameter dict
        overlaid on this cavity's own parameters. Used by UQ and optimisation.

        (This used to call ``type(self)(name=..., geo_filepath=...)``, which no
        model's constructor accepts — every non-elliptical UQ died with
        ``TypeError: Pillbox.__init__() got an unexpected keyword argument 'name'``.)
        """
        # Deferred: breaks the base <-> cavities import cycle.
        from cavsim2d.cavity.cavities import Cavities
        spawn = Cavities(folder, _skip_project_init=True)
        os.makedirs(folder, exist_ok=True)

        for key, row in difference.iterrows():
            params = dict(self.parameters)
            for k, v in row.items():
                if k in params:
                    params[k] = v

            scav = self.rebuild(params)
            scav.name = str(key)
            scav.projectDir = folder
            scav.self_dir = os.path.join(folder, str(key))
            scav.uq_dir = os.path.join(scav.self_dir, 'uq')
            geo_dir = os.path.join(scav.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)

            scav.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            scav.write_geometry(scav.parameters, scav.n_cells, scav.beampipe,
                                write=scav.geo_filepath)
            scav._write_geometry_snapshot()

            spawn.cavities_list.append(scav)
            spawn.cavities_dict[scav.name] = scav
            spawn.shape_space[scav.name] = scav.shape
            spawn.shape_space_multicell[scav.name] = scav.shape_multicell

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
                print(f"Warning: parameter '{name}' not found in {os.path.basename(input_geo)}")
                info(fr"Here are available parameters: {(self.parameters.keys())}")

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(input_geo))

        with open(output_path, "w") as f:
            f.write(content)

        # print(f"Updated .geo file written to: {output_path}")
        return output_path


