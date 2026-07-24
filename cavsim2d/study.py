from IPython.core.display import HTML, display_html, Math
from IPython.core.display_functions import display
from abc import ABC, abstractmethod
from pathlib import Path
from cavsim2d.models.base import Cavity
from cavsim2d.constants import *
from cavsim2d.processes import *
from cavsim2d.utils.shared_functions import *
from scipy.special import *
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
from cavsim2d.processes.tune import normalize_cell_type_config
from cavsim2d.utils.config_validation import validate_tune_config
from cavsim2d.utils.config_validation import validate_eigenmode_config
from cavsim2d.utils.config_validation import validate_wakefield_config
from cavsim2d.utils.config_validation import validate_optimisation_config
import re
from cavsim2d.solvers.ABCI.abci import resolve_mrot
from cavsim2d.solvers.solver_objects import (OptimisationSolver, StudyEigenmode,
                                             StudyWakefield, merge_config,
                                             DEFAULT_EIGENMODE_CONFIG,
                                             DEFAULT_WAKEFIELD_CONFIG, _maybe_show)
from cavsim2d.utils.config_validation import require
from cavsim2d.utils.style import house_style, WARM

class Study:
    """A study: a manager over several RF-device simulations (cavities, guns, …).

    Groups devices under one project folder to run and compare analyses across
    them. A single device can also be analysed on its own without a ``Study`` —
    see :meth:`Cavity.set_workspace` / :meth:`Cavity.save`.
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
        self.cavities_list = cavities_list
        self.cavities_dict = {}
        if cavities_list is None or cavities_list == []:
            self.cavities_list = []
        else:
            self.add_cavity(cavities_list, names_list)

        self.name = 'cavities'
        if name:
            if not isinstance(name, str):
                raise ValueError('Study name must be a string.')
            self.name = name

        if not _skip_project_init and folder is not None:
            self.save(folder, overwrite)

        self.uq_nodes = {}
        self.uq_fm_results_all_modes = {}
        self.rf_config = None
        self.power_qois_uq = {}
        self.shape_space = {}
        self.shape_space_multicell = {}
        self.sweep_results = None
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

        self.E_acc = np.linspace(0.5, 30, 100) * 1e6  # V/m
        self.set_cavities_field()

    @property
    def tuned(self):
        """Return a Study view containing each cavity's tuned version.

        Each element is `cav.tuned` — a full cavity object with its own
        folder (geometry, eigenmode, etc.), so you can call `.plot`,
        `.eigenmode.run`, etc. on it. Cavities without a tuned version
        are omitted.
        """
        tuned_cavs = []
        tuned_names = []
        for cav in self.cavities_list:
            t = cav.tuned
            if t is not None:
                tuned_cavs.append(t)
                tuned_names.append(t.name or f'{cav.name}_tuned')
        if not tuned_cavs:
            warning('No cavities in this Study have a tuned version yet. '
                    'Run run_tune(...) first.')
            return None
        view = Study(self.projectDir, name=f'{self.name}_tuned',
                     _skip_project_init=True)
        for tc, nm in zip(tuned_cavs, tuned_names):
            view.add_cavity(tc, nm)
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
            cavs.projectDir = self.projectDir

            if names:
                cavs.set_name(names)
            else:
                cavs.set_name(f'cav_{len(self.cavities_list)}')

            if isinstance(plot_labels, list):
                cavs.set_plot_label(plot_labels[0])
            else:
                cavs.set_plot_label(plot_labels)

            cavs.create()
            self.cavities_list.append(cavs)
            self.cavities_dict[cavs.name] = cavs
            self.shape_space[cavs.name] = cavs.shape
            self.shape_space_multicell[cavs.name] = cavs.shape_multicell

        else:
            if names is not None:
                require(len(cavs) == len(names), 'Number of cavities does not correspond to number of names.')
            else:
                names = [f'cav_{ii}' for ii in range(len(self.cavities_list))]

            if plot_labels is not None:
                require(len(cavs) == len(plot_labels), 'Number of cavities does not correspond to number of labels.')
            else:
                plot_labels = names

            for i1, cav in enumerate(cavs):
                cav.projectDir = self.projectDir
                cav.set_name(names[i1])
                cav.create()

                cav.set_plot_label(plot_labels[i1])

                self.cavities_list.append(cav)
                self.cavities_dict[cav.name] = cav

                self.shape_space[cav.name] = cav.shape
                self.shape_space_multicell[cav.name] = cav.shape_multicell

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
            error('Please specify a folder to write the simulation results to.')
            return
        else:
            try:
                self.projectDir = project_folder
                success, i = self._create_project(overwrite)
                if not success:
                    error(f"Project {project_folder} could not be created. Please check the folder and try again.")
                    return
                else:
                    if i == 1:
                        done(f"Project {project_folder} created successfully.")
                    if i == 2:
                        done(f"Project {project_folder} created already exists. \n\tSet `overwrite=True` to overwrite.")

            except Exception as e:
                error("Exception occurred: ", e)
                return

        if project_folder is None:
            self.projectDir = Path(os.getcwd())

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
                except Exception as e:
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

    def _create_project(self, overwrite):
        project_name = self.name
        project_dir = self.projectDir

        if project_name != '':
            # check if folder already exist
            e = self._check_if_path_exists(project_dir, project_name, overwrite)

            proposed_path = Path(project_dir)
            if proposed_path.name != project_name:
                proposed_path = proposed_path / project_name

            if not e:
                # Create project directory (flat — cavity objects go directly inside)
                try:
                    os.makedirs(str(proposed_path), exist_ok=True)
                    self.projectDir = str(proposed_path)
                    return True, 1
                except Exception as e:
                    self.projectDir = str(proposed_path)
                    error("An exception occurred in created project: ", e)
                    return False, 0
            else:
                self.projectDir = str(proposed_path)
                return True, 2
        else:
            error('\tPlease enter a valid project name')
            proposed_path = Path(project_dir)
            if proposed_path.name != project_name:
                proposed_path = proposed_path / project_name
            self.projectDir = str(proposed_path)
            return False, 0

    @staticmethod
    def _check_if_path_exists(directory, folder, overwrite):
        path = f"{directory}/{folder}"
        if os.path.exists(path):
            x = 'n'
            if overwrite:
                x = 'y'

            if x == 'y':
                try:
                    directory_list = os.listdir(path)

                    # Flat structure: just check it's a project folder
                    if len(directory_list) < 20:
                        shutil.rmtree(path)
                        return False
                    else:
                        error('\tIt seems that the folder specified is not a cavity project folder. Please check folder'
                              'again to avoid deleting important files.')
                        return True

                except Exception as e:
                    error("Exception occurred: ", e)
                    return True
            else:
                return True
        else:
            return False

    def sweep(self, sweep_config):
        self.sweep_results = {}
        for cav in self.cavities_list:
            cav.sweep(sweep_config)
            self.sweep_results[cav.name] = cav.sweep_results

    # NOTE: Cavity.sweep currently raises NotImplementedError — the parameter
    # sweep was not carried through the geometry refactor (see the method for
    # the manual workaround). This wrapper is kept so the API shape is stable
    # once sweep is ported.

    def plot_dispersion(self, pol=None, bands=None, ax=None, break_axis=True,
                        breaks=None, light_line=True, **kwargs):
        """Overlay each cavity's dispersion diagram. ``pol`` selects the
        polarisation(s): ``None`` (default) shows every computed polarisation,
        or name one ('monopole', 'dipole', ...) or a list. ``bands`` selects
        passbands; ``break_axis`` splits the y-axis where passbands are far apart,
        ``breaks`` places those splits by hand, and ``light_line`` overlays the
        speed-of-light line (see :meth:`Cavity.plot_dispersion`)."""
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
        return Cavity._render_dispersion(series, n, ax=ax, break_axis=break_axis,
                                         breaks=breaks, light_lines=light_lines or None,
                                         **kwargs)

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

            The legacy ``'parameters'``/``'cell_types'`` pair is still
            accepted (with a deprecation warning).

        Recognised keys
        ---------------
        freqs (alias ``freq``) : float | list
            Target frequency in MHz (a scalar for all cavities, or one per cavity).
        cell_type : dict
            ``{cell type: tune variable(s)}`` as above. A tune variable is any
            name the model exposes via ``cav.tune_variables()`` (e.g. ``'Req'``,
            ``'L'`` for elliptical, ``'R6'`` for the gun, ``'p3_r'`` for a spline).
        processes : int
            Number of parallel worker processes (default 1).
        rerun : bool
            Recompute even if a tuned result exists (default True).
        tolerance (alias ``tol``) : float
            Convergence tolerance on the frequency.
        maxiter : int
            Maximum secant iterations per stage.
        eigenmode_config : dict
            Eigenmode sub-config used for each tuning solve (see
            :meth:`run_eigenmode` keys).
        uq_config : dict
            Optional UQ-averaged tuning (see :meth:`run_uq` keys).

        Returns
        -------

        """

        if tune_config is None and not kwargs:
            tune_config = {
                'cell_type': {'mid-cell': 'Req'},
                'freqs': [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self],
            }
            info(f'Tune variable and frequency not entered, defaulting to {json.dumps(tune_config, indent=4)}')
        else:
            # Config keys may also be passed as kwargs; kwargs override the dict.
            tune_config = {**(tune_config or {}), **kwargs}

        validate_tune_config(tune_config)

        if 'freqs' not in tune_config.keys():
            tune_config['freqs'] = [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self]
            info(f'Target frequency not entered, defaulting to {tune_config["freqs"]}')

        if ('cell_type' not in tune_config.keys()
                and 'parameters' not in tune_config.keys()
                and 'cell_types' not in tune_config.keys()):
            tune_config['cell_type'] = {'mid-cell': 'Req'}
            info('"cell_type" not entered, defaulting to {"mid-cell": "Req"}')

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

        rerun = True
        if 'rerun' in tune_config:
            rerun = tune_config['rerun']
            if not isinstance(rerun, bool):
                raise ValueError("tune_config['rerun'] must be a boolean.")
        else:
            tune_config['rerun'] = rerun

        if rerun:
            run_tune_parallel(self.cavities_dict, tune_config, solver='NGSolveMEVP', resume=False)

        self.tune_config = tune_config
        # get tune results (also attaches cav.tuned lazily from disk)
        self.get_tune_res()

        # Automatically run eigenmode simulation on tuned cavities if config provided
        if 'eigenmode_config' in tune_config:
            tuned_view = self.tuned
            if tuned_view:
                info(f"Automatically running eigenmode simulation for tuned cavities...")
                ec = tune_config['eigenmode_config']

                # Adjust n_cells if specified in config
                if 'n_cells' in ec:
                    for cav in tuned_view.cavities_list:
                        cav.set_n_cells(ec['n_cells'])

                tuned_view.run_eigenmode(ec)

    def get_tune_res(self):
        """Reload tune results from disk and attach tuned cavities.

        After ``run_tune`` the subprocesses have written
        ``<cav>/tuned/tune_info/tune_res.json`` and the tuned geometry.
        The parent‐process cavity objects still have no ``_tuned_cavity``,
        so we force the ``cav.tuned`` lazy loader to rebuild them from disk.
        """
        for cav in self.cavities_dict.values():
            cav._tuned_cavity = None  # force reload via property
            tuned = cav.tuned
            if tuned is not None:
                cav.tune_results = dict(tuned.tune_results)
                cav.freq = tuned.freq
                self.tune_results[cav.name] = dict(tuned.tune_results)
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
            Recompute even if results already exist (default True).
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

        if eigenmode_config is None:
            eigenmode_config = {}
        # Config keys may also be passed as kwargs; kwargs override the dict.
        eigenmode_config = {**eigenmode_config, **kwargs}

        validate_eigenmode_config(eigenmode_config)

        # Merge over the complete defaults so the saved config records every
        # setting the run used (same pattern as EigenmodeSolver.run).
        eigenmode_config = merge_config(DEFAULT_EIGENMODE_CONFIG, eigenmode_config)

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

        # Save eigenmode config for traceability
        for cav in self.cavities_list:
            ec_path = Path(cav.self_dir) / 'eigenmode' / 'eigenmode_config.json'
            # Ensure folder exists (might be empty if analysis failed but we still want the log)
            os.makedirs(ec_path.parent, exist_ok=True)
            with open(ec_path, 'w') as f:
                json.dump(eigenmode_config, f, indent=4, default=str)

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
                                "sigma_SR [mm]": 4.32,  # <- Bunch length
                                "sigma_BS [mm]": 15.2,  # <- Bunch length
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
            Recompute even if results exist (default True).
        MT, NFS : int
            Number of mesh lines / frequency samples for the solver.
        DDR_SIG, DDZ_SIG : float
            Radial / longitudinal mesh density (mesh lines per sigma).
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

        if wakefield_config is None:
            wakefield_config = {}
        # Config keys may also be passed as kwargs; kwargs override the dict.
        wakefield_config = {**wakefield_config, **kwargs}

        validate_wakefield_config(wakefield_config)

        # Merge over the complete defaults so the saved config records every
        # setting the run used (same pattern as WakefieldSolver.run).
        wakefield_config = merge_config(DEFAULT_WAKEFIELD_CONFIG, wakefield_config)

        wakefield_config['target'] = run_wakefield_s

        wakefield_config_keys = wakefield_config.keys()

        if 'operating_points' in wakefield_config_keys:
            self.operating_points = wakefield_config['operating_points']
            for cav in self.cavities_list:
                cav.operating_points = self.operating_points

        rerun = True
        if 'rerun' in wakefield_config_keys:
            if isinstance(wakefield_config['rerun'], bool):
                rerun = wakefield_config['rerun']

        uq_config = {}
        # Truthy check (not just presence): the complete merged config carries
        # 'uq_config' with explicit None when UQ is disabled.
        if wakefield_config.get('uq_config'):



            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

            require('objectives' in wakefield_config['uq_config'].keys(), 'Please enter objectives in uq_config.')

            # Keep the FULL objective form (goal, name, [args]). process_objectives
            # and the UQ wakefield reader both key on obj[1]=name and obj[2]=args
            # (e.g. a ZL/ZT objective's frequency windows), exactly as the
            # optimiser's objectives_unprocessed does. The old code truncated to
            # ['', goal, name] — dropping the frequency range and shifting the QOI
            # name into the goal slot, so every ZL/ZT objective came back empty.
            objectives_unprocessed = list(wakefield_config['uq_config']['objectives'])
            objectives, weights = process_objectives(objectives_unprocessed)
            wakefield_config['uq_config']['objectives_unprocessed'] = objectives_unprocessed
            wakefield_config['uq_config']['objectives'] = objectives
            uq_config = wakefield_config['uq_config']
        else:
            wakefield_config['uq_config'] = uq_config

        if rerun:
            MROT = 2
            MT = 10
            NFS = 10000
            wakelength = 50
            bunch_length = 25
            DDR_SIG = 0.1
            DDZ_SIG = 0.1

            # `processes` was already type- and range-checked by
            # validate_wakefield_config above; only the default is needed here.
            wakefield_config.setdefault('processes', 1)
            processes = wakefield_config['processes']

            # MROT is the canonical key (0 longitudinal, 1 transverse, 2 both);
            # names and the deprecated 'polarisation' alias are also accepted.
            # Note: for wakefield this is the ABCI beam mode, not the eigenmode
            # azimuthal order that 'polarisation' means in eigenmode_config.
            if 'polarisation' in wakefield_config_keys and 'MROT' not in wakefield_config_keys:
                info("wakefield_config['polarisation'] is deprecated; use 'MROT' "
                     "(0=longitudinal, 1=transverse, 2=both).")
            resolved_mrot = resolve_mrot(wakefield_config, default=MROT)
            wakefield_config['MROT'] = resolved_mrot
            wakefield_config['polarisation'] = resolved_mrot  # back-compat for downstream reads

            # MT / NFS were already type-checked by validate_wakefield_config above.
            wakefield_config.setdefault('MT', MT)
            wakefield_config.setdefault('NFS', NFS)

            # check input configs

            # beam config
            wake_config = {}
            beam_config = {}
            mesh_config = {}
            if 'beam_config' not in wakefield_config_keys:
                wakefield_config['beam_config'] = beam_config
            if 'wake_config' not in wakefield_config_keys:
                wakefield_config['wake_config'] = wake_config
            if 'mesh_config' not in wakefield_config_keys:
                wakefield_config['mesh_config'] = mesh_config

            if 'bunch_length' in beam_config.keys():
                require(not isinstance(wakefield_config['beam_config']['bunch_length'], str),
                        'Bunch length must be of type integer or float.')
            else:
                beam_config['bunch_length'] = bunch_length

            # wake config
            if 'wakelength' in wake_config.keys():
                require(not isinstance(wake_config['wakelength'], str), 'Wakelength must be of type integer or float.')
            else:
                wake_config['wakelength'] = wakelength

            if 'DDR_SIG' not in mesh_config.keys():
                mesh_config['DDR_SIG'] = DDR_SIG

            if 'DDZ_SIG' not in mesh_config.keys():
                mesh_config['DDZ_SIG'] = DDZ_SIG
                
            run_wakefield_parallel(self.cavities_dict, wakefield_config)

        self.wakefield_config = wakefield_config
        self.get_wakefield_qois(uq_config)

    def get_wakefield_qois(self, uq_config):
        for key, cav in self.cavities_dict.items():
            # try:
            # main run loss/kick factors — always reported now
            self.wakefield_qois[cav.name] = cav.wakefield.qois
            # operating-point QOIs (P_HOM, ...) — only when op-points were given
            cav.get_wakefield_qois(self.wakefield_config)
            self.wakefield_qois_op[cav.name] = cav.wakefield_qois
            if uq_config:
                # Wakefield UQ writes to the cavity's own uq/ dir (the same file
                # the eigenmode branch and the optimiser use), not the old
                # pre-refactor <projectDir>/wakefield/ABCI/<name>/ layout.
                cav.get_uq_hom_results(os.path.join(cav.uq_dir, "uq.json"))
                cav_uq_hom_results = cav.uq_hom_results
                if 'operating_points' in uq_config:
                    # separate into opearating points
                    op_points = uq_config['operating_points']
                    uq_hom_results_op = {}
                    for op, val in op_points.items():
                        uq_hom_results_op[op] = {}
                        for k, v in val.items():
                            if 'sigma' in k:
                                sig_id = k.split('_')[-1].split(' ')[0]
                                ident = fr'_{op}_{sig_id}_{v}mm'
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
        no_of_generations (alias ``no_of_generation``) : int
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
        self.optimisation.run(optimisation_config, resume=resume)

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

    def config_sample(self, kind):
        return self.cavities_list[0].config_sample(kind)

    def set_cavities_field(self):
        """
        Sets cavities analysis field range.

        Returns
        -------

        """
        for cav in self.cavities_list:
            cav.set_Eacc(Eacc=self.E_acc)

    def compare_power(self, E_acc=None):
        if E_acc is not None:
            self.E_acc = E_acc
            self.set_cavities_field()

        self.p_qois = []
        results = []
        for i, cav in enumerate(self.cavities_list):
            # E_acc_Pin(cavity, op_field[i], ls[i], fig, ax, ax_right, ax_right2)
            results.append(self.qois(cav, cav.op_field * 1e-6, E_acc))

        self.returned_results = results

    def qois_fm(self):
        """
        Retrieves the fundamental mode quantities of interest

        Returns
        -------
        Dictionary containing fundamental mode quantities of interest (normed optional).
        """
        results = []
        for cav in self.cavities_list:
            results.append({
                r"freq [MHz]": cav.freq,
                r"Epk/Eacc []": cav.e,
                r"Bpk/Eacc [mT/MV/m]": cav.b,
                r"kcc [%]": cav.k_cc,
                r"R/Q [Ohm]": cav.R_Q,
                r"G [Ohm]": cav.G,
                r"GR/Q [Ohm^2]": cav.GR_Q
            })

        results_norm_units = []
        for cav in self.cavities_list:
            results_norm_units.append({
                r"$e_\mathrm{pk}$": cav.e,
                r"$b_\mathrm{pk}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$r/q$": cav.R_Q,
                r"$g$": cav.G,
                r"$g\cdot r/q $": cav.GR_Q
            })

        return results

    def summary(self):
        """Fundamental-mode QOIs for all cavities as a DataFrame, indexed by
        cavity name. In Jupyter it renders as an HTML table; elsewhere use
        ``.to_string()`` / ``.to_markdown()``. Requires eigenmode results
        (run ``run_eigenmode`` first)."""
        names = [cav.name for cav in self.cavities_list]
        return pd.DataFrame(self.qois_fm(), index=names)

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
                r"$E_\mathrm{pk}/E_\mathrm{acc} ~[\cdot]$": cav.e,
                r"$B_\mathrm{pk}/E_\mathrm{acc} ~\mathrm{[mT/MV/m]}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$R/Q~ \mathrm{[\Omega]}$": cav.R_Q,
                r"$G ~\mathrm{[\Omega]}$": cav.G,
                r"$G\cdot R/Q ~\mathrm{[10^4\Omega^2]}$": cav.GR_Q * 1e-4,
                r"$|k_\parallel| ~\mathrm{[V/pC]}$": cav.k_loss[opt],
                r"$|k_\perp| ~\mathrm{[V/pC/m]}$": cav.k_kick[opt],
                r"$P_\mathrm{HOM}/cav~ \mathrm{[kW]}$": cav.phom[opt]
            })

        results_norm_units = []
        for cav in self.cavities_list:
            results_norm_units.append({
                r"$e_\mathrm{pk}/e_\mathrm{acc}$": cav.e,
                r"$b_\mathrm{pk}/e_\mathrm{acc}$": cav.b,
                r"$k_\mathrm{cc}$": cav.k_cc,
                r"$r/q$": cav.R_Q,
                r"$g$": cav.G,
                # r"$g\cdot r/q $": cav.GR_Q,
                # r"$|k_\mathrm{FM}|$": cav.k_fm,
                r"$|k_\parallel|$": cav.k_loss[opt],
                r"$k_\perp$": cav.k_kick[opt],
                r"$p_\mathrm{HOM}/cav$": cav.phom[opt]
            })

        return results

    def plot_uq_geometries(self):
        fig, axd = plt.subplot_mosaic([[cav.name for cav in self.cavities_list]], layout='constrained')

        for cav, ax in zip(self.cavities_list, axd.values()):
            # plot nominal
            cav.plot('geometry', ax=ax, mid_cell=True, zorder=10, show=False)
            directory = os.path.join(cav.self_dir, 'eigenmode')
            tag = f'{cav.name}_Q'
            uq_geom_folders = self.find_folders_with_tag(directory, tag)

            # load uq geometries
            for uq_geom_folder in uq_geom_folders:
                if os.path.exists(f'{uq_geom_folder}/monopole/geodata.n'):
                    # read geometry
                    cav_geom = pd.read_csv(f'{uq_geom_folder}/monopole/geodata.n', header=None,
                                           skipfooter=1, sep='\\s+', engine='python')

                    cav_geom = cav_geom[[1, 0]]
                    ax.plot(cav_geom[1], cav_geom[0], ls='--', lw=1, c='gray')
            ax.set_title(cav.name)
        return ax

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

    def plot_power_comparison(self, fig=None, ax_list=None):
        """
        Can be called using ``cavities.plot_power_comparison()``

        Parameters
        ----------
        fig: matplotlib figure
        ax_list: list of matplotlib axes object

        Returns
        -------

        """
        if fig is not None:
            fig = fig
            ax1, ax2, ax3 = ax_list
        else:
            # create figure
            fig = plt.figure()
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[:, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 1])
        # def E_acc_Pin(self, cavity, E_acc, op_field, ls='-', ):

        # ax_right2._get_lines.prop_cycler = ax._get_lines.prop_cycler
        # ax_right2.spines["right"].set_position(("axes", 1.2))
        for i, cavity in enumerate(self.cavities_list):
            ax1.plot(cavity.E_acc * 1e-6, cavity.pstat / cavity.n_cav,
                     ls=self.ls[i], lw=2, c='tab:orange',
                     label=r"$P_\mathrm{static/cav}$" + fr"{cavity.name}")

            ax1.plot(cavity.E_acc * 1e-6, cavity.pdyn / cavity.n_cav,
                     ls=self.ls[i], lw=2, c='tab:blue', label=r"$P_\mathrm{dynamic/cav}$" + fr"{cavity.name}")

            # p1, = ax1.plot(cavity.E_acc * 1e-6, cavity.p_wp/cavity.n_cav,
            #                ls=self.ls[i], lw=2, c='k', label=r"$P_\mathrm{wp/beam}$" + fr"{cavity.name}")

            p2, = ax2.plot(cavity.E_acc * 1e-6, cavity.n_cav, ls=self.ls[i], lw=2, c='tab:red',
                           label=fr"{cavity.name}")

            p3, = ax3.plot(cavity.E_acc * 1e-6, cavity.p_in * 1e-3, ls=self.ls[i], lw=2, c='tab:purple',
                           label=fr"{cavity.name}")

            ax1.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax1.set_ylabel(r"$P_\mathrm{stat, dyn}$/cav [W]")
            ax2.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax2.set_ylabel(r"$N_\mathrm{cav/beam}$")
            ax3.set_xlabel(r"$E_\mathrm{acc}$ [MV/m]")
            ax3.set_ylabel(r"$P_\mathrm{in/cav}$ [kW]")

            ax1.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax1.text(cavity.op_field * 1e-6 - 1, 0.3, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90, transform=ax1.get_xaxis_transform())
            ax2.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax2.text(cavity.op_field * 1e-6 - 1, 0.5, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90, transform=ax2.get_xaxis_transform())
            ax3.axvline(cavity.op_field * 1e-6, ls=':', c='k')
            ax3.text(cavity.op_field * 1e-6 - 1, 0.3, f"{cavity.op_field * 1e-6} MV/m",
                     size=14, rotation=90,
                     transform=ax3.get_xaxis_transform())

            # ax.axvline(7.13, ls='--', c='k')
            # ax.axvline(10, ls='--', c='k')
            # ax.axvline(15, ls='--', c='k')
            # ax_right2.axhline(500, ls='--', c='k')
            # ax_right2.axhline(1000, ls='--', c='k')
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            # ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
            # ax_right.yaxis.set_major_locator(ticker.MultipleLocator(100))
            # ax_right2.yaxis.set_major_locator(ticker.MultipleLocator(200))

            # ax.yaxis.label.set_color(p1.get_color())
            # ax_right.yaxis.label.set_color(p2.get_color())
            # ax_right2.yaxis.label.set_color(p3.get_color())

            ax1.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            ax2.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            ax3.set_xlim(min(cavity.E_acc) * 1e-6, max(cavity.E_acc) * 1e-6)
            # # ax.set_ylim(0, 50)
            # ax_right.set_ylim(100, 400)f
            # ax_right2.set_ylim(0, 700)
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax3.set_yscale('log')

            # tkw = dict(size=4, width=1.5)
            # ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
            # ax_right.tick_params(axis='y', colors=p2.get_color(), **tkw)
            # ax_right2.tick_params(axis='y', colors=p3.get_color(), **tkw)
            # ax.tick_params(axis='x', **tkw)

            ax1.minorticks_on()
            # ax.grid(True, which='both', axis='both')

        # dummy lines with NO entries, just to create the black style legend
        dummy_lines = []
        for b_idx, b in enumerate(self.cavities_list):
            dummy_lines.append(ax1.plot([], [], c="gray", ls=self.ls[b_idx])[0])

        lines = ax1.get_lines()
        legend1 = ax1.legend([lines[i] for i in range(3)],
                             [r"$P_\mathrm{stat}$", r"$P_\mathrm{dyn}$"], loc=3)
        legend2 = ax1.legend([dummy_lines[i] for i in range(len(self.cavities_list))],
                             [cavity.name for cavity in self.cavities_list],
                             loc=0)
        ax1.add_artist(legend1)

        # ax1.legend(ncol=len(cavities))
        ax2.legend(loc='upper left')
        ax3.legend(loc=3)

        label = [r"$\mathbf{Z^*}$", 'Z', r"$\mathbf{W^*}$", 'W']
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_comparison.png")

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

    def _power_bar_impl(self):
        """
        Plots bar chart of power quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (12, 3)
        # plot barchart
        data = np.array([list(d.values()) for d in self.returned_results])
        data_col_max = data.max(axis=0)

        x = list(self.returned_results[0].keys())
        X = np.arange(len(x))

        fig, ax = plt.subplots()
        ax.margins(x=0)
        width = 0.15  # 1 / len(x)
        for i, cav in enumerate(self.cavities_list):
            ax.bar(X + i * width, data[i] / data_col_max, width=width, label=cav.name)

        ax.set_xticks([r + width for r in range(len(x))], x)
        # label = ["C3794_H (2-Cell)", "C3795_H (5-Cell)"]

        ax.axhline(1.05, c='k')
        ax.set_ylim(-0.01, 1.5 * ax.get_ylim()[-1])
        ax.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_power_comparison_bar.png")
        return ax

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

        # display formula used
        display(Math(r'Dims: {}x{}m \\ Area: {}m^2 \\ Volume: {}m^3'.format(2, round(3, 2), 5, 5)))

        for ii, cav in enumerate(self.cavities_list):
            # set cavity intrinsic quality factor and inv_eta
            cav.Q0 = rf_config['Q0 []'][ii]
            cav.inv_eta = rf_config['inv_eta []'][ii]
            if 'Eacc [MV/m]' in rf_config.keys():
                cav.Eacc_rf_config = rf_config['Eacc [MV/m]'][ii]  # change this later, not effective way

            self.power_qois[cav.name] = self.get_power_qois(cav, rf_config, op_points_list)
            self.power_qois_uq[cav.name] = self.get_power_qois(cav, rf_config, op_points_list, uq=True)

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
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
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
                    # for metric, values in metrics[op_pt]['SR'].items():
                    dd_nominal[cav] = metrics[op_pt]['SR']

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.power_qois_uq.items():
                    for metric, values in metrics[op_pt]['SR'].items():
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

    # def plot_compare_hom_bar_(self, opt, ncols=3, uq=False):
    #     """
    #     Plot bar chart of higher-order mode's quantities of interest
    #
    #     Returns
    #     -------
    #
    #     """
    #     # plt.rcParams["figure.figsize"] = (15 / 27 * 6 * len(self.cavities_list), 3)
    #     plt.rcParams["figure.figsize"] = (12, 4)
    #
    #     if not uq:
    #         # plot barchart
    #         self.hom_results = self.qois_hom(opt)
    #         df = pd.DataFrame.from_dict(self.hom_results)
    #         fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')
    #         # Plot each column in a separate subplot
    #         labels = [cav.plot_label for cav in self.cavities_list]
    #         for key, ax in axd.items():
    #             ax.bar(df.index, df[key], label=labels, color=matplotlib.colormaps['Set2'].colors[:len(df)],
    #                    edgecolor='k', width=1)
    #             ax.set_xticklabels([])
    #             ax.set_ylabel(key)
    #             h, l = ax.get_legend_handles_labels()
    #     else:
    #         # Step 1: Flatten the dictionary into a DataFrame
    #         rows = []
    #         for cav, metrics in self.uq_hom_results.items():
    #             for metric, values in metrics.items():
    #                 rows.append({
    #                     'cavity': cav,
    #                     'metric': metric,
    #                     'mean': values['expe'][0],
    #                     'std': values['stdDev'][0]
    #                 })
    #
    #         df = pd.DataFrame(rows)
    #
    #         labels = [cav.plot_label for cav in self.cavities_list]
    #
    #         # Step 2: Create a Mosaic Plot
    #         metrics = df['metric'].unique()
    #         num_metrics = len(metrics)
    #
    #         layout = [[metric for metric in metrics]]
    #         fig, axd = plt.subplot_mosaic(layout, layout='constrained')
    #
    #         # Plot each metric on a separate subplot
    #         for metric, ax in axd.items():
    #             sub_df = df[df['metric'] == metric]
    #             ax.bar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], label=labels, capsize=5,
    #                    color=matplotlib.colormaps['Set2'].colors[:len(df)], edgecolor='k',
    #                    width=1)
    #
    #             ax.set_xticklabels([])
    #             ax.set_xticks([])
    #             ax.set_ylabel(metric)
    #             h, l = ax.get_legend_handles_labels()
    #
    #     if not ncols:
    #         ncols = min(4, len(self.cavities_list))
    #     fig.legend(h, l, loc='outside upper center', borderaxespad=0, ncol=ncols)
    #
    #     # fig.set_tight_layout(True)
    #
    #     # save plots
    #     fname = [cav.name for cav in self.cavities_list]
    #     fname = '_'.join(fname)
    #
    #     self.save_all_plots(f"{fname}_hom_bar.png")
    #
    #     return axd

    def _hom_bar_impl(self, op_points_list, ncols=3, uq=False, figsize=(12, 3), qois=None):
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
        if qois is None:
            qois = ['k_loss', 'k_kick', 'p_hom']

        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

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
                    ax.bar(df.index, df[key], color=colors[i], ec='k', label=label)
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
                        if fr'{opt}_SR' in kk:
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in metrics[opt]['SR'].items():
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

    def _hom_scatter_impl(self, op_points_list, ncols=3, uq=False, figsize=(12, 3), qois=None):
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
        if qois is None:
            qois = ['k_loss', 'k_kick', 'p_hom']

        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

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
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
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
                        if fr'{opt}_SR' in kk:
                            dd_nominal[cav] = vv

                df_nominal = pd.DataFrame.from_dict(dd_nominal).T
                df_nominal_list.append(df_nominal)

                rows = []
                # get uq opt
                for cavity, metrics in self.uq_hom_results.items():
                    for metric, values in metrics[opt]['SR'].items():
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
            self.fm_results = self.qois_fm()

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
            selected_qois = ['freq [MHz]', 'Epk/Eacc []', 'Bpk/Eacc [mT/MV/m]', 'R/Q [Ohm]']
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
            self.fm_results = self.qois_fm()
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
            metrics = [m for q in selected_qois for m in available if _bare(m) == q]
            if not metrics:
                raise ValueError("None of the specified qois are present in the UQ fundamental mode results.")

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

                    # plot nominal
                    ax.scatter(df_nominal.index, df_nominal[_bare(metric)], facecolor='none',
                               label='Design Point', ec='k', lw=1, s=75,
                               zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                ax.set_xlabel(LABELS[_bare(metric)])

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

    def _all_scatter_impl(self, opt, ncols=3):
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
                    ax.scatter(df.index, df[key], color=colors[i], ec='k', label=label)
                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(key)

            h, l = ax.get_legend_handles_labels()
        else:
            # get nominal qois
            dd_nominal = {}
            for cav, ops_id in self.wakefield_qois_op.items():
                for kk, vv in ops_id.items():
                    if fr'{opt}_SR' in kk:
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
                for metric, values in metrics[opt]['SR'].items():
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
    def plot_all_scatter(self, opt, ncols=3, show=True):
        """Scatter of fundamental-mode + HOM/power QOIs across every cavity."""
        r = self._all_scatter_impl(opt, ncols=ncols)
        _maybe_show(show)
        return r

    def plot_all_bar(self, opt, ncols=3, show=True):
        """Bar chart of fundamental-mode + HOM/power QOIs across every cavity."""
        r = self._all_bar_impl(opt, ncols=ncols)
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

    def plot_surface_fields(self):
        """
        Plot surface fields of cavities

        Returns
        -------

        """
        for cav in self.cavities_list:
            # normalize fields
            e_surf = np.abs(cav.surface_field['0'])
            e_surf_norm = e_surf / e_surf.max()

            plt.plot(e_surf_norm, label=cav.name)

        plt.axhline(1.02, c='k')
        plt.ylim(-0.01, 1.5)
        plt.xlabel(r'$L_\mathrm{surf}$ [mm]')
        plt.ylabel(r'$|E_\mathrm{surf}|/|E_\mathrm{surf}|_\mathrm{max}$')
        plt.legend(loc='upper center', ncol=len(self.cavities_list))
        plt.tight_layout()

        # save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_surface_fields.png")

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
        Epk_Eacc_list = [cav.e for cav in self.cavities_list]
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
            q = 1.6021773e-19
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

    # def plot_dispersion(self):
    #     """
    #     Plot dispersion curve for the cavities
    #
    #     Returns
    #     -------
    #
    #     """
    #     fig, ax = plt.subplots()
    #     ax.margins(x=0)
    #     for cav in self.cavities_list:
    #         x = range(1, cav.n_cells + 1)
    #         ax.plot(x, cav.d_slans_all_results['FREQUENCY'][0:cav.n_cells], marker='o', mec='k',
    #                 label=f'{cav.name} (kcc={round(cav.k_cc, 2)} %)')
    #         ax.set_xlabel('Mode Number')
    #         ax.set_ylabel('Frequency [MHz]')
    #
    #     plt.legend()
    #
    #     # save plots
    #     fname = [cav.name for cav in self.cavities_list]
    #     fname = '_'.join(fname)
    #
    #     self.save_all_plots(f"{fname}_dispersion.png")
    #
    #     plt.show()
    #
    # # def ql_pin(self, labels, geometry, RF, QOI, Machine, p_data=None):
    # #     """
    # #     Calculate the value of input power as a function of loaded quality factor
    # #
    # #     Parameters
    # #     ----------
    # #     labels: list, array like
    # #         Descriptive labels on matplotlib plot
    # #     geometry: list, array like
    # #         List of grouped geometric input parameters
    # #     RF: list, array like
    # #         List of grouped radio-frequency (RF) properties
    # #     QOI:
    # #         List of quantities of interest for cavities
    # #     Machine:
    # #         List of grouped machine related materials
    # #     p_data:
    # #
    # #
    # #     Returns
    # #     -------
    # #
    # #     """
    # #     # check if entries are of same length
    # #
    # #     it = iter(geometry)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(RF)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(QOI)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     it = iter(Machine)
    # #     the_len = len(next(it))
    # #     if not all(len(l) == the_len for l in it):
    # #         raise ValueError('not all lists have same length!')
    # #
    # #     n_cells, l_cells, G, b = [np.array(x) for x in geometry]
    # #     E_acc, Vrf = [np.array(x) for x in RF]
    # #
    # #     fig, ax = plt.subplots()
    # #     ax.margins(x=0)
    # #
    # #     # QOI
    # #     f0, R_Q = [np.array(x) for x in QOI]
    # #
    # #     # Machine
    # #     I0, rho, E0 = [np.array(x) for x in Machine]
    # #
    # #     l_active = 2 * n_cells * l_cells
    # #     l_cavity = l_active + 8 * l_cells
    # #
    # #     # CALCULATED
    # #     v_cav = E_acc * l_active
    # #
    # #     U_loss = 88.46 * E0 ** 4 / rho * 1e-6  # GeV # energy lost per turn per beam
    # #     v_loss = U_loss * 1e9  # V # v loss per beam
    # #
    # #     phi = np.arccos(v_loss / Vrf)
    # #     delta_f = -R_Q * f0 * I0 * np.sin(phi) / (2 * v_cav)  # optimal df
    # #     QL_0_x = v_cav / (R_Q * I0 * np.cos(phi))  # optimal Q loaded
    # #
    # #     QL_0 = np.linspace(1e4, 1e9, 1000000)
    # #
    # #     xy_list = [(0.15, 0.13), (0.1, 0.16), (0.1, 0.19), (0.1, 0.21)]
    # #     for i in range(len(E_acc)):
    # #         f1_2 = f0[i] / (2 * QL_0)  # 380.6
    #
    # #         pin = v_cav[i] ** 2 / (4 * R_Q[i] * QL_0) * \
    # #               ((1 + ((R_Q[i] * QL_0 * I0[i]) / v_cav[i]) * np.cos(phi[i])) ** 2 +
    # #                ((delta_f[i] / f1_2) + ((R_Q[i] * QL_0 * I0[i]) / v_cav[i]) * np.sin(phi[i])) ** 2)
    # #
    # #         # material/ wall power
    # #         e_acc = np.linspace(0.5, 25, 1000) * 1e6  # MV/m
    # #
    # #         txt = labels[i]
    # #
    # #         if "*" in labels[i]:
    # #             l = ax.plot(QL_0, pin * 1e-3, label=txt, lw=4,
    # #                         ls='--')
    # #         else:
    # #             l = ax.plot(QL_0, pin * 1e-3, label=txt, lw=4)
    # #
    # #         # add annotations
    # #
    # #         # annotext = ax.annotate(txt, xy=xy_list[i], xycoords='figure fraction', size=8, rotation=0,
    # #         #                        c=l[0].get_color())
    # #
    # #     if p_data:
    # #         # plot QL with penetration
    # #         ax_2 = ax.twinx()
    # #         data = fr.excel_reader(p_data)
    # #         data_ = data[list(data.keys())[0]]
    # #         ax_2.plot(data_["QL"], data_["penetration"], lw=4)
    # #
    # #     # plot decorations
    # #     ax.set_xlabel(r"$Q_{L,0}$")
    # #     ax.set_ylabel(r"$P_\mathrm{in} ~[\mathrm{kW}]$")
    # #     ax.set_xscale('log')
    # #     ax.set_xlim(5e3, 1e9)
    # #     ax.set_ylim(0, 3000)
    # #     ax.legend(loc='upper left')  #
    # #     ax.minorticks_on()
    # #     # ax.grid(which='both')
    # #     fig.show()

    def define_operating_points(self, ops):
        self.operating_points = ops

    @staticmethod
    def linspace(start, stop, step=1.):
        """
        Like np.linspace but uses step instead of num
        This is inclusive to stop, so if start=1, stop=3, step=0.5
        Output is: array([1., 1.5, 2., 2.5, 3.])
        """
        if start < stop:
            ll = np.linspace(start, stop, int((stop - start) / abs(step) + 1))
            if stop not in ll:
                ll = np.append(ll, stop)

            return ll
        else:
            ll = np.linspace(stop, start, int((start - stop) / abs(step) + 1))
            if start not in ll:
                ll = np.append(ll, start)
            return ll

    @staticmethod
    def lineTo(prevPt, nextPt, step):
        if prevPt[0] == nextPt[0]:
            # vertical line
            # chwxk id nextPt is greater
            if prevPt[1] < nextPt[1]:
                py = np.linspace(prevPt[1], nextPt[1], step)
            else:
                py = np.linspace(nextPt[1], prevPt[1], step)
                py = py[::-1]
            px = np.ones(len(py)) * prevPt[0]

        elif prevPt[1] == nextPt[1]:
            # horizontal line
            if prevPt[0] < nextPt[1]:
                px = np.linspace(prevPt[0], nextPt[0], step)
            else:
                px = np.linspace(nextPt[0], prevPt[0], step)

            py = np.ones(len(px)) * prevPt[1]
        else:
            # calculate angle to get appropriate step size for x and y
            ang = np.arctan((nextPt[1] - prevPt[1]) / (nextPt[0] - prevPt[0]))
            if prevPt[0] < nextPt[0] and prevPt[1] < nextPt[1]:
                px = np.arange(prevPt[0], nextPt[0], step * np.cos(ang))
                py = np.arange(prevPt[1], nextPt[1], step * np.sin(ang))
            elif prevPt[0] > nextPt[0] and prevPt[1] < nextPt[1]:
                px = np.arange(nextPt[0], prevPt[0], step * np.cos(ang))
                px = px[::-1]
                py = np.arange(prevPt[1], nextPt[1], step * np.sin(ang))
            elif prevPt[0] < nextPt[0] and prevPt[1] > nextPt[1]:
                px = np.arange(prevPt[0], nextPt[0], step * np.cos(ang))
                py = np.arange(nextPt[1], prevPt[1], step * np.sin(ang))
                py = py[::-1]
            else:
                px = np.arange(nextPt[0], prevPt[0], step * np.cos(ang))
                px = px[::-1]
                py = np.arange(nextPt[1], prevPt[1], step * np.sin(ang))
                py = py[::-1]

        # plt.plot(px, py)

    @staticmethod
    def arcTo2(x_center, y_center, a, b, step, start_angle, end_angle):
        u = x_center  # x-position of the center
        v = y_center  # y-position of the center
        a = a  # radius on the x-axis
        b = b  # radius on the y-axis
        sa = (start_angle / 360) * 2 * np.pi  # convert angle to radians
        ea = (end_angle / 360) * 2 * np.pi  # convert angle to radians

        if ea < sa:
            # end point of curve
            x_end, y_end = u + a * np.cos(sa), v + b * np.sin(sa)

            t = np.arange(ea, sa, np.pi / 100)
            # t = np.linspace(ea, sa, 100)
            # check if end angle is included, include if not
            if sa not in t:
                t = np.append(t, sa)
            t = t[::-1]
        else:
            # end point of curve
            x_end, y_end = u + a * np.cos(ea), v + b * np.sin(ea)

            t = np.arange(sa, ea, np.pi / 100)
            # t = np.linspace(ea, sa, 100)
            if ea not in t:
                t = np.append(t, ea)

        return [x_end, y_end]

    @staticmethod
    def arcTo(x_center, y_center, a, b, step, start, end):
        u = x_center  # x-position of the center
        v = y_center  # y-position of the center
        a = a  # radius on the x-axis
        b = b  # radius on the y-axis

        t = np.arange(0, 2 * np.pi, np.pi / 100)

        x = u + a * np.cos(t)
        y = v + b * np.sin(t)
        pts = np.column_stack((x, y))
        inidx = np.all(np.logical_and(np.array(start) < pts, pts < np.array(end)), axis=1)
        inbox = pts[inidx]
        inbox = inbox[inbox[:, 0].argsort()]

        # plt.plot(inbox[:, 0], inbox[:, 1])

        return inbox

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
                R_Q = r"$R/Q ~[\Omega$] " + "".join([fr"& {round(cav.R_Q, 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(cav.G, 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(cav.k_cc, 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(cav.e, 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for cav in self.cavities_list]) + r" \\"

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
                    [fr"& {round(cav.R_Q, 2)} " for cav in self.cavities_list]) + r" \\"
                G = r"$G$ ~[$\Omega$] " + "".join([fr"& {round(cav.G, 2)} " for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for cav in self.cavities_list]) + r" \\"
                kcc = r"$k_\mathrm{cc}~[\%]$ " + "".join(
                    [fr"& {round(cav.k_cc, 2)} " for cav in self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ []" + "".join(
                    [fr"& {round(cav.e, 2)} " for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} ~\left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for cav in self.cavities_list]) + r" \\"

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
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}~[]$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ ~[kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
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
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$~[]" + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} ~[\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Phom = r"$P_\mathrm{HOM}$/cav [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom, 4)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
                        for op_pt in op_pts_list for cav in
                        self.cavities_list]) + r" \\"

                PHOM = r"$P_\mathrm{HOM}$ [kW] " + "".join(
                    [
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
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
                    [fr"& {round(cav.R_Q, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                G = r"$G$ [$\Omega$] " + "".join(
                    [fr"& {round(cav.G, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                GR_Q = r"$G\cdot R/Q ~[10^4\Omega^2]$ " + "".join(
                    [fr"& {round(cav.GR_Q * 1e-4, 2)} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                epk = r"$E_{\mathrm{pk}}/E_{\mathrm{acc}}$ [] " + "".join(
                    [fr"& {round(cav.e, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"
                bpk = r"$B_{\mathrm{pk}}/E_{\mathrm{acc}} \left[\mathrm{\frac{mT}{MV/m}}\right]$ " + "".join(
                    [fr"& {round(cav.b, 2)} " for op_pt in op_pts_list for cav in self.cavities_list]) + r" \\"

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
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Ncav']} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Q0 = r"$Q_\mathrm{0}$ " + "".join(
                    [fr"& {cav.rf_performance_qois[op_pt]['SR']['Q0 []']:.2E} " for op_pt in op_pts_list for cav in
                     self.cavities_list]) + r" \\"
                Pin = r"$P_\mathrm{in}\mathrm{/cav} [\mathrm{kW}]$ " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pin/cav [kW]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pstat = r"$P_\mathrm{stat}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pstat/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pdyn = r"$P_\mathrm{dyn}$/cav [W] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pdyn/cav [W]'], 2)} " for op_pt in op_pts_list
                     for cav in self.cavities_list]) + r" \\"

                Pwp = r"$P_\mathrm{wp}$/cav [kW] " + "".join(
                    [fr"& {round(cav.rf_performance_qois[op_pt]['SR']['Pwp/cav [kW]'], 2)} " for op_pt in op_pts_list
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
                        fr"& {'/'.join([str(round(phom * cav.rf_performance_qois[op_pt]['SR']['Ncav'], 2)) for phom in [vv for kk, vv in cav.phom.items() if fr'{op_pt}' in kk]])} "
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
            error("Either NGSolve or ABCI results not available. Please use '<cav>.set_eigenmode_qois(<folder>)' "
                  "or '<cav>.set_wakefield_qois(<folder>)' to fix this. Error: ", e)

    def make_excel_summary(self):
        try:
            data = {'Name': [cav.name for cav in self.cavities_list],
                    'Project': [cav.project for cav in self.cavities_list],
                    'Type': [cav.type for cav in self.cavities_list],
                    'CW/Pulsed': [cav.cw_pulsed for cav in self.cavities_list],
                    'Material': [cav.material for cav in self.cavities_list],
                    'N_cells': [cav.n_cells for cav in self.cavities_list],
                    'Freq [MHz]': [cav.op_freq for cav in self.cavities_list],
                    'Beta': [cav.beta for cav in self.cavities_list],
                    'T_oper [K]': [cav.op_temp for cav in self.cavities_list],
                    'I0 [mA]': [cav.I0 for cav in self.cavities_list],
                    'sigma [mm]': [cav.sigma for cav in self.cavities_list],
                    'A_i [mm]': [round(cav.shape_space['IC'][0], 2) for cav in self.cavities_list],
                    'B_i [mm]': [round(cav.shape_space['IC'][1], 2) for cav in self.cavities_list],
                    'a_i [mm]': [round(cav.shape_space['IC'][2], 2) for cav in self.cavities_list],
                    'b_i [mm]': [round(cav.shape_space['IC'][3], 2) for cav in self.cavities_list],
                    'R_i [mm]': [round(cav.shape_space['IC'][4], 2) for cav in self.cavities_list],
                    'L_i [mm]': [round(cav.shape_space['IC'][5], 2) for cav in self.cavities_list],
                    'Req [mm]': [round(cav.shape_space['IC'][6], 2) for cav in self.cavities_list],
                    'alpha_i [deg]': [round(cav.shape_space['IC'][7], 2) for cav in self.cavities_list],
                    'A_el [mm]': [round(cav.shape_space['OC'][0], 2) for cav in self.cavities_list],
                    'B_el [mm]': [round(cav.shape_space['OC'][1], 2) for cav in self.cavities_list],
                    'a_el [mm]': [round(cav.shape_space['OC'][2], 2) for cav in self.cavities_list],
                    'b_el [mm]': [round(cav.shape_space['OC'][3], 2) for cav in self.cavities_list],
                    'R_el [mm]': [round(cav.shape_space['OC'][4], 2) for cav in self.cavities_list],
                    'L_el [mm]': [round(cav.shape_space['OC'][5], 2) for cav in self.cavities_list],
                    # 'Req [mm]': [round(cav.shape_space['OC'][6], 2) for cav in self.cavities_list],
                    'alpha__el [deg]': [round(cav.shape_space['OC'][7], 2) for cav in self.cavities_list],
                    'A_er [mm]': [round(cav.shape_space['OC'][0], 2) for cav in self.cavities_list],
                    'B_er [mm]': [round(cav.shape_space['OC'][1], 2) for cav in self.cavities_list],
                    'a_er [mm]': [round(cav.shape_space['OC'][2], 2) for cav in self.cavities_list],
                    'b_er [mm]': [round(cav.shape_space['OC'][3], 2) for cav in self.cavities_list],
                    'R_er [mm]': [round(cav.shape_space['OC'][4], 2) for cav in self.cavities_list],
                    'L_er [mm]': [round(cav.shape_space['OC'][5], 2) for cav in self.cavities_list],
                    # 'Req [mm]': [round(cav.shape_space['OC'][6], 2) for cav in self.cavities_list],
                    'alpha_er [deg]': [round(cav.shape_space['OC'][7], 2) for cav in self.cavities_list],
                    'R_shunt [Ohm]': ['' for cav in self.cavities_list],
                    'R/Q [Ohm]': [cav.R_Q for cav in self.cavities_list],
                    'k_cc [%]': [cav.k_cc for cav in self.cavities_list],
                    'field flatness [%]': [cav.ff for cav in self.cavities_list],
                    'L_active [m]': [cav.l_active for cav in self.cavities_list],
                    'Epk/Eacc []': [cav.e for cav in self.cavities_list],
                    'Bpk/Eacc [mT/MV/m]': [cav.b for cav in self.cavities_list],
                    'G [Ohm]': [cav.G for cav in self.cavities_list],
                    'R/Q.G [Ohm^2]': [cav.GR_Q for cav in self.cavities_list],
                    '|k_loss| [V/pC]': [cav.k_loss for cav in self.cavities_list],
                    '|k_kick| [V/pC/m]': [cav.k_kick for cav in self.cavities_list],
                    'P_HOM/cav [kW]': [cav.phom for cav in self.cavities_list],
                    'Reference': [cav.reference for cav in self.cavities_list]
                    }

            df = pd.DataFrame.from_dict(data)
            df.to_excel(
                Path(self.projectDir) / "PostprocessingData" / "Data" / f"{self.name}_excel_summary.xlsx",
                sheet_name='Cavities')
        except Exception as e:
            error("Either SLANS or ABCI results not available. Please use '<cav>.set_slans_qois(<folder>)' "
                  "or '<cav>.set_wakefield_qois(<folder>)' to fix this.")

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

    def save_all_plots(self, plot_name):
        """
        Save all plots
        Parameters
        ----------
        plot_name: str
            Name of saved plot

        Returns
        -------

        """
        fname = [cav.name for cav in self.cavities_list]
        if self.projectDir != '':
            # check if folder exists
            if os.path.exists(fr"{self.projectDir}\PostProcessingData\Plots"):
                # create new subdirectory
                if not os.path.exists(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}"):
                    os.mkdir(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}")

                save_folder = fr"{self.projectDir}\PostProcessingData\Plots\{'_'.join(fname)}"
                plt.savefig(f"{save_folder}/{plot_name}", dpi=300)
            else:
                if not os.path.exists(fr"{self.projectDir}\PostProcessingData"):
                    os.mkdir(fr"{self.projectDir}\PostProcessingData")
                    os.mkdir(fr"{self.projectDir}\PostProcessingData\Plots")
                    os.mkdir(fr"{self.projectDir}\PostprocessingData\Plots\{'_'.join(fname)}")

                save_folder = fr"{self.projectDir}\PostProcessingData\Plots\{'_'.join(fname)}"
                plt.savefig(f"{save_folder}/{plot_name}", dpi=300)

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
        c = 299792458
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
        c = 299792458  # m/s

        if mode == 'TM01':
            freq = (c * p_TM01) / (2 * np.pi * Ri * 1e9) * 1e3
        else:
            freq = (c * p_TE11) / (2 * np.pi * Ri * 1e9) * 1e3

        return freq

    def __repr__(self):
        names = list(self.cavities_dict.keys())
        shown = names if len(names) <= 6 else names[:6] + ['...']
        return (f"Study(project={getattr(self, 'name', None)!r}, "
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


