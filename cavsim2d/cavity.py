import ast
import operator as op
import os.path
import subprocess
from abc import ABC, abstractmethod
from distutils import dir_util
import ipywidgets as widgets
from IPython.core.display import HTML, display_html
from IPython.core.display_functions import display
from ipywidgets import HBox, VBox, Label
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks
from scipy.special import *
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.optimisation import Optimisation
from cavsim2d.processes import *
from cavsim2d.utils.shared_functions import *
from cavsim2d.constants import *

# Safe arithmetic evaluator for simple expressions
_ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}

class Cavities(Optimisation):
    """
    Cavities object is an object containing several Cavity objects.
    """

    def __init__(self, folder, name=None, cavities_list=None, names_list=None, overwrite=False):
        """Constructs all the necessary attributes of the Cavity object

        Parameters
        ----------
        cavities_list: list, array like
            List containing Cavity objects.

        """

        super().__init__()

        self.projectDir = folder
        self.cavities_list = cavities_list
        self.cavities_dict = {}
        if cavities_list is None or cavities_list == []:
            self.cavities_list = []
        else:
            self.add_cavity(cavities_list, names_list)

        self.name = 'cavities'
        if name:
            assert isinstance(name, str), error('Please enter valid project name.')
            self.name = name

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
        self.wakefield_qois = {}
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
                assert len(cavs) == len(names), "Number of cavities does not correspond to number of names."
            else:
                names = [f'cav_{ii}' for ii in range(len(self.cavities_list))]

            if plot_labels is not None:
                assert len(cavs) == len(plot_labels), "Number of cavities does not correspond to number of labels."
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

            if not e:
                # create project structure in folders
                project_dir_structure = {
                    f'{project_name}':
                        {
                            'Cavities': None,
                            'OperatingPoints': None,
                            'PostData': {
                                'Plots': None,
                                'Data': None
                            },
                            'Reference': None
                        }
                }
                try:
                    make_dirs_from_dict(project_dir_structure, project_dir)
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    return True, 1
                except Exception as e:
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    error("An exception occurred in created project: ", e)
                    return False, 0
            else:
                # self.projectDir = os.path.join(project_dir, project_name)
                self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                return True, 2
        else:
            error('\tPlease enter a valid project name')
            self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
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

                    if 'Cavities' in directory_list \
                            and 'PostData' in directory_list \
                             and len(directory_list) < 6:
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

    def plot_dispersion(self):
        for cav in self.cavities_list:
            cav.plot_dispersion()

    def check_uq_config(self, uq_config):
        uq_ok = {}
        for cav in self.cavities_list:
            res = cav.check_uq_config(uq_config)
            uq_ok[cav.name] = res
        info(uq_ok)

    def run_tune(self, tune_config=None):
        """

        Parameters
        ----------
        tune_config: dict

            .. code-block:: python

                tune_config = {
                            'freqs': 801.58,
                            'parameters': 'Req',
                            'cell_types': 'mid-cell',
                            'processes': 1,
                            'rerun': True
                        }

        Returns
        -------

        """

        if tune_config is None:
            # set default tune_config
            tune_config = {
                'parameters': ['Req' for _ in self],
                'freqs': [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self],
                'cell_types': ['mid_cell' for _ in self]
            }
            info(f'Tune variable and frequency not entered, defaulting to {json.dumps(tune_config, indent=4)}')

        if 'freqs' not in tune_config.keys():
            # set default tune_config
            tune_config['freqs'] = [c0 / (4 * cav.L * 1e-3) * 1e-6 for cav in self]
            info(f'Target frequency not entered, defaulting to {tune_config["freqs"]}')

        if 'parameters' not in tune_config.keys():
            # set default tune_config
            tune_config['parameters'] = ['Req' for _ in self]
            info(f'"parameters" not entered, defaulting to {tune_config["parameters"]}')

        if 'cell_types' not in tune_config.keys():
            # set default tune_config
            tune_config['cell_types'] = ['mid_cell' for _ in self]
            info(f'"cell_types" not entered, defaulting to {tune_config["cell_types"]}')

        if 'uq_config' in tune_config.keys():
            uq_config = tune_config['uq_config']
            if 'delta' in uq_config.keys():
                assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                     "be equal to the number of "
                                                                                     "variables.")
            if 'epsilon' in uq_config.keys():
                assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                    "The number of epsilons must "
                    "be equal to the number of "
                    "variables.")

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

        if 'eigenmode_config' in tune_config.keys():
            eigenmode_config = tune_config['eigenmode_config']

            if 'uq_config' in eigenmode_config.keys():
                uq_config_eig = eigenmode_config['uq_config']
                if 'delta' in uq_config_eig.keys():
                    assert len(uq_config_eig['delta']) == len(uq_config_eig['variables']), error(
                        "The number of deltas must "
                        "be equal to the number of "
                        "variables.")
                if 'epsilon' in uq_config_eig.keys():
                    assert len(uq_config_eig['epsilon']) == len(uq_config_eig['variables']), error(
                        "The number of epsilons must "
                        "be equal to the number of "
                        "variables.")

                if 'epsilon' in uq_config_eig.keys() and 'uq_config' in uq_config_eig.keys():
                    info('epsilon and delta are both entered. Epsilon is preferred.')

        rerun = True
        if 'rerun' in tune_config:
            rerun = tune_config['rerun']
            assert isinstance(rerun, bool), error('rerun should be boolean.')
        else:
            tune_config['rerun'] = rerun

        if rerun:
            run_tune_parallel(self.cavities_dict, tune_config, solver='NGSolveMEVP', resume=False)

        self.tune_config = tune_config
        # get tune results
        self.get_tune_res()

    def get_tune_res(self):
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_ngsolve_tune_res()
                self.tune_results[cav.name] = cav.tune_results
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    def run_eigenmode(self, eigenmode_config=None):
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

        Returns
        -------
        None

        """

        if eigenmode_config is None:
            eigenmode_config = {}

        eigenmode_config['target'] = run_eigenmode_s

        rerun = True
        if 'rerun' in eigenmode_config.keys():
            if isinstance(eigenmode_config['rerun'], bool):
                rerun = eigenmode_config['rerun']

        uq_config = {}
        if 'uq_config' in eigenmode_config.keys():
            uq_config = eigenmode_config['uq_config']
            if uq_config:
                if 'delta' in uq_config.keys():
                    assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                         "be equal to the number of "
                                                                                         "variables.")

                if 'epsilon' in uq_config.keys():
                    assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                        "The number of epsilons must "
                        "be equal to the number of "
                        "variables.")

                if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                    info('Epsilon and delta are both entered. Epsilon is preferred.')

        if rerun:
            # add save directory field to eigenmode_config
            eigenmode_config['solver_save_directory'] = 'NGSolveMEVP'
            eigenmode_config['opt'] = False

            run_parallel(self.cavities_dict, eigenmode_config,
                                   self.projectDir)

        self.eigenmode_config = eigenmode_config
        self.get_eigenmode_qois(uq_config)

    def get_eigenmode_qois(self, uq_config):
        # get results
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_eigenmode_qois()
                self.eigenmode_qois[cav.name] = cav.eigenmode_qois
                self.eigenmode_qois_all_modes[cav.name] = cav.eigenmode_qois_all_modes
                if uq_config:
                    cav.get_uq_fm_results(cav.uq_dir)
                    self.uq_fm_results[cav.name] = cav.uq_fm_results
                    self.uq_nodes[cav.name] = cav.uq_nodes
                    self.uq_fm_results_all_modes[cav.name] = cav.uq_fm_results_all_modes
            except FileNotFoundError as e:
                error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")
                return False

    def run_wakefield(self, wakefield_config=None):
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

        Returns
        -------

        """

        if wakefield_config is None:
            wakefield_config = {}
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
        if 'uq_config' in wakefield_config_keys:

            if 'delta' in uq_config.keys():
                assert len(uq_config['delta']) == len(uq_config['variables']), error("The number of deltas must "
                                                                                     "be equal to the number of "
                                                                                     "variables.")

            if 'epsilon' in uq_config.keys():
                assert len(uq_config['epsilon']) == len(uq_config['variables']), error(
                    "The number of epsilons must "
                    "be equal to the number of "
                    "variables.")

            if 'epsilon' in uq_config.keys() and 'uq_config' in uq_config.keys():
                info('epsilon and delta are both entered. Epsilon is preferred.')

            assert 'objectives' in wakefield_config['uq_config'].keys(), error('Please enter objectives in uq_config.')

            objectives_unprocessed = []
            # adjust objectives to match signature with optimisation
            for obj in wakefield_config['uq_config']['objectives']:
                if isinstance(obj, list):
                    objectives_unprocessed.append(['', obj[0], obj[1]])
                else:
                    objectives_unprocessed.append(obj)

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

            processes = 1
            if 'processes' in wakefield_config.keys():
                assert wakefield_config['processes'] > 0, error('Number of proceses must be greater than zero.')
                processes = wakefield_config['processes']
            else:
                wakefield_config['processes'] = processes

            if 'polarisation' in wakefield_config_keys:
                assert wakefield_config['polarisation'] in [0, 1, 2], error(
                    'Polarisation should be 0 for longitudinal, '
                    '1 for transverse, 2 for both.')
            else:
                wakefield_config['polarisation'] = MROT

            if 'MT' in wakefield_config_keys:
                assert isinstance(wakefield_config['MT'], int), error(
                    'MT must be integer between 4 and 20, with 4 and 20 '
                    'included.')
            else:
                wakefield_config['MT'] = MT

            if 'NFS' in wakefield_config_keys:
                assert isinstance(wakefield_config['NFS'], int), error('NFS must be integer.')
            else:
                wakefield_config['NFS'] = NFS

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
                assert not isinstance(wakefield_config['beam_config']['bunch_length'], str), error(
                    'Bunch length must be of type integer or float.')
            else:
                beam_config['bunch_length'] = bunch_length

            # wake config
            if 'wakelength' in wake_config.keys():
                assert not isinstance(wake_config['wakelength'], str), error(
                    'Wakelength must be of type integer or float.')
            else:
                wake_config['wakelength'] = wakelength

            if 'DDR_SIG' not in mesh_config.keys():
                mesh_config['DDR_SIG'] = DDR_SIG

            if 'DDZ_SIG' not in mesh_config.keys():
                mesh_config['DDZ_SIG'] = DDZ_SIG

            run_parallel(self.cavities_dict, wakefield_config)

        self.wakefield_config = wakefield_config
        self.get_wakefield_qois(uq_config)

    def get_wakefield_qois(self, uq_config):
        for key, cav in self.cavities_dict.items():
            try:
                cav.get_abci_data()
                cav.get_wakefield_qois(self.wakefield_config)
                self.wakefield_qois[cav.name] = cav.wakefield_qois
                if uq_config:
                    cav.get_uq_hom_results(os.path.join(self.projectDir, "SimulationData", "ABCI", cav.name, "uq.json"))
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
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    def plot(self, what, ax=None, scale_x=None, **kwargs):
        for ii, cav in enumerate(self.cavities_list):
            if what.lower() == 'geometry':
                ax = cav.plot('geometry', ax, **kwargs)

            if what.lower() == 'zl':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('zl', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('zl', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'zt':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('zt', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('zt', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'wpl':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('wpl', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('wpl', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'wpt':
                if scale_x is None:
                    scale_x = [1 for _ in self.cavities_list]
                else:
                    if isinstance(scale_x, list):
                        assert len(scale_x) == len(self.cavities_list), error(
                            'Length of scale_x must be same as number of Cavity objects.')
                    else:
                        scale_x = [scale_x for _ in self.cavities_list]

                if ax:
                    ax = cav.plot('wpt', ax, scale_x=scale_x[ii], **kwargs)
                else:
                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.margins(x=0)
                    ax = cav.plot('wpt', ax, scale_x=scale_x[ii], **kwargs)

            if what.lower() == 'convergence':
                ax = cav.plot('convergence', ax)
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
            cav.plot('geometry', ax=ax, mid_cell=True, zorder=10)
            directory = os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', cav.name)
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

        plt.show()

    def plot_compare_bar(self):
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

        plt.show()

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

    def plot_compare_power_scatter(self, rf_config, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
        """
        Plots bar chart of power quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = figsize

        self.rf_config = rf_config

        assert len(rf_config['Q0 []']) == len(self.cavities_list), error(
            'Lengh of Q0 [] must equal number of Cavity objects.')
        assert len(rf_config['inv_eta []']) == len(self.cavities_list), error(
            'Lengh of inv_eta [] must equal number of Cavity objects.')
        if 'Eacc [MV/m]' in rf_config.keys():
            assert len(rf_config['Eacc [MV/m]']) == len(self.cavities_list), error('length of accelerating field '
                                                                                   'Eacc [MV/m] must equal number '
                                                                                   'of Cavity objects in Cavities.')
        if 'V [GV]' in rf_config.keys():
            assert len(rf_config['V [GV]']) == len(op_points_list), error('length of RF Voltage '
                                                                          'V [GV] must equal number '
                                                                          'of operating points.')

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
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

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

    def plot_compare_eigenmode(self, kind='scatter', uq=False, ncols=3):
        if kind == 'scatter' or kind == 's':
            self.plot_compare_fm_scatter(uq=uq, ncols=ncols)
        if kind == 'bar' or kind == 'b':
            self.plot_compare_fm_bar(uq=uq, ncols=ncols)

    def plot_compare_wakefield(self, opt, kind='scatter', uq=False, ncols=3, figsize=(12, 3)):
        if kind == 'scatter' or kind == 's':
            self.plot_compare_hom_scatter(opt, uq=uq, ncols=ncols, figsize=figsize)
        if kind == 'bar' or kind == 'b':
            self.plot_compare_hom_bar(opt, uq=uq, ncols=ncols)

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

    def plot_compare_hom_bar(self, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
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
        plt.rcParams["figure.figsize"] = figsize
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

            df = pd.DataFrame.from_dict(self.hom_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

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
                for cav, ops_id in self.wakefield_qois.items():
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
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

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

    def plot_compare_hom_scatter(self, op_points_list, ncols=3, uq=False, figsize=(12, 3)):
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
        plt.rcParams["figure.figsize"] = figsize
        if isinstance(op_points_list, str):
            op_points_list = [op_points_list]

        if not uq:
            self.hom_results = self.qois_hom(op_points_list[0])

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

            # Step 1: Flatten the dictionary into a DataFrame
            df_list, df_nominal_list = [], []
            for opt in op_points_list:

                # get nominal qois
                dd_nominal = {}
                for cav, ops_id in self.wakefield_qois.items():
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
            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

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

    def plot_compare_fm_bar(self, ncols=3, uq=False):
        """
        Plot bar chart of fundamental mode quantities of interest

        Returns
        -------

        """
        plt.rcParams["figure.figsize"] = (12, 4)

        if not uq:
            self.fm_results = self.qois_fm()

            df = pd.DataFrame.from_dict(self.fm_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

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

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            num_metrics = len(metrics)

            layout = [[metric for metric in metrics]]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained')

            for metric, ax in axd.items():
                sub_df = df[df['metric'] == metric]
                ax.bar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], label=labels, capsize=5,
                       color=matplotlib.colormaps['Set2'].colors[:len(df)], edgecolor='k',
                       width=1)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_ylabel(LABELS[metric])
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

    def plot_compare_fm_scatter(self, ncols=3, uq=False):
        """
        Plot scatter chart of higher-order mode quantities of interest.

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
        plt.rcParams["figure.figsize"] = (12, 3)

        if not uq:
            self.fm_results = self.qois_fm()
            df = pd.DataFrame.from_dict(self.fm_results)
            fig, axd = plt.subplot_mosaic([list(df.columns)], layout='constrained')

            labels = [cav.plot_label for cav in self.cavities_list]
            colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors

            # Plot each column in a separate subplot
            for key, ax in axd.items():
                for i, label in enumerate(labels):
                    ax.scatter(df.index, df[key], color=colors[i], label=label)
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
            # colors = matplotlib.colormaps['Set2'].colors[:len(labels)]  # Ensure unique colors
            colors = [cav.color for cav in self.cavities_list]

            # Step 2: Create a Mosaic Plot
            metrics = df['metric'].unique()
            layout = [[metric for metric in metrics]]
            # layout = [metrics[:4], [*metrics[4:], '.']]
            fig, axd = plt.subplot_mosaic(layout, layout='constrained', figsize=(21, 3))

            # Plot each metric on a separate subplot
            for metric, ax in axd.items():
                for i, label in enumerate(labels):
                    sub_df = df[(df['metric'] == metric) & (df['cavity'] == label)]
                    scatter_points = ax.scatter(sub_df['cavity'], sub_df['mean'], color=colors[i], s=150,
                                                fc='none', ec=colors[i], label='Mean', lw=3, zorder=100)
                    ax.errorbar(sub_df['cavity'], sub_df['mean'], yerr=sub_df['std'], capsize=10, lw=3,
                                color=scatter_points.get_edgecolor()[0])

                    # plot nominal
                    ax.scatter(df_nominal.index, df_nominal[metric], facecolor='none', label='Design Point',
                               ec='k', lw=1, s=75,
                               zorder=100)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.margins(0.3)
                # ax.set_ylabel(LABELS[metric])
                ax.set_xlabel(LABELS[metric])

            h, l = ax.get_legend_handles_labels()

        # by_label = dict(zip(l, h))
        # if 'nominal' in by_label.keys():
        #     nominal_handle = by_label.pop('nominal')  # Remove 'nominal' entry
        #     # Reinsert 'nominal' as the last entry
        #     by_label['nominal'] = nominal_handle

        # Set legend
        if not ncols:
            ncols = min(4, len(self.cavities_list))
        fig.legend(*reorder_legend(h, l, ncols), loc='outside upper center', borderaxespad=0, ncol=ncols)

        # Save plots
        fname = [cav.name for cav in self.cavities_list]
        fname = '_'.join(fname)

        self.save_all_plots(f"{fname}_fm_scatter.png")

        return axd

    def plot_compare_all_scatter(self, opt, ncols=3):
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
            for cav, ops_id in self.wakefield_qois.items():
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

    def plot_compare_all_bar(self, opt, ncols=3):
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
        plt.ylabel('$|E_\mathrm{0, z}|/|E_\mathrm{0, z}|_\mathrm{max}$')
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
        plt.xlabel('$L_\mathrm{surf}$ [mm]')
        plt.ylabel('$|E_\mathrm{surf}|/|E_\mathrm{surf}|_\mathrm{max}$')
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

    def run_abci(self):
        for cav in self.cavities_list:
            cav.run_abci()

    def run_multipacting(self):
        for cav in self.cavities_list:
            cav.run_multipacting()

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
                os.path.join(self.projectDir, "SimulationData", "Data", f"{self.name}_excel_summary.xlsx"),
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

    def run_optimisation(self, optimisation_config):
        self.start_optimisation(self.projectDir, optimisation_config)

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
                error('Invalid key. Cavities does not contain a Cavity named {key}.')
        else:
            raise TypeError("Invalid argument type. Must be int or str.")


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
        self.eigenmode_dir = None
        self.wakefield_dir = None
        self.uq_dir = None

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

        # If a geometry file is provided, skip dimension‐based setup:
        if geo_filepath:
            self._init_from_geo(geo_filepath)
            self.get_geometric_parameters()

    def _init_from_geo(self, filepath, kind='geo'):
        """
        Load geometry from a file (e.g. a .geo)
        """
        self.step_geo, self.mesh, self.bcs = ngsolve_mevp.load_geo(filepath)

    def get_geometric_parameters(self):
        with open(self.geo_filepath, "r") as file:
            for line in file:
                match = re.match(r'\s*(\w+)\s*=\s*DefineNumber\[\s*([\d.eE+-]+)\s*,\s*Name\s*"Parameters/[^"]*"\s*\];',
                                 line)
                if match:
                    var_name, var_value = match.groups()
                    self.parameters[var_name] = float(var_value)

    def create(self):
        pass

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
        if how == 'cross':
            pass
        else:
            for key, interval_def in sweep_config.items():
                # save nominal variable value form shape space
                current_var = self.shape['IC'][VAR_TO_INDEX_DICT[key]]
                par_vals = np.linspace(interval_def[0], interval_def[1], interval_def[2], endpoint=True)
                self.sweep_results[f'{key}'] = {}
                for val in par_vals:
                    if which == 'eigenmode':
                        # change value
                        self.shape['IC'][VAR_TO_INDEX_DICT[key]] = val
                        res = self.run_eigenmode()
                        if res:
                            self.sweep_results[f'{key}'][val] = copy.deepcopy(self.eigenmode_qois)
                            if uq_config:
                                self.sweep_results_uq[f'{key}'][val] = copy.deepcopy(self.uq_fm_results)

                # replace initial value
                self.shape['IC'][VAR_TO_INDEX_DICT[key]] = current_var

    def study_mesh_convergence(self, h=2, h_passes=7, h_step=1.5, p=2, p_passes=6, p_step=1):
        """

        Parameters
        ----------
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

                run_eigenmode_s({self.name: self}, eigenmode_config, '')
                # read results
                self.get_eigenmode_qois()

                convergence_dict_h[ih] = self.eigenmode_qois
                convergence_dict_h[ih]['h'] = hs
                convergence_dict_h[ih]['p'] = p
                convergence_dict_h[ih]['No of Mesh Elements'] = self.eigenmode_qois['No of Mesh Elements']

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

    def calculate_rel_errors(self, df):
        # Specify the columns to exclude
        columns_to_exclude = ['h', 'p', 'No of Mesh Elements']
        df_to_compute = df.drop(columns=columns_to_exclude)
        df_prev = df_to_compute.shift(1)
        relative_errors = (df_to_compute - df_prev).abs() / df_prev.abs()
        relative_errors.columns = [f'rel_error_{col}' for col in df_to_compute.columns]
        excluded_columns = df[columns_to_exclude]
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

        if boundary_cond:
            self.bc = boundary_cond

        if self.cell_parameterisation == 'multicell':
            self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape, self.shape_multicell,
                              self.n_modes,
                              freq_shift, self.bc,
                              SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)
        else:
            self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape, self.shape_multicell,
                              self.n_modes,
                              freq_shift, self.bc,
                              SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)

        # load quantities of interest
        try:
            self.get_eigenmode_qois()
            if uq_config:
                self.get_uq_fm_results(
                    os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', f'{self.name}', 'uq.json'))
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

        if operating_points is None:
            wp_dict = {}
        exist = False

        # check if R/Q is set
        if self.R_Q == 0:
            self.get_eigenmode_qois()
            self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        if not exist:
            if solver == 'ABCI':
                self._run_abci(self.name, self.n_cells, self.n_modules, self.shape,
                               MROT=MROT, MT=MT, NFS=NFS, UBT=wakelength, bunch_length=bunch_length,
                               DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG,
                               parentDir=SOFTWARE_DIRECTORY, projectDir=self.projectDir, WG_M=WG_M, marker=marker,
                               operating_points=operating_points, freq=self.freq, R_Q=self.R_Q)

                try:
                    self.get_abci_data()
                    self.get_wakefield_qois()
                except FileNotFoundError:
                    error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

        else:
            try:
                self.get_abci_data()
                self.get_wakefield_qois()
            except FileNotFoundError:
                error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

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

    @staticmethod
    def _run_ngsolve(name, n_cells, n_modules, shape, shape_multi, n_modes, f_shift, bc, parentDir, projectDir,
                     sub_dir='',
                     uq_config=None):
        parallel = False
        start_time = time.time()
        # create folders for all keys
        ngsolve_mevp.createFolder(name, projectDir, subdir=sub_dir)

        if 'OC_R' in shape.keys():
            OC_R = 'OC_R'
        else:
            OC_R = 'OC'

        # ngsolve_mevp.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
        #                     n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
        #                     parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        if shape['CELL PARAMETERISATION'] == 'flattop':
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())

            ngsolve_mevp.cavity_flattop(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                        n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc,
                                        beampipes=shape['BP'],
                                        parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

        elif shape['CELL PARAMETERISATION'] == 'multicell':
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())
            ngsolve_mevp.cavity_multicell(n_cells, n_modules, shape_multi['IC'], shape_multi['OC'], shape_multi[OC_R],
                                          n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc,
                                          beampipes=shape['BP'],
                                          parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        else:
            # write_cst_paramters(f"{key}_n{n_cell}", shape['IC'], shape['OC'], shape['OC_R'],
            #                     projectDir=projectDir, cell_type="None", solver=select_solver.lower())
            ngsolve_mevp.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape[OC_R],
                                n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
                                parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)

        # run UQ
        if uq_config:
            objectives = uq_config['objectives']
            solver_dict = {'eigenmode': ngsolve_mevp}
            solver_args_dict = {'eigenmode':
                                    {'n_cells': n_cells, 'n_modules': n_modules, 'f_shift': f_shift, 'bc': bc,
                                     'beampipes': shape['BP']
                                     },
                                'parentDir': parentDir,
                                'projectDir': projectDir,
                                'analysis folder': 'NGSolveMEVP',
                                'cell_type': 'mid cell',
                                'optimisation': False
                                }

            uq_cell_complexity = 'simplecell'
            if 'cell_complexity' in uq_config.keys():
                uq_cell_complexity = uq_config['cell_complexity']

            if uq_cell_complexity == 'multicell':
                shape_space = {name: shape_multi}
                uq_parallel_multicell(shape_space, objectives, solver_dict, solver_args_dict, uq_config)
            else:
                shape_space = {name: shape}
                uq_parallel(shape_space, objectives, solver_dict, solver_args_dict, 'eigenmode')

        done(f'Done with Cavity {name}. Time: {time.time() - start_time}')

    def set_wall_material(self, wm):
        self.wall_material = wm

    def get_ngsolve_tune_res(self):
        """

        Parameters
        ----------
        tune_variable: {'A', 'B', 'a'. 'b', 'Ri', 'L', 'Req'}
            Tune variable.
        cell_type: {'mid cell', 'end-mid cell', 'mid-end cell', 'single cell'}
            Type of cell to tune

        Returns
        -------

        """
        tune_res = 'tune_res.json'
        if os.path.exists(os.path.join(self.self_dir, 'eigenmode', 'tune_res.json')):
            with open(os.path.join(self.self_dir, 'eigenmode', 'tune_res.json'),
                      'r') as json_file:
                self.tune_results = json.load(json_file)
            self.freq = self.tune_results['FREQ']
            # self.shape['IC'] = self.tune_results['IC']
            # self.shape['OC'] = self.tune_results['OC']
            # self.shape['OC_R'] = self.tune_results['OC_R']
            # self.mid_cell = self.shape['IC']
            # self.end_cell_left = self.shape['OC']
            # self.end_cell_right = self.shape['OC_R']

        else:
            error("Tune results not found. Please tune the cavity")

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        qois = 'qois.json'
        assert os.path.exists(
            os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)), (
            error('Eigenmode result does not exist, please run eigenmode simulation.'))
        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole',
                               'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
            self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

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

    def plot_dispersion(self, ax=None, show_continuous=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()

        df = pd.DataFrame.from_dict(self.eigenmode_qois_all_modes, orient='index')
        freqs = df['freq [MHz]'][1:self.n_cells + 1]

        k = np.linspace(np.pi / self.n_cells, np.pi, self.n_cells, endpoint=True)
        axis_label = ['$' + f'({i + 1}/{self.n_cells})' + r'\pi$' if i - 1 != self.n_cells else r'$\pi$' for i in
                      range(self.n_cells)]
        ax.plot(k, freqs, marker='o', mec='k', label=self.name, **kwargs)
        ax.set_xticklabels(axis_label)
        ax.set_ylabel('$f$ [MHz]')
        ax.set_xlabel('$k$')

        # if show_continuous:
        #     # calculate continuous
        #     k = np.linspace(0, np.pi, 100, endpoint=True)
        #     f_pi_over_2 = (freqs[0] + freqs[-1])/2
        #     freq_c = np.sqrt(freqs[0]**2 + freqs[-1]**2 - 2*freqs[0]*freqs[-1]*np.cos(k))
        #     ax.plot(k, freq_c, c='k', lw=2)

        return ax

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
            ax.text(
                0.95, 0.05,  # Position (normalized coordinates)
                '$\eta=' + fr'{self.ff:.2f}\%' + '$',  # Text content
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
            if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                           'Ez_0_abs.csv')):
                with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                       'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
                ax.text(
                    0.95, 0.05,  # Position (normalized coordinates)
                    '$\eta=$' + fr'{self.ff:.2f}\%' + '$',  # Text content
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
            df = pd.DataFrame({k: {sub_k: v[0] for sub_k, v in sub_dict.items()} for k, sub_dict in
                               self.uq_fm_results_all_modes.items()})
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 4))

            # Generate KDE data
            for col in ['freq_1 [MHz]', 'freq_2 [MHz]', 'freq_3 [MHz]', 'freq_4 [MHz]', 'freq_5 [MHz]', 'freq_6 [MHz]',
                        'freq_7 [MHz]', 'freq_8 [MHz]', 'freq_9 [MHz]']:
                mean = df[col]["expe"]
                std = df[col]["stdDev"]
                x_values = np.linspace(mean - 10, mean + 10, 100000)  # Adjust range as needed
                if std / mean < 1e-4:  # set a very narrow range for the x-axis around the mean.
                    x = np.linspace(mean - 10, mean + 10, 100000)
                    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
                    ax.plot(x, y_values * 1e-8, label=col)
                else:
                    y_values = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / std) ** 2)
                    ax.plot(x_values, y_values * 1e-8, label=col)

            # Plot settings
            ax.set_title("KDE Plots for Frequencies")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Density")
            ax.set_xlim(1270, 1310)
            # ax.set_ylim(0, 3)
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
        for dirr in os.listdir(os.path.join(folder, 'Cavities')):
            if 'Q' in dirr.split('_')[-1]:
                with open(os.path.join(folder, 'Cavities', dirr, 'eigenmode', 'monopole', 'qois.json'), 'r') as json_file:
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
        qois = 'qois.json'

        if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'ABCI', self.name, 'qois.json')):
            with open(os.path.join(self.projectDir, 'SimulationData', 'ABCI', self.name, 'qois.json')) as json_file:
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

    def get_abci_data(self):
        self.abci_data = {'Long': ABCIData(self.wakefield_dir, 'longitudinal', 0),
                          'Trans': ABCIData(self.wakefield_dir, 'transversal', 1)}

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

        top_folder = os.path.join(self.projectDir, "SimulationData", "ABCI", self.name)
        efield_contour = get_wakefield_data(fr'{top_folder}/Cavity_MROT_0.top')
        ani = animate_frames(efield_contour)
        display_html(HTML(animate_frames(efield_contour).to_jshtml()))

        if save:
            # Save the animation as an MP4 file
            ani.save(fr'{top_folder}/{self.name}_e_field_animation.mp4', writer='ffmpeg', dpi=150)

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

    def get_fields(self, mode=1):
        gfu_E, gfu_H = ngsolve_mevp.load_fields(os.path.join(self.projectDir,
                                                             'SimulationData', 'NGSolveMEVP',
                                                             self.name, 'monopole'),
                                                mode)
        return gfu_E, gfu_H

    def plot(self, what, ax=None, scale_x=1, **kwargs):
        if what.lower() == 'geometry':
            if 'mid_cell' in kwargs.keys():
                new_kwargs = {key: val for key, val in kwargs.items() if key != 'mid_cell'}
                ax = write_cavity_geometry_cli(self.mid_cell, self.mid_cell, self.mid_cell,
                                               'none', 1, scale=1, ax=ax, plot=True, **new_kwargs)
            elif 'end_cell_left' in kwargs.keys():
                ax = write_cavity_geometry_cli(self.end_cell_left, self.end_cell_left, self.end_cell_left,
                                               'left', 1, scale=1, ax=ax, plot=True, **kwargs)
            elif 'end_cell_right' in kwargs.keys():
                ax = write_cavity_geometry_cli(self.end_cell_right, self.end_cell_right, self.end_cell_right,
                                               'right', 1, scale=1, ax=ax, plot=True, **kwargs)
            else:
                ax = write_cavity_geometry_cli(self.mid_cell, self.end_cell_left, self.end_cell_right,
                                               self.beampipe, self.n_cells, scale=1, ax=ax, plot=True, **kwargs)
            ax.set_xlabel('$z$ [mm]')
            ax.set_ylabel(r"$r$ [mm]")
            return ax

        if what.lower() == 'zl':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Long.)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Long.)', **kwargs)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Trans.)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * scale_x * 1e3, y, label=fr'{self.name} (Wake. Trans.)', **kwargs)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
            return ax

        if what.lower() == 'wpl':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Long'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Longitudinal wake potentials)', **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Long'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Longitudinal wake potentials)', **kwargs)

            ax.set_xlabel('Distance from Bunch Head S [m]')
            ax.set_ylabel(r"Scaled Wake Potentials $W (S)$ [V/pC]")
            return ax
        if what.lower() == 'wpt':
            if ax:
                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Transversal wake potentials)', lw=3, **kwargs)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)

                if 'c' not in kwargs.keys():
                    kwargs['c'] = self.color

                x, y, _ = self.abci_data['Trans'].get_data('Wake Potentials')
                ax.plot(x, y, label=fr'{self.name} (Transversal wake potentials)', lw=3, **kwargs)
            ax.set_xlabel('Distance from Bunch Head S [m]')
            ax.set_ylabel(r"Scaled Wake Potentials $W (S)$ [V/pC/m]")
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

    def plot_mesh(self, plotter='ngsolve'):
        mesh_path = os.path.join(self.self_dir, 'eigenmode', 'monopole')

        if os.path.exists(mesh_path):
            ngsolve_mevp.plot_mesh(mesh_path, plotter=plotter)
        else:
            error('The folder or mesh does not exist. Please check that a mesh file was writtten.')

    def plot_fields(self, mode=1, which='E', plotter='ngsolve'):
        field_path = os.path.join(self.self_dir, 'eigenmode', 'monopole')

        if os.path.exists(field_path):
            ngsolve_mevp.plot_fields(field_path, mode, which, plotter)
        else:
            error('The foler or field file does not exist. Please check that a mesh file was writtten.')

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
    def geo_to_abc(self, wakefield_config=None, **kwargs):
        # Read input
        with open(self.geo_filepath) as f:
            lines = [l.split('%', 1)[0].strip() for l in f]

        # 1) Build symbol table from DefineNumber
        symbols = {}
        define_re = re.compile(r"(\w+)\s*=\s*DefineNumber\[\s*([^,\]]+)")
        for line in lines:
            m = define_re.search(line)
            if m:
                symbols[m.group(1)] = m.group(2).strip()
        # Evaluate until fixed
        changed = True
        while changed:
            changed = False
            for k, v in list(symbols.items()):
                if isinstance(v, str):
                    try:
                        symbols[k] = self.eval_expr(v, symbols)
                        changed = True
                    except:
                        pass

        # 2) Parse geometry segments
        point_re = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\}")
        ellipse_re = re.compile(r"Ellipse\(\d+\)\s*=\s*\{([^}]+)\}")

        points = {}  # pid -> (r,z)
        segments = []  # list of ('point', r, z) or ('ellipse', end_pid)

        # First pass: collect points structs and z-values
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

        # Second pass: build segment list
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
                # remove last 3 point segments
                if len(segments) >= 3:
                    segments = segments[:-3]
                # add ellipse indicator + endpoint
                mid_pid = ids[1]
                end_pid = ids[3]
                # segments.append(('ellipse', mid_pid))
                segments.append(('ellipse', mid_pid, end_pid))

        # Create geometry if not crated
        self.create()

        # 3) Write output
        wakefield_folder_structure = {
            'wakefield': {
                'longitudinal': None,
                'transversal': None
            }
        }
        make_dirs_from_dict(wakefield_folder_structure, self.self_dir)

        MROT = wakefield_config['polarisation']
        if  MROT == 2:
            for m in range(2):
                self._write_abc(points, segments, zmin, symbols, m, wakefield_config, **kwargs)
        else:
            self._write_abc(points, segments, zmin, symbols, MROT, wakefield_config, **kwargs)

    def _write_abc(self, points, segments, zmin, symbols, MROT, wakefield_config, **kwargs):
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

        with open(os.path.join(self.self_dir, 'wakefield', MROT_DICT[MROT], 'cavity.abc'), 'w') as out:
            # Header
            out.write(f' &FILE LSAV = {LSAV}, ITEST = 0, LREC = F, LCPUTM = {LCPUTM} &END \n')
            out.write(' SAMPLE INPUT #1 A SIMPLE CAVITY STRUCTURE \n')
            out.write(' &BOUN  IZL = 3, IZR = 3  &END \n')
            out.write(f' &MESH DDR = {mesh_DDR}, DDZ = {mesh_DDZ} &END \n')
            out.write(" #CAVITYSHAPE\n0.\n")

            # Body
            for seg in segments:
                if seg[0] == 'point':
                    _, r, z = seg
                    out.write(f"{r:.16f} {z:.16f}\n")
                elif seg[0] == 'ellipse':
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
                f' &WAKE  UBT = {int(UBT)}, LCRBW = {LCRBW}, LCBACK = {LCBACK}, LCRBW = {LCRBW}, ZSEP = {ZSEP} &END \n')  # , NFS = {NFS}
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
                def make_dirs_from_dict(d, current_dir=fr"{project_dir}"):
                    for key, val in d.items():
                        os.mkdir(os.path.join(current_dir, key))
                        if type(val) == dict:
                            make_dirs_from_dict(val, os.path.join(current_dir, key))

                # create project structure in folders
                project_dir_structure = {
                    f'{project_name}':
                        {
                            'Cavities': None,
                            'OperatingPoints': None,
                            'SimulationData': {
                                'SLANS': None,
                                'SLANS_Opt': None,
                                'NGSolveMEVP': None,
                                'NativeEig': None,
                                'ABCI': None,
                                'CavitiesAnalysis': None
                            },
                            'PostprocessingData': {
                                'Plots': None,
                                'Data': None,
                                'CSTData': None
                            },
                            'Reference': None
                        }
                }
                try:
                    make_dirs_from_dict(project_dir_structure)
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    return True
                except Exception as e:
                    self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                    error("An exception occurred in created project: ", e)
                    return False
            else:
                self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
                return True
        else:
            info('\tPlease enter a valid project name')
            self.projectDir = f2b_slashes(fr"{project_dir}\{project_name}")
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

                    if 'Cavities' in directory_list \
                            and 'PostprocessingData' in directory_list \
                            and 'SimulationData' in directory_list and len(directory_list) < 6:
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
        path = os.path.join(self.projectDir, 'SimulationData', 'SLANS', f'_process_{invar}')
        if os.path.exists(path):
            shutil.rmtree(path)
            dir_util._path_created = {}

        os.makedirs(path)

    @staticmethod
    def _copyFiles(invar, parentDir, projectDir, name):
        src = os.path.join(parentDir, 'exe', 'SLANS_exe')
        dst = os.path.join(projectDir, 'SimulationData', 'SLANS', f'_process_{invar}', 'SLANS_exe')

        dir_util.copy_tree(src, dst)

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

    @abstractmethod
    def create(self):
        pass


class EllipticalCavity(Cavity):
    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k', plot_label=None):
        """
        All of the old “dimension‐based” logic has been moved here.
        Assumes self.kind, self.name, self.color, etc. are already set.
        """
        super().__init__()

        self.projectDir = None

        self.plot_label = plot_label
        self.name = name
        self.kind = 'elliptical cavity'
        self.beampipe = beampipe
        self.color = color
        self.geo_filepath = None
        self.self_dir = None

        self.n_cells = n_cells
        self.cell_parameterisation = cell_parameterisation

        # Basic counters / containers
        self.n_modes = (n_cells + 1) if n_cells is not None else None
        self.n_modules = 1
        self.no_of_modules = 1

        # Handle the special case: mid_cell can be a dict with keys OC, OC_R, IC
        if isinstance(mid_cell, dict):
            end_cell_left = mid_cell['OC']
            end_cell_right = mid_cell['OC_R']
            mid_cell = mid_cell['IC']

        self.mid_cell = np.array(mid_cell)[:7]
        if end_cell_left is not None:
            self.end_cell_left = np.array(end_cell_left)[:7]
        else:
            self.end_cell_left = np.copy(self.mid_cell)

        if end_cell_right is not None:
            self.end_cell_right = np.array(end_cell_right)[:7]
        else:
            self.end_cell_right = np.copy(self.end_cell_left)

        # Unpack 7 parameters
        (self.A, self.B, self.a, self.b,
         self.Ri, self.L, self.Req) = self.mid_cell[:7]

        (self.A_el, self.B_el, self.a_el,
         self.b_el, self.Ri_el, self.L_el,
         self.Req_el) = self.end_cell_left[:7]

        (self.A_er, self.B_er, self.a_er,
         self.b_er, self.Ri_er, self.L_er,
         self.Req_er) = self.end_cell_right[:7]

        # Active length & cavity length
        self.l_active = (
                        2 * (self.n_cells - 1) * self.L +
                        self.L_el +
                        self.L_er
                        ) * 1e-3
        self.l_cavity = self.l_active + 8 * self.L * 1e-3

        # Build self.shape dictionary
        self.shape = {
            "IC": update_alpha(self.mid_cell[:7], self.cell_parameterisation),
            "OC": update_alpha(self.end_cell_left[:7], self.cell_parameterisation),
            "OC_R": update_alpha(self.end_cell_right[:7], self.cell_parameterisation),
            "BP": beampipe,
            "n_cells": self.n_cells,
            "CELL PARAMETERISATION": self.cell_parameterisation,
            "kind": self.kind
        }

        self.to_multicell()
        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            # define different paths for easier reference later
            self.eigenmode_dir = os.path.join(self.self_dir, 'eigenmode')
            self.wakefield_dir = os.path.join(self.self_dir, 'wakefield')
            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(self.projectDir, 'Cavities', self.name, 'geometry', 'geodata.geo')
            if mode is None:
                self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)
            else:
                self.write_quarter_geometry(self.parameters, beampipe, write=self.geo_filepath)

    def write_geometry(self, parameters, n_cells, BP, scale=1, ax=None, bc=None, tangent_check=False,
                                  ignore_degenerate=False, plot=False, write=None, dimension=False,
                                  contour=False, **kwargs):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        GEO = """
        """
        # IC, OC, OC_R = shape['IC'], shape['OC'], shape['OC_R']
        # BP = self.beampipe
        # n_cell = self.n_cells

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_aspect('equal')

        names = ['A_m', 'B_m', 'a_m', 'b_m', 'Ri_m', 'L_m', 'Req_m',
                 'A_el', 'B_el', 'a_el', 'b_el', 'Ri_el', 'L_el', 'Req_el',
                 'A_er', 'B_er', 'a_er', 'b_er', 'Ri_er', 'L_er', 'Req_er']

        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, \
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, \
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = (parameters[n]*1e-3 for n in names)

        Req = Req_m  # CHANGE THIS, Req should be same

        L_bp = 4 * L_m
        if dimension or contour:
            L_bp = 1 * L_m

        if BP.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif BP.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif BP.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        step = 0.0005

        # calculate shift
        shift = (L_bp_r + L_bp_l + L_el + (n_cells - 1) * 2 * L_m + L_er) / 2

        # calculate angles outside loop
        # CALCULATE x1_el, y1_el, x2_el, y2_el

        df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, tangent_check=tangent_check)
        x1el, y1el, x2el, y2el = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1, y1, x2, y2
        df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, tangent_check=tangent_check)
        x1, y1, x2, y2 = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, tangent_check=tangent_check)
        x1er, y1er, x2er, y2er = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # # define parameters
            # cav.write(f'\nA_el = DefineNumber[{A_el}, Name "Parameters/End cell 1 Equator ellipse major axis"];')
            # cav.write(f'\nB_el = DefineNumber[{B_el}, Name "Parameters/End cell 1 Equator ellipse minor axis"];')
            # cav.write(f'\na_el = DefineNumber[{a_el}, Name "Parameters/End cell 1 Iris ellipse major axis"];')
            # cav.write(f'\nb_el = DefineNumber[{b_el}, Name "Parameters/End cell 1 Iris ellipse minor axis"];')
            # cav.write(f'\nRi_el = DefineNumber[{Ri_el}, Name "Parameters/End cell 1 Iris radius"];')
            # cav.write(f'\nL_el = DefineNumber[{L_el}, Name "Parameters/End cell 1 Half cell length"];')
            # # cav.write(f'\nReq_el = DefineNumber[{Req}, Name "Parameters/End cell 1 Equator radius"];\n')
            #
            # cav.write(f'\nA_m = DefineNumber[{A_el}, Name "Parameters/Mid cell Equator ellipse major axis"];')
            # cav.write(f'\nB_m = DefineNumber[{B_el}, Name "Parameters/Mid cell Equator ellipse minor axis"];')
            # cav.write(f'\na_m = DefineNumber[{a_el}, Name "Parameters/Mid cell Iris ellipse major axis"];')
            # cav.write(f'\nb_m = DefineNumber[{b_el}, Name "Parameters/Mid cell Iris ellipse minor axis"];')
            # cav.write(f'\nRi_m = DefineNumber[{Ri_el}, Name "Parameters/Mid cell Iris radius"];')
            # cav.write(f'\nL_m = DefineNumber[{L_el}, Name "Parameters/Mid cell Half cell length"];')
            # cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Mid cell Equator radius"];\n')
            #
            # cav.write(f'\nA_er = DefineNumber[{A_el}, Name "Parameters/End cell 2 Equator ellipse major axis"];')
            # cav.write(f'\nB_er = DefineNumber[{B_el}, Name "Parameters/End cell 2 Equator ellipse minor axis"];')
            # cav.write(f'\na_er = DefineNumber[{a_el}, Name "Parameters/End cell 2 Iris ellipse major axis"];')
            # cav.write(f'\nb_er = DefineNumber[{b_el}, Name "Parameters/End cell 2 Iris ellipse minor axis"];')
            # cav.write(f'\nRi_er = DefineNumber[{Ri_el}, Name "Parameters/End cell 2 Iris radius"];')
            # cav.write(f'\nL_er = DefineNumber[{L_el}, Name "Parameters/End cell 2 Half cell length"];')
            # # cav.write(f'\nReq_er = DefineNumber[{Req}, Name "Parameters/End cell 2 Equator radius"];\n')

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])
            pts = lineTo(start_point, [-shift, Ri_el], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [-shift, Ri_el]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            if bc:
                # draw left boundary condition
                ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                        [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
                        [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pts = lineTo(pt, [L_bp_l - shift, Ri_el], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0]])
                pt = [L_bp_l - shift, Ri_el]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

            for n in range(1, n_cells + 1):
                if n == 1:
                    # DRAW ARC:
                    if plot and dimension:
                        ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                                                 height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                                    xytext=(L_bp_l - shift, Ri_el + b_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))
                        ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                                    xytext=(L_bp_l - shift, Ri_el + b_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_el + b_el]
                    majax_pt = [L_bp_l - shift + a_el, Ri_el + b_el]
                    end_pt = [-shift + x1el, y1el]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                    pt = [-shift + x1el, y1el]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2el, y2el], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2el, y2el]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    if plot and dimension:
                        ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                                 height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                                    xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                    xytext=(L_el + L_bp_l - shift, Req - B_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

                    start_pt = pt
                    center_pt = [L_el + L_bp_l - shift, Req - B_el]
                    majax_pt = [L_el + L_bp_l - shift - A_el, Req - B_el]
                    end_pt = [L_bp_l + L_el - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                    pt = [L_bp_l + L_el - shift, Req]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    if n_cells == 1:
                        if L_bp_r > 0:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper

                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            if plot and dimension:
                                ax.scatter(L_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_er),
                                                                         width=2 * A_er,
                                                                         height=2 * B_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_er),
                                            xytext=(L_el + L_bp_l - shift + A_er, Req - B_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                            xytext=(L_el + L_bp_l - shift, Req - B_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n',
                                        ha='center',
                                        va='center')
                                ax.text(L_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                        va='center', ha='left', rotation=90)

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])

                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                         width=2 * a_er,
                                                                         height=2 * b_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])

                            geo.append([pt[1], pt[0], 0])

                            # calculate new shift
                            shift = shift - (L_el + L_er)
                        else:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                         width=2 * a_er,
                                                                         height=2 * b_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er - shift, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])
                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            # pt_indx += 1

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                    else:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper

                        start_pt = pt
                        center_pt = [L_el + L_bp_l - shift, Req - B_m]
                        majax_pt = [L_el + L_bp_l - shift + a_m, Req - B_m]
                        end_pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                          end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                    [L_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                        pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # STRAIGHT LINE TO NEXT POINT
                        pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                        # for pp in pts:
                        #     geo.append([pp[1], pp[0], 0])
                        pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]

                        pt_indx = add_point(cav, pt, pt_indx)
                        curve_indx = add_line(cav, pt_indx, curve_indx)
                        curve.append(curve_indx)

                        geo.append([pt[1], pt[0], 0])

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        start_pt = pt
                        center_pt = [L_el + L_m + L_bp_l - shift, Ri_m + b_m]
                        majax_pt = [L_el + L_m + L_bp_l - shift - A_m, Ri_m + b_m]
                        end_pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                          end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                    [L_bp_l + L_el + L_m - shift, Ri_m])
                        pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # calculate new shift
                        shift = shift - (L_el + L_m)
                        # ic(shift)

                elif n > 1 and n != n_cells:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_m, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_m, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift + A_m, Req - B_m]
                    end_pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_m + L_m - x2 + 2 * L_bp_l - shift, y2])
                    pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_m + L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_m + L_m + L_bp_l - shift - a_m, Ri_m + b_m]
                    end_pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_m + L_m - shift, Ri_m])
                    pt = [L_bp_l + L_m + L_m - shift, Ri_m]

                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # calculate new shift
                    shift = shift - 2 * L_m
                else:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_er, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_er, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_er]
                    majax_pt = [L_m + L_bp_l - shift + A_er, Req - B_er]
                    end_pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_er + L_bp_l - shift, Ri_er + b_er]
                    majax_pt = [L_m + L_er + L_bp_l - shift - a_er, Ri_er + b_er]
                    end_pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_m + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    if L_bp_r > 0:
                        geo.append([pt[1], pt[0], 0])
                    else:
                        geo.append([pt[1], pt[0], 1])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_r + L_bp_l + (n_cells - 1) * 2 * L_m + L_el + L_er) / 2

            if L_bp_r > 0:  # if there's a problem, check here.
                pts = lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cells - 1) * L_m + L_el + L_er - shift, Ri_er], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0], 0])
                pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 1])

            # END PATH
            pts = lineTo(pt, [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0],
                         step)  # to add beam pipe to right
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
            # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
            geo.append([pt[1], pt[0], 2])

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

        # write geometry
        # if write:
        #     try:
        #         df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
        #         # change point data precision
        #         df['r'] = df['r'].round(8)
        #         df['z'] = df['z'].round(8)
        #         # drop duplicates
        #         df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
        #         df.to_csv(write, sep='\t', index=False)
        #     except FileNotFoundError as e:
        #         error('Check file path:: ', e)

        # append start point
        # geo.append([start_point[1], start_point[0], 0])

        if bc:
            # draw right boundary condition
            ax.plot([shift, shift], [-Ri_er, Ri_er],
                    [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                    [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

        # CLOSE PATH
        # lineTo(pt, start_point, step)
        # geo.append([start_point[1], start_point[0], 0])
        geo = np.array(geo)

        if plot:

            if dimension:
                top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
            else:
                # recenter asymmetric cavity to center
                shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
                if n_cells == 1:
                    shift_to_center = L_er + L_bp_r
                else:
                    shift_to_center = n_cells * L_m + L_bp_r

                top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
                bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3,
                                 c=top[0].get_color(),
                                 **kwargs)

            # plot legend without duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        return ax

    def write_quarter_geometry(self, parameters, bp='none', bc=None, tangent_check=False,
                                          ignore_degenerate=False, plot=False, write=None, dimension=False,
                                          contour=False, **kwargs):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        GEO = """
        """

        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        if bp == 'none':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)
        elif bp == 'left':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_el']*1e-3 for n in names)
        elif bp == 'right':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_er']*1e-3 for n in names)
        else:
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)

        L_bp = 4 * L
        if dimension or contour:
            L_bp = 1 * L

        if bp == 'left' or bp == 'right':
            L_bp_l = L_bp
        else:
            L_bp_l = 0

        step = 0.0005

        # calculate shift
        shift = (L_bp_l + L) / 2

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # define parameters
            cav.write(f'\nA = DefineNumber[{A}, Name "Parameters/Equator ellipse major axis"];')
            cav.write(f'\nB = DefineNumber[{B}, Name "Parameters/Equator ellipse minor axis"];')
            cav.write(f'\na = DefineNumber[{a}, Name "Parameters/Iris ellipse major axis"];')
            cav.write(f'\nb = DefineNumber[{b}, Name "Parameters/Iris ellipse minor axis"];')
            cav.write(f'\nRi = DefineNumber[{Ri}, Name "Parameters/Iris radius"];')
            cav.write(f'\nL = DefineNumber[{L}, Name "Parameters/Half cell length"];')
            cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Equator radius"];\n')

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])

            pt = [-shift, 'Ri']

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pt = [L_bp_l - shift, 'Ri']

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

            df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp_l, tangent_check=tangent_check)
            x1, y1, x2, y2 = df[0]
            if not ignore_degenerate:
                msg = df[-2]
                if msg != 1:
                    error('Parameter set leads to degenerate geometry.')
                    # save figure of error
                    return

            start_pt = pt
            center_pt = [L_bp_l - shift, 'Ri + b']
            majax_pt = [f'{L_bp_l - shift} + a', 'Ri + b']
            end_pt = [-shift + x1, y1]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L_bp_l - shift, Ri + b, a, b, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            # geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pt = [-shift + x2, y2]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

            start_pt = pt
            center_pt = [f'L + {L_bp_l - shift}', 'Req - B']
            majax_pt = [f'L + {L_bp_l - shift} - A', 'Req - B']
            end_pt = [f'{L_bp_l - shift} + L', 'Req']
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L + L_bp_l - shift, Req - B, A, B, step, pt, [L_bp_l + L - shift, Req])
            pt = [f'{L_bp_l - shift} + L', 'Req']

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_l + L) / 2

            # END PATH
            # pts = lineTo(pt, [L+ L_bp_l + - shift, 0],
            #              step)  # to add beam pipe to right

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [L + L_bp_l - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            geo.append([pt[1], pt[0], 2])

            pmcs = [1]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

    def get_geometric_parameters(self):
        parameter_names = ["A", "B", "a", "b", "Ri", "L", "Req"]
        shape_keys = {"IC": 'm', "OC": 'el', "OC_R": 'er'}

        for key in shape_keys.keys():
            values = self.shape[key]
            for name, value in zip(parameter_names, values):
                self.parameters[f"{name}_{shape_keys[key]}"] = value


class SplineCavity(Cavity):
    def __init__(self, shape, name='SplineCavity', kind='Berzier'):
        # self.shape_space = {
        #     'IC': [L, Req, Ri, S, L_bp],
        #     'BP': beampipe
        # }
        super().__init__(name)
        self.self_dir = None
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name

        if 'n_cells' in shape.keys():
            n_cells = shape['n_cells']
        else:
            self.n_cells = 1

        if 'beampipe' in shape.keys():
            beampipe = shape['beampipe']
        else:
            self.beampipe = 'none'

        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = kind
        self.n_cells = 1
        if 'n_cells' in shape.keys():
            self.n_cells = shape['n_cells']

        self.shape = {
            "geometry": shape['geometry'],
            'BP': 'none',
            'CELL PARAMETERISATION': self.cell_parameterisation,
            'kind': self.kind}

        self.shape_multicell = {'kind': self.kind}

        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            # define different paths for easier reference later
            self.eigenmode_dir = os.path.join(self.self_dir, 'eigenmode')
            self.wakefield_dir = os.path.join(self.self_dir, 'wakefield')
            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(self.self_dir, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters, n_cells, beampipe,
                                write=self.geo_filepath)

    def get_geometric_parameters(self):
        self.parameters = self.shape['geometry']

    def write_geometry(self, parameters, n_cells=1, beampipe='none', write=None):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        parameters = np.array(list(parameters.values()))*1e-3

        Ri_el = parameters[0][0]
        L_m = abs(parameters[0][1] - parameters[-1][1])

        step = 0.0005
        shift = 0

        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # SHIFT POINT TO START POINT
            start_point = [shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)

            pt = [-shift, Ri_el]
            pt_indx = add_point(cav, parameters[0], pt_indx)

            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # cavity cell surface
            if self.kind == 'BSpline':
                pt_indx, curve_indx = add_bspline(cav, pt_indx, curve_indx, parameters[1:], n_cells)
                curve.append(curve_indx)
            if self.kind == 'Bezier':
                for i in range(n_cells):
                    pt_indx, curve_indx = add_bezierspline(cav, pt_indx, curve_indx, parameters[1:])
                    curve.append(curve_indx)

                    if i < n_cells-1:
                        parameters[:, 0] += parameters[-1][0]


            # right iris
            pt_indx = add_point(cav, [parameters[-1][0], 0], pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

    def write_quarter_geometry(self, parameters, bp='none', bc=None, tangent_check=False,
                                          ignore_degenerate=False, plot=False, write=None, dimension=False,
                                          contour=False, **kwargs):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        GEO = """
        """

        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        if bp == 'none':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)
        elif bp == 'left':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_el']*1e-3 for n in names)
        elif bp == 'right':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_er']*1e-3 for n in names)
        else:
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)

        L_bp = 4 * L
        if dimension or contour:
            L_bp = 1 * L

        if bp == 'left' or bp == 'right':
            L_bp_l = L_bp
        else:
            L_bp_l = 0

        step = 0.0005

        # calculate shift
        shift = (L_bp_l + L) / 2

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # define parameters
            cav.write(f'\nA = DefineNumber[{A}, Name "Parameters/Equator ellipse major axis"];')
            cav.write(f'\nB = DefineNumber[{B}, Name "Parameters/Equator ellipse minor axis"];')
            cav.write(f'\na = DefineNumber[{a}, Name "Parameters/Iris ellipse major axis"];')
            cav.write(f'\nb = DefineNumber[{b}, Name "Parameters/Iris ellipse minor axis"];')
            cav.write(f'\nRi = DefineNumber[{Ri}, Name "Parameters/Iris radius"];')
            cav.write(f'\nL = DefineNumber[{L}, Name "Parameters/Half cell length"];')
            cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Equator radius"];\n')

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])

            pt = [-shift, 'Ri']

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pt = [L_bp_l - shift, 'Ri']

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

            df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp_l, tangent_check=tangent_check)
            x1, y1, x2, y2 = df[0]
            if not ignore_degenerate:
                msg = df[-2]
                if msg != 1:
                    error('Parameter set leads to degenerate geometry.')
                    # save figure of error
                    return

            start_pt = pt
            center_pt = [L_bp_l - shift, 'Ri + b']
            majax_pt = [f'{L_bp_l - shift} + a', 'Ri + b']
            end_pt = [-shift + x1, y1]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L_bp_l - shift, Ri + b, a, b, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            # geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pt = [-shift + x2, y2]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

            start_pt = pt
            center_pt = [f'L + {L_bp_l - shift}', 'Req - B']
            majax_pt = [f'L + {L_bp_l - shift} - A', 'Req - B']
            end_pt = [f'{L_bp_l - shift} + L', 'Req']
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L + L_bp_l - shift, Req - B, A, B, step, pt, [L_bp_l + L - shift, Req])
            pt = [f'{L_bp_l - shift} + L', 'Req']

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_l + L) / 2

            # END PATH
            # pts = lineTo(pt, [L+ L_bp_l + - shift, 0],
            #              step)  # to add beam pipe to right

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [L + L_bp_l - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            geo.append([pt[1], pt[0], 2])

            pmcs = [1]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')


class EllipticalCavityFlatTop(Cavity):
    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 color='k', plot_label=None):
        """
        All of the old “dimension‐based” logic has been moved here.
        Assumes self.kind, self.name, self.color, etc. are already set.
        """
        super().__init__()

        self.projectDir = None
        self.plot_label = plot_label
        self.name = name
        self.kind = 'elliptical cavity flat top'
        self.beampipe = beampipe
        self.color = color

        self.n_cells = n_cells
        self.cell_parameterisation = 'flattop'

        # Basic counters / containers
        self.n_modes = (n_cells + 1) if n_cells is not None else None
        self.n_modules = 1
        self.no_of_modules = 1

        # Handle the special case: mid_cell can be a dict with keys OC, OC_R, IC
        if isinstance(mid_cell, dict):
            end_cell_left = mid_cell['OC']
            end_cell_right = mid_cell['OC_R']
            mid_cell = mid_cell['IC']

        # Must have at least length = 8 in each cell‐parameter list
        assert mid_cell is not None and len(mid_cell) > 7, \
            ValueError(
                "Flattop cavity mid‐cells require at least 8 input parameters, with the 8th representing length (l).")
        if end_cell_left is not None:
            assert len(end_cell_left) > 7, \
                ValueError(
                    "Flattop cavity left end‐cells require at least 8 input parameters, with the 8th representing "
                    "length (l).")
        if end_cell_right is not None:
            assert len(end_cell_right) > 7, \
                ValueError(
                    "Flattop cavity right end‐cells require at least 8 input parameters, with the 8th representing "
                    "length (l).")

        # Truncate or pad to exactly 8 elements
        self.mid_cell = np.array(mid_cell)[:8]
        self.end_cell_left = np.array(end_cell_left)[:8] if end_cell_left is not None else self.mid_cell
        self.end_cell_right = np.array(end_cell_right)[:8] if end_cell_right is not None else self.mid_cell

        # Ensure end_cell_left / end_cell_right exist
        if end_cell_left is None:
            self.end_cell_left = self.mid_cell
        if end_cell_right is None:
            self.end_cell_right = self.end_cell_left

        # Unpack
        (self.A, self.B, self.a, self.b,
         self.Ri, self.L, self.Req, self.l) = self.mid_cell[:8]

        (self.A_el, self.B_el, self.a_el, self.b_el,
         self.Ri_el, self.L_el, self.Req_el, self.l_el) = self.end_cell_left[:8]

        (self.A_er, self.B_er, self.a_er, self.b_er,
         self.Ri_er, self.L_er, self.Req_er, self.l_er) = self.end_cell_right[:8]

        # Active length & cavity length
        self.l_active = (
                                2 * (self.n_cells - 1) * self.L +
                                (self.n_cells - 2) * self.l +
                                self.L_el + self.l_el +
                                self.L_er + self.l_er
                        ) * 1e-3
        self.l_cavity = self.l_active + 8 * (self.L + self.l) * 1e-3

        # Build self.shape dictionary
        self.shape = {
            "IC": update_alpha(self.mid_cell[:8], self.cell_parameterisation),
            "OC": update_alpha(self.end_cell_left[:8], self.cell_parameterisation),
            "OC_R": update_alpha(self.end_cell_right[:8], self.cell_parameterisation),
            "BP": beampipe,
            "n_cells": self.n_cells,
            "CELL PARAMETERISATION": self.cell_parameterisation,
            "kind": self.kind,
            "geo_file": None
        }

        # Produce multicell representation
        self.to_multicell()
        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, tune=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)
            self.geo_filepath = os.path.join(self.projectDir, 'Cavities', self.name, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters, n_cells, beampipe,
                                write=self.geo_filepath)

    def write_geometry(self, parameters, n_cells, BP, scale=1, ax=None, bc=None, tangent_check=False,
                       ignore_degenerate=False, plot=False, write=None, dimension=False,
                       contour=False, **kwargs):
        """
        Write cavity geometry

        Parameters
        ----------
        BP
        OC_R
        OC
        ignore_degenerate
        file_path: str
            File path to write geometry to
        n_cell: int
            Number of cavity cells
        mid_cell: list, ndarray
            Array of cavity middle cells' geometric parameters
        end_cell_left: list, ndarray
            Array of cavity left end cell's geometric parameters
        end_cell_right: list, ndarray
            Array of cavity left end cell's geometric parameters
        beampipe: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        plot: bool
            If True, the cavity geometry is plotted for viewing

        Returns
        -------

        """

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_aspect('equal')

        names = ['A_m', 'B_m', 'a_m', 'b_m', 'Ri_m', 'L_m', 'Req_m', 'l_m',
                 'A_el', 'B_el', 'a_el', 'b_el', 'Ri_el', 'L_el', 'Req_el', 'l_el',
                 'A_er', 'B_er', 'a_er', 'b_er', 'Ri_er', 'L_er', 'Req_er', 'l_er']

        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, l, \
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, l_el, \
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, l_er = (parameters[n]*1e-3 for n in names)
        Req = Req_m

        step = 0.005

        L_bp = 4 * L_m
        if dimension or contour:
            L_bp = 1 * L_m

        if BP.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif BP.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif BP.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        # calculate shift
        shift = (L_bp_r + L_bp_l + L_el + (n_cells - 1) * 2 * L_m + L_er + (n_cells - 2) * l + l_el + l_er) / 2

        # calculate angles outside loop
        # CALCULATE x1_el, y1_el, x2_el, y2_el
        df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, l_el / 2, tangent_check=tangent_check)
        x1el, y1el, x2el, y2el = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1, y1, x2, y2
        df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, l / 2, tangent_check=tangent_check)
        x1, y1, x2, y2 = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1_er, y1_er, x2_er, y2_er
        df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, l_er / 2, tangent_check=tangent_check)
        x1er, y1er, x2er, y2er = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        geo = []

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        geo.append([start_point[1], start_point[0], 3])

        lineTo(start_point, [-shift, Ri_el], step)
        pt = [-shift, Ri_el]
        geo.append([pt[1], pt[0], 2])

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            lineTo(pt, [L_bp_l - shift, Ri_el], step)
            pt = [L_bp_l - shift, Ri_el]

            geo.append([pt[1], pt[0], 2])

        for n in range(1, n_cells + 1):
            if n == 1:
                # DRAW ARC:
                if plot and dimension:
                    ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                    ellipse = plt.matplotlib.patches.Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                                             height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                                             facecolor='none')
                    ax.add_patch(ellipse)
                    ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                                xytext=(L_bp_l - shift, Ri_el + b_el),
                                arrowprops=dict(arrowstyle='->', color='black'))
                    ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                                xytext=(L_bp_l - shift, Ri_el + b_el),
                                arrowprops=dict(arrowstyle='->', color='black'))

                    ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', va='center', ha='center')
                    ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                            va='center', ha='center', rotation=90)

                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                if plot and dimension:
                    ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                    ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                             height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                             facecolor='none')
                    ax.add_patch(ellipse)
                    ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                                xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                                arrowprops=dict(arrowstyle='<-', color='black'))
                    ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                xytext=(L_el + L_bp_l - shift, Req - B_el),
                                arrowprops=dict(arrowstyle='->', color='black'))

                    ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', va='center',
                            ha='center')
                    ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                            va='center', ha='center', rotation=90)

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_el + l_el - shift, Req], step)
                pt = [L_bp_l + L_el + l_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                if plot and dimension:
                    ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                    # Plot the straight line
                    line_start = [L_bp_l + L_el - shift, Req]
                    line_end = pt
                    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r', zorder=200)

                    ax.annotate('', xy=(line_start[0], line_start[1] + 0.5), xytext=(line_end[0], line_end[1] + 0.5),
                                arrowprops=dict(arrowstyle='<->', color='black'))
                    ax.text((line_start[0] + line_end[0]) / 2, line_start[1] + 0.7,
                            f'{round(l_el, 2)}\n', va='center', ha='center')

                if n_cells == 1:
                    if L_bp_r > 0:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + L_bp_l + l_el - shift, Req - B_er, A_er, B_er, step, pt,
                                    [L_el + l_el + L_er - x2er + + L_bp_l + L_bp_r - shift, y2er])
                        pt = [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse((L_el + l_el + L_bp_l - shift, Req - B_er),
                                                                     width=2 * A_er,
                                                                     height=2 * B_er, angle=0, edgecolor='gray',
                                                                     ls='--',
                                                                     facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_bp_l - shift, Req - B_er),
                                        xytext=(L_el + l_el + L_bp_l - shift + A_er, Req - B_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_bp_l - shift, Req),
                                        xytext=(L_el + l_el + L_bp_l - shift, Req - B_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                    va='center', ha='left', rotation=90)

                        # STRAIGHT LINE TO NEXT POINT
                        lineTo(pt, [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                        pt = [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                        geo.append([pt[1], pt[0], 2])

                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse(
                                (L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                width=2 * a_er,
                                height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er),
                                    f'{round(a_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                    va='center', ha='center', rotation=90)

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])
                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])

                        geo.append([pt[1], pt[0], 2])

                        # calculate new shift
                        shift = shift - (L_el + l_el + L_er)
                    else:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + L_bp_l + l_el - shift, Req - B_er, A_er, B_er, step, pt,
                                    [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                        pt = [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        # STRAIGHT LINE TO NEXT POINT
                        lineTo(pt, [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                        pt = [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                        geo.append([pt[1], pt[0], 2])

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse(
                                (L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                width=2 * a_er,
                                height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er),
                                    f'{round(a_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                    va='center', ha='center', rotation=90)

                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])

                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])
                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_bp_l + L_el + l_el - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_el + l_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                    pt = [L_el + l_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + l_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    pt = [L_el + l_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + l_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_el + l_el + L_m - shift, Ri_m])
                    pt = [L_bp_l + L_el + l_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # calculate new shift
                    shift = shift - (L_el + L_m + l_el)
                    # ic(shift)

            elif n > 1 and n != n_cells:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                for pp in pts:
                    geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + l - shift, Req], step)
                pt = [L_bp_l + L_m + l - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l + l - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_m + L_m + l - x2 + 2 * L_bp_l - shift, y2])
                pt = [L_m + L_m + l - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])

                geo.append([pt[1], pt[0], 2])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m + l - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_m + L_m + l - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + l + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_m + L_m + l - shift, Ri_m])
                pt = [L_bp_l + L_m + L_m + l - shift, Ri_m]

                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # calculate new shift
                shift = shift - 2 * L_m - l
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + l_er - shift, Req], step)
                pt = [L_bp_l + L_m + l_er - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + l_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                            [L_m + L_er + l_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                pt = [L_m + L_er + l_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er + l_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                pt = [L_m + L_er + l_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + l_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er + l_er - shift, Ri_er])
                pt = [L_bp_l + L_m + L_er + l_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

        # BEAM PIPE
        # reset shift

        shift = (L_bp_r + L_bp_l + L_el + l_el + (n_cells - 1) * 2 * L_m + (n_cells - 2) * l + L_er + l_er) / 2
        pts = lineTo(pt, [
            L_bp_r + L_bp_l + 2 * (n_cells - 1) * L_m + (n_cells - 2) * l + l_el + l_er + L_el + L_er - shift,
            Ri_er], step)

        if L_bp_r != 0:
            pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cells - 2) * l + l_el + l_er - shift,
                  Ri_er]
            for pp in pts:
                geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

        # END PATH
        pts = lineTo(pt, [
            2 * (n_cells - 1) * L_m + L_el + L_er + (n_cells - 2) * l + l_el + l_er + L_bp_l + L_bp_r - shift, 0],
                     step)  # to add beam pipe to right
        pt = [2 * (n_cells - 1) * L_m + L_el + L_er + (n_cells - 2) * l + l_el + l_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cells * L_er + L_bp_l - shift, 0], step)
        geo.append([pt[1], pt[0], 2])

        # CLOSE PATH
        lineTo(pt, start_point, step)
        geo.append([pt[1], pt[0], 3])

        # # write geometry
        # if write:
        #     try:
        #         df = pd.DataFrame(geo, columns=['r', 'z'])
        #         df.to_csv(write, sep='\t', index=False)
        #     except FileNotFoundError as e:
        #         error('Check file path:: ', e)

        # append start point
        geo.append([start_point[1], start_point[0], 3])

        if bc:
            # draw right boundary condition
            ax.plot([shift, shift], [-Ri_er, Ri_er],
                    [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                    [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

        # CLOSE PATH
        # lineTo(pt, start_point, step)
        # geo.append([start_point[1], start_point[0], 3])

        geo = np.array(geo)

        if plot:
            if dimension:
                top = ax.plot(geo[:, 1], geo[:, 0], **kwargs)
            else:
                # recenter asymmetric cavity to center
                shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
                if n_cells == 1:
                    shift_to_center = L_er + L_bp_r
                else:
                    shift_to_center = n_cells * L_m + L_bp_r

                top = ax.plot(geo[:, 1] - shift_left + shift_to_center, geo[:, 0], **kwargs)
                # bottom = ax.plot(geo[:, 1] - shift_left + shift_to_center, -geo[:, 0], c=top[0].get_color(), **kwargs)

            # plot legend wthout duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            return ax


    def get_geometric_parameters(self):
        parameter_names = ["A", "B", "a", "b", "Ri", "L", "Req", "l"]
        shape_keys = {"IC": 'm', "OC": 'el', "OC_R": 'er'}

        for key in shape_keys.keys():
            values = self.shape[key]
            for name, value in zip(parameter_names, values):
                self.parameters[f"{name}_{shape_keys[key]}"] = value


class RFGun(Cavity):
    def __init__(self, shape, name='gun'):
        # self.shape_space = {
        #     'IC': [L, Req, Ri, S, L_bp],
        #     'BP': beampipe
        # }
        super().__init__(name)
        self.self_dir = None
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name
        self.n_cells = 1
        self.beampipe = 'none'
        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = 'vhf gun'

        self.shape = {
            "geometry": shape['geometry'],
            'BP': 'none',
            'CELL PARAMETERISATION': self.cell_parameterisation,
            'kind': self.kind}

        self.shape_multicell = {'kind': self.kind}

        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            # define different paths for easier reference later
            self.eigenmode_dir = os.path.join(self.self_dir, 'eigenmode')
            self.wakefield_dir = os.path.join(self.self_dir, 'wakefield')
            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(self.self_dir, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters,
                                write=self.geo_filepath)

    def get_geometric_parameters(self):
        self.parameters = self.shape['geometry']

    def write_geometry(self, parameters, write=None):
        names = [
            "y1", "R2", "T2", "L3", "R4", "L5", "R6", "L7", "R8",
            "T9", "R10", "T10", "L11", "R12", "L13", "R14", "x"
        ]

        y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, \
            T10, L11, R12, L13, R14, x = (parameters[n] for n in names)


        # calcualte R9
        R9 = (((y1 + R2 * np.sin(T2) + L3 * np.cos(T2) + R4 * np.sin(T2) + L5 + R6) -
               (R14 + L13 + R12 * np.sin(T10) + L11 * np.cos(T10) + R10 * np.sin(T10) + x + R8 * (
                       1 - np.sin(T9))))) / np.sin(T9)

        step = 5 * 1e-2
        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)

        with open(write.replace('.n', '.geo'), 'w') as cav:

            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            start_pt = [0, 0]
            pt_indx = add_point(cav, start_pt, pt_indx)

            geo.append([start_pt[1], start_pt[0], 1])

            # DRAW LINE CONNECTING ARCS
            lineTo(start_pt, [0, y1], step)
            pt = [0, y1]
            geo.append([pt[1], pt[0], 0])

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # DRAW ARC:
            start_pt = pt
            center_pt = [-R2, y1]
            majax_pt = [-R2 + R2, y1]
            end_pt = [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(-R2, y1, R2, R2, pt, [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)], 0, T2, step)
            pt = [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)], step)
            pt = [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R4 * np.cos(T2), pt[1] + R4 * np.sin(T2)]
            majax_pt = [pt[0] + R4 * np.cos(T2) + R4, pt[1] + R4 * np.sin(T2)]
            end_pt = [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R4 * np.cos(T2), pt[1] + R4 * np.sin(T2), R4, R4,
            #                  pt, [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)], -(np.pi - T2), np.pi, step)
            pt = [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0], pt[1] + L5], step)
            pt = [pt[0], pt[1] + L5]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R6, pt[1]]
            majax_pt = [pt[0] + R6 + R6, pt[1]]
            end_pt = [pt[0] + R6, pt[1] + R6]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R6, pt[1], R6, R6, pt, [pt[0] + R6, pt[1] + R6], np.pi, np.pi / 2, step)
            pt = [pt[0] + R6, pt[1] + R6]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] + L7, pt[1]], step)
            pt = [pt[0] + L7, pt[1]]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0], pt[1] - R8]
            majax_pt = [pt[0] + R8, pt[1] - R8]
            end_pt = [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0], pt[1] - R8, R8, R8, pt, [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))],
            #                  np.pi / 2, T9, step)
            pt = [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] - R9 * np.cos(T9), pt[1] - R9 * np.sin(T9)]
            majax_pt = [pt[0] - R9 * np.cos(T9) + R9, pt[1] - R9 * np.sin(T9)]
            end_pt = [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] - R9 * np.cos(T9), pt[1] - R9 * np.sin(T9), R9, R9,
            #                  pt, [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)], T9, 0, step)
            pt = [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] - R10, pt[1]]
            majax_pt = [pt[0] - R10 + R10, pt[1]]
            end_pt = [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] - R10, pt[1], R10, R10,
            #                  pt, [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)], 0, -T10, step)
            pt = [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)], step)
            pt = [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R12 * np.cos(T10), pt[1] - R12 * np.sin(T10)]
            majax_pt = [pt[0] + R12 * np.cos(T10) + R12, pt[1] - R12 * np.sin(T10)]
            end_pt = [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R12 * np.cos(T10), pt[1] - R12 * np.sin(T10), R12, R12,
            #                  pt, [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)], (np.pi - T10), np.pi,
            #                  step)
            pt = [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0], pt[1] - L13], step)
            pt = [pt[0], pt[1] - L13]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R14, pt[1]]
            majax_pt = [pt[0] + R14 + R14, pt[1]]
            end_pt = [pt[0] + R14, pt[1] - R14]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R14, pt[1], R14, R14, pt, [pt[0] + R14, pt[1] - R14], np.pi, -np.pi / 2, step)
            pt = [pt[0] + R14, pt[1] - R14]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] + 10 * y1, pt[1]], step)
            pt = [pt[0] + 10 * y1, pt[1]]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 1])

            # line
            lineTo(pt, [pt[0], pt[1] - y1], step)
            pt = [pt[0], pt[1] - x]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            geo.append([pt[1], pt[0], 2])

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

        # start_pt = [0, 0]
        # geo.append([start_pt[1], start_pt[0], 1])

        # pandss
        # df = pd.DataFrame(geo)
        # # print(df[df.duplicated(keep=False)])
        #
        # geo = np.array(geo)
        # _, idx = np.unique(geo[:, 0:2], axis=0, return_index=True)
        # geo = geo[np.sort(idx)]

        # print('length of geometry:: ', len(geo))

        # top = plt.plot(geo[:, 1], geo[:, 0])
        # bottom = plt.plot(geo[:, 1], -geo[:, 0], c=top[0].get_color())

        # # plot legend wthout duplicates
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())
        # plt.gca().set_aspect('equal')
        # plt.xlim(left=-30 * 1e-2)
        #
        # # # write geometry
        # # if write:
        # #     try:
        # #         df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
        # #         # change point data precision
        # #         df['r'] = df['r'].round(8)
        # #         df['z'] = df['z'].round(8)
        # #         # drop duplicates
        # #         df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
        # #         df.to_csv(write, sep='\t', index=False)
        # #     except FileNotFoundError as e:
        # #         error('Check file path:: ', e)
        #
        # return plt.gca()

    def plot(self, what, ax=None, **kwargs):
        # file_path = os.path.join(self.projectDir, "SimulationData", "NGSolveMEVP", self.name, "monopole", "geodata.n")
        if what.lower() == 'geometry':
            ax = self.write_geometry(self.parameters)
            ax.set_xlabel('$z$ [m]')
            ax.set_ylabel(r"$r$ [m]")
            return ax

        if what.lower() == 'zl':
            if ax:
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
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

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        # print('it is here')
        qois = 'qois.json'
        assert os.path.exists(
            os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)), (
            error('Eigenmode result does not exist, please run eigenmode simulation.'))
        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
            self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        self.freq = self.eigenmode_qois['freq [MHz]']
        self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        self.GR_Q = self.eigenmode_qois['GR/Q [Ohm^2]']
        self.G = self.GR_Q / self.R_Q
        self.Q = self.eigenmode_qois['Q []']
        self.e = self.eigenmode_qois['Epk/Eacc []']
        self.b = self.eigenmode_qois['Bpk/Eacc [mT/MV/m]']
        self.Epk_Eacc = self.e
        self.Bpk_Eacc = self.b

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')

            ax.legend(loc="upper right")
            if show_min_max:
                minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(10000 * (maxz - minz)) / 50, width=100)
                Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                ax.plot(self.Ez_0_abs['z(0, 0)'][peaks], Ez_0_abs_peaks, marker='o', ls='')
                ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                ax.axhline(max(Ez_0_abs_peaks), c='k')
        else:
            if os.path.exists(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                           'Ez_0_abs.csv')):
                with open(os.path.join(self.projectDir, 'SimulationData', 'NGSolveMEVP', self.name, 'monopole',
                                       'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
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

    def __str__(self):
        p = dict()
        # p[self.name] = {
        #     'tune': self.tune_results,
        #     'fm': self.eigenmode_qois,
        #     'hom': self.wakefield_qois,
        #     'uq': {
        #         'tune': 0,
        #         'fm': self.uq_fm_results,
        #         'hom': self.uq_hom_results
        #     }
        # }
        return fr"{json.dumps(p, indent=4)}"


class Pillbox(Cavity):
    def __init__(self, n_cells, dims, beampipe='none'):
        super().__init__(n_cells, beampipe)

        L, Req, Ri, S, L_bp = dims
        self.n_cells = n_cells
        self.n_modes = n_cells + 1
        self.kind = 'pillbox'
        self.n_modules = 1  # change later to enable module run
        self.L = L
        self.Req = Req
        self.Ri = Ri
        self.S = S
        self.L_bp = L_bp
        self.beampipe = beampipe
        self.bc = 33
        self.cell_parameterisation = 'simplecell'

        self.shape = {
            'IC': [L, Req, Ri, S, L_bp],
            'BP': beampipe
        }
        self.shape_multicell = None

        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))


            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            # define different paths for easier reference later
            self.eigenmode_dir = os.path.join(self.self_dir, 'eigenmode')
            self.wakefield_dir = os.path.join(self.self_dir, 'wakefield')
            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(self.projectDir, 'Cavities', self.name, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)

    def write_geometry(self, parameters, n_cells, beampipe='none', write=None, plot=False, **kwargs):
        """

        Parameters
        ----------
        file_path
        n_cell
        cell_par
        beampipe
        plot
        kwargs

        Returns
        -------

        """

        names = ['L', 'Req', 'Ri', 'S', 'L_bp']
        L, Req, Ri, S, L_bp = (parameters[n]*1e-3 for n in names)

        step = 0.001

        if beampipe.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif beampipe.lower() == 'none':
            L_bp_l = 0.000
            L_bp_r = 0.000
        elif beampipe.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif beampipe.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            shift = (L_bp_l + L_bp_r + n_cells * L + (n_cells - 1) * S) / 2

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            # cav.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # lineTo(start_point, [-shift, Ri], step)
            pt = [-shift, Ri]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)
            # cav.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # add beampipe
            if L_bp_l > 0:
                # lineTo(pt, [-shift + L_bp_l, Ri], step)
                pt = [-shift + L_bp_l, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            for n in range(1, n_cells + 1):
                if n == 1:
                    # lineTo(pt, [-shift + L_bp_l, Req], step)
                    pt = [-shift + L_bp_l, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Req], step)
                    pt = [-shift + L_bp_l + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Ri], step)
                    pt = [-shift + L_bp_l + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L
                elif n > 1:
                    # lineTo(pt, [-shift + L_bp_l + S, Ri], step)
                    pt = [-shift + L_bp_l + S, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S, Req], step)
                    pt = [-shift + L_bp_l + S, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Req], step)
                    pt = [-shift + L_bp_l + S + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Ri], step)
                    pt = [-shift + L_bp_l + S + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L

            if L_bp_r > 0:
                # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri], step)
                pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # END PATH
            # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0], step)
            pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

    def get_geometric_parameters(self):
        self.parameters = {
            'Ri': self.Ri,
            'L': self.L,
            'Req': self.Req,
            'S': self.S,
            'L_bp': self.L_bp
        }

    def plot(self, what, ax=None, **kwargs):
        if what.lower() == 'geometry':
            ax = plot_pillbox_geometry(self.n_cells, self.L, self.Req, self.Ri, self.S, self.L_bp, self.beampipe)
            ax.set_xlabel('$z$ [m]')
            ax.set_ylabel(r"$r$ [m]")
            return ax

        if what.lower() == 'zl':
            if ax:
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
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

    def run_tune(self, tune_variable, cell_type='Mid Cell', freq=None, solver='SLANS', proc=0, resume=False, n_cells=1):
        """
        Tune current cavity geometry

        Parameters
        ----------
        n_cells: int
            Number of cells used for tuning.
        resume: bool
            Option to resume tuning or not. Only for shape space with multiple entries.
        proc: int
            Processor number
        solver: {'SLANS', 'Native'}
            Solver to be used. Native solver is still under development. Results are not as accurate as that of SLANS.
        freq: float
            Reference frequency in MHz
        cell_type: {'mid cell', 'end-mid cell', 'mid-end cell', 'end-end cell', 'single cell'}
            Type of cell to tune
        tune_variable: {'Req', 'L'}
            Tune variable. Currently supports only the tuning of the equator radius ``Req`` and half-cell length ``L``

        Returns
        -------

        """

        iter_set = ['Linear Interpolation', TUNE_ACCURACY, 10]

        if freq is None:
            # calculate freq from mid cell length
            beta = 1
            freq = beta * c0 / (4 * self.mid_cell[5])
            info("Calculated freq from mid cell half length: ", freq)

        # create new shape space based on cell_type
        # if cell_type.lower() == 'mid cell':
        shape_space = {
            f'{self.name}':
                {
                    'IC': self.shape_space['IC'],
                    'OC': self.shape_space['OC'],
                    'OC_R': self.shape_space['OC_R'],
                    "BP": 'none',
                    'FREQ': freq
                }
        }

        if len(self.slans_tune_res.keys()) != 0:
            run_tune = input("This cavity has already been tuned. Run tune again? (y/N)")
            if run_tune.lower() == 'y':
                self.run_tune_ngsolve(shape_space, resume, proc, self.bc,
                                      SOFTWARE_DIRECTORY, self.projectDir, self.name,
                                      tune_variable, iter_set, cell_type,
                                      progress_list=[], convergence_list=self.convergence_list, n_cells=n_cells)

            # read tune results and update geometry
            try:
                self.get_ngsolve_tune_res(tune_variable, cell_type)
            except FileNotFoundError:
                error("Could not find the tune results. Please run tune again.")
        else:

            self.run_tune_ngsolve(shape_space, resume, proc, self.bc,
                                  SOFTWARE_DIRECTORY, self.projectDir, self.name,
                                  tune_variable, iter_set, cell_type,
                                  progress_list=[], convergence_list=self.convergence_list, n_cells=n_cells)
            try:
                self.get_ngsolve_tune_res(tune_variable, cell_type)
            except FileNotFoundError:
                error("Oops! Something went wrong. Could not find the tune results. Please run tune again.")

    @staticmethod
    def run_tune_ngsolve(shape, resume, p, bc, parentDir, projectDir, filename,
                         tune_variable, iter_set, cell_type, progress_list, convergence_list, n_cells):
        tuner.tune_ngsolve(shape, bc, parentDir, projectDir, filename, resume=resume, proc=p,
                           tune_variable=tune_variable, iter_set=iter_set,
                           cell_type=cell_type, sim_folder='Optimisation',
                           progress_list=progress_list, convergence_list=convergence_list,
                           save_last=True,
                           n_cell_last_run=n_cells)  # last_key=last_key This would have to be tested again #val2

    def run_eigenmode(self, solver='ngsolve', freq_shift=0, boundary_cond=None, subdir='',
                      uq_config=None):
        """
        Run eigenmode analysis on cavity

        Parameters
        ----------
        solver: {'SLANS', 'NGSolve'}
            Solver to be used. Native solver is still under development. Results are not as accurate as that of SLANS.
        freq_shift:
            Frequency shift. Eigenmode solver searches for eigenfrequencies around this value
        boundary_cond: int
            Boundary condition of left and right cell/beampipe ends
        subdir: str
            Sub directory to save results to
        uq_config: None | dict
            Provides inputs required for uncertainty quantification. Default is None and disables uncertainty quantification.

        Returns
        -------

        """

        if boundary_cond:
            self.bc = boundary_cond

        self._run_ngsolve(self.name, self.n_cells, self.n_modules, self.shape_space, self.n_modes, freq_shift,
                          self.bc, SOFTWARE_DIRECTORY, self.projectDir, sub_dir='', uq_config=uq_config)
        # load quantities of interest
        try:
            self.get_eigenmode_qois()
        except FileNotFoundError as e:
            error(f"Could not find eigenmode results. Please rerun eigenmode analysis:: {e}")

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

        """

        if operating_points is None:
            wp_dict = {}
        exist = False

        if not exist:
            if solver == 'ABCI':
                self._run_abci(self.name, self.n_cells, self.n_modules, self.shape_space,
                               MROT=MROT, MT=MT, NFS=NFS, UBT=wakelength, bunch_length=bunch_length,
                               DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG,
                               parentDir=SOFTWARE_DIRECTORY, projectDir=self.projectDir, WG_M=WG_M, marker=marker,
                               operating_points=operating_points, freq=self.freq, R_Q=self.R_Q)

                try:
                    self.get_abci_data()
                    self.get_wakefield_qois()
                except FileNotFoundError:
                    error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

        else:
            try:
                self.get_abci_data()
                self.get_wakefield_qois()
            except FileNotFoundError:
                error("Could not find the abci wakefield results. Please rerun wakefield analysis.")

    @staticmethod
    def _run_ngsolve(name, n_cells, n_modules, shape, n_modes, f_shift, bc, parentDir, projectDir, sub_dir='',
                     uq_config=None):
        start_time = time.time()
        # create folders for all keys
        ngsolve_mevp.createFolder(name, projectDir, subdir=sub_dir)
        ngsolve_mevp.pillbox(n_cells, n_modules, shape['IC'],
                             n_modes=n_modes, fid=f"{name}", f_shift=f_shift, bc=bc, beampipes=shape['BP'],
                             parentDir=parentDir, projectDir=projectDir, subdir=sub_dir)
        # run UQ
        if uq_config:
            uq(name, shape, ["freq", "R/Q", "Epk/Eacc", "Bpk/Eacc"],
               n_cells=n_cells, n_modules=n_modules, n_modes=n_modes,
               f_shift=f_shift, bc=bc, parentDir=parentDir, projectDir=projectDir)

        done(f'Done with Cavity {name}. Time: {time.time() - start_time}')

    @staticmethod
    def _run_abci(name, n_cells, n_modules, shape, MROT=0, MT=4.0, NFS=10000, UBT=50.0, bunch_length=20.0,
                  DDR_SIG=0.1, DDZ_SIG=0.1,
                  parentDir=None, projectDir=None,
                  WG_M=None, marker='', operating_points=None, freq=0, R_Q=0):

        # run abci code
        if WG_M is None:
            WG_M = ['']

        start_time = time.time()
        # run both polarizations if MROT == 2
        for ii in WG_M:
            try:
                if MROT == 2:
                    for m in range(2):
                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                         fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                         projectDir=projectDir,
                                         WG_M=ii, marker=ii)
                else:
                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                     fid=name, MROT=MROT, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir, projectDir=projectDir,
                                     WG_M=ii, marker=ii)
            except KeyError:
                if MROT == 2:
                    for m in range(2):
                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                         fid=name, MROT=m, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                         projectDir=projectDir,
                                         WG_M=ii, marker=ii)
                else:
                    abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                     fid=name, MROT=MROT, MT=MT, NFS=NFS, UBT=UBT, bunch_length=bunch_length,
                                     DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir, projectDir=projectDir,
                                     WG_M=ii, marker=ii)

        done(f'Cavity {name}. Time: {time.time() - start_time}')
        if len(operating_points.keys()) > 0:
            try:
                if freq != 0 and R_Q != 0:
                    d = {}
                    # save qois
                    for key, vals in operating_points.items():
                        WP = key
                        I0 = float(vals['I0 [mA]'])
                        Nb = float(vals['Nb [1e11]'])
                        sigma_z = [float(vals["sigma_SR [mm]"]), float(vals["sigma_BS [mm]"])]
                        bl_diff = ['SR', 'BS']

                        # info("Running wakefield analysis for given operating points.")
                        for i, s in enumerate(sigma_z):
                            for ii in WG_M:
                                fid = f"{WP}_{bl_diff[i]}_{s}mm{ii}"
                                try:
                                    for m in range(2):
                                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC_R'],
                                                         fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                         bunch_length=s,
                                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                                         projectDir=projectDir,
                                                         WG_M=ii, marker=ii, sub_dir=f"{name}")
                                except KeyError:
                                    for m in range(2):
                                        abci_geom.cavity(n_cells, n_modules, shape['IC'], shape['OC'], shape['OC'],
                                                         fid=fid, MROT=m, MT=MT, NFS=NFS, UBT=10 * s * 1e-3,
                                                         bunch_length=s,
                                                         DDR_SIG=DDR_SIG, DDZ_SIG=DDZ_SIG, parentDir=parentDir,
                                                         projectDir=projectDir,
                                                         WG_M=ii, marker=ii, sub_dir=f"{name}")

                                dirc = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                                # try:
                                k_loss = abs(ABCIData(dirc, f'{fid}', 0).loss_factor['Longitudinal'])
                                k_kick = abs(ABCIData(dirc, f'{fid}', 1).loss_factor['Transverse'])
                                # except:
                                #     k_loss = 0
                                #     k_kick = 0

                                d[fid] = get_qois_value(freq, R_Q, k_loss, k_kick, s, I0, Nb, n_cells)

                    # save qoi dictionary
                    run_save_directory = os.path.join(projectDir, "SimulationData", "ABCI", name, marker)
                    with open(os.path.join(run_save_directory, "qois.json"), "w") as f:
                        json.dump(d, f, indent=4, separators=(',', ': '))

                    done("Done with the secondary analysis for working points")
                else:
                    info("To run analysis for working points, eigenmode simulation has to be run first"
                         "to obtain the cavity operating frequency and R/Q")
            except KeyError:
                error('The working point entered is not valid. See below for the proper input structure.')
                show_valid_operating_point_structure()


class Dakota:
    def __init__(self, folder, name, scripts_folder=None):
        self.nodes = None
        self.sim_results = None

        assert f'/' in folder, error('Please ensure directory paths use forward slashes.')
        self.projectDir = folder
        if scripts_folder is None:
            self.scripts_folder = r'D:/Dropbox/CavityDesignHub/analysis_modules/uq/dakota_scripts'
        else:
            self.scripts_folder = scripts_folder
        self.name = name

    def write_input_file(self, **kwargs):
        keys = kwargs.keys()
        assert 'variables_config' in keys, error('please enter keyword "variables config"')
        assert 'interface_config' in keys, error('please enter keyword "interface config"')
        assert 'method_config' in keys, error('please enter keyword "method config"')

        variables_config = kwargs['variables_config']
        interface_config = kwargs['interface_config']
        method_config = kwargs['method_config']

        # check if folder exists, if not, create folder
        if not os.path.exists(os.path.join(self.projectDir, self.name)):
            try:
                os.mkdir(os.path.join(self.projectDir, self.name))
            except FileExistsError:
                error("Could not create folder. Make sure target location exists.")

        with open(os.path.join(self.projectDir, self.name, f'{self.name}.in'), 'w') as f:
            self.environment(f)
            self.method(f, **method_config)
            self.variables(f, **variables_config)
            self.interface(f, **interface_config)

    def environment(self, f):
        f.write('environment\n')
        f.write('\ttabular_data\n')
        f.write("\t\ttabular_data_file = 'sim_result_table.dat'\n")
        f.write('\tresults_output\n')
        f.write("\t\tresults_output_file = 'result_output_file.dat'\n")

        f.write('\n')

    def method(self, f, **kwargs):
        keys = kwargs.keys()
        assert 'method' in keys, error('Please enter "method" in "method config".')
        method = kwargs['method']

        f.write('method\n')
        f.write(f'\t{method}\n')
        f.write("\t\texport_expansion_file ='expansion_file.dat'\n")
        f.write("\t\tcubature_integrand = 3\n")
        f.write("\t\tsamples_on_emulator = 10000\n")
        f.write("\t\tseed = 12347\n")
        f.write("\t\tvariance_based_decomp interaction_order = 1\n")

        f.write('\n')

    def variables(self, f, **kwargs):
        """

        Parameters
        ----------
        f: File
            File
        kind: {'uniform_uncertain', 'normal_uncertain', 'beta_uncertain'}
            Type of distribution of the variables
        descriptors: list, ndarray
            Uncertain variable names
        kwargs: kwargs
            Other relevant arguments eg. {means: [], 'std_deviations': [], 'lower_bounds': [], 'upper_bounds': []}


        Returns
        -------

        """

        keys = kwargs.keys()
        assert 'kind' in keys, error('Please enter "kind"')
        assert 'lower_bounds' in keys, error('Please enter keyword "lower_bounds"')
        assert 'upper_bounds' in keys, error('Please enter keyword "upper bounds"')
        kind = kwargs['kind']
        upper_bounds = kwargs['upper_bounds']
        lower_bounds = kwargs['lower_bounds']

        assert len(upper_bounds) == len(lower_bounds), error("Length of upper and lower bounds must be equal.")

        if "descriptors" in keys:
            descriptors = kwargs['descriptors']
        else:
            info('"descriptors" not entered. Using default parameter labelling.')
            descriptors = [f'p{n}' for n in range(len(upper_bounds))]
        assert len(descriptors) == len(kwargs['upper_bounds'])

        f.write("variables\n")
        f.write(f"\t{kind} = {len(descriptors)}\n")
        f.write(
            "\tdescriptors       =   " + '\t\t\t'.join(['"' + descriptor + '"' for descriptor in descriptors]) + '\n')

        if 'means' in kwargs.keys():
            assert len(descriptors) == len(kwargs['means'])
            f.write("\tmeans      =   " + '\t\t\t'.join([str(mean) for mean in kwargs['means']]) + '\n')

        if 'std_deviations' in kwargs.keys():
            assert len(descriptors) == len(kwargs['std_deviations'])
            f.write("\tstd_deviations      =   " + '\t\t\t'.join([str(std) for std in kwargs['std_deviations']]) + '\n')

        if 'lower_bounds' in kwargs.keys():
            f.write("\tlower_bounds      =   " + '\t\t\t'.join([str(lb) for lb in kwargs['lower_bounds']]) + '\n')

        if 'upper_bounds' in kwargs.keys():
            f.write("\tupper_bounds      =   " + '\t\t\t'.join([str(ub) for ub in kwargs['upper_bounds']]) + '\n')

        f.write('\n')

    def interface(self, f, **kwargs):

        keys = kwargs.keys()
        assert 'analysis_driver' in keys, error('please enter keyword "analysis driver"')
        assert 'responses' in keys, error('Please enter "responses"')

        analysis_driver = kwargs['analysis_driver']
        responses = kwargs['responses']

        nodes_only = False
        if 'nodes_only' in keys:
            nodes_only = kwargs['nodes_only']
            responses = ['f1']

        processes = 1
        if 'processes' in keys:
            processes = kwargs['processes']

        f.write("interface\n")
        f.write("#\tcommon options\n")
        f.write("#\tfork\n")
        f.write("\tparameters_file = 'params.in'\n")
        f.write("\tresults_file    = 'results.out'\n")
        f.write(f"\tsystem asynchronous evaluation_concurrency = {processes}\n")
        f.write(f"\tanalysis_driver = '{analysis_driver} {len(responses)} {nodes_only} {self.scripts_folder}'\n")
        f.write("#\tparameters_file = 'params.in'\n")
        f.write("#\tresults_file    = 'results.out'\n")
        f.write("#\tfile_tag\n")
        f.write("#\tfile_save\n")
        f.write("#\taprepro\n")
        f.write('\n')

        self.responses(f, responses)

    def responses(self, f, responses):
        f.write("responses\n")
        f.write(f"\tresponse_functions = {len(responses)}\n")
        f.write("\tno_gradients\n")
        f.write("\tno_hessians\n")

        f.write('\n')

    def nodes_to_cst_sweep_input(self, partitions=1):
        # save parts
        row_partition = len(self.nodes.index) // partitions
        for i in range(partitions):
            if i < partitions - 1:
                df_part = self.nodes.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = self.nodes.loc[i * row_partition:]

            df_part.to_csv(fr"{self.projectDir}/{self.name}/cst_sweep_files/cst_par_in_{i + 1}.txt", sep="\t",
                           index=None)

    def run_analysis(self, write_cst=True, partitions=1):
        cwd = os.path.join(self.projectDir, self.name)
        dakota_in = f'{os.path.join(self.projectDir, self.name, f"{self.name}.in")}'
        dakota_out = f'{os.path.join(self.projectDir, self.name, f"{self.name}.out")}'

        subprocess.run(['dakota', '-i', dakota_in, '-o', dakota_out], cwd=cwd, shell=True)

        # read results
        filepath = fr"{self.projectDir}/{self.name}/sim_result_table.dat"
        self.sim_results = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        self.nodes = self.sim_results.drop(self.sim_results.filter(regex='response|interface|eval_id').columns, axis=1)
        self.sim_results.to_excel(fr"{self.projectDir}/{self.name}/nodes.xlsx", index=False)

        if write_cst:
            # check if folder exist and clear
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
            else:
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

            # post processes
            self.nodes_to_cst_sweep_input(partitions)
        else:
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

    def plot_sobol_indices(self, filepath, objectives, which=None, kind='stacked', orientation='vertical',
                           normalise=True,
                           group=None, reorder_index=None,
                           selection_index=None, figsize=None):

        if figsize is None:
            figsize = (8, 4)

        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall("\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall("\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []

        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=['main', 'total', 'vars'])
            df = df.astype({'main': 'float', 'total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        df_merge_interaction = pd.DataFrame(columns=['interaction', 'var1', 'var2'])
        # ic(result_interaction.items())
        for i, (k, v) in enumerate(result_interaction.items()):
            df = pd.DataFrame(v, columns=['interaction', 'var1', 'var2'])
            df = df.astype({'interaction': 'float'})
            if i == 0:
                df_merge_interaction = df
            else:
                # df_merge_interaction = pd.merge(df_merge_interaction, df, on=['var1', 'var2'])
                pass
            # ic(df_merge_interaction)

            # combine var columns
            # df_merge_interaction["vars"] = df_merge_interaction[["var1", "var2"]].agg(','.join, axis=1)
            # df.plot.bar(x='var', y='Main')

        # ic(df_merge)

        # group columns
        if group:
            df_merge_T = df_merge.T
            df_merge_T.columns = df_merge_T.loc['vars']
            df_merge_T = df_merge_T.drop('vars')
            for g in group:
                df_merge_T[','.join(g)] = df_merge_T[g].sum(axis=1)

                # drop columns
                df_merge_T = df_merge_T.drop(g, axis=1)

            # reorder index
            if reorder_index:
                df_merge_T = df_merge_T[reorder_index]
            df_merge = df_merge_T.T.reset_index()
            # ic(df_merge)

        # ic(df_merge_interaction)

        if normalise:
            # normalise dataframe columns
            for column in df_merge.columns:
                if 'main' in column or 'total' in column:
                    df_merge[column] = df_merge[column].abs() / df_merge[column].abs().sum()

        # ic(df_merge)
        # filter df
        for w in which:
            if w.lower() == 'main' or w.lower() == 'total':
                dff = df_merge.filter(regex=f'{w.lower()}|vars')
            else:
                # create new column which is a combination of the two variable names
                dff = df_merge_interaction.filter(regex=f'{w.lower()}|vars')
                if not dff.empty:
                    dff['vars'] = df_merge_interaction[['var1', 'var2']].apply(lambda x: '_'.join(x), axis=1)

            cmap = 'tab20'

            if not dff.empty:
                if kind.lower() == 'stacked':
                    dff_T = dff.set_index('vars').T
                    if orientation == 'vertical':
                        ax = dff_T.plot.bar(stacked=True, rot=0, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff_T.plot.barh(stacked=True, rot=0, edgecolor='k', figsize=(8, 4))  # , cmap=cmap
                        ax.invert_yaxis()
                        # for bars in ax.containers:
                        #     ax.bar_label(bars, fmt='%.2f', label_type='center', color='white', fontsize=5)

                        ax.set_xlim(left=0)
                        ax.set_yticklabels(objectives)
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')
                else:
                    if orientation == 'vertical':
                        ax = dff.plot.bar(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axhline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff.plot.barh(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axvline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')

            else:
                error(f"No {w} found.")

    def get_sobol_indices(self, filepath, objectives, which=None, selection_index=None):
        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall("\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall("\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []
        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=[f'{objectives[i].replace("$", "")}_main',
                                          f'{objectives[i].replace("$", "")}_total', 'vars'])
            df = df.astype(
                {f'{objectives[i].replace("$", "")}_main': 'float', f'{objectives[i].replace("$", "")}_total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        return df_merge

    def quadrature_nodes_to_cst_par_input(self, filefolder, n=2):
        filepath = fr"{filefolder}\sim_result_table.dat"
        df = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        df.drop(df.filter(regex='response|interface|eval_id').columns, axis=1, inplace=True)
        df.to_excel(fr"{filefolder}\cubature_nodes_pars.xlsx", index=False)

        # save parts
        row_partition = len(df.index) // n
        for i in range(n):
            if i < n - 1:
                df_part = df.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = df.loc[i * row_partition:]

            df_part.to_csv(fr"{filefolder}\cst_par_in_{i + 1}.txt", sep="\t", index=None)

    def get_pce(self, filefolder):

        filepath = fr"{filefolder}\uq_pce_expansion.dat"
        df = pd.read_csv(filepath, sep='\\s+', header=None)

        poly = 0
        for row in df.iterrows():
            poly += 0

    def quote_to_float(self, value):
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            return float(value[1:-1])
        else:
            return value

    def combine_params_output(self, folder, N):
        for i in range(N):
            if i == 0:
                df = pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)
            else:
                df = pd.concat([df, pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)])

        # rearrange column according to the reference column order
        df_reference = pd.read_excel(fr"{folder}\cubature_nodes_pars.xlsx")
        columns = list(df_reference.columns)
        # check if 3D Run ID in column and drop if yes
        if ' 3D Run ID' in list(df.columns):
            columns.append(' 3D Run ID')

        columns = list(df_reference.columns) + (df.columns.drop(columns).tolist())

        df = df[columns]

        df.to_excel(fr"{folder}\cubature_nodes.xlsx", index=False)

    def plot_sobol(self, config):
        obj = config['obj']
        group = config['group']
        reorder_index = ['reorder_index']
        filefolder = config['folder']
        kind = config['kind']
        normalise = config['normalise']
        which = config['which']
        orientation = config['orientation']
        selection_index = config['selection_index']

        # obj = [r"$Q_\mathrm{ext, FM}$", r"$\max(Q_\mathrm{ext, dip})$"]
        # obj = [r"$S_\mathrm{max}~\mathrm{[dB]}$", r"$S_\mathrm{min}~\mathrm{[dB]}$", r"$f(S_\mathrm{max})~\mathrm{[MHz]}$", r"$f(S_\mathrm{min})~\mathrm{[MHz]}$"]
        # obj =
        # obj = [r"$freq [MHz]$",	fr"$R/Q [Ohm]$", r"$Epk/Eacc []$",	r"$Bpk/Eacc [mT/MV/m]$", 'G', 'kcc', 'ff']
        # obj =
        # obj = [r"$freq [MHz]$", 'kcc']
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', group=group, reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', selection_index=selection_index, normalise=False)#

        self.plot_sobol_indices(fr"{filefolder}\dakota.out", obj, which=which, kind=kind,
                                selection_index=selection_index, orientation=orientation, normalise=normalise)  #


class OperationPoints:
    def __init__(self, filepath=None):
        self.op_points = {}

        if filepath:
            if os.path.exists(filepath):
                self.op_points = self.load_operating_point(filepath)

    def load_operating_point(self, filepath):
        with open(filepath, 'r') as f:
            op_points = json.load(f)

        self.op_points = op_points
        return op_points

    def get_default_operating_points(self):
        self.op_points = pd.DataFrame({
            "Z_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1280,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 56,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 132,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2023": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2023": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1400,
                "V [GV]": 0.12,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 56,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 2.76
            },
            "W_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 135,
                "V [GV]": 1.0,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0801,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 65.99,
                "tau_xy [ms]": 131.98,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 112,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.55,
                "sigma_BS [mm]": 7.02,
                "Nb [1e11]": 2.29
            },
            "H_2022": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 53.4,
                "V [GV]": 2.1,
                "Eacc [MV/m]": 10.61,
                "nu_s []": 0.0328,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 19.6,
                "tau_xy [ms]": 39.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 528,
                "T [K]": 4.5,
                "sigma_SR [mm]": 2.5,
                "sigma_BS [mm]": 4.45,
                "Nb [1e11]": 1.51
            },
            "ttbar_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10,
                "V [GV]": 9.2,
                "Eacc [MV/m]": 20.12,
                "nu_s []": 0.0826,
                "alpha_p [1e-5]": 0.733,
                "tau_z [ms]": 5.63,
                "tau_xy [ms]": 11.26,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 488,
                "T [K]": 2,
                "sigma_SR [mm]": 1.67,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.26
            },
            "Z_booster_2022": {
                "freq [MHz]": 801.58,
                "E [GeV]": 45.6,
                "I0 [mA]": 128,
                "V [GV]": 0.14,
                "Eacc [MV/m]": 6.23,
                "nu_s []": 0.0370,
                "alpha_p [1e-5]": 2.85,
                "tau_z [ms]": 354.91,
                "tau_xy [ms]": 709.82,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 120,
                "T [K]": 4.5,
                "sigma_SR [mm]": 4.32,
                "sigma_BS [mm]": 15.2,
                "Nb [1e11]": 0.276
            },
            "Z_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 45.6,
                "I0 [mA]": 1390,
                "V [GV]": 0.10,
                "Eacc [MV/m]": 5.72,
                "nu_s []": 0.025,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 424.6,
                "tau_xy [ms]": 849.2,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.5,
                "sigma_BS [mm]": 12.1,
                "Nb [1e11]": 1.7
            },
            "W_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 80,
                "I0 [mA]": 147,
                "V [GV]": 0.75,
                "Eacc [MV/m]": 11.91,
                "nu_s []": 0.0506,
                "alpha_p [1e-5]": 1.48,
                "tau_z [ms]": 78.7,
                "tau_xy [ms]": 157.4,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 52,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.0,
                "sigma_BS [mm]": 6.0,
                "Nb [1e11]": 1.5
            },
            "H_2018": {
                "freq [MHz]": 400.79,
                "E [GeV]": 120,
                "I0 [mA]": 29,
                "V [GV]": 2.0,
                "Eacc [MV/m]": 11.87,
                "nu_s []": 0.036,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 23.4,
                "tau_xy [ms]": 46.8,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 136,
                "T [K]": 4.5,
                "sigma_SR [mm]": 3.15,
                "sigma_BS [mm]": 5.3,
                "Nb [1e11]": 1.8
            },
            "ttbar_2018": {
                "freq [MHz]": 801.58,
                "E [GeV]": 182.5,
                "I0 [mA]": 10.8,
                "V [GV]": 10.93,
                "Eacc [MV/m]": 24.72,
                "nu_s []": 0.087,
                "alpha_p [1e-5]": 0.73,
                "tau_z [ms]": 6.8,
                "tau_xy [ms]": 13.6,
                "f_rev [kHz]": 3.07,
                "beta_xy [m]": 50,
                "N_c []": 584,
                "T [K]": 2,
                "sigma_SR [mm]": 1.97,
                "sigma_BS [mm]": 2.54,
                "Nb [1e11]": 2.3
            }
        })


def show_welcome():
    import base64
    filename = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    image_path = os.path.join(filename, 'docs/images/cavsim2d_logo_1x.png')

    # Convert the image to base64
    with open(image_path, "rb") as img_file:
        encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

    # HTML and CSS to display image and colourful text on the same line
    message = f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{encoded_image}" style="height: 32px;">
        <p style="margin: 0; font-size: 16px; 
                  background: -webkit-linear-gradient(left, #EFC3CA, #5DE2E7, #FE9900, #E7DDFF, #FFDE59);
                  -webkit-background-clip: text; 
                  -webkit-text-fill-color: transparent;">
            <b>CAV-SIM-2D</b> loaded successfully!
        </p>
    </div>
    """

    # Display the HTML
    display(HTML(message))
