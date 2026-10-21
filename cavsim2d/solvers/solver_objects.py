"""The solver namespaces, re-exported from their own modules.

Each analysis namespace lives in its own module (``tune_solver``,
``eigenmode_solver``, ``wakefield_solver``, ``multipacting_solver``,
``optimisation_solver``, ``study_namespaces``), the default configs in
``configs``. Importing from here keeps working.
"""
from cavsim2d.solvers.configs import (_ConfigSampleMixin, _documented, _user_config,
                                      _VALIDATORS, CONFIG_NOTES, config_sample, CONFIG_SAMPLES,
                                      ConfigSample, DEFAULT_EIGENMODE_CONFIG,
                                      DEFAULT_MULTIPACTING_CONFIG, DEFAULT_OPTIMISATION_CONFIG,
                                      DEFAULT_TUNE_CONFIG, DEFAULT_UQ_CONFIG,
                                      DEFAULT_WAKEFIELD_CONFIG, EIGENMODE_NOTES, merge_config,
                                      MULTIPACTING_NOTES, OPTIMISATION_NOTES, TUNE_NOTES,
                                      UQ_NOTES, WAKEFIELD_NOTES)
from cavsim2d.solvers._shared import (_accelerating_mode_row, _gaussian_sum, _has_field,
                                      _has_mesh, _maybe_show, _MODE_FILTER_CAVEAT,
                                      _reference_series, _require_vacuum,
                                      _solved_with_open_boundary, _z_axis_label)
from cavsim2d.solvers.tune_solver import TuneSolver
from cavsim2d.solvers.eigenmode_solver import EigenmodeSolver
from cavsim2d.solvers.wakefield_solver import WakefieldSolver
from cavsim2d.solvers.multipacting_solver import MultipactingSolver
from cavsim2d.solvers.optimisation_solver import OptimisationSolver
from cavsim2d.solvers.study_namespaces import _StudyNamespace, StudyEigenmode, StudyWakefield

__all__ = [
    '_accelerating_mode_row', '_ConfigSampleMixin', '_documented', '_gaussian_sum',
    '_has_field', '_has_mesh', '_maybe_show', '_MODE_FILTER_CAVEAT',
    '_reference_series', '_require_vacuum', '_solved_with_open_boundary',
    '_StudyNamespace', '_user_config', '_VALIDATORS', '_z_axis_label', 'CONFIG_NOTES',
    'config_sample', 'CONFIG_SAMPLES', 'ConfigSample', 'DEFAULT_EIGENMODE_CONFIG',
    'DEFAULT_MULTIPACTING_CONFIG', 'DEFAULT_OPTIMISATION_CONFIG', 'DEFAULT_TUNE_CONFIG',
    'DEFAULT_UQ_CONFIG', 'DEFAULT_WAKEFIELD_CONFIG', 'EIGENMODE_NOTES',
    'EigenmodeSolver', 'merge_config', 'MULTIPACTING_NOTES', 'MultipactingSolver',
    'OPTIMISATION_NOTES', 'OptimisationSolver', 'StudyEigenmode', 'StudyWakefield',
    'TUNE_NOTES', 'TuneSolver', 'UQ_NOTES', 'WAKEFIELD_NOTES', 'WakefieldSolver',
]
