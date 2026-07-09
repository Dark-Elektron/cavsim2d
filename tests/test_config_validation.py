"""Config validation warns on typos and stays silent on valid configs.

These tests don't need the solver, so they run everywhere."""
import warnings

import pytest

from cavsim2d.utils.config_validation import (
    validate_eigenmode_config, validate_wakefield_config,
    validate_tune_config, validate_optimisation_config, validate_uq_config,
)


def _warnings_for(fn, cfg):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn(cfg)
    return [str(w.message) for w in caught]


def test_unknown_key_warns_with_suggestion():
    msgs = _warnings_for(validate_eigenmode_config,
                         {'processes': 1, 'boundary_condition': 'mm'})
    assert any("boundary_condition" in m and "boundary_conditions" in m for m in msgs)


def test_cell_complexity_footgun_flagged():
    msgs = _warnings_for(validate_uq_config,
                         {'variables': ['A'], 'cell complexity': 'multicell'})
    assert any("cell_complexity" in m for m in msgs)


def test_valid_configs_are_silent():
    assert _warnings_for(validate_eigenmode_config, {
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'polarisation': ['monopole', 'dipole'], 'n_modes': 4,
    }) == []
    assert _warnings_for(validate_eigenmode_config, {
        'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
        'polarisation': 'monopole', 'nmodes': 4,
    }) == []
    assert _warnings_for(validate_wakefield_config, {
        'processes': 1, 'rerun': True, 'MROT': 'dipole', 'wakelength': 5,
        'bunch_length': 25,
    }) == []
    assert _warnings_for(validate_optimisation_config, {
        'initial_points': 5, 'no_of_generation': 2, 'bounds': {}, 'objectives': [],
        'mutation_factor': 4, 'crossover_factor': 4, 'chaos_factor': 2,
    }) == []


def test_bad_processes_raises():
    with pytest.raises(ValueError, match="greater than zero"):
        validate_eigenmode_config({'processes': 0})
    with pytest.raises(ValueError, match="must be an integer"):
        validate_eigenmode_config({'processes': 1.5})


def test_delta_variables_length_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        validate_uq_config({'variables': ['A', 'B'], 'delta': [0.05]})


def test_nested_config_validated():
    # a typo inside a nested uq_config should be caught
    msgs = _warnings_for(validate_tune_config, {
        'freqs': 801.58, 'cell_type': {'mid-cell': 'Req'},
        'uq_config': {'variables': ['A'], 'objctives': ['R/Q [Ohm]']},
    })
    assert any("objctives" in m for m in msgs)
