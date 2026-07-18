"""Tuning regression: an elliptical mid-cell tunes its equator radius to hit a
target frequency, and the tuned cavity is reachable via ``cav.tuned``."""
import os

import numpy as np
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d import Cavities, EllipticalCavity

_GUN_GEOM = {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
             'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
             'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
             'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}
_SPLINE_GEOM = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
                'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}


def _gun():
    """A fresh shape dict each time — RFGun keeps ``shape['geometry']`` by
    reference, so tuning one would mutate a shared module-level dict."""
    return {'geometry': dict(_GUN_GEOM)}


def test_tune_hits_target_frequency(project_dir):
    target = 801.58
    midcell = np.array(MIDCELL + [0])  # trailing alpha slot
    cav = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
    cavs = Cavities(project_dir)
    cavs.add_cavity(cav, 'tune')

    tune_config = {
        'freqs': target,
        'cell_type': {'mid-cell': 'Req'},
        'processes': 1,
        'rerun': True,
        'eigenmode_config': {'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'},
    }
    cavs.run_tune(tune_config)

    tuned = cav.tuned
    assert tuned is not None, "cav.tuned should be populated after tuning"
    res = cav.tune.qois['mid-cell']
    assert abs(res['FREQ'] - target) < 0.5, res['FREQ']
    # tuning moved the equator radius away from its starting value
    assert res['parameters']['Req_m'] != MIDCELL[6]


# --- three defects that made non-elliptical tuning fail silently -------------

def test_uses_cell_suffixes_is_a_model_capability():
    """The tuner must ask the model, not compare `kind` strings.

    `kind` is 'elliptical cavity flat top' for the flat top, so
    `kind == 'elliptical cavity'` skipped the Req -> Req_m suffixing and the
    tuner then looked up a bare 'Req' that does not exist.
    """
    from cavsim2d import (EllipticalCavity, EllipticalCavityFlatTop, Pillbox,
                                 CircularWaveguide)
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ft = tesla + [20]
    assert EllipticalCavity(1, tesla, tesla, tesla, beampipe='none').uses_cell_suffixes
    assert EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='none').uses_cell_suffixes
    assert not Pillbox(1, [100, 100, 20, 0, 0], beampipe='none').uses_cell_suffixes
    assert not CircularWaveguide(100.0, 150.0).uses_cell_suffixes


def test_printing_survives_a_console_that_cannot_encode_the_message():
    """A U+2248 in a tuner error message raised UnicodeEncodeError on the Windows
    cp1252 console — so a tuning *failure* crashed instead of reporting itself."""
    import io
    import sys
    import cavsim2d.utils.printing as printing

    class CP1252Buffer(io.StringIO):
        encoding = 'cp1252'

    msg = 'Req_m ≈ 171.2 → unreachable'
    old = sys.stdout
    sys.stdout = CP1252Buffer()
    try:
        printing.error(msg)          # must not raise
        written = sys.stdout.getvalue()
    finally:
        sys.stdout = old
    assert 'Req_m' in written and 'unreachable' in written
    assert '≈' not in written   # degraded, not crashed


def test_rebuild_is_the_single_per_type_hook():
    """Tuning, UQ and optimisation all reconstruct a cavity from a parameter dict.
    Every model provides `rebuild`; the base machinery is generic."""
    from cavsim2d import (EllipticalCavity, EllipticalCavityFlatTop, Pillbox,
                                 RFGun, SplineCavity, CircularWaveguide)
    from cavsim2d.models.base import Cavity
    for cls in (EllipticalCavity, EllipticalCavityFlatTop, Pillbox, RFGun,
                SplineCavity, CircularWaveguide):
        assert 'rebuild' in cls.__dict__, f'{cls.__name__} must implement rebuild()'
        # the generic machinery is inherited, not duplicated per type
        assert 'clone_for_tuning' not in cls.__dict__
        assert '_load_tuned_from_disk' not in cls.__dict__
    assert 'rebuild' in Cavity.__dict__


def test_a_model_without_rebuild_says_so():
    """The base hook names itself, instead of blaming the elliptical tuner."""
    from cavsim2d.models.base import Cavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    from cavsim2d import EllipticalCavity
    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    with pytest.raises(NotImplementedError, match='rebuild'):
        Cavity.rebuild(cav, cav.parameters)


def test_rebuild_round_trips_every_model():
    """rebuild(self.parameters) reproduces an equivalent cavity."""
    import numpy as np
    from cavsim2d import (EllipticalCavity, EllipticalCavityFlatTop, Pillbox,
                                 RFGun, SplineCavity, CircularWaveguide)
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ft = tesla + [20]
    geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
            'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}
    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    cavities = [EllipticalCavity(2, tesla, tesla, tesla, beampipe='both'),
                EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both'),
                Pillbox(1, [100, 100, 20, 0, 50], beampipe='both'),
                RFGun(gun),
                SplineCavity({'geometry': dict(geom)}, kind='Bezier'),
                CircularWaveguide(100.0, 150.0)]
    for cav in cavities:
        clone = cav.rebuild(cav.parameters)
        assert type(clone) is type(cav)
        assert clone.n_cells == cav.n_cells
        for k, v in cav.parameters.items():
            assert np.allclose(np.asarray(clone.parameters[k], dtype=float),
                               np.asarray(v, dtype=float)), (type(cav).__name__, k)


def test_flattop_now_tunes(project_dir):
    """Previously a silent no-op: run_tune returned normally with cav.tuned = None."""
    from cavsim2d import Cavities, EllipticalCavityFlatTop
    ft = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20, 20]
    cav = EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both')
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['FT'])
    cavs.run_tune({'freqs': 790.0, 'cell_type': {'mid-cell': 'Req'}, 'processes': 1})
    assert cav.tuned is not None, 'run_tune reported success but tuned nothing'
    assert cav.tuned.parameters['Req_m'] != pytest.approx(171.20)


# --- the tune variable resolver asks the model, not a hardcoded list ---------

def test_every_model_declares_its_own_tune_variables():
    """`tune_variables()` is derived from the model's own parameters, so adding a
    geometry needs no edit to a central whitelist."""
    from cavsim2d import (EllipticalCavity, EllipticalCavityFlatTop, Pillbox,
                                 RFGun, SplineCavity, CircularWaveguide)
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ft = tesla + [20]

    el = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    assert el.tune_variables() == {'A', 'B', 'a', 'b', 'Ri', 'L', 'Req'}
    # the flat top has a flat-top length; a plain elliptical cavity does not.
    assert 'l' in EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='none').tune_variables()
    assert 'l' not in el.tune_variables()

    # non-elliptical models name their own parameters
    assert 'R6' in RFGun(_gun()).tune_variables()
    assert 'S' in Pillbox(1, [100, 100, 20, 0, 0], beampipe='none').tune_variables()
    assert CircularWaveguide(100.0, 150.0).tune_variables() == {'L', 'R'}

    # a spline is parameterised by control points, so each gives two handles
    sc = SplineCavity({'geometry': dict(_SPLINE_GEOM)}, kind='Bezier')
    assert {'p3_z', 'p3_r'} <= sc.tune_variables()
    assert 'p3' not in sc.tune_variables()


def test_unknown_tune_variable_names_the_models_own_parameters():
    from cavsim2d import RFGun
    from cavsim2d.processes.tune import _resolve_suffixed_var
    with pytest.raises(ValueError, match=r"Unknown tune variable 'Req'.*RFGun"):
        _resolve_suffixed_var('Req', 'mid-cell', RFGun(_gun()))
    # and the message lists what it *does* accept
    with pytest.raises(ValueError, match='R6'):
        _resolve_suffixed_var('Req', 'mid-cell', RFGun(_gun()))


def test_a_geometry_without_parameters_says_so():
    """An imported mesh/CAD geometry has no parameters — that is a different
    failure from misspelling a variable, and must not be reported as one."""
    from cavsim2d import Pillbox
    from cavsim2d.processes.tune import _resolve_suffixed_var
    cav = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cav.parameters = {}                       # as if built from a .geo/CAD file
    with pytest.raises(ValueError, match='no tunable parameters'):
        _resolve_suffixed_var('Req', 'mid-cell', cav)


def test_unsuffixed_models_keep_their_full_parameter_name():
    """The old resolver split on '_' to find the bare name, so the pillbox's
    'L_bp' silently resolved to 'L' — tuning the wrong parameter."""
    from cavsim2d import Pillbox, EllipticalCavity
    from cavsim2d.processes.tune import _resolve_suffixed_var
    pb = Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')
    assert _resolve_suffixed_var('L_bp', 'mid-cell', pb) == 'L_bp'
    assert _resolve_suffixed_var('Req', 'mid-cell', pb) == 'Req'      # no '_m'

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    el = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    assert _resolve_suffixed_var('Req', 'mid-cell', el) == 'Req_m'
    assert _resolve_suffixed_var('Req', 'end-cell-right', el) == 'Req_er'
    assert _resolve_suffixed_var('Req_el', 'mid-cell', el) == 'Req_el'


def test_bad_tune_variable_is_rejected_before_any_process_is_spawned():
    from cavsim2d import RFGun
    from cavsim2d.processes.tune import validate_cell_type_config
    with pytest.raises(ValueError, match='Unknown tune variable'):
        validate_cell_type_config({'mid-cell': ['nope']}, [RFGun(_gun())])


def test_expand_variable_asks_the_model_not_substring_matching():
    """UQ expanded a variable by testing `k_var in parameter_key`."""
    from cavsim2d import EllipticalCavity, Pillbox, RFGun, SplineCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    # a bare name on an elliptical cavity means 'this quantity in every cell'
    el = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    assert sorted(el.expand_variable('Req')) == ['Req_el', 'Req_er', 'Req_m']
    assert el.expand_variable('Req_m') == ['Req_m']

    # 'L' must not drag in the pillbox's 'L_bp'
    pb = Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')
    assert pb.expand_variable('L') == ['L']
    assert pb.expand_variable('L_bp') == ['L_bp']

    # a spline coordinate expands to itself (it matched no key, giving k=0
    # random variables and a ZeroDivisionError inside the quadrature)
    sc = SplineCavity({'geometry': dict(_SPLINE_GEOM)}, kind='Bezier')
    assert sc.expand_variable('p3_r') == ['p3_r']

    with pytest.raises(ValueError, match='Unknown variable'):
        RFGun(_gun()).expand_variable('Req')


def test_spline_uq_perturbs_a_control_point_coordinate(project_dir):
    """`k = 0` random variables -> ZeroDivisionError in the Stroud3 quadrature."""
    import json
    from cavsim2d import Cavities, SplineCavity
    cav = SplineCavity({'geometry': dict(_SPLINE_GEOM)}, kind='Bezier')
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['SC'])
    cav.run_eigenmode({'processes': 1, 'rerun': True, 'uq_config': {
        'variables': ['p3_r', 'p2_r'], 'objectives': ['monopole:freq [MHz]'],
        'delta': [0.01, 0.01], 'processes': 1, 'distribution': 'gaussian',
        'method': ['Stroud3']}})
    with open(os.path.join(cav.self_dir, 'uq', 'uq.json')) as fh:
        uq = json.load(fh)
    # the perturbation actually moved the geometry
    assert uq['monopole:freq [MHz]']['stdDev'][0] > 0


def test_perturbation_reaches_a_non_scalar_parameter():
    """apply_perturbation wrote cav.parameters[var] directly, so a variable
    living inside a control point ([z, r]) could not be perturbed at all."""
    from cavsim2d import SplineCavity
    from cavsim2d.utils.shapes import apply_perturbation
    cav = SplineCavity({'geometry': dict(_SPLINE_GEOM)}, kind='Bezier')
    out = apply_perturbation(cav, [[0.10]], ['p3_r'], 'mul')
    perturbed = list(out.values())[0]
    z, r = perturbed.parameters['p3']
    assert z == pytest.approx(85)                 # z untouched
    assert r == pytest.approx(103 * 1.10)         # r scaled
    assert cav.parameters['p3'] == [85, 103]      # base cavity unchanged


def test_qoi_readers_keep_the_base_signature():
    """RFGun.get_eigenmode_qois() narrowed the base's (self, config=None) to
    (self), so every caller that passes a config — UQ does — hit a TypeError."""
    import inspect
    from cavsim2d import (EllipticalCavity, EllipticalCavityFlatTop, Pillbox,
                                 RFGun, SplineCavity, CircularWaveguide)
    from cavsim2d.models.base import Cavity
    base = inspect.signature(Cavity.get_eigenmode_qois)
    for cls in (EllipticalCavity, EllipticalCavityFlatTop, Pillbox, RFGun,
                SplineCavity, CircularWaveguide):
        if 'get_eigenmode_qois' in cls.__dict__:
            assert inspect.signature(cls.get_eigenmode_qois) == base, cls.__name__


def test_rfgun_tunes(project_dir):
    """Blocked by a hardcoded elliptical-only variable whitelist."""
    from cavsim2d import Cavities, RFGun
    cav = RFGun(_gun())
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['GUN'])
    cavs.run_tune({'freqs': 210.0, 'cell_type': {'mid-cell': 'R6'}, 'processes': 1,
                   'rerun': True})
    assert cav.tuned is not None
    assert cav.tuned.parameters['R6'] != pytest.approx(_GUN_GEOM['R6'])
    assert cav.tune.qois['mid-cell']['FREQ'] == pytest.approx(210.0, abs=0.5)


def test_spline_tunes_a_control_point_coordinate(project_dir):
    """A vector-valued parameter survives the round trip through tune_res.json:
    `_load_tuned_from_disk` used float(v), which raised on [z, r] and then
    silently dropped the tuned coordinate."""
    from cavsim2d import Cavities, SplineCavity
    cav = SplineCavity({'geometry': dict(_SPLINE_GEOM)}, kind='Bezier')
    cavs = Cavities(project_dir)
    cavs.add_cavity([cav], ['SC'])
    cavs.run_tune({'freqs': 1488.02, 'cell_type': {'mid-cell': 'p3_r'}, 'processes': 1,
                   'rerun': True})
    assert cav.tuned is not None
    z, r = cav.tuned.parameters['p3']
    assert z == pytest.approx(85)                      # only the r handle moved
    assert r != pytest.approx(103)
    # the tuned geometry really is at the target, not just the recorded number
    cav.tuned.run_eigenmode({'processes': 1, 'rerun': True})
    assert cav.tuned.eigenmode.qois['freq [MHz]'] == pytest.approx(1488.02, abs=0.5)


def test_flattop_create_accepts_the_tuner_mode_keyword(project_dir):
    """Its create() took `tune=`, so the tuner's `mode='tune'` raised a TypeError
    that tune_function reported as 'degenerate geometry'."""
    import inspect
    from cavsim2d import EllipticalCavityFlatTop
    sig = inspect.signature(EllipticalCavityFlatTop.create)
    assert 'mode' in sig.parameters and 'tune' not in sig.parameters
