"""Polarisation-qualified objective names (P2-2).

Monopole and m-pole `qois.json` share 18 QOI key names, so an objective must say
which polarisation it means. Bare names are rejected rather than silently taken
to mean the monopole.
"""
import json
import os

import pytest

pytest.importorskip("ngsolve")

from cavsim2d.solvers.objectives import (SEP, Objective, parse_objective, parse_objectives,
                                         objective_polarisations, canonical_objective,
                                         is_qualified, read_objective_values)


def test_parse_objective_forms():
    o = parse_objective('monopole:R/Q [Ohm]')
    assert (o.pol, o.mode, o.qoi) == ('monopole', None, 'R/Q [Ohm]')
    assert o.column == 'monopole:R/Q [Ohm]'

    o = parse_objective('dipole:2:freq [MHz]')
    assert (o.pol, o.mode, o.qoi) == ('dipole', 2, 'freq [MHz]')
    assert o.column == 'dipole:2:freq [MHz]'

    # a QOI name may contain anything except a colon
    o = parse_objective('dipole:R/Q_t [Ohm/m^(2(m-1))]')
    assert o.qoi == 'R/Q_t [Ohm/m^(2(m-1))]'

    # azimuthal number instead of a name; whitespace tolerated
    assert parse_objective('1:freq [MHz]').pol == 'dipole'
    assert parse_objective(' quadrupole : 3 : G [Ohm] ').column == 'quadrupole:3:G [Ohm]'


def test_bare_objective_is_rejected():
    """'freq [MHz]' used to silently mean the monopole's — ambiguous once a
    second polarisation is solved."""
    with pytest.raises(ValueError, match='does not name a polarisation'):
        parse_objective('freq [MHz]')


def test_parse_objective_errors():
    with pytest.raises(ValueError, match='unknown polarisation'):
        parse_objective('octopole:freq [MHz]')       # typo of 'octupole'
    with pytest.raises(ValueError, match='1-based'):
        parse_objective('dipole:0:freq [MHz]')
    with pytest.raises(ValueError, match='must be a 1-based mode index'):
        parse_objective('dipole:x:freq [MHz]')
    with pytest.raises(ValueError, match='empty QOI name'):
        parse_objective('dipole:')
    with pytest.raises(ValueError, match='must be a string'):
        parse_objective(5)
    with pytest.raises(ValueError, match='more than once'):
        parse_objectives(['dipole:freq [MHz]', 'dipole:freq [MHz]'])


def test_helpers():
    assert objective_polarisations(['monopole:freq [MHz]', 'dipole:2:R/Q [Ohm]']) == [0, 1]
    assert canonical_objective('1:freq [MHz]') == 'dipole:freq [MHz]'
    assert canonical_objective('ZL') == 'ZL'         # wakefield objective, untouched
    assert is_qualified('ZL') is False
    assert is_qualified('dipole:freq [MHz]') is True
    assert Objective('dipole', None, 'x', 'dipole:x') == parse_objective('dipole:x')


TESLA = [42, 42, 12, 19, 35, 57.7, 103.353]


@pytest.fixture(scope='module')
def solved(tmp_path_factory):
    from cavsim2d.cavity import Cavities, EllipticalCavity
    d = tmp_path_factory.mktemp('obj')
    cavs = Cavities(str(d))
    cav = EllipticalCavity(2, TESLA, TESLA, TESLA, beampipe='both')
    cavs.add_cavity([cav], ['C'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 5})
    return cav


def test_read_objective_values(solved):
    vals = read_objective_values(solved.eigenmode_dir,
                                 ['monopole:freq [MHz]', 'dipole:freq [MHz]',
                                  'dipole:R/Q_t [Ohm/m^(2(m-1))]'])
    assert set(vals) == {'monopole:freq [MHz]', 'dipole:freq [MHz]',
                         'dipole:R/Q_t [Ohm/m^(2(m-1))]'}
    # the dipole passband sits above the monopole one
    assert vals['dipole:freq [MHz]'] > vals['monopole:freq [MHz]']

    # 'dipole:freq' is the primary mode of interest, which defaults to mode 1
    by_mode = read_objective_values(solved.eigenmode_dir, ['dipole:1:freq [MHz]'])
    assert by_mode['dipole:1:freq [MHz]'] == pytest.approx(vals['dipole:freq [MHz]'])


def test_read_objective_values_rejects_unknowns(solved):
    """These used to be silently dropped, so the statistics were quietly computed
    over whichever objectives happened to match."""
    with pytest.raises(ValueError, match=r"unknown QOI 'R/Q \[Ohms\]'"):
        read_objective_values(solved.eigenmode_dir, ['monopole:R/Q [Ohms]'])
    with pytest.raises(ValueError, match='does not exist'):
        read_objective_values(solved.eigenmode_dir, ['dipole:99:freq [MHz]'])
    with pytest.raises(ValueError, match='Add .*sextupole'):
        read_objective_values(solved.eigenmode_dir, ['sextupole:freq [MHz]'])


def test_modes_of_interest_written_to_disk(solved):
    """qois.json is the primary mode; qois_moi.json holds every mode of interest."""
    pol_dir = os.path.join(solved.eigenmode_dir, 'monopole')
    with open(os.path.join(pol_dir, 'qois.json')) as fh:
        primary = json.load(fh)
    with open(os.path.join(pol_dir, 'qois_moi.json')) as fh:
        moi = json.load(fh)
    assert list(moi) == [primary['mode_of_interest']]
    assert moi[primary['mode_of_interest']]['freq [MHz]'] == primary['freq [MHz]']


def test_arbitrary_number_of_modes_of_interest(tmp_path):
    """A polarisation may have any number of modes of interest, not two.

    Each is written to qois_moi.json keyed by its 1-based index; the first listed
    is primary and lands in qois.json; each matches qois_all_modes.
    """
    from cavsim2d.cavity import Cavities, EllipticalCavity
    cavs = Cavities(str(tmp_path))
    cav = EllipticalCavity(3, TESLA, TESLA, TESLA, beampipe='both')
    cavs.add_cavity([cav], ['C'])
    cavs.run_eigenmode({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm',
                        'polarisation': ['monopole', 'dipole'], 'n_modes': 7,
                        'mode_of_interest': {'monopole': [4, 1, 2, 3], 'dipole': [1, 2, 3]}})

    expected = {'monopole': ['4', '1', '2', '3'], 'dipole': ['1', '2', '3']}
    for pol, keys in expected.items():
        pol_dir = os.path.join(cav.eigenmode_dir, pol)
        with open(os.path.join(pol_dir, 'qois_moi.json')) as fh:
            moi = json.load(fh)
        with open(os.path.join(pol_dir, 'qois.json')) as fh:
            primary = json.load(fh)
        with open(os.path.join(pol_dir, 'qois_all_modes.json')) as fh:
            all_modes = json.load(fh)

        assert list(moi) == keys                      # order preserved
        assert primary['mode_of_interest'] == keys[0]  # first listed is primary
        assert primary['freq [MHz]'] == moi[keys[0]]['freq [MHz]']
        for k, q in moi.items():
            assert q['freq [MHz]'] == pytest.approx(all_modes[str(int(k) - 1)]['freq [MHz]'])

    # every mode of interest is addressable as an objective
    vals = read_objective_values(cav.eigenmode_dir,
                                 ['monopole:1:freq [MHz]', 'monopole:4:freq [MHz]',
                                  'dipole:1:freq [MHz]', 'dipole:3:freq [MHz]',
                                  'monopole:freq [MHz]'])
    # the bare monopole objective is the primary mode, which here is mode 4
    assert vals['monopole:freq [MHz]'] == pytest.approx(vals['monopole:4:freq [MHz]'])
    # frequencies rise with mode index within each polarisation
    assert vals['monopole:1:freq [MHz]'] < vals['monopole:4:freq [MHz]']
    assert vals['dipole:1:freq [MHz]'] < vals['dipole:3:freq [MHz]']
