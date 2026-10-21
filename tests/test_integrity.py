"""Folder integrity: results are never replaced unseen, and a cavity folder never
holds results for two different geometries.

Every analysis stamps its results with the geometry it solved
(``geometry_stamp.json``); a run asks before replacing anything, refuses to leave
results computed for another geometry beside its own, and reading such results
warns.
"""
import json
import os
import warnings

import pandas as pd
import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL, requires_abci
from cavsim2d import CavsimWarning, EllipticalCavity, Pillbox, Study
from cavsim2d.models.base import abci_bunch_length
from cavsim2d.solvers.solver_objects import CONFIG_SAMPLES
from cavsim2d.utils import config_validation, integrity
from cavsim2d.utils.integrity import (describe_change, fingerprint_hash,
                                      geometry_fingerprint, read_stamp, stale_units,
                                      write_stamp)

FAST = {'boundary_conditions': 'mm', 'mesh_config': {'h': 20, 'p': 2}}
OTHER = MIDCELL[:6] + [MIDCELL[6] + 4.0]          # same cell, larger Req


def _cav(cell=MIDCELL, name='c'):
    return EllipticalCavity(1, list(cell), list(cell), list(cell), beampipe='none', name=name)


def _no_stdin(monkeypatch):
    """Make input() behave as in a script or under nbconvert: nobody to ask."""
    def _raise(*_):
        raise EOFError
    monkeypatch.setattr('builtins.input', _raise)


def _fake_wakefield(cav, cell):
    """A wakefield result block stamped with *cell*'s geometry, written without
    ABCI so the guard can be exercised anywhere."""
    folder = os.path.join(cav.self_dir, 'wakefield')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'qois.json'), 'w') as f:
        json.dump({'|k_loss| [V/pC]': 1.0}, f)
    write_stamp(folder, _cav(cell))
    return folder


# -- the fingerprint ------------------------------------------------------------

def test_fingerprint_tracks_geometry_not_names():
    a, b = _cav(name='a'), _cav(name='b')
    assert fingerprint_hash(geometry_fingerprint(a)) == fingerprint_hash(geometry_fingerprint(b))
    c = _cav(OTHER)
    assert fingerprint_hash(geometry_fingerprint(a)) != fingerprint_hash(geometry_fingerprint(c))
    text = describe_change(geometry_fingerprint(a), geometry_fingerprint(c))
    assert 'Req_m' in text and f'{MIDCELL[6]:g}' in text


# -- every analysis stamps what it computed ---------------------------------------

def test_eigenmode_results_are_stamped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    stamp = read_stamp(os.path.join(cav.self_dir, 'eigenmode', 'monopole'))
    assert stamp is not None
    assert stamp['hash'] == fingerprint_hash(geometry_fingerprint(cav))
    # the geometry snapshot Study.load rebuilds from describes the same geometry
    with open(os.path.join(cav.self_dir, 'geometry', 'model.json')) as f:
        assert json.load(f)['parameters']['Req_m'] == pytest.approx(MIDCELL[6])


# -- tuning leaves the cavity's own results alone ---------------------------------

def test_tune_does_not_overwrite_the_cavitys_eigenmode_results(tmp_path, monkeypatch):
    """The secant tuner used to probe the cavity itself, so every probe solve
    rewrote <cav>/eigenmode/monopole/qois.json; after a tune the cavity's own
    results were the last probe's four-key dict at the TUNED frequency."""
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    qpath = os.path.join(cav.self_dir, 'eigenmode', 'monopole', 'qois.json')
    with open(qpath) as f:
        before = f.read()
    params = dict(cav.parameters)

    cav.tune.run(freq=790.0, cell_type={'mid-cell': 'Req'},
                 eigenmode_config=FAST)

    with open(qpath) as f:
        assert f.read() == before
    assert cav.parameters == params                # the cavity itself is untouched
    assert cav.tuned is not None
    assert cav.tuned.parameters['Req_m'] != pytest.approx(params['Req_m'])
    assert not stale_units(cav)
    assert not os.path.exists(os.path.join(cav.self_dir, 'tuned', 'tune_info', 'probe'))
    # the saved config survives the run (the run's own wipe used to delete it)
    assert cav.tune.config['freqs'] == 790.0


def test_study_tune_solves_the_tuned_cavity_in_its_own_folder(project_dir):
    """study.run_tune(solve_tuned=True) built study.tuned with add_cavity,
    which re-homed every tuned cavity to its parent's folder, so the follow-up
    eigenmode run wrote the tuned geometry's results over the parent's. An
    elliptical cavity with a .geo hid it; a pillbox did not."""
    study = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    study.add_cavity(pb, 'PB')
    study.run_eigenmode({'boundary_conditions': 'mm'})
    f_parent = pb.eigenmode.qois['freq [MHz]']

    study.run_tune({'freqs': 1100, 'cell_type': {'mid-cell': 'Req'},
                    'eigenmode_config': {'boundary_conditions': 'mm'}, 'solve_tuned': True})

    pb.eigenmode._invalidate()
    assert pb.eigenmode.qois['freq [MHz]'] == pytest.approx(f_parent)
    assert os.path.dirname(pb.tuned.self_dir) == pb.self_dir
    assert pb.tuned.eigenmode.qois['freq [MHz]'] == pytest.approx(1100, abs=1.0)


def test_solve_tuned_is_explicit(tmp_path, monkeypatch):
    """eigenmode_config sets the mesh of the tuning solves; solving the tuned
    cavity afterwards is its own switch, the same for cav.tune.run and
    study.run_tune (the study used to do it whenever eigenmode_config was given)."""
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    tuned = cav.tune.run(freq=790.0, cell_type={'mid-cell': 'Req'},
                         eigenmode_config=FAST).cavity.tuned
    assert not os.path.exists(os.path.join(tuned.self_dir, 'eigenmode'))
    cav.tune.run(freq=790.0, cell_type={'mid-cell': 'Req'}, eigenmode_config=FAST,
                 solve_tuned=True)
    assert cav.tuned.eigenmode.qois['freq [MHz]'] == pytest.approx(790.0, abs=0.5)


def test_a_tuned_version_is_announced_and_must_match_its_parent(tmp_path, monkeypatch):
    """A run on a cavity that has a tuned version says the results go to the
    untuned cavity, once per session. A tuned version of an earlier geometry is
    no longer the cavity's tuned version: study.tuned leaves it out."""
    monkeypatch.chdir(tmp_path)
    study = Study('proj')
    cav = _cav()
    study.add_cavity(cav, 'c')
    cav.tune.run(freq=790.0, cell_type={'mid-cell': 'Req'}, eigenmode_config=FAST)
    assert integrity.current_tuned(cav) is cav.tuned

    with warnings.catch_warnings(record=True) as first:
        warnings.simplefilter('always')
        cav.eigenmode.run(FAST)
    notices = [str(w.message) for w in first if 'has a tuned version' in str(w.message)]
    assert len(notices) == 1 and cav.tuned.self_dir in notices[0]
    assert not os.path.exists(os.path.join(cav.tuned.self_dir, 'eigenmode'))
    with warnings.catch_warnings(record=True) as again:
        warnings.simplefilter('always')
        cav.eigenmode.run(FAST, force=True)
    assert not any('tuned version' in str(w.message) for w in again)

    cav.parameters['Req_m'] = cav.parameters['Req_el'] = cav.parameters['Req_er'] = OTHER[6]
    assert integrity.current_tuned(cav) is None
    with pytest.warns(CavsimWarning, match='Left out of study.tuned'):
        assert study.tuned is None


# -- the run guard ----------------------------------------------------------------

def test_stale_results_from_another_analysis_block_a_silent_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.set_workspace(str(tmp_path / 'c'))
    wake = _fake_wakefield(cav, OTHER)
    _no_stdin(monkeypatch)
    with pytest.raises(RuntimeError, match='wakefield'):
        cav.eigenmode.run(FAST)
    assert os.path.exists(wake)                      # nothing deleted unseen
    assert not os.path.exists(os.path.join(cav.self_dir, 'eigenmode', 'monopole'))

    with pytest.warns(UserWarning, match='deleting results'):
        cav.eigenmode.run(FAST, force=True)
    assert not os.path.exists(wake)
    assert cav.eigenmode.qois['freq [MHz]'] > 0


def test_answering_no_runs_and_deletes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.set_workspace(str(tmp_path / 'c'))
    wake = _fake_wakefield(cav, OTHER)
    prompts = []
    monkeypatch.setattr('builtins.input', lambda msg='': prompts.append(msg) or 'n')
    cav.eigenmode.run(FAST)
    assert len(prompts) == 1 and 'wakefield' in prompts[0] and 'Req_m' in prompts[0]
    assert os.path.exists(wake)
    assert not os.path.exists(os.path.join(cav.self_dir, 'eigenmode', 'monopole'))

    monkeypatch.setattr('builtins.input', lambda msg='': 'y')
    cav.eigenmode.run(FAST)
    assert not os.path.exists(wake)
    assert os.path.exists(os.path.join(cav.self_dir, 'eigenmode', 'monopole', 'qois.json'))


def test_a_cavity_reusing_a_name_cannot_mix_into_the_folder(tmp_path, monkeypatch):
    """Two different cavities given the same name used to share one folder: the
    second's results landed beside the first's with nothing to tell them apart."""
    monkeypatch.chdir(tmp_path)
    first = _cav(name='same')
    first.eigenmode.run(FAST)
    _fake_wakefield(first, MIDCELL)          # a second analysis for the first geometry

    second = _cav(OTHER, name='same')       # different geometry, same folder
    _no_stdin(monkeypatch)
    with pytest.raises(RuntimeError, match='different geometry'):
        second.eigenmode.run(FAST)
    # the first cavity's results are untouched
    with open(os.path.join(first.self_dir, 'eigenmode', 'monopole', 'qois.json')) as f:
        assert json.load(f)['freq [MHz]'] == pytest.approx(first.eigenmode.qois['freq [MHz]'])


def test_rerunning_one_analysis_keeps_the_others(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    wake = _fake_wakefield(cav, MIDCELL)                # same geometry: not stale
    cav.eigenmode.run(FAST, force=True)
    assert os.path.exists(os.path.join(wake, 'qois.json'))


def test_rerun_false_recomputes_results_for_an_old_geometry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    f0 = cav.eigenmode.qois['freq [MHz]']
    cav.eigenmode.run(FAST, rerun=False)              # valid results: reused
    assert cav.eigenmode.qois['freq [MHz]'] == f0

    cav.parameters['Req_m'] = cav.parameters['Req_el'] = cav.parameters['Req_er'] = OTHER[6]
    cav.eigenmode.run(FAST, rerun=False, force=True)  # stale: recomputed, not served
    assert cav.eigenmode.qois['freq [MHz]'] < f0 - 1


def test_reading_results_for_another_geometry_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert cav.eigenmode.qois                   # matching: no warning
    cav.parameters['Req_m'] += 1.0
    integrity._warned.clear()
    with pytest.warns(UserWarning, match='different geometry'):
        cav.eigenmode.qois


def test_study_rejects_two_cavities_with_one_name(project_dir):
    study = Study(project_dir)
    study.add_cavity(_cav(), 'same')
    with pytest.raises(ValueError, match='same'):
        study.add_cavity(_cav(OTHER), 'same')


def test_study_names_a_list_of_cavities(project_dir):
    study = Study(project_dir)
    study.add_cavity([_cav(), _cav(OTHER)])
    assert sorted(study.cavities_dict) == ['cav_0', 'cav_1']


def test_study_constructor_takes_cavities(tmp_path):
    """Study(folder, cavities_list=...) raised AttributeError, and its cavities
    were created outside the study folder."""
    study = Study(str(tmp_path / 'proj'), cavities_list=[_cav()], names_list=['a'])
    assert os.path.dirname(study['a'].self_dir) == study.projectDir


def test_study_load_finds_the_study_folder(tmp_path):
    """Study('proj') keeps its cavities in proj/cavities/; Study.load('proj')
    used to look only in proj/ and return an empty study."""
    study = Study(str(tmp_path / 'proj'))
    study.add_cavity(_cav(), 'a')
    assert list(Study.load(str(tmp_path / 'proj')).cavities_dict) == ['a']


def test_chaining_survives_every_rebuild(tmp_path):
    """rebuild() works from the parameter dict, which holds no chaining, so the
    tuned cavity of a module, its UQ/sweep variants and a module reloaded by
    Study.load all came back as a single cavity: a different geometry."""
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both', chain=3, spacing=50)
    study = Study(str(tmp_path / 'proj'))
    study.add_cavity(cav, 'mod')
    loaded = Study.load(str(tmp_path / 'proj'))['mod']
    assert (loaded.chain, loaded.spacing) == (3, 50)
    variants = cav.spawn(pd.DataFrame({'Req_m': [MIDCELL[6]]}, index=['v']), str(tmp_path / 'sw'))
    assert (variants['v'].chain, variants['v'].spacing) == (3, 50)
    tuned = cav.clone_for_tuning(dict(cav.parameters), os.path.join(cav.self_dir, 'tuned'))
    assert (tuned.chain, tuned.spacing) == (3, 50)
    assert fingerprint_hash(geometry_fingerprint(loaded)) == fingerprint_hash(geometry_fingerprint(cav))


def test_elliptical_setters_change_the_solved_geometry():
    """The base setters updated only cav.shape, while profile() and the geometry
    stamp read cav.parameters, so set_mid_cell changed nothing that was solved."""
    one = _cav()
    one.set_end_cell_left(OTHER)
    assert one.parameters['Req_el'] == pytest.approx(OTHER[6])
    three = EllipticalCavity(3, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    before = geometry_fingerprint(three)
    three.set_mid_cell(MIDCELL[:5] + [MIDCELL[5] + 1.0, MIDCELL[6]])
    assert three.parameters['L_m'] == pytest.approx(MIDCELL[5] + 1.0)
    assert geometry_fingerprint(three) != before


def test_show_views_fall_back_outside_a_notebook(tmp_path, monkeypatch):
    """Draw() renders nothing outside a notebook, and says nothing; the show_*
    views warn and draw the matplotlib view instead."""
    monkeypatch.chdir(tmp_path)
    with pytest.warns(UserWarning, match='only renders in a Jupyter notebook'):
        ax = _cav().show_geometry(show=False)
    assert hasattr(ax, 'triplot')


# -- overwrite=True only ever deletes cavsim2d's own folders ----------------------

def test_save_overwrite_refuses_a_folder_it_did_not_make(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.set_workspace(str(tmp_path / 'ws' / 'c'))
    precious = tmp_path / 'notes'
    precious.mkdir()
    (precious / 'thesis.txt').write_text('irreplaceable')
    with pytest.raises(ValueError, match='not a cavsim2d'):
        cav.save(str(precious), overwrite=True)
    assert (precious / 'thesis.txt').exists()


def test_study_overwrite_refuses_a_folder_it_did_not_make(tmp_path):
    folder = tmp_path / 'proj' / 'cavities'
    folder.mkdir(parents=True)
    (folder / 'thesis.txt').write_text('irreplaceable')
    with pytest.raises(ValueError, match='not a cavsim2d'):
        Study(str(tmp_path / 'proj'), overwrite=True)
    assert (folder / 'thesis.txt').exists()


# -- config samples ---------------------------------------------------------------

def test_config_sample_spellings_on_every_namespace(project_dir):
    cav = _cav()
    for solver in (cav.tune, cav.eigenmode, cav.wakefield, cav.multipacting):
        sample = solver.config_sample
        assert sample == solver.cfg_sample == solver.sample_config == solver.sample_cfg
        assert solver.config_sample() == sample        # callable too
        sample['processes'] = 99                         # a fresh copy each time
        assert solver.config_sample.get('processes') != 99
    for kind in ('tune', 'eigenmode', 'wakefield', 'multipacting', 'uq', 'optimisation'):
        assert cav.cfg_sample(kind, show=False) == cav.config_sample(kind, show=False)
    study = Study(project_dir)                           # no cavity needed
    assert study.cfg_sample('tune', show=False) == cav.tune.config_sample
    assert study.eigenmode.config_sample == cav.eigenmode.config_sample
    assert study.optimisation.config_sample == cav.config_sample('optimisation', show=False)
    with pytest.raises(ValueError, match='eigenmode'):
        cav.config_sample('eigenmod')


def test_samples_display_notes_as_valid_python():
    """A displayed sample notes each key's meaning and unit, and pastes back as the
    same dict."""
    cav = _cav()
    for kind in CONFIG_SAMPLES:
        sample = cav.config_sample(kind, show=False)
        assert eval(repr(sample)) == dict(sample)
    shown = repr(cav.tune.config_sample)
    assert "'freqs': None," in shown and '# target [MHz]' in shown
    edited = cav.eigenmode.config_sample
    edited['my_key'] = 1                                 # a key without a note
    assert "    'my_key': 1,\n" in repr(edited)


def test_samples_are_valid_configs():
    """The samples are templates to edit and run, so they must pass validation:
    every key of a default config is one the validator accepts."""
    validators = {'eigenmode': config_validation.validate_eigenmode_config,
                  'wakefield': config_validation.validate_wakefield_config,
                  'tune': config_validation.validate_tune_config,
                  'multipacting': config_validation.validate_multipacting_config,
                  'uq': config_validation.validate_uq_config,
                  'optimisation': config_validation.validate_optimisation_config}
    assert set(validators) == set(CONFIG_SAMPLES)
    cav = _cav()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for kind, validate in validators.items():
            validate(cav.config_sample(kind, show=False))
    for kind, sample in CONFIG_SAMPLES.items():
        assert set(sample) <= config_validation.KNOWN_KEYS[kind]
    assert 'cell_type' in cav.tune.config_sample
    assert 'parameters' not in cav.tune.config_sample       # no legacy keys


# -- documented aliases reach the solver ------------------------------------------

def _capture_tune(monkeypatch):
    seen = {}

    def fake(cavs_dict, cfg, **_):
        seen.update(cfg)
    monkeypatch.setattr('cavsim2d.processes.tune.run_tune_parallel', fake)
    return seen


def test_tune_aliases(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = _capture_tune(monkeypatch)
    cav = _cav()
    cav.tune.run(freq=1400, tolerance=1e-3, parameters='Req', cell_types='mid-cell')
    assert seen['freqs'] == 1400
    assert seen['tol'] == 1e-3
    assert seen['cell_type'] == {'mid-cell': ['Req']}
    for alias in ('freq', 'tolerance', 'parameters', 'cell_types'):
        assert alias not in seen


def test_unknown_keys_warn_on_cavity_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _capture_tune(monkeypatch)
    with pytest.warns(UserWarning, match="unrecognised key 'freqz'"):
        _cav().tune.run(freqz=1300)


def test_overwrite_is_an_alias_of_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST)
    monkeypatch.setattr('builtins.input',
                        lambda *_: pytest.fail('overwrite=True must not ask'))
    cav.eigenmode.run(FAST, overwrite=True)


def test_nmodes_alias_reaches_the_solver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cav = _cav()
    cav.eigenmode.run(FAST, nmodes=4)
    assert len(cav.eigenmode.qois_df) == 4


# -- the cavity-level attributes always match the namespaces ----------------------

def test_qoi_attributes_follow_every_run_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fresh = _cav(name='fresh')
    assert fresh.eigenmode.qois == {} and fresh.wakefield.qois == {} and fresh.tune.qois == {}
    assert fresh.eigenmode_qois == {}

    cav = _cav()
    cav.eigenmode.run(FAST)                 # the namespace path
    assert cav.eigenmode_qois == cav.eigenmode.qois
    assert cav.freq == pytest.approx(cav.eigenmode.qois['freq [MHz]'])
    assert cav.wakefield_qois == cav.wakefield.qois == {}


def test_cavity_has_run_tune(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seen = _capture_tune(monkeypatch)
    _cav().run_tune({'freqs': 1300, 'cell_type': {'mid-cell': 'Req'}})
    assert seen['freqs'] == 1300


# -- study wakefield keeps the bunch and wake length it was given -----------------

def test_study_wakefield_passes_bunch_and_wake_length(project_dir, monkeypatch):
    """Study.run_wakefield nested a hard-coded 25 mm bunch and 50 m wake under
    beam_config/wake_config, which the deck writer reads first: every study-level
    run used them whatever bunch_length/wakelength it was given."""
    seen = {}
    monkeypatch.setattr('cavsim2d.study.run_wakefield_parallel',
                        lambda cavs, cfg, *a, **k: seen.update(cfg))
    study = Study(project_dir)
    study.add_cavity(_cav(), 'c')
    study.run_wakefield(bunch_length=10, wakelength=1, MROT=0)
    assert abci_bunch_length(seen) == pytest.approx(0.010)
    assert (seen.get('wake_config') or {}).get('wakelength', seen.get('wakelength')) == 1


@requires_abci
def test_study_wakefield_deck_uses_the_requested_bunch(project_dir):
    study = Study(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    study.add_cavity(cav, 'c')
    study.run_wakefield(bunch_length=10, wakelength=1, MROT=0)
    with open(os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'cavity.abc')) as f:
        deck = f.read()
    assert 'SIG = 0.01,' in deck and 'UBT = 1,' in deck
    assert read_stamp(os.path.join(cav.self_dir, 'wakefield')) is not None
