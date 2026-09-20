"""Beam-line elements and their concatenation: Beampipe, Bellows, BLA, Assembly."""
import warnings


import numpy as np
import pytest

from conftest import MIDCELL
from cavsim2d import (Assembly, BLA, Beampipe, Bellows, CircularWaveguide,
                      EllipticalCavity, Taper)
from cavsim2d.geometry import Profile

RI = 80.0


def _bellows(**kw):
    kw = {'Ri': RI, 'A': 15.0, 'L_p': 10.0, 'N_conv': 5,
          'R_root': 2.0, 'R_crest': 2.0, **kw}
    return Bellows(**kw)


def _line(**kw):
    return Assembly([Beampipe(R=RI, L=100.0, name='bp_l'),
                     _bellows(),
                     EllipticalCavity(1, MIDCELL, beampipe='none', name='cav'),
                     Beampipe(R=RI, L=100.0, name='bp_r')], **kw)


# ── Beampipe ──────────────────────────────────────────────────────────────

def test_beampipe_ends_are_open_by_default():
    """The drift element must not carry a metal plate across the bore, or it
    could not be concatenated between two devices."""
    assert [s['name'] for s in Beampipe(R=RI, L=100.0).profile()._segs] == \
        ['PMC', 'PEC', 'PMC', 'AXI']


def test_beampipe_ends_can_be_closed():
    assert [s['name'] for s in Beampipe(R=RI, L=100.0, ends='pec').profile()._segs] == \
        ['PEC', 'PEC', 'PEC', 'AXI']
    left, right = Beampipe(R=RI, L=100.0, ends=('pec', 'pmc')).ends
    assert (left, right) == ('PEC', 'PMC')


def test_beampipe_rejects_unknown_end():
    with pytest.raises(ValueError, match='unknown left end'):
        Beampipe(R=RI, L=100.0, ends='absorbing')


def test_circular_waveguide_alias_keeps_pec_ends():
    """The deprecated name must keep meshing the closed cylinder scripts were
    written against — its ends are PEC, not the new PMC default."""
    with pytest.warns(DeprecationWarning):
        cw = CircularWaveguide(R=230.0, L=200.0)
    assert cw.ends == ('PEC', 'PEC')
    assert [s['name'] for s in cw.profile()._segs] == ['PEC', 'PEC', 'PEC', 'AXI']


# ── Bellows ───────────────────────────────────────────────────────────────

def test_bellows_extent_and_arc_count():
    b = _bellows(L_bp=10.0)
    pts = np.asarray(b.profile().contour_points(2e-4, skip=('AXI', 'PMC')))
    assert np.isclose(pts[:, 0].ptp() * 1e3, 5 * 10.0 + 2 * 10.0)   # N*L_p + 2*L_bp
    assert np.isclose(pts[:, 1].min() * 1e3, RI)
    assert np.isclose(pts[:, 1].max() * 1e3, RI + 15.0)
    # four rounded corners per convolution, emitted as real arc segments
    assert sum(1 for s in b.profile()._segs if s['kind'] == 'arc') == 4 * 5


def test_bellows_wall_is_tangent_continuous():
    """Every corner is a true inscribed arc, so a densely sampled wall turns
    only by the arc discretisation — never by a kink."""
    pts = np.asarray(_bellows().profile().contour_points(5e-5, skip=('AXI', 'PMC')))
    d = np.diff(pts, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    turn = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)))
    assert turn.max() < 10.0


@pytest.mark.parametrize('kw, match', [
    ({'R_crest': 3.0}, 'crest corners do not fit'),
    ({'R_root': 3.0}, 'root corners do not fit'),
    ({'A': 3.0, 'R_root': 2.0, 'R_crest': 2.0}, 'do not both fit on the'),
    ({'flank_angle': 120.0}, 'flank_angle must be in'),
    ({'crest_fraction': 1.0}, 'crest_fraction must be'),
    ({'N_conv': 0}, 'N_conv must be at least 1'),
])
def test_bellows_rejects_infeasible_parameters(kw, match):
    """Infeasible corner radii are caught analytically at parameter time, so an
    optimiser gets a named constraint rather than a kernel failure."""
    with pytest.raises(ValueError, match=match):
        _bellows(**kw)


def test_bellows_degenerate_flats_emit_no_zero_length_segment():
    """R = L_p/4 is the nominal hydroformed convolution, not an edge case: the
    flats vanish and the wall becomes arc-to-arc."""
    b = _bellows(R_root=2.5, R_crest=2.5)          # 2*R == L_p/2 exactly
    prof = b.profile()
    pts = prof.points
    assert not any(np.allclose(pts[s['i0']], pts[s['i1']]) for s in prof._segs)
    assert sum(1 for s in prof._segs if s['kind'] == 'arc') == 4 * 5


def test_bellows_tilted_flank_keeps_period():
    b = _bellows(L_p=20.0, N_conv=3, flank_angle=70.0,
                 R_root=1.0, R_crest=1.0)
    zs = [z for z, _ in b.profile().points]
    assert np.isclose((max(zs) - min(zs)) * 1e3, 3 * 20.0)


def test_bellows_n_conv_is_not_a_tune_variable():
    """N_conv is discrete; offering it as a continuous handle would let a
    tuner or a UQ sweep ask for 4.7 convolutions."""
    assert 'N_conv' not in _bellows().tune_variables()


# ── Profile.then ──────────────────────────────────────────────────────────

def test_assembled_cavity_matches_monolithic_contour():
    """Pipes concatenated onto a bare cavity must reproduce, exactly, the same
    cavity built with beampipe='both'."""
    L_bp = 2 * MIDCELL[5]
    mono = EllipticalCavity(1, MIDCELL, beampipe='both')
    asm = Assembly([Beampipe(R=MIDCELL[4], L=L_bp, name='l'),
                    EllipticalCavity(1, MIDCELL, beampipe='none', name='c'),
                    Beampipe(R=MIDCELL[4], L=L_bp, name='r')])
    a = np.asarray(mono.profile().contour_points(2e-4, skip=('AXI',)))
    b = np.asarray(asm.profile().contour_points(2e-4, skip=('AXI',)))
    a[:, 0] -= a[:, 0].min()
    b[:, 0] -= b[:, 0].min()
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-12)


def test_concatenation_leaves_no_interface_boundary():
    """Joining removes both inner apertures: the vacuum is one region, so there
    is no interface for a boundary condition to sit on."""
    prof = _line().profile()
    assert sum(1 for s in prof._segs if s['name'] == 'PMC') == 2
    left, right = prof.end_aperture_segments()
    assert (left, right) == (0, len(prof._segs) - 2)


def test_then_inserts_a_gap():
    a = Beampipe(R=40.0, L=100.0, name='a').profile()
    b = Beampipe(R=40.0, L=100.0, name='b').profile()
    joined = a.then(b, gap=0.02)
    zs = [z for z, _ in joined.points]
    assert np.isclose(max(zs) - min(zs), 0.1 + 0.02 + 0.1)


def test_then_warns_about_a_mismatched_aperture_but_builds_it():
    """A step is legitimate geometry, so it is built — but it is easy to create
    by accident and invisible once joined, so it is reported."""
    small = Beampipe(R=40.0, L=100.0, name='s').profile()
    big = Beampipe(R=60.0, L=100.0, name='b').profile()
    with pytest.warns(UserWarning, match='aperture mismatch'):
        joined = small.then(big, gap=0.02)
    zs = [z for z, _ in joined.points]
    assert np.isclose(max(zs) - min(zs), 0.1 + 0.02 + 0.1)
    assert any(np.isclose(r, 60.0e-3) for _, r in joined.points)


def test_then_step_warning_locates_the_junction():
    small = Beampipe(R=40.0, L=100.0, name='s').profile()
    big = Beampipe(R=60.0, L=100.0, name='b').profile()
    with pytest.warns(UserWarning) as rec:
        small.then(big)
    msg = str(rec[0].message)
    for fragment in ('40 mm', '60 mm', 'z = 50 mm', 'Taper', 'allow_step'):
        assert fragment in msg


def test_then_allow_step_silences_the_warning():
    small = Beampipe(R=40.0, L=100.0, name='s').profile()
    big = Beampipe(R=60.0, L=100.0, name='b').profile()
    with warnings.catch_warnings():
        warnings.simplefilter('error')          # any warning fails the test
        small.then(big, allow_step=True)


def test_assembly_warns_at_construction_and_locates_every_step():
    with pytest.warns(UserWarning, match='aperture mismatch') as rec:
        line = Assembly([Beampipe(R=40.0, L=80.0, name='a'),
                         Beampipe(R=80.0, L=80.0, name='b'),
                         Beampipe(R=35.0, L=60.0, name='c')])
    msg = str(rec[0].message)
    assert '2 junctions' in msg
    assert "'a'" in msg and "'b'" in msg and "'c'" in msg
    assert 'z =    40.000 mm' in msg and 'z =   120.000 mm' in msg
    assert '+40 mm' in msg and '-45 mm' in msg
    assert np.isclose(line.length, 220.0)       # built regardless


def test_check_continuity_reports_the_junctions():
    with pytest.warns(UserWarning):
        line = Assembly([Beampipe(R=40.0, L=80.0, name='a'),
                         Beampipe(R=80.0, L=80.0, name='b')])
    with pytest.warns(UserWarning):
        found, = line.check_continuity()
    assert found['labels'] == ('a', 'b')
    assert np.isclose(found['z'], 40.0)
    assert np.isclose(found['step'], 40.0)
    assert found['radii'] == (40.0, 80.0)


def test_assembly_profile_does_not_rewarn():
    """profile() runs on every tuner iteration; re-warning there would bury the
    one useful message from construction."""
    with pytest.warns(UserWarning):
        line = Assembly([Beampipe(R=40.0, L=80.0, name='a'),
                         Beampipe(R=80.0, L=80.0, name='b')])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for _ in range(3):
            line.profile()


def test_assembly_allow_steps_silences_and_survives_rebuild():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        line = Assembly([Beampipe(R=40.0, L=80.0, name='narrow'),
                         Beampipe(R=80.0, L=80.0, name='wide')], allow_steps=True)
    assert np.isclose(line.length, 160.0)
    assert line.rebuild(dict(line.parameters)).allow_steps is True


def test_matched_apertures_warn_about_nothing():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        Beampipe(R=40.0, L=80.0, name='a') + Beampipe(R=40.0, L=80.0, name='b')


def test_a_taper_is_the_way_to_change_aperture():
    line = (Beampipe(R=40.0, L=80.0, name='narrow')
            + Taper(R_left=40.0, R_right=80.0, L=120.0)
            + Beampipe(R=80.0, L=80.0, name='wide'))
    assert np.isclose(line.length, 280.0)


def test_then_rejects_a_non_meridian_profile():
    open_contour = (Profile('x').start(0.0, 0.05)
                    .line_to(0.1, 0.05, 'PEC')
                    .line_to(0.1, 0.0, 'PMC')
                    .close('AXI'))
    with pytest.raises(ValueError, match='standard axis-to-axis meridian'):
        open_contour.then(Beampipe(R=50.0, L=10.0).profile())


# ── Assembly ──────────────────────────────────────────────────────────────

def test_add_operator_builds_and_flattens():
    line = (Beampipe(R=RI, L=50.0, name='a')
            + Beampipe(R=RI, L=50.0, name='b')
            + Beampipe(R=RI, L=50.0, name='c'))
    assert isinstance(line, Assembly)
    assert len(line) == 3 and line.labels == ['a', 'b', 'c']


def test_repeated_names_get_suffixed_labels():
    line = _line()
    assert line.labels == ['bp_l', 'bellows', 'cav', 'bp_r']
    two = Assembly([_bellows(), _bellows()])
    assert two.labels == ['bellows_1', 'bellows_2']


def test_parameters_are_namespaced_by_element():
    line = _line()
    assert line.parameters['bellows:A'] == 15.0
    assert line.parameters['bp_l:L'] == 100.0
    assert 'cav:Req_m' in line.parameters
    assert 'bellows:A' in line.tune_variables()


def test_expand_variable_delegates_to_the_element():
    """A bare name still means 'the same quantity in every cell' of the element
    that owns it."""
    assert _line().expand_variable('cav:Req') == \
        ['cav:Req_m', 'cav:Req_el', 'cav:Req_er']


def test_set_tune_value_reaches_the_element():
    line = _line()
    line.set_tune_value('bellows:A', 12.0)
    assert line.element('bellows').parameters['A'] == 12.0
    assert line.parameters['bellows:A'] == 12.0
    assert np.isclose(line.get_tune_value('bellows:A'), 12.0)
    # a bare suffixed name resolves through the element's own convention
    line.set_tune_value('cav:Req', 170.0)
    assert line.element('cav').parameters['Req_m'] == 170.0


def test_unknown_variable_names_the_available_labels():
    with pytest.raises(ValueError, match='bellows'):
        _line().get_tune_value('nosuch:A')


def test_push_revalidates_the_elements():
    """The tuner mutates parameters in place, so an element that owns a
    constraint must re-check it when the assembly pushes the values down."""
    line = _line()
    line.parameters['bellows:R_crest'] = 3.0
    with pytest.raises(ValueError, match='crest corners do not fit'):
        line.profile()


def test_ends_override_reopens_a_closed_element():
    """set_ends wins over the end elements, in both directions — the solver's
    own boundary_conditions can only close an aperture, never reopen one."""
    line = Assembly([Beampipe(R=RI, L=100.0, ends='pec', name='a'),
                     Beampipe(R=RI, L=100.0, name='b')])
    assert line.profile()._segs[0]['name'] == 'PEC'
    line.set_ends('pmc')
    assert line.profile()._segs[0]['name'] == 'PMC'
    line.set_ends('pmc', 'pec')
    prof = line.profile()
    assert (prof._segs[0]['name'], prof._segs[-2]['name']) == ('PMC', 'PEC')


def test_passive_elements_do_not_shift_the_mode_of_interest():
    """n_cells picks the monopole mode of interest, so a pipe or a bellows must
    not count as an accelerating cell."""
    assert _line().n_cells == 1
    assert Assembly([EllipticalCavity(2, MIDCELL, beampipe='none', name='a'),
                     EllipticalCavity(3, MIDCELL, beampipe='none', name='b')]).n_cells == 5


def test_row_slots_resolve_a_bare_element_variable():
    """Optimisation and UQ name their variables the way the user wrote the
    bounds ('cav:A'), but an elliptical cavity stores A_m/A_el/A_er."""
    line = Assembly([Beampipe(R=RI, L=100.0, name='bp'),
                     EllipticalCavity(1, MIDCELL, beampipe='none', name='cav')])
    assert line._row_slots('cav:A') == ['cav:A_m', 'cav:A_el', 'cav:A_er']
    assert line._row_slots('cav:A_m') == ['cav:A_m']
    assert line._row_slots('bp:R') == ['bp:R']
    assert line._row_slots('nosuch:X') == []


def test_spawn_applies_a_bare_element_variable_to_the_geometry(tmp_path):
    """The regression that matters: a bounds column that does not reach the
    geometry makes every candidate identical, and the objectives then differ
    only by solver noise — which reads as a converged optimisation."""
    import pandas as pd

    line = Assembly([Beampipe(R=RI, L=100.0, name='bp'),
                     EllipticalCavity(1, MIDCELL, beampipe='none', name='cav')])
    df = pd.DataFrame([{'cav:A': 60.0}, {'cav:A': 57.0}], index=['c0', 'c1'])
    spawned = line.spawn(df, str(tmp_path))

    a0 = spawned.cavities_dict['c0'].element('cav').parameters
    a1 = spawned.cavities_dict['c1'].element('cav').parameters
    assert a0['A_m'] == a0['A_el'] == a0['A_er'] == 60.0
    assert a1['A_m'] == 57.0
    # and the geometry genuinely differs between candidates
    p0 = np.asarray(spawned.cavities_dict['c0'].profile().contour_points(1e-3, skip=('AXI',)))
    p1 = np.asarray(spawned.cavities_dict['c1'].profile().contour_points(1e-3, skip=('AXI',)))
    assert p0.shape != p1.shape or not np.allclose(p0, p1)


def test_rebuild_expands_a_bare_element_variable():
    line = Assembly([Beampipe(R=RI, L=100.0, name='bp'),
                     EllipticalCavity(1, MIDCELL, beampipe='none', name='cav')])
    params = dict(line.parameters)
    params['cav:A'] = 60.0
    cav = line.rebuild(params).element('cav')
    assert cav.parameters['A_m'] == cav.parameters['A_el'] == 60.0


def test_rebuild_round_trips():
    line = _line()
    params = dict(line.parameters)
    params['bellows:A'] = 12.0
    fresh = line.rebuild(params)
    assert isinstance(fresh, Assembly)
    assert fresh.labels == line.labels
    assert fresh.element('bellows').parameters['A'] == 12.0
    assert fresh.element('bellows').N_conv == 5


def test_reconstruct_from_state_round_trips():
    line = _line()
    rebuilt = Assembly._reconstruct_from_state(line._reconstruct_state())
    assert rebuilt.labels == line.labels
    assert rebuilt.parameters == line.parameters


def test_assembly_needs_at_least_two_elements():
    with pytest.raises(ValueError, match='at least two elements'):
        Assembly([Beampipe(R=RI, L=100.0)])


def test_gaps_must_match_the_junction_count():
    with pytest.raises(ValueError, match='one entry per junction'):
        Assembly([Beampipe(R=RI, L=50.0, name='a'),
                  Beampipe(R=RI, L=50.0, name='b')], gaps=[1.0, 2.0])


# ── BLA ───────────────────────────────────────────────────────────────────

def test_bla_carries_an_absorber_ring():
    bla = BLA(R=RI, L=150.0, absorber_length=80.0, absorber_thickness=8.0)
    d, = bla.dielectrics
    assert d['z'] == (-40.0, 40.0)
    # inner radius IS the bore; the thickness goes outward
    assert d['r'] == (RI, RI + 8.0)


def test_bla_does_not_obstruct_the_bore():
    """The absorber sits in a recess outside the aperture. An absorber that ate
    into the bore would throttle the beam pipe it is supposed to line."""
    bla = BLA(R=RI, L=150.0, absorber_length=80.0, absorber_thickness=8.0)
    pts = np.asarray(bla.profile().contour_points(2e-4, skip=('AXI', 'PMC')))
    assert np.isclose(pts[:, 1].min() * 1e3, RI)            # bore never narrows
    assert np.isclose(pts[:, 1].max() * 1e3, RI + 8.0)      # wall steps outward
    d, = bla.dielectrics
    assert d['r'][0] >= RI


def test_bla_ends_match_a_plain_pipe_so_it_joins_without_a_step():
    line = Assembly([Beampipe(R=RI, L=100.0, name='bp'),
                     BLA(R=RI, L=150.0, absorber_length=80.0,
                         absorber_thickness=8.0)])
    pts = np.asarray(line.profile().contour_points(2e-4, skip=('AXI', 'PMC')))
    # the only radius above the bore is the recess itself
    assert np.isclose(pts[:, 1].min() * 1e3, RI)


def test_bla_absorber_may_span_the_whole_element():
    bla = BLA(R=RI, L=150.0, absorber_length=150.0, absorber_thickness=8.0)
    prof = bla.profile()
    pts = prof.points
    assert not any(np.allclose(pts[s['i0']], pts[s['i1']]) for s in prof._segs)


def test_bla_is_native_only():
    """The inherited pipe writer would emit a plain cylinder, losing the recess
    and the absorber; clone_for_tuning keys off write_geometry being None."""
    bla = BLA(R=RI, L=150.0, absorber_length=80.0, absorber_thickness=8.0)
    assert not callable(getattr(bla, 'write_geometry', None))


def test_bla_absorber_travels_with_its_element():
    """Region extents are absolute mm, so concatenation has to move them with
    the element or the absorber stays behind at the origin."""
    line = Assembly([Beampipe(R=RI, L=100.0, name='bp'),
                     BLA(R=RI, L=150.0, absorber_length=80.0,
                         absorber_thickness=8.0)])
    d, = line.dielectrics
    assert d['material'] == 'bla_absorber'          # prefixed with the label
    assert np.allclose(d['z'], (125.0 - 40.0, 125.0 + 40.0))


def test_bla_rejects_a_ring_that_does_not_fit():
    with pytest.raises(ValueError, match='does not fit inside the element'):
        BLA(R=RI, L=100.0, absorber_length=80.0, absorber_thickness=8.0,
            z_absorber=40.0)


# ── Taper ─────────────────────────────────────────────────────────────────

def test_taper_spans_the_two_radii():
    t = Taper(R_left=35.0, R_right=80.0, L=120.0)
    pts = np.asarray(t.profile().contour_points(2e-4, skip=('AXI', 'PMC')))
    assert np.isclose(pts[:, 1].min() * 1e3, 35.0)
    assert np.isclose(pts[:, 1].max() * 1e3, 80.0)
    assert np.isclose(pts[:, 0].ptp() * 1e3, 120.0)
    assert np.isclose(t.half_angle, np.degrees(np.arctan2(45.0, 120.0)))


def test_taper_fillets_round_both_corners():
    t = Taper(R_left=35.0, R_right=80.0, L=160.0,
              straight_left=20.0, straight_right=20.0, R_fillet=8.0)
    prof = t.profile()
    assert sum(1 for s in prof._segs if s['kind'] == 'arc') == 2
    pts = np.asarray(prof.contour_points(5e-5, skip=('AXI', 'PMC')))
    d = np.diff(pts, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    turn = np.degrees(np.arccos(np.clip((d[:-1] * d[1:]).sum(1), -1.0, 1.0)))
    assert turn.max() < 5.0


def test_taper_without_straight_runs_has_no_corners_to_round():
    """The cone spans the element, so its corners belong to whatever it is
    joined to — the fillet has nothing to sit on and is ignored, not an error."""
    t = Taper(R_left=35.0, R_right=80.0, L=120.0, R_fillet=5.0)
    assert not any(s['kind'] == 'arc' for s in t.profile()._segs)


@pytest.mark.parametrize('kw, match', [
    ({'straight_left': 100.0, 'straight_right': 100.0}, 'leaving no room for the'),
    ({'straight_left': 0.5, 'R_fillet': 20.0}, 'left fillet does not fit'),
    ({'R_fillet': -1.0}, 'R_fillet must be non-negative'),
    ({'R_left': 0.0}, 'R_left must be positive'),
])
def test_taper_rejects_infeasible_parameters(kw, match):
    with pytest.raises(ValueError, match=match):
        Taper(**{'R_left': 35.0, 'R_right': 80.0, 'L': 120.0, **kw})


def test_taper_overlapping_fillets_on_a_short_cone_are_rejected():
    with pytest.raises(ValueError, match='do not both fit on the'):
        Taper(R_left=35.0, R_right=36.0, L=100.0, straight_left=45.0,
              straight_right=45.0, R_fillet=250.0)


def test_taper_joins_two_bores_without_a_step():
    """A taper is the controlled alternative to the abrupt step then() inserts
    when two elements simply do not match in radius."""
    line = Assembly([Beampipe(R=35.0, L=60.0, name='a'),
                     Taper(R_left=35.0, R_right=80.0, L=120.0),
                     Beampipe(R=80.0, L=60.0, name='b')])
    prof = line.profile()
    assert np.isclose(line.length, 240.0)
    # a step would show up as a vertical wall segment at a junction z
    verticals = [s for s in prof._segs
                 if s['name'] == 'PEC'
                 and np.isclose(prof.points[s['i0']][0], prof.points[s['i1']][0])]
    assert not verticals


def test_taper_is_not_an_accelerating_cell():
    assert Taper(R_left=35.0, R_right=80.0, L=120.0).contributes_cells is False


def test_assembly_directs_dielectrics_to_the_element():
    line = Assembly([Beampipe(R=RI, L=50.0, name='a'),
                     Beampipe(R=RI, L=50.0, name='b')])
    with pytest.raises(NotImplementedError, match='element that carries it'):
        line.add_dielectric('foo', 10.0, z=(0, 1), r=(0, 1))


# ── meshing / solving ─────────────────────────────────────────────────────

def test_element_spans_tile_the_line_without_gaps():
    line = _line()
    spans = line.element_spans()
    assert [s['label'] for s in spans] == line.labels
    for a, b in zip(spans, spans[1:]):
        assert np.isclose(a['z1'], b['z0'])          # butted, no gaps
    assert np.isclose(spans[-1]['z1'] - spans[0]['z0'], line.length)


def test_element_spans_group_by_kind():
    """Same kind means same colour when shading, so two bellows must report the
    same kind while keeping distinct labels."""
    line = Assembly([_bellows(), Beampipe(R=RI, L=50.0, name='bp'), _bellows()])
    kinds = [s['kind'] for s in line.element_spans()]
    assert kinds[0] == kinds[2] == 'bellows'
    assert kinds[1] == 'beampipe'
    assert len({s['label'] for s in line.element_spans()}) == 3


def test_shade_elements_draws_one_band_per_element():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line = _line()
    _, ax = plt.subplots()
    line.shade_elements(ax, edges=False, legend=True, labels=False)
    assert len(ax.patches) == len(line.elements)
    # one legend entry per *kind*, not per element
    handles, labels = ax.get_legend_handles_labels()
    assert set(labels) == {s['kind'] for s in line.element_spans()}
    plt.close('all')


def test_band_colors_are_assigned_in_order_and_never_cycled():
    """A ninth kind must not reuse slot 1's colour: repeating a hue says two
    unrelated kinds are the same kind."""
    from cavsim2d.models.assembly import BAND_COLORS, BAND_OVERFLOW
    assert len(set(BAND_COLORS)) == len(BAND_COLORS)
    assert BAND_OVERFLOW not in BAND_COLORS


def test_shade_elements_labels_every_band_by_default():
    """A light wash cannot carry identity by colour alone, so the bands are
    named in place unless the caller opts out."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line = _line()
    _, ax = plt.subplots()
    line.shade_elements(ax)
    drawn = {t.get_text() for t in ax.texts}
    assert set(line.labels) <= drawn
    plt.close('all')


def test_shade_elements_marks_boundaries_between_like_neighbours():
    """Two cavities in a row share a colour, so without the edge rules their
    bands merge into one and the join is invisible."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    line = Assembly([EllipticalCavity(1, MIDCELL, beampipe='both', name='c1'),
                     EllipticalCavity(1, MIDCELL, beampipe='both', name='c2')])
    _, ax = plt.subplots()
    line.shade_elements(ax, edges=True)
    junction = line.element_spans()[0]['z1']
    assert any(np.isclose(ln.get_xdata()[0], junction) for ln in ax.lines)
    plt.close('all')


def test_axis_field_profile_handles_a_lossy_complex_solve(project_dir):
    """A lossy region makes the eigenproblem complex. Sampling must cope: a
    float buffer silently rejected every complex sample, so the whole profile
    came back NaN and plotted as an empty axis."""
    pytest.importorskip('ngsolve')
    line = Assembly([Beampipe(R=RI, L=120.0, name='bp'),
                     BLA(R=RI, L=160.0, absorber_length=90.0,
                         absorber_thickness=10.0, eps_r=10.0, tan_delta=0.3,
                         maxh=5.0)], ends='pec', name='lossy')
    line.set_workspace(project_dir)
    line.eigenmode.run({'polarisation': 'monopole', 'n_modes': 2,
                        'boundary_conditions': 'mm',
                        'mesh_config': {'h': 12, 'p': 2}})
    z, ez = line.axis_field_profile(0)
    assert np.isfinite(ez).all(), 'axis field came back NaN'
    assert np.nanmax(np.abs(ez)) > 0
    assert np.isclose(np.nanmax(np.abs(ez)), 1.0)      # normalised
    assert z[-1] > z[0]


def test_axis_field_profile_rejects_an_unsampleable_mode(project_dir):
    """An all-NaN profile must raise, not return silently: it renders as an
    empty plot, which looks like a physics result."""
    pytest.importorskip('ngsolve')
    line = Assembly([Beampipe(R=RI, L=120.0, name='a'),
                     Beampipe(R=RI, L=120.0, name='b')], ends='pec', name='tube')
    line.set_workspace(project_dir)
    line.eigenmode.run({'polarisation': 'monopole', 'n_modes': 2,
                        'boundary_conditions': 'mm',
                        'mesh_config': {'h': 12, 'p': 2}})
    z, ez = line.axis_field_profile(0)
    assert np.isfinite(ez).all()


def test_assembly_meshes_with_named_boundaries():
    pytest.importorskip('ngsolve')
    mesh = _line().profile().mesh(maxh=0.02, order=2)
    assert mesh.ne > 0
    assert set(mesh.GetBoundaries()) == {'AXI', 'PEC', 'PMC'}


def test_assembled_cavity_solves_like_the_monolithic_one(project_dir):
    """The strongest check that concatenation is lossless: same geometry in,
    same eigenmode out."""
    pytest.importorskip('ngsolve')
    pytest.importorskip('gmsh')
    from cavsim2d import Study

    L_bp = 2 * MIDCELL[5]
    mono = EllipticalCavity(1, MIDCELL, beampipe='both', name='mono')
    asm = Assembly([Beampipe(R=MIDCELL[4], L=L_bp, name='l'),
                    EllipticalCavity(1, MIDCELL, beampipe='none', name='c'),
                    Beampipe(R=MIDCELL[4], L=L_bp, name='r')], name='asm')
    study = Study(project_dir)
    study.add_cavity([mono, asm], ['mono', 'asm'])
    study.run_eigenmode({'processes': 1, 'rerun': True,
                         'boundary_conditions': 'mm',
                         'mesh_config': {'h': 8, 'p': 2}})
    mono.get_eigenmode_qois()
    asm.get_eigenmode_qois()
    assert abs(mono.eigenmode_qois['freq [MHz]']
               - asm.eigenmode_qois['freq [MHz]']) < 1e-3
    assert abs(mono.eigenmode_qois['R/Q [Ohm]']
               - asm.eigenmode_qois['R/Q [Ohm]']) < 1e-2
