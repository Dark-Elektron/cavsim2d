"""Dielectric (eps_r != 1) support in the NGSolve eigenmode solver.

The physics is pinned against closed-form answers, not against previously
recorded numbers:

- a **uniform** fill scales every eigenfrequency by exactly 1/sqrt(eps_r);
- an axially **half-filled** pillbox matches the two-layer TM dispersion
  relation to ~1e-4;
- a **thin** dielectric ring matches Slater cavity-perturbation theory;
- and a region filled with vacuum reproduces the single-domain answer, which is
  what proves the sub-domain split itself is neutral.

Dielectric **loss** is pinned the same way. A uniform lossy fill has a closed-form
Q on both paths — ``1/tan_delta`` perturbatively, ``tan_delta / (2(sqrt(1 +
tan_delta^2) - 1))`` from the exact complex eigenvalue — so neither path is
checked against the other where an anchor exists, and the two are compared only
where the physics says they must agree (small tan_delta) and must not (large).

The last one is the load-bearing regression: a non-conformally glued interface
leaves the sub-domains electrically disconnected, and the solve then returns
plausible-looking spurious modes of the isolated pieces rather than failing.
"""
import json
import os
import warnings

import numpy as np
import pytest

from cavsim2d import Pillbox
from cavsim2d.solvers.NGSolve.eigen_ngsolve import (LOSSY_TAN_DELTA, NGSolveMEVP,
                                                    material_cfs, material_loss_cfs,
                                                    max_tan_delta, parse_materials,
                                                    resolve_loss_model, surface_resistance)

C0 = 299792458.0
R_PB, LZ_PB = 0.10, 0.15


# ── helpers ────────────────────────────────────────────────────────────────

def _split_pillbox(zsplit, maxh=0.008):
    """PEC cylinder cut at ``zsplit`` into 'Domain' and 'ceramic'.

    Built directly in netgen.occ (not through Profile) so the *solver* half of
    the feature is tested independently of the geometry half.
    """
    from netgen.occ import WorkPlane, Glue, OCCGeometry
    from ngsolve import Mesh, BND

    outer = WorkPlane().MoveTo(0, 0).Rectangle(LZ_PB, R_PB).Face()
    if zsplit >= LZ_PB:
        outer.name = 'Domain'
        shape = outer
    else:
        slab = WorkPlane().MoveTo(zsplit, 0).Rectangle(LZ_PB - zsplit, R_PB).Face()
        vac, die = outer - slab, outer * slab
        vac.name, die.name = 'Domain', 'ceramic'
        shape = Glue([vac, die])

    mesh = Mesh(OCCGeometry(shape, dim=2).GenerateMesh(maxh=maxh))
    seen = {}
    for el in mesh.Elements(BND):
        if el.index in seen:
            continue
        vs = [mesh[v].point for v in el.vertices]
        zc = sum(p[0] for p in vs) / len(vs)
        rc = sum(p[1] for p in vs) / len(vs)
        on_wall = abs(rc - R_PB) < 1e-12 or abs(zc) < 1e-12 or abs(zc - LZ_PB) < 1e-12
        seen[el.index] = 'AXI' if abs(rc) < 1e-12 else ('PEC' if on_wall else 'IF')
    for idx, name in seen.items():
        mesh.ngmesh.SetBCName(idx, name)
    return mesh


def _solve(mesh, n=4, materials=None, p=2, m=0, loss_model='lossless'):
    freqs, _, _ = NGSolveMEVP()._solve_modes(mesh, p, m, n, materials=materials,
                                             loss_model=loss_model)
    return np.sort(freqs)[:n]


def _qois(mesh, materials, loss_model, n=3, p=2, m=0):
    """Full per-mode QOIs for a lossy solve, on whichever path *loss_model* picks."""
    solver = NGSolveMEVP()
    freqs, gE, gH = solver._solve_modes(mesh, p, m, n, materials=materials,
                                        loss_model=loss_model)
    return [solver.evaluate_qois(mesh, gE, gH, freqs, m, mode_idx=i, n_cells=1, L=75,
                                 materials=materials, loss_model=loss_model,
                                 q_diel=solver._last_dielectric_q)
            for i in range(n)]


def _uniform(eps_r, tan_delta):
    """materials mapping that fills the whole split pillbox with one lossy medium."""
    return {m: {'eps_r': eps_r, 'tan_delta': tan_delta} for m in ('Domain', 'ceramic')}


def _two_layer_roots(eps2, d1, d2, n=2):
    """Roots of the exact two-layer TM dispersion relation for a pillbox.

        (kz1/eps1) tan(kz1 d1) + (kz2/eps2) tan(kz2 d2) = 0,
        kzi^2 = (w/c0)^2 eps_i - (x01/R)^2
    """
    from scipy.special import jn_zeros
    from scipy.optimize import brentq

    x01 = jn_zeros(0, 1)[0]

    def kz(f, er):
        return np.sqrt((2 * np.pi * f / C0) ** 2 * er - (x01 / R_PB) ** 2 + 0j)

    def disp(f):
        a, b = kz(f, 1.0), kz(f, eps2)
        return (a * np.tan(a * d1) + b / eps2 * np.tan(b * d2)).real

    grid = np.linspace(300e6, 1.4e9, 40000)
    vals = [disp(f) for f in grid]
    roots = []
    for i in range(len(grid) - 1):
        lo, hi = vals[i], vals[i + 1]
        # skip tan() poles, which also change sign
        if np.isfinite(lo) and np.isfinite(hi) and lo * hi < 0 and abs(lo - hi) < 50:
            roots.append(brentq(disp, grid[i], grid[i + 1]) * 1e-6)
    return np.array(roots[:n])


# ── the sub-domain split must be physically neutral ────────────────────────

def test_vacuum_filled_region_reproduces_single_domain():
    """Two materials that are both vacuum must give the single-domain spectrum.

    This is the regression that catches a non-conformal glue: if the interface
    nodes are duplicated the sub-domains decouple, and extra modes appear that
    are resonances of the isolated pieces rather than of the cavity.
    """
    one = _solve(_split_pillbox(LZ_PB))
    two = _solve(_split_pillbox(0.075))
    assert np.allclose(two, one, rtol=1e-4), (one, two)


def test_region_mesh_is_conformal_and_has_no_spurious_modes():
    """The same neutrality, through the *model* API and a thin ring.

    A thin ring is the demanding case: it is where the glue actually failed, and
    a spurious half-wave mode of the isolated bore lands *below* the fundamental,
    so it silently becomes the reported mode of interest.
    """
    solver = NGSolveMEVP()

    plain = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    f_plain = _solve(solver._build_mesh(plain, 6e-3, 3), n=3, p=3)

    ringed = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    ringed.add_dielectric('quartz', 1.0, z=(-1e4, 1e4), r=(2.5, 3.0), maxh=0.3)
    f_ring = _solve(solver._build_mesh(ringed, 6e-3, 3), n=3, p=3,
                    materials=solver.resolve_materials(ringed))

    assert np.allclose(f_ring, f_plain, rtol=1e-4), (f_plain, f_ring)


def test_clockwise_profile_still_glues_conformally():
    """Profiles are traced clockwise by convention; a clockwise OCC face makes
    Glue fail to identify the shared edges. The mesh must come out conformal
    anyway — Profile.mesh() asserts this, so a regression raises here."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    prof = cav.profile()
    assert prof._signed_area() < 0, "pillbox profile is expected to be clockwise"

    prof.add_region('quartz', z=(-1.0, 1.0), r=(2.5e-3, 3.0e-3))
    mesh = prof.mesh(maxh=6e-3, order=1)          # raises if non-conformal
    pts = np.array([mesh[v].point for v in mesh.vertices])
    _, counts = np.unique(np.round(pts, 12), axis=0, return_counts=True)
    assert int((counts > 1).sum()) == 0


# ── closed-form physics ────────────────────────────────────────────────────

@pytest.mark.parametrize('eps_r', [4.0, 9.0])
def test_uniform_fill_scales_every_frequency_by_sqrt_eps(eps_r):
    """A uniformly filled cavity has f = f_vac / sqrt(eps_r), exactly."""
    mesh_v, mesh_d = _split_pillbox(0.075), _split_pillbox(0.075)
    f_vac = _solve(mesh_v)
    f_eps = _solve(mesh_d, materials={'Domain': eps_r, 'ceramic': eps_r})
    assert np.allclose(f_vac / f_eps, np.sqrt(eps_r), rtol=1e-6), (f_vac, f_eps)


def test_half_filled_pillbox_matches_two_layer_dispersion():
    """Axially half-filled: the real dielectric-interface test, against the
    exact transcendental dispersion relation rather than a scaling law."""
    f = _solve(_split_pillbox(0.075), n=2, materials={'ceramic': 4.0})
    roots = _two_layer_roots(4.0, 0.075, 0.075, n=2)
    assert np.allclose(f, roots, rtol=1e-4), (f, roots)


def test_thin_ring_matches_slater_perturbation_theory():
    """A thin dielectric ring on the aperture shifts TM010 by the amount
    first-order cavity-perturbation theory predicts::

        df/f = -(eps_r - 1)/2 * (integral over sample |E|^2) / (integral |E|^2)

    E_z is tangential to the ring, so E (not D) is continuous and the plain
    Slater form applies. 15 % tolerance: the formula is first order and ignores
    the field distortion the ring itself causes.
    """
    from scipy.special import jn_zeros, j1
    solver = NGSolveMEVP()
    eps_r, r0, r1, L_cav, R_cav = 3.8, 2.5e-3, 3.0e-3, 0.100, 0.100

    def tm010(cav):
        return _solve(solver._build_mesh(cav, 6e-3, 3), n=1, p=3,
                      materials=solver.resolve_materials(cav))[0]

    base = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    base.add_dielectric('quartz', 1.0, z=(-1e4, 1e4), r=(2.5, 3.0), maxh=0.3)
    ring = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    ring.add_dielectric('quartz', eps_r, z=(-1e4, 1e4), r=(2.5, 3.0), maxh=0.3)

    f0, f1 = tm010(base), tm010(ring)
    measured = (f1 - f0) / f0

    # Slater: J0 ~ 1 over the ring (x01*r/R << 1); the disc average of J0^2 is J1(x01)^2.
    x01 = jn_zeros(0, 1)[0]
    v_sample = np.pi * (r1 ** 2 - r0 ** 2) * L_cav
    v_cavity = np.pi * R_cav ** 2 * L_cav
    predicted = -(eps_r - 1) / 2 * v_sample / (v_cavity * j1(x01) ** 2)

    assert measured < 0, "a dielectric must lower the resonant frequency"
    assert abs(measured - predicted) < 0.15 * abs(predicted), (measured, predicted)


def test_stored_energy_is_eps_weighted():
    """U must carry eps_r inside the integral. With a uniform fill at fixed field
    normalisation, U scales by eps_r — everything downstream (R/Q, Q, G, Rsh)
    inherits this, so getting it wrong corrupts every figure of merit at once."""
    mesh = _split_pillbox(0.075)
    solver = NGSolveMEVP()

    def energy(materials):
        freqs, gE, gH = solver._solve_modes(mesh, 2, 0, 2, materials=materials)
        # normalise the mode so the comparison is at equal field amplitude
        gE[0].vec.data = 1.0 / np.linalg.norm(gE[0].vec.FV().NumPy()) * gE[0].vec
        q = solver.evaluate_qois(mesh, gE, gH, freqs, mode_idx=0, n_cells=1, L=75,
                                 materials=materials)
        return q['U [J]'], q

    u1, q1 = energy(None)
    u4, _ = energy({'Domain': 4.0, 'ceramic': 4.0})
    assert 'U [J]' in q1 and 'Ploss [W]' in q1
    assert np.isclose(u4 / u1, 4.0, rtol=1e-3), (u1, u4)


def test_per_material_peak_field_is_reported():
    """Epk inside the dielectric is reported separately: E jumps at the
    interface, so the PEC-wall sample cannot see the in-dielectric peak."""
    mesh = _split_pillbox(0.075)
    solver = NGSolveMEVP()
    mats = {'ceramic': 4.0}
    freqs, gE, gH = solver._solve_modes(mesh, 2, 0, 2, materials=mats)
    q = solver.evaluate_qois(mesh, gE, gH, freqs, mode_idx=0, n_cells=1, L=75,
                             materials=mats)
    assert q['Epk_ceramic [MV/m]'] > 0


# ── dielectric loss ────────────────────────────────────────────────────────

@pytest.mark.parametrize('tan_delta,expected', [
    (0.0, 'lossless'),
    (1e-4, 'perturbation'),
    (LOSSY_TAN_DELTA, 'perturbation'),        # the threshold itself is still cheap
    (0.1, 'lossy'),
])
def test_auto_loss_model_follows_the_loss_tangent(tan_delta, expected):
    """With no explicit choice the loss tangent decides, and the threshold is the
    point where perturbation theory stops being trustworthy."""
    mats = {'ceramic': {'eps_r': 3.8, 'tan_delta': tan_delta}}
    assert resolve_loss_model(mats, None) == expected
    assert resolve_loss_model(mats, {'loss_model': 'auto'}) == expected


def test_explicit_lossless_is_honoured_but_warns_when_it_should_not_be():
    """The user's choice wins — a warning, not a refusal, and never a silent
    downgrade of a lossy material to a lossless answer."""
    lossy = {'ceramic': {'eps_r': 3.8, 'tan_delta': 0.1}}
    with pytest.warns(UserWarning, match='no longer reliable'):
        assert resolve_loss_model(lossy, {'loss_model': 'lossless'}) == 'perturbation'
    # below the threshold there is nothing to warn about
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert resolve_loss_model({'ceramic': {'eps_r': 3.8, 'tan_delta': 1e-4}},
                                  {'loss_model': 'lossless'}) == 'perturbation'
    # and the expensive path on a lossless material is pointless, so it says so
    with pytest.warns(UserWarning, match='no material has a loss tangent'):
        assert resolve_loss_model({'ceramic': 3.8}, {'loss_model': 'lossy'}) == 'lossless'
    with pytest.raises(ValueError, match="not one of"):
        resolve_loss_model(lossy, {'loss_model': 'perturbative'})


def test_perturbative_q_of_a_uniform_fill_is_one_over_tan_delta():
    """The filling factor is 1, so Q_diel = int eps'|E|^2 / int eps''|E|^2 = 1/tan_delta
    for EVERY mode — no mode shape enters. Exact, not approximate."""
    mesh = _split_pillbox(0.075)
    qs = _qois(mesh, _uniform(4.0, 1e-3), 'perturbation')
    for q in qs:
        assert q['Q model'] == 'perturbation'
        assert q['Q_diel []'] == pytest.approx(1e3, rel=1e-6)
        assert q['tan_delta []'] == pytest.approx(1e-3)
        # total Q is the parallel combination of wall and dielectric
        assert 1 / q['Q []'] == pytest.approx(1 / q['Q_wall []'] + 1 / q['Q_diel []'],
                                              rel=1e-9)
        # G is a wall quantity by definition: it must follow Q_wall, not the
        # dielectric-loaded total Q.
        w = 2 * np.pi * q['freq [MHz]'] * 1e6
        assert q['G [Ohm]'] == pytest.approx(q['Q_wall []'] * surface_resistance(w))
        assert q['G [Ohm]'] > q['Q []'] * surface_resistance(w)


def test_lossy_q_of_a_uniform_fill_matches_the_exact_complex_eigenvalue():
    """A uniform fill scales lambda by exactly 1/(eps'(1 - j tan_delta)), so

        Q = Re(sqrt(lam)) / (2 Im(sqrt(lam))) = t / (2 (sqrt(1 + t^2) - 1))

    for every mode. This pins the complex solver against closed form — not
    against the perturbative answer, which it is allowed to disagree with.
    """
    t = 0.2
    mesh = _split_pillbox(0.075)
    exact = t / (2 * (np.sqrt(1 + t ** 2) - 1))
    qs = _qois(mesh, _uniform(4.0, t), 'lossy')
    for q in qs:
        assert q['Q model'] == 'lossy'
        assert q['Q_diel []'] == pytest.approx(exact, rel=1e-6), q['Q_diel []']
    # ... and it is NOT 1/t: that difference is exactly what the lossy path buys.
    assert abs(qs[0]['Q_diel []'] - 1 / t) > 1e-3 * exact


def test_lossy_frequency_of_a_uniform_fill_matches_closed_form():
    """Re(f) of the lossy solve follows |eps|: f = f_vac / Re(sqrt(eps'(1 - jt)))."""
    t = 0.2
    mesh = _split_pillbox(0.075)
    f_vac = _solve(mesh, n=3)
    f_lossy = np.array([q['freq [MHz]'] for q in _qois(mesh, _uniform(4.0, t), 'lossy')])
    expected = f_vac * (1 / np.sqrt(4.0 * (1 - 1j * t))).real
    assert np.allclose(np.sort(f_lossy), np.sort(expected), rtol=1e-5), (f_lossy, expected)


def test_paths_agree_at_small_tan_delta_and_separate_at_large():
    """The whole justification for keeping two paths: they must be the same
    answer where perturbation theory holds, and different where it does not."""
    mesh = _split_pillbox(0.075)

    small = _uniform(4.0, 1e-4)
    qp = _qois(mesh, small, 'perturbation', n=2)
    ql = _qois(mesh, small, 'lossy', n=2)
    for a, b in zip(qp, ql):
        assert b['Q_diel []'] == pytest.approx(a['Q_diel []'], rel=1e-6)
        assert b['freq [MHz]'] == pytest.approx(a['freq [MHz]'], rel=1e-8)

    big = _uniform(4.0, 0.3)
    qp = _qois(mesh, big, 'perturbation', n=2)
    ql = _qois(mesh, big, 'lossy', n=2)
    for a, b in zip(qp, ql):
        # perturbation misses both the Q correction and the frequency pull
        assert abs(b['Q_diel []'] - a['Q_diel []']) > 0.01 * a['Q_diel []']
        assert abs(b['freq [MHz]'] - a['freq [MHz]']) > 1e-4 * a['freq [MHz]']


def test_a_half_filled_lossy_cavity_beats_its_filling_factor():
    """Only part of the energy sits in the lossy medium, so Q_diel must exceed
    1/tan_delta — the sanity check that the loss integral is weighted by the
    field and not by the volume."""
    mesh = _split_pillbox(0.075)
    mats = {'ceramic': {'eps_r': 3.8, 'tan_delta': 0.01}}
    for model in ('perturbation', 'lossy'):
        q = _qois(mesh, mats, model, n=1)[0]
        assert q['Q_diel []'] > 1 / 0.01, (model, q['Q_diel []'])
        assert q['Pdiel [W]'] > 0


def test_absolute_qois_are_reproducible_and_share_the_lossless_scale():
    """An eigenvector has no intrinsic amplitude, so every absolute QOI is reported
    at whatever scale the eigensolver returned. PINVIT normalises; Arnoldi does
    not, and its scale is not even stable between two runs of the same problem.

    Unnormalised, this bites twice over: `Pdiel`, `U`, `Ploss` and `Epk` change
    run to run on the lossy path, and they disagree with the lossless path even at
    tan_delta -> 0, where the two solve the same cavity. Both are silent — every
    ratio (Q, Q_diel, R/Q, G) is scale-free and looks perfectly correct.
    """
    mesh = _split_pillbox(0.075)
    absolute = ('U [J]', 'Ploss [W]', 'Pdiel [W]', 'Epk [MV/m]', 'Hpk [A/m]')

    # (a) two lossy runs of the same problem agree
    tiny = _uniform(4.0, 1e-4)
    a, b = _qois(mesh, tiny, 'lossy', n=2), _qois(mesh, tiny, 'lossy', n=2)
    for qa, qb in zip(a, b):
        for key in absolute:
            assert qa[key] == pytest.approx(qb[key], rel=1e-9), key

    # (b) and they agree with the lossless path, which is the same cavity at this
    # loss tangent
    ref = _qois(mesh, tiny, 'perturbation', n=2)
    for qa, qr in zip(a, ref):
        for key in absolute:
            assert qa[key] == pytest.approx(qr[key], rel=1e-6), key

    # (c) the shared convention is the real path's: int eps_r' |E|^2 r dA = 1, so
    # U = 0.5 * eps0 * 2pi for the monopole regardless of mode, mesh or material
    for q in a + ref:
        assert q['U [J]'] == pytest.approx(0.5 * 8.85418782e-12 * 2 * np.pi, rel=1e-9)

    # (d) at real loss the absolute quantities are allowed - required - to differ,
    # by the same margin the Q does
    big = _uniform(4.0, 0.05)
    ql = _qois(mesh, big, 'lossy', n=1)[0]
    qp = _qois(mesh, big, 'perturbation', n=1)[0]
    assert ql['Pdiel [W]'] != pytest.approx(qp['Pdiel [W]'], rel=1e-6)
    assert ql['U [J]'] == pytest.approx(qp['U [J]'], rel=1e-9)   # normalisation holds


def test_a_lossless_run_reports_no_loss_bookkeeping():
    """The default path must be untouched: no Q_diel, no Pdiel, and Q is the wall
    Q it always was."""
    mesh = _split_pillbox(0.075)
    q = _qois(mesh, {'ceramic': 3.8}, 'lossless', n=1)[0]
    assert q['Q model'] == 'lossless'
    for key in ('Q_diel []', 'Q_wall []', 'Pdiel [W]', 'tan_delta []'):
        assert key not in q
    assert q['Q []'] == pytest.approx(2 * np.pi * q['freq [MHz]'] * 1e6 *
                                      q['U [J]'] / q['Ploss [W]'])

    # a vacuum cavity gets no material keys at all, not even the tag
    solver = NGSolveMEVP()
    freqs, gE, gH = solver._solve_modes(mesh, 2, 0, 1)
    plain = solver.evaluate_qois(mesh, gE, gH, freqs, mode_idx=0, n_cells=1, L=75)
    assert 'Q model' not in plain


def test_lossy_fields_round_trip_through_disk(tmp_path):
    """Complex coefficient vectors must survive save/load: rebuilding them on a
    REAL space would drop the phase, and with it the loss."""
    mesh = _split_pillbox(0.075)
    solver = NGSolveMEVP()
    mats = _uniform(4.0, 0.2)
    folder = str(tmp_path)
    freqs, gE, _ = solver._solve_modes(mesh, 2, 0, 2, save_dir=folder,
                                       materials=mats, loss_model='lossy')
    with open(os.path.join(folder, 'field_meta.json')) as f:
        assert json.load(f)['complex'] is True

    gE2, gH2 = solver.load_fields(folder, mode=1)
    assert gE2[0].space.is_complex
    q = solver.evaluate_qois(solver.load_mesh(folder), gE2, gH2, freqs, mode_idx=0,
                             n_cells=1, L=75, materials=mats, loss_model='lossy',
                             q_diel=solver._last_dielectric_q)
    assert q['Q model'] == 'lossy'
    assert np.iscomplexobj(np.asarray(gE2[0].vec.FV().NumPy()))


def test_config_can_force_either_path_end_to_end(tmp_path, monkeypatch):
    """The knob users actually turn: eigenmode_config['loss_model'] on a run,
    with tan_delta coming from the model API."""
    monkeypatch.chdir(tmp_path)
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.set_name('lossycav')
    cav.add_dielectric('quartz', 9.8, tan_delta=0.05, z=(-1e4, 1e4), r=(2.5, 3.0),
                       maxh=0.5)
    assert cav.dielectrics[0]['tan_delta'] == 0.05
    mats = NGSolveMEVP.resolve_materials(cav)
    assert mats == {'quartz': {'eps_r': 9.8, 'tan_delta': 0.05}}
    # auto picks the complex solver above the threshold
    assert resolve_loss_model(mats, None) == 'lossy'

    cav.eigenmode.run(polarisation='monopole', n_modes=2, loss_model='lossless',
                      mesh_config={'h': 8, 'p': 2})
    q = cav.eigenmode.qois
    assert q['Q model'] == 'perturbation'
    assert q['Q_diel []'] > 0


def test_config_materials_can_sweep_tan_delta_without_touching_geometry():
    """A loss sweep is a config override, and it must MERGE onto the declared
    permittivity rather than resetting it to vacuum."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0))
    got = NGSolveMEVP.resolve_materials(cav, {'materials': {'quartz': {'tan_delta': 0.2}}})
    assert got == {'quartz': {'eps_r': 3.8, 'tan_delta': 0.2}}
    # and a complex override supersedes a declared tan_delta instead of colliding
    cav.clear_dielectrics()
    cav.add_dielectric('quartz', 3.8, tan_delta=0.01, z=(-1e4, 1e4), r=(2.5, 3.0))
    got = NGSolveMEVP.resolve_materials(cav, {'materials': {'quartz': 3.8 - 0.76j}})
    assert parse_materials(got) == {'quartz': (3.8, 0.76)}


def test_tan_delta_is_a_uq_variable_and_invalidates_the_cache():
    """Loss is a perturbable property like any other, and changing it must
    invalidate cached results — a rerun that served the old Q would look right."""
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    cav.add_dielectric('quartz', 4.4, tan_delta=1e-3, z=(-1e4, 1e4), r=(2.0, 2.5))
    assert 'quartz:tan_delta' in cav.dielectric_variables()
    assert cav.dielectric_values()['quartz:tan_delta'] == 1e-3
    cav.set_tune_value('quartz:tan_delta', 0.02)
    assert cav.get_tune_value('quartz:tan_delta') == 0.02
    assert cav._dielectric_signature()['_dielectric:quartz:tan_delta'] == 0.02
    with pytest.raises(ValueError, match='gain'):
        cav.set_dielectric_value('quartz:tan_delta', -1e-4)
    with pytest.raises(ValueError, match='gain'):
        cav.add_dielectric('bad', 4.0, tan_delta=-1e-4, z=(-1, 1), r=(1, 2))


def test_loss_model_config_key_is_validated():
    from cavsim2d.utils.config_validation import validate_eigenmode_config
    validate_eigenmode_config({'loss_model': 'lossy'})
    validate_eigenmode_config({'loss_model': None})
    with pytest.raises(ValueError, match='loss_model'):
        validate_eigenmode_config({'loss_model': 'perturbative'})


# ── loud failures ──────────────────────────────────────────────────────────

def test_unknown_material_name_raises():
    """A typo must not silently fall back to vacuum."""
    mesh = _split_pillbox(0.075)
    with pytest.raises(ValueError, match='not a region of this mesh'):
        material_cfs(mesh, {'quatrz': 3.8})


def test_magnetic_properties_are_still_rejected():
    """mu_r is refused rather than ignored: it changes the stiffness form, not
    just the mass form, so accepting it would report a plausible wrong answer."""
    mesh = _split_pillbox(0.075)
    with pytest.raises(NotImplementedError, match='mu_r'):
        material_cfs(mesh, {'ceramic': {'eps_r': 3.8, 'mu_r': 2.0}})


def test_complex_permittivity_spellings_agree_and_conflicts_raise():
    """eps_r*(1 - j tan_delta) can be written either way, but not both ways at
    once — two spellings of one quantity in one entry is an input error, not a
    thing to silently pick a winner for."""
    assert parse_materials({'c': {'eps_r': 4.0, 'tan_delta': 0.05}}) == {'c': (4.0, 0.2)}
    assert parse_materials({'c': 4.0 - 0.2j}) == {'c': (4.0, 0.2)}
    # exp(+jwt) makes the loss term negative-imaginary; the sign of the shorthand
    # is not load-bearing, the modulus is.
    assert parse_materials({'c': 4.0 + 0.2j}) == {'c': (4.0, 0.2)}
    assert max_tan_delta({'c': {'eps_r': 4.0, 'tan_delta': 0.05}}) == pytest.approx(0.05)
    assert max_tan_delta({'a': 4.0, 'b': {'eps_r': 2.0, 'tan_delta': 0.3}}) == pytest.approx(0.3)
    assert max_tan_delta(None) == 0.0

    with pytest.raises(ValueError, match='BOTH'):
        parse_materials({'c': {'eps_r': 4.0 - 0.2j, 'tan_delta': 0.05}})
    with pytest.raises(ValueError, match='gain'):
        parse_materials({'c': {'eps_r': 4.0, 'tan_delta': -1e-3}})
    with pytest.raises(ValueError, match='positive'):
        parse_materials({'c': {'eps_r': -4.0}})


def test_loss_coefficient_is_none_without_loss():
    """No loss tangent anywhere -> None, which is the signal that keeps a
    lossless run's QOIs byte-identical to what they always were."""
    mesh = _split_pillbox(0.075)
    assert material_loss_cfs(mesh, None) is None
    assert material_loss_cfs(mesh, {'ceramic': 3.8}) is None
    assert material_loss_cfs(mesh, {'ceramic': {'eps_r': 3.8, 'tan_delta': 0.0}}) is None
    assert material_loss_cfs(mesh, {'ceramic': {'eps_r': 3.8, 'tan_delta': 1e-4}}) is not None
    # the real part is unaffected by the loss tangent
    assert material_cfs(mesh, {'ceramic': {'eps_r': 1.0, 'tan_delta': 0.5}}) is None


def test_all_vacuum_materials_return_no_coefficient_function():
    """eps_r == 1 everywhere must return None so the vacuum path assembles the
    exact expressions it always did."""
    mesh = _split_pillbox(0.075)
    assert material_cfs(mesh, None) is None
    assert material_cfs(mesh, {'ceramic': 1.0}) is None
    assert material_cfs(mesh, {'ceramic': 2.0}) is not None


def test_config_materials_must_name_a_declared_region():
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0))
    assert NGSolveMEVP.resolve_materials(cav)['quartz'] == {'eps_r': 3.8}
    # override an existing region: allowed
    got = NGSolveMEVP.resolve_materials(cav, {'materials': {'quartz': {'eps_r': 4.5}}})
    assert got['quartz']['eps_r'] == 4.5
    # name one that was never declared: refused
    with pytest.raises(ValueError, match='not a dielectric region'):
        NGSolveMEVP.resolve_materials(cav, {'materials': {'alumina': 9.8}})


def test_duplicate_and_degenerate_regions_raise():
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0))
    with pytest.raises(ValueError, match='already has a dielectric region'):
        cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0))
    with pytest.raises(ValueError, match='zero extent'):
        cav.add_dielectric('slab', 3.8, z=(0, 0), r=(2.5, 3.0))
    with pytest.raises(ValueError, match='positive'):
        cav.add_dielectric('bad', -1.0, z=(-1, 1), r=(2.5, 3.0))


def test_vacuum_only_solvers_refuse_a_dielectric_cavity():
    """Wakefield and multipacting have no material model; they must raise rather
    than return a vacuum answer for a dielectric-loaded cavity."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.5, 3.0))
    from cavsim2d.solvers.solver_objects import _require_vacuum
    for analysis in ('wakefield', 'multipacting'):
        with pytest.raises(NotImplementedError, match='does not model'):
            _require_vacuum(cav, analysis)
    cav.clear_dielectrics()
    _require_vacuum(cav, 'wakefield')          # no longer raises


def test_dielectrics_invalidate_the_geometry_snapshot():
    """Changing eps_r must be visible to the cache check, or a rerun=False run
    silently serves the previous (vacuum) results."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    assert cav._dielectric_signature() == {}
    cav.add_dielectric('quartz', 3.8, z=(-70.0, 70.0), r=(2.5, 3.0))
    sig = cav._dielectric_signature()
    assert sig['_dielectric:quartz:eps_r'] == 3.8
    assert sig['_dielectric:quartz:r0'] == 2.5 and sig['_dielectric:quartz:r1'] == 3.0

    other = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    other.add_dielectric('quartz', 4.5, z=(-70.0, 70.0), r=(2.5, 3.0))
    assert other._dielectric_signature() != sig


def test_solved_dielectric_fields_reload(tmp_path, monkeypatch):
    """A full run must persist and reload. The saved mesh has to carry its
    material names through the .vol round-trip, or rebuilding the system on
    reload raises 'not a region of this mesh'."""
    monkeypatch.chdir(tmp_path)
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    cav.set_name('quartz_rt')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.2)
    cav.eigenmode.run(mesh_config={'h': 4, 'p': 3}, n_modes=2)

    q = cav.eigenmode.qois
    assert q['freq [MHz]'] > 100
    assert q['Epk_quartz [MV/m]'] > 0
    assert q['U [J]'] > 0

    folder = os.path.join(cav.eigenmode.folder, 'monopole')
    meta = json.load(open(os.path.join(folder, 'field_meta.json')))
    assert meta['materials'] == {'quartz': {'eps_r': 3.8}}

    mesh = NGSolveMEVP().load_mesh(folder)
    assert 'quartz' in mesh.GetMaterials()
    gfu_E, gfu_H = NGSolveMEVP().load_fields(folder, 0)
    assert len(gfu_E) == len(gfu_H) > 0


def test_dielectrics_survive_spawn_and_clone(tmp_path):
    """rebuild() works from the parameter dict, which does not contain the
    regions — so every generic clone path has to carry them, or a sweep or UQ
    study over a loaded cavity silently becomes a vacuum one."""
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.2)

    bare = cav.rebuild(dict(cav.parameters))
    assert bare.dielectrics == []                 # rebuild alone cannot know
    assert cav._carry_dielectrics_to(bare).dielectrics == cav.dielectrics

    clone = cav.clone_for_tuning(dict(cav.parameters), str(tmp_path / 'tuned'))
    assert clone.dielectrics == cav.dielectrics


def test_dielectric_uq_variables_are_declared_and_addressable():
    """Regions declare their own '<material>:<field>' handles, the same way the
    geometry declares its parameters — nothing central to update."""
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    assert cav.dielectric_variables() == set()
    cav.add_dielectric('quartz', 4.4, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.2)

    assert cav.dielectric_variables() == {
        'quartz:eps_r', 'quartz:tan_delta', 'quartz:r_in', 'quartz:r_out',
        'quartz:wall', 'quartz:z_lo', 'quartz:z_hi'}
    assert cav.dielectric_values()['quartz:wall'] == 0.5
    assert cav.expand_variable('quartz:eps_r') == ['quartz:eps_r']

    # the generic accessors UQ drives must route to the region
    cav.set_tune_value('quartz:eps_r', 4.6)
    assert cav.get_tune_value('quartz:eps_r') == 4.6

    # 'wall' holds the outer radius and moves the inner one: a tube in a bore has
    # its OD set by the aperture, so it is the wall that varies.
    cav.set_tune_value('quartz:wall', 0.55)
    assert cav.dielectrics[0]['r'] == (1.95, 2.5)

    with pytest.raises(ValueError, match='Unknown variable'):
        cav.expand_variable('quartz:epsr')            # typo, not silently ignored
    with pytest.raises(ValueError, match='wall'):
        cav.set_dielectric_value('quartz:wall', 3.0)  # thicker than the radius


def test_dielectric_uq_perturbs_and_spawns(tmp_path, monkeypatch):
    """A UQ over eps_r and wall must actually vary them in the spawned cavities.

    The regression this guards: dielectric values live outside `cav.parameters`,
    so `shapes_to_dataframe` and `spawn` both had to learn about them. Miss
    either and the UQ runs happily on N identical cavities and reports a
    standard deviation of zero.
    """
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    cav.set_name('uqdiel')
    cav.add_dielectric('quartz', 4.4, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.25)
    cav.eigenmode.run(polarisation='monopole', n_modes=3,
                      mesh_config={'h': 5, 'p': 3},
                      uq_config={'variables': ['quartz:eps_r', 'quartz:wall'],
                                 'objectives': ['monopole:freq [MHz]'],
                                 'delta': [0.1, 0.01], 'method': ['stroud3'],
                                 'perturbation_mode': ['add', [0.1, 0.01]]})

    nodes = pd.read_csv(os.path.join(cav.uq_dir, 'nodes.csv'), sep='\t')
    assert {'quartz:eps_r', 'quartz:wall'} <= set(nodes.columns)
    assert nodes['quartz:eps_r'].nunique() > 1
    assert nodes['quartz:wall'].nunique() > 1

    uq = json.load(open(os.path.join(cav.uq_dir, 'uq.json')))
    stats = uq['monopole:freq [MHz]']
    assert stats['expe'][0] > 100
    assert stats['stdDev'][0] > 1e-3, 'perturbations did not reach the solver'


def test_dielectrics_survive_a_state_round_trip():
    """Study.load rebuilds cavities from _reconstruct_state; the regions have to
    come back with them or a reloaded study silently becomes vacuum."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-70.0, 70.0), r=(2.5, 3.0), maxh=0.3)
    state = cav._reconstruct_state()
    back = type(cav)._reconstruct_from_state(state)
    assert back.dielectrics == cav.dielectrics


def test_dielectric_face_colour_defaults_yellow_and_multiple_regions():
    """add_dielectric colours each region's face — default yellow (1, 1, 0),
    overridable with an (r, g, b) tuple — and the colour reaches the drawn mesh.
    More than one dielectric can be added; each keeps its own material and colour."""
    from ngsolve.webgui import Draw
    cav = Pillbox(1, [20, 37.5, 2.5, 0, 5], beampipe='both')
    cav.add_dielectric('quartz', 4.5, z=(-1e4, 1e4), r=(2.0, 2.5), maxh=0.2)   # default
    cav.add_dielectric('ceramic', 9.8, z=(-3.0, 3.0), r=(5.0, 7.0), maxh=0.4,
                       color=(0.1, 0.3, 1.0))                                   # override
    assert cav.dielectrics[0]['color'] == (1.0, 1.0, 0.0)
    assert cav.dielectrics[1]['color'] == (0.1, 0.3, 1.0)

    mesh = NGSolveMEVP()._build_mesh(cav, 4e-3, 1)
    assert {'quartz', 'ceramic'}.issubset(set(mesh.GetMaterials()))

    # the colours reach the webgui render (Draw(mesh) paints each region's face)
    data = Draw(mesh, show=False).GetData()
    by_name = {}
    for name, col in zip(data['names'], data['colors']):
        by_name.setdefault(name, tuple(round(c, 3) for c in col[:3]))
    assert by_name['quartz'] == (1.0, 1.0, 0.0)         # yellow, the default
    assert by_name['ceramic'] == (0.1, 0.3, 1.0)        # the per-region override


def test_dielectric_colour_survives_state_round_trip():
    """A custom dielectric colour must come back through _reconstruct_state
    (Study.load), not silently revert to the yellow default."""
    cav = Pillbox(1, [100, 100, 5, 0, 20], beampipe='both')
    cav.add_dielectric('quartz', 3.8, z=(-70.0, 70.0), r=(2.5, 3.0),
                       color=(0.9, 0.2, 0.1))
    back = type(cav)._reconstruct_from_state(cav._reconstruct_state())
    assert back.dielectrics[0]['color'] == (0.9, 0.2, 0.1)
