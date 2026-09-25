"""Wakefield smoke test (Windows + bundled ABCI.exe only): ABCI runs and
produces longitudinal + transversal output that the reader can load."""
import os

import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL, requires_abci
from cavsim2d import Study, EllipticalCavity


@requires_abci
def test_wakefield_runs_and_writes_output(project_dir):
    cavs = Study(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['WF'])

    cavs.run_wakefield({'processes': 1, 'rerun': True,
                        'wakelength': 5, 'bunch_length': 25})

    wf = os.path.join(cav.self_dir, 'wakefield')
    assert os.path.exists(os.path.join(wf, 'longitudinal', 'cavity.top'))
    assert os.path.exists(os.path.join(wf, 'transversal', 'cavity.top'))

    # normalised impedance loads and plots without error, via cav.wakefield
    assert not cav.wakefield.wake_z.empty
    ax = cav.wakefield.plot_impedance()
    assert ax is not None
    assert cav.wakefield.qois['|k_loss| [V/pC]'] != 0

    # the cumulative loss-factor SPECTRUM k_loss(F) vs frequency is captured as
    # its own (fk, k_loss(f)) columns and plotted by plot_k_loss — distinct from
    # the scalar |k_loss| above (the frequency-resolved curve the user wanted)
    import numpy as np
    wz = cav.wakefield.wake_z
    assert 'fk [MHz]' in wz.columns
    kcol = next(c for c in wz.columns if c.startswith('k_loss(f)'))
    sub = wz[['fk [MHz]', kcol]].dropna().sort_values('fk [MHz]')
    assert len(sub) > 10
    # cumulative curve ends near the total loss factor (integrated up to f_max)
    assert sub[kcol].iloc[-1] == pytest.approx(cav.wakefield.qois['|k_loss| [V/pC]'],
                                               rel=0.25)
    kax = cav.wakefield.plot_k_loss()
    assert kax is not None and kax.lines


@requires_abci
def test_pillbox_wakefield_end_to_end(project_dir):
    """Non-elliptical geometries run through ABCI too (P2-4)."""
    from cavsim2d import Study, Pillbox
    cavs = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')   # L_bp = 50 mm
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 10})
    top = os.path.join(pb.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
    assert os.path.getsize(top) > 0
    assert not pb.wakefield.wake_z.empty
    assert pb.wakefield.qois['|k_loss| [V/pC]'] != 0


def test_abci_abort_raises_instead_of_silently_producing_nothing(tmp_path):
    """ABCI exits 0 even when it refuses to run, leaving empty .pot/.top files.

    Beam pipes are now added automatically, so the old trigger (a pillbox with
    L_bp = 0) no longer aborts. The guard itself is what matters, so drive it
    directly with a log ABCI would have written.
    """
    from cavsim2d.solvers.ABCI.abci import _raise_if_abci_aborted

    (tmp_path / 'cavity.out').write_text(
        ' NUMBER OF MESH LINES IN R      : NR   =         84\n'
        '0*** STOP *** THE BEAM PIPES AT BOTH ENDS ARE TOO SHORT.\n'
        ' THEY MUST HAVE AT LEAST 5 MESH LENGTH.\n')
    with pytest.raises(RuntimeError, match='ABCI refused to run'):
        _raise_if_abci_aborted(str(tmp_path))

    # a clean log passes through
    (tmp_path / 'cavity.out').write_text(' NORMAL COMPLETION\n')
    _raise_if_abci_aborted(str(tmp_path))


@requires_abci
def test_pillbox_without_beampipe_now_runs(project_dir):
    """L_bp = 0 used to make ABCI refuse; the pipes are added for it now."""
    from cavsim2d import Study, Pillbox
    cavs = Study(project_dir)
    pb = Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')
    cavs.add_cavity([pb], ['PB'])
    cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
    top = os.path.join(pb.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
    assert os.path.getsize(top) > 0


# --- seam 2: the wakefield contour comes from Profile, not the .geo text -----

def test_contour_points_are_accurate():
    """Curved segments are sampled to the requested spacing."""
    import numpy as np
    from scipy.special import ellipe
    from cavsim2d.geometry import Profile

    a, b = 0.05, 0.03
    p = Profile().start(a, 0.0)
    p.ellipse_arc_to(0.0, b, center=(0.0, 0.0), semi_z=a, semi_r=b, boundary='PEC')
    p.line_to(0.0, 0.0, 'AXI')
    p.close('PMC')
    pts = np.asarray(p.contour_points(1e-4, skip=('AXI', 'PMC')))
    sampled = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    exact = a * ellipe(1 - (b / a) ** 2)          # quarter-ellipse perimeter
    assert abs(sampled - exact) / exact < 1e-5


def test_beampipes_are_added_only_where_missing():
    """ABCI needs a pipe at each end; cavities that have one are left alone."""
    import numpy as np
    from cavsim2d import EllipticalCavity, Pillbox, RFGun
    from cavsim2d.geometry.beampipes import abci_contour, beampipe_lengths

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ddz = 1.25e-3
    min_pipe = 6 * ddz

    prof = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both').profile()
    _, al, ar = abci_contour(prof, ddz, min_pipe)
    assert (al, ar) == (0.0, 0.0)                 # already has pipes -> untouched

    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')
    prof = cav.profile()
    raw = prof.contour_points(ddz, skip=('AXI',))
    axial = max(z for z, _ in raw) - min(z for z, _ in raw)
    pts, al, ar = abci_contour(prof, ddz, min_pipe)
    assert al == pytest.approx(3 * axial)         # default: 3x the axial length
    assert ar == pytest.approx(3 * axial)
    left, right = beampipe_lengths(pts)
    assert left >= min_pipe and right >= min_pipe
    assert abs(pts[0][1]) < 1e-12 and abs(pts[-1][1]) < 1e-12   # still ends on the axis

    # a gun already has a downstream drift tube: only the cathode side gets a pipe
    gun = RFGun({'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                              'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                              'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                              'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}})
    _, al, ar = abci_contour(gun.profile(), ddz, min_pipe)
    assert al > 0 and ar == 0.0

    _, al, ar = abci_contour(Pillbox(1, [100, 100, 20, 0, 0], beampipe='none').profile(),
                             ddz, min_pipe)
    assert al > 0 and ar > 0


def test_abci_shape_keeps_arcs():
    """ABCI emits NaN wake potentials when an arc meeting a beam pipe tangentially
    (an elliptical iris) is handed to it as a dense polyline."""
    from cavsim2d import EllipticalCavity, Pillbox
    from cavsim2d.geometry.beampipes import abci_shape

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    items, _, _ = abci_shape(EllipticalCavity(1, tesla, tesla, tesla, beampipe='both').profile(),
                             1.25e-3, 7.5e-3)
    assert [it[0] for it in items].count('arc') == 4     # 2 iris + 2 equator

    items, _, _ = abci_shape(Pillbox(1, [100, 100, 20, 0, 50], beampipe='both').profile(),
                             1.25e-3, 7.5e-3)
    assert all(it[0] == 'point' for it in items)         # all straight lines


@requires_abci
def test_wakefield_matches_the_legacy_geo_path(project_dir):
    """The Profile-based deck reproduces the old .geo-parsed deck exactly."""
    from cavsim2d import Study, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def k_loss(legacy):
        cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
        cavs = Study(os.path.join(project_dir, 'legacy' if legacy else 'profile'))
        cavs.add_cavity([cav], ['C'])
        original = EllipticalCavity.profile
        if legacy:
            EllipticalCavity.profile = lambda self: None
        try:
            cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
            top = os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
            with open(top, errors='replace') as fh:
                assert 'NaN' not in fh.read()
            return cav.wakefield.qois['|k_loss| [V/pC]']
        finally:
            EllipticalCavity.profile = original

    assert k_loss(False) == pytest.approx(k_loss(True), abs=1e-9)


@requires_abci
def test_wakefield_runs_for_every_cavity_type(project_dir):
    """Flat-top, spline and gun had no working wakefield: the flat top writes no
    .geo at all, and the spline wall is a curve the old regex could not see."""
    import numpy as np
    from cavsim2d import (Study, EllipticalCavity, Pillbox, RFGun,
                                 SplineCavity, EllipticalCavityFlatTop)
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    ft = tesla + [20]
    gun = {'geometry': {'y1': 1.5e-2, 'R2': 3e-2, 'T2': np.deg2rad(45), 'L3': 24e-2,
                        'R4': 5e-2, 'L5': 11e-2, 'R6': 6e-2, 'L7': 19e-2, 'R8': 4e-2,
                        'T9': np.deg2rad(8), 'R10': 3e-2, 'T10': np.deg2rad(40),
                        'L11': 5e-2, 'R12': 3e-2, 'L13': 3e-2, 'R14': 3e-2, 'x': 1e-2}}
    geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
            'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}

    cases = [('PB', Pillbox(1, [100, 100, 20, 0, 0], beampipe='none')),
             ('ELN', EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')),
             ('FT', EllipticalCavityFlatTop(1, ft, ft, ft, beampipe='both')),
             ('SC', SplineCavity({'geometry': dict(geom)}, kind='Bezier')),
             ('GUN', RFGun(gun))]
    for name, cav in cases:
        cavs = Study(os.path.join(project_dir, name))
        cavs.add_cavity([cav], [name])
        cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
        top = os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'cavity.top')
        assert os.path.getsize(top) > 0
        with open(top, errors='replace') as fh:
            assert 'NaN' not in fh.read(), name + ': ABCI produced NaN wake potentials'
        assert cav.wakefield.qois['|k_loss| [V/pC]'] != 0


@requires_abci
def test_added_beampipes_do_not_change_the_cavity_wake(project_dir):
    """The same elliptical cavity with beampipe='both' and with beampipe='none'
    (pipes added automatically) must give the same loss factor."""
    from cavsim2d import Study, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]

    def k(bp, tag):
        cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe=bp)
        cavs = Study(os.path.join(project_dir, tag))
        cavs.add_cavity([cav], [tag])
        cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 0, 'wakelength': 3})
        return cav.wakefield.qois['|k_loss| [V/pC]']

    assert k('both', 'both') == pytest.approx(k('none', 'none'), rel=2e-3)


# --- the wakefield seam is solver-agnostic ----------------------------------

def test_backend_registry():
    from cavsim2d.solvers.wakefield import get_backend, BACKENDS
    assert 'abci' in BACKENDS
    assert get_backend('abci').name == 'abci'
    assert get_backend(None).name == 'abci'          # default
    with pytest.raises(ValueError, match='Unknown wakefield solver'):
        get_backend('does-not-exist')


@requires_abci
def test_run_writes_the_normalised_schema(project_dir):
    """A run persists wakefield/<pol>/qois.json, read back via cav.wakefield.qois."""
    import json
    from cavsim2d import Study, EllipticalCavity
    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
    cavs = Study(project_dir)
    cavs.add_cavity([cav], ['C'])
    cavs.run_wakefield({'processes': 1, 'rerun': True, 'MROT': 2, 'wakelength': 3})

    ljson = os.path.join(cav.self_dir, 'wakefield', 'longitudinal', 'qois.json')
    assert os.path.exists(ljson)
    with open(ljson) as fh:
        assert '|k_loss| [V/pC]' in json.load(fh)
    # the main run's loss/kick factors are always reported (merged qois.json)
    assert os.path.exists(os.path.join(cav.self_dir, 'wakefield', 'qois.json'))
    q = cav.wakefield.qois
    assert q['|k_loss| [V/pC]'] != 0 and '|k_kick| [V/pC/m]' in q
    # no operating points -> no qois_op
    assert not os.path.exists(os.path.join(cav.self_dir, 'wakefield', 'qois_op.json'))
    assert cav.wakefield.qois_op == {}


class _DummyWakefield:
    """A canned backend that produces the normalised schema without any solver —
    used to prove the seam is genuinely solver-agnostic."""
    name = 'dummy'

    @staticmethod
    def wakefield_dir(cav):
        from pathlib import Path
        return Path(cav.self_dir) / 'wakefield'

    def run(self, cav, config, subdir=''):
        base = self.wakefield_dir(cav)
        if subdir:
            base = base / subdir
        for pol in ('longitudinal', 'transversal'):
            (base / pol).mkdir(parents=True, exist_ok=True)

    def read(self, cav):
        return self.read_dir(str(self.wakefield_dir(cav)))

    def read_dir(self, folder):
        import numpy as np
        import pandas as pd
        from cavsim2d.solvers.wakefield import WakefieldResult
        f = np.linspace(100.0, 2000.0, 60)          # MHz
        z = np.abs(np.sin(f / 180.0)) * 1000.0
        wz = pd.DataFrame({'f [MHz]': f, '|Z| [Ohm]': z, 'Re(Z) [Ohm]': z,
                           'Im(Z) [Ohm]': z * 0.1,
                           's [m]': np.linspace(0, 3, 60), 'W [V/pC]': -z})
        wt = wz.rename(columns=lambda c: c.replace('Ohm', 'Ohm/m').replace('V/pC', 'V/pC/m'))
        qois = {'|k_loss| [V/pC]': 0.42, 'k_FM [V/pC]': 0.10,
                'k_loss_HOM [V/pC]': 0.32, '|k_kick| [V/pC/m]': 1.5}
        return WakefieldResult(wz, wt, qois)

    def write_qois(self, cav, result=None):
        from cavsim2d.solvers.wakefield.base import WakefieldBackend
        WakefieldBackend.write_qois(self, cav, result if result is not None else self.read(cav))

    @classmethod
    def read_qois(cls, cav):
        from cavsim2d.solvers.wakefield.base import WakefieldBackend
        return WakefieldBackend.read_qois(cav)


def test_wakefield_backend_is_swappable(project_dir):
    """Swap ABCI for a canned backend via wakefield_config['solver']: the run,
    cav.wakefield.qois, plot_impedance and a ZL objective all work through it —
    no ABCI, proving nothing downstream is tied to a specific solver."""
    import matplotlib
    matplotlib.use('Agg')
    from cavsim2d import Study, EllipticalCavity
    from cavsim2d.solvers.wakefield import register_backend, BACKENDS
    from cavsim2d.processes.wakefield import get_wakefield_objectives_value

    register_backend(_DummyWakefield())
    try:
        tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
        cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='both')
        cavs = Study(project_dir)
        cavs.add_cavity([cav], ['D'])
        cavs.run_wakefield({'solver': 'dummy', 'processes': 1, 'rerun': True})

        # normalised schema written and read back
        assert os.path.exists(os.path.join(cav.self_dir, 'wakefield',
                                           'longitudinal', 'qois.json'))
        assert cav.wakefield.qois['|k_loss| [V/pC]'] == 0.42

        # plotting reads the normalised frames, not any ABCI object; the dummy
        # backend reports [Ohm], so the self-describing unit conversion is
        # exercised here too
        ax = cav.wakefield.plot_impedance()
        assert ax is not None and ax.lines

        # a ZL objective computes through the same frames
        df_wake, keys = get_wakefield_objectives_value(
            {cav.name: cav}, [['min', 'ZL', [0.1, 0.5, 1.0]]],
            project_dir, solver='dummy')
        assert not df_wake.empty
    finally:
        BACKENDS.pop('dummy', None)


@requires_abci
def test_wakefield_field_line_animation(project_dir):
    """save_fields=True turns on ABCI's electric-field-line output (LPLE), and
    cav.wakefield.animate_fields() reads those snapshots into an animation. Without
    save_fields there are no snapshots, and it returns None with a message rather
    than an empty animation (the old plot_animate_wakefield gave 0 frames)."""
    import matplotlib
    matplotlib.use('Agg')

    cavs = Study(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['WFANIM'])

    # nothing run yet -> no cavity.top -> no animation, clear return (not a crash)
    assert cav.wakefield.animate_fields(embed=False) is None

    # save_fields -> ABCI writes electric-field-line frames that animate
    cav.wakefield.run({'MROT': 0, 'wakelength': 10, 'bunch_length': 25,
                       'save_fields': True})
    anim = cav.wakefield.animate_fields(embed=False)
    assert anim is not None
    assert anim._save_count >= 1                 # at least one field-line frame

def test_abci_deck_carries_the_requested_mesh(tmp_path):
    """wakefield_config['mesh_config'] must reach ABCI's own field mesh.

    geo_to_abc already read DDR/DDZ to size the contour sampling and the
    minimum beam pipe, so the key looked supported; the deck writer ignored it
    and pinned the mesh at its default. A caller sweeping DDR got a differently
    sampled wall solved on an unchanged mesh, and the convergence check they
    wrote came back flat. Writing the deck needs no ABCI binary, so this runs
    everywhere.
    """
    import glob
    from cavsim2d import Bellows

    cav = Bellows(Ri=35.0, A=8.0, L_p=6.0, N_conv=4,
                  R_root=1.2, R_crest=1.2, name='deck')
    cav.set_workspace(str(tmp_path / 'deck'))
    cav.geo_to_abc({'MROT': 0, 'wakelength': 10, 'bunch_length': 25,
                    'mesh_config': {'DDR': 0.0007, 'DDZ': 0.0009}})

    decks = glob.glob(str(tmp_path / 'deck' / 'wakefield' / '**' / '*.abc'),
                      recursive=True)
    assert decks, 'no ABCI deck was written'
    text = open(decks[0]).read()
    assert 'DDR = 0.0007' in text
    assert 'DDZ = 0.0009' in text


def test_abci_mesh_follows_a_short_bunch():
    """DDR_SIG/DDZ_SIG set the mesh as a fraction of the bunch length, capped at
    the 1.25 mm default. They were documented but never read, so a 4.32 mm bunch
    was meshed at 1.25 mm: 3.5 steps per sigma."""
    from cavsim2d.models.base import abci_mesh_steps, ABCI_DEFAULT_STEP

    # the default 25 mm bunch keeps the mesh it always had
    assert abci_mesh_steps({'bunch_length': 25}) == (ABCI_DEFAULT_STEP, ABCI_DEFAULT_STEP)
    # a short bunch is resolved at 0.1 sigma, nested or top-level
    assert abci_mesh_steps({'beam_config': {'bunch_length': 4.32}}) == pytest.approx(
        (0.000432, 0.000432))
    assert abci_mesh_steps({'bunch_length': 4.32,
                            'mesh_config': {'DDZ_SIG': 0.2}}) == pytest.approx(
        (0.000432, 0.000864))
    # an explicit step always wins
    assert abci_mesh_steps({'bunch_length': 4.32,
                            'mesh_config': {'DDR': 0.001}}) == pytest.approx(
        (0.001, 0.000432))


def test_abci_deck_mesh_for_a_short_bunch(tmp_path):
    """The deck carries the bunch-scaled mesh, not the fixed 1.25 mm default."""
    import glob
    from cavsim2d import Bellows

    cav = Bellows(Ri=35.0, A=8.0, L_p=6.0, N_conv=4,
                  R_root=1.2, R_crest=1.2, name='deck')
    cav.set_workspace(str(tmp_path / 'deck'))
    cav.geo_to_abc({'MROT': 0, 'wakelength': 1, 'bunch_length': 5})
    decks = glob.glob(str(tmp_path / 'deck' / 'wakefield' / '**' / '*.abc'),
                      recursive=True)
    text = open(decks[0]).read()
    assert 'DDR = 0.0005' in text and 'DDZ = 0.0005' in text


def test_operating_point_bunch_lengths():
    """One bunch length, several named ones, and the old sigma_<label> keys."""
    from cavsim2d.data_module.operating_points import bunch_lengths, bunch_tag

    assert bunch_lengths({'sigma [mm]': 4.32}) == {'': 4.32}
    assert bunch_lengths({'sigma': 4.32}) == {'': 4.32}
    assert bunch_lengths({'sigma [mm]': {'SR': 4.32, 'BS': 15.2}}) == {'SR': 4.32, 'BS': 15.2}
    # legacy spelling, order kept, so SR stays the primary
    assert list(bunch_lengths({'sigma_SR [mm]': 4.32, 'sigma_BS [mm]': 15.2})) == ['SR', 'BS']
    with pytest.raises(KeyError):
        bunch_lengths({'I0 [mA]': 1280})

    # an int and a float bunch length file under the same id
    assert bunch_tag('Z', 'SR', 25) == bunch_tag('Z', 'SR', 25.0) == 'Z_SR_25.0mm'
    assert bunch_tag('Z', '', 4.32) == 'Z_4.32mm'
