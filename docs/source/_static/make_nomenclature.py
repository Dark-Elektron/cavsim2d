"""Generate the beam-line element nomenclature figures for the docs."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cavsim2d import BLA, Bellows, Taper
from cavsim2d.utils.style import apply_style, WARM

apply_style()

OUT = r'C:\Users\Soske\Documents\git_projects\cavsim2d\docs\source\_static'
WALL, DIM, ACC = '#1d1d1d', WARM[0], WARM[1]


def wall(ax, cav, **kw):
    pts = np.asarray(cav.profile().contour_points(1e-4, skip=('AXI', 'PMC'))) * 1e3
    ax.plot(pts[:, 0], pts[:, 1], color=WALL, lw=1.8, zorder=3, **kw)
    return pts


def axis(ax, z0, z1):
    ax.plot([z0, z1], [0, 0], color='0.55', lw=0.9, ls=(0, (7, 5)), zorder=1)
    ax.annotate('beam axis', xy=(z1, 0), xytext=(z1 - 1, 1.4),
                color='0.45', fontsize=8, ha='right')


def dim(ax, p0, p1, label, offset=(0, 0), color=DIM, fs=9, ha='center', va='center'):
    """A double-headed dimension line from p0 to p1, labelled at its middle."""
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.1,
                                shrinkA=0, shrinkB=0, mutation_scale=10))
    mid = ((p0[0] + p1[0]) / 2 + offset[0], (p0[1] + p1[1]) / 2 + offset[1])
    ax.text(mid[0], mid[1], label, color=color, fontsize=fs, ha=ha, va=va,
            bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))


def tick(ax, z, r0, r1, color=DIM):
    ax.plot([z, z], [r0, r1], color=color, lw=0.7, ls=':', zorder=2)


def call(ax, text, xy, xytext, color=ACC, fs=9, ha='center'):
    ax.annotate(text, xy=xy, xytext=xytext, color=color, fontsize=fs, ha=ha,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.0,
                                shrinkA=0, shrinkB=2))


def finish(ax, title, z0, z1, r1, r0=-4.0):
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(z0, z1)
    ax.set_ylim(r0, r1)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('r [mm]')
    ax.set_aspect('equal', adjustable='box')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ── bellows ───────────────────────────────────────────────────────────────

def bellows_figure():
    Ri, A, L_p, N, Rr, Rc, Lbp = 35.0, 12.0, 20.0, 3, 3.0, 3.0, 12.0
    b = Bellows(Ri=Ri, A=A, L_p=L_p, N_conv=N, R_root=Rr, R_crest=Rc,
                crest_fraction=0.5, L_bp=Lbp)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    wall(ax, b)
    z_start = -Lbp                                  # corrugation starts here
    axis(ax, -Lbp - 2, N * L_p + Lbp + 2)

    # bore and depth
    tick(ax, z_start - 6, 0, Ri)
    dim(ax, (z_start - 6, 0), (z_start - 6, Ri), 'Ri  (bore)', offset=(2.6, 0), ha='left')
    z_cr = z_start + L_p * 0.5                      # over the first crest
    tick(ax, z_cr, Ri, Ri + A)
    dim(ax, (z_cr, Ri), (z_cr, Ri + A), 'A', offset=(-2.6, 0), ha='right')
    ax.plot([z_start - 8, N * L_p + 4], [Ri, Ri], color=DIM, lw=0.6, ls=':', zorder=1)

    # one period, and the flats inside it
    r_top = Ri + A + 7
    dim(ax, (z_start + L_p, r_top), (z_start + 2 * L_p, r_top), 'L_p', offset=(0, 2.2))
    for z in (z_start + L_p, z_start + 2 * L_p):
        tick(ax, z, Ri, r_top)

    w = L_p / 2.0                                   # crest_fraction = 0.5
    r_fl = Ri + A + 2.0
    dim(ax, (z_start + L_p + w / 2, r_fl), (z_start + L_p + 1.5 * w, r_fl),
        'crest flat', offset=(0, 1.9), fs=8)
    r_rt = Ri - 5.5
    dim(ax, (z_start + 1.5 * L_p + w / 2, r_rt), (z_start + 2.5 * L_p - w / 2, r_rt),
        'root flat', offset=(0, -1.9), fs=8)
    ax.text(z_start + 2.1 * L_p, Ri - 11.5,
            'crest_fraction = crest flat / (crest flat + root flat)',
            color=DIM, fontsize=8, ha='center')

    # corner radii
    call(ax, 'R_crest', (z_start + 2 * L_p + w / 2 + Rc * 0.3, Ri + A - Rc * 0.35),
         (z_start + 2.55 * L_p, Ri + A + 9))
    call(ax, 'R_root', (z_start + 2.5 * L_p + w / 2 - Rr * 0.3, Ri + Rr * 0.35),
         (z_start + 3.0 * L_p, Ri + 12))

    # end pipe and the whole corrugated run
    dim(ax, (-Lbp, Ri - 12.5), (0.0, Ri - 12.5), 'L_bp', offset=(0, -2.0), fs=8)
    dim(ax, (0.0, Ri - 18.0), (N * L_p, Ri - 18.0),
        'N_conv * L_p', offset=(0, -2.0), fs=8)
    for z in (-Lbp, 0.0, N * L_p):
        tick(ax, z, Ri - 19.0, Ri)

    # flank
    ax.annotate('flank_angle\n(90 deg = vertical)',
                xy=(z_start + L_p * 1.0, Ri + A / 2), xytext=(z_start - 4, Ri + A + 13),
                color=ACC, fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.0))

    finish(ax, 'Bellows parameters', -Lbp - 10, N * L_p + Lbp + 4, Ri + A + 22, -6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'beamline_bellows_nomenclature.png'), dpi=160)
    plt.close(fig)


# ── beam line absorber ────────────────────────────────────────────────────

def bla_figure():
    R, L, L_abs, t = 35.0, 150.0, 80.0, 10.0
    b = BLA(R=R, L=L, absorber_length=L_abs, absorber_thickness=t)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    wall(ax, b)
    axis(ax, -L / 2 - 2, L / 2 + 2)

    d, = b.dielectrics
    ax.fill_between([d['z'][0], d['z'][1]], d['r'][0], d['r'][1],
                    color=WARM[1], alpha=0.45, lw=0, zorder=2)
    ax.text(0, R + t / 2, 'absorber', color='#6b1f0a', fontsize=9,
            ha='center', va='center', zorder=4)

    tick(ax, -L / 2 - 6, 0, R)
    dim(ax, (-L / 2 - 6, 0), (-L / 2 - 6, R), 'R', offset=(-5, 0), ha='right')
    ax.text(-L / 2 - 6, R / 2 - 6, '(free aperture)', color=DIM, fontsize=7.5,
            ha='right')

    dim(ax, (L / 2 + 7, R), (L / 2 + 7, R + t), 'absorber_\nthickness',
        offset=(6, 0), ha='left', fs=8)
    ax.plot([-L / 2, L / 2 + 9], [R, R], color=DIM, lw=0.6, ls=':', zorder=1)

    dim(ax, (-L_abs / 2, R + t + 6), (L_abs / 2, R + t + 6),
        'absorber_length', offset=(0, 2.4), fs=9)
    for z in (-L_abs / 2, L_abs / 2):
        tick(ax, z, R, R + t + 6)
    dim(ax, (-L / 2, -7), (L / 2, -7), 'L', offset=(0, -2.4))
    for z in (-L / 2, L / 2):
        tick(ax, z, -8, R)

    call(ax, 'z_absorber\n(ring centre, from the element centre)',
         (0, R + t), (26, R + t + 19), fs=8, ha='left')
    ax.text(0, -15,
            'the ring inner radius IS the bore: the thickness goes outward, '
            'into a recess in the wall',
            color=DIM, fontsize=8.5, ha='center')

    finish(ax, 'Beam line absorber (BLA) parameters',
           -L / 2 - 22, L / 2 + 30, R + t + 28, -19)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'beamline_bla_nomenclature.png'), dpi=160)
    plt.close(fig)


# ── taper ─────────────────────────────────────────────────────────────────

def taper_figure():
    Rl, Rr, L, sl, sr, fil = 35.0, 80.0, 160.0, 30.0, 30.0, 12.0
    t = Taper(R_left=Rl, R_right=Rr, L=L, straight_left=sl, straight_right=sr,
              R_fillet=fil)
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    wall(ax, t)
    axis(ax, -L / 2 - 2, L / 2 + 2)

    tick(ax, -L / 2 - 6, 0, Rl)
    dim(ax, (-L / 2 - 6, 0), (-L / 2 - 6, Rl), 'R_left', offset=(-4, 0), ha='right')
    tick(ax, L / 2 + 6, 0, Rr)
    dim(ax, (L / 2 + 6, 0), (L / 2 + 6, Rr), 'R_right', offset=(4, 0), ha='left')

    dim(ax, (-L / 2, Rr + 10), (L / 2, Rr + 10), 'L', offset=(0, 2.4))
    dim(ax, (-L / 2, Rl - 8), (-L / 2 + sl, Rl - 8), 'straight_left',
        offset=(0, -2.3), fs=8)
    dim(ax, (L / 2 - sr, Rl - 8), (L / 2, Rl - 8), 'straight_right',
        offset=(0, -2.3), fs=8)
    for z in (-L / 2, -L / 2 + sl, L / 2 - sr, L / 2):
        tick(ax, z, Rl - 9, Rr + 10)

    call(ax, 'R_fillet', (-L / 2 + sl + 3, Rl + 3), (-L / 2 + sl - 6, Rl + 26),
         ha='right')
    call(ax, 'R_fillet', (L / 2 - sr - 3, Rr - 3), (L / 2 - sr + 8, Rr - 26),
         ha='left')

    # half angle
    zc0, zc1 = -L / 2 + sl, L / 2 - sr
    ax.plot([zc0, zc1 + 12], [Rl, Rl], color=ACC, lw=0.7, ls=':', zorder=2)
    ax.annotate('half_angle\n(derived)', xy=(zc0 + 18, Rl + 5),
                xytext=(zc0 - 6, Rr + 24), color=ACC, fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=ACC, lw=1.0))

    finish(ax, 'Taper parameters', -L / 2 - 26, L / 2 + 28, Rr + 34, -14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'beamline_taper_nomenclature.png'), dpi=160)
    plt.close(fig)


bellows_figure()
bla_figure()
taper_figure()
print('wrote nomenclature figures to', OUT)
