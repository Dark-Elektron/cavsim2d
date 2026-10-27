"""Example 3 — Frequency tuning.

Tunes a single-cell cavity's equator radius (Req) so the fundamental mode
lands on a target frequency, solves the tuned cavity, and saves the geometry
before and after. Results land under
    <SIM_ROOT>/tune/
"""
import os

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d import EllipticalCavity, Study
from cavsim2d.utils.style import WARM

TARGET_FREQ = 801.58  # MHz


def main():
    banner(f"Example 3: Tune Req to {TARGET_FREQ} MHz")
    out = plots_dir('tune')

    study = Study(project_dir('tune'))
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none')
    study.add_cavity(cav, 'cell')

    study.run_tune({
        'freqs': TARGET_FREQ,
        'cell_type': {'mid-cell': 'Req'},
        # the mesh of the tuning solves; solve_tuned then runs the eigenmode
        # analysis on the tuned cavity with the same settings
        'eigenmode_config': {'boundary_conditions': 'mm', 'mesh_config': {'h': 15, 'p': 3}},
        'solve_tuned': True,
    })

    res = cav.tune.qois['mid-cell']
    print(f"\nReq {MIDCELL[6]:.3f} -> {res['parameters']['Req_m']:.3f} mm, "
          f"tuned to {res['FREQ']:.4f} MHz")

    # The tuned geometry is a cavity of its own, with its own results
    q = cav.tuned.eigenmode.qois
    print(f"Tuned cavity: freq {q['freq [MHz]']:.4f} MHz, R/Q {q['R/Q [Ohm]']:.2f} Ohm, "
          f"field flatness {q['ff [%]']:.1f} %")

    # Untuned and tuned geometry overlaid: tuned=True draws the tuned version
    ax = cav.plot('geometry', show=False, label='before tuning')
    cav.plot('geometry', ax=ax, tuned=True, show=False, label='after tuning',
             color=WARM[0], ls='--')
    ax.legend()
    savefig(os.path.join(out, 'geometry_before_after.png'))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
