"""Example 3 — Frequency tuning.

Tunes a single-cell cavity's equator radius (Req) so the fundamental mode
lands on a target frequency, then reports the tuned parameters and runs an
eigenmode solve on the tuned cavity. Results land under
    <SIM_ROOT>/examples/tune/
"""
import os
import pprint

import numpy as np

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)

TARGET_FREQ = 801.58  # MHz


def main():
    banner(f"Example 3: Tune Req to {TARGET_FREQ} MHz")
    proj = project_dir('tune')
    out = plots_dir('tune')

    cavs = Cavities(proj)
    midcell = np.array(MIDCELL + [0])  # trailing alpha slot
    cav = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
    cavs.add_cavity(cav, 'tuned_cell')

    # geometry before tuning
    try:
        ax = cav.plot('geometry', label='before tuning')
    except Exception as e:
        ax = None
        print(f"    [warn] before-geometry plot failed: {e!r}")

    tune_config = {
        'freqs': TARGET_FREQ,
        'cell_type': {'mid-cell': 'Req'},
        'processes': 1,
        'rerun': True,
        'eigenmode_config': {'processes': 1, 'rerun': True,
                             'boundary_conditions': 'mm'},
    }
    cavs.run_tune(tune_config)

    print("\nTune results:")
    pp.pprint(cav.tune.qois)

    tuned = cav.tuned
    if tuned is not None:
        res = cav.tune.qois['mid-cell']
        print(f"\nTuned FREQ: {res['FREQ']:.3f} MHz  "
              f"(Req {MIDCELL[6]:.2f} -> {res['parameters']['Req_m']:.3f} mm)")
        # overlay tuned geometry on the same axes
        try:
            if ax is not None:
                tuned.plot('geometry', ax=ax, label='after tuning')
                ax.legend()
                savefig(os.path.join(out, 'geometry_before_after.png'), ax.figure)
        except Exception as e:
            print(f"    [warn] after-geometry plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
