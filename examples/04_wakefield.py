"""Example 4 — Wakefield / impedance analysis (Windows + bundled ABCI.exe).

Runs the ABCI wakefield solver on a two-cell cavity and saves the longitudinal
and transverse impedance spectra and wake potentials. Results land under
    <SIM_ROOT>/examples/wakefield/
"""
import os
import platform

from _common import (MIDCELL, ENDCELL_L, ENDCELL_R,
                     project_dir, plots_dir, savefig, banner)
from cavsim2d.cavity import Cavities, EllipticalCavity


def main():
    banner("Example 4: Wakefield / impedance (ABCI)")

    exe = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'cavsim2d', 'solvers', 'ABCI', 'ABCI.exe')
    if not (platform.system() == 'Windows' and os.path.exists(exe)):
        print("    Skipping: wakefield needs the bundled ABCI.exe on Windows "
              "(or wine). See the README 'Third party code' section.")
        return

    proj = project_dir('wakefield')
    out = plots_dir('wakefield')

    cavs = Cavities(proj)
    cav = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
    cavs.add_cavity([cav], ['wake_cell'])

    wakefield_config = {
        'processes': 1,
        'rerun': True,
        'wakelength': 20,     # metres of wake to compute
        'bunch_length': 25,   # mm
    }
    cavs.run_wakefield(wakefield_config)

    # Load ABCI output and save the standard impedance / wake plots.
    cav.get_abci_data()
    for what, fname, label in [
        ('zl', 'longitudinal_impedance.png', 'longitudinal impedance'),
        ('zt', 'transverse_impedance.png', 'transverse impedance'),
        ('wpl', 'longitudinal_wake.png', 'longitudinal wake potential'),
        ('wpt', 'transverse_wake.png', 'transverse wake potential'),
    ]:
        try:
            ax = cav.plot(what)
            savefig(os.path.join(out, fname), ax.figure)
        except Exception as e:
            print(f"    [warn] {label} plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
