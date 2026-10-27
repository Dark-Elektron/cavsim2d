"""Example 4 — Wakefield / impedance analysis with ABCI.

Runs the ABCI wakefield solver on a two-cell cavity and saves the longitudinal
and transverse impedance spectra, the wake potential and the loss-factor
spectrum. Results land under
    <SIM_ROOT>/wakefield/

cavsim2d does not ship ABCI. Download it from KEK (https://abci.kek.jp/abci.htm)
and set CAVSIM2D_ABCI to the executable or its folder; this example skips
itself when ABCI is not found.
"""
import os

from _common import (MIDCELL, ENDCELL_L, ENDCELL_R,
                     project_dir, plots_dir, savefig, banner)
from cavsim2d import EllipticalCavity, Study
from cavsim2d.solvers.ABCI.abci import ABCI_URL, abci_exe_path


def main():
    banner("Example 4: Wakefield / impedance (ABCI)")
    if not os.path.isfile(abci_exe_path()):
        print(f"    Skipping: ABCI not found. Download it from {ABCI_URL} and set "
              f"CAVSIM2D_ABCI to the executable or its folder.")
        return
    out = plots_dir('wakefield')

    study = Study(project_dir('wakefield'))
    cav = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
    study.add_cavity([cav], ['two_cell'])

    study.run_wakefield({
        'MROT': 2,             # longitudinal and transverse
        'wakelength': 20,      # metres of wake to compute
        'bunch_length': 25,    # mm
    })

    print("\nLoss and kick factors:")
    for key, value in cav.wakefield.qois.items():
        print(f"    {key}: {value}")

    for kind in ('longitudinal', 'transverse'):
        cav.wakefield.plot_impedance(kind, show=False)
        savefig(os.path.join(out, f'{kind}_impedance.png'))
    cav.wakefield.plot_wake('longitudinal', show=False)
    savefig(os.path.join(out, 'longitudinal_wake.png'))
    cav.wakefield.plot_k_loss(show=False)
    savefig(os.path.join(out, 'loss_factor_spectrum.png'))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
