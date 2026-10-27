"""Example 2 — Higher-order-mode (m-pole) eigenmodes.

Solves the monopole, dipole and quadrupole passbands of a single-cell cavity
on the same 2D meridian mesh, prints the per-polarisation figures of merit,
and saves the spectrum and dipole field plots. Results land under
    <SIM_ROOT>/eigenmode_mpole/
"""
import os

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d import EllipticalCavity, Study


def main():
    banner("Example 2: m-pole eigenmodes (monopole, dipole, quadrupole)")
    out = plots_dir('eigenmode_mpole')

    study = Study(project_dir('eigenmode_mpole'))
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    study.add_cavity([cav], ['mpole'])

    study.run_eigenmode({
        'boundary_conditions': 'mm',
        # names or azimuthal mode numbers, e.g. [0, 1, 2]
        'polarisation': ['monopole', 'dipole', 'quadrupole'],
        'n_modes': 4,     # modes per polarisation
        'mesh_config': {'h': 15, 'p': 3},
    })

    # Every mode of every polarisation, in one table
    cols = ['polarisation', 'freq [MHz]', 'R/Q [Ohm]', 'G [Ohm]']
    df = cav.eigenmode.qois_df
    print("\nModes:")
    print(df[[c for c in cols if c in df.columns]].round(3).to_string())

    # The lowest dipole mode's figures of merit (a transverse R/Q)
    print("\nLowest dipole mode:")
    for key, value in cav.eigenmode.mpole_qois('dipole', all_modes=False).items():
        print(f"    {key}: {value}")

    cav.eigenmode.plot_spectrum(show=False)
    savefig(os.path.join(out, 'spectrum.png'))

    # 'E'/'H' are the in-plane envelopes; 'Ephi'/'Hphi' the azimuthal ones.
    for which, fname in [('E', 'dipole_E_inplane.png'),
                         ('Ephi', 'dipole_E_azimuthal.png')]:
        cav.show_fields(which=which, plotter='matplotlib', pol='dipole', show=False)
        savefig(os.path.join(out, fname))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
