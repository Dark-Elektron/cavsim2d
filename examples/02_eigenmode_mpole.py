"""Example 2 — Higher-order-mode (m-pole) eigenmodes.

Solves the monopole, dipole and quadrupole passbands of a single-cell cavity
on the same 2D meridian mesh, prints the per-polarisation figures of merit,
and saves dipole field plots (in-plane and azimuthal envelopes). Results land
under
    <SIM_ROOT>/examples/eigenmode_mpole/cavities/mpole/eigenmode/<pol>/
"""
import os
import pprint

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)


def main():
    banner("Example 2: m-pole eigenmodes (monopole, dipole, quadrupole)")
    proj = project_dir('eigenmode_mpole')
    out = plots_dir('eigenmode_mpole')

    cavs = Cavities(proj)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['mpole'])

    eigenmode_config = {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
        # names or azimuthal mode numbers, e.g. [0, 1, 2]
        'polarisation': ['monopole', 'dipole', 'quadrupole'],
        'n_modes': 4,     # physical modes per m-pole solve
    }
    cavs.run_eigenmode(eigenmode_config)

    # --- Per-polarisation results ---------------------------------------
    cav.get_eigenmode_qois()
    print("\nMonopole fundamental QOIs:")
    pp.pprint(cav.eigenmode_qois)

    for pol in ('dipole', 'quadrupole'):
        print(f"\n{pol.capitalize()} modes:")
        for mode in cav.eigenmode.mpole_modes(pol):
            print(f"    {mode}")
        print(f"  {pol} fundamental QOIs:")
        qois = cav.eigenmode.mpole_qois(pol, all_modes=False)
        pp.pprint(qois)

    # --- Dipole field plots ---------------------------------------------
    # 'E'/'H' are the in-plane envelopes; 'Ephi'/'Hphi' the azimuthal ones.
    for which, fname in [('E', 'dipole_E_inplane.png'),
                         ('Ephi', 'dipole_E_azimuthal.png')]:
        try:
            cav.plot_fields(mode=0, which=which, plotter='matplotlib', pol='dipole')
            savefig(os.path.join(out, fname))
        except Exception as e:
            print(f"    [warn] dipole {which} plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
