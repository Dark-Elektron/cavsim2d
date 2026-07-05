"""Example 1 — Monopole eigenmode analysis.

Builds a single-cell and a two-cell TESLA-like cavity, runs the fundamental
(monopole) eigenmode solve, prints the figures of merit, and saves geometry,
field, axis-field and comparison plots. All artefacts land under
    <SIM_ROOT>/examples/eigenmode/
"""
import os
import pprint

from _common import (SIM_ROOT, MIDCELL, ENDCELL_L, ENDCELL_R,
                     project_dir, plots_dir, savefig, banner)
from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)


def main():
    banner("Example 1: Monopole eigenmode")
    proj = project_dir('eigenmode')
    out = plots_dir('eigenmode')

    cavs = Cavities(proj)
    cav1 = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cav2 = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
    cavs.add_cavity([cav1, cav2], ['single_cell', 'two_cell'])

    eigenmode_config = {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
    }
    cavs.run_eigenmode(eigenmode_config)

    print("\nFundamental-mode QOIs:")
    pp.pprint(cavs.eigenmode_qois)

    # --- Plots -----------------------------------------------------------
    # Geometry
    try:
        ax = cav1.plot('geometry', label='single cell')
        savefig(os.path.join(out, 'geometry_single_cell.png'), ax.figure)
    except Exception as e:
        print(f"    [warn] geometry plot failed: {e!r}")

    # Field magnitude (fundamental pi-mode)
    try:
        cav1.plot_fields(mode=1, which='E', plotter='matplotlib')
        savefig(os.path.join(out, 'field_E_single_cell.png'))
    except Exception as e:
        print(f"    [warn] field plot failed: {e!r}")

    # On-axis accelerating field / field flatness (two-cell)
    try:
        cav2.get_eigenmode_qois()
        cav2.plot_axis_field()
        savefig(os.path.join(out, 'axis_field_two_cell.png'))
    except Exception as e:
        print(f"    [warn] axis-field plot failed: {e!r}")

    # Dispersion / passband (two-cell has a 2-mode passband)
    try:
        ax = cav2.plot_dispersion()
        savefig(os.path.join(out, 'dispersion_two_cell.png'), ax.figure)
    except Exception as e:
        print(f"    [warn] dispersion plot failed: {e!r}")

    # Bar comparison across cavities
    try:
        cavs.plot_compare_fm_bar()
        savefig(os.path.join(out, 'compare_fm_bar.png'))
    except Exception as e:
        print(f"    [warn] comparison plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
