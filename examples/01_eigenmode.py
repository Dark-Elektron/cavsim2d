"""Example 1 — Monopole eigenmode analysis.

Builds a single-cell and a two-cell cavity, runs the monopole eigenmode solve,
prints the figures of merit, and saves geometry, field, on-axis field,
dispersion and comparison plots. Everything lands under
    <SIM_ROOT>/eigenmode/
"""
import os

from _common import (MIDCELL, ENDCELL_L, ENDCELL_R,
                     project_dir, plots_dir, savefig, banner)
from cavsim2d import EllipticalCavity, Study


def main():
    banner("Example 1: Monopole eigenmode")
    out = plots_dir('eigenmode')

    study = Study(project_dir('eigenmode'))
    cav1 = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cav2 = EllipticalCavity(2, MIDCELL, ENDCELL_L, ENDCELL_R, beampipe='both')
    study.add_cavity([cav1, cav2], ['single_cell', 'two_cell'])

    study.run_eigenmode({'boundary_conditions': 'mm',
                         'mesh_config': {'h': 15, 'p': 3}})

    print("\nFundamental-mode figures of merit:")
    print(study.summary().round(3).to_string())

    cav1.plot('geometry', show=False)
    savefig(os.path.join(out, 'geometry_single_cell.png'))

    cav1.show_fields(which='E', plotter='matplotlib', show=False)
    savefig(os.path.join(out, 'field_E_single_cell.png'))

    # On-axis accelerating field: both cells in phase (the pi-mode)
    cav2.plot_axis_field(show=False)
    savefig(os.path.join(out, 'axis_field_two_cell.png'))

    # A two-cell cavity has a two-mode passband
    cav2.plot_dispersion(show=False)
    savefig(os.path.join(out, 'dispersion_two_cell.png'))

    study.eigenmode.plot_fm_scatter(show=False)
    savefig(os.path.join(out, 'compare_fm_scatter.png'))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
