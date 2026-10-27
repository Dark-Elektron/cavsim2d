"""Example 5 — Eigenmode with uncertainty quantification (UQ).

Propagates Gaussian manufacturing tolerances on the ellipse semi-axes (A, B,
a, b) through the eigenmode solve by Stroud-3 quadrature and reports the mean
and standard deviation of the figures of merit. Results land under
    <SIM_ROOT>/eigenmode_uq/
"""
import os

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d import EllipticalCavity, Study


def main():
    banner("Example 5: Eigenmode + UQ (Stroud-3 quadrature)")
    out = plots_dir('eigenmode_uq')

    study = Study(project_dir('eigenmode_uq'))
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    study.add_cavity([cav], ['uq_cell'])

    study.run_eigenmode({
        'boundary_conditions': 'mm',
        'mesh_config': {'h': 15, 'p': 3},
        'uq_config': {
            'variables': ['A', 'B', 'a', 'b'],
            # Objectives name a polarisation, since monopole and m-pole results
            # share QOI names; 'dipole:2:freq [MHz]' picks a 1-based mode.
            'objectives': ['monopole:freq [MHz]', 'monopole:R/Q [Ohm]',
                           'monopole:Epk/Eacc []', 'monopole:Bpk/Eacc [mT/MV/m]',
                           'monopole:G [Ohm]'],
            # delta is the standard deviation of each variable, in mm (an
            # additive perturbation, not a fraction)
            'delta': [0.2, 0.2, 0.2, 0.2],
            'distribution': 'gaussian',
            'method': ['Quadrature', 'Stroud3'],
            'cell_type': 'mid-cell',
            'cell_complexity': 'simplecell',
        },
    })

    print("\nMean and standard deviation of each objective:")
    for name, stats in study.uq_fm_results['uq_cell'].items():
        print(f"    {name}: {stats['expe'][0]:.4g} +/- {stats['stdDev'][0]:.3g}")

    # Each QOI as its mean with a +/-1 sigma error bar, beside the design point
    study.eigenmode.plot_fm_scatter(uq=True, show=False)
    savefig(os.path.join(out, 'uq_fm_scatter.png'))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
