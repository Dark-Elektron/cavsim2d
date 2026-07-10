"""Example 5 — Eigenmode with uncertainty quantification (UQ).

Propagates Gaussian geometry uncertainties (on A, B, a, b) through the
eigenmode solve via Stroud-3 quadrature and reports the mean and standard
deviation of the figures of merit. Results land under
    <SIM_ROOT>/examples/eigenmode_uq/
"""
import pprint

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)


def main():
    banner("Example 5: Eigenmode + UQ (Stroud-3 quadrature)")
    proj = project_dir('eigenmode_uq')
    out = plots_dir('eigenmode_uq')

    cavs = Cavities(proj)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['uq_cell'])

    eigenmode_config = {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
        'uq_config': {
            'variables': ['A', 'B', 'a', 'b'],
            # Objectives name a polarisation: monopole and m-pole results share
            # QOI names. 'dipole:2:freq [MHz]' picks a specific 1-based mode.
            'objectives': ["monopole:Epk/Eacc []", "monopole:Bpk/Eacc [mT/MV/m]",
                           "monopole:R/Q [Ohm]", "monopole:G [Ohm]", "monopole:freq [MHz]"],
            'delta': [0.05, 0.05, 0.05, 0.05],   # 5% std on each variable
            'processes': 1,
            'distribution': 'gaussian',
            'method': ['Quadrature', 'Stroud3'],
            'cell_type': 'mid-cell',
            'cell_complexity': 'simplecell',
        },
    }
    cavs.run_eigenmode(eigenmode_config)

    print("\nUQ results (mean / std of each objective):")
    pp.pprint(cavs.uq_fm_results)

    # Bar comparison with UQ error bars.
    try:
        cavs.plot_compare_fm_bar(uq=True)
        savefig(f"{out}/uq_compare_fm_bar.png")
    except Exception as e:
        print(f"    [warn] UQ comparison plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
