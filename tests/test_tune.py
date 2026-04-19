"""Test: Tune + Eigenmode using the new solver-as-object API."""
import sys
import numpy as np
import pprint

sys.path.insert(0, r'c:\Users\Soske\Documents\git_projects\cavsim2d')

from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)


def main():
    sim_dir = r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations'
    # TESLA‐like midcell known to tune cleanly
    midcell = np.array([62.22, 66.13, 30.22, 23.11, 72.0, 93.5, 171.20, 0])

    cav_tune_eig = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
    ax = cav_tune_eig.plot('geometry', label='Before tuning')

    cavs = Cavities(sim_dir)
    cavs.add_cavity(cav_tune_eig, 'cav_tune_eig')

    tune_config = {
        'freqs': 801.58,
        'parameters': 'L',
        'cell_types': 'mid-cell',
        'processes': 1,
        'rerun': True,
    }

    cavs.run_tune(tune_config)
    pp.pprint(cavs.tune_results)

    tuned = cav_tune_eig.tuned
    assert tuned is not None, "cav.tuned should be populated after tuning"
    print(f"\ntuned self_dir: {tuned.self_dir}")
    print(f"tuned freq:     {tuned.freq}")

    tuned.plot('geometry', ax, label='After tuning')

    tuned.eigenmode.run({'processes': 1, 'rerun': True, 'boundary_conditions': 'mm'})
    print(f"tuned eigenmode qois: {tuned.eigenmode.qois}")


if __name__ == '__main__':
    main()
