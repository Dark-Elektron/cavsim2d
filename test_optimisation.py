"""Test: Optimisation using the new solver-as-object API."""
import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = r'c:\Users\Soske\Documents\git_projects\cavsim2d'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cavsim2d.cavity.cavities import Cavities
import numpy as np

def run_test():
    # Use a temporary directory for simulations
    sim_dir = Path(r'c:\Users\Soske\Documents\git_projects\cavsim2d\tmp_simulations')
    sim_dir.mkdir(parents=True, exist_ok=True)

    cavs = Cavities(str(sim_dir))
    cell_type = 'single-cell'

    optimisation_config = {
        'initial_points': 2,      # Small number for testing
        'no_of_generation': 1,    # Small number for testing
        'method': {
            'LHS': {'seed': 5},
        },
        'bounds': {
            'A': [40, 45],
            'B': [40, 45],
            'a': [10, 15],
            'b': [15, 20],
            'Ri': [30, 35],
            'L': [50, 60],
            'Req': [100, 110],
        },
        'objectives': [
            ['min', 'freq [MHz]'],
            ['min', 'Epk/Eacc []']
        ],
        'tune_config': {
            'freqs': 801.58,
            'parameters': 'Req',
            'cell_types': cell_type,
            'processes': 1,    # Single process for testing
            'eigenmode_config': {
                'n_cells': 1,
                'n_modules': 1,
                'f_shift': 0,
                'bc': 33,
                'beampipes': 'both',
            },
        },
        'mutation_factor': 0.5,
        'crossover_factor': 0.7,
        'elites_for_crossover': 2,
        'chaos_factor': 0.1,
        'weights': [1, 1],
    }

    print("Starting test optimization...")
    print(f"Results will be saved to: {sim_dir / 'optimisation'}")
    try:
        # New API: cavs.optimisation is an OptimisationSolver instance
        # cavs.run_optimisation() delegates to cavs.optimisation.run()
        cavs.run_optimisation(optimisation_config)

        # Access results via solver object
        print("\nOptimisation complete!")
        print(f"Results folder: {cavs.optimisation.folder}")
        if cavs.optimisation.history is not None:
            print(f"History shape: {cavs.optimisation.history.shape}")
        if cavs.optimisation.pareto is not None:
            print(f"Pareto front shape: {cavs.optimisation.pareto.shape}")

    except Exception as e:
        print(f"Optimization failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
