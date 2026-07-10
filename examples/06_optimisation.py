"""Example 6 — Multi-objective shape optimisation.

Runs a small evolutionary optimisation (a couple of generations) that tunes
each candidate to the target frequency and minimises Epk/Eacc and Bpk/Eacc,
then saves the Pareto front and convergence plots. Results land under
    <SIM_ROOT>/examples/optimisation/optimisation/
"""
import os

from _common import project_dir, plots_dir, savefig, banner
from cavsim2d.cavity import Cavities


def main():
    banner("Example 6: Multi-objective optimisation")
    proj = project_dir('optimisation')
    out = plots_dir('optimisation')

    cavs = Cavities(proj)

    # mutation/crossover/chaos factors are integer offspring counts.
    # Bounds bracket the known-good TESLA-like mid-cell so the inner tune
    # converges. Keep the run small so the example finishes quickly.
    config = {
        'initial_points': 6,
        'no_of_generation': 2,
        'method': {'LHS': {'seed': 5}},
        'bounds': {
            'A': [58, 66], 'B': [62, 70], 'a': [26, 34], 'b': [20, 27],
            'Ri': [74, 82], 'L': [90, 97], 'Req': [165, 190],
        },
        'objectives': [
            ['min', 'monopole:Epk/Eacc []'],
            ['min', 'monopole:Bpk/Eacc [mT/MV/m]'],
        ],
        'tune_config': {
            'freqs': 801.58,
            'cell_type': {'mid-cell': 'Req'},
            'processes': 1,
            'eigenmode_config': {'n_cells': 1, 'processes': 1,
                                 'boundary_conditions': 'mm'},
        },
        'mutation_factor': 4,
        'crossover_factor': 4,
        'elites_for_crossover': 2,
        'chaos_factor': 2,
        'weights': [1, 1],
    }
    cavs.run_optimisation(config)

    opt = cavs.optimisation
    print(f"\nEvaluated candidates: {len(opt.history)}")
    print(f"Pareto-front size:    {len(opt.pareto)}")

    # Pareto front and history in objective space.
    try:
        fig, _ = opt.plot_pareto(kind='scatter', normalise=False)
        if fig is not None:
            savefig(os.path.join(out, 'pareto_front.png'), fig)
    except Exception as e:
        print(f"    [warn] pareto plot failed: {e!r}")

    try:
        fig, _ = opt.plot_history(kind='scatter', normalise=False, color_by_gen=True)
        if fig is not None:
            savefig(os.path.join(out, 'history.png'), fig)
    except Exception as e:
        print(f"    [warn] history plot failed: {e!r}")

    try:
        fig, _ = opt.plot_convergence()
        if fig is not None:
            savefig(os.path.join(out, 'convergence.png'), fig)
    except Exception as e:
        print(f"    [warn] convergence plot failed: {e!r}")

    print(f"\nDone. Results under: {proj}")


if __name__ == '__main__':
    main()
