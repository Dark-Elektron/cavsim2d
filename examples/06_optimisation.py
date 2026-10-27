"""Example 6 — Multi-objective shape optimisation.

Runs a small evolutionary optimisation of a single cell. Every candidate is
tuned to the target frequency before its objectives, Epk/Eacc and Bpk/Eacc, are
evaluated. Saves the Pareto front, the history and the convergence plot.
Results land under
    <SIM_ROOT>/optimisation/
"""
import os

from _common import MIDCELL, project_dir, plots_dir, savefig, banner
from cavsim2d import EllipticalCavity, Study


def main():
    banner("Example 6: Multi-objective optimisation")
    out = plots_dir('optimisation')

    study = Study(project_dir('optimisation'))
    # The cavity added to the study is the template every candidate copies.
    study.add_cavity([EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='none')],
                     ['template'])

    # Bounds bracket the template so the inner tune converges; the run is
    # kept small so the example finishes quickly.
    study.run_optimisation({
        'initial_points': 6,
        'no_of_generation': 2,
        'method': {'LHS': {'seed': 5}},
        'bounds': {'A': [58, 66], 'B': [62, 70], 'a': [26, 34], 'b': [20, 27]},
        'objectives': [['min', 'monopole:Epk/Eacc []'],
                       ['min', 'monopole:Bpk/Eacc [mT/MV/m]']],
        'tune_config': {
            'freqs': 801.58,
            'cell_type': {'mid-cell': 'Req'},
            'eigenmode_config': {'boundary_conditions': 'mm',
                                 'mesh_config': {'h': 15, 'p': 3}},
        },
        # offspring counts per generation
        'mutation_factor': 4,
        'crossover_factor': 4,
        'elites_for_crossover': 2,
        'chaos_factor': 2,
    })

    opt = study.optimisation
    print(f"\nEvaluated candidates: {len(opt.history)}")
    print(f"Pareto-front size:    {len(opt.pareto)}")
    print(opt.pareto[list(opt.objective_vars)].round(3).to_string())

    opt.plot_pareto(kind='scatter', normalise=False, show=False)
    savefig(os.path.join(out, 'pareto_front.png'))
    opt.plot_history(kind='scatter', normalise=False, color_by_gen=True, show=False)
    savefig(os.path.join(out, 'history.png'))
    opt.plot_convergence(show=False)
    savefig(os.path.join(out, 'convergence.png'))

    print(f"\nDone. Results under: {study.projectDir}")


if __name__ == '__main__':
    main()
