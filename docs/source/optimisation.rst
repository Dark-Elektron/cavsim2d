Geometry Optimisation
=====================

The geometry optimisation module uses a multi-objective genetic algorithm (GA) to automatically search the cavity design space for Pareto-optimal shapes that satisfy specified RF performance targets. It integrates the eigenmode solver, frequency tuning, and — optionally — the wakefield solver into a single automated pipeline.

Interface
*********
To run the geometry optimisation:

.. code-block:: python

    cavs.run_optimisation(optimisation_config)

Configuration Dictionary
************************
Optimisation is driven by a nested configuration dictionary:

.. code-block:: python

    optimisation_config = {
        'tune_config': {
            'freqs': 1300,
            'cell_type': {'mid-cell': 'Req'},
            'processes': 1
        },
        'bounds': {
            'A': [20.0, 80.0],
            'B': [20.0, 80.0],
            'a': [10.0, 60.0],
            'b': [10.0, 60.0],
            'Ri': [60.0, 85.0],
            'L': [93.5, 93.5],
            'Req': [170.0, 170.0]
        },
        'objectives': [
            ['min', 'monopole:Epk/Eacc []'],
            ['min', 'monopole:Bpk/Eacc [mT/MV/m]'],
            ['max', 'monopole:R/Q [Ohm]'],
            ['min', 'ZL', [1, 2, 5]],
            ['min', 'ZT', [1, 2, 3, 5]]
        ],
        'initial_points': 5,
        'method': {
            'LHS': {'seed': 5}
        },
        'no_of_generation': 2,
        'crossover_factor': 5,
        'elites_for_crossover': 2,
        'mutation_factor': 5,
        'chaos_factor': 5
    }

Settings description:

``tune_config``
   *(dict)* The tuning setup applied to every generated candidate geometry. This ensures each design is tuned to the exact target frequency before its RF objectives are evaluated. See :doc:`tuning` for details.

``bounds``
   *(dict)* Defines the search bounds ``[lower, upper]`` for each geometric variable (in mm). Variables that should remain constant must be entered with identical upper and lower bounds (e.g. ``'L': [93.5, 93.5]``).

``objectives``
   *(list of lists)* Specifies the optimisation objectives. Each entry is a list with the structure ``[direction, metric, ...]``:

   - **Direction:** ``'min'`` (minimise), ``'max'`` (maximise), or ``'equal'`` (target a specific value).
   - **Metric:** A QOI key prefixed with the polarisation name (e.g. ``'monopole:Epk/Eacc []'``, ``'monopole:freq [MHz]'``).
   - For wakefield impedance objectives (``'ZL'`` or ``'ZT'``), the third element is a list of frequency intervals (in GHz) over which the peak impedance is evaluated.

``initial_points``
   *(int)* Number of initial candidate designs to evaluate before the GA evolution begins.

``method``
   *(dict)* Method used for initial point generation. The default is Latin Hypercube Sampling (``LHS``), which provides good space-filling coverage.

``no_of_generation``
   *(int)* Number of GA generations to evolve.

``crossover_factor`` / ``mutation_factor`` / ``chaos_factor``
   *(int)* Weights governing how many offspring are generated via crossover, random mutations, and random chaos injections in each generation.

``elites_for_crossover``
   *(int)* Number of top-performing candidate designs retained in the parent pool for crossover.

Accessing Results
*****************
The optimisation results (Pareto front, per-generation fitness values, and the final population) are saved to the project folder and can be inspected programmatically:

.. code-block:: python

    # Inspect final population and Pareto front
    print(cavs.optimisation_results)

See the worked example: :doc:`examples/optimisation/pareto`.
