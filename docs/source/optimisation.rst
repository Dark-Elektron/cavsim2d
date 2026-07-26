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
Results are saved to ``<project>/optimisation/`` and exposed as DataFrames under the ``optimisation`` namespace:

.. code-block:: python

    opt = cavs.optimisation
    opt.history          # every evaluated candidate (geometry + objectives + generation)
    opt.pareto           # the non-dominated designs
    opt.pareto_history   # the Pareto front from each generation
    opt.objective_vars   # the objective column names

Each row carries the candidate's geometry variables and its objective values, so the front can be sorted and picked from directly, e.g. ``opt.pareto.sort_values('monopole:Epk/Eacc []')``.

Visualisation
*************
The objective space can be viewed several ways via ``kind=`` (``'scatter'``, ``'pcp'`` parallel-coordinates, ``'radar'``, ``'heatmap'``), with ``normalise=True|False``:

.. code-block:: python

    opt.plot_pareto(kind='scatter')      # pairwise scatter matrix of objectives
    opt.plot_pareto(kind='pcp')          # parallel coordinates — one line per design
    opt.plot_pareto(kind='radar')        # a polygon fingerprint per design
    opt.plot_history(color_by_gen=True)  # every candidate, coloured by generation
    opt.plot_pareto_history()            # the front from each generation
    opt.plot_convergence()               # fitness against generation

Objectives may combine eigenmode QOIs (e.g. ``'monopole:Epk/Eacc []'``, ``'monopole:R/Q [Ohm]'``) with wakefield impedance targets (``'ZL'`` / ``'ZT'`` over frequency intervals), so a single run can trade off surface fields, shunt impedance and higher-order-mode impedance together.

Worked examples:

- :doc:`examples/optimisation/pareto` — a minimal two-objective run,
- :doc:`examples/optimisation/visualising_results` — a three-objective run and the full set of result views,
- :doc:`examples/optimisation/cavity_types` — optimising a non-elliptical (spline) cavity over its own variables.
