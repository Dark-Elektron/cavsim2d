Uncertainty Quantification
==========================

The uncertainty quantification (UQ) module propagates fabrication tolerances and alignment errors through the solver chain to compute the statistical distribution (mean and standard deviation) of the RF figures of merit. This enables designers to assess how manufacturing variability impacts cavity performance before committing to a fabrication run.

Interface
*********
UQ capabilities are integrated directly into existing analyses by nesting a ``uq_config`` sub-dictionary inside the solver configuration (e.g. inside ``eigenmode_config`` or ``wakefield_config``):

.. code-block:: python

    uq_config = {
        'variables': ['L', 'Req'],
        'delta': [0.05, 0.05],
        'method': ['Quadrature', 'Stroud3'],
        'distribution': 'gaussian',
        'cell_type': 'mid-cell',
        'cell_complexity': 'simplecell'
    }

    eigenmode_config = {
        'processes': 3,
        'rerun': True,
        'boundary_conditions': 'mm',
        'uq_config': uq_config
    }

    cavs.run_eigenmode(eigenmode_config)

This runs the eigenmode solver at a set of perturbed geometry points (quadrature nodes) and aggregates the results into statistical moments.

Configuration Settings
**********************

``variables``
   *(list of str)* The geometric variables to perturb, e.g. ``['L', 'Req']``. Each variable corresponds to a dimension of the elliptical cell parameterisation (see :doc:`quickstart` for the parameter naming convention).

``delta``
   *(list of float)* The perturbation magnitude for each variable (in mm). Interpreted as the standard deviation when ``distribution`` is ``'gaussian'``, or the half-width when ``distribution`` is ``'uniform'``.

``method``
   *(list)* The UQ evaluation method. The recommended default is ``['Quadrature', 'Stroud3']``, which uses Stroud's third-order symmetric cubature rule. This minimises the number of simulation evaluations while capturing second-order statistical moments.

``distribution``
   *(str, default:* ``'gaussian'`` *)* Probability distribution model: ``'gaussian'`` (normal) or ``'uniform'``.

``cell_type``
   *(str, default:* ``'mid-cell'`` *)* The type of cell to which the perturbation is applied.

``cell_complexity``
   *(str, default:* ``'simplecell'`` *)* Determines whether perturbations are applied to a single cell (``'simplecell'``) or the full multi-cell structure.

``processes``
   *(int, default: 1)* Number of parallel processes for evaluating the perturbed quadrature nodes.

Visualising Results
*******************
Once the UQ simulation finishes, the standard deviations are populated in the results database. You can compare the fundamental-mode quantities with error bars across different cavities:

.. code-block:: python

    cavs.eigenmode.plot_fm_bar(uq=True)

.. important::

   Running UQ perturbations near degenerate geometries can lead to errors. For example, applying perturbations to a cavity with geometric dimensions close to physical constraints (like some reentrant cavity designs) may produce invalid or self-intersecting shapes. Use ``cav.inspect()`` to verify that the perturbation range defined by ``delta`` does not push the geometry beyond its valid limits.

See the worked examples in :doc:`Advanced: UQ everywhere <examples/advanced/index>`.
