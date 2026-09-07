Cavity Tuning
=============

The tuning module adjusts a selected geometric parameter of the cavity cells to converge on a specified target frequency. This is essential when designing cavities that must operate at an exact frequency (e.g. 1300 MHz for the TESLA/ILC standard) before evaluating final RF properties.

The algorithm iteratively modifies the chosen variable (typically the equator radius :math:`R_{\text{eq}}`), re-runs the eigenmode solver at each step, and uses a root-finding procedure to home in on the target :math:`\pi`-mode frequency.

Interface
*********
To run tuning:

.. code-block:: python

    cavs.run_tune(tune_config)

Or for a single cavity:

.. code-block:: python

    tesla.run_tune(tune_config)

Configuration Dictionary
************************
The tuning process is controlled by a configuration dictionary:

.. code-block:: python

    tune_config = {
        'freqs': 1300,
        'cell_type': {'mid-cell': 'Req'},
        'processes': 1,
        'rerun': True
    }

Settings description:

``freqs``
   *(float or list)* Target frequency in MHz. For multiple cavities, pass a list of target frequencies matching the number of loaded cavities.

``cell_type``
   *(dict)* Maps each cell block to the geometric variable to adjust. The key selects the cell block:

   - ``'mid-cell'`` — the repeating interior cells,
   - ``'end-cell'`` — the outer end cells (``'end-cell-left'`` / ``'end-cell-right'`` address one end),
   - ``'single-cell'`` — the sole cell of a one-cell cavity.

   The value selects the geometric variable, typically ``'Req'`` (equator radius) for elliptical cells, but any handle the model declares works — ``'A'``, ``'B'``, ``'L'`` for elliptical cells, ``'Req'`` for a pillbox, ``'R6'`` for the RF gun, or a spline control-point coordinate such as ``'p2_r'``. The value may also be a **list** of handles to tune in sequence.

   Scheduling several cell types in one call runs them **in order**, so later stages see the earlier ones already tuned:

   .. code-block:: python

       'cell_type': {'mid-cell': 'Req', 'end-cell': 'L'}

   Each model declares its own handles — the tuner asks the model rather than assuming a fixed list. See :doc:`examples/tuning/cavity_types` for the same call driving four different geometries.

Multicell cavities
******************
In a multicell cavity the operating mode is the :math:`\pi`-mode of the fundamental passband. Two things set it:

- The **interior (mid) cells** fix the passband through their shared equator radius ``Req``. Tuning the mid cell places the :math:`\pi`-mode.
- The **end cells** are detuned by the beam pipes. Left uncorrected they are under-excited, so the field is not flat. Tuning them **independently** — usually through the end-cell half-length ``L`` — restores field flatness and pins the :math:`\pi`-mode on target.

Both jobs are done in one call by scheduling ``'mid-cell'`` and ``'end-cell'`` together. The tuned mid and end parameters are stored on the single tuned cavity, whose :math:`\pi`-mode is the target frequency. See :doc:`examples/tuning/multicell` for the worked two-stage example, including the field-flatness before/after comparison.

Two-cell cavities: a design family
**********************************
A symmetric **2-cell** cavity is field-flat by mirror symmetry for *any* cell split at a fixed operating frequency, so requiring the mid and end cells to reach the target frequency independently is optional — it pins one design out of a family that all operate on target with a flat field. By default a 2-cell tune therefore returns that **family** rather than a single geometry: it sweeps the mid/end frequency split (retuning the shared ``Req`` so every member stays on target) and returns ``n_family`` designs. The each-cell-at-frequency member (mid and end frequencies equal) is kept as the representative ``cav.tuned``; the full family is ``cav.tune.family``. See :doc:`examples/tuning/two_cell_design_family` for why no member is privileged (``Epk/Eacc`` is invariant while ``R/Q``, ``G`` and the HOM wake vary). To trace this family directly for any elliptical cavity, call ``cav.tune.family_scan``.

``equal_cell_freq``
   *(bool, default: False)* 2-cell cavities only. If ``True``, pins the single each-cell-at-frequency design (the standard two-stage tune) instead of returning the family. Enforcing it also stiffens the tune — a strongly-coupled (wide-iris) cell can be unable to reach the target on its own even when the assembly can.

``n_family``
   *(int, default: 5)* Number of members returned by a 2-cell family tune.

``family_fraction``
   *(float, default: 0.03)* Half-width of the end-length sweep for the 2-cell family, as a fraction of the nominal.

``processes``
   *(int, default: 1)* Number of parallel workers when tuning multiple cavities simultaneously.

``rerun``
   *(bool, default: True)* If ``True``, forces a re-run of the tuning calculations even if cached results are found.

Constraining a QoI, not just frequency
**************************************
Frequency is not the only quantity you may need to *pin*. An equality target on a figure of merit — say ``R/Q = 157`` — is a **constraint**, not an objective: handed to the optimiser as ``|R/Q - 157|`` it is merely traded off against the other objectives, and the Pareto front fills with near-misses. Pin it in the tune instead, by adding a ``qoi_targets`` entry to the tune config. The tune variables stay in ``cell_type``; the tuner then moves them to hit the frequency **and** the QoI targets together, by a damped multivariable Newton on the assembled cavity (in one dimension this is the same secant the frequency tune uses):

.. code-block:: python

    cav.tune.run({
        'freqs': 801.58,
        'cell_type': {'end-cell': ['Req', 'L']},    # two tune variables
        'qoi_targets': {'R/Q [Ohm]': 157},           # the equality constraint
    })
    cav.tuned          # the tuned cavity, sitting on freq AND R/Q
    cav.tune.qois      # the achieved frequency and QoIs

``qoi_targets`` accepts any accelerating-mode QoI columns (``'R/Q [Ohm]'``, ``'ff [%]'``, …). The number of tune variables (summed across ``cell_type``) versus the number of targets (frequency + ``qoi_targets``) decides the outcome:

- **more targets than variables** — over-constrained; raises ``ValueError``,
- **equal counts** — a unique design,
- **fewer targets than variables** — under-constrained; a *family* is tuned (as a plain 2-cell frequency tune already does).

For a symmetric **2-cell** this runs on the full assembly and is exact — ``ff = 100 %`` by symmetry, and ``Req`` (the shared equator radius) is applied to both cells regardless of the ``cell_type`` key. For ``n > 2`` a **staged, per-cup** tune is used instead — see the next section.

Staged tuning for many-cell cavities
************************************
For ``n > 2`` the field flatness is a real constraint and an assembly-wide Newton is expensive. The π-mode **R/Q is additive over half-cells (cups)** — an *N*-cell cavity is ``2N-2`` interior (mid) cups and 2 beampipe (end) cups, because each end cell is one end-cup plus one interior mid-cup — and the sum is exact to ~0.03 % for frequency-matched cups. So the tune is **staged cup-by-cup**: give a **total** R/Q and let the tuner distribute it.

.. code-block:: python

    cav.tune.run({
        'freqs': 801.58,
        'qoi_targets': {'R/Q [Ohm]': 450},         # the cavity total
        'cell_type': {'mid-cell': 'Req',            # 1 var  -> frequency only
                      'end-cell': ['L', 'A']},      # 2 vars -> frequency + R/Q
    })

The **variable count carries the intent**: a cell with one tune variable is tuned to frequency only (its cup R/Q falls out); a cell with two is tuned to frequency **and** an R/Q share. The mid cell is tuned first (frequency-only, via the shared ``Req``) and its cup R/Q measured; the remaining cavity R/Q — ``total - (2N-2)·mid_cup`` — is assigned to the end cups, and the end cell is tuned to ``(freq, its share)`` via **local** handles ``(L, A)``. Every cup is tuned to the same frequency, so the field is flat by construction and you write no arithmetic. To pick the split yourself, add ``qoi_split`` (per-section R/Q, of the total)::

    'qoi_split': {'end-cell': 90},   # the two end cups take 90 between them; the mid derives the rest

Two nuances the mechanism relies on:

- **R/Q must be tuned with a cell-local handle, never** ``Req``. ``Req`` is the *shared* equator radius; moving it to change one cell's R/Q detunes every other cell (it drove one test's Req to 178 mm and left the end cell unable to reach the target frequency). Frequency is set by ``Req`` (shared) or the end length ``L`` (local); R/Q by the equator ellipse semi-axis ``A`` (local).
- **Not every split is reachable.** A wide-iris end cell at the shared ``Req`` may be unable to reach the target frequency at all (its frequency floors *above* target), or unable to reach a very low R/Q share. The tune then stops with an informative **no-solution** message — which cell, the share it could not meet, and the closest it reached — and leaves the per-cup convergence on ``cav.tune.convergence`` / ``cav.tune.plot_convergence`` so you can adjust the total, the split, or the geometry.

The achieved assembled ``(freq, ff, R/Q)`` is reported on ``cav.tune.qois``; because the split assumes additivity, compare it against your target — for a frequency-only mid it tracks to well under 1 %.

Accessing Results
*****************
Once tuning finishes, the tuned geometric variables and final frequencies are stored directly on the cavity:

.. code-block:: python

    # View final tuned parameters for a cavity
    tuned_qois = tesla.tune.qois
    pp.pprint(tuned_qois)

.. tip::

   Tuning is automatically integrated into the optimisation loop (see :doc:`optimisation`). Every candidate geometry generated by the genetic algorithm is tuned to the target frequency before its objectives are evaluated.

Worked examples:

- :doc:`examples/tuning/tune_to_frequency` — a single cell to a target frequency,
- :doc:`examples/tuning/multicell` — mid and end cells of a 9-cell cavity, with field flattening,
- :doc:`examples/tuning/cavity_types` — the same call across elliptical, pillbox, RF-gun and spline geometries.
