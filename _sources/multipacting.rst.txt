Multipacting
============

The multipacting module tracks emitted electrons in a cavity's eigenmode field
with secondary emission and reports the field levels at which a resonant electron
avalanche builds up. It follows the solver-object pattern: everything is reached
through ``cav.multipacting``. See :doc:`theory/multipacting` for the physics.

The electromagnetic field is the **monopole eigenmode**; ``run()`` solves it
first if it has not been computed, so a single call is enough.

Interface
*********
.. code-block:: python

    from cavsim2d import EllipticalCavity

    tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
    cav = EllipticalCavity(1, tesla, tesla, tesla, beampipe='none')

    cav.multipacting.run({
        'epks': [e * 1e6 for e in range(1, 46)],   # peak fields to sweep [V/m]
        'phis': None,                              # launch phases (default 72)
        'proc_count': None,                        # parallel workers (auto)
    })

When called without a config, sensible defaults are used (a 0–80 MV/m sweep,
72 launch phases, an emission band at the equator, the bundled copper SEY).

Configuration
*************
``run()`` accepts a dictionary (or keyword arguments):

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Meaning
   * - ``mode``
     - 0-based eigenmode index (default the fundamental, the accelerating mode).
   * - ``epks``
     - Peak surface fields to sweep, in **V/m**. Default: 0–80 MV/m in 192 steps.
   * - ``phis``
     - Initial RF launch phases [rad]. Default: 72 phases over :math:`[0, 2\pi]`.
   * - ``xrange``
     - Emission-site z-interval [m]. Default: a small band at the equator (where
       elliptical-cavity multipacting concentrates).
   * - ``v_init``
     - Emission energy [eV] (default 2).
   * - ``loss_model``
     - ``'field'`` (absorb on an unfavourable surface field, default),
       ``'wait'`` (re-launch uncounted), or ``'always'`` (re-launch and count).
   * - ``proc_count``
     - Parallel workers for the field sweep. ``None`` (default) auto-picks from
       the CPU count; ``1`` runs in-process.
   * - ``t_max``
     - Track duration [s] per launch (default ``1000e-10`` ≈ 13 RF cycles).
   * - ``eigenmode_config``
     - Passed to the auto eigenmode solve (e.g. a finer ``mesh_config`` for
       near-wall accuracy).
   * - ``pec_maxh``
     - Local mesh size [mm] on the **PEC wall**, just for multipacting (see below).

.. note::
   Each parallel worker is a plain subprocess, so the sweep works identically
   from notebooks and scripts (no ``if __name__ == '__main__':`` guard needed)
   on every platform. ``proc_count=1`` runs in-process. Per-worker progress is
   written to ``multipacting/mworker_{p}.log``; if a worker crashes, its
   traceback is included in the raised error.

The **complete** merged configuration (defaults included) is what runs and what
``multipacting/config.json`` records, so a saved config always reproduces the run.

Stepwise setup and inspection
*****************************
Everything can be bundled into ``run(config)``, but each step can also be set —
and **inspected** — individually beforehand: stage the mesh, look at it, stage
the emission band, look at the emission points, *then* run.

.. code-block:: python

    mp = cav.multipacting
    mp.set_mesh_parameters(h=6, p=3, pec_maxh=1.0)   # multipacting's own mesh
    mp.show_mesh()                                   # inspect it (no solve)

    mp.set_xrange([-0.005, 0.005])                   # emission band [m]
    mp.show_emission_points()                        # the sites the tracker will use

    mp.set_epks([e * 1e6 for e in range(1, 46)])     # sweep values [V/m]
    mp.set_phis(np.linspace(0, 2 * np.pi, 72))       # launch phases

    mp.run(proc_count=1)                             # uses everything staged

``set_*()`` calls chain (each returns the solver). :meth:`show_mesh` previews
the mesh multipacting would use *right now* — the staged own-field mesh when
mesh parameters / ``pec_maxh`` were set, otherwise the eigenmode mesh;
:meth:`show_emission_points` marks the wall points the staged (or given)
``xrange`` selects.

Staged values are merged **under** the ``run()`` config. If a key is given both
ways, ``run()`` emits a ``UserWarning`` naming the key(s) — the config entry at
the point of calling ``run()`` takes precedence over the previously staged value.

Surface-refined mesh (``pec_maxh``)
***********************************
The surviving electrons live in a sub-millimetre layer at the wall and bounce
~20 times, so multipacting needs more near-wall field resolution than the
eigenmode QOIs do. ``pec_maxh`` refines the mesh **on the PEC surface only**:

.. code-block:: python

    cav.multipacting.run({'pec_maxh': 1.0, ...})    # 1 mm wall elements

With ``pec_maxh`` set, multipacting solves the monopole eigenmode **on its own
surface-refined mesh**, stored under ``multipacting/field/`` — the shared
``cav.eigenmode`` results (and everything computed from them: QOIs, impedance,
dispersion) are **left untouched**. Each analysis stays internally consistent
with its own mesh, so nothing is recomputed or replaced behind your back.
Without ``pec_maxh``, the existing eigenmode field is reused as before
(``run()`` solves it first if missing).

The multipacting-owned mesh keeps a **straight boundary** (the field itself is
solved at full order ``p``): the tracker's collision surface is the polyline
through the wall vertices, and only on a straight mesh does that polyline
coincide exactly with the element edges — the construction the original
PyMultipact validated. On a *curved* mesh the two disagree by the chord
sagitta, and computed impact points can fall outside the mesh at concave wall
sections (a ``Meshpoint not in mesh`` failure). For the same reason, if you
track with a **wide emission band** (an ``xrange`` reaching towards the iris),
prefer setting mesh parameters / ``pec_maxh`` — the curved eigenmode mesh of
the reuse path is only guaranteed near the (convex) equator, where the default
emission band sits. With the fine ``pec_maxh`` wall the geometric error of the
straight boundary is micrometres, and the fundamental agrees with the curved
eigenmode mesh to :math:`\sim 10^{-4}` relative.

Secondary emission yield
************************
A copper-like SEY curve is bundled and used by default. Supply your own
two-column table (impact energy [eV], yield):

.. code-block:: python

    cav.multipacting.set_sey('my_material.sey')
    cav.multipacting.plot_sey()

Results and visualisation
*************************
.. code-block:: python

    mp = cav.multipacting
    mp.counter          # counter function c20/c0 vs peak field
    mp.epk              # peak-field axis [MV/m]
    mp.final_energy     # mean final impact energy [eV]
    mp.results          # the full saved result dict (incl. 'sweep_time [s]')

    mp.plot_counter()              # multipacting barriers (c20/c0 vs Epk)
    mp.plot_final_energy()         # impact energy vs Epk, SEY crossovers marked
    mp.plot_enhanced_counter()     # e20/c0 (avalanche where > 1)
    mp.plot_distance_map(epk_i=10) # MultiPac-style d20 map for one field level
    mp.plot_trajectories()         # interactive viewer (notebook; ipywidgets)

Every ``plot_*`` (and ``show_*``) method **displays the figure by default**
(``show=True``): a plot appears without a manual ``plt.show()``. Pass
``show=False`` to keep the axes live for overlaying or to reuse the returned
``ax``:

.. code-block:: python

    ax = mp.plot_counter(show=False)          # don't display yet
    other.multipacting.plot_counter(ax=ax)    # overlay a second cavity, then show

``plot_counter(launchable_norm=True)`` divides by the launchable fraction
(~0.5), matching MultiPac's normalisation.

The peak-field sweep shows a **live progress bar** over the field levels as the
workers finish them (``run({'progress': False})`` to silence it), prints its
wall-clock time when it finishes, and records it as
``mp.results['sweep_time [s]']``.

``plot_trajectories(color_by='energy')`` (or ``'velocity'``) gradient-colours
the viewed path — on a **log** scale, since wall-hugging multipacting electrons
(:math:`\sim 10^2` eV) and gap-crossing electrons (:math:`\sim 10^5` eV)
coexist in the same result.

Animation
*********
``animate_trajectories()`` animates the surviving trajectories as moving heads
with a short fading trace, gradient-coloured by ``'energy'`` (default) or
``'velocity'``. In a notebook it plays **inline** automatically:

.. code-block:: python

    mp.animate_trajectories()                        # plays inline (all survivors)
    mp.animate_trajectories(epk_i=83)                # one field level
    mp.animate_trajectories(epk_i=[80, 83],          # a group of levels,
                            phi_i=range(0, 72, 8),   # phases,
                            traj=[0, 1, 2])          # and/or trajectories
    mp.animate_trajectories(epk_i=83, save='mp.gif')        # write GIF
    mp.animate_trajectories(epk_i=83, save='mp.mp4')        # write MP4 (ffmpeg)

``epk_i`` / ``phi_i`` / ``traj`` select by field level, launch phase and
trajectory index (int or list; ``None`` = all, the default). ``epk_i``
restricts the field levels; **within** them, ``traj`` (by bright-set index) and
``phi_i`` (by launch phase) are **combined as a union** — giving both shows the
named trajectories *and* those at the named phases, so adding a selector only
adds trajectories (it never intersects them to nothing). Out-of-range indices
are skipped with a warning. ``trail`` sets the trace length, ``step``
subsamples long tracks, ``fps`` the playback rate. ``zoom='auto'`` (default)
frames the trajectories — the orbits are sub-millimetre, invisible at
full-cavity scale; ``zoom=None`` shows the whole cavity.

Rendering every frame is the slow part when many particles survive, so a
**progress bar with timing** is shown (``progress=True``). The call always
returns the :class:`~matplotlib.animation.FuncAnimation`, so the two-step form
still works::

    anim = mp.animate_trajectories(embed=False)   # build without playing inline
    anim.save('animation.gif')                    # …then save (with a progress bar)

Inline playback is controlled by ``embed`` (``'auto'`` plays in a notebook
unless ``save`` was given; ``True``/``False`` forces it).

See the worked example: :doc:`examples/multipacting/tesla`.
