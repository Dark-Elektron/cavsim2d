Convergence, timing and accuracy
================================

Every number the eigenmode solver reports carries two errors: the mesh's, from
discretising the cavity, and the solver's, from stopping the eigensolver at a
finite tolerance. These examples measure both, and what each costs in time.

The first two refine the mesh: one against a closed-form answer, one for every
figure of merit on a real cavity, ending with the mesh settings to use. The third
holds the mesh fixed and compares solver settings (preconditioner, tolerance, how
many modes to converge, the sparse direct solver) by time per simulation and by
the error of every frequency and figure of merit, so you can trade accuracy for
speed knowingly.

They double as a regression suite. Re-run them after changing a solver, a method
or a default, and any change in timing or accuracy shows up directly: a new
setting is one more entry in ``cav.eigenmode.benchmark(variants=...)``.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   mesh_convergence
   convergence_qois
   solver_settings
