Eigenmode
=========

Worked examples of the eigenmode solver, ordered from the simplest geometry to
the most involved analysis: start with a cavity whose answer is known in closed
form, move on to figures of merit and passbands, then to numerical convergence —
first of the frequency, then of every figure of merit, which is what qualifies a
mesh for a tolerance study — then to dielectric loading and loss, and finally to
cavities whose modes leave through the beam pipe, and the three ways of computing
their external Q.

Two eigenmode examples appear later, once the tools they rely on have been
introduced: reconstructing the impedance from open-boundary modes is compared
against the wakefield solver (:doc:`../wakefield/index`), and the multicell
sensitivity study needs tuning and uncertainty quantification
(:doc:`../advanced/index`).

.. toctree::
   :maxdepth: 1
   :titlesonly:

   pillbox
   elliptical_tesla
   cavity_types
   compare_cavities
   dispersion
   mesh_convergence
   convergence_qois
   dielectric_quartz_tube
   dielectric_loss
   external_q_methods
