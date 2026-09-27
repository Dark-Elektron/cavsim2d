Eigenmode
=========

Worked examples of the eigenmode solver, ordered from the simplest geometry to
the most involved analysis: start with a cavity whose answer is known in closed
form, move on to figures of merit and passbands, then to dielectric loading and
loss, and finally to cavities whose modes leave through the beam pipe, and the
three ways of computing their external Q.

How fine a mesh has to be, and what the solver settings cost in time and buy in
accuracy, have their own section next: :doc:`../convergence/index`.

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
   dielectric_quartz_tube
   dielectric_loss
   external_q_methods
