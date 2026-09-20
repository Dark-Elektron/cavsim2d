Beam lines
==========

Worked examples of the beam-line elements and of concatenating devices into a
single simulatable structure. The two standalone elements come first, then the
assembly that joins them, then a module of cavities, then optimising a cavity
with its neighbours in place.

A beam line is built by adding devices together::

    line = (Beampipe(R=50, L=120)
            + Taper(R_left=50, R_right=80, L=80)
            + EllipticalCavity(2, mid_cell, beampipe='none')
            + Bellows(Ri=80, A=12, L_p=10, N_conv=4, R_root=2, R_crest=2)
            + BLA(R=50, L=160, absorber_length=90, absorber_thickness=10))

The result is an :class:`~cavsim2d.models.assembly.Assembly`, which is itself a
:class:`~cavsim2d.models.base.Cavity`: it meshes, solves, tunes and optimises
like any single geometry. Use ``chain=`` instead when you want *one* cavity
repeated into a module — see :doc:`../studies/index`.

Each element's parameters are named and dimensioned in :ref:`geometry-beamline`.

.. toctree::
   :maxdepth: 1
   :titlesonly:

   taper
   bellows
   beam_line_assembly
   multi_cavity_module
   optimise_assembly
