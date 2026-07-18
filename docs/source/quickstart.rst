Quickstart Tutorial
===================

This tutorial walks you through setting up your first project, defining an elliptical cavity geometry, running an eigenmode analysis, and plotting the resulting mesh and electromagnetic fields.

Core Concepts
*************

The core workflow of ``cavsim2d`` revolves around two main classes:

1. **``Cavity`` (and subclasses like ``EllipticalCavity``):** Represents a single physical RF cavity device, holding its cell dimensions, geometry configurations, and local analysis results.
2. **``Cavities`` (Study Manager):** A study manager class that groups multiple ``Cavity`` objects under a single project folder, handles directory layouts, triggers parallelised analyses, and facilitates comparisons between different designs.

Step 1: Set Up the Project and Geometry
***************************************

We begin by importing the necessary classes and initializing a ``Cavities`` project manager. We then define the dimensions for a 9-cell TESLA cavity and construct an ``EllipticalCavity`` object.

Create a new Python script (e.g., ``run_tutorial.py``) and add the following:

.. code-block:: python

    import sys
    import pprint
    from cavsim2d import Cavities, EllipticalCavity

    # Initialize a pretty printer for readable output
    pp = pprint.PrettyPrinter(indent=4)

    # Initialize the project folder
    # This directory will hold all meshes, solver inputs, and result JSON files
    cavs = Cavities(project_folder='./tutorial_project', overwrite=True)

    # Define dimensions for a standard 9-cell TESLA cavity (in mm)
    # IC: [A, B, a, b, Ri, L, Req]
    n_cells = 9
    midcell = [42, 42, 12, 19, 35, 57.7, 103.353]
    endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
    endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]

    # Create the elliptical cavity object
    tesla = EllipticalCavity(n_cells, midcell, endcell_l, endcell_r, beampipe='both')

Step 2: Inspect and Add the Cavity
**********************************

You can inspect the geometry properties or plot the cavity boundary profile using matplotlib:

.. code-block:: python

    # Plot the geometry profile
    tesla.plot('geometry')

    # Add the cavity to the study manager
    cavs.add_cavity([tesla], names=['TESLA'], plot_labels=['TESLA'])

Step 3: Run the Eigenmode Solve
*******************************

Running the eigenmode analysis is a single call. Behind the scenes, ``cavsim2d`` meshes the geometry and invokes the unified product-space NGSolve electromagnetic solver.

.. code-block:: python

    # Run the eigenmode solver
    cavs.run_eigenmode()

    # Print the fundamental mode Quantities of Interest (QOIs)
    print("Fundamental Mode Quantities of Interest:")
    pp.pprint(cavs.eigenmode_qois)

Step 4: Visualise Mesh and Fields
*********************************

Meshes and field profiles are properties of the individual ``Cavity`` objects. You can access individual cavities from the manager by index or by name to plot them:

.. code-block:: python

    # Plot the finite element mesh
    cavs[0].show_mesh()

    # Plot the electric (E) and magnetic (H) field magnitudes for the fundamental mode (mode 1)
    cavs['TESLA'].show_fields(mode=1, which='E')
    cavs['TESLA'].show_fields(mode=1, which='H')

Next Steps
**********
Now that you have successfully completed a basic simulation run, explore the analysis module guides for more advanced workflows:

- :doc:`eigenmode` — configure the solver, adaptive meshing, and extract RF figures of merit.
- :doc:`tuning` — adjust cavity dimensions to converge on a target frequency.
- :doc:`wakefield` — compute beam-induced wake potentials and impedance spectra with ABCI.
- :doc:`optimisation` — run multi-objective genetic algorithm searches over the design space.
- :doc:`uq` — propagate fabrication tolerances to quantify statistical spreads of RF parameters.
