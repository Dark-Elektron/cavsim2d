Installation
############

This section outlines how to install ``cavsim2d`` and configure its required external solver dependencies.

Toolkit Setup
*************

Install the released package from PyPI:

.. code-block:: bash

    pip install cavsim2d

Or, to follow development or change the code, clone the repository from GitHub and install it in editable mode:

.. code-block:: bash

    git clone https://github.com/Dark-Elektron/cavsim2d.git
    cd cavsim2d
    pip install -e .

Either way the core dependencies, NGSolve and gmsh among them, are installed with it. So are the Jupyter widget packages that NGSolve's interactive viewer needs, so ``cav.show_fields()``, ``cav.show_mesh()`` and ``cav.show_geometry()`` render in a notebook (JupyterLab, Jupyter Notebook 7 or VS Code) with nothing else to install. Recent NGSolve releases fetch the viewer's JavaScript from cdn.jsdelivr.net, so the notebook needs internet access for it. Outside a notebook, or when the viewer cannot render, these methods draw a static matplotlib view instead and say why.

To run the example notebooks you also need a Jupyter kernel and front end. The optional ``jupyter`` extra installs JupyterLab and the kernel: ``pip install "cavsim2d[jupyter]"`` from PyPI, or ``pip install -e ".[jupyter]"`` in a clone. VS Code users only need the kernel, which VS Code offers to install.

To improve the readability of printed configurations and outputs, it is also recommended to install ``pprintpp``:

.. code-block:: bash

    pip install pprintpp

Third-Party Solvers
*******************

ABCI Setup (Wakefield Solver)
=============================
Wakefield and impedance analyses in ``cavsim2d`` are driven by Yong Ho Chin's ABCI (Azimuthal Beam Cavity Interaction) code, which solves Maxwell's equations in the time domain for axisymmetric structures.

ABCI is free to use but not open source, and KEK asks that it be obtained from its own page, so ``cavsim2d`` does not include it. Eigenmode, tuning, UQ and optimisation work without it.

1. Download ABCI from the `ABCI page at KEK <https://abci.kek.jp/abci.htm>`_. For 64-bit Windows this is ``ABCI_MP64_12_5.zip`` at the time of writing.
2. Unzip it anywhere. ABCI needs no installation.
3. Set the ``CAVSIM2D_ABCI`` environment variable to the executable (for example ``ABCI_MP64_12_5.exe``), or to the folder that holds ``ABCI.exe``. On Windows, ``setx`` makes it persist for new terminals and Jupyter sessions:

   .. code-block:: powershell

       setx CAVSIM2D_ABCI "C:\ABCI\ABCI_MP64_12_5.exe"

   On Linux or macOS:

   .. code-block:: bash

       export CAVSIM2D_ABCI="$HOME/abci/ABCI_MP64_12_5.exe"

   You can also set it from Python before the first wakefield run:

   .. code-block:: python

       import os
       os.environ['CAVSIM2D_ABCI'] = r'C:\ABCI\ABCI_MP64_12_5.exe'

ABCI is a Windows binary. On Linux and macOS it runs through `wine <https://www.winehq.org>`_, which must be installed. If ABCI or wine is missing, a wakefield run stops at once with a message saying where to get it.

If someone else wants ABCI, point them to the KEK page rather than passing on a copy.

NGSolve Setup (Eigenmode Solver)
================================
Eigenmode simulations are run in-memory via the NGSolve electromagnetic finite-element engine, which is a required Python dependency. It is installed automatically during the `pip` installation of `cavsim2d`.
