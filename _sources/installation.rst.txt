Installation
############

This section outlines how to install ``cavsim2d`` and configure its required external solver dependencies.

Toolkit Setup
*************

To install ``cavsim2d``, first clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/Dark-Elektron/cavsim2d.git

Navigate to the project root directory and install in editable mode:

.. code-block:: bash

    pip install -e .

To improve the readability of printed configurations and outputs, it is also recommended to install ``pprintpp``:

.. code-block:: bash

    pip install pprintpp

Third-Party Solvers
*******************

ABCI Setup (Wakefield Solver)
=============================
Wakefield and impedance analyses in ``cavsim2d`` are driven by Yang Ho Chin's **ABCI** (Azimuthal Beam Cavity Interaction) code, which solves Maxwell's equations in the time domain for axisymmetric structures.

1. Download the latest 64-bit version of ABCI (e.g., ``ABCI_MP64_12_5.zip``) from the official `ABCI website <https://abci.kek.jp/abci.htm>`_.
2. If the website download links are broken, you can fetch the zip file directly via PowerShell:

   .. code-block:: powershell

       wget http://abci.kek.jp/ABCI_MP64_12_5.zip -O ABCI_MP64_12_5.zip

3. Create an ``ABCI`` directory in ``cavsim2d/solvers`` (or ``cavsim2d/solver``):

   .. code-block:: bash

       mkdir -p cavsim2d/solver/ABCI

4. Extract and copy the executable ``ABCI_MP64_12_5.exe`` (or similar) into the newly created folder, renaming it to ``ABCI.exe``:

   .. code-block:: bash

       cp ABCI_MP_12_5/ABCI_MP64_12_5.exe cavsim2d/solver/ABCI/ABCI.exe

NGSolve Setup (Eigenmode Solver)
================================
Eigenmode simulations are run in-memory via the **NGSolve** electromagnetic finite-element engine, which is a required Python dependency. It is installed automatically during the `pip` installation of `cavsim2d`.
