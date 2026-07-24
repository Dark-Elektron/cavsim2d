Wakefield Analysis
==================

The wakefield module computes beam-induced electromagnetic wake potentials and impedance spectra using the time-domain `ABCI <https://abci.kek.jp/abci.htm>`_ (Azimuthal Beam Cavity Interaction) solver. These quantities are essential for evaluating beam stability limits, higher-order mode (HOM) power dissipation, and coupled-bunch instability thresholds.

.. note::

   ABCI must be installed separately. See :doc:`installation` for setup instructions.

Interface
*********
To run the wakefield analysis:

.. code-block:: python

    cavs.run_wakefield(wakefield_config)

Configuration Dictionary
************************
The simulation parameters are defined in a configuration dictionary. Every key
can equally be passed as a keyword argument — ``cav.wakefield.run(wakelength=80)``
— and kwargs override the dictionary. The config is merged over a complete set
of defaults, and the **merged** dict is what runs and what
``wakefield/config.json`` records:

.. code-block:: python

    wakefield_config = {
        'wakelength': 50,
        'bunch_length': 25,
        'MROT': 'both',
        'processes': 1,
        'rerun': True,
        'operating_points': op_points
    }

Settings description:

``wakelength``
   *(float, default: 50)* Length of the wake potential to compute (in metres). Longer wake lengths yield finer frequency resolution in the impedance spectrum.

``bunch_length``
   *(float, default: 25)* Longitudinal RMS bunch length :math:`\sigma_z` (in mm). This determines the driving bunch shape used in the ABCI simulation.

``MROT``
   *(int or str, default:* ``'both'`` *)* Specifies which multipole component of the wake to compute:

   - ``0`` or ``'monopole'`` — longitudinal wake only,
   - ``1`` or ``'dipole'`` — transverse wake only,
   - ``'both'`` (or ``2``) — compute both simultaneously.

``processes``
   *(int, default: 1)* Number of parallel solver processes.

``rerun``
   *(bool, default: True)* If ``True``, forces a re-run of ABCI even if cached output files exist.

``operating_points``
   *(dict)* Dictionary defining physical operating parameters of the beam. When provided, ``cavsim2d`` computes loss/kick factors and HOM power dissipation for the specified design scenarios.

Operating Points
****************
Operating points describe the beam parameters at a given machine energy. Each entry maps a label to its beam parameters:

.. code-block:: python

    op_points = {
        "Z": {
            "freq [MHz]": 400.79,
            "E [GeV]": 45.6,
            "I0 [mA]": 1280,
            "sigma_SR [mm]": 4.32,
            "Nb [1e11]": 2.76
        }
    }

Multiple operating points can be included in the dictionary to compare HOM power across different machine configurations.

Visualisation and Plotting
**************************
Wakefield results live under the ``wakefield`` namespace, mirroring the eigenmode
one, so a single cavity and a whole study plot the same way:

.. code-block:: python

    # single cavity — longitudinal impedance (|Z| vs f), in kOhm by default
    ax = cav.wakefield.plot_impedance()
    cav.wakefield.plot_impedance('transverse', ax=ax, unit='M')   # MOhm/m

    # wake potential (W vs s)
    cav.wakefield.plot_wake()

    # cumulative loss / kick factor *spectra* k(F) vs frequency
    # (ABCI's 'Loss Factor Spectrum Integrated upto F')
    cav.wakefield.plot_k_loss()
    cav.wakefield.plot_k_kick()

    # whole study — one curve per cavity, shared axis
    ax = study.wakefield.plot_impedance()
    study.wakefield.plot_k_loss()

Because both solvers report kOhm by default and take the same ``unit`` argument,
an eigenmode-reconstructed spectrum overlays a wakefield one with no rescaling:

.. code-block:: python

    ax = cav.eigenmode.plot_impedance()   # reconstructed from the modes
    cav.wakefield.plot_impedance(ax=ax)   # from the wake solve — same axis

The frames themselves are available as ``cav.wakefield.impedance(kind, unit=...)``
(and the raw ``cav.wakefield.wake_z`` / ``wake_t``). **All** wakefield plotting now
lives on the namespace: the old ``cav.plot('ZL'/'ZT'/'wpl'/'wpt')`` calls have been
removed in favour of ``plot_impedance()`` / ``plot_wake()`` above.

The scalar ``cav.wakefield.qois['|k_loss| [V/pC]']`` (a single number) is the total
loss factor; the per-cavity comparison of that number across a study is drawn by the
comparison plots (e.g. ``study.wakefield.plot_hom_bar(...)``). ``plot_k_loss()`` instead
draws the frequency-resolved cumulative spectrum k(F).

Accessing Results
*****************
Wakefield and HOM quantities of interest can be read directly:

.. code-block:: python

    # Loss and kick factors
    print(cavs.wakefield_qois)

See the worked example: :doc:`examples/wakefield/impedance`. To run the
wakefield under geometric uncertainty (a ``uq_config`` with ``ZL``/``ZT``
impedance objectives), see :doc:`examples/advanced/wakefield_uq`.
