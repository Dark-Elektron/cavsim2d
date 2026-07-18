Multipacting Analysis
=====================

Multipacting is a resonant electron avalanche: electrons emitted from a cavity
wall are accelerated by the RF field, strike the wall, release secondary
electrons, and — if the geometry and field level phase-lock the orbit and the
secondary yield exceeds unity — multiply. It can clamp the achievable RF power.
``cavsim2d`` predicts the field levels at which this resonates by tracking
electrons in the cavity's eigenmode field with secondary emission, following the
PyMultipact method :cite:p:`Udongwo2024`.

The electromagnetic field is the monopole eigenmode computed by the
:doc:`eigenmode solver <eigenmode>` — no separate field import is needed. Only
the in-plane electric field :math:`\mathbf{E}_{rz}` and the azimuthal magnetic
field :math:`H_\varphi` enter the particle dynamics.

Relativistic Particle Motion
----------------------------
Each electron (charge :math:`-q_0`, rest mass :math:`m_0`) follows the
relativistic Lorentz force in the time-harmonic field. Writing the field at a
point as the eigenmode envelope times the RF phase
:math:`\mathrm{e}^{\mathrm{i}(\omega t + \phi)}` (with launch phase :math:`\phi`),
the velocity update integrated by the solver is

.. math::
   \frac{\mathrm{d}\mathbf{u}}{\mathrm{d}t}
   = \frac{q_0}{m_0}\sqrt{1 - \left(\frac{|\mathbf{u}|}{c_0}\right)^2}
   \left[ \mathbf{E} + \mathbf{u}\times\mathbf{B}
          - \frac{1}{c_0^2}(\mathbf{u}\cdot\mathbf{E})\,\mathbf{u} \right],

with :math:`\mathbf{B} = \mu_0 H_\varphi\,\hat{\boldsymbol{\varphi}}`. The system
is advanced with a fixed-step **RK4** integrator; the time step is a fixed
fraction of the RF period. Wall crossings are detected geometrically (the
straight segment between successive positions is intersected against the local
wall polyline) and the impact point and RF phase are recovered by sub-step
interpolation.

The magnetic field is reconstructed from the electric field as
:math:`H_\varphi = \tfrac{\mathrm{i}}{\mu_0\omega}\,\nabla\times\mathbf{E}_{rz}`.
The factor :math:`\mathrm{i}` carries the physical 90-degree phase between
:math:`\mathbf{E}` and :math:`\mathbf{H}` that the Lorentz integration needs;
the eigenmode QOIs use only :math:`|H|`, so the field is rebuilt here from the
curl rather than reusing the magnitude-only stored value.

Secondary Emission
------------------
At each impact the kinetic energy is

.. math::
   E_{\mathrm{impact}} = (\gamma - 1)\, m_0 c_0^2, \qquad
   \gamma = \left(1 - (|\mathbf{u}|/c_0)^2\right)^{-1/2},

and the number of secondaries released is the secondary-emission-yield (SEY)
curve :math:`\delta(E_{\mathrm{impact}})`, a material property read from a table
(a copper-like default ships with ``cavsim2d``). Multipacting can grow only where
:math:`\delta > 1`. Whether a secondary actually leaves the wall depends on the
surface field direction at impact (:math:`\mathbf{E}\cdot\hat{\mathbf n} \ge 0`);
the ``loss_model`` selects what happens otherwise ('field' absorbs, the paper
behaviour; 'wait' re-launches uncounted; 'always' re-launches and counts).

Multipacting Metrics
--------------------
The analysis launches a cloud of electrons from the wall over a grid of emission
sites and RF launch phases, for each value of a **peak-field sweep**, and tracks
them for a fixed number of RF cycles. An electron surviving to :math:`N = 20`
impacts is a resonant ("bright") trajectory. The metrics reported against the
peak surface field :math:`E_{\mathrm{pk}}` are:

- **Counter function** :math:`c_{20}/c_0` — the fraction of launched electrons
  that reach 20 impacts. Peaks locate the multipacting barriers.
- **Enhanced counter** :math:`e_{20}/c_0` — the same, weighted by the product of
  the secondary yields along each trajectory; :math:`e_{20}/c_0 > 1` signals a
  self-sustaining avalanche.
- **Final impact energy** :math:`E_{\mathrm{f},20}` — the mean impact energy of
  the 20-hit electrons, compared against the SEY crossover energies.
- **Distance function** :math:`d_{20}` — the distance in (position, RF-phase)
  space between an electron's launch point and its 20th impact (the Ylä-Oijala
  distance metric used by MultiPac),

  .. math::
     d_{20} = \sqrt{\,\lVert \mathbf{x}_{20} - \mathbf{x}_0 \rVert^2
              + \kappa\,\lvert \mathrm{e}^{\mathrm{i}\varphi_{20}}
              - \mathrm{e}^{\mathrm{i}\varphi_0}\rvert^2\,},
     \qquad \kappa = \frac{\lambda}{2\pi},

  whose minima over (site, phase) locate the fixed points of the resonant orbits.

The method is benchmarked against MultiPac on the TESLA cavity
:cite:p:`Udongwo2024,Zhu2003`. See the :doc:`user guide <../multipacting>` for the
``cav.multipacting`` interface and the worked
:doc:`example <../examples/multipacting/tesla>`.
