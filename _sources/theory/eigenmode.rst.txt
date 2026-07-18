Eigenmode Analysis
==================

Maxwell Eigenvalue Problem — Unified Formulation (:math:`m \ge 0`)
------------------------------------------------------------------

Governing Equations
^^^^^^^^^^^^^^^^^^^
We start from the time-harmonic Maxwell's equations in a source-free, lossless vacuum domain :math:`\Omega` with a perfectly conducting boundary (PEC) :math:`\partial\Omega_{\text{PEC}}`:

.. math::
   \nabla \times \mathbf{E} = -i\omega\mu_0 \mathbf{H}

.. math::
   \nabla \times \mathbf{H} = i\omega\varepsilon_0 \mathbf{E}

where :math:`\omega` is the angular frequency, :math:`\mu_0` is the vacuum permeability, and :math:`\varepsilon_0` is the vacuum permittivity. Taking the curl of the first equation and substituting the second yields the vector wave equation for the electric field :math:`\mathbf{E}`:

.. math::
   \nabla \times \left( \nabla \times \mathbf{E} \right) = k^2 \mathbf{E}, \quad k^2 = \frac{\omega^2}{c_0^2}

where :math:`c_0 = 1/\sqrt{\mu_0\varepsilon_0}` is the speed of light. The boundary condition on PEC walls is:

.. math::
   \mathbf{n} \times \mathbf{E} = \mathbf{0} \quad \text{on } \partial\Omega_{\text{PEC}}

.. math::
   \mathbf{n} \times (\nabla \times \mathbf{E}) = \mathbf{0} \quad \text{on } \partial\Omega_{\text{PMC}}

Fourier Azimuthal Ansatz and Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For 2D axisymmetric geometries, we adopt cylindrical coordinates :math:`(z, r, \phi)`. We expand the 3D electric field in a Fourier series in azimuth :math:`\phi`:

.. math::
   \mathbf{E}(z, r, \phi) = \mathbf{E}_{rz}(z, r) \cos(m\phi) + E_\phi(z, r) \sin(m\phi) \mathbf{e}_\phi

where :math:`m \ge 0` is the azimuthal mode number (angular harmonic).

Weak Formulation
^^^^^^^^^^^^^^^^^^^^^^^^
The fields are discretized on the product space :math:`H(\text{curl}, \Omega_{2D}) \times H^1(\Omega_{2D})`. The trial functions are :math:`(\mathbf{u}, u_\phi)` and the test functions are :math:`(\mathbf{v}, v_\phi)`. The scalar :math:`H^1` space is constructed with an order :math:`p+1` and is zero on the axis :math:`r=0` and PEC walls (:code:`dirichlet="PEC|AXI"`). The elevated polynomial order
:math:`p+1` for the nodal space :math:`H^1_{p+1}` is not an independent choice but a consequence of the exact sequence property underlying the Nédélec/hierarchical construction of :math:`H(\mathrm{curl})_p`: the space is built such that

.. math::
   \nabla H^1_{p+1}(\Omega_{2D}) \subset H(\mathrm{curl})_p(\Omega_{2D})

holds exactly at the discrete level. 

Substituting the scaled ansatz into the wave equation, integrating against the cylindrical volume measure :math:`\mathrm{d}V = 2\pi r \mathrm{d}r \mathrm{d}z`, and integrating over :math:`\phi` yields the bilinear forms for all :math:`m \ge 0` :cite:p:`Chinellato2005`:

.. math::
   a((\mathbf{u}, u_\phi), (\mathbf{v}, v_\phi)) = \int_{\Omega_{2D}} \left[ r (\nabla \times \mathbf{u}) \cdot (\nabla \times \mathbf{v}) + \frac{1}{r} \left( m^2 \mathbf{u}\cdot\mathbf{v} + m\mathbf{u}\cdot\nabla v_\phi + m\nabla u_\phi \cdot \mathbf{v} + \nabla u_\phi \cdot \nabla v_\phi \right) \right] \mathrm{d}A

.. math::
   b((\mathbf{u}, u_\phi), (\mathbf{v}, v_\phi)) = \int_{\Omega_{2D}} \left[ r \mathbf{u}\cdot\mathbf{v} + \frac{1}{r} u_\phi v_\phi \right] \mathrm{d}A

where :math:`\mathrm{d}A = \mathrm{d}z \mathrm{d}r`.


Gradient Kernel Projection
^^^^^^^^^^^^^^^^^^^^^^^^^^
A major challenge in Maxwell eigenvalue computations is the presence of an infinite-dimensional kernel of electrostatic gradient fields that correspond to spurious modes with eigenvalue :math:`k^2 = 0`. To see how this kernel relates to the product-space formulation, we present the step-by-step derivation of the gradient operator and its real-valued transformation.

Derivation of the 3D Gradient
"""""""""""""""""""""""""""""
Let :math:`\Psi(r, z, \phi)` be a 3D scalar potential. For a given azimuthal mode number :math:`m`, we decompose it into a 2D meridian potential :math:`\psi(r, z)` multiplied by the angular phase:

.. math::
   \Psi(r, z, \phi) = \psi(r, z) e^{\mathrm{i}m\phi}

The 3D gradient operator :math:`\nabla` in cylindrical coordinates :math:`(r, z, \phi)` is defined as:

.. math::
   \nabla \Psi = \frac{\partial \Psi}{\partial r}\hat{\mathbf{r}} + \frac{\partial \Psi}{\partial z}\hat{\mathbf{z}} + \frac{1}{r}\frac{\partial \Psi}{\partial \phi}\hat{\boldsymbol{\phi}}

Substituting our ansatz :math:`\Psi = \psi e^{\mathrm{i}m\phi}` into the definition:

.. math::
   \nabla(\psi e^{\mathrm{i}m\phi}) = \frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial r}\hat{\mathbf{r}} + \frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial z}\hat{\mathbf{z}} + \frac{1}{r}\frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial \phi}\hat{\boldsymbol{\phi}}

Applying the partial derivatives:

- For the radial component: :math:`\frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial r} = \frac{\partial \psi}{\partial r}e^{\mathrm{i}m\phi}`
- For the axial component: :math:`\frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial z} = \frac{\partial \psi}{\partial z}e^{\mathrm{i}m\phi}`
- For the azimuthal component: :math:`\frac{1}{r}\frac{\partial(\psi e^{\mathrm{i}m\phi})}{\partial \phi} = \frac{1}{r}\left(\mathrm{i}m \psi e^{\mathrm{i}m\phi}\right)`

Gathering the components yields:

.. math::
   \nabla(\psi e^{\mathrm{i}m\phi}) = \left( \frac{\partial \psi}{\partial r}\hat{\mathbf{r}} + \frac{\partial \psi}{\partial z}\hat{\mathbf{z}} + \frac{\mathrm{i}m}{r}\psi\hat{\boldsymbol{\phi}} \right) e^{\mathrm{i}m\phi}

Let the 2D meridian gradient be denoted by :math:`\nabla_{rz} \psi = \frac{\partial \psi}{\partial r}\hat{\mathbf{r}} + \frac{\partial \psi}{\partial z}\hat{\mathbf{z}}`. The 3D gradient vector mapped to the product space components :math:`\begin{pmatrix} \mathbf{u} \\ u_\phi \end{pmatrix}` can be written by factoring out :math:`e^{\mathrm{i}m\phi}`:

.. math::
   \nabla_{3\mathrm{D}} \psi = \begin{pmatrix} \nabla_{rz} \psi \\ \frac{\mathrm{i}m}{r}\psi \end{pmatrix}

Transformation to a Real-Valued Formulation
"""""""""""""""""""""""""""""""""""""""""""
To avoid dealing with the coordinate singularity :math:`\frac{1}{r}` and the imaginary unit :math:`\mathrm{i}` directly inside the finite element spaces, a substitution is made for the azimuthal field component. By setting the azimuthal vector component variable to :math:`u_\phi = \mathrm{i}r E_\phi`, the operator naturally simplifies.

When mapping a scalar field :math:`\psi` to this modified product space, the substitution scales the azimuthal slot by :math:`\mathrm{i}r`. Multiplying the analytical component :math:`\frac{\mathrm{i}m}{r}\psi` by :math:`\mathrm{i}r`:

.. math::
   u_\phi = (\mathrm{i}r) \cdot \left(\frac{\mathrm{i}m}{r}\psi\right) = (\mathrm{i} \cdot \mathrm{i}) \cdot m \cdot \psi = - m \psi

Therefore, in this transformed product space coordinate system, the exact continuous gradient operator acts as:

.. math::
   G_{\text{continuous}}\psi = \begin{pmatrix} \nabla_{rz} \psi \\ -m\psi \end{pmatrix}

Discrete Projector Construction
"""""""""""""""""""""""""""""""
For any :math:`m \ge 0`, the gradient kernel :math:`\mathcal{K}_m` in the product space takes the form:

.. math::
   \mathcal{K}_m = \{ (\nabla \psi, -m \psi) : \psi \in H^1(\Omega_{2D}), \psi = 0 \text{ on } \partial\Omega_{\text{PEC}} \}

To project out this gradient kernel during the iterative PINVIT solver steps, we construct the discrete block gradient matrix :math:`G_m`:

.. math::
   G_m = \begin{bmatrix} G_{\text{rz}} \\ -m I \end{bmatrix}

where :math:`G_{\text{rz}}` is the discrete gradient operator mapping from the scalar potential space to the edge :math:`H(\text{curl})` space. We define a :math:`b`-orthogonal projector :math:`P_m` that filters out this kernel:

.. math::
   P_m = I - G_m (G_m^T B G_m)^{-1} G_m^T B

where :math:`B` is the mass matrix corresponding to the bilinear form :math:`b`. The projector :math:`P_m` is applied to wrap the preconditioned inverse system inside the PINVIT eigensolver to ensure rapid convergence without spurious mode pollution.

Quantities of Interest (QOIs)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Longitudinal QOIs
"""""""""""""""""
Once the fields are computed, the following figures of merit are evaluated:

1. **Stored Energy** :math:`U`:

   .. math::
      U = \frac{1}{2} \varepsilon_0 \int_{\Omega} |\mathbf{E}|^2 \mathrm{d}V = \pi \varepsilon_0 \int_{\Omega_{2D}} \left( r |\mathbf{E}_{rz}|^2 + \frac{1}{r} |u_\phi|^2 \right) \mathrm{d}z \mathrm{d}r

2. **Accelerating Voltage** :math:`V_{\text{acc}}` **and Gradient** :math:`E_{\text{acc}}`:

   Integrated along the :math:`z`-axis (:math:`r=0`) for a particle traveling at speed :math:`\beta c_0` :cite:p:`Wangler2008`:

   .. math::
      V_{\text{acc}} = \left| \int_{z_{\text{min}}}^{z_{\text{max}}} E_z(z, 0) e^{i \frac{\omega z}{\beta c_0}} \mathrm{d}z \right|, \quad E_{\text{acc}} = \frac{V_{\text{acc}}}{L_{\text{active}}}

   where :math:`L_{\text{active}} = 2 L_{\text{cell}} N_{\text{cells}}` is the active cavity length.

3. **Shunt Impedance over Q** :math:`(R/Q)`:

   .. math::
      \frac{R}{Q} = \frac{V_{\text{acc}}^2}{\omega U}

4. **Surface Power Dissipation** :math:`P_{\text{loss}}`:

   .. math::
      P_{\text{loss}} = \frac{1}{2} R_s \int_{\partial\Omega_{\text{PEC}}} |\mathbf{H}|^2 \mathrm{d}S = \pi R_s \int_{\Gamma_{\text{PEC}}} r \left( |H_{rz}|^2 + |H_\phi|^2 \right) \mathrm{d}s

   where :math:`R_s = \sqrt{\frac{\omega\mu_0}{2\sigma}}` is the surface resistance of the cavity wall with conductivity :math:`\sigma` :cite:p:`Padamsee2008`.

5. **Quality Factor** :math:`Q` **and Geometry Factor** :math:`G`:

   .. math::
      Q = \frac{\omega U}{P_{\text{loss}}}, \quad G = Q R_s = \frac{\omega \mu_0 \int_{\Omega} |\mathbf{E}|^2 \mathrm{d}V}{\int_{\partial\Omega} |\mathbf{H}|^2 \mathrm{d}S}

6. **Peak Fields** :math:`E_{\text{pk}}` **and** :math:`B_{\text{pk}}`:

   Pointwise maxima along the PEC boundary:

   .. math::
      E_{\text{pk}} = \max_{\Gamma_{\text{PEC}}} |\mathbf{E}|, \quad B_{\text{pk}} = \mu_0 \max_{\Gamma_{\text{PEC}}} |\mathbf{H}|


Transverse QOIs and Panofsky--Wenzel Relation
""""""""""""""""""""""""""""""""""""""""""""""
For deflecting higher-order modes (:math:`m \ge 1`), the accelerating force is transverse. Since the longitudinal field :math:`E_z` vanishes on the axis for :math:`m \ge 1`, the longitudinal voltage is integrated along an off-axis line at radius :math:`r_0` (typically half the beam aperture radius):

.. math::
   V_z(r_0) = \left| \int_{z_{\text{min}}}^{z_{\text{max}}} E_z(z, r_0) e^{i \frac{\omega z}{\beta c_0}} \mathrm{d}z \right|

By the Panofsky--Wenzel theorem, the transverse kick voltage :math:`V_t` is related to the gradient of the longitudinal voltage:

.. math::
   V_t = \frac{m V_z(r_0)}{k r_0}

The transverse shunt impedance over Q is normalized by the off-axis factor to remain independent of the choice of :math:`r_0`:

.. math::
   \left( \frac{R}{Q} \right)_t = \frac{V_t^2}{\omega U r_0^{2(m-1)}}


Impedance Spectrum from Eigenfrequencies and Quality Factors
------------------------------------------------------------

The impedance spectrum the beam sees can be *reconstructed* from the eigenmode
results alone: each resonant mode behaves as a parallel :math:`RLC` resonator,
completely characterised by its resonant frequency :math:`f_i`, its
:math:`(R/Q)_i` and its quality factor :math:`Q_i` — exactly the quantities the
eigenmode solver reports. The longitudinal beam-coupling impedance is the sum of
Breit--Wigner resonator terms

.. math::
   Z_\parallel(f) = \sum_i \frac{R_i}{1 + \mathrm{i}\, Q_i \left( \dfrac{f}{f_i} - \dfrac{f_i}{f} \right)},
   \qquad
   R_i = \frac{1}{2}\, Q_i \left( \frac{R}{Q} \right)_i

where :math:`R_i` is the shunt impedance of mode :math:`i` (the factor
:math:`1/2` reflects the accelerator convention :math:`R/Q = V^2/(\omega U)`
used above). On resonance, :math:`Z_\parallel(f_i) = R_i` is real; the
:math:`3\,\text{dB}` full width of each peak is :math:`f_i / Q_i`. As
:math:`f \to 0` every longitudinal term vanishes.

For the transverse (deflecting, :math:`m \ge 1`) impedance, the transverse
:math:`(R/Q)_t` from the Panofsky--Wenzel relation above (in :math:`\Omega`) is
converted to a transverse shunt impedance in :math:`\Omega/\mathrm{m}` by the
factor :math:`\omega_i / c_0`, and each resonator term carries an additional
:math:`f_i / f` :

.. math::
   Z_\perp(f) = \sum_i \frac{f_i}{f} \cdot
   \frac{R_{t,i}}{1 + \mathrm{i}\, Q_i \left( \dfrac{f}{f_i} - \dfrac{f_i}{f} \right)},
   \qquad
   R_{t,i} = \frac{1}{2}\, Q_i \left( \frac{R}{Q} \right)_{t,i} \frac{\omega_i}{c_0}

whose DC limit is finite and purely inductive,
:math:`Z_\perp(0) = \mathrm{i} \sum_i R_{t,i} / Q_i`. The cavity is
axisymmetric, so the two transverse planes are degenerate and a single
transverse :math:`R/Q` describes both.

Two caveats. First, the eigenmode solver reports the *unloaded* :math:`Q_0`
(wall losses only); if the mode is damped by couplers or absorbers, the beam
sees the loaded :math:`Q_L`, which should be substituted for :math:`Q_i` —
lowering :math:`Q` broadens the resonance and reduces its peak while leaving
:math:`R/Q` (a pure geometry quantity) unchanged. Second, the reconstruction
contains exactly the modes that were solved and nothing else: no
broadband/resistive-wall contribution and nothing above the highest computed
mode. Within that range, however, it is essentially exact — it has no
wake-length truncation, so it complements a wakefield solve, whose spectral
resolution :math:`\Delta f \sim c_0 / (2 L_{\text{wake}})` limits how well
narrow (high-:math:`Q`) resonances are resolved. ``cavsim2d`` exposes both
through the same frame layout (``cav.eigenmode.impedance()`` vs
``cav.wakefield.impedance()``) so the reconstructed and the simulated spectra
can be overlaid directly.


Zienkiewicz--Zhu (ZZ) Error Estimator
--------------------------------------

A Posteriori Error Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To achieve high precision with minimum degrees of freedom, ``cavsim2d`` implements an adaptive mesh refinement (AMR) loop driven by a recovery-based a posteriori error estimator. 

The standard finite-element approximation yields a discontinuous magnetic field :math:`\mathbf{H} \propto \nabla \times \mathbf{E}` across element boundaries. The Zienkiewicz--Zhu (ZZ) technique reconstructs a continuous recovered field :math:`\mathbf{H}^*` in the :math:`H^1` space via patch recovery :cite:p:`ZienkiewiczZhu1987,ZienkiewiczZhu1992`:

.. math::
   \mathbf{H}^* = \sum_{j} H^*_j \phi_j

where :math:`\phi_j` are the continuous nodal basis functions. The recovered nodal values :math:`H^*_j` are computed by a least-squares projection of the discontinuous FE field :math:`\mathbf{H}` over a patch of elements surrounding node :math:`j`.

Element-wise Error Indicator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The error in element :math:`K` is estimated as the difference between the recovered field and the raw FE field:

.. math::
   \eta_K^2 = \int_K r \left| \mathbf{H} - \mathbf{H}^* \right|^2 \mathrm{d}A

.. _dorfler-marking:

Doerfler Marking and Refinement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In each refinement step, we compute the indicators :math:`\eta_K` for all elements. Let :math:`\eta_{\text{max}} = \max_K \eta_K`. We mark elements for refinement using the Doerfler marking strategy with threshold parameter :math:`\theta \in (0, 1]` :cite:p:`Doerfler1996`:

.. math::
   \text{Mark } K \quad \text{if } \eta_K > \theta \eta_{\text{max}}

Marked elements are subdivided using Netgen's 2D local refinement algorithms, and the polynomial order :math:`p` is re-applied to curve the newly created elements along the boundary. The refinement terminates when the maximum error across all solved physical modes falls below a tolerance:

.. math::
   \max_{\text{modes}} \left( \max_K \eta_K \right) < \text{tol}
