Uncertainty Quantification and Sensitivity Analysis
===================================================

Manufacturing tolerances and tuning errors perturb a cavity's geometry, and
those perturbations propagate to the figures of merit. ``cavsim2d`` answers two
questions: *how much does a QOI scatter* (uncertainty quantification), and
*which input dimensions drive that scatter* (sensitivity analysis)
:cite:p:`Saltelli2008`.

Problem Setup
-------------
Let :math:`\mathbf{x} = (x_1, \dots, x_d)` be the uncertain geometric inputs
(half-cell parameters, or shape control points) with a joint probability density
:math:`\rho(\mathbf{x})` on a domain :math:`\Gamma`, and let :math:`Q(\mathbf{x})`
be a scalar QOI obtained by solving the (eigenmode or wakefield) model at
:math:`\mathbf{x}`. The uncertain inputs are taken independent and each mapped to
a reference interval :math:`[-1, 1]`.

Moment Propagation by Cubature
------------------------------
The statistics of :math:`Q` are its weighted moments under :math:`\rho`:

.. math::
   \mathbb{E}[Q] = \int_\Gamma Q(\mathbf{x})\, \rho(\mathbf{x})\, \mathrm{d}\mathbf{x},
   \qquad
   \mathrm{Var}[Q] = \int_\Gamma \big(Q(\mathbf{x}) - \mathbb{E}[Q]\big)^2 \rho(\mathbf{x})\, \mathrm{d}\mathbf{x}

Rather than sampling these integrals with Monte Carlo, ``cavsim2d`` evaluates
them with a **cubature rule**: a small set of nodes :math:`\{\mathbf{x}^{(i)}\}_{i=1}^{N}`
and weights :math:`\{w_i\}` chosen so that the weighted sum is exact for
polynomials up to a given total degree. The model is solved once per node, and
the moments are the weighted sums

.. math::
   \mathbb{E}[Q] \approx \sum_{i=1}^{N} w_i\, Q(\mathbf{x}^{(i)}),
   \qquad
   \mathrm{Var}[Q] \approx \sum_{i=1}^{N} w_i\, \big(Q(\mathbf{x}^{(i)}) - \mathbb{E}[Q]\big)^2

The skewness and (non-excess) kurtosis follow from the higher central moments,
normalised by the standard deviation :math:`\sigma = \sqrt{\mathrm{Var}[Q]}`:

.. math::
   \gamma_1 = \frac{1}{\sigma^3}\sum_{i=1}^{N} w_i\, \big(Q(\mathbf{x}^{(i)}) - \mathbb{E}[Q]\big)^3,
   \qquad
   \gamma_2 = \frac{1}{\sigma^4}\sum_{i=1}^{N} w_i\, \big(Q(\mathbf{x}^{(i)}) - \mathbb{E}[Q]\big)^4

The default rule is a degree-3 **Stroud** cubature :cite:p:`Stroud1971`, which
needs only :math:`\mathcal{O}(2d)` nodes for :math:`d` inputs — far fewer model
solves than Monte Carlo for the same accuracy on smooth responses. Degree-5
Stroud, tensor Gauss--Legendre, and Latin-hypercube / Monte-Carlo sampling are
also available (the sampling rules fall back to uniform weights
:math:`w_i = 1/N`).

Sensitivity: Sobol Indices
--------------------------
Variance-based (Sobol) sensitivity analysis attributes the output variance
:math:`\mathrm{Var}[Q]` to the inputs through the ANOVA/Sobol decomposition
:cite:p:`Sobol2001`. The **first-order index** of input :math:`x_j`,

.. math::
   S_j = \frac{\mathrm{Var}_{x_j}\!\big(\mathbb{E}[\, Q \mid x_j \,]\big)}{\mathrm{Var}[Q]},

is the fraction of the variance removed on average by fixing :math:`x_j` alone —
its main effect. The **total-effect index**,

.. math::
   S_{T_j} = 1 - \frac{\mathrm{Var}_{\mathbf{x}_{\sim j}}\!\big(\mathbb{E}[\, Q \mid \mathbf{x}_{\sim j} \,]\big)}{\mathrm{Var}[Q]},

additionally captures every interaction :math:`x_j` participates in, so
:math:`S_{T_j} \ge S_j`, with equality when :math:`x_j` has no interactions. The
inputs with the largest :math:`S_{T_j}` are the ones whose tolerances matter
most. The indices are estimated from the model evaluations (over a surrogate or
directly), giving a ranked, dimensionless attribution of the QOI's scatter to the
geometry.
