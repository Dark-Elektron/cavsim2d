"""Waveguide-port operators for the axisymmetric solvers.

A port is a **transparent boundary condition**: it terminates a beam pipe in the
exact modal impedance of that pipe, so a wave above cutoff leaves and does not come
back, while one below cutoff is stored rather than absorbed. That is what a PML and
a lossy absorber only approximate.

The condition is the modal admittance operator

    n x H = Y(w) E_t,      Y(w) = sum_n y_n(w) <., e_n> e_n

with ``e_n`` the transverse profile of the n-th pipe mode and
``y_n = w eps0 / beta_n`` its wave admittance. In a weak form it is a **boundary
bilinear term**, and because ``Y`` is diagonal in the modal basis it is a sum of
rank-1 outer products over the port degrees of freedom -- cheap to build, and
cheap to update when ``w`` changes.

Two solvers share this module:

- :mod:`eigen_ports` puts ``Y(w)`` into the eigenproblem. Since
  ``beta_n(w) = sqrt(k0^2 - k_c,n^2)`` depends on the eigenvalue, that makes it a
  **nonlinear** eigenproblem, whose complex ``w`` gives ``Q_ext`` directly.
- :mod:`frequency_domain_ngsolve` freezes ``w`` and solves a linear system per
  frequency.

Azimuthal order
---------------
The 2D formulation solves one azimuthal order ``m`` at a time and the orders do not
couple, so a port basis is per-``m`` as well:

- ``m = 0`` (monopole): TM_0n and TE_0n. At m = 0 those two blocks decouple from
  each other: a TM (accelerating) mode radiates only into TM_0n, a TE mode only into
  TE_0n, whose cutoff is higher. TE_11 does not appear at all -- it is an m = 1 mode.
  This is the piece that surprises people coming from 3D, where TE_11 is the lowest
  pipe mode of all.
- ``m >= 1``: TE_mn and TM_mn both couple. The two polarisations of a given mode
  (the "H" and "V" of TE_11) are the *same* 2D solution rotated by 90 degrees, so
  they contribute **one** port mode here, not two -- the degeneracy is already
  factored out by the ``cos(m phi)/sin(m phi)`` ansatz. Adding both would
  double-count.

Transverse profiles
-------------------
With ``E_r ~ cos(m phi)`` and ``E_phi ~ sin(m phi)``, the transverse field of a pipe
mode of cutoff ``k`` is

    TM_mn:  e_r = k J_m'(k r),       e_phi = -m J_m(k r) / r
    TE_mn:  e_r = -m J_m(k r) / r,   e_phi = k J_m'(k r)

normalised so ``integral (e_r^2 + e_phi^2) r dr = 1`` over the pipe. The wave
admittances are ``y_TM = w eps0 / beta`` and ``y_TE = beta / (w mu0)``.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import jn_zeros, jnp_zeros, jv, jvp

from ngsolve import CoefficientFunction, IfPos, LinearForm, ds, x, y  # type: ignore

from cavsim2d.constants import c0, mu0
from cavsim2d.solvers.NGSolve.eigen_ngsolve import get_boundary_nodes

C0 = c0
MU0 = mu0
EPS0 = 1.0 / (MU0 * C0 ** 2)
Z0 = MU0 * C0

#: Terms in the J_1 ascending series. It alternates, so it loses digits to
#: cancellation as ``u`` grows: ~1e-13 at u ~ 9 (the third zero of J_0), but useless
#: by u ~ 37. :func:`cutoffs` refuses mode counts that would take it out of range.
_J1_TERMS = 30

#: Largest ``u = k_c R`` the series is trusted to. Beyond it the cancellation eats
#: the answer and the port profile would be silently wrong.
_J1_MAX_ARG = 12.0


def j1_cf(k_c, r=None):
    """``J_1(k_c * r)`` as an NGSolve CoefficientFunction.

    Bessel functions are not ngsolve primitives, so the ascending series

        J_1(u) = (u/2) sum_m (-1)^m / (m! (m+1)!) (u^2/4)^m

    is evaluated by Horner in ngsolve arithmetic. Exact, not a fit: checked against
    ``scipy.special.j1`` to 2e-14 over the range :data:`_J1_MAX_ARG` allows.
    """
    r = y if r is None else r
    u = k_c * r
    q = u * u / 4.0
    coeffs = [((-1.0) ** m) / (float(math.factorial(m)) * float(math.factorial(m + 1)))
              for m in range(_J1_TERMS)]
    acc = CoefficientFunction(coeffs[-1])
    for a in reversed(coeffs[:-1]):
        acc = a + q * acc
    return (u / 2.0) * acc


def _series_cf(n, u, terms=_J1_TERMS):
    """``sum_s (-1)^s / (s! (s+n)!) (u^2/4)^s`` by Horner, as a CoefficientFunction."""
    q = u * u / 4.0
    coeffs = [((-1.0) ** s) / (float(math.factorial(s)) * float(math.factorial(s + n)))
              for s in range(terms)]
    acc = CoefficientFunction(coeffs[-1])
    for a in reversed(coeffs[:-1]):
        acc = a + q * acc
    return acc


def _power_cf(base, n):
    out = CoefficientFunction(1.0)
    for _ in range(int(n)):
        out = out * base
    return out


def jn_cf(n, k, r=None):
    """``J_n(k r)`` as a CoefficientFunction, by its ascending series (see
    :func:`j1_cf`, of which this is the general order)."""
    r = y if r is None else r
    u = k * r
    return _power_cf(u / 2.0, n) * _series_cf(n, u)


def jn_over_u_cf(n, k, r=None):
    """``J_n(u) / u`` at ``u = k r``, for ``n >= 1``: regular on the axis, where a
    literal division would be 0/0."""
    if int(n) < 1:
        raise ValueError('J_0(u)/u is singular on the axis; n must be >= 1.')
    r = y if r is None else r
    u = k * r
    return _power_cf(u / 2.0, int(n) - 1) * _series_cf(n, u) / 2.0


def jn_prime_cf(n, k, r=None):
    """``J_n'(k r)``, the derivative with respect to the argument."""
    if int(n) == 0:
        return -jn_cf(1, k, r)
    return 0.5 * (jn_cf(int(n) - 1, k, r) - jn_cf(int(n) + 1, k, r))


def mode_profile(m, k_c, kind, radius):
    """``(e_r, e_phi)`` of the TE/TM mode ``m`` with cutoff *k_c*, normalised so
    ``integral (e_r^2 + e_phi^2) r dr = 1`` over ``[0, radius]``. See the module
    docstring for the profiles. Returns CoefficientFunctions in ``r = y`` (``e_phi``
    is ``None`` for TM_0n and ``e_r`` for TE_0n, which have no such component)."""
    m, kind = int(m), str(kind).upper()

    def np_profile(r):
        u = k_c * r
        d = k_c * jvp(m, u)
        s = (m * k_c * (jv(m, u) / u if u > 1e-12 else (0.5 if m == 1 else 0.0))
             if m else 0.0)
        return (d, -s) if kind == 'TM' else (-s, d)

    norm2, _ = quad(lambda r: r * sum(v * v for v in np_profile(r)), 0.0, float(radius),
                    limit=200, epsabs=0.0, epsrel=1e-12)
    norm = np.sqrt(norm2)
    d = k_c * jn_prime_cf(m, k_c) / norm
    s = m * k_c * jn_over_u_cf(m, k_c) / norm if m else None
    if kind == 'TM':
        return d, (None if s is None else -s)
    return (None if s is None else -s), d


def cutoffs(radius, n_modes, m=0, kind='TM'):
    """Cutoff wavenumbers of the first *n_modes* pipe modes of order *m*.

    ``TM_mn`` sits at the n-th zero of ``J_m``; ``TE_mn`` at the n-th zero of
    ``J_m'`` (for ``m = 0`` that excludes the trivial zero at the origin).
    """
    kind = str(kind).upper()
    if kind == 'TM':
        zeros = jn_zeros(int(m), int(n_modes))
    elif kind == 'TE':
        zeros = jnp_zeros(int(m), int(n_modes))
    else:
        raise ValueError(f"kind must be 'TM' or 'TE'; got {kind!r}.")
    if zeros[-1] > _J1_MAX_ARG:
        raise ValueError(
            f"{n_modes} {kind}_{m}n port modes need J-series arguments up to "
            f"{zeros[-1]:.1f}, past the {_J1_MAX_ARG} this implementation is accurate "
            f"to (the alternating series loses ~13 digits by u ~ 37). Use fewer port "
            f"modes: the ones past the first evanescent pair contribute a reactance "
            f"that is already tiny.")
    return zeros / float(radius)


def propagation_constant(k0, k_c):
    """``beta`` of a mode with cutoff *k_c* at free-space wavenumber *k0*.

    Above cutoff ``beta`` is real and positive: a wave that leaves and does not come
    back. Below it ``beta = -i|beta|``, the branch on which ``exp(-i beta z)``
    **decays**. Taking the other root turns the port into a source, and every
    downstream quantity -- ``Re Z``, ``Q_ext`` -- comes out with the wrong sign.

    Accepts a complex *k0*, which the nonlinear eigenproblem needs: there the
    eigenvalue itself is complex and ``beta`` is evaluated on it.
    """
    k0c = complex(k0)
    if abs(k0c.imag) < 1e-30:
        k0r, k_c = float(k0c.real), float(k_c)
        if k0r > k_c:
            return complex(np.sqrt(k0r ** 2 - k_c ** 2), 0.0)
        return complex(0.0, -np.sqrt(k_c ** 2 - k0r ** 2))
    # complex k0: pick the sheet with Re >= 0 (outgoing) and Im <= 0 (decaying)
    b = np.sqrt(k0c ** 2 - float(k_c) ** 2 + 0j)
    if b.real < 0 or (abs(b.real) < 1e-30 and b.imag > 0):
        b = -b
    return complex(b)


def port_planes(mesh, bnd='PMC'):
    """``[(z, radius, sign), ...]`` for the pipe ends carrying boundary *bnd*.

    Both ends share one boundary name, so they are told apart by which side of the
    mid-plane they sit on; *sign* is +1 for the right-hand end.
    """
    pts = np.array(list(get_boundary_nodes(mesh, bnd)))
    if pts.size == 0:
        raise RuntimeError(
            f"the mesh has no {bnd!r} boundary, so there are no pipe ends to put "
            "ports on. Mesh with boundary_conditions='mm' — a port replaces that "
            "natural condition rather than a Dirichlet one.")
    zl, zr = float(pts[:, 0].min()), float(pts[:, 0].max())
    if abs(zr - zl) < 1e-9:
        raise RuntimeError("both pipe ends came out at the same z; a port solve needs "
                           "a beam pipe at each end.")
    rl = float(pts[np.isclose(pts[:, 0], zl), 1].max())
    rr = float(pts[np.isclose(pts[:, 0], zr), 1].max())
    return [(zl, rl, -1.0), (zr, rr, +1.0)]


def projection_vectors(mesh, fes, n_port_modes=3, m=0, kind='TM', bnd='PMC'):
    """``(vectors, cutoffs)`` -- one projection vector and cutoff per port mode.

    ``P_n[i] = integral (r (v_i)_r e_r + (v_i)_phi e_phi) dr`` over that end, the
    overlap of test function ``i`` with the mode's transverse profile (the azimuthal
    unknown is ``u_phi = r E_phi``, so its term carries no extra ``r``). With the
    profile normalised to ``integral (e_r^2 + e_phi^2) r dr = 1`` the modal
    coefficient is exactly the wave admittance, with no leftover factor.

    *kind* is ``'TM'`` or ``'TE'``; call it once per family. Any ``m``.
    """
    v, v_phi = fes.TestFunction()
    planes = port_planes(mesh, bnd)
    z_mid = 0.5 * (planes[0][0] + planes[1][0])
    vecs, kcs = [], []
    for _z_port, radius, sign in planes:
        side = IfPos(sign * (x - z_mid), 1.0, 0.0)
        for k_c in cutoffs(radius, n_port_modes, m=m, kind=kind):
            e_r, e_phi = mode_profile(m, float(k_c), kind, radius)
            integrand = CoefficientFunction(0.0)
            # .Trace(): an HCurl test function has no volume value on a boundary
            # form. On the port plane that trace IS the radial component.
            if e_r is not None:
                integrand = integrand + y * v.Trace()[1] * e_r
            if e_phi is not None:
                integrand = integrand + v_phi * e_phi
            lf = LinearForm(fes)
            lf += (integrand * side) * ds(definedon=mesh.Boundaries(bnd))
            lf.Assemble()
            vecs.append(np.array(lf.vec.FV().NumPy(), dtype=complex).copy())
            kcs.append(float(k_c))
    return vecs, np.array(kcs)
