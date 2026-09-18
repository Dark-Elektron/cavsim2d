r"""Lorentz radiation pressure on a cavity wall, and the detuning it causes.

Theory
------
The RF field exerts a pressure on the conducting wall. With fields written as peak
amplitudes the time-averaged energy densities are :math:`u_H=\tfrac14\mu_0|H|^2` and
:math:`u_E=\tfrac14\varepsilon_0|E|^2`, and the outward pressure is their difference:

.. math::  P = \tfrac14\left(\mu_0|H|^2 - \varepsilon_0|E|^2\right)

so the magnetic field near the equator pushes the wall **out** and the electric field near
the iris pulls it **in**. On a perfect conductor the tangential electric and normal
magnetic components vanish, so :math:`|E|` is purely normal there and :math:`|H|` purely
tangential.

Given a wall displacement :math:`\mathbf{u}`, Slater's theorem gives the frequency shift
as the work the pressure does against that displacement, normalised by the stored energy:

.. math::  \frac{\Delta\omega}{\omega}
           = -\frac{1}{U}\oint_S P\,(\mathbf{u}\cdot\mathbf{n})\,\mathrm{d}S

Pushing the wall **in** where the magnetic field is strong raises the frequency; pushing
it in where the electric field is strong lowers it. The static deflection is itself
proportional to the pressure, and the pressure to :math:`E_\mathrm{acc}^2`, so the shift
is quoted as the Lorentz detuning coefficient

.. math::  K_L = \frac{\Delta f}{E_\mathrm{acc}^2}\quad[\mathrm{Hz/(MV/m)^2}]

which is negative for an unstiffened cavity: the inward pull at the iris and the outward
push at the equator both lower the frequency of the accelerating mode.

Scope
-----
The pressure follows from the RF solution with no further modelling. The **displacement
does not** - it depends on the wall thickness, the elastic constants, and above all on how
the cavity is held, none of which the RF model knows. :func:`slater_shift` therefore takes
the displacement as an argument rather than computing it: supply one from a mechanical
solve and this module turns it into a frequency shift.

Validation
----------
A TM010 pillbox has :math:`f = c\,x_{01}/(2\pi R)`, which depends on the radius and not on
the length. Both identities are reproduced: a uniform outward displacement of the barrel
gives :math:`\Delta f/f = -\alpha/R` to 0.001%, and the same displacement of the end
plates gives zero.
"""
import numpy as np
from ngsolve import (Conj, GridFunction, H1, Integrate, InnerProduct, Norm,
                     VectorH1, specialcf, y)

from cavsim2d.solvers.NGSolve.eigen_ngsolve import get_boundary_nodes

__all__ = ['MU0', 'EPS0', 'pressure_cf', 'wall_pressure', 'stored_energy',
           'slater_shift', 'wall_force', 'detuning_coefficient']

MU0 = 4e-7 * np.pi
EPS0 = 8.8541878128e-12


def _projected_fields(mesh, gfu_E, gfu_H, mode, order):
    """E and H projected into continuous H1, so their traces integrate on a boundary.

    Two separate reasons, both of which silently corrupt a wall integral:

    * **H** is reconstructed as ``curl(E)/(mu0 w)``, and a GridFunction curl cannot
      be SIMD-evaluated on a boundary - the integral fails outright with "SIMD:
      don't know how I shall evaluate".
    * **E** lives in HCurl, whose boundary trace keeps only the TANGENTIAL part. On
      a perfect conductor that part is zero by definition, so an unprojected
      ``|E|^2`` evaluates to nothing on the wall and the electric half of the
      pressure disappears without any error being raised.

    The eigenmode solver projects H the same way for its wall-loss integral. On a
    PML mesh the projection is restricted to the physical region: H1 is continuous,
    so projecting across the interface would average in the complex-stretched
    layer's own field.
    """
    u_gf, uphi_gf = gfu_E[mode].components
    H_inplane, H_phi = gfu_H[mode]
    kw = ({'definedon': mesh.Materials('phys')}
          if 'pml' in mesh.GetMaterials() else {})
    ein = GridFunction(VectorH1(mesh, order=order, complex=True))
    ein.Set(u_gf, **kw)
    hphi = GridFunction(H1(mesh, order=order, complex=True))
    hphi.Set(H_phi, **kw)
    hin = GridFunction(VectorH1(mesh, order=order, complex=True))
    hin.Set(H_inplane, **kw)
    return ein, uphi_gf, hin, hphi


def _field_norms(gfu_E, gfu_H, mode=0, m=0, mesh=None, project=False):
    """``(|E|^2, |H|^2)`` as coefficient functions.

    A monopole keeps all of E in the in-plane block and all of H in the azimuthal
    one, so the other terms drop out. The general form is kept because an m>=1 mode
    populates both.

    Set *project* for a boundary integral; see :func:`_projected_H`.
    """
    u_gf, uphi_gf = gfu_E[mode].components
    if project:
        u_gf, uphi_gf, H_inplane, H_phi = _projected_fields(
            mesh, gfu_E, gfu_H, mode, u_gf.space.globalorder)
        e2 = InnerProduct(u_gf, u_gf)
        h2 = InnerProduct(H_inplane, H_inplane) + Norm(H_phi) ** 2
        if m != 0:
            e2 = e2 + Norm(uphi_gf) ** 2 / (y * y + 1e-30)
        return e2, h2
    H_inplane, H_phi = gfu_H[mode]
    e2 = InnerProduct(u_gf, u_gf)
    h2 = InnerProduct(H_inplane, H_inplane) + Norm(H_phi) ** 2
    if m != 0:
        # E_phi = u_phi / r. The wall reaches the axis only at a beam-pipe end cap,
        # where u_phi vanishes anyway, so the guard only avoids a 0/0.
        e2 = e2 + Norm(uphi_gf) ** 2 / (y * y + 1e-30)
    return e2, h2


def _azimuthal_factor(m):
    """Integral over phi: 2*pi for a monopole, pi for m>=1 (cos^2 averages to 1/2)."""
    return 2 * np.pi if m == 0 else np.pi


def pressure_cf(gfu_E, gfu_H, mode=0, m=0, scale=1.0, mesh=None, project=False):
    """Lorentz radiation pressure as a coefficient function, in pascals.

    Positive pushes the wall outward (magnetic), negative inward (electric). *scale*
    multiplies the field amplitude, so the pressure scales with its square - pass the
    factor that takes the stored solution to the gradient you want.
    """
    e2, h2 = _field_norms(gfu_E, gfu_H, mode, m, mesh, project)
    return 0.25 * (MU0 * h2 - EPS0 * e2) * (scale ** 2)


def stored_energy(mesh, gfu_E, gfu_H, mode=0, m=0, eps_cf=None):
    r"""Total stored energy of the mode, in joules.

    The time-averaged electric and magnetic energies are equal at resonance, so the
    total is twice either one: :math:`U=\tfrac12\varepsilon_0\int|E|^2\,\mathrm{d}V`.
    """
    e2, _ = _field_norms(gfu_E, gfu_H, mode, m)
    eps = EPS0 if eps_cf is None else EPS0 * eps_cf
    return float(Integrate(0.5 * eps * e2 * y, mesh).real) * _azimuthal_factor(m)


def wall_pressure(mesh, gfu_E, gfu_H, mode=0, m=0, scale=1.0, wall='PEC',
                  points=None):
    """Sample the pressure on the wall.

    Parameters
    ----------
    points : array_like, optional
        ``(n, 2)`` array of ``(z, r)`` in metres to evaluate at. Defaults to the
        mesh's boundary nodes, which are ordered by coordinate rather than along the
        contour - pass ``profile.contour_points(ds)`` when the samples must follow
        the wall, as a line plot needs.

    Returns
    -------
    (points, p) : ndarray, ndarray
        Sample coordinates in metres, and the pressure at each in pascals.
    """
    cf = pressure_cf(gfu_E, gfu_H, mode, m, scale)
    if points is None:
        points = sorted(get_boundary_nodes(mesh, wall))
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    vals = np.asarray(cf(mesh(pts[:, 0], pts[:, 1]))).ravel().real
    return pts, vals


def slater_shift(mesh, gfu_E, gfu_H, normal_displacement, freq_hz,
                 mode=0, m=0, wall='PEC', scale=1.0, U=None):
    r"""Frequency shift from a wall displacement, by Slater's theorem, in hertz.

    Parameters
    ----------
    normal_displacement : CoefficientFunction
        The OUTWARD normal component of the wall displacement, in metres. A
        GridFunction from a mechanical solve is the intended input; an analytic
        expression built from ``ngsolve.x`` / ``ngsolve.y`` also works.
    freq_hz : float
        Unperturbed frequency.
    U : float, optional
        Stored energy. Computed from the fields when omitted; pass it to avoid
        recomputing it across a sweep.

    Returns
    -------
    float
        :math:`\Delta f`, negative when the wall moves so as to lower the frequency,
        which is the usual sense for an unstiffened cavity.

    Notes
    -----
    The mesh is a meridian, so the surface element of the body of revolution carries
    the factor ``r``, and the azimuthal integral contributes ``2*pi`` (``pi`` for
    m>=1).
    """
    p_cf = pressure_cf(gfu_E, gfu_H, mode, m, scale, mesh, project=True)
    if U is None:
        U = stored_energy(mesh, gfu_E, gfu_H, mode, m)
    work = float(Integrate(p_cf * normal_displacement * y, mesh,
                           definedon=mesh.Boundaries(wall)).real)
    return -freq_hz * work * _azimuthal_factor(m) / U


def wall_force(mesh, gfu_E, gfu_H, mode=0, m=0, wall='PEC', scale=1.0):
    """Net force the field exerts on *wall*, as ``(F_z, F_r)`` in newtons.

    A cheap sanity check on the pressure: for a mode symmetric about the mid-plane
    the axial components cancel, so ``F_z`` should come out near zero.
    """
    p_cf = pressure_cf(gfu_E, gfu_H, mode, m, scale, mesh, project=True)
    n = specialcf.normal(2)
    az = _azimuthal_factor(m)
    fz = float(Integrate(p_cf * n[0] * y, mesh,
                         definedon=mesh.Boundaries(wall)).real) * az
    fr = float(Integrate(p_cf * n[1] * y, mesh,
                         definedon=mesh.Boundaries(wall)).real) * az
    return fz, fr


def detuning_coefficient(delta_f_hz, eacc_mv_per_m):
    r"""Lorentz detuning coefficient :math:`K_L`, in Hz/(MV/m)^2."""
    return delta_f_hz / (float(eacc_mv_per_m) ** 2)
