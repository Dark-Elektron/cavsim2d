"""Tangency and alpha-angle solvers for elliptical cell contours."""
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, fsolve, minimize_scalar


def update_alpha(cell, cell_parameterisation='simplecell'):
    """
    Update geometry json file variables to include the value of alpha

    Parameters
    ----------
    cell:
        Cavity geometry parameters

    Returns
    -------
    List of cavity geometry parameters

    """
    A, B, a, b, Ri, L, Req = cell[:7]
    alpha = calculate_alpha(A, B, a, b, Ri, L, Req, 0)
    if cell_parameterisation == 'simplecell':
        cell = [A, B, a, b, Ri, L, Req, alpha[0]]
    elif cell_parameterisation == 'flattop':
        cell = [A, B, a, b, Ri, L, Req, cell[7], alpha[0]]

    return np.array(cell)


def calculate_alpha(A, B, a, b, Ri, L, Req, L_bp):
    """
    Calculates the largest angle the tangent line of two ellipses makes with the horizontal axis

    Parameters
    ----------
    A: float
    B: float
    a: float
    b: float
    Ri: float
    L: float
    Req: float
    L_bp: float

    Returns
    -------
    alpha: float
        Largest angle the tangent line of two ellipses makes with the horizontal axis
    error_msg: int
        State of the iteration, failed or successful. Refer to

    """

    df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp)
    x1, y1, x2, y2 = df[0]
    error_msg = df[-2]

    alpha = 180 - np.arctan2(y2 - y1, (x2 - x1)) * 180 / np.pi
    return alpha, error_msg


def tangent_coords(A, B, a, b, Ri, L, Req, L_bp, lft=0, tangent_check=False):
    """
    Tangent points of the straight wall joining the iris and equator ellipses.
    See :func:`wall_tangent` for the method.

    Parameters
    ----------
    A: float
        Equator ellipse dimension
    B: float
        Equator ellipse dimension
    a: float
        Iris ellipse dimension
    b: float
        Iris ellipse dimension
    Ri: float
        Iris radius
    L: float
        Cavity half cell length
    Req: float
        Cavity equator radius
    L_bp: float
        Cavity beampipe length
    tangent_check: bool
        If set to True, the calculated tangent line as well as the ellipses are plotted and shown

    Returns
    -------
    df: tuple
        ``(x, infodict, ier, mesg)``, the shape of ``scipy.optimize.fsolve``'s
        full output: ``x = [x1, y1, x2, y2]`` (iris then equator tangent point,
        shifted by *L_bp*) and ``ier == 1`` on success. On failure ``ier`` is 5
        and ``x`` is a best-effort estimate.
    """
    data = ([0 + L_bp, Ri + b, L + L_bp, Req - B], [a, b, A, B])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])

    # The returned tuple keeps fsolve's full_output shape, (x, infodict, ier, mesg),
    # because every caller reads df[0] for the points and df[-2] == 1 for success.
    pts = wall_tangent(A, B, a, b, Ri, L, Req)
    if pts is not None:
        df = (pts + np.array([L_bp, 0.0, L_bp, 0.0]), {}, 1, 'The solution converged.')
    else:
        # No tangent exists. Callers that pass ignore_degenerate still read the
        # points, so hand them the old Newton estimate, flagged as a failure.
        df = (_legacy_fsolve(data, a, b, A, B, Ri, L, Req, L_bp), {}, 5,
              'The iris and equator ellipses overlap: no tangent line joins them.')

    x1, y1, x2, y2 = df[0]
    # alpha = 180 - np.arctan2(y2 - y1, (x2 - x1)) * 180 / np.pi

    if tangent_check:
        shift_x = -L - lft
        h, k, p, q = data[0]
        a, b, A, B = data[1]
        el_ab = Ellipse((shift_x + h, k), 2 * a, 2 * b, alpha=0.5)
        el_AB = Ellipse((shift_x + p, q), 2 * A, 2 * B, alpha=0.5)

        ax = plt.gca()
        ax.add_artist(el_ab)
        ax.add_artist(el_AB)

        x1, y1, x2, y2 = df[0]
        ax.plot([shift_x + x1, shift_x + x2], [y1, y2], label=fr'{df[-2]}:: {df[-1]}')
        ax.legend()

    return df


def wall_tangent(A, B, a, b, Ri, L, Req):
    """Tangent points ``[x1, y1, x2, y2]`` of a half-cell wall, iris plane at ``z = 0``.

    Returns ``None`` when the iris and equator ellipses overlap, so no tangent line
    can join them.

    The wall runs from the iris ellipse (metal on one side of the line) to the
    equator ellipse (vacuum on the other), so it is an *internal* common tangent.
    Writing the line as ``n . X = c`` with unit normal ``n = (cos t, sin t)``
    pointing from metal to vacuum, it touches the iris ellipse where ``c`` equals
    that ellipse's support value and the equator ellipse where ``c`` equals the
    equator's lowest value along ``n``. Equating the two leaves one scalar
    equation in ``t``::

        g(t) = n . (C_eq - C_iris) - |(A n_z, B n_r)| - |(a n_z, b n_r)| = 0

    Two disjoint ellipses have exactly two internal tangents, one on each side of
    the maximum of ``g``. The wall is the one on the ``-pi/2`` side, since the
    iris arc leaves the iris bottom and turns toward the equator, so that root is
    bracketed and solved with Brent's method.

    This replaces a four-unknown Newton solve whose residuals divided by
    ``(y1 - k)`` and ``(x2 - p)``. Those vanish for an exactly vertical wall
    (``a + A == L``), which never converged. Newton could also converge to the
    other internal tangent, or to a point that is not a tangent at all when the
    ellipses overlap, and still report success. That gave a broken contour with
    no error.
    """
    h, k, p, q = 0.0, Ri + b, L, Req - B
    d_z, d_r = p - h, q - k

    def g(t):
        nz, nr = np.cos(t), np.sin(t)
        return nz * d_z + nr * d_r - np.hypot(A * nz, B * nr) - np.hypot(a * nz, b * nr)

    lo = -np.pi / 2
    peak = minimize_scalar(lambda t: -g(t), bounds=(lo, np.pi / 2), method='bounded',
                           options={'xatol': 1e-14})
    t_max = float(peak.x)
    if not g(t_max) > 0:
        return None
    t = brentq(g, lo, t_max, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500)

    nz, nr = np.cos(t), np.sin(t)
    s_iris = np.hypot(a * nz, b * nr)
    s_eq = np.hypot(A * nz, B * nr)
    return np.array([h + a * a * nz / s_iris, k + b * b * nr / s_iris,
                     p - A * A * nz / s_eq, q - B * B * nr / s_eq])


def _legacy_fsolve(data, a, b, A, B, Ri, L, Req, L_bp):
    """The original Newton estimate of the tangent points, kept only as the
    best-effort points returned alongside a failure flag."""
    checks = {"non-reentrant": [[0.5, -0.5], [0.75, -0.25], [0.25, -0.75], [0.9, -0.1]],
              "reentrant": [[1.1, -1.1], [1.9, -1.9], [1.5, -1.5], [1.75, -1.75]]}
    guesses = checks['reentrant'] if a + A > L else checks['non-reentrant']
    df = None
    for f_b, f_B in guesses:
        df = fsolve(ellipse_tangent,
                    np.array([a + L_bp, Ri + f_b * b, L - A + L_bp, Req + f_B * B]),
                    args=data, fprime=jac, xtol=1.49012e-12, full_output=True)
        if df[-2] == 1:
            break
    return df[0]


def ellipse_tangent(z, *data):
    """
    Calculates the coordinates of the tangent line that connects two ellipses

    .. _ellipse tangent:

    .. figure:: ../images/ellipse_tangent.png
       :alt: ellipse tangent
       :align: center
       :width: 500px

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
        See :numref:`ellipse tangent`
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``


    Returns
    -------
    list of four non-linear functions

    Note
    -----
    The four returned non-linear functions are

    .. math::

       f_1 = \\frac{A^2b^2(x_1 - h)(y_2-q)}{a^2B^2(x_2-p)(y_1-k)} - 1

       f_2 = \\frac{(x_1 - h)^2}{a^2} + \\frac{(y_1-k)^2}{b^2} - 1

       f_3 = \\frac{(x_2 - p)^2}{A^2} + \\frac{(y_2-q)^2}{B^2} - 1

       f_4 = \\frac{-b^2(x_1-x_2)(x_1-h)}{a^2(y_1-y_2)(y_1-k)} - 1
    """

    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    return f1, f2, f3, f4


def jac(z, *data):
    """
    Computes the Jacobian of the non-linear system of ellipse tangent equations

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
        See :numref:`ellipse tangent`
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``

    Returns
    -------
    J: array like
        Array of the Jacobian

    """
    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    # f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    # f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    # f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    # f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    df1_dx1 = A ** 2 * b ** 2 * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))
    df1_dy1 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k) ** 2)
    df1_dx2 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) ** 2 * (y1 - k))
    df1_dy2 = A ** 2 * b ** 2 * (x1 - h) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))

    df2_dx1 = 2 * (x1 - h) / a ** 2
    df2_dy1 = 2 * (y1 - k) / b ** 2
    df2_dx2 = 0
    df2_dy2 = 0

    df3_dx1 = 0
    df3_dy1 = 0
    df3_dx2 = 2 * (x2 - p) / A ** 2
    df3_dy2 = 2 * (y2 - q) / B ** 2

    df4_dx1 = -b ** 2 * ((x1 - x2) + (x1 - h)) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy1 = -b ** 2 * (x1 - x2) * (x1 - h) * ((y1 - y2) + (y1 - k)) / (a ** 2 * ((y1 - y2) * (y1 - k)) ** 2)
    df4_dx2 = b ** 2 * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy2 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) ** 2 * (y1 - k))

    J = [[df1_dx1, df1_dy1, df1_dx2, df1_dy2],
         [df2_dx1, df2_dy1, df2_dx2, df2_dy2],
         [df3_dx1, df3_dy1, df3_dx2, df3_dy2],
         [df4_dx1, df4_dy1, df4_dx2, df4_dy2]]

    return J


#: Lengths below this (metres) count as zero when rounding a wall: a flat that
#: short is treated as absent rather than emitted as a degenerate edge.
CORNER_TOL = 1e-12


def inscribed_corner(vertex, prev_pt, next_pt, radius, tol=CORNER_TOL):
    """The arc of radius *radius* that rounds the corner at *vertex*.

    Returns ``(t_in, t_out, centre)`` — the two tangent points and the arc
    centre, as numpy arrays — or ``None`` when the corner is straight, the
    radius is zero, or an adjacent edge has no length.

    For a wedge of angle ``phi`` between the incoming and outgoing walls the
    tangent points sit ``radius / tan(phi / 2)`` back from the vertex along each
    edge, and the centre is on the bisector at ``radius / sin(phi / 2)``. That
    is one formula for every corner: a right angle gives an offset of exactly
    ``radius``, and a leaning wall changes ``phi``, not the method.
    """
    v = np.asarray(vertex, dtype=float)
    u_in = v - np.asarray(prev_pt, dtype=float)
    u_out = np.asarray(next_pt, dtype=float) - v
    n_in, n_out = np.linalg.norm(u_in), np.linalg.norm(u_out)
    if radius <= 0 or n_in < tol or n_out < tol:
        return None
    u_in, u_out = u_in / n_in, u_out / n_out
    phi = np.arccos(float(np.clip(np.dot(-u_in, u_out), -1.0, 1.0)))
    if phi > np.pi - 1e-9:                      # collinear: nothing to round
        return None
    bisector = u_out - u_in
    norm = np.linalg.norm(bisector)
    if norm < tol:
        return None
    d = radius / np.tan(phi / 2.0)
    centre = v + (radius / np.sin(phi / 2.0)) * (bisector / norm)
    return v - d * u_in, v + d * u_out, centre


def corner_offset(vertex, prev_pt, next_pt, radius):
    """How far back from *vertex* a corner of *radius* reaches, along each edge.

    This is the quantity a feasibility check compares against the available
    flat, so it is exposed separately from :func:`inscribed_corner`: a model can
    reject an impossible radius analytically, before anything is built.
    """
    got = inscribed_corner(vertex, prev_pt, next_pt, radius)
    if got is None:
        return 0.0
    return float(np.linalg.norm(got[0] - np.asarray(vertex, dtype=float)))


def emit_rounded_wall(prof, vertices, boundary, tol=CORNER_TOL):
    """Draw *vertices* onto *prof*, rounding each one by its own radius.

    ``vertices`` is ``[(z, r, radius), ...]`` in metres, walked in order; the
    first and last are endpoints and their radius is ignored. A corner whose
    radius is zero stays sharp, and a flat that the rounding consumes entirely
    is skipped rather than emitted as a zero-length edge — which is the nominal
    case for a deep convolution, not an exotic one.

    The current point must already be ``vertices[0]``.
    """
    pts = [(float(v[0]), float(v[1])) for v in vertices]
    cursor = np.asarray(pts[0], dtype=float)
    for i in range(1, len(vertices) - 1):
        corner = inscribed_corner(pts[i], pts[i - 1], pts[i + 1], float(vertices[i][2]), tol)
        if corner is None:
            # A corner with no radius, or one whose adjacent run has zero length
            # (a taper with no straight section). Only emit when the point
            # actually moves, or the wire picks up a degenerate edge.
            here = np.asarray(pts[i], dtype=float)
            if np.linalg.norm(here - cursor) > tol:
                prof.line_to(pts[i][0], pts[i][1], boundary)
                cursor = here
            continue
        t_in, t_out, centre = corner
        if np.linalg.norm(t_in - cursor) > tol:
            prof.line_to(t_in[0], t_in[1], boundary)
        prof.circle_arc_to(t_out[0], t_out[1], centre, boundary)
        cursor = t_out
    end = np.asarray(pts[-1], dtype=float)
    if np.linalg.norm(end - cursor) > tol:
        prof.line_to(end[0], end[1], boundary)
    return prof
