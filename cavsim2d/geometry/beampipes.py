"""Beam pipes on a meridian contour.

ABCI refuses to run a structure that does not have a beam pipe at *each* end:

    *** STOP *** THE BEAM PIPES AT BOTH ENDS ARE TOO SHORT.
    THEY MUST HAVE AT LEAST 5 MESH LENGTH.

That is a requirement of the wake code, not of the cavity. Several geometries
(RF gun, spline cavity, a pillbox with ``L_bp = 0``, any cavity with
``beampipe='none'``) have no pipe in their parameterisation, so one is added
before the contour is handed to the solver.

Coordinates are ``[(z, r), ...]`` in metres, ordered along the wall from the
left axis point to the right axis point (the axis segment itself excluded).
"""
import numpy as np

#: Default pipe length, as a multiple of the device's own axial length.
DEFAULT_PIPE_FACTOR = 3.0


def axial_length(points):
    """Axial extent of a contour."""
    zs = [z for z, _ in points]
    return max(zs) - min(zs)


def _pipe_run(points, indices, tol):
    """Axial length of the constant-radius run starting at ``indices[0]``."""
    z0, r0 = points[indices[0]]
    z_far = z0
    for i in indices[1:]:
        z, r = points[i]
        if abs(r - r0) > tol:
            break
        z_far = z
    return abs(z_far - z0)


def beampipe_lengths(points, tol=1e-9):
    """Existing pipe length at each end, ``(left, right)``.

    A "pipe" is a constant-radius run adjacent to the aperture. The first and
    last contour points sit on the axis; the aperture points are their
    neighbours.
    """
    n = len(points)
    if n < 4:
        return 0.0, 0.0
    left = _pipe_run(points, range(1, n), tol)
    right = _pipe_run(points, range(n - 2, -1, -1), tol)
    return left, right


def ensure_beampipes(points, min_length, pipe_length=None, tol=1e-9):
    """Return ``(points, added_left, added_right)`` with a pipe at each end.

    An end whose existing pipe is shorter than *min_length* is extended to
    *pipe_length* (default: :data:`DEFAULT_PIPE_FACTOR` times the contour's axial
    length). The aperture radius, and the rest of the wall, are untouched: the
    axis point moves outward and a new aperture point is inserted at the new z,
    which turns the old aperture point into the inner end of a straight pipe.
    """
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) < 4:
        raise ValueError('a contour needs at least four points to carry beam pipes')

    if pipe_length is None:
        pipe_length = DEFAULT_PIPE_FACTOR * axial_length(pts)
    if pipe_length <= 0:
        raise ValueError('pipe_length must be positive')

    left, right = beampipe_lengths(pts, tol)
    added_left = added_right = 0.0

    if left < min_length:
        added_left = pipe_length - left
        z_ap, r_ap = pts[1]
        # drop the old axis point; the axis now starts further out
        pts = [(z_ap - added_left, 0.0), (z_ap - added_left, r_ap)] + pts[1:]

    if right < min_length:
        added_right = pipe_length - right
        z_ap, r_ap = pts[-2]
        pts = pts[:-1] + [(z_ap + added_right, r_ap), (z_ap + added_right, 0.0)]

    return pts, added_left, added_right


def abci_contour(profile, ds, min_pipe_length, pipe_length=None):
    """Densified wall contour for ABCI, with beam pipes guaranteed at both ends.

    Returns ``(points, added_left, added_right)`` in metres. The axis segment is
    excluded — ABCI wants the wall from one axis point to the other.

    Note this is a *polyline*: use it only for walls with no conic arcs. ABCI's
    mesher produces NaN wake potentials when an arc that meets a beam pipe
    tangentially is replaced by many near-collinear points. Prefer
    :func:`abci_shape`, which keeps ABCI's native arc primitive.
    """
    points = profile.contour_points(ds, skip=('AXI',))
    if len(points) >= 2 and points[0][0] > points[-1][0]:
        points = points[::-1]           # always left-to-right in z
    return ensure_beampipes(points, min_pipe_length, pipe_length)


def _arc_centre(p0, pm, p1):
    """Circumcentre of three points, or None if they are collinear."""
    (ax, ay), (bx, by), (cx, cy) = p0, pm, p1
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-18:
        return None
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
          + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
          + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    return ux, uy


def abci_shape(profile, ds, min_pipe_length, pipe_length=None):
    """ABCI wall description: points plus native arcs, with beam pipes guaranteed.

    Returns ``(items, added_left, added_right)`` where each item is either
    ``('point', z, r)`` or ``('arc', (z_c, r_c), (z_e, r_e))`` — an arc from the
    previous point, given by its centre and end point, which is what ABCI's
    ``-3.`` marker introduces.

    Straight segments contribute their endpoint. Circular and elliptical arcs
    become one ABCI arc each (an ellipse is described by its own centre — the
    historical encoding; see the note in :mod:`cavsim2d.models.base`). Splines
    have no ABCI primitive and are densified to *ds*.

    Densifying an arc instead would be more faithful, but ABCI's mesher yields
    NaN wake potentials when an arc that meets a beam pipe tangentially (as an
    elliptical cavity's iris does) is fed to it as a polyline.
    """
    segs = [s for s in profile._segs if s['name'] not in ('AXI',)]
    if not segs:
        raise ValueError('profile has no wall segments')

    pts = profile.points
    items = [('point',) + tuple(map(float, pts[segs[0]['i0']]))]

    for seg in segs:
        end = tuple(map(float, pts[seg['i1']]))
        if seg['kind'] == 'line':
            items.append(('point',) + end)
        elif seg['kind'] == 'ellipse':
            items.append(('arc', tuple(map(float, seg['center'])), end))
        elif seg['kind'] == 'arc':
            centre = _arc_centre(pts[seg['i0']], seg['mid'], pts[seg['i1']])
            if centre is None:
                items.append(('point',) + end)
            else:
                items.append(('arc', centre, end))
        elif seg['kind'] == 'spline':
            n = max(3, int(np.ceil(profile._segment_length(seg) / ds)) + 1)
            for p in profile._segment_points(seg, n)[1:]:
                items.append(('point',) + tuple(map(float, p)))
        else:
            raise ValueError(f"unknown segment kind {seg['kind']!r}")

    # Beam pipes act on the wall's leading/trailing vertices. An arc contributes
    # its end point as a vertex; with beampipe='none' the aperture point *is* an
    # arc's end, not a separate point item.
    def vertex(item):
        return (item[1], item[2]) if item[0] == 'point' else item[2]

    vertices = [vertex(it) for it in items]
    left, right = beampipe_lengths(vertices)
    if pipe_length is None:
        zs = [v[0] for v in vertices]
        pipe_length = DEFAULT_PIPE_FACTOR * (max(zs) - min(zs))

    added_left = added_right = 0.0
    if left < min_pipe_length:
        added_left = pipe_length - left
        z_ap, r_ap = vertices[1]
        items = [('point', z_ap - added_left, 0.0),
                 ('point', z_ap - added_left, r_ap)] + items[1:]
    if right < min_pipe_length:
        added_right = pipe_length - right
        z_ap, r_ap = vertices[-2]
        items = items[:-1] + [('point', z_ap + added_right, r_ap),
                              ('point', z_ap + added_right, 0.0)]

    return items, added_left, added_right
