"""A backend-agnostic meridian profile for axisymmetric structures.

A :class:`Profile` is the *blueprint*: an ordered, closed list of boundary
segments (straight lines and circular arcs) in the (z, r) meridian plane, each
tagged with a boundary name (``'AXI'`` the axis, ``'PEC'`` a conducting wall,
``'PMC'`` a symmetry / aperture plane). ``mesh()`` builds it natively with
netgen.occ — exact conic edges, no gmsh round-trip, no ``.geo`` file — and the
solver only ever sees the resulting boundary-tagged mesh.

Coordinates are in metres (the mesh scale the solver expects).

Example (a box)::

    p = (Profile()
         .start(0, 0)
         .line_to(0, 0.1, 'PMC')     # up the left plane
         .line_to(0.2, 0.1, 'PEC')   # along the wall
         .line_to(0.2, 0, 'PMC')     # down the right plane
         .close('AXI'))              # back along the axis
    mesh = p.mesh(maxh=0.02, order=3)
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.special import comb


class Profile:
    def __init__(self, name='profile'):
        self.name = name
        self._pts = []      # ordered boundary points [(z, r), ...]
        self._segs = []     # [{'kind','i0','i1','name', ('mid')}, ...]

    # -- construction -------------------------------------------------------

    def start(self, z, r):
        """Set the starting point of the contour."""
        self._pts = [(float(z), float(r))]
        self._segs = []
        return self

    def line_to(self, z, r, boundary):
        """Straight segment from the current point to (z, r), tagged *boundary*."""
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'line', 'i0': i0, 'i1': i0 + 1, 'name': boundary})
        return self

    def arc_to(self, z, r, through, boundary):
        """Circular arc from the current point to (z, r) passing through the
        point ``through=(z_m, r_m)``, tagged *boundary*."""
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'arc', 'i0': i0, 'i1': i0 + 1,
                           'name': boundary, 'mid': (float(through[0]), float(through[1]))})
        return self

    def circle_arc_to(self, z, r, center, boundary):
        """Circular arc from the current point to (z, r) about ``center``.

        The short sweep is taken (every arc in the supported geometries is < pi).
        Exact: the arc midpoint is placed on the circle and the segment is built
        as a three-point ``ArcOfCircle``.
        """
        p0 = self._pts[-1]
        cz, cr = float(center[0]), float(center[1])
        radius = np.hypot(p0[0] - cz, p0[1] - cr)
        a0 = np.arctan2(p0[1] - cr, p0[0] - cz)
        a1 = np.arctan2(r - cr, z - cz)
        dt = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi     # short sweep, in (-pi, pi]
        am = a0 + dt / 2.0
        mid = (cz + radius * np.cos(am), cr + radius * np.sin(am))
        return self.arc_to(z, r, through=mid, boundary=boundary)

    def ellipse_arc_to(self, z, r, center, semi_z, semi_r, boundary):
        """Exact elliptical arc from the current point to (z, r).

        The arc lies on the ellipse centred at ``center=(zc, rc)`` with
        semi-axis ``semi_z`` along z and ``semi_r`` along r. The short sweep
        between the two endpoints is taken. Built with OCC's exact conic
        (``Ellipse(...).Trim(t0, t1).Edge()``) — no polyline approximation.
        """
        i0 = len(self._pts) - 1
        self._pts.append((float(z), float(r)))
        self._segs.append({'kind': 'ellipse', 'i0': i0, 'i1': i0 + 1, 'name': boundary,
                           'center': (float(center[0]), float(center[1])),
                           'semi_z': float(semi_z), 'semi_r': float(semi_r)})
        return self

    def spline_to(self, poles, boundary, kind='bspline', degree=3):
        """Free-form spline from the current point through ``poles``.

        ``poles`` are the *control* points after the current one; the last is the
        segment's end point. The current point is the first pole, so the curve
        starts and ends on the contour (both curve types are clamped) but need not
        pass through the interior poles — the same convention as gmsh's ``BSpline``
        and ``Bezier``. ``kind`` is ``'bspline'`` or ``'bezier'``.
        """
        kind = kind.lower()
        if kind not in ('bspline', 'bezier'):
            raise ValueError(f"spline kind must be 'bspline' or 'bezier', got {kind!r}")
        poles = [(float(z), float(r)) for z, r in poles]
        if len(poles) < 2:
            raise ValueError('a spline segment needs at least two poles')
        i0 = len(self._pts) - 1
        self._pts.append(poles[-1])
        self._segs.append({'kind': 'spline', 'i0': i0, 'i1': i0 + 1, 'name': boundary,
                           'interior': poles[:-1], 'spline_kind': kind,
                           'degree': int(degree)})
        return self

    # -- spline helpers -----------------------------------------------------

    @classmethod
    def _spline_poles(cls, seg, pts):
        """Full control polygon: start point, interior poles, end point."""
        return [pts[seg['i0']]] + list(seg['interior']) + [pts[seg['i1']]]

    @staticmethod
    def _clamped_uniform_knots(n_poles, degree):
        """Knot vector of a clamped uniform B-spline (what OCC and gmsh both use)."""
        n_internal = n_poles - degree - 1
        internal = list(np.arange(1, n_internal + 1) / (n_internal + 1)) if n_internal > 0 else []
        return np.array([0.0] * (degree + 1) + internal + [1.0] * (degree + 1))

    @classmethod
    def _spline_degree(cls, seg, pts):
        return min(seg['degree'], len(cls._spline_poles(seg, pts)) - 1)

    @staticmethod
    def _insert_knot(knots, poles, degree, u):
        """Boehm's algorithm: insert knot *u* once, leaving the curve unchanged."""
        k = int(np.searchsorted(knots, u, side='right')) - 1
        new = list(poles[:k - degree + 1])
        for i in range(k - degree + 1, k + 1):
            denom = knots[i + degree] - knots[i]
            a = 0.0 if denom == 0 else (u - knots[i]) / denom
            new.append((1.0 - a) * poles[i - 1] + a * poles[i])
        new.extend(poles[k:])
        return np.insert(knots, k + 1, u), np.array(new)

    @classmethod
    def _bspline_to_bezier(cls, poles, degree):
        """Split a clamped uniform B-spline into its exact Bezier segments.

        netgen's ``BSplineCurve`` builds an *unclamped* uniform B-spline, which
        does not start or end at the outer poles, so it cannot be used directly.
        Raising every internal knot to multiplicity ``degree`` (Boehm insertion)
        decomposes the very same curve into Bezier arcs, which netgen represents
        exactly — no approximation, and the endpoints land back on the contour.
        """
        poles = np.asarray(poles, dtype=float)
        knots = cls._clamped_uniform_knots(len(poles), degree)
        for u in sorted({float(k) for k in knots if 0.0 < k < 1.0}):
            while int(np.sum(np.isclose(knots, u))) < degree:
                knots, poles = cls._insert_knot(knots, poles, degree, u)
        n_seg = (len(poles) - 1) // degree
        return [poles[i * degree: i * degree + degree + 1] for i in range(n_seg)]

    @classmethod
    def _spline_points(cls, seg, pts, n=48):
        """Sample points along a spline segment (for boundary matching)."""
        poles = np.asarray(cls._spline_poles(seg, pts), dtype=float)
        u = np.linspace(0.0, 1.0, n)
        if seg['spline_kind'] == 'bezier':
            m = len(poles) - 1
            k = np.arange(m + 1)
            basis = comb(m, k)[None, :] * (u[:, None] ** k[None, :]) * ((1 - u)[:, None] ** (m - k)[None, :])
            return [tuple(p) for p in basis @ poles]
        deg = cls._spline_degree(seg, pts)
        knots = cls._clamped_uniform_knots(len(poles), deg)
        return [tuple(p) for p in BSpline(knots, poles, deg)(u)]

    # -- ellipse helpers ----------------------------------------------------

    @staticmethod
    def _ellipse_frame(semi_z, semi_r):
        """Return (major, minor, xdir) with major >= minor; xdir is the unit
        direction of the major axis in the (z, r) plane."""
        if semi_z >= semi_r:
            return semi_z, semi_r, (1.0, 0.0)
        return semi_r, semi_z, (0.0, 1.0)

    @classmethod
    def _ellipse_param(cls, p, center, major, minor, xdir):
        """Parameter t with P = C + major*cos(t)*xdir + minor*sin(t)*ydir."""
        ux, uy = xdir
        vx, vy = -uy, ux                       # ydir = xdir rotated +90 deg
        dz, dr = p[0] - center[0], p[1] - center[1]
        du = (dz * ux + dr * uy) / major
        dv = (dz * vx + dr * vy) / minor
        return np.arctan2(dv, du)

    @classmethod
    def _ellipse_span(cls, seg, pts):
        """(center, major, minor, xdir, t_lo, t_hi) for an ellipse segment,
        taking the short sweep between its endpoints."""
        c = seg['center']
        major, minor, xdir = cls._ellipse_frame(seg['semi_z'], seg['semi_r'])
        t0 = cls._ellipse_param(pts[seg['i0']], c, major, minor, xdir)
        t1 = cls._ellipse_param(pts[seg['i1']], c, major, minor, xdir)
        dt = t1 - t0
        while dt <= -np.pi:
            dt += 2 * np.pi
        while dt > np.pi:
            dt -= 2 * np.pi
        t_lo, t_hi = (t0, t0 + dt) if dt >= 0 else (t0 + dt, t0)
        return c, major, minor, xdir, t_lo, t_hi

    @classmethod
    def _ellipse_points(cls, seg, pts, n=24):
        """Sample points along an ellipse segment, ordered from ``i0`` to ``i1``."""
        c, major, minor, xdir, t_lo, t_hi = cls._ellipse_span(seg, pts)
        ux, uy = xdir
        vx, vy = -uy, ux
        out = []
        for t in np.linspace(t_lo, t_hi, n):
            ct, st = np.cos(t), np.sin(t)
            out.append((c[0] + major * ct * ux + minor * st * vx,
                        c[1] + major * ct * uy + minor * st * vy))
        # _ellipse_span sorts (t_lo, t_hi), which reverses the sweep when it runs
        # in the decreasing-parameter direction. Order from i0 so that a walk over
        # consecutive segments (contour_points) stays continuous.
        p0 = pts[seg['i0']]
        if (np.hypot(out[0][0] - p0[0], out[0][1] - p0[1])
                > np.hypot(out[-1][0] - p0[0], out[-1][1] - p0[1])):
            out.reverse()
        return out

    def close(self, boundary):
        """Straight segment from the current point back to the start, tagged
        *boundary* (typically the axis)."""
        i0 = len(self._pts) - 1
        self._segs.append({'kind': 'line', 'i0': i0, 'i1': 0, 'name': boundary})
        return self

    # -- contour sampling ---------------------------------------------------

    def _arc_points(self, seg, n):
        """Sample a three-point circular arc segment."""
        p0 = np.asarray(self._pts[seg['i0']], dtype=float)
        p1 = np.asarray(self._pts[seg['i1']], dtype=float)
        pm = np.asarray(seg['mid'], dtype=float)

        # circumcentre of the three points
        ax, ay = p0
        bx, by = pm
        cx, cy = p1
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-18:                       # collinear -> straight
            return [tuple(p0), tuple(p1)]
        ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
              + (cx ** 2 + cy ** 2) * (ay - by)) / d
        uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx)
              + (cx ** 2 + cy ** 2) * (bx - ax)) / d
        centre = np.array([ux, uy])
        radius = np.linalg.norm(p0 - centre)

        a0 = np.arctan2(p0[1] - uy, p0[0] - ux)
        am = np.arctan2(pm[1] - uy, pm[0] - ux)
        a1 = np.arctan2(p1[1] - uy, p1[0] - ux)

        def unwrap(a, ref):
            while a - ref > np.pi:
                a -= 2 * np.pi
            while a - ref < -np.pi:
                a += 2 * np.pi
            return a

        am = unwrap(am, a0)
        a1 = unwrap(a1, am)
        return [(ux + radius * np.cos(t), uy + radius * np.sin(t))
                for t in np.linspace(a0, a1, n)]

    def _segment_length(self, seg):
        """Approximate arclength of a segment, for choosing a sample count."""
        if seg['kind'] == 'line':
            p0 = np.asarray(self._pts[seg['i0']])
            p1 = np.asarray(self._pts[seg['i1']])
            return float(np.linalg.norm(p1 - p0))
        pts = np.asarray(self._segment_points(seg, 33))
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def _segment_points(self, seg, n):
        """Sample *n* points along one segment, endpoints included."""
        if seg['kind'] == 'line':
            p0 = np.asarray(self._pts[seg['i0']], dtype=float)
            p1 = np.asarray(self._pts[seg['i1']], dtype=float)
            return [tuple(p0 + (p1 - p0) * t) for t in np.linspace(0.0, 1.0, max(2, n))]
        if seg['kind'] == 'arc':
            return self._arc_points(seg, max(3, n))
        if seg['kind'] == 'ellipse':
            return self._ellipse_points(seg, self._pts, n=max(3, n))
        if seg['kind'] == 'spline':
            return self._spline_points(seg, self._pts, n=max(3, n))
        raise ValueError(f"unknown segment kind {seg['kind']!r}")

    def contour_points(self, ds, skip=('AXI',)):
        """The meridian wall as an ordered, densified ``[(z, r), ...]`` polyline.

        Curved segments are sampled at roughly *ds* spacing; straight ones keep
        their two endpoints. Segments whose boundary name is in *skip* are left
        out — by default the axis, since external wake codes want the wall only.

        This is the geometry seam for wakefield solvers: a solver takes the
        contour, not a ``.geo`` file. (The old ABCI writer regex-parsed the
        ``.geo`` text and understood only ``Point`` and ``Ellipse``, so it could
        not see a spline wall and had no source at all for a cavity with no
        ``.geo``.)
        """
        if ds <= 0:
            raise ValueError('ds must be positive')
        out = []
        for seg in self._segs:
            if seg['name'] in skip:
                continue
            if seg['kind'] == 'line':
                n = 2
            else:
                n = int(np.ceil(self._segment_length(seg) / ds)) + 1
                n = max(3, min(n, 2000))
            pts = self._segment_points(seg, n)
            if out and np.allclose(out[-1], pts[0], atol=1e-12):
                pts = pts[1:]
            out.extend(tuple(map(float, p)) for p in pts)
        return out

    # -- queries ------------------------------------------------------------

    @property
    def points(self):
        """Ordered boundary points [(z, r), ...] (open — start not repeated)."""
        return list(self._pts)

    def _seg_midpoint(self, seg):
        p0 = self._pts[seg['i0']]
        p1 = self._pts[seg['i1']]
        if seg['kind'] == 'arc':
            return seg['mid']
        if seg['kind'] == 'ellipse':
            pts = self._ellipse_points(seg, self._pts, n=3)
            return pts[1]
        if seg['kind'] == 'spline':
            pts = self._spline_points(seg, self._pts, n=3)
            return pts[1]
        return ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)

    def boundary_names(self):
        return sorted({s['name'] for s in self._segs})

    # -- netgen.occ backend -------------------------------------------------

    def to_occ_face(self):
        """Build a netgen.occ Face from the profile segments (lines / arcs).

        Boundary names are *not* set here: OCCGeometry's shape-healing rebuilds
        edges (and strips names) whenever the profile has collinear adjacent
        segments with different tags — e.g. a pillbox end plane, whose beam
        aperture and metal plate are collinear but PMC vs PEC. Instead the
        boundaries are named on the generated mesh (:meth:`_name_boundaries`),
        which is robust to that healing.
        """
        # Deferred: netgen is an optional heavy dependency. Keep `cavsim2d`
        # importable (and Profile constructible) on installs without it.
        from netgen.occ import (Segment, Wire, Face, Pnt, ArcOfCircle,
                                Ellipse, gp_Ax2d, gp_Pnt2d, gp_Dir2d, BezierCurve)

        if len(self._segs) < 3:
            raise ValueError("A profile needs at least 3 segments to bound a face.")

        edges = []
        for s in self._segs:
            p0 = self._pts[s['i0']]
            p1 = self._pts[s['i1']]
            if s['kind'] == 'arc':
                m = s['mid']
                edges.append(ArcOfCircle(Pnt(p0[0], p0[1], 0),
                                         Pnt(m[0], m[1], 0),
                                         Pnt(p1[0], p1[1], 0)))
            elif s['kind'] == 'ellipse':
                c, major, minor, xdir, t_lo, t_hi = self._ellipse_span(s, self._pts)
                ax = gp_Ax2d(gp_Pnt2d(c[0], c[1]), gp_Dir2d(xdir[0], xdir[1]))
                edges.append(Ellipse(ax, major, minor).Trim(t_lo, t_hi).Edge())
            elif s['kind'] == 'spline':
                poles = self._spline_poles(s, self._pts)
                if s['spline_kind'] == 'bezier':
                    edges.append(BezierCurve([Pnt(z, r, 0) for z, r in poles]))
                else:
                    # exact Bezier decomposition; netgen's BSplineCurve is unclamped
                    for bez in self._bspline_to_bezier(poles, self._spline_degree(s, self._pts)):
                        edges.append(BezierCurve([Pnt(z, r, 0) for z, r in bez]))
            else:
                edges.append(Segment(Pnt(p0[0], p0[1], 0), Pnt(p1[0], p1[1], 0)))
        face = Face(Wire(edges))
        face.name = 'Domain'
        return face

    @staticmethod
    def _point_segment_distance(q, p0, p1):
        qx, qy = q
        ax, ay = p0
        bx, by = p1
        abx, aby = bx - ax, by - ay
        denom = abx * abx + aby * aby
        t = 0.0 if denom == 0 else ((qx - ax) * abx + (qy - ay) * aby) / denom
        t = max(0.0, min(1.0, t))
        return np.hypot(qx - (ax + t * abx), qy - (ay + t * aby))

    def _seg_distance(self, q, s):
        """Distance from point *q* to segment *s* (exact for lines; arcs and
        ellipse arcs are sampled into a fine polyline)."""
        p0, p1 = self._pts[s['i0']], self._pts[s['i1']]
        if s['kind'] == 'line':
            return self._point_segment_distance(q, p0, p1)
        if s['kind'] == 'arc':
            m = s['mid']
            return min(self._point_segment_distance(q, p0, m),
                       self._point_segment_distance(q, m, p1))
        if s['kind'] == 'spline':
            pts = self._spline_points(s, self._pts, n=64)
        else:
            pts = self._ellipse_points(s, self._pts, n=32)
        return min(self._point_segment_distance(q, pts[i], pts[i + 1])
                   for i in range(len(pts) - 1))

    def _segment_of(self, q):
        """Name of the profile segment that point *q* lies on (nearest)."""
        best_name, best_d = None, float('inf')
        for s in self._segs:
            d = self._seg_distance(q, s)
            if d < best_d:
                best_d, best_name = d, s['name']
        return best_name

    def _name_boundaries(self, mesh):
        """Assign boundary names on *mesh* by matching each boundary region to
        the profile segment its elements lie on (works regardless of whether
        OCC preserved the edge names)."""
        # Deferred: ngsolve is an optional heavy dependency.
        from ngsolve import BND
        index_name = {}
        for el in mesh.Elements(BND):
            idx = el.index
            if idx in index_name:
                continue
            vs = [mesh[v].point for v in el.vertices]
            q = (sum(p[0] for p in vs) / len(vs), sum(p[1] for p in vs) / len(vs))
            index_name[idx] = self._segment_of(q)
        for idx, name in index_name.items():
            mesh.ngmesh.SetBCName(idx, name)

    @classmethod
    def _spline_speed(cls, seg, pts, u):
        """|dC/du| of a spline segment at parameters ``u``."""
        poles = np.asarray(cls._spline_poles(seg, pts), dtype=float)
        if seg['spline_kind'] == 'bezier':
            m = len(poles) - 1
            dpoles = m * np.diff(poles, axis=0)          # derivative is a degree m-1 Bezier
            k = np.arange(m)
            basis = comb(m - 1, k)[None, :] * (u[:, None] ** k[None, :]) \
                * ((1 - u)[:, None] ** (m - 1 - k)[None, :])
            d = basis @ dpoles
        else:
            deg = cls._spline_degree(seg, pts)
            knots = cls._clamped_uniform_knots(len(poles), deg)
            d = BSpline(knots, poles, deg).derivative()(u)
        return np.linalg.norm(d, axis=1)

    def _stationary_corners(self, rtol=1e-6):
        """Contour points where the tangent vanishes (``|dC/du| = 0``).

        Only spline segments can have these: a control polygon that reverses on
        itself — e.g. the iris of a multicell B-spline built by repeating the
        polygon — gives a vanishing tangent exactly at a knot. netgen's high-order
        curving fails on small elements there.
        """
        corners = []
        for s in self._segs:
            if s['kind'] != 'spline':
                continue
            u = np.linspace(0.0, 1.0, 1025)
            speed = self._spline_speed(s, self._pts, u)
            scale = speed.max()
            if scale <= 0:
                continue
            interior = np.flatnonzero(speed[1:-1] < rtol * scale) + 1
            pts = np.asarray(self._spline_points(s, self._pts, n=1025))
            for i in interior:
                p = tuple(pts[i])
                if not any(np.hypot(p[0] - c[0], p[1] - c[1]) < 1e-9 for c in corners):
                    corners.append(p)
        return corners

    def mesh(self, maxh, order=1):
        """Return a boundary-tagged NGSolve mesh of the profile."""
        # Deferred: netgen/ngsolve are optional heavy dependencies.
        from netgen.occ import OCCGeometry
        from ngsolve import Mesh
        mesh = Mesh(OCCGeometry(self.to_occ_face(), dim=2).GenerateMesh(maxh=maxh))
        self._name_boundaries(mesh)
        if order and order > 1:
            try:
                mesh.Curve(order)
            except Exception as exc:
                corners = self._stationary_corners()
                if not corners:
                    raise
                where = ', '.join('(%.6g, %.6g)' % c for c in corners[:3])
                raise RuntimeError(
                    f"netgen could not build the order-{order} curved mesh at maxh={maxh:g} "
                    f"for profile {self.name!r}.\n"
                    f"The contour has {len(corners)} stationary corner(s) - points where the "
                    f"spline's tangent vanishes — near {where}.\n"
                    "This happens when a control polygon reverses on itself, e.g. the iris of a "
                    "multicell B-spline built by repeating the polygon. The corner is real "
                    "geometry (gmsh only survives it by rounding it off), but netgen's high-order "
                    "curving fails on small elements there.\n"
                    "Workarounds: use kind='Bezier' (one curve per cell, no stationary corner), "
                    "coarsen maxh, or lower the mesh order."
                ) from exc
        return mesh


def mesh_from_profile(profile, maxh, order=1):
    """Convenience wrapper: mesh a :class:`Profile`."""
    return profile.mesh(maxh=maxh, order=order)
