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


class MaterialRegion:
    """An axisymmetric sub-domain of a :class:`Profile` carrying its own material.

    A region is a rectangular ring in the (z, r) meridian plane — i.e. an annular
    cylinder in 3D — which covers the shapes dielectric loading actually takes:
    ceramic windows, dielectric-loaded accelerating liners, absorber rings.
    Coordinates are in metres, like the rest of the profile.

    The region is *clipped* to the profile it is added to, so it may be specified
    generously (e.g. ``r=(0.09, 1.0)`` for "everything outside r = 90 mm"); only
    the part inside the cavity becomes a sub-domain.

    Arbitrary region outlines are the natural extension: give this class a
    segment chain instead of a rectangle and implement :meth:`to_occ_face` /
    :meth:`boundary_distance` from it. Nothing outside this class assumes the
    rectangle.
    """

    def __init__(self, material, z, r, color=(1.0, 1.0, 0.0)):
        self.material = str(material)
        # OCC face colour (r, g, b), default yellow — carried onto the meshed
        # face so the region shows up when the geometry/mesh is drawn.
        self.color = tuple(float(c) for c in color)
        if len(z) != 2 or len(r) != 2:
            raise ValueError("region 'z' and 'r' must each be a (lo, hi) pair, "
                             f"got z={z!r}, r={r!r}.")
        self.z = (float(min(z)), float(max(z)))
        self.r = (float(min(r)), float(max(r)))
        if self.z[0] == self.z[1] or self.r[0] == self.r[1]:
            raise ValueError(
                f"region {self.material!r} has zero extent (z={self.z}, r={self.r}); "
                "a material region needs a non-degenerate z and r span, in metres.")
        if self.r[0] < 0:
            raise ValueError(f"region {self.material!r} has r < 0 ({self.r}); the "
                             "meridian plane is r >= 0.")

    def __repr__(self):
        return f"MaterialRegion({self.material!r}, z={self.z}, r={self.r})"

    def edges(self):
        """The region outline as a list of ``((z0, r0), (z1, r1))`` line segments."""
        (z0, z1), (r0, r1) = self.z, self.r
        c = [(z0, r0), (z1, r0), (z1, r1), (z0, r1)]
        return [(c[i], c[(i + 1) % 4]) for i in range(4)]

    def boundary_distance(self, q):
        """Distance from point *q* to the region's outline."""
        return min(Profile._point_segment_distance(q, a, b) for a, b in self.edges())

    def to_occ_face(self):
        """A netgen.occ Face covering the region (before clipping to the profile)."""
        # Deferred: netgen is an optional heavy dependency.
        from netgen.occ import WorkPlane
        return (WorkPlane().MoveTo(self.z[0], self.r[0])
                .Rectangle(self.z[1] - self.z[0], self.r[1] - self.r[0]).Face())


class Profile:
    #: Material name given to whatever is left after every region is cut out.
    #: Matches the ``Physical Surface("Domain")`` the gmsh writers emit.
    default_material = 'Domain'

    def __init__(self, name='profile'):
        self.name = name
        self._pts = []      # ordered boundary points [(z, r), ...]
        self._segs = []     # [{'kind','i0','i1','name', ('mid')}, ...]
        self._regions = []  # [MaterialRegion, ...] — sub-domains, see add_region

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
        """Every boundary name the meshed profile will carry — the contour tags
        plus one ``IF_<material>`` interface per material region."""
        return sorted({s['name'] for s in self._segs} | set(self.interface_names()))

    # -- material regions ---------------------------------------------------

    def add_region(self, material, *, z, r, color=(1.0, 1.0, 0.0)):
        """Add a material sub-domain, an axisymmetric rectangular ring.

        ``material`` names the sub-domain — it becomes the mesh material name the
        solver looks up the permittivity by. ``z=(z0, z1)`` and ``r=(r0, r1)`` are
        its extent in **metres**, clipped to the profile, so a generous span is
        fine::

            p.add_region('ceramic', z=(0.10, 0.15), r=(0.0, 0.10))

        ``color`` is the region's ``(r, g, b)`` face colour (default yellow),
        carried onto the meshed face so the region is visible when the geometry or
        mesh is drawn (e.g. ``ngsolve.Draw``); it does not affect the solve.

        Regions are cut **in order**: each is clipped against what earlier
        regions have already claimed, so where two overlap the one added first
        wins. A region left with nothing at all — fully outside the profile, or
        fully swallowed by an earlier one — raises.
        """
        if material == self.default_material:
            raise ValueError(
                f"region name {material!r} is the background material name; pick "
                "another name for the region, or change Profile.default_material.")
        if any(reg.material == material for reg in self._regions):
            raise ValueError(f"a region named {material!r} was already added to "
                             f"profile {self.name!r}.")
        self._regions.append(MaterialRegion(material, z, r, color=color))
        return self

    def regions(self):
        """The material regions, in the order they were added."""
        return list(self._regions)

    def materials(self):
        """Every material name the meshed profile will carry, background first."""
        return [self.default_material] + [reg.material for reg in self._regions]

    def interface_names(self):
        """The boundary name given to each region's internal interface."""
        return [f'IF_{reg.material}' for reg in self._regions]

    # -- netgen.occ backend -------------------------------------------------

    def to_occ_face(self):
        """Build a netgen.occ Face from the profile segments (lines / arcs).

        Boundary names are *not* set here: OCCGeometry's shape-healing rebuilds
        edges (and strips names) whenever the profile has collinear adjacent
        segments with different tags — e.g. a pillbox end plane, whose beam
        aperture and metal plate are collinear but PMC vs PEC. Instead the
        boundaries are named on the generated mesh (:meth:`_name_boundaries`),
        which is robust to that healing. (Per-edge ``maxh`` hints are stripped by
        the same healing, so local refinement is done with ``RestrictH`` size
        points in :meth:`mesh`, not on the OCC edges.)
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
        face.name = self.default_material
        return face

    def _signed_area(self):
        """Signed area of the contour polygon: > 0 counter-clockwise, < 0 clockwise.

        Only the *sign* is used (to orient the OCC face), so the straight-line
        polygon through the segment endpoints is accurate enough — no closed
        contour flips sign because of an arc's bulge.
        """
        p = np.asarray(self._pts, dtype=float)
        if len(p) < 3:
            return 0.0
        z, r = p[:, 0], p[:, 1]
        return 0.5 * float(np.sum(z * np.roll(r, -1) - np.roll(z, -1) * r))

    def _assert_conformal(self, mesh):
        """Raise if the meshed regions are not conformally joined.

        A non-conformal interface carries two coincident nodes, so the HCurl space
        gets independent DOFs on each side, tangential E is free to jump, and the
        sub-domains stop being electrically connected. The solve still succeeds —
        it just returns modes of the disconnected pieces. That is a silent,
        physically plausible wrong answer, so it is checked rather than trusted.
        """
        pts = np.array([mesh[v].point for v in mesh.vertices], dtype=float)
        if not len(pts):
            return
        _, counts = np.unique(np.round(pts, 12), axis=0, return_counts=True)
        n_dup = int((counts > 1).sum())
        if n_dup:
            raise RuntimeError(
                f"profile {self.name!r}: the material regions "
                f"{[r.material for r in self._regions]} did not mesh conformally — "
                f"{n_dup} interface node(s) are duplicated, so the sub-domains are "
                "electrically disconnected and the eigenmode solve would return "
                "spurious modes of the isolated pieces. This usually means OCC could "
                "not identify the shared edges when gluing; check that the region "
                "bounds do not almost-but-not-quite coincide with a contour segment.")

    def to_occ_shape(self):
        """The meshable netgen.occ shape: the profile face, split into one named
        face per material region plus the background.

        With no regions this is exactly :meth:`to_occ_face` — one face named
        ``default_material`` — so the vacuum path is unchanged.

        With regions, each is intersected with what is left of the profile, named
        after its material, and the remainder keeps ``default_material``. The
        pieces are ``Glue``-d so netgen meshes them **conformally**: the interface
        carries one set of nodes, which is what makes the HCurl space enforce
        tangential-E continuity across it (the physical dielectric interface
        condition) with no special treatment in the solver.
        """
        # Deferred: netgen is an optional heavy dependency.
        from netgen.occ import Glue

        face = self.to_occ_face()
        if not self._regions:
            return face

        # Orientation matters, and silently. A face built from a CLOCKWISE wire
        # produces boolean results whose shared edges ``Glue`` cannot identify:
        # the mesh then carries two coincident nodes on every interface, the
        # sub-domains are electrically disconnected, and the solve returns
        # plausible-looking spurious modes (an isolated sub-domain resonating
        # against its own natural-BC walls). Profiles are conventionally traced
        # clockwise — up the end plane, along the wall, back along the axis — so
        # this flip is the normal case, not the exception. Only done when there
        # are regions to glue, so the single-domain path is untouched.
        if self._signed_area() < 0:
            face = face.Reversed()
            face.name = self.default_material

        rest, pieces = face, []
        for reg in self._regions:
            rface = reg.to_occ_face()
            piece = rest * rface                       # clip to what is still free
            if not len(piece.faces):
                raise ValueError(
                    f"material region {reg.material!r} (z={reg.z}, r={reg.r}) does not "
                    f"overlap profile {self.name!r} — check the units (metres) and that "
                    "the region is not already fully covered by an earlier region.")
            piece.name = reg.material
            # Colour the region's faces (default yellow) so a dielectric is visible
            # when the geometry/mesh is drawn (e.g. ngsolve Draw). Cosmetic only —
            # the solve reads the material name, not the colour — so never let a
            # colour-assignment quirk in some netgen build break meshing.
            try:
                piece.faces.col = reg.color
            except Exception:
                pass
            pieces.append(piece)
            rest = rest - rface
        if len(rest.faces):
            rest.name = self.default_material
            pieces.append(rest)
        return Glue(pieces)

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

    def _nearest_segment(self, q):
        """``(name, distance)`` of the profile segment nearest to point *q*."""
        best_name, best_d = None, float('inf')
        for s in self._segs:
            d = self._seg_distance(q, s)
            if d < best_d:
                best_d, best_name = d, s['name']
        return best_name, best_d

    def _segment_of(self, q):
        """Name of the profile segment that point *q* lies on (nearest)."""
        return self._nearest_segment(q)[0]

    def _boundary_name_at(self, q):
        """Boundary name for a mesh boundary element represented by point *q*.

        With material regions the mesh carries interior interface edges as extra
        boundary regions. Naming those by nearest *contour* segment — what the
        single-domain code did — would tag an interface ``'PEC'``, the solver
        would apply a Dirichlet condition on it, and the answer would be silently
        wrong. So compare the two distances directly: whichever of the outer
        contour and the region outlines is nearer wins.

        This needs no tolerance and stays correct where a region *touches* the
        wall: there the region edge coincides with the contour, both distances are
        ~0, and the tie goes to the contour — which is right, because that edge
        really is the cavity wall.
        """
        name, d_outer = self._nearest_segment(q)
        best_if, d_region = None, float('inf')
        for reg in self._regions:
            d = reg.boundary_distance(q)
            if d < d_region:
                d_region, best_if = d, f'IF_{reg.material}'
        return name if d_outer <= d_region else best_if

    def _name_boundaries(self, mesh):
        """Assign boundary names on *mesh* by matching each boundary region to
        the profile segment its elements lie on (works regardless of whether
        OCC preserved the edge names). Interior material interfaces are named
        ``IF_<material>`` — see :meth:`_boundary_name_at`."""
        # Deferred: ngsolve is an optional heavy dependency.
        from ngsolve import BND
        index_name = {}
        for el in mesh.Elements(BND):
            idx = el.index
            if idx in index_name:
                continue
            vs = [mesh[v].point for v in el.vertices]
            q = (sum(p[0] for p in vs) / len(vs), sum(p[1] for p in vs) / len(vs))
            index_name[idx] = self._boundary_name_at(q)
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

    def _region_size_points(self, mp, region_maxh):
        """Add ``RestrictH`` size points over each named region's area.

        A thin region — the 0.5 mm wall of a dielectric tube in a cavity meshed at
        ``h = 20 mm`` — would otherwise be crossed by a single element, or fail to
        mesh at all. Sizing is applied over the region's *area* (not just its
        outline) so the shell is resolved through its thickness.
        """
        by_name = {reg.material: reg for reg in self._regions}
        for name, h in region_maxh.items():
            reg = by_name.get(name)
            if reg is None:
                raise ValueError(
                    f"region_maxh: no material region named {name!r} in profile "
                    f"{self.name!r} (has {sorted(by_name)}).")
            h = float(h)
            if h <= 0:
                raise ValueError(f"region_maxh[{name!r}] must be positive, got {h!r}.")
            nz = max(2, int(np.ceil((reg.z[1] - reg.z[0]) / h)) + 1)
            nr = max(2, int(np.ceil((reg.r[1] - reg.r[0]) / h)) + 1)
            for zv in np.linspace(reg.z[0], reg.z[1], nz):
                for rv in np.linspace(reg.r[0], reg.r[1], nr):
                    mp.RestrictH(x=float(zv), y=float(rv), z=0, h=h)

    def mesh(self, maxh, order=1, edge_maxh=None, region_maxh=None):
        """Return a boundary-tagged NGSolve mesh of the profile.

        ``edge_maxh`` (boundary name -> local maxh, metres) refines specific
        boundaries below the global ``maxh`` — e.g. ``{'PEC': 1e-3}`` resolves
        the near-wall field for multipacting while the interior stays coarse.
        Implemented with ``RestrictH`` mesh-size points sampled along the named
        boundary at its target size (per-edge OCC ``maxh`` hints do not survive
        the shape-healing that :meth:`to_occ_face` relies on). Material interfaces
        are addressable here as ``IF_<material>``.

        ``region_maxh`` (material name -> local maxh, metres) does the same over a
        material region's **area**, which is what a thin dielectric shell needs.
        """
        # Deferred: netgen/ngsolve are optional heavy dependencies.
        from netgen.occ import OCCGeometry
        from ngsolve import Mesh
        from netgen.meshing import MeshingParameters

        geo = OCCGeometry(self.to_occ_shape(), dim=2)
        if edge_maxh or region_maxh:
            all_names = {s['name'] for s in self._segs} | set(self.interface_names())
            mp = MeshingParameters(maxh=maxh)
            if region_maxh:
                self._region_size_points(mp, region_maxh)
            for name, h in (edge_maxh or {}).items():
                if name not in all_names:
                    raise ValueError(f"edge_maxh: no boundary named {name!r} in this "
                                     f"profile (has {sorted(all_names)}).")
                if name.startswith('IF_'):
                    # A material interface is not on the contour, so take its
                    # outline straight from the region.
                    reg = next(r for r in self._regions if f'IF_{r.material}' == name)
                    corners = [a for a, _ in reg.edges()] + [reg.edges()[0][0]]
                    pts = np.asarray(corners, dtype=float)
                else:
                    skip = tuple(all_names - {name})
                    # contour_points densifies curved segments at ~h but keeps only
                    # the two endpoints of straight ones, so a flat/tangent wall
                    # section would get no interior size points and stay coarse.
                    # Subdivide every polyline gap wider than h so the refinement
                    # is uniform along the whole boundary.
                    pts = np.asarray(self.contour_points(float(h), skip=skip), dtype=float)
                dense = [pts[0]]
                for a, b in zip(pts[:-1], pts[1:]):
                    d = float(np.hypot(*(b - a)))
                    n_sub = max(1, int(np.ceil(d / float(h))))
                    for k in range(1, n_sub + 1):
                        dense.append(a + (b - a) * (k / n_sub))
                for z, r in dense:
                    mp.RestrictH(x=float(z), y=float(r), z=0, h=float(h))
            mesh = Mesh(geo.GenerateMesh(mp=mp))
        else:
            mesh = Mesh(geo.GenerateMesh(maxh=maxh))
        if self._regions:
            self._assert_conformal(mesh)
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
