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
        """Sample points along an ellipse segment (for boundary matching)."""
        c, major, minor, xdir, t_lo, t_hi = cls._ellipse_span(seg, pts)
        ux, uy = xdir
        vx, vy = -uy, ux
        out = []
        for t in np.linspace(t_lo, t_hi, n):
            ct, st = np.cos(t), np.sin(t)
            out.append((c[0] + major * ct * ux + minor * st * vx,
                        c[1] + major * ct * uy + minor * st * vy))
        return out

    def close(self, boundary):
        """Straight segment from the current point back to the start, tagged
        *boundary* (typically the axis)."""
        i0 = len(self._pts) - 1
        self._segs.append({'kind': 'line', 'i0': i0, 'i1': 0, 'name': boundary})
        return self

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
        from netgen.occ import Segment, Wire, Face, Pnt, ArcOfCircle

        if len(self._segs) < 3:
            raise ValueError("A profile needs at least 3 segments to bound a face.")

        from netgen.occ import Ellipse, gp_Ax2d, gp_Pnt2d, gp_Dir2d

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

    def mesh(self, maxh, order=1):
        """Return a boundary-tagged NGSolve mesh of the profile."""
        from netgen.occ import OCCGeometry
        from ngsolve import Mesh
        mesh = Mesh(OCCGeometry(self.to_occ_face(), dim=2).GenerateMesh(maxh=maxh))
        self._name_boundaries(mesh)
        if order and order > 1:
            mesh.Curve(order)
        return mesh


def mesh_from_profile(profile, maxh, order=1):
    """Convenience wrapper: mesh a :class:`Profile`."""
    return profile.mesh(maxh=maxh, order=order)
