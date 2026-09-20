"""Bellows — a corrugated beam-pipe section.

A bellows is drawn as a square-wave corrugation on the bore whose every corner
is rounded: root flat, flank, crest flat, flank, repeated ``N_conv`` times. The
rounding is emitted as real :class:`~cavsim2d.geometry.Profile` arc segments, not
applied by the CAD kernel afterwards, because the segment list is what the ABCI
writer, the multipacting contour and the plotters all read — a fillet that
existed only in the OCC face would give the eigenmode solver one geometry and the
wake solver another.

Corner construction
-------------------
Every corner is the same problem: an arc of radius *R* inscribed in the wedge
between the incoming and outgoing wall directions. For a wedge of angle
``phi`` the tangent points sit ``R / tan(phi / 2)`` back from the vertex along
each edge, and the centre is on the bisector at ``R / sin(phi / 2)``. With
vertical flanks every wedge is a right angle and that offset is just ``R``; with
a tilted flank (``flank_angle < 90``) every wedge is ``180 - flank_angle`` and the
offset is ``R * tan(flank_angle / 2)``. It is one routine either way, so a
vendor drawing with a leaning flank costs a parameter, not a second geometry
path.
"""
import os

import numpy as np

from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile, emit_rounded_wall

#: Flats shorter than this (metres) are treated as absent, so a crest whose
#: fillet radius reaches L_p/4 becomes arc-to-arc instead of emitting a
#: zero-length edge that OCC would reject.
FLAT_TOL = 1e-12


class Bellows(Cavity):
    """A corrugated bellows section of ``N_conv`` convolutions.

    All lengths are in **mm**, matching every other model.

    Parameters
    ----------
    Ri : float
        Bore (root) radius — the radius of the pipe the bellows corrugates.
    A : float
        Convolution depth, so the crest radius is ``Ri + A``.
    L_p : float
        Convolution period (one root flat + two flanks + one crest flat).
    N_conv : int
        Number of convolutions.
    R_root, R_crest : float
        Corner radii at the root and at the crest. They may differ.
    crest_fraction : float, optional
        Share of the per-period *flat* length given to the crest, default 0.5
        (an equal-duty corrugation). The rest goes to the root flat.
    flank_angle : float, optional
        Flank angle from the axis in degrees, default 90 (vertical flanks).
        Smaller values lean the flank outward, giving a trapezoidal convolution.
    L_bp : float, optional
        Straight beam pipe carried at each end, default 0.

    Raises
    ------
    ValueError
        If the corner radii do not fit — checked analytically at construction,
        so an optimiser or a UQ sweep gets a named constraint violation rather
        than an opaque failure from the meshing kernel.

    Examples
    --------
    >>> b = Bellows(Ri=35.0, A=8.0, L_p=6.0, N_conv=10,
    ...             R_root=1.2, R_crest=1.2)
    >>> round(b.length, 3)
    60.0
    """

    #: Passive: it carries no accelerating cell (see Cavity.contributes_cells).
    contributes_cells = False

    def __init__(self, Ri, A, L_p, N_conv, R_root, R_crest,
                 crest_fraction=0.5, flank_angle=90.0, L_bp=0.0,
                 name='bellows', color='k', plot_label=None,
                 chain=1, spacing=None):
        super().__init__(n_cells=1, beampipe='none', name=name, color=color,
                         plot_label=plot_label, chain=chain, spacing=spacing)
        self.kind = 'bellows'
        self.n_cells = 1
        self.n_modes = 1
        self.color = color
        # N_conv is discrete, so it is deliberately *not* a parameter: it must
        # not show up as a continuous tune/UQ handle (see Cavity.tune_variables).
        self.N_conv = int(N_conv)
        if self.N_conv < 1:
            raise ValueError(f'N_conv must be at least 1, got {N_conv!r}.')
        self.parameters = {
            'Ri': float(Ri), 'A': float(A), 'L_p': float(L_p),
            'R_root': float(R_root), 'R_crest': float(R_crest),
            'crest_fraction': float(crest_fraction),
            'flank_angle': float(flank_angle), 'L_bp': float(L_bp),
        }
        self.beampipe = 'none'
        self.cell_parameterisation = 'simplecell'
        self.shape = {'IC': [float(Ri), float(A), float(L_p),
                             float(R_root), float(R_crest)], 'BP': 'none'}
        self.shape_multicell = None
        self.check_feasible()

    # -- geometry ------------------------------------------------------------

    @property
    def length(self):
        """Axial length of the corrugated run (excluding the end pipes), mm."""
        return self.N_conv * float(self.parameters['L_p'])

    def _layout(self):
        """Per-period lengths in **mm**: ``(w_root, w_crest, flank_dz, offsets)``.

        ``offsets`` is ``(d_root, d_crest)`` — how far each corner's tangent
        points sit back from its vertex.
        """
        p = self.parameters
        A, L_p = float(p['A']), float(p['L_p'])
        theta = np.radians(float(p['flank_angle']))
        # Axial extent consumed by one flank. A vertical flank consumes none.
        flank_dz = 0.0 if abs(theta - np.pi / 2) < 1e-12 else A / np.tan(theta)
        flats = L_p - 2.0 * flank_dz
        f = float(p['crest_fraction'])
        w_crest = f * flats
        w_root = flats - w_crest
        # The wedge at every corner is (180 - flank_angle); the tangent-point
        # offset R / tan(wedge / 2) reduces to R * tan(flank_angle / 2).
        t = np.tan(theta / 2.0)
        return w_root, w_crest, flank_dz, (float(p['R_root']) * t,
                                           float(p['R_crest']) * t)

    def check_feasible(self):
        """Validate the corner radii against the flats and the flank.

        Raises :class:`ValueError` naming the violated constraint. Called at
        construction and again by :meth:`rebuild`, so an infeasible point is
        rejected at parameter time — before any mesh is attempted.
        """
        p = self.parameters
        A = float(p['A'])
        theta = float(p['flank_angle'])
        if not 0.0 < theta <= 90.0:
            raise ValueError(
                f'flank_angle must be in (0, 90] degrees, got {theta!r}.')
        if not 0.0 < float(p['crest_fraction']) < 1.0:
            raise ValueError('crest_fraction must be strictly between 0 and 1, '
                             f"got {p['crest_fraction']!r}.")
        for key in ('Ri', 'A', 'L_p'):
            if float(p[key]) <= 0:
                raise ValueError(f'{key} must be positive, got {p[key]!r}.')
        for key in ('R_root', 'R_crest', 'L_bp'):
            if float(p[key]) < 0:
                raise ValueError(f'{key} must be non-negative, got {p[key]!r}.')

        w_root, w_crest, _, (d_root, d_crest) = self._layout()
        if w_root < 0 or w_crest < 0:
            raise ValueError(
                f'the flanks do not fit in the period: at flank_angle='
                f'{theta} deg a depth A={A} mm needs '
                f'{2 * (A / np.tan(np.radians(theta))):.4g} mm of the '
                f"L_p={p['L_p']} mm period, leaving no room for the flats.")
        flank = A / np.sin(np.radians(theta))
        if 2.0 * d_root > w_root + FLAT_TOL:
            raise ValueError(
                f"root corners do not fit: 2 * R_root * tan(flank_angle/2) = "
                f"{2 * d_root:.4g} mm exceeds the {w_root:.4g} mm root flat. "
                f"Reduce R_root below {w_root / (2 * d_root) * float(p['R_root']):.4g} mm, "
                f'or lengthen L_p.')
        if 2.0 * d_crest > w_crest + FLAT_TOL:
            raise ValueError(
                f"crest corners do not fit: 2 * R_crest * tan(flank_angle/2) = "
                f"{2 * d_crest:.4g} mm exceeds the {w_crest:.4g} mm crest flat. "
                f"Reduce R_crest below {w_crest / (2 * d_crest) * float(p['R_crest']):.4g} mm, "
                f'or lengthen L_p.')
        if d_root + d_crest > flank + FLAT_TOL:
            raise ValueError(
                f'the root and crest corners do not both fit on the '
                f'{flank:.4g} mm flank: they need '
                f'{d_root + d_crest:.4g} mm. Reduce R_root / R_crest, or '
                f'deepen the convolution (A).')
        return True

    def _vertices(self):
        """Sharp-corner wall vertices in **metres**: ``[(z, r, radius), ...]``.

        The first and last entries carry ``radius = 0``: they are the wall's
        endpoints on the beam pipe, collinear with it, so they are not corners.
        """
        p = self.parameters
        Ri = float(p['Ri']) * 1e-3
        A = float(p['A']) * 1e-3
        L_bp = float(p['L_bp']) * 1e-3
        R_root = float(p['R_root']) * 1e-3
        R_crest = float(p['R_crest']) * 1e-3
        w_root, w_crest, flank_dz, _ = self._layout()
        w_root, w_crest, flank_dz = (w_root * 1e-3, w_crest * 1e-3, flank_dz * 1e-3)

        z = 0.0
        out = [(z, Ri, 0.0)]                        # wall start (on the pipe)
        z += w_root / 2.0
        for i in range(self.N_conv):
            out.append((z, Ri, R_root))             # root corner, turning up
            z += flank_dz
            out.append((z, Ri + A, R_crest))        # crest corner, levelling off
            z += w_crest
            out.append((z, Ri + A, R_crest))        # crest corner, turning down
            z += flank_dz
            out.append((z, Ri, R_root))             # root corner, levelling off
            if i < self.N_conv - 1:
                z += w_root
        z += w_root / 2.0
        out.append((z, Ri, 0.0))                    # wall end (on the pipe)
        if L_bp > 0:
            out = ([(out[0][0] - L_bp, Ri, 0.0)] + out
                   + [(out[-1][0] + L_bp, Ri, 0.0)])
        return out

    def profile(self):
        """Meridian boundary as a :class:`~cavsim2d.geometry.Profile` (metres).

        The wall runs left to right at the bore radius, rounding each corner
        with an exact circular arc; both ends are ``'PMC'`` apertures so the
        element composes and the solver can retag them.
        """
        try:
            self.check_feasible()
            verts = self._vertices()
        except (KeyError, TypeError, ValueError):
            return None

        z0, r0 = verts[0][0], verts[0][1]
        z1 = verts[-1][0]
        prof = Profile(self.name).start(z0, 0.0).line_to(z0, r0, 'PMC')
        emit_rounded_wall(prof, verts, 'PEC')
        prof.line_to(z1, 0.0, 'PMC')
        return self._chained(prof.close('AXI'))

    # -- model plumbing ------------------------------------------------------

    def get_geometric_parameters(self):
        return self.parameters

    def create(self, n_cells=None, beampipe=None, mode=None):
        """Provision the workspace. The bellows is native-only — its wall is a
        run of exact arcs that the gmsh ``.geo`` fallback has no encoding for, so
        no ``.geo`` is written and ``profile()`` is the single source."""
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
            self._write_geometry_snapshot()

    def rebuild(self, parameters, beampipe=None):
        """A fresh bellows from its parameter dict (``N_conv`` comes from self)."""
        return type(self)(
            Ri=float(parameters['Ri']), A=float(parameters['A']),
            L_p=float(parameters['L_p']), N_conv=self.N_conv,
            R_root=float(parameters['R_root']),
            R_crest=float(parameters['R_crest']),
            crest_fraction=float(parameters.get('crest_fraction', 0.5)),
            flank_angle=float(parameters.get('flank_angle', 90.0)),
            L_bp=float(parameters.get('L_bp', 0.0)),
            name=self.name, color=self.color, plot_label=self.plot_label,
            chain=getattr(self, 'chain', 1), spacing=getattr(self, 'spacing', None))

    def _reconstruct_state(self):
        state = super()._reconstruct_state()
        state['N_conv'] = int(self.N_conv)
        return state

    @classmethod
    def _reconstruct_from_state(cls, state):
        stub = cls.__new__(cls)
        stub.N_conv = int(state.get('N_conv', 1))
        stub.name = state.get('name', 'bellows')
        stub.color = state.get('color', 'k')
        stub.plot_label = state.get('plot_label', None)
        cav = stub.rebuild(dict(state['parameters']))
        cav.name = state.get('name', cav.name)
        return cav
