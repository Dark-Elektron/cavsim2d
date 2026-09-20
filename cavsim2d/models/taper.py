"""Taper — a conical transition between two beam-pipe radii.

A :class:`Taper` joins a bore of radius ``R_left`` to one of ``R_right`` over a
length *L*. It is the element that makes a change of aperture a controlled piece
of geometry rather than the abrupt step
:meth:`~cavsim2d.geometry.Profile.then` inserts when two elements simply do not
match: a gentle cone reflects far less, and its wall angle is the quantity a
wakefield or trapped-mode study is sensitive to.

The two transition corners can be rounded, using the same inscribed-arc routine
the bellows uses (:func:`~cavsim2d.geometry.tangency.inscribed_corner`), which
needs a straight run to sit on — hence ``straight_left`` / ``straight_right``.
With no straight runs the cone spans the whole element and the corners belong to
whatever it is concatenated onto, so there is nothing to round and the fillet is
ignored.
"""
import os

import numpy as np

from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile, corner_offset, emit_rounded_wall


class Taper(Cavity):
    """A conical transition from bore ``R_left`` to bore ``R_right``.

    All lengths are in **mm**.

    Parameters
    ----------
    R_left, R_right : float
        Bore radius at the upstream and downstream ends.
    L : float
        Overall axial length of the element.
    straight_left, straight_right : float, optional
        Straight runs of pipe at each end, before and after the cone
        (default 0, i.e. the cone spans the whole element). They are what a
        corner radius is rounded against.
    R_fillet : float, optional
        Radius rounding both transition corners (default 0, sharp). Requires a
        straight run on the corresponding side.

    Raises
    ------
    ValueError
        If the straight runs leave no room for the cone, or a fillet does not
        fit — checked analytically at construction, like every other element, so
        an optimiser gets a named constraint rather than a meshing failure.

    Examples
    --------
    >>> t = Taper(R_left=35.0, R_right=80.0, L=120.0)
    >>> round(t.half_angle, 2)
    20.56
    """

    #: Passive: it carries no accelerating cell (see Cavity.contributes_cells).
    contributes_cells = False

    def __init__(self, R_left, R_right, L, straight_left=0.0, straight_right=0.0,
                 R_fillet=0.0, name='taper', color='k', plot_label=None,
                 chain=1, spacing=None):
        super().__init__(n_cells=1, beampipe='none', name=name, color=color,
                         plot_label=plot_label, chain=chain, spacing=spacing)
        self.kind = 'taper'
        self.n_cells = 1
        self.n_modes = 1
        self.color = color
        self.beampipe = 'none'
        self.cell_parameterisation = 'simplecell'
        self.parameters = {
            'R_left': float(R_left), 'R_right': float(R_right), 'L': float(L),
            'straight_left': float(straight_left),
            'straight_right': float(straight_right),
            'R_fillet': float(R_fillet),
        }
        self.shape = {'IC': [float(R_left), float(R_right), float(L)], 'BP': 'none'}
        self.shape_multicell = None
        self.check_feasible()

    # -- geometry ------------------------------------------------------------

    @property
    def cone_length(self):
        """Axial length of the conical run, mm."""
        p = self.parameters
        return (float(p['L']) - float(p['straight_left'])
                - float(p['straight_right']))

    @property
    def half_angle(self):
        """Cone half-angle from the axis, in degrees (0 for a straight pipe)."""
        p = self.parameters
        dr = abs(float(p['R_right']) - float(p['R_left']))
        dz = self.cone_length
        if dz <= 0:
            return 90.0
        return float(np.degrees(np.arctan2(dr, dz)))

    def _vertices(self):
        """Wall vertices in **metres**: ``[(z, r, radius), ...]``.

        The first and last carry radius 0 — they are the wall's endpoints, not
        corners.
        """
        p = self.parameters
        R_l = float(p['R_left']) * 1e-3
        R_r = float(p['R_right']) * 1e-3
        half = float(p['L']) * 1e-3 / 2.0
        s_l = float(p['straight_left']) * 1e-3
        s_r = float(p['straight_right']) * 1e-3
        fil = float(p['R_fillet']) * 1e-3
        return [(-half, R_l, 0.0),
                (-half + s_l, R_l, fil),
                (half - s_r, R_r, fil),
                (half, R_r, 0.0)]

    def check_feasible(self):
        """Validate the runs and the fillet radius.

        Raises :class:`ValueError` naming the violated constraint. Called at
        construction and again by :meth:`profile`, so a search point that cannot
        be built is rejected at parameter time.
        """
        p = self.parameters
        for key in ('R_left', 'R_right', 'L'):
            if float(p[key]) <= 0:
                raise ValueError(f'{key} must be positive, got {p[key]!r}.')
        for key in ('straight_left', 'straight_right', 'R_fillet'):
            if float(p[key]) < 0:
                raise ValueError(f'{key} must be non-negative, got {p[key]!r}.')
        if self.cone_length < 0:
            raise ValueError(
                f"the straight runs ({p['straight_left']} + {p['straight_right']} = "
                f"{float(p['straight_left']) + float(p['straight_right']):.4g} mm) "
                f"exceed the element length L={p['L']} mm, leaving no room for the "
                'cone.')

        if float(p['R_fillet']) <= 0:
            return True
        verts = self._vertices()
        slant = float(np.hypot(verts[2][0] - verts[1][0], verts[2][1] - verts[1][1]))
        offsets = []
        for i, side, run in ((1, 'left', float(p['straight_left']) * 1e-3),
                             (2, 'right', float(p['straight_right']) * 1e-3)):
            d = corner_offset([verts[i][0], verts[i][1]],
                              [verts[i - 1][0], verts[i - 1][1]],
                              [verts[i + 1][0], verts[i + 1][1]],
                              verts[i][2])
            offsets.append(d)
            if d > run + 1e-12:
                raise ValueError(
                    f'the {side} fillet does not fit: it reaches {d * 1e3:.4g} mm '
                    f'back along a straight_{side} of {run * 1e3:.4g} mm. Reduce '
                    f'R_fillet, or lengthen straight_{side}.')
        if sum(offsets) > slant + 1e-12:
            raise ValueError(
                f'the two fillets do not both fit on the {slant * 1e3:.4g} mm cone: '
                f'they need {sum(offsets) * 1e3:.4g} mm. Reduce R_fillet, or '
                'lengthen the taper.')
        return True

    def profile(self):
        """Meridian boundary as a :class:`~cavsim2d.geometry.Profile` (metres).

        Both ends are ``'PMC'`` apertures at their own bore radius, so the taper
        concatenates onto a pipe of ``R_left`` upstream and ``R_right``
        downstream with no step at either junction.
        """
        try:
            self.check_feasible()
            verts = self._vertices()
        except (KeyError, TypeError, ValueError):
            return None

        (z0, r0, _), (z1, r1, _) = verts[0], verts[-1]
        prof = Profile(self.name).start(z0, 0.0).line_to(z0, r0, 'PMC')
        emit_rounded_wall(prof, verts, 'PEC')
        prof.line_to(z1, 0.0, 'PMC')
        return self._chained(prof.close('AXI'))

    # -- model plumbing ------------------------------------------------------

    def get_geometric_parameters(self):
        return self.parameters

    def create(self, n_cells=None, beampipe=None, mode=None):
        """Provision the workspace. Native-only: ``profile()`` is the single
        geometry source, so no gmsh ``.geo`` is written."""
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
            self._write_geometry_snapshot()

    def rebuild(self, parameters, beampipe=None):
        """A fresh taper from its parameter dict."""
        return type(self)(
            R_left=float(parameters['R_left']),
            R_right=float(parameters['R_right']),
            L=float(parameters['L']),
            straight_left=float(parameters.get('straight_left', 0.0)),
            straight_right=float(parameters.get('straight_right', 0.0)),
            R_fillet=float(parameters.get('R_fillet', 0.0)),
            name=self.name, color=self.color, plot_label=self.plot_label,
            chain=getattr(self, 'chain', 1), spacing=getattr(self, 'spacing', None))
