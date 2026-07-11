from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import numpy as np
import os

class SplineCavity(Cavity):
    # 'Berzier' was the historical (misspelt) default. It matched neither branch of
    # write_geometry, so the .geo came out with no cavity wall at all — a degenerate
    # triangle. Accept it as an alias for 'Bezier' rather than silently mis-meshing.
    _KIND_ALIASES = {'bspline': 'bspline', 'bezier': 'bezier', 'berzier': 'bezier'}

    def __init__(self, shape, name='SplineCavity', kind='Bezier'):
        """A cavity whose wall is a spline through control points.

        Parameters
        ----------
        shape : dict
            Geometry description with these keys:

            - ``'geometry'`` (**required**) : the control points, each a
              ``[z, r]`` pair in **millimetres**, keyed ``'p0'``, ``'p1'``, … in
              order along the wall (from the axis upstream, up and over the wall,
              back to the axis). For ``kind='bezier'`` they are the Bézier control
              polygon (one curve per cell); for ``kind='bspline'`` the poles.

              Two forms, mirroring :class:`EllipticalCavity`:

              * a single control-point dict used for every cell; or
              * per-cell dicts under ``'mid_cell'`` / ``'end_cell_left'`` /
                ``'end_cell_right'`` (aliases ``'mid'`` / ``'end cell left'`` /
                ``'end cell right'``, or ``'IC'`` / ``'OC'`` / ``'OC_R'``). Each
                end cell defaults to the mid cell when omitted, and the mid cell
                falls back to whichever end cell is given.
            - ``'n_cells'`` (optional, default 1) : number of cells.
            - ``'beampipe'`` (optional, default ``'none'``) : ``'none'``,
              ``'left'``, ``'right'`` or ``'both'``.
            - ``'beampipe_length'`` (optional, mm) : straight-pipe length; defaults
              to one cell width.
        name : str
            Cavity name (used for its output folder).
        kind : {'Bezier', 'bspline'}
            Spline type for the wall. ``'berzier'`` is accepted as a legacy alias.

        Examples
        --------
        >>> geom = {'p0': [0, 35], 'p1': [0, 70], 'p2': [30, 103],
        ...         'p3': [85, 103], 'p4': [115, 70], 'p5': [115, 35]}
        >>> cav = SplineCavity({'geometry': geom, 'n_cells': 9}, kind='Bezier')
        >>> # or distinct end cells:
        >>> SplineCavity({'geometry': {'mid_cell': geom, 'end_cell_left': geom},
        ...               'n_cells': 9, 'beampipe': 'both'})
        """
        super().__init__(name)
        self.self_dir = None
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name

        # These were previously written to *local* variables (not self.*), so
        # `n_cells` and `beampipe` from the shape dict were silently dropped.
        self.n_cells = shape.get('n_cells', 1)
        self.beampipe = shape.get('beampipe', 'none')
        # Optional straight beam-pipe length (mm); None -> default (one cell
        # width) when a beam pipe is requested. See profile().
        self.beampipe_length = shape.get('beampipe_length', None)

        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = kind

        self.shape = {
            "geometry": shape['geometry'],
            'BP': self.beampipe,
            'CELL PARAMETERISATION': self.cell_parameterisation,
            'kind': self.kind}

        self.shape_multicell = {'kind': self.kind}

        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        if self.projectDir:
            # Create cavity directory directly inside project folder
            self.self_dir = os.path.join(self.projectDir, self.name)
            geo_dir = os.path.join(self.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)

            self.uq_dir = os.path.join(self.self_dir, 'uq')

            # Meshing and the ABCI deck both come from the native profile(); the
            # legacy .geo writer only understands a single flat control polygon.
            # For per-cell (nested) geometry, stay native-only (geo_filepath =
            # None), as multicell elliptical cavities do.
            nested = any(isinstance(v, dict) for v in (self.parameters or {}).values())
            if nested:
                self.geo_filepath = None
            else:
                self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
                self.write_geometry(self.parameters, n_cells, beampipe,
                                    write=self.geo_filepath)

    def get_geometric_parameters(self):
        self.parameters = self.shape['geometry']

    #: A control point is ``[z, r]``; each contributes two scalar tune handles.
    COORD_SUFFIXES = ('_z', '_r')

    def tune_variables(self):
        """Control-point coordinates, e.g. ``'p2_r'``.

        The wall is parameterised by points rather than by scalars, so the base
        implementation (which only accepts scalar parameters) would report none.
        Each point ``p2 = [z, r]`` supplies two handles instead.
        """
        return {f'{key}{suffix}'
                for key, value in (self.parameters or {}).items()
                if isinstance(key, str) and np.ndim(value) == 1 and len(value) == 2
                for suffix in self.COORD_SUFFIXES}

    def _split_coordinate(self, name):
        """``'p2_r'`` -> ``('p2', 1)``."""
        for index, suffix in enumerate(self.COORD_SUFFIXES):
            key = name[:-len(suffix)]
            if name.endswith(suffix) and key in self.parameters:
                return key, index
        raise ValueError(
            f"Unknown tune variable {name!r} for {type(self).__name__}. "
            f"It accepts: {', '.join(sorted(self.tune_variables()))}.")

    def get_tune_value(self, name):
        key, index = self._split_coordinate(name)
        return self.parameters[key][index]

    def set_tune_value(self, name, value):
        key, index = self._split_coordinate(name)
        point = list(self.parameters[key])
        point[index] = value
        self.parameters[key] = point

    def spline_kind(self):
        """Normalised curve type: ``'bspline'`` or ``'bezier'`` (None if unknown)."""
        return self._KIND_ALIASES.get(str(self.kind).lower())

    # geometry keys accepted for per-cell control points (elliptical-style).
    _CELL_ALIASES = {
        'mid_cell': 'mid_cell', 'mid': 'mid_cell', 'mid cell': 'mid_cell', 'ic': 'mid_cell',
        'end_cell_left': 'end_cell_left', 'end cell left': 'end_cell_left',
        'end_cell': 'end_cell_left', 'end cell': 'end_cell_left',
        'left': 'end_cell_left', 'oc': 'end_cell_left',
        'end_cell_right': 'end_cell_right', 'end cell right': 'end_cell_right',
        'right': 'end_cell_right', 'oc_r': 'end_cell_right',
    }

    @staticmethod
    def _poly(cell):
        """Control polygon ``(N, 2)`` array from a ``{p0:[z,r], ...}`` dict,
        ordered by point index."""
        keys = sorted(cell, key=lambda s: int(''.join(c for c in str(s) if c.isdigit()) or 0))
        return np.array([cell[k] for k in keys], dtype=float)

    def _cell_length_m(self):
        """Mid-cell axial width (iris to iris) in metres, for the dispersion light
        line — the z-span of the mid-cell control polygon."""
        cells = self._cell_polys()
        if not cells:
            return None
        mid = cells[len(cells) // 2]              # an interior cell for a multicell
        width = float(mid[-1][0] - mid[0][0])     # mm
        return width * 1e-3 if width > 0 else None

    def _cell_polys(self):
        """The ``n_cells`` control polygons in cell order (end-left, mid…, end-right).

        The ``geometry`` may be a bare ``{p0:…, p5:…}`` dict (one control polygon,
        used for every cell), or give per-cell polygons under ``mid_cell`` /
        ``end_cell_left`` / ``end_cell_right`` (aliases ``mid``/``end cell
        left``/``end cell right``, or ``IC``/``OC``/``OC_R``). Following the
        elliptical rules, each end cell defaults to the mid cell when omitted, and
        the mid cell falls back to whichever end cell is given."""
        geom = self.parameters or {}
        if not geom:
            return None
        nested = any(isinstance(v, dict) for v in geom.values())
        if not nested:
            mid = self._poly(geom)
            left = right = mid
        else:
            norm = {self._CELL_ALIASES.get(str(k).lower().strip(), str(k).lower()): v
                    for k, v in geom.items()}
            mid_src = norm.get('mid_cell') or norm.get('end_cell_left') or norm.get('end_cell_right')
            if mid_src is None:
                return None
            mid = self._poly(mid_src)
            left = self._poly(norm['end_cell_left']) if 'end_cell_left' in norm else mid
            right = self._poly(norm['end_cell_right']) if 'end_cell_right' in norm else mid
        n = max(1, int(self.n_cells))
        if n == 1:
            return [mid]
        return [left] + [mid] * (n - 2) + [right]

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` (metres) — the native
        netgen.occ path, with the cavity wall as an exact spline.

        The contour is: axis -> first control point (PMC aperture), the spline wall
        (PEC), down to the axis (PMC aperture), then back along the axis (AXI).
        For ``'bezier'`` one curve per cell; for ``'bspline'`` a single curve
        through every cell's poles. Cells may differ (mid vs end cells) — see
        :meth:`_cell_polys`.
        """
        kind = self.spline_kind()
        if kind is None:
            return None
        cells = self._cell_polys()
        if cells is None:
            return None
        try:
            cells = [np.asarray(c, dtype=float) * 1e-3 for c in cells]
        except (TypeError, ValueError):
            return None
        if any(c.ndim != 2 or c.shape[0] < 3 or c.shape[1] != 2 for c in cells):
            return None

        # Place each cell after the previous: shift so its first pole sits at the
        # running z cursor. Cells connect because the spline interpolates its
        # endpoints and adjacent apertures share the iris radius.
        placed, z_cursor = [], 0.0
        for c in cells:
            s = c.copy()
            s[:, 0] += z_cursor - c[0][0]
            placed.append(s)
            z_cursor += c[-1][0] - c[0][0]         # this cell's z width

        bp = (self.beampipe or 'none').lower()
        r_ap_l = float(placed[0][0][1])            # left aperture radius (p0)
        r_ap_r = float(placed[-1][-1][1])          # right aperture radius (last p)
        default_bp = float(placed[0][-1][0] - placed[0][0][0])   # first cell width
        L_bp = (self.beampipe_length * 1e-3) if self.beampipe_length else default_bp
        L_bp_l = L_bp if bp in ('both', 'left') else 0.0
        L_bp_r = L_bp if bp in ('both', 'right') else 0.0

        prof = Profile('spline')
        z0 = -L_bp_l
        prof.start(z0, 0.0)
        prof.line_to(z0, r_ap_l, 'PMC')            # left aperture
        if L_bp_l > 0:
            prof.line_to(0.0, r_ap_l, 'PEC')       # left beam pipe up to the first cell

        # Each cell contributes its poles after the first: the previous point
        # (the aperture, or the previous cell's last pole) already supplies p0.
        if kind == 'bspline':
            all_poles = []
            for s in placed:
                all_poles.extend(s[1:].tolist())
            prof.spline_to(all_poles, 'PEC', kind='bspline', degree=3)
        else:
            for s in placed:
                prof.spline_to(s[1:].tolist(), 'PEC', kind='bezier')

        z_end = float(placed[-1][-1][0])
        if L_bp_r > 0:
            z_end += L_bp_r
            prof.line_to(z_end, r_ap_r, 'PEC')     # right beam pipe
        prof.line_to(z_end, 0.0, 'PMC')            # right aperture
        prof.close('AXI')
        return prof

    def write_geometry(self, parameters, n_cells=1, beampipe='none', write=None):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        parameters = np.array(list(parameters.values()))*1e-3

        Ri_el = parameters[0][0]
        L_m = abs(parameters[0][1] - parameters[-1][1])

        step = 0.0005
        shift = 0

        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # SHIFT POINT TO START POINT
            start_point = [shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)

            pt = [-shift, Ri_el]
            pt_indx = add_point(cav, parameters[0], pt_indx)

            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # cavity cell surface
            kind = self.spline_kind()
            if kind is None:
                raise ValueError(f"unknown spline kind {self.kind!r}; "
                                 "expected 'BSpline' or 'Bezier'")
            if kind == 'bspline':
                pt_indx, curve_indx = add_bspline(cav, pt_indx, curve_indx, parameters[1:], n_cells)
                curve.append(curve_indx)
            if kind == 'bezier':
                for i in range(n_cells):
                    pt_indx, curve_indx = add_bezierspline(cav, pt_indx, curve_indx, parameters[1:])
                    curve.append(curve_indx)

                    if i < n_cells-1:
                        parameters[:, 0] += parameters[-1][0]


            # right iris
            pt_indx = add_point(cav, [parameters[-1][0], 0], pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            pmcs = [1, curve[-2]]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')

    def write_quarter_geometry(self, parameters, bp='none', bc=None, tangent_check=False,
                                          ignore_degenerate=False, plot=False, write=None, dimension=False,
                                          contour=False, **kwargs):
        """
        Plot cavity geometry

        Parameters
        ----------
        tangent_check
        bc
        ax
        ignore_degenerate
        IC: list, ndarray
            Inner Cell geometric parameters list
        OC: list, ndarray
            Left outer Cell geometric parameters list
        OC_R: list, ndarray
            Right outer Cell geometric parameters list
        BP: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        n_cell: int
            Number of cavity cells
        scale: float
            Scale of the cavity geometry

        Returns
        -------

        """

        GEO = """
        """

        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        if bp == 'none':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)
        elif bp == 'left':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_el']*1e-3 for n in names)
        elif bp == 'right':
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_er']*1e-3 for n in names)
        else:
            A, B, a, b, Ri, L, Req = (parameters[f'{n}_m']*1e-3 for n in names)

        L_bp = 4 * L
        if dimension or contour:
            L_bp = 1 * L

        if bp == 'left' or bp == 'right':
            L_bp_l = L_bp
        else:
            L_bp_l = 0

        step = 0.0005

        # calculate shift
        shift = (L_bp_l + L) / 2

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # define parameters
            cav.write(f'\nA = DefineNumber[{A}, Name "Parameters/Equator ellipse major axis"];')
            cav.write(f'\nB = DefineNumber[{B}, Name "Parameters/Equator ellipse minor axis"];')
            cav.write(f'\na = DefineNumber[{a}, Name "Parameters/Iris ellipse major axis"];')
            cav.write(f'\nb = DefineNumber[{b}, Name "Parameters/Iris ellipse minor axis"];')
            cav.write(f'\nRi = DefineNumber[{Ri}, Name "Parameters/Iris radius"];')
            cav.write(f'\nL = DefineNumber[{L}, Name "Parameters/Half cell length"];')
            cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Equator radius"];\n')

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])

            pt = [-shift, 'Ri']

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pt = [L_bp_l - shift, 'Ri']

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

            df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp_l, tangent_check=tangent_check)
            x1, y1, x2, y2 = df[0]
            if not ignore_degenerate:
                msg = df[-2]
                if msg != 1:
                    error('Parameter set leads to degenerate geometry.')
                    # save figure of error
                    return

            start_pt = pt
            center_pt = [L_bp_l - shift, 'Ri + b']
            majax_pt = [f'{L_bp_l - shift} + a', 'Ri + b']
            end_pt = [-shift + x1, y1]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L_bp_l - shift, Ri + b, a, b, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            # geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pt = [-shift + x2, y2]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

            start_pt = pt
            center_pt = [f'L + {L_bp_l - shift}', 'Req - B']
            majax_pt = [f'L + {L_bp_l - shift} - A', 'Req - B']
            end_pt = [f'{L_bp_l - shift} + L', 'Req']
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcTo(L + L_bp_l - shift, Req - B, A, B, step, pt, [L_bp_l + L - shift, Req])
            pt = [f'{L_bp_l - shift} + L', 'Req']

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_l + L) / 2

            # END PATH
            # pts = lineTo(pt, [L+ L_bp_l + - shift, 0],
            #              step)  # to add beam pipe to right

            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [L + L_bp_l - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            geo.append([pt[1], pt[0], 2])

            pmcs = [1]
            axis = [curve[-1]]
            pecs = [x for x in curve if (x not in pmcs and x not in axis)]

            cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
            cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

            cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
            cav.write(f"\nPlane Surface(1) = {{{1}}};")
            cav.write(f"\nReverse Surface {1};")
            cav.write(f'\nPhysical Surface("Domain") = {1};')
    def rebuild(self, parameters, beampipe=None):
        """A fresh SplineCavity from its control-point dict."""
        return SplineCavity(
            {'geometry': dict(parameters), 'n_cells': self.n_cells,
             'beampipe': beampipe if beampipe is not None else self.beampipe,
             'beampipe_length': self.beampipe_length},
            name=self.name, kind=self.kind)



