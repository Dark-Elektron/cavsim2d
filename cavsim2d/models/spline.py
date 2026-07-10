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
        # self.shape_space = {
        #     'IC': [L, Req, Ri, S, L_bp],
        #     'BP': beampipe
        # }
        super().__init__(name)
        self.self_dir = None
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name

        if 'n_cells' in shape.keys():
            n_cells = shape['n_cells']
        else:
            self.n_cells = 1

        if 'beampipe' in shape.keys():
            beampipe = shape['beampipe']
        else:
            self.beampipe = 'none'

        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = kind
        self.n_cells = 1
        if 'n_cells' in shape.keys():
            self.n_cells = shape['n_cells']

        self.shape = {
            "geometry": shape['geometry'],
            'BP': 'none',
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

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` (metres) — the native
        netgen.occ path, with the cavity wall as an exact spline.

        The contour is: axis -> first control point (PMC aperture), the spline wall
        (PEC), down to the axis (PMC aperture), then back along the axis (AXI).
        Control-point semantics match the ``.geo`` writer: for ``'bezier'`` one
        curve per cell, for ``'bspline'`` a single curve through every cell's poles.
        """
        kind = self.spline_kind()
        if kind is None:
            return None
        try:
            poles = np.array(list(self.parameters.values()), dtype=float) * 1e-3
        except (TypeError, ValueError):
            return None
        if poles.ndim != 2 or poles.shape[0] < 3 or poles.shape[1] != 2:
            return None

        prof = Profile('spline')
        prof.start(0.0, 0.0)
        prof.line_to(poles[0][0], poles[0][1], 'PMC')       # left aperture

        # Cells repeat the control polygon, each shifted by the running last-z, which
        # is exactly what add_bspline / write_geometry do (the shift compounds).
        rest = poles[1:].copy()
        if kind == 'bspline':
            all_poles = []
            for i in range(self.n_cells):
                all_poles.extend(rest.tolist())
                if i < self.n_cells - 1:
                    rest = rest + np.array([rest[-1][0], 0.0])
            prof.spline_to(all_poles, 'PEC', kind='bspline', degree=3)
            z_end = all_poles[-1][0]
        else:
            for i in range(self.n_cells):
                prof.spline_to(rest.tolist(), 'PEC', kind='bezier')
                if i < self.n_cells - 1:
                    rest = rest + np.array([rest[-1][0], 0.0])
            z_end = rest[-1][0]

        prof.line_to(z_end, 0.0, 'PMC')                     # right aperture
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
        return SplineCavity({'geometry': dict(parameters), 'n_cells': self.n_cells},
                            name=self.name, kind=self.kind)



