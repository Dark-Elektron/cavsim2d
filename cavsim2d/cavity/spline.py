from cavsim2d.cavity.base import Cavity
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import numpy as np
import os

class SplineCavity(Cavity):
    def __init__(self, shape, name='SplineCavity', kind='Berzier'):
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
            if self.kind == 'BSpline':
                pt_indx, curve_indx = add_bspline(cav, pt_indx, curve_indx, parameters[1:], n_cells)
                curve.append(curve_indx)
            if self.kind == 'Bezier':
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


