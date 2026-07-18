from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
from scipy.signal import find_peaks
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class RFGun(Cavity):
    def __init__(self, shape, name='gun'):
        """A VHF-type RF gun, its wall a chain of arcs and straight segments.

        Parameters
        ----------
        shape : dict
            ``{'geometry': {...}}``. The ``'geometry'`` dict describes the
            meridian wall as an ordered chain of primitives, running from the
            cathode plane out to the exit aperture. Unlike the other cavity
            types, **gun parameters are in metres (SI), and angles in radians**.
            The keys, in wall order:

            - ``y1``  : cathode-plane aperture radius.
            - ``R2``  : radius of the first (cathode-nose) arc; ``T2`` its angle.
            - ``L3``  : straight length after that arc (at angle ``T2``).
            - ``R4``  : blend-arc radius; ``L5`` the following straight length.
            - ``R6``  : radius of the arc into the barrel; ``L7`` the barrel length.
            - ``R8``  : nose-cone entry-arc radius; ``T9`` its angle (``R9`` is
              derived internally so the wall stays tangent).
            - ``R10`` : exit-nose arc radius; ``T10`` its angle.
            - ``L11`` : straight length after the nose; ``R12`` the next arc radius.
            - ``L13`` : straight length to the exit; ``R14`` the exit-aperture arc.
            - ``x``   : downstream drift-tube length to the exit aperture.
        name : str
            Cavity name (used for its output folder).

        Notes
        -----
        The gun already ships with a downstream drift tube (``x``), so a wakefield
        run adds a beam pipe only on the cathode side.
        """
        super().__init__(name)
        self.self_dir = None
        self.cell_parameterisation = 'simplecell'  # consider removing
        self.name = name
        self.n_cells = 1
        self.beampipe = 'none'
        self.n_modes = 1
        self.axis_field = None
        self.bc = 'mm'
        self.projectDir = None
        self.kind = 'vhf gun'

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
            self.write_geometry(self.parameters,
                                write=self.geo_filepath)

    def get_geometric_parameters(self):
        self.parameters = self.shape['geometry']

    @staticmethod
    def _r9(p):
        """Radius of the arc that closes the gun contour, fixed by the others."""
        y1, R2, T2, L3, R4, L5, R6, R8, T9 = (p['y1'], p['R2'], p['T2'], p['L3'],
                                              p['R4'], p['L5'], p['R6'], p['R8'], p['T9'])
        R10, T10, L11, R12, L13, R14, x = (p['R10'], p['T10'], p['L11'], p['R12'],
                                           p['L13'], p['R14'], p['x'])
        return (((y1 + R2 * np.sin(T2) + L3 * np.cos(T2) + R4 * np.sin(T2) + L5 + R6) -
                 (R14 + L13 + R12 * np.sin(T10) + L11 * np.cos(T10) + R10 * np.sin(T10)
                  + x + R8 * (1 - np.sin(T9))))) / np.sin(T9)

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` — the native netgen.occ
        path, with exact circular arcs.

        Mirrors :meth:`write_geometry` primitive for primitive. Gun parameters are
        already in metres, so no unit conversion. Boundary tags follow the ``.geo``
        writer: the cathode plane (first segment) and the exit aperture (last
        segment) are PMC, the axis is AXI, every wall in between is PEC.
        """
        p = self.parameters
        try:
            y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, T10, L11, R12, L13, R14, x = (
                float(p[n]) for n in ('y1', 'R2', 'T2', 'L3', 'R4', 'L5', 'R6', 'L7',
                                      'R8', 'T9', 'R10', 'T10', 'L11', 'R12', 'L13',
                                      'R14', 'x'))
            R9 = self._r9(p)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None

        prof = Profile('rfgun')
        prof.start(0.0, 0.0)
        prof.line_to(0.0, y1, 'PMC')                                  # cathode plane
        z, r = 0.0, y1

        z, r = R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)
        prof.circle_arc_to(z, r, center=(-R2, y1), boundary='PEC')

        z, r = z - L3 * np.cos(T2), r + L3 * np.sin(T2)
        prof.line_to(z, r, 'PEC')

        c = (z + R4 * np.cos(T2), r + R4 * np.sin(T2))
        z, r = z - (R4 - R4 * np.cos(T2)), r + R4 * np.sin(T2)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z, r + L5
        prof.line_to(z, r, 'PEC')

        c = (z + R6, r)
        z, r = z + R6, r + R6
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z + L7, r
        prof.line_to(z, r, 'PEC')

        c = (z, r - R8)
        z, r = z + R8 * np.cos(T9), r - (R8 - R8 * np.sin(T9))
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        c = (z - R9 * np.cos(T9), r - R9 * np.sin(T9))
        z, r = z + (R9 - R9 * np.cos(T9)), r - R9 * np.sin(T9)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        c = (z - R10, r)
        z, r = z - (R10 - R10 * np.cos(T10)), r - R10 * np.sin(T10)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z - L11 * np.sin(T10), r - L11 * np.cos(T10)
        prof.line_to(z, r, 'PEC')

        c = (z + R12 * np.cos(T10), r - R12 * np.sin(T10))
        z, r = z - (R12 - R12 * np.cos(T10)), r - R12 * np.sin(T10)
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z, r - L13
        prof.line_to(z, r, 'PEC')

        c = (z + R14, r)
        z, r = z + R14, r - R14
        prof.circle_arc_to(z, r, center=c, boundary='PEC')

        z, r = z + 10 * y1, r
        prof.line_to(z, r, 'PEC')

        z, r = z, r - x
        prof.line_to(z, r, 'PMC')                                     # exit aperture
        prof.close('AXI')
        return prof

    def write_geometry(self, parameters, n_cells=None, beampipe=None, write=None, **kwargs):
        # n_cells / beampipe accepted for signature compatibility with the other
        # models, so the generic clone_for_tuning can call every write_geometry
        # the same way. A gun has one cell and no beampipe parameterisation.
        names = [
            "y1", "R2", "T2", "L3", "R4", "L5", "R6", "L7", "R8",
            "T9", "R10", "T10", "L11", "R12", "L13", "R14", "x"
        ]

        y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, \
            T10, L11, R12, L13, R14, x = (parameters[n] for n in names)


        # calcualte R9
        R9 = (((y1 + R2 * np.sin(T2) + L3 * np.cos(T2) + R4 * np.sin(T2) + L5 + R6) -
               (R14 + L13 + R12 * np.sin(T10) + L11 * np.cos(T10) + R10 * np.sin(T10) + x + R8 * (
                       1 - np.sin(T9))))) / np.sin(T9)

        step = 5 * 1e-2
        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)

        with open(write.replace('.n', '.geo'), 'w') as cav:

            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            start_pt = [0, 0]
            pt_indx = add_point(cav, start_pt, pt_indx)

            geo.append([start_pt[1], start_pt[0], 1])

            # DRAW LINE CONNECTING ARCS
            lineTo(start_pt, [0, y1], step)
            pt = [0, y1]
            geo.append([pt[1], pt[0], 0])

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # DRAW ARC:
            start_pt = pt
            center_pt = [-R2, y1]
            majax_pt = [-R2 + R2, y1]
            end_pt = [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(-R2, y1, R2, R2, pt, [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)], 0, T2, step)
            pt = [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)], step)
            pt = [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R4 * np.cos(T2), pt[1] + R4 * np.sin(T2)]
            majax_pt = [pt[0] + R4 * np.cos(T2) + R4, pt[1] + R4 * np.sin(T2)]
            end_pt = [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R4 * np.cos(T2), pt[1] + R4 * np.sin(T2), R4, R4,
            #                  pt, [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)], -(np.pi - T2), np.pi, step)
            pt = [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0], pt[1] + L5], step)
            pt = [pt[0], pt[1] + L5]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R6, pt[1]]
            majax_pt = [pt[0] + R6 + R6, pt[1]]
            end_pt = [pt[0] + R6, pt[1] + R6]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R6, pt[1], R6, R6, pt, [pt[0] + R6, pt[1] + R6], np.pi, np.pi / 2, step)
            pt = [pt[0] + R6, pt[1] + R6]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] + L7, pt[1]], step)
            pt = [pt[0] + L7, pt[1]]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0], pt[1] - R8]
            majax_pt = [pt[0] + R8, pt[1] - R8]
            end_pt = [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0], pt[1] - R8, R8, R8, pt, [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))],
            #                  np.pi / 2, T9, step)
            pt = [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] - R9 * np.cos(T9), pt[1] - R9 * np.sin(T9)]
            majax_pt = [pt[0] - R9 * np.cos(T9) + R9, pt[1] - R9 * np.sin(T9)]
            end_pt = [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] - R9 * np.cos(T9), pt[1] - R9 * np.sin(T9), R9, R9,
            #                  pt, [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)], T9, 0, step)
            pt = [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] - R10, pt[1]]
            majax_pt = [pt[0] - R10 + R10, pt[1]]
            end_pt = [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] - R10, pt[1], R10, R10,
            #                  pt, [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)], 0, -T10, step)
            pt = [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)], step)
            pt = [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R12 * np.cos(T10), pt[1] - R12 * np.sin(T10)]
            majax_pt = [pt[0] + R12 * np.cos(T10) + R12, pt[1] - R12 * np.sin(T10)]
            end_pt = [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R12 * np.cos(T10), pt[1] - R12 * np.sin(T10), R12, R12,
            #                  pt, [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)], (np.pi - T10), np.pi,
            #                  step)
            pt = [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0], pt[1] - L13], step)
            pt = [pt[0], pt[1] - L13]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # DRAW ARC:
            start_pt = pt
            center_pt = [pt[0] + R14, pt[1]]
            majax_pt = [pt[0] + R14 + R14, pt[1]]
            end_pt = [pt[0] + R14, pt[1] - R14]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            # pts = arcToTheta(pt[0] + R14, pt[1], R14, R14, pt, [pt[0] + R14, pt[1] - R14], np.pi, -np.pi / 2, step)
            pt = [pt[0] + R14, pt[1] - R14]
            # for pp in pts:
            #     if (np.around(pp, 12) != np.around(pt, 12)).all():
            #         geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # line
            lineTo(pt, [pt[0] + 10 * y1, pt[1]], step)
            pt = [pt[0] + 10 * y1, pt[1]]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 1])

            # line
            lineTo(pt, [pt[0], pt[1] - y1], step)
            pt = [pt[0], pt[1] - x]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            geo.append([pt[1], pt[0], 2])

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

        # start_pt = [0, 0]
        # geo.append([start_pt[1], start_pt[0], 1])

        # pandss
        # df = pd.DataFrame(geo)
        # # print(df[df.duplicated(keep=False)])
        #
        # geo = np.array(geo)
        # _, idx = np.unique(geo[:, 0:2], axis=0, return_index=True)
        # geo = geo[np.sort(idx)]

        # print('length of geometry:: ', len(geo))

        # top = plt.plot(geo[:, 1], geo[:, 0])
        # bottom = plt.plot(geo[:, 1], -geo[:, 0], c=top[0].get_color())

        # # plot legend wthout duplicates
        # handles, labels = plt.gca().get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # plt.legend(by_label.values(), by_label.keys())
        # plt.gca().set_aspect('equal')
        # plt.xlim(left=-30 * 1e-2)
        #
        # # # write geometry
        # # if write:
        # #     try:
        # #         df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
        # #         # change point data precision
        # #         df['r'] = df['r'].round(8)
        # #         df['z'] = df['z'].round(8)
        # #         # drop duplicates
        # #         df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
        # #         df.to_csv(write, sep='\t', index=False)
        # #     except FileNotFoundError as e:
        # #         error('Check file path:: ', e)
        #
        # return plt.gca()

    def plot(self, what, ax=None, **kwargs):
        if what.lower() == 'geometry':
            # Geometry from the Profile via the model-independent base plotter.
            return super().plot(what, ax=ax, **kwargs)

        if what.lower() == 'convergence':
            try:
                if ax:
                    self._plot_convergence(ax)
                else:
                    fig, ax = plt.subplot_mosaic([['conv', 'abs_err']], layout='constrained', figsize=(12, 4))
                    self._plot_convergence(ax)
                return ax
            except ValueError:
                info("Convergence data not available.")

    def get_eigenmode_qois(self, config=None):
        """Read the eigenmode QOIs, plus the gun's on-axis |Ez| profile.

        This used to reimplement the base reader with a monopole-only body and a
        narrower signature, so any caller passing a config (UQ does) hit a
        TypeError. Only the |Ez| profile is gun-specific.
        """
        super().get_eigenmode_qois(config)

        ez_path = os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')
        if os.path.exists(ez_path):
            with open(ez_path) as csv_file:
                self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
        self.R_Q = self.eigenmode_qois['R/Q [Ohm]']
        self.GR_Q = self.eigenmode_qois['GR/Q [Ohm^2]']
        self.G = self.GR_Q / self.R_Q
        self.Q = self.eigenmode_qois['Q []']
        self.e = self.eigenmode_qois['Epk/Eacc []']
        self.b = self.eigenmode_qois['Bpk/Eacc [mT/MV/m]']
        self.Epk_Eacc = self.e
        self.Bpk_Eacc = self.b

    def plot_axis_field(self, show_min_max=True):
        fig, ax = plt.subplots(figsize=(12, 3))
        if len(self.Ez_0_abs['z(0, 0)']) != 0:
            ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')

            ax.legend(loc="upper right")
            if show_min_max:
                minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(10000 * (maxz - minz)) / 50, width=100)
                Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                ax.plot(self.Ez_0_abs['z(0, 0)'][peaks], Ez_0_abs_peaks, marker='o', ls='')
                ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                ax.axhline(max(Ez_0_abs_peaks), c='k')
        else:
            if os.path.exists(os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')):
                with open(os.path.join(self._eigenmode_pol_dir('monopole'), 'Ez_0_abs.csv')) as csv_file:
                    self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')
                ax.plot(self.Ez_0_abs['z(0, 0)'], self.Ez_0_abs['|Ez(0, 0)|'], label='$|E_z(0,0)|$')
                ax.legend(loc="upper right")
                if show_min_max:
                    minz, maxz = min(self.Ez_0_abs['z(0, 0)']), max(self.Ez_0_abs['z(0, 0)'])
                    peaks, _ = find_peaks(self.Ez_0_abs['|Ez(0, 0)|'], distance=int(5000 * (maxz - minz)) / 100,
                                          width=100)
                    Ez_0_abs_peaks = self.Ez_0_abs['|Ez(0, 0)|'][peaks]
                    ax.axhline(min(Ez_0_abs_peaks), c='r', ls='--')
                    ax.axhline(max(Ez_0_abs_peaks), c='k')
            else:
                error('Axis field plot data not found.')

    def __str__(self):
        p = dict()
        # p[self.name] = {
        #     'tune': self.tune_results,
        #     'fm': self.eigenmode_qois,
        #     'hom': self.wakefield_qois,
        #     'uq': {
        #         'tune': 0,
        #         'fm': self.uq_fm_results,
        #         'hom': self.uq_hom_results
        #     }
        # }
        return fr"{json.dumps(p, indent=4)}"
    def rebuild(self, parameters, beampipe=None):
        """A fresh RFGun from its geometry parameter dict."""
        return RFGun({'geometry': dict(parameters)}, name=self.name)



