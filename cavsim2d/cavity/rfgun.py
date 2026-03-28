from cavsim2d.cavity.base import Cavity
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
        # self.shape_space = {
        #     'IC': [L, Req, Ri, S, L_bp],
        #     'BP': beampipe
        # }
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
            cav_dir_structure = {
                self.name: {
                    'geometry': None,
                }
            }

            if os.path.exists(os.path.join(self.projectDir, 'Cavities')):
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))
            else:
                os.mkdir(os.path.join(self.projectDir, 'Cavities'))
                make_dirs_from_dict(cav_dir_structure, os.path.join(self.projectDir, 'Cavities'))

            # write geometry file to folder
            self.self_dir = os.path.join(self.projectDir, 'Cavities', self.name)

            # define different paths for easier reference later
            self.eigenmode_dir = os.path.join(self.self_dir, 'eigenmode')
            self.wakefield_dir = os.path.join(self.self_dir, 'wakefield')
            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(self.self_dir, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters,
                                write=self.geo_filepath)

    def get_geometric_parameters(self):
        self.parameters = self.shape['geometry']

    def write_geometry(self, parameters, write=None):
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
            ax = self.write_geometry(self.parameters)
            ax.set_xlabel('$z$ [m]')
            ax.set_ylabel(r"$r$ [m]")
            return ax

        if what.lower() == 'zl':
            if ax:
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Long'].get_data('Longitudinal Impedance Magnitude')
                ax.plot(x * 1e3, y)

            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\parallel} ~[\mathrm{k\Omega}]$")
            return ax
        if what.lower() == 'zt':
            if ax:
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            else:
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.margins(x=0)
                x, y, _ = self.abci_data['Trans'].get_data('Transversal Impedance Magnitude')
                ax.plot(x * 1e3, y)
            ax.set_xlabel('f [MHz]')
            ax.set_ylabel(r"$Z_{\perp} ~[\mathrm{k\Omega/m}]$")
            return ax

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

    def get_eigenmode_qois(self):
        """
        Get quantities of interest written by the SLANS code
        Returns
        -------

        """
        # print('it is here')
        qois = 'qois.json'
        assert os.path.exists(
            os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)), (
            error('Eigenmode result does not exist, please run eigenmode simulation.'))
        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', qois)) as json_file:
            self.eigenmode_qois = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'qois_all_modes.json')) as json_file:
            self.eigenmode_qois_all_modes = json.load(json_file)

        with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
            self.Ez_0_abs = pd.read_csv(csv_file, sep='\t')

        self.freq = self.eigenmode_qois['freq [MHz]']
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
            if os.path.exists(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')):
                with open(os.path.join(self.self_dir, 'eigenmode', 'monopole', 'Ez_0_abs.csv')) as csv_file:
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


