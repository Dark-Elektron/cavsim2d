from cavsim2d.cavity.base import Cavity
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class EllipticalCavityFlatTop(Cavity):
    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 color='k', plot_label=None):
        """
        All of the old “dimension‐based” logic has been moved here.
        Assumes self.kind, self.name, self.color, etc. are already set.
        """
        super().__init__()

        self.projectDir = None
        self.plot_label = plot_label
        self.name = name
        self.kind = 'elliptical cavity flat top'
        self.beampipe = beampipe
        self.color = color

        self.n_cells = n_cells
        self.cell_parameterisation = 'flattop'

        # Basic counters / containers
        self.n_modes = (n_cells + 1) if n_cells is not None else None
        self.n_modules = 1
        self.no_of_modules = 1

        # Handle the special case: mid_cell can be a dict with keys OC, OC_R, IC
        if isinstance(mid_cell, dict):
            end_cell_left = mid_cell['OC']
            end_cell_right = mid_cell['OC_R']
            mid_cell = mid_cell['IC']

        # Must have at least length = 8 in each cell‐parameter list
        assert mid_cell is not None and len(mid_cell) > 7, \
            ValueError(
                "Flattop cavity mid‐cells require at least 8 input parameters, with the 8th representing length (l).")
        if end_cell_left is not None:
            assert len(end_cell_left) > 7, \
                ValueError(
                    "Flattop cavity left end‐cells require at least 8 input parameters, with the 8th representing "
                    "length (l).")
        if end_cell_right is not None:
            assert len(end_cell_right) > 7, \
                ValueError(
                    "Flattop cavity right end‐cells require at least 8 input parameters, with the 8th representing "
                    "length (l).")

        # Truncate or pad to exactly 8 elements
        self.mid_cell = np.array(mid_cell)[:8]
        self.end_cell_left = np.array(end_cell_left)[:8] if end_cell_left is not None else self.mid_cell
        self.end_cell_right = np.array(end_cell_right)[:8] if end_cell_right is not None else self.mid_cell

        # Ensure end_cell_left / end_cell_right exist
        if end_cell_left is None:
            self.end_cell_left = self.mid_cell
        if end_cell_right is None:
            self.end_cell_right = self.end_cell_left

        # Unpack
        (self.A, self.B, self.a, self.b,
         self.Ri, self.L, self.Req, self.l) = self.mid_cell[:8]

        (self.A_el, self.B_el, self.a_el, self.b_el,
         self.Ri_el, self.L_el, self.Req_el, self.l_el) = self.end_cell_left[:8]

        (self.A_er, self.B_er, self.a_er, self.b_er,
         self.Ri_er, self.L_er, self.Req_er, self.l_er) = self.end_cell_right[:8]

        # Active length & cavity length
        self.l_active = (
                                2 * (self.n_cells - 1) * self.L +
                                (self.n_cells - 2) * self.l +
                                self.L_el + self.l_el +
                                self.L_er + self.l_er
                        ) * 1e-3
        self.l_cavity = self.l_active + 8 * (self.L + self.l) * 1e-3

        # Build self.shape dictionary
        self.shape = {
            "IC": update_alpha(self.mid_cell[:8], self.cell_parameterisation),
            "OC": update_alpha(self.end_cell_left[:8], self.cell_parameterisation),
            "OC_R": update_alpha(self.end_cell_right[:8], self.cell_parameterisation),
            "BP": beampipe,
            "n_cells": self.n_cells,
            "CELL PARAMETERISATION": self.cell_parameterisation,
            "kind": self.kind,
            "geo_file": None
        }

        # Produce multicell representation
        self.to_multicell()
        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, tune=None):
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
            self.geo_filepath = os.path.join(self.projectDir, 'Cavities', self.name, 'geometry', 'geodata.geo')
            self.write_geometry(self.parameters, n_cells, beampipe,
                                write=self.geo_filepath)

    def write_geometry(self, parameters, n_cells, BP, scale=1, ax=None, bc=None, tangent_check=False,
                       ignore_degenerate=False, plot=False, write=None, dimension=False,
                       contour=False, **kwargs):
        """
        Write cavity geometry

        Parameters
        ----------
        BP
        OC_R
        OC
        ignore_degenerate
        file_path: str
            File path to write geometry to
        n_cell: int
            Number of cavity cells
        mid_cell: list, ndarray
            Array of cavity middle cells' geometric parameters
        end_cell_left: list, ndarray
            Array of cavity left end cell's geometric parameters
        end_cell_right: list, ndarray
            Array of cavity left end cell's geometric parameters
        beampipe: str {"left", "right", "both", "none"}
            Specify if beam pipe is on one or both ends or at no end at all
        plot: bool
            If True, the cavity geometry is plotted for viewing

        Returns
        -------

        """

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_aspect('equal')

        names = ['A_m', 'B_m', 'a_m', 'b_m', 'Ri_m', 'L_m', 'Req_m', 'l_m',
                 'A_el', 'B_el', 'a_el', 'b_el', 'Ri_el', 'L_el', 'Req_el', 'l_el',
                 'A_er', 'B_er', 'a_er', 'b_er', 'Ri_er', 'L_er', 'Req_er', 'l_er']

        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, l, \
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, l_el, \
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er, l_er = (parameters[n]*1e-3 for n in names)
        Req = Req_m

        step = 0.005

        L_bp = 4 * L_m
        if dimension or contour:
            L_bp = 1 * L_m

        if BP.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif BP.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif BP.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        # calculate shift
        shift = (L_bp_r + L_bp_l + L_el + (n_cells - 1) * 2 * L_m + L_er + (n_cells - 2) * l + l_el + l_er) / 2

        # calculate angles outside loop
        # CALCULATE x1_el, y1_el, x2_el, y2_el
        df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, l_el / 2, tangent_check=tangent_check)
        x1el, y1el, x2el, y2el = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1, y1, x2, y2
        df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, l / 2, tangent_check=tangent_check)
        x1, y1, x2, y2 = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1_er, y1_er, x2_er, y2_er
        df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, l_er / 2, tangent_check=tangent_check)
        x1er, y1er, x2er, y2er = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        geo = []

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        geo.append([start_point[1], start_point[0], 3])

        lineTo(start_point, [-shift, Ri_el], step)
        pt = [-shift, Ri_el]
        geo.append([pt[1], pt[0], 2])

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            lineTo(pt, [L_bp_l - shift, Ri_el], step)
            pt = [L_bp_l - shift, Ri_el]

            geo.append([pt[1], pt[0], 2])

        for n in range(1, n_cells + 1):
            if n == 1:
                # DRAW ARC:
                if plot and dimension:
                    ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                    ellipse = plt.matplotlib.patches.Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                                             height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                                             facecolor='none')
                    ax.add_patch(ellipse)
                    ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                                xytext=(L_bp_l - shift, Ri_el + b_el),
                                arrowprops=dict(arrowstyle='->', color='black'))
                    ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                                xytext=(L_bp_l - shift, Ri_el + b_el),
                                arrowprops=dict(arrowstyle='->', color='black'))

                    ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', va='center', ha='center')
                    ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                            va='center', ha='center', rotation=90)

                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                if plot and dimension:
                    ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                    ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                             height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                             facecolor='none')
                    ax.add_patch(ellipse)
                    ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                                xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                                arrowprops=dict(arrowstyle='<-', color='black'))
                    ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                xytext=(L_el + L_bp_l - shift, Req - B_el),
                                arrowprops=dict(arrowstyle='->', color='black'))

                    ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', va='center',
                            ha='center')
                    ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                            va='center', ha='center', rotation=90)

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_el + l_el - shift, Req], step)
                pt = [L_bp_l + L_el + l_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                if plot and dimension:
                    ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                    # Plot the straight line
                    line_start = [L_bp_l + L_el - shift, Req]
                    line_end = pt
                    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r', zorder=200)

                    ax.annotate('', xy=(line_start[0], line_start[1] + 0.5), xytext=(line_end[0], line_end[1] + 0.5),
                                arrowprops=dict(arrowstyle='<->', color='black'))
                    ax.text((line_start[0] + line_end[0]) / 2, line_start[1] + 0.7,
                            f'{round(l_el, 2)}\n', va='center', ha='center')

                if n_cells == 1:
                    if L_bp_r > 0:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + L_bp_l + l_el - shift, Req - B_er, A_er, B_er, step, pt,
                                    [L_el + l_el + L_er - x2er + + L_bp_l + L_bp_r - shift, y2er])
                        pt = [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse((L_el + l_el + L_bp_l - shift, Req - B_er),
                                                                     width=2 * A_er,
                                                                     height=2 * B_er, angle=0, edgecolor='gray',
                                                                     ls='--',
                                                                     facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_bp_l - shift, Req - B_er),
                                        xytext=(L_el + l_el + L_bp_l - shift + A_er, Req - B_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_bp_l - shift, Req),
                                        xytext=(L_el + l_el + L_bp_l - shift, Req - B_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                    va='center', ha='left', rotation=90)

                        # STRAIGHT LINE TO NEXT POINT
                        lineTo(pt, [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                        pt = [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                        geo.append([pt[1], pt[0], 2])

                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse(
                                (L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                width=2 * a_er,
                                height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er),
                                    f'{round(a_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                    va='center', ha='center', rotation=90)

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])
                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])

                        geo.append([pt[1], pt[0], 2])

                        # calculate new shift
                        shift = shift - (L_el + l_el + L_er)
                    else:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        pts = arcTo(L_el + L_bp_l + l_el - shift, Req - B_er, A_er, B_er, step, pt,
                                    [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                        pt = [L_el + l_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        # STRAIGHT LINE TO NEXT POINT
                        lineTo(pt, [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                        pt = [L_el + l_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                        geo.append([pt[1], pt[0], 2])

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        if plot and dimension:
                            ax.scatter(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                            ellipse = plt.matplotlib.patches.Ellipse(
                                (L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                width=2 * a_er,
                                height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                facecolor='none')
                            ax.add_patch(ellipse)
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='<-', color='black'))
                            ax.annotate('', xy=(L_el + l_el + L_er + L_bp_l - shift, Ri_er),
                                        xytext=(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                        arrowprops=dict(arrowstyle='->', color='black'))

                            ax.text(L_el + l_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er),
                                    f'{round(a_er, 2)}\n',
                                    va='center', ha='center')
                            ax.text(L_el + l_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                    va='center', ha='center', rotation=90)

                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])

                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])

                        pts = arcTo(L_el + l_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                    [L_bp_l + L_el + l_el + L_er - shift, Ri_er])
                        pt = [L_bp_l + L_el + l_el + L_er - shift, Ri_er]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 2])
                        geo.append([pt[1], pt[0], 2])
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_bp_l + L_el + l_el - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_el + l_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                    pt = [L_el + l_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + l_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    pt = [L_el + l_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + l_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_el + l_el + L_m - shift, Ri_m])
                    pt = [L_bp_l + L_el + l_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # calculate new shift
                    shift = shift - (L_el + L_m + l_el)
                    # ic(shift)

            elif n > 1 and n != n_cells:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                for pp in pts:
                    geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + l - shift, Req], step)
                pt = [L_bp_l + L_m + l - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l + l - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_m + L_m + l - x2 + 2 * L_bp_l - shift, y2])
                pt = [L_m + L_m + l - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])

                geo.append([pt[1], pt[0], 2])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m + l - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_m + L_m + l - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + l + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_m + L_m + l - shift, Ri_m])
                pt = [L_bp_l + L_m + L_m + l - shift, Ri_m]

                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # calculate new shift
                shift = shift - 2 * L_m - l
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + l_er - shift, Req], step)
                pt = [L_bp_l + L_m + l_er - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + l_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                            [L_m + L_er + l_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                pt = [L_m + L_er + l_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er + l_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                pt = [L_m + L_er + l_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + l_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er + l_er - shift, Ri_er])
                pt = [L_bp_l + L_m + L_er + l_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

        # BEAM PIPE
        # reset shift

        shift = (L_bp_r + L_bp_l + L_el + l_el + (n_cells - 1) * 2 * L_m + (n_cells - 2) * l + L_er + l_er) / 2
        pts = lineTo(pt, [
            L_bp_r + L_bp_l + 2 * (n_cells - 1) * L_m + (n_cells - 2) * l + l_el + l_er + L_el + L_er - shift,
            Ri_er], step)

        if L_bp_r != 0:
            pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cells - 2) * l + l_el + l_er - shift,
                  Ri_er]
            for pp in pts:
                geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

        # END PATH
        pts = lineTo(pt, [
            2 * (n_cells - 1) * L_m + L_el + L_er + (n_cells - 2) * l + l_el + l_er + L_bp_l + L_bp_r - shift, 0],
                     step)  # to add beam pipe to right
        pt = [2 * (n_cells - 1) * L_m + L_el + L_er + (n_cells - 2) * l + l_el + l_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cells * L_er + L_bp_l - shift, 0], step)
        geo.append([pt[1], pt[0], 2])

        # CLOSE PATH
        lineTo(pt, start_point, step)
        geo.append([pt[1], pt[0], 3])

        # # write geometry
        # if write:
        #     try:
        #         df = pd.DataFrame(geo, columns=['r', 'z'])
        #         df.to_csv(write, sep='\t', index=False)
        #     except FileNotFoundError as e:
        #         error('Check file path:: ', e)

        # append start point
        geo.append([start_point[1], start_point[0], 3])

        if bc:
            # draw right boundary condition
            ax.plot([shift, shift], [-Ri_er, Ri_er],
                    [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                    [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

        # CLOSE PATH
        # lineTo(pt, start_point, step)
        # geo.append([start_point[1], start_point[0], 3])

        geo = np.array(geo)

        if plot:
            if dimension:
                top = ax.plot(geo[:, 1], geo[:, 0], **kwargs)
            else:
                # recenter asymmetric cavity to center
                shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
                if n_cells == 1:
                    shift_to_center = L_er + L_bp_r
                else:
                    shift_to_center = n_cells * L_m + L_bp_r

                top = ax.plot(geo[:, 1] - shift_left + shift_to_center, geo[:, 0], **kwargs)
                # bottom = ax.plot(geo[:, 1] - shift_left + shift_to_center, -geo[:, 0], c=top[0].get_color(), **kwargs)

            # plot legend wthout duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            return ax


    def get_geometric_parameters(self):
        parameter_names = ["A", "B", "a", "b", "Ri", "L", "Req", "l"]
        shape_keys = {"IC": 'm', "OC": 'el', "OC_R": 'er'}

        for key in shape_keys.keys():
            values = self.shape[key]
            for name, value in zip(parameter_names, values):
                self.parameters[f"{name}_{shape_keys[key]}"] = value


