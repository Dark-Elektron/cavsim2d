from cavsim2d.cavity.base import Cavity
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class EllipticalCavity(Cavity):
    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k', plot_label=None):
        """
        All of the old “dimension‐based” logic has been moved here.
        Assumes self.kind, self.name, self.color, etc. are already set.
        """
        super().__init__()

        self.projectDir = None

        self.plot_label = plot_label
        self.name = name
        self.kind = 'elliptical cavity'
        self.beampipe = beampipe
        self.color = color
        self.geo_filepath = None
        self.self_dir = None

        self.n_cells = n_cells
        self.cell_parameterisation = cell_parameterisation

        # Basic counters / containers
        self.n_modes = (n_cells + 1) if n_cells is not None else None
        self.n_modules = 1
        self.no_of_modules = 1

        # Handle the special case: mid_cell can be a dict with keys OC, OC_R, IC
        if isinstance(mid_cell, dict):
            end_cell_left = mid_cell['OC']
            end_cell_right = mid_cell['OC_R']
            mid_cell = mid_cell['IC']

        self.mid_cell = np.array(mid_cell)[:7]
        if end_cell_left is not None:
            self.end_cell_left = np.array(end_cell_left)[:7]
        else:
            self.end_cell_left = np.copy(self.mid_cell)

        if end_cell_right is not None:
            self.end_cell_right = np.array(end_cell_right)[:7]
        else:
            self.end_cell_right = np.copy(self.end_cell_left)

        # Equator radius Req (index 6) is physically shared across all cells
        # of a multi-cell cavity. If the user passed mismatched values,
        # unify them toward the canonical source (mid-cell for n_cells >= 2,
        # end-cell-left for a single-cell cavity) and warn — otherwise the
        # geometry writer silently uses OC_R's Req for every tangent check,
        # producing misleading "degenerate geometry" errors.
        self._unify_equator_radius()

        # Unpack 7 parameters
        (self.A, self.B, self.a, self.b,
         self.Ri, self.L, self.Req) = self.mid_cell[:7]

        (self.A_el, self.B_el, self.a_el,
         self.b_el, self.Ri_el, self.L_el,
         self.Req_el) = self.end_cell_left[:7]

        (self.A_er, self.B_er, self.a_er,
         self.b_er, self.Ri_er, self.L_er,
         self.Req_er) = self.end_cell_right[:7]

        # Active length & cavity length
        self.l_active = (
                        2 * (self.n_cells - 1) * self.L +
                        self.L_el +
                        self.L_er
                        ) * 1e-3
        self.l_cavity = self.l_active + 8 * self.L * 1e-3

        # Build self.shape dictionary
        self.shape = {
            "IC": update_alpha(self.mid_cell[:7], self.cell_parameterisation),
            "OC": update_alpha(self.end_cell_left[:7], self.cell_parameterisation),
            "OC_R": update_alpha(self.end_cell_right[:7], self.cell_parameterisation),
            "BP": beampipe,
            "n_cells": self.n_cells,
            "CELL PARAMETERISATION": self.cell_parameterisation,
            "kind": self.kind
        }

        self.to_multicell()
        self.get_geometric_parameters()

    def create(self, n_cells=None, beampipe=None, mode=None):
        if n_cells is None:
            n_cells = self.n_cells
        if beampipe is None:
            beampipe = self.beampipe

        # In tune mode, just rewrite the geometry file with current parameters
        # (self_dir and geo_filepath are already set from spawn)
        if mode == 'tune' and self.geo_filepath:
            self.write_quarter_geometry(self.parameters, beampipe, write=self.geo_filepath)
            return
        if mode == 'tune-endcell' and self.geo_filepath:
            self.write_endcell_tune_geometry(self.parameters, beampipe, write=self.geo_filepath)
            return

        # If self_dir is already set and geometry exists, skip re-creation
        # (e.g. when spawned by optimisation with flat folder structure)
        if self.self_dir and self.geo_filepath and os.path.exists(self.geo_filepath):
            return

        if self.projectDir:
            # Create cavity directory directly inside project folder (no Cavities/ subfolder)
            self.self_dir = os.path.join(self.projectDir, self.name)
            geo_dir = os.path.join(self.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)

            self.uq_dir = os.path.join(self.self_dir, 'uq')

            self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            if mode is None:
                self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)
            elif mode == 'tune-endcell':
                self.write_endcell_tune_geometry(self.parameters, beampipe, write=self.geo_filepath)
            else:
                self.write_quarter_geometry(self.parameters, beampipe, write=self.geo_filepath)

    def _unify_equator_radius(self):
        """Force Req to be identical on mid, end-left and end-right cells.

        For multi-cell cavities the canonical source is the mid-cell; for
        single-cell cavities it is end-cell-left. A warning is emitted if
        any sibling had to be adjusted, so the user knows their input was
        inconsistent.
        """
        req_m, req_el, req_er = (float(self.mid_cell[6]),
                                 float(self.end_cell_left[6]),
                                 float(self.end_cell_right[6]))
        if np.isclose(req_m, req_el) and np.isclose(req_m, req_er):
            return

        canonical = req_m if (self.n_cells or 1) >= 2 else req_el
        source = 'mid-cell' if (self.n_cells or 1) >= 2 else 'end-cell-left'
        warning(
            f"Req differs across cells (mid={req_m}, end-left={req_el}, "
            f"end-right={req_er}). Req is physically shared across all cells — "
            f"unifying to the {source} value ({canonical}). Adjust your input "
            f"or call cav._unify_equator_radius() explicitly to silence this."
        )
        self.mid_cell[6] = canonical
        self.end_cell_left[6] = canonical
        self.end_cell_right[6] = canonical

    def write_endcell_tune_geometry(self, parameters, bp, write=None):
        """Build a 1-cell geometry for end-cell tuning: beampipe + end-cell
        half + adjacent mid-cell half, with PMC at both iris/beampipe ends.

        The adjacent half uses the current mid-cell parameters, so the
        solve returns the same pi-mode frequency the end-cell has inside
        the real multicell cavity — which is what the secant root-finder
        needs to hit the target.

        ``bp`` selects the end-cell being tuned:
        - 'left'  : end-LEFT cell tuned; end-right parameters are replaced
                    with the current mid-cell parameters for the adjacent
                    half of the geometry.
        - 'right' : mirrored — end-RIGHT cell tuned; end-left params are
                    replaced with mid-cell params.
        """
        params = dict(parameters)
        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        if bp == 'left':
            for n in names:
                params[f'{n}_er'] = params[f'{n}_m']
        elif bp == 'right':
            for n in names:
                params[f'{n}_el'] = params[f'{n}_m']
        else:
            # Should not be called with 'none'/'both'; fall back to full geom.
            pass
        self.write_geometry(params, n_cells=1, BP=bp, write=write)

    def write_geometry(self, parameters, n_cells, BP, scale=1, ax=None, bc=None, tangent_check=False,
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
        # IC, OC, OC_R = shape['IC'], shape['OC'], shape['OC_R']
        # BP = self.beampipe
        # n_cell = self.n_cells

        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.set_aspect('equal')

        names = ['A_m', 'B_m', 'a_m', 'b_m', 'Ri_m', 'L_m', 'Req_m',
                 'A_el', 'B_el', 'a_el', 'b_el', 'Ri_el', 'L_el', 'Req_el',
                 'A_er', 'B_er', 'a_er', 'b_er', 'Ri_er', 'L_er', 'Req_er']

        A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, \
            A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, \
            A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = (parameters[n]*1e-3 for n in names)

        Req = Req_m  # CHANGE THIS, Req should be same

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

        step = 0.0005

        # calculate shift
        shift = (L_bp_r + L_bp_l + L_el + (n_cells - 1) * 2 * L_m + L_er) / 2

        # calculate angles outside loop
        # CALCULATE x1_el, y1_el, x2_el, y2_el

        df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, tangent_check=tangent_check)
        x1el, y1el, x2el, y2el = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        # CALCULATE x1, y1, x2, y2
        df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, tangent_check=tangent_check)
        x1, y1, x2, y2 = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, tangent_check=tangent_check)
        x1er, y1er, x2er, y2er = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            # # define parameters
            # cav.write(f'\nA_el = DefineNumber[{A_el}, Name "Parameters/End cell 1 Equator ellipse major axis"];')
            # cav.write(f'\nB_el = DefineNumber[{B_el}, Name "Parameters/End cell 1 Equator ellipse minor axis"];')
            # cav.write(f'\na_el = DefineNumber[{a_el}, Name "Parameters/End cell 1 Iris ellipse major axis"];')
            # cav.write(f'\nb_el = DefineNumber[{b_el}, Name "Parameters/End cell 1 Iris ellipse minor axis"];')
            # cav.write(f'\nRi_el = DefineNumber[{Ri_el}, Name "Parameters/End cell 1 Iris radius"];')
            # cav.write(f'\nL_el = DefineNumber[{L_el}, Name "Parameters/End cell 1 Half cell length"];')
            # # cav.write(f'\nReq_el = DefineNumber[{Req}, Name "Parameters/End cell 1 Equator radius"];\n')
            #
            # cav.write(f'\nA_m = DefineNumber[{A_el}, Name "Parameters/Mid cell Equator ellipse major axis"];')
            # cav.write(f'\nB_m = DefineNumber[{B_el}, Name "Parameters/Mid cell Equator ellipse minor axis"];')
            # cav.write(f'\na_m = DefineNumber[{a_el}, Name "Parameters/Mid cell Iris ellipse major axis"];')
            # cav.write(f'\nb_m = DefineNumber[{b_el}, Name "Parameters/Mid cell Iris ellipse minor axis"];')
            # cav.write(f'\nRi_m = DefineNumber[{Ri_el}, Name "Parameters/Mid cell Iris radius"];')
            # cav.write(f'\nL_m = DefineNumber[{L_el}, Name "Parameters/Mid cell Half cell length"];')
            # cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Mid cell Equator radius"];\n')
            #
            # cav.write(f'\nA_er = DefineNumber[{A_el}, Name "Parameters/End cell 2 Equator ellipse major axis"];')
            # cav.write(f'\nB_er = DefineNumber[{B_el}, Name "Parameters/End cell 2 Equator ellipse minor axis"];')
            # cav.write(f'\na_er = DefineNumber[{a_el}, Name "Parameters/End cell 2 Iris ellipse major axis"];')
            # cav.write(f'\nb_er = DefineNumber[{b_el}, Name "Parameters/End cell 2 Iris ellipse minor axis"];')
            # cav.write(f'\nRi_er = DefineNumber[{Ri_el}, Name "Parameters/End cell 2 Iris radius"];')
            # cav.write(f'\nL_er = DefineNumber[{L_el}, Name "Parameters/End cell 2 Half cell length"];')
            # cav.write(f'\nReq_er = DefineNumber[{Req}, Name "Parameters/End cell 2 Equator radius"];\n')

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])
            pts = lineTo(start_point, [-shift, Ri_el], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [-shift, Ri_el]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            if bc:
                # draw left boundary condition
                ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                        [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
                        [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pts = lineTo(pt, [L_bp_l - shift, Ri_el], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0]])
                pt = [L_bp_l - shift, Ri_el]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

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

                        ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_el + b_el]
                    majax_pt = [L_bp_l - shift + a_el, Ri_el + b_el]
                    end_pt = [-shift + x1el, y1el]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                    pt = [-shift + x1el, y1el]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2el, y2el], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2el, y2el]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

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

                        ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

                    start_pt = pt
                    center_pt = [L_el + L_bp_l - shift, Req - B_el]
                    majax_pt = [L_el + L_bp_l - shift - A_el, Req - B_el]
                    end_pt = [L_bp_l + L_el - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                    pt = [L_bp_l + L_el - shift, Req]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    if n_cells == 1:
                        if L_bp_r > 0:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper

                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            if plot and dimension:
                                ax.scatter(L_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_er),
                                                                         width=2 * A_er,
                                                                         height=2 * B_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_er),
                                            xytext=(L_el + L_bp_l - shift + A_er, Req - B_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                            xytext=(L_el + L_bp_l - shift, Req - B_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n',
                                        ha='center',
                                        va='center')
                                ax.text(L_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                        va='center', ha='left', rotation=90)

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])

                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                         width=2 * a_er,
                                                                         height=2 * b_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])

                            geo.append([pt[1], pt[0], 0])

                            # calculate new shift
                            shift = shift - (L_el + L_er)
                        else:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                         width=2 * a_er,
                                                                         height=2 * b_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er - shift, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])
                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            # pt_indx += 1

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                    else:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper

                        start_pt = pt
                        center_pt = [L_el + L_bp_l - shift, Req - B_m]
                        majax_pt = [L_el + L_bp_l - shift + a_m, Req - B_m]
                        end_pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                          end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                    [L_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                        pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # STRAIGHT LINE TO NEXT POINT
                        pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                        # for pp in pts:
                        #     geo.append([pp[1], pp[0], 0])
                        pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]

                        pt_indx = add_point(cav, pt, pt_indx)
                        curve_indx = add_line(cav, pt_indx, curve_indx)
                        curve.append(curve_indx)

                        geo.append([pt[1], pt[0], 0])

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        start_pt = pt
                        center_pt = [L_el + L_m + L_bp_l - shift, Ri_m + b_m]
                        majax_pt = [L_el + L_m + L_bp_l - shift - A_m, Ri_m + b_m]
                        end_pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                          end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                    [L_bp_l + L_el + L_m - shift, Ri_m])
                        pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # calculate new shift
                        shift = shift - (L_el + L_m)
                        # ic(shift)

                elif n > 1 and n != n_cells:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_m, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_m, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift + A_m, Req - B_m]
                    end_pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_m + L_m - x2 + 2 * L_bp_l - shift, y2])
                    pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_m + L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_m + L_m + L_bp_l - shift - a_m, Ri_m + b_m]
                    end_pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_m + L_m - shift, Ri_m])
                    pt = [L_bp_l + L_m + L_m - shift, Ri_m]

                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # calculate new shift
                    shift = shift - 2 * L_m
                else:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_er, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_er, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_er]
                    majax_pt = [L_m + L_bp_l - shift + A_er, Req - B_er]
                    end_pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_er + L_bp_l - shift, Ri_er + b_er]
                    majax_pt = [L_m + L_er + L_bp_l - shift - a_er, Ri_er + b_er]
                    end_pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_m + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    if L_bp_r > 0:
                        geo.append([pt[1], pt[0], 0])
                    else:
                        geo.append([pt[1], pt[0], 1])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_r + L_bp_l + (n_cells - 1) * 2 * L_m + L_el + L_er) / 2

            if L_bp_r > 0:  # if there's a problem, check here.
                pts = lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cells - 1) * L_m + L_el + L_er - shift, Ri_er], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0], 0])
                pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 1])

            # END PATH
            pts = lineTo(pt, [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0],
                         step)  # to add beam pipe to right
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [2 * (n_cells - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
            # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
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

        # write geometry
        # if write:
        #     try:
        #         df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
        #         # change point data precision
        #         df['r'] = df['r'].round(8)
        #         df['z'] = df['z'].round(8)
        #         # drop duplicates
        #         df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
        #         df.to_csv(write, sep='\t', index=False)
        #     except FileNotFoundError as e:
        #         error('Check file path:: ', e)

        # append start point
        # geo.append([start_point[1], start_point[0], 0])

        if bc:
            # draw right boundary condition
            ax.plot([shift, shift], [-Ri_er, Ri_er],
                    [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                    [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

        # CLOSE PATH
        # lineTo(pt, start_point, step)
        # geo.append([start_point[1], start_point[0], 0])
        geo = np.array(geo)

        if plot:

            if dimension:
                top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
            else:
                # recenter asymmetric cavity to center
                shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
                if n_cells == 1:
                    shift_to_center = L_er + L_bp_r
                else:
                    shift_to_center = n_cells * L_m + L_bp_r

                top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
                bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3,
                                 c=top[0].get_color(),
                                 **kwargs)

            # plot legend without duplicates
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

        return ax

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

    def get_geometric_parameters(self):
        parameter_names = ["A", "B", "a", "b", "Ri", "L", "Req"]
        shape_keys = {"IC": 'm', "OC": 'el', "OC_R": 'er'}

        for key in shape_keys.keys():
            values = self.shape[key]
            for name, value in zip(parameter_names, values):
                self.parameters[f"{name}_{shape_keys[key]}"] = value

    def spawn(self, difference, folder):
        """Spawn candidate cavities into a flat folder structure.

        Each candidate cavity gets its own subfolder directly under `folder`:
            folder/G0_C0_P/
            folder/G0_C1_P/
            ...

        No project structure (cavities/Cavities/) is created.
        """
        from cavsim2d.cavity.cavities import Cavities

        # Create container without project directory structure
        spawn = Cavities(folder, _skip_project_init=True)
        os.makedirs(folder, exist_ok=True)

        for key, params_diff in difference.iterrows():
            # modify parameters
            mid_cells_mod = self._modify_parameters(
                ['A_m', 'B_m', 'a_m', 'b_m', 'Ri_m', 'L_m', 'Req_m'],
                params_diff,
                np.copy(self.mid_cell))
            endcell_l_mod = self._modify_parameters(
                ['A_el', 'B_el', 'a_el', 'b_el', 'Ri_el', 'L_el', 'Req_el'],
                params_diff,
                np.copy(self.end_cell_left))
            endcell_r_mod = self._modify_parameters(
                ['A_er', 'B_er', 'a_er', 'b_er', 'Ri_er', 'L_er', 'Req_er'],
                params_diff,
                np.copy(self.end_cell_right))

            name = key
            scav = EllipticalCavity(self.n_cells, mid_cells_mod, endcell_l_mod, endcell_r_mod, beampipe=self.beampipe)
            scav.name = name

            # Set self_dir directly — flat structure, no Cavities/ nesting
            scav.projectDir = folder
            scav.self_dir = os.path.join(folder, str(name))
            os.makedirs(scav.self_dir, exist_ok=True)

            # Write geometry
            geo_dir = os.path.join(scav.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)
            scav.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            scav.write_geometry(scav.parameters, scav.n_cells, scav.beampipe, write=scav.geo_filepath)

            spawn.cavities_list.append(scav)
            spawn.cavities_dict[name] = scav
            spawn.shape_space[name] = scav.shape
            spawn.shape_space_multicell[name] = scav.shape_multicell

        return spawn


    def _modify_parameters(self, columns, row, values):

        # Create a mapping from column name → index
        col_to_index = {col: i for i, col in enumerate(columns)}

        # Update only existing keys in the row
        for key, val in row.items():
            if key in col_to_index:
                values[col_to_index[key]] = val

        return values

    def clone_for_tuning(self, tuned_parameters, tuned_self_dir, beampipe=None):
        """Return a fresh EllipticalCavity living in ``tuned_self_dir``.

        ``tuned_parameters`` is the full suffixed parameter dict
        (A_m, B_m, ..., A_el, ..., A_er, ...) that the tuner has updated.
        The clone gets its own ``geometry/`` folder with a freshly written
        geodata.geo. Result caches are empty so subsequent eigenmode /
        wakefield runs write into the clone's own folders.
        """
        if beampipe is None:
            beampipe = self.beampipe

        mid = np.array([tuned_parameters[f'{n}_m'] for n in
                        ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])
        left = np.array([tuned_parameters[f'{n}_el'] for n in
                         ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])
        right = np.array([tuned_parameters[f'{n}_er'] for n in
                          ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])

        clone = EllipticalCavity(
            n_cells=self.n_cells,
            mid_cell=mid,
            end_cell_left=left,
            end_cell_right=right,
            beampipe=beampipe,
            name=self.name,
            cell_parameterisation=self.cell_parameterisation,
            color=self.color,
            plot_label=self.plot_label,
        )
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_self_dir)

        geo_dir = os.path.join(clone.self_dir, 'geometry')
        os.makedirs(geo_dir, exist_ok=True)
        clone.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
        clone.write_geometry(clone.parameters, clone.n_cells, clone.beampipe,
                             write=clone.geo_filepath)

        clone.uq_dir = os.path.join(clone.self_dir, 'uq')

        return clone

    def _load_tuned_from_disk(self, tuned_dir):
        """Rebuild the tuned EllipticalCavity from a persisted ``tuned/`` folder.

        Reads tuned parameters from ``tune_info/tune_res.json`` (if present)
        and rebuilds the cavity. Falls back to the current cavity's shape
        if tune_res.json is missing. The file is keyed by cell type — the
        final stage's parameter snapshot is the cumulative tuned geometry.
        """
        from cavsim2d.processes.tune import last_stage_result

        tuned_dir = Path(tuned_dir)
        tune_res_path = tuned_dir / 'tune_info' / 'tune_res.json'

        tune_res = None
        if tune_res_path.exists():
            with open(tune_res_path, 'r') as f:
                tune_res = json.load(f)
            last = last_stage_result(tune_res)
            tuned_params = (last or {}).get('parameters', {}) or {}
        else:
            tuned_params = {}

        # Fall back to the source cavity's own parameters for any suffix that
        # wasn't touched by this tune run (e.g. single-cell tunes only produce
        # `_m` keys; end-cell-r may be absent when not tuned separately).
        full_params = dict(self.parameters)
        full_params.update(tuned_params)

        mid = np.array([full_params[f'{n}_m'] for n in
                        ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])
        left = np.array([full_params[f'{n}_el'] for n in
                         ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])
        right = np.array([full_params[f'{n}_er'] for n in
                          ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']])

        clone = EllipticalCavity(
            n_cells=self.n_cells,
            mid_cell=mid,
            end_cell_left=left,
            end_cell_right=right,
            beampipe=self.beampipe,
            name=self.name,
            cell_parameterisation=self.cell_parameterisation,
            color=self.color,
            plot_label=self.plot_label,
        )
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_dir)
        geo_filepath = tuned_dir / 'geometry' / 'geodata.geo'
        if geo_filepath.exists():
            clone.geo_filepath = str(geo_filepath)
        clone.uq_dir = str(tuned_dir / 'uq')

        if tune_res is not None:
            clone.tune_results = tune_res
            last = last_stage_result(tune_res)
            if last is not None and 'FREQ' in last:
                clone.freq = last['FREQ']

        return clone


