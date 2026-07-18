import json
from pathlib import Path
from cavsim2d.models.base import Cavity
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from cavsim2d.study import Cavities
from cavsim2d.geometry import Profile
from cavsim2d.geometry.tangency import tangent_coords
from cavsim2d.geometry.contours import (elliptical_profile_from_half_cells, half_cell_sequence,
                                        continuity_violations, DegenerateGeometry)
from cavsim2d.geometry.writers.multipac import writeCavityForMultipac, writeCavityForMultipac_multicell

class EllipticalCavity(Cavity):
    """Multi-cell elliptical RF cavity.

    Each half-cell is parameterised by ``[A, B, a, b, Ri, L, Req]`` (mm). See the
    :meth:`__init__` docstring for the labelled geometry sketch — i.e.
    ``help(EllipticalCavity)`` prints the parameterisation.
    """
    uses_cell_suffixes = True
    def __init__(self, n_cells=None, mid_cell=None, end_cell_left=None,
                 end_cell_right=None, beampipe='none', name='cavity',
                 cell_parameterisation='simplecell', color='k', plot_label=None):
        r"""A multi-cell elliptical cavity.

        Geometry / parameterisation
        ---------------------------
        A half-cell is one quarter of the meridian outline: an **iris** ellipse
        near the aperture, joined by a straight wall segment to an **equator**
        ellipse at the outer wall. The seven parameters ``[A, B, a, b, Ri, L,
        Req]`` (all in **mm**) fix it::

            r
            ^                        equator ellipse
          Req |- - - - - - - - - .--''''''--.   (z semi-axis A, r semi-axis B)
              |                ,-'            \
              |              /                 |
              |            /  <- wall, angle alpha to the z-axis
              |          ,'                    |
              |    _..--'   iris ellipse       |
           Ri |--''         (z semi-axis a,    |
              |              r semi-axis b)     |
              +----------------------------------------> z
              |<--------------- L -------------->|
                       (half-cell length; full cell = 2 L)

        The full cell mirrors this quarter about the equator plane (z = L) and
        the axis (r = 0); a multi-cell cavity chains ``n_cells`` of them. Only the
        upper half (r >= 0) is the analysed axisymmetric domain — that is what
        ``cav.plot('geometry')`` draws. See the figure ``docs/images/cavity.jpg``
        and the :doc:`Eigenmode <eigenmode>` guide for the labelled parameters.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        mid_cell, end_cell_left, end_cell_right : sequence of floats, **mm**
            The seven half-cell parameters ``[A, B, a, b, Ri, L, Req]`` (an
            optional 8th slot ``alpha`` is accepted and ignored on input):

            - ``A``, ``B`` : semi-axes (z, r) of the **equator** ellipse.
            - ``a``, ``b`` : semi-axes (z, r) of the **iris** ellipse.
            - ``Ri``       : iris (aperture / beam-pipe) radius.
            - ``L``        : half-cell length (iris plane to equator plane); the
              full cell length is ``2*L``.
            - ``Req``      : equator radius.
            - ``alpha``    : wall inclination angle [deg], derived from the above
              (the optional 8th slot); it is a computed result, not an input.

            ``mid_cell`` sets the interior cells; ``end_cell_left`` /
            ``end_cell_right`` set the two ends (default to ``mid_cell``). A
            dict ``{'IC': mid, 'OC': left, 'OC_R': right}`` is also accepted.
        beampipe : {'none', 'left', 'right', 'both'}
            Which ends carry a beam pipe.
        name : str
            Cavity name (used for its output folder).
        color, plot_label : optional
            Styling for overlay plots.

        Examples
        --------
        >>> tesla = [42, 42, 12, 19, 35, 57.7, 103.353]
        >>> EllipticalCavity(9, tesla, tesla, tesla, beampipe='both')
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
        # (self_dir and geo_filepath are already set from spawn).
        # Scratch quarter/end-cell geometries during tuning intentionally
        # don't update the snapshot — the snapshot reflects the *full*
        # multicell geometry the user's parameters correspond to.
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

            if getattr(self, '_half_cells', None) is not None:
                # Independently varying cells: only the native Profile path can express
                # them. Write no .geo — write_geometry would emit a uniform-mid-cell
                # contour, and a silent fallback to it would give wrong physics. With
                # geo_filepath None, any fallback fails loudly instead.
                self.geo_filepath = None
                self._write_geometry_snapshot()
                return

            self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            if mode is None:
                self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)
            elif mode == 'tune-endcell':
                self.write_endcell_tune_geometry(self.parameters, beampipe, write=self.geo_filepath)
            else:
                self.write_quarter_geometry(self.parameters, beampipe, write=self.geo_filepath)
            self._write_geometry_snapshot()

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

            # Req is physically shared across all cells — pre-align the three
            # cells' index-6 value to the canonical source (mid-cell for
            # multi-cell cavities, end-cell-left for single-cell) so the
            # EllipticalCavity constructor's _unify_equator_radius() doesn't
            # warn every time optimisation mutates only one cell's Req.
            canonical_req = mid_cells_mod[6] if (self.n_cells or 1) >= 2 else endcell_l_mod[6]
            mid_cells_mod[6] = canonical_req
            endcell_l_mod[6] = canonical_req
            endcell_r_mod[6] = canonical_req

            name = key
            scav = EllipticalCavity(self.n_cells, mid_cells_mod, endcell_l_mod, endcell_r_mod, beampipe=self.beampipe)
            scav.name = name

            # Set self_dir directly — flat structure, no Cavities/ nesting
            scav.projectDir = folder
            scav.self_dir = os.path.join(folder, str(name))
            scav.uq_dir = os.path.join(scav.self_dir, 'uq')
            os.makedirs(scav.self_dir, exist_ok=True)

            # Write geometry
            geo_dir = os.path.join(scav.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)
            scav.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            scav.write_geometry(scav.parameters, scav.n_cells, scav.beampipe, write=scav.geo_filepath)
            scav._write_geometry_snapshot()

            spawn.cavities_list.append(scav)
            spawn.cavities_dict[name] = scav
            spawn.shape_space[name] = scav.shape
            spawn.shape_space_multicell[name] = scav.shape_multicell

        return spawn


    def spawn_half_cells(self, perturbed, folder):
        """Container of cavities carrying explicit per-half-cell geometry.

        The multicell counterpart of :meth:`spawn`. ``perturbed`` maps a name to a
        ``(2 * n_cells, 7)`` half-cell array. No ``.geo`` is written: only the native
        :class:`~cavsim2d.geometry.Profile` path can express independently varying
        cells, and ``write_geometry`` would silently emit a uniform-mid-cell contour.
        """
        spawn = Cavities(folder, _skip_project_init=True)
        os.makedirs(folder, exist_ok=True)

        for name, half_cells in perturbed.items():
            scav = EllipticalCavity(self.n_cells, self.mid_cell, self.end_cell_left,
                                    self.end_cell_right, beampipe=self.beampipe)
            scav.name = name
            scav.set_half_cells(half_cells)

            scav.projectDir = folder
            scav.self_dir = os.path.join(folder, str(name))
            scav.uq_dir = os.path.join(scav.self_dir, 'uq')
            os.makedirs(scav.self_dir, exist_ok=True)
            os.makedirs(os.path.join(scav.self_dir, 'geometry'), exist_ok=True)
            scav.geo_filepath = None            # native profile() only

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

    HALF_CELL_VARS = ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')

    def half_cells(self):
        """The ``(2 * n_cells, 7)`` half-cell parameter array, in **mm**.

        This is the canonical per-cell representation: row ``2k`` is the forward
        (left) half of cell ``k+1``, row ``2k+1`` its backward (right) half, each
        carrying its own ``(A, B, a, b, Ri, L, Req)``. Adjacent halves share an
        equator ``Req`` within a cell and an iris ``Ri`` across the cell boundary
        (see :func:`~cavsim2d.geometry.contours.continuity_violations`).

        Derived from ``self.parameters`` (all mid-cells identical) unless
        :meth:`set_half_cells` has installed an explicit, independently varying set.
        """
        if getattr(self, '_half_cells', None) is not None:
            return np.array(self._half_cells, dtype=float)
        cells = {suf: [float(self.parameters[f'{n}_{suf}']) for n in self.HALF_CELL_VARS]
                 for suf in ('m', 'el', 'er')}
        halves = half_cell_sequence(cells['m'], cells['el'], cells['er'], self.n_cells)
        for h in halves:
            h[6] = cells['m'][6]                     # writers force Req = Req_m
        return np.array(halves, dtype=float)

    def set_half_cells(self, half_cells):
        """Install an explicit per-half-cell parameter array (mm), letting every
        cell vary independently.

        Overrides ``self.parameters`` as the geometry source for :meth:`profile`.
        The array must satisfy the equator/iris continuity constraints; pass
        ``None`` to revert to the parameter-derived (uniform mid-cell) geometry.
        """
        if half_cells is None:
            self._half_cells = None
            self.geo_filepath = None            # let create() rewrite the uniform .geo
            return self
        arr = np.asarray(half_cells, dtype=float)
        expected = (2 * self.n_cells, len(self.HALF_CELL_VARS))
        if arr.shape != expected:
            raise ValueError(f'half_cells must have shape {expected}, got {arr.shape}')
        bad = continuity_violations(arr)
        if bad:
            raise ValueError('half-cell geometry is discontinuous:\n  ' + '\n  '.join(bad))
        self._half_cells = arr
        # Any .geo on disk describes the uniform-mid-cell contour and no longer
        # matches this geometry. Drop the reference so a fallback cannot silently
        # solve the wrong cavity (see NGSolveMEVP._build_mesh).
        self.geo_filepath = None
        return self

    def _cell_length_m(self):
        """Mid-cell axial length (iris to iris), metres: twice the half-cell L."""
        return 2 * self.parameters['L_m'] * 1e-3

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` (metres) — the native
        netgen.occ geometry path, with *exact* ellipse arcs.

        Covers every elliptical geometry: single-cell and multicell, symmetric and
        asymmetric end cells, any beampipe configuration, and (via
        :meth:`set_half_cells`) independently varying cells. Returns ``None`` only
        for degenerate parameter sets, so the solver falls back to the gmsh ``.geo``
        writer, which reports the failure.

        Reads ``self.parameters`` rather than ``self.mid_cell``: that dict is the
        live source of truth ``write_geometry`` uses, and the tuner mutates it in
        place while searching.
        """
        try:
            halves = self.half_cells() * 1e-3
            L_bp = 4 * float(self.parameters['L_m']) * 1e-3
        except (KeyError, TypeError, ValueError):
            return None
        try:
            return elliptical_profile_from_half_cells(halves, self.beampipe, L_bp,
                                                      name='elliptical')
        except (DegenerateGeometry, ValueError):
            return None

    def export_multipac(self, file_path, plot=False, multicell=False):
        """Write this cavity's contour in the Multipac input format.

        ``file_path`` is the geometry *file* to write (not a directory). Set
        ``multicell=True`` for a per-cell (non-uniform) parameterisation.
        """
        writer = writeCavityForMultipac_multicell if multicell else writeCavityForMultipac
        writer(file_path, self.n_cells, self.mid_cell,
               end_cell_left=self.end_cell_left, end_cell_right=self.end_cell_right,
               beampipe=self.beampipe, plot=plot)
        return file_path
    
    def rebuild(self, parameters, beampipe=None):
        """A fresh EllipticalCavity from a suffixed parameter dict (A_m, ..., Req_er)."""
        names = ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')
        cells = [np.array([float(parameters[f'{n}_{suf}']) for n in names])
                 for suf in ('m', 'el', 'er')]
        return EllipticalCavity(
            n_cells=self.n_cells,
            mid_cell=cells[0], end_cell_left=cells[1], end_cell_right=cells[2],
            beampipe=self.beampipe if beampipe is None else beampipe,
            name=self.name,
            cell_parameterisation=self.cell_parameterisation,
            color=self.color,
            plot_label=self.plot_label,
        )





