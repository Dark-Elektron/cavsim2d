from abc import ABC, abstractmethod
from cavsim2d.models.base import Cavity
from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from cavsim2d.geometry import Profile

class Pillbox(Cavity):
    def __init__(self, n_cells, dims, beampipe='none'):
        """A right-cylinder (pillbox) cavity, optionally multi-cell.

        Parameters
        ----------
        n_cells : int
            Number of cells (barrels).
        dims : sequence of 5 floats ``[L, Req, Ri, S, L_bp]``, all in **mm**
            - ``L``    : cell (barrel) length along the axis.
            - ``Req``  : cavity radius (the barrel / equator radius).
            - ``Ri``   : iris/aperture radius, i.e. the beam-pipe radius.
            - ``S``    : inter-cell drift length — the straight gap between
              adjacent cells at radius ``Ri`` (0 for a single cell, or to butt
              cells directly together).
            - ``L_bp`` : beam-pipe length added at each end selected by
              ``beampipe``.
        beampipe : {'none', 'left', 'right', 'both'}
            Which ends carry a beam pipe of length ``L_bp``.

        Examples
        --------
        >>> Pillbox(1, [100, 100, 20, 0, 50], beampipe='both')  # L=Req=100 mm
        """
        super().__init__(n_cells, beampipe)

        L, Req, Ri, S, L_bp = dims
        self.n_cells = n_cells
        self.n_modes = n_cells + 1
        self.kind = 'pillbox'
        self.n_modules = 1  # change later to enable module run
        self.L = L
        self.Req = Req
        self.Ri = Ri
        self.S = S
        self.L_bp = L_bp
        self.beampipe = beampipe
        self.bc = 33
        self.cell_parameterisation = 'simplecell'

        self.shape = {
            'IC': [L, Req, Ri, S, L_bp],
            'BP': beampipe
        }
        self.shape_multicell = None

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
            self.write_geometry(self.parameters, n_cells, beampipe, write=self.geo_filepath)
            self._write_geometry_snapshot()

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` (metres) — the native
        netgen.occ geometry path, no ``.geo`` round-trip.

        Reads ``self.parameters`` (the live dict ``write_geometry`` uses and the
        tuner mutates in place), never the values captured at ``__init__``.
        """

        names = ('L', 'Req', 'Ri', 'S', 'L_bp')
        try:
            L, Req, Ri, S, L_bp = (float(self.parameters[n]) * 1e-3 for n in names)
        except (KeyError, TypeError, ValueError):
            return None

        n = self.n_cells
        bp = self.beampipe.lower()
        L_bp_l = L_bp if bp in ('both', 'left') else 0.0
        L_bp_r = L_bp if bp in ('both', 'right') else 0.0
        shift = (L_bp_l + L_bp_r + n * L + (n - 1) * S) / 2.0

        p = Profile('pillbox')
        z = -shift
        p.start(z, 0.0)
        p.line_to(z, Ri, 'PMC')                  # left aperture
        if L_bp_l > 0:
            z += L_bp_l
            p.line_to(z, Ri, 'PEC')              # left beampipe
        for cell in range(1, n + 1):
            if cell > 1:                         # inter-cell drift of length S
                z += S
                p.line_to(z, Ri, 'PEC')
            p.line_to(z, Req, 'PEC')             # up the end plate
            z += L
            p.line_to(z, Req, 'PEC')             # along the barrel
            p.line_to(z, Ri, 'PEC')              # down the end plate
        if L_bp_r > 0:
            z += L_bp_r
            p.line_to(z, Ri, 'PEC')              # right beampipe
        p.line_to(z, 0.0, 'PMC')                 # right aperture
        p.close('AXI')
        return p

    def write_geometry(self, parameters, n_cells, beampipe='none', write=None, plot=False, **kwargs):
        """

        Parameters
        ----------
        file_path
        n_cell
        cell_par
        beampipe
        plot
        kwargs

        Returns
        -------

        """

        names = ['L', 'Req', 'Ri', 'S', 'L_bp']
        L, Req, Ri, S, L_bp = (parameters[n]*1e-3 for n in names)

        step = 0.001

        if beampipe.lower() == 'both':
            L_bp_l = L_bp
            L_bp_r = L_bp
        elif beampipe.lower() == 'none':
            L_bp_l = 0.000
            L_bp_r = 0.000
        elif beampipe.lower() == 'left':
            L_bp_l = L_bp
            L_bp_r = 0.000
        elif beampipe.lower() == 'right':
            L_bp_l = 0.000
            L_bp_r = L_bp
        else:
            L_bp_l = 0.000
            L_bp_r = 0.000

        geo = []
        curve = []
        pt_indx = 1
        curve_indx = 1
        curve.append(curve_indx)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')

            shift = (L_bp_l + L_bp_r + n_cells * L + (n_cells - 1) * S) / 2

            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            # cav.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # lineTo(start_point, [-shift, Ri], step)
            pt = [-shift, Ri]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)
            # cav.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # add beampipe
            if L_bp_l > 0:
                # lineTo(pt, [-shift + L_bp_l, Ri], step)
                pt = [-shift + L_bp_l, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            for n in range(1, n_cells + 1):
                if n == 1:
                    # lineTo(pt, [-shift + L_bp_l, Req], step)
                    pt = [-shift + L_bp_l, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Req], step)
                    pt = [-shift + L_bp_l + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + L, Ri], step)
                    pt = [-shift + L_bp_l + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L
                elif n > 1:
                    # lineTo(pt, [-shift + L_bp_l + S, Ri], step)
                    pt = [-shift + L_bp_l + S, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S, Req], step)
                    pt = [-shift + L_bp_l + S, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Req], step)
                    pt = [-shift + L_bp_l + S + L, Req]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # lineTo(pt, [-shift + L_bp_l + S + L, Ri], step)
                    pt = [-shift + L_bp_l + S + L, Ri]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    shift -= L

            if L_bp_r > 0:
                # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri], step)
                pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, Ri]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                # fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

            # END PATH
            # lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0], step)
            pt = [-shift + L_bp_l + (n_cells - 1) * S + L_bp_r, 0]

            pt_indx = add_point(cav, pt, pt_indx)
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

    def get_geometric_parameters(self):
        self.parameters = {
            'Ri': self.Ri,
            'L': self.L,
            'Req': self.Req,
            'S': self.S,
            'L_bp': self.L_bp
        }



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
    def rebuild(self, parameters, beampipe=None):
        """A fresh Pillbox from an unsuffixed parameter dict (L, Req, Ri, S, L_bp)."""
        dims = [float(parameters[k]) for k in ('L', 'Req', 'Ri', 'S', 'L_bp')]
        return Pillbox(self.n_cells, dims,
                       beampipe=self.beampipe if beampipe is None else beampipe)

