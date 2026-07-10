from abc import ABC, abstractmethod
from cavsim2d.cavity.base import Cavity
from cavsim2d.constants import *
from cavsim2d.data_module.abci_data import ABCIData
from cavsim2d.utils.shared_functions import *
import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

class Pillbox(Cavity):
    def __init__(self, n_cells, dims, beampipe='none'):
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

    def clone_for_tuning(self, tuned_parameters, tuned_self_dir, beampipe=None):
        """Return a fresh Pillbox in ``tuned_self_dir`` carrying the tuned
        (unsuffixed) parameter dict, with its own geometry written out."""
        if beampipe is None:
            beampipe = self.beampipe
        dims = [tuned_parameters['L'], tuned_parameters['Req'],
                tuned_parameters['Ri'], tuned_parameters['S'],
                tuned_parameters['L_bp']]
        clone = Pillbox(self.n_cells, dims, beampipe=beampipe)
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_self_dir)

        geo_dir = os.path.join(clone.self_dir, 'geometry')
        os.makedirs(geo_dir, exist_ok=True)
        clone.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
        clone.write_geometry(clone.parameters, clone.n_cells, clone.beampipe,
                             write=clone.geo_filepath)
        clone.uq_dir = os.path.join(clone.self_dir, 'uq')
        clone._write_geometry_snapshot()
        return clone

    def _load_tuned_from_disk(self, tuned_dir):
        """Rebuild the tuned Pillbox from a persisted ``tuned/`` folder,
        reading the tuned (unsuffixed) parameters from tune_res.json."""
        from cavsim2d.processes.tune import last_stage_result

        params = dict(self.parameters)
        tune_res_path = os.path.join(str(tuned_dir), 'tune_info', 'tune_res.json')
        tune_res = {}
        if os.path.exists(tune_res_path):
            with open(tune_res_path) as f:
                tune_res = json.load(f)
            last = last_stage_result(tune_res)
            if last and last.get('parameters'):
                for k, v in last['parameters'].items():
                    try:
                        params[k] = float(v)
                    except (TypeError, ValueError):
                        pass

        dims = [params['L'], params['Req'], params['Ri'], params['S'], params['L_bp']]
        clone = Pillbox(self.n_cells, dims, beampipe=self.beampipe)
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone.self_dir = str(tuned_dir)
        clone.geo_filepath = os.path.join(clone.self_dir, 'geometry', 'geodata.geo')
        clone.tune_results = tune_res
        return clone

    def plot(self, what, ax=None, **kwargs):
        if what.lower() == 'geometry':
            ax = plot_pillbox_geometry(self.n_cells, self.L, self.Req, self.Ri, self.S, self.L_bp, self.beampipe)
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
