from cavsim2d.models.base import Cavity
from cavsim2d.utils.shared_functions import *
from cavsim2d.geometry import Profile
import os

class CircularWaveguide(Cavity):
    """Circular Waveguide Cavity primitive.

    A cylindrical PEC-closed cavity of radius *R* and length *L*.
    To ensure correct numerical boundary conditions on the axis for monopole modes,
    a tiny region (radius *Ri* = 1.0 mm) at the center of the endplates is PMC,
    while the rest of the endplates and the outer wall are PEC.
    """
    def __init__(self, R, L):
        super().__init__(n_cells=1, beampipe='none')
        self.R = R      # Radius in mm
        self.L = L      # Length in mm
        self.kind = 'circular_waveguide'
        self.name = 'circular_waveguide'
        self.n_cells = 1
        self.n_modes = 1
        
        self.get_geometric_parameters()

    def get_geometric_parameters(self):
        self.parameters = {
            'R': self.R,
            'L': self.L
        }

    def create(self, n_cells=None, beampipe=None, mode=None):
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            geo_dir = os.path.join(self.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)
            self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            self.write_geometry(self.parameters, write=self.geo_filepath)
            self._write_geometry_snapshot()

    def profile(self):
        """Meridian boundary as a unified :class:`Profile` (metres) — the native
        netgen.occ path, no ``.geo`` round-trip.

        A rectangle in the (z, r) half-plane: both end planes and the barrel are
        PEC, the axis is AXI. There is no PMC region — this matches the ``.geo``
        writer, which tags all three walls PEC despite the class docstring.
        """
        try:
            R = float(self.parameters['R']) * 1e-3
            L = float(self.parameters['L']) * 1e-3
        except (KeyError, TypeError, ValueError):
            return None

        shift = L / 2.0
        return (Profile('circular_waveguide')
                .start(-shift, 0.0)
                .line_to(-shift, R, 'PEC')      # left end plane
                .line_to(shift, R, 'PEC')       # barrel
                .line_to(shift, 0.0, 'PEC')     # right end plane
                .close('AXI'))

    def write_geometry(self, parameters, write=None, **kwargs):
        """Write a Gmsh .geo file for the circular waveguide cavity."""
        R_m = parameters['R'] * 1e-3
        L_m = parameters['L'] * 1e-3
        shift = L_m / 2.0

        os.makedirs(os.path.dirname(write), exist_ok=True)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write('SetFactory("OpenCASCADE");\n')
            
            # Points defining the rectangular half-section in r-z plane
            cav.write(f"Point(1) = {{-{shift:.16e}, 0, 0}};\n")
            cav.write(f"Point(2) = {{-{shift:.16e}, {R_m:.16e}, 0}};\n")
            cav.write(f"Point(3) = {{{shift:.16e}, {R_m:.16e}, 0}};\n")
            cav.write(f"Point(4) = {{{shift:.16e}, 0, 0}};\n")
            
            cav.write("Line(1) = {1, 2};\n")  # Left endplane inner part (PMC)
            cav.write("Line(2) = {2, 3};\n")  # Left endplane outer part (PEC)
            cav.write("Line(3) = {3, 4};\n")  # Outer cylinder equator wall (PEC)
            cav.write("Line(4) = {4, 1};\n")  # Axis (AXI)
            
            cav.write('\nPhysical Line("PEC") = {1, 2, 3};\n')
            cav.write('Physical Line("AXI") = {4};\n')
            cav.write("\nCurve Loop(1) = {1, 2, 3, 4};\n")
            cav.write("Plane Surface(1) = {1};\n")
            cav.write("Reverse Surface 1;\n")
            cav.write('Physical Surface("Domain") = {1};\n')

    def clone_for_tuning(self, tuned_parameters, tuned_self_dir, beampipe=None):
        clone = CircularWaveguide(tuned_parameters['R'], tuned_parameters['L'], tuned_parameters.get('Ri', 1.0))
        clone.name = self.name
        clone.projectDir = self.projectDir
        clone.self_dir = tuned_self_dir
        os.makedirs(clone.self_dir, exist_ok=True)
        clone.geo_filepath = os.path.join(clone.self_dir, 'geometry', 'geodata.geo')
        clone.write_geometry(tuned_parameters, write=clone.geo_filepath)
        return clone
