"""Test: Eigenmode analysis using the new solver-as-object API."""
import sys
import numpy as np
import pprint

sys.path.insert(0, r'c:\Users\Soske\Documents\git_projects\cavsim2d')

from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)

# --- Setup ---
sim_dir = r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations'
cavs = Cavities(sim_dir)

# Define geometry parameters
midcell = [62.22, 66.13, 30.22, 23.11, 72, 93.5, 171.20]  # A, B, a, b, Ri, L, Req
endcell_l = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]
endcell_r = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]

# Create cavity
cav1 = EllipticalCavity(2, midcell, endcell_l, endcell_r, beampipe='both')
cavs.add_cavity([cav1], ['C3795'])

# --- Run eigenmode (Cavities-level API, still works) ---
eigenmode_config = {
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm'
}
cavs.run_eigenmode(eigenmode_config)

# --- Access results ---
pp.pprint(cavs.eigenmode_qois)

# --- Per-cavity solver object API ---
# cav1.eigenmode.run(eigenmode_config)  # alternative per-cavity API
# print(cav1.eigenmode.qois)
