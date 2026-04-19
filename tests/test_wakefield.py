"""Test: Wakefield analysis using the new solver-as-object API."""
import sys
import pprint

sys.path.insert(0, r'c:\Users\Soske\Documents\git_projects\cavsim2d')

from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)

# --- Setup ---
sim_dir = r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations'
cavs = Cavities(sim_dir)

# Define geometry parameters
n_cells = 2
midcell = [62.22, 66.13, 30.22, 23.11, 72, 93.5, 171.20]   # A, B, a, b, Ri, L, Req
endcell_l = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]
endcell_r = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]

# Create cavity
cav1 = EllipticalCavity(n_cells, midcell, endcell_l, endcell_r, beampipe='both')
cavs.add_cavity([cav1], ['C3795_WF'])

# --- Step 1: Tune first ---
tune_config = {
    'freqs': 801.58,
    'parameters': 'Req',
    'cell_types': 'mid-cell',
    'processes': 1,
    'rerun': True,
    'eigenmode_config': {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
    },
}
cavs.run_tune(tune_config)

# --- Step 2: Eigenmode ---
eigenmode_config = {
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm'
}
cavs.run_eigenmode(eigenmode_config)
pp.pprint(cavs.eigenmode_qois)

# --- Step 3: Wakefield ---
wakefield_config = {
    'processes': 1,
    'rerun': True,
    'MROT': 2,
    'wakelength': 50,
    'bunch_length': 25,
    'MT': 10,
    'NFS': 10000,
}
cavs.run_wakefield(wakefield_config)
pp.pprint(cavs.wakefield_qois)
