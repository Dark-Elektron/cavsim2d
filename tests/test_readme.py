"""Test: Full README demo using the new solver-as-object API."""
import sys
import pprint

sys.path.insert(0, r'c:\Users\Soske\Documents\git_projects\cavsim2d')

from cavsim2d.cavity import Cavities, EllipticalCavity

pp = pprint.PrettyPrinter(indent=4)

# --- Setup ---
sim_dir = r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations'
cavs = Cavities(sim_dir)

# =====================================================
# 1. Basic eigenmode analysis
# =====================================================

# Define geometry parameters
midcell = [42, 42, 12, 19, 35, 57.7, 103.353]     # A, B, a, b, Ri, L, Req
endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.716, 103.353]
endcell_r = [42, 42, 9, 12.8, 39, 56.815, 103.353]

# Create cavity
n_cells = 1
tesla = EllipticalCavity(n_cells, midcell, endcell_l, endcell_r, beampipe='both')
cavs.add_cavity(tesla, 'TESLA')

# Run eigenmode  
eigenmode_config = {
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm'
}
cavs.run_eigenmode(eigenmode_config)
pp.pprint(cavs.eigenmode_qois)

# Compare FM quantities
# cavs.plot_compare_fm_bar()

# =====================================================
# 2. Tune to target frequency
# =====================================================

cavs2 = Cavities(sim_dir)

midcell_t = [42, 42, 12, 19, 35, 57.7, 100]  # deliberately off-target Req
tesla_t = EllipticalCavity(1, midcell_t, midcell_t, midcell_t, beampipe='none')
cavs2.add_cavity(tesla_t, 'TESLA_tune')

tune_config = {
    'freqs': 1300,
    'parameters': 'Req_m',
    'cell_types': 'mid-cell',
    'processes': 1,
    'rerun': True,
    'eigenmode_config': {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
    },
}
cavs2.run_tune(tune_config)
pp.pprint(cavs2.tune_results)

# =====================================================
# 3. Eigenmode + UQ
# =====================================================

cavs3 = Cavities(sim_dir)

midcell_u = [42, 42, 12, 19, 35, 57.7, 103.353]
tesla_u = EllipticalCavity(1, midcell_u, midcell_u, midcell_u, beampipe='both')
cavs3.add_cavity(tesla_u, 'TESLA_UQ')

eigenmode_uq_config = {
    'processes': 1,
    'rerun': True,
    'boundary_conditions': 'mm',
    'uq_config': {
        'variables': ['A', 'B', 'a', 'b'],
        'objectives': ["Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]"],
        'delta': [0.05, 0.05, 0.05, 0.05],
        'processes': 1,
        'distribution': 'gaussian',
        'method': ['Quadrature', 'Stroud3'],
        'cell_type': 'mid-cell',
        'cell complexity': 'multicell',
        'tune_config': {
            'freqs': 1300,
            'parameters': 'Req',
            'cell_types': 'mid-cell',
            'processes': 1,
            'rerun': True,
            'eigenmode_config': {
                'processes': 1,
                'rerun': True,
                'boundary_conditions': 'mm',
            },
        },
    }
}
cavs3.run_eigenmode(eigenmode_uq_config)
pp.pprint(cavs3.eigenmode_qois)
# pp.pprint(cavs3.uq_fm_results)
