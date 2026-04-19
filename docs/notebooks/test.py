import sys
sys.path.append("../..")
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from cavsim2d.cavity import Cavity, EllipticalCavity, Cavities, OperationPoints
import pprint
pp = pprint.PrettyPrinter(indent=4)

# define geometry parameters
midcell = np.array([42.0, 23.0, 42.5, 57.5, 70.24238959559739, 93.5, 170.0, 0])
endcell = np.array([47.0, 23.0, 42.5, 57.5, 70.59919028624745, 93.5, 170.0, 0])
endcell_r = np.array([47.0, 23.0, 42.5, 57.5, 70.59919028624745, 93.5, 170.0, 0])

# create cavity
cav_tune_eig = EllipticalCavity(1, midcell, midcell, midcell, beampipe='none')
ax = cav_tune_eig.plot('geometry', label='Before tuning')

cavs_tune_eig = Cavities(r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations',
overwrite=True
)

cavs_tune_eig.add_cavity(cav_tune_eig, 'cav_tune_eig')

tune_config = {
    'freqs': 801.58,
    'parameters': 'L',
    'cell_types': 'mid-cell',
    'processes': 1,
    'rerun': True,
    'eigenmode_config': {
        'processes': 3,
        'rerun': True,
        'boundary_conditions': 'mm',
    },
}
cavs_tune_eig.run_tune(tune_config)
pp.pprint(cavs_tune_eig.tune_results)

# plot geometry after tuning
# cavs_tune_eig.plot('geometry', ax, label='After tuning')