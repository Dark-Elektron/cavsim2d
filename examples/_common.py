"""Shared setup for the cavsim2d example scripts.

Each example writes its results (geometry, meshes, qois.json, ABCI output, ...)
and its plots under ``SIM_ROOT`` so they can be inspected after the run. The
scripts are headless: every figure is saved to a PNG instead of shown.
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')  # headless: figures are saved, never shown
import matplotlib.pyplot as plt

# Make the in-repo package importable even without `pip install`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cavsim2d.utils.style import apply_style  # noqa: E402 (after the path setup)

apply_style()
# The plot methods show their figure by default; under the headless backend
# that is a no-op that warns, so say nothing.
warnings.filterwarnings('ignore', message='.*non-interactive.*')

# Where all example outputs land: $CAVSIM2D_EXAMPLES_DIR, else examples/output/.
SIM_ROOT = os.environ.get('CAVSIM2D_EXAMPLES_DIR') or os.path.join(_HERE, 'output')

# An 800 MHz single mid-cell and its end cells: A, B, a, b, Ri, L, Req  [mm]
MIDCELL = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20]
ENDCELL_L = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]
ENDCELL_R = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]


def project_dir(name):
    """Per-example project directory under SIM_ROOT."""
    return os.path.join(SIM_ROOT, name)


def plots_dir(name):
    """Per-example directory for saved figures (created if needed)."""
    p = os.path.join(SIM_ROOT, name, 'plots')
    os.makedirs(p, exist_ok=True)
    return p


def savefig(path, fig=None):
    """Save *fig* (or the current figure) and close every figure."""
    fig = fig or plt.gcf()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"    saved plot: {path}")
    plt.close('all')


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70, flush=True)
