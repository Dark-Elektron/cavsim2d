"""Shared setup for the cavsim2d example scripts.

Each example writes its results (geodata, meshes, qois.json, ABCI output, ...)
and saved plots under ``SIM_ROOT`` so they can be inspected after the run,
exactly like the notebooks do — but headless, so every figure is saved to a
PNG instead of shown interactively.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless: figures are saved, never shown
import matplotlib.pyplot as plt

# Make the in-repo package importable even without `pip install`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Where all example outputs land (same location the notebooks use).
SIM_ROOT = r'C:\Users\Soske\Documents\git_projects\cavsim2d_simulations'

# A TESLA-like single mid-cell: A, B, a, b, Ri, L, Req  [mm]
MIDCELL = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20]
ENDCELL_L = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]
ENDCELL_R = [62.58, 57.54, 17.21, 12, 80, 93.795, 171.20]


def project_dir(name):
    """Per-example project directory under SIM_ROOT (created if needed)."""
    p = os.path.join(SIM_ROOT, 'examples', name)
    os.makedirs(p, exist_ok=True)
    return p


def plots_dir(name):
    """Per-example directory for saved figures (created if needed)."""
    p = os.path.join(SIM_ROOT, 'examples', name, 'plots')
    os.makedirs(p, exist_ok=True)
    return p


def savefig(path, fig=None):
    """Save *fig* (or the current figure) and close it. Never raises — a
    plotting hiccup should not abort the demo."""
    try:
        fig = fig or plt.gcf()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"    saved plot: {path}")
    except Exception as e:  # noqa: BLE001 - demo robustness
        print(f"    [warn] could not save {path}: {e!r}")
    finally:
        plt.close('all')


def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70, flush=True)
