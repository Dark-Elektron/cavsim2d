#!/usr/bin/env python
"""
Benchmark: Cylindrical Waveguide Cavity
=======================================
Compares cavsim2d's NGSolve eigenmode solver against closed-form analytical
solutions for a circular waveguide cavity (cylinder).

Uses the CircularWaveguide primitive class from the library and runs
the h-p refinement sweep using the built-in study_mesh_convergence module.

Outputs (saved to benchmark/cylindrical_waveguide/)
--------------------------------------------------
- convergence_results.csv: raw simulation data and errors.
- convergence_plots: log-log relative error plots vs DOFs and vs time.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib – headless backend
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make the in-repo cavsim2d importable
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from cavsim2d.cavity import Study, CircularWaveguide

# Analytical solutions for TM010 mode of R=230 mm, L=200 mm copper cavity
F_ANALYTICAL   = 498.880556     # MHz
Q_ANALYTICAL   = 36651.90884     # dimension-less
RQ_ANALYTICAL  = 220.375309     # Ohm

def main():
    print("=" * 70)
    print("  BENCHMARK: Cylindrical Waveguide Cavity")
    print(f"  Analytical TM010: f = {F_ANALYTICAL:.6f} MHz, Q = {Q_ANALYTICAL:.2f}, R/Q = {RQ_ANALYTICAL:.4f} Ohm")
    print("=" * 70)

    # Set up directories
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cylindrical_waveguide")
    os.makedirs(output_dir, exist_ok=True)

    # Instantiate cavities container
    cavs = Study(output_dir)
    # Radius R = 230 mm, Length L = 200 mm
    cav = CircularWaveguide(R=230.0, L=200.0)
    cavs.add_cavity([cav], ['circular_waveguide'])

    # Set direct_solver to sparsecholesky for systems without MKL Pardiso
    eigenmode_config = {
        'processes': 1,
        'rerun': True,
        'boundary_conditions': 'mm',
        'direct_solver': 'sparsecholesky'
    }
    cav.eigenmode_config = eigenmode_config

    # Run convergence study using the library's built-in module. h-refinement
    # is now error-driven (adaptive): start from maxh=20 mm and let the solver
    # refine, sweeping the polynomial order p from 3 to 5.
    print("Running convergence study sweep...")
    cav.study_mesh_convergence(h=20.0, p=3, p_passes=3, p_step=1,
                               tol=1e-12, max_refinements=6)

    # Load results and keep the fundamental TM010 mode (monopole, mode 0); the
    # table carries one row per (order, refinement level, polarisation, mode).
    df = cav.convergence_df_data.copy()
    df = df[(df['polarisation'] == 'monopole') & (df['mode_index'] == 0)].copy()
    for col in ('freq [MHz]', 'Q []', 'R/Q [Ohm]'):
        df[col] = pd.to_numeric(df[col])

    # Relative errors compared to the exact analytical values
    df['rel_err_freq'] = np.abs(df['freq [MHz]'] - F_ANALYTICAL) / F_ANALYTICAL
    df['rel_err_Q'] = np.abs(df['Q []'] - Q_ANALYTICAL) / Q_ANALYTICAL
    df['rel_err_RoQ'] = np.abs(df['R/Q [Ohm]'] - RQ_ANALYTICAL) / RQ_ANALYTICAL

    # Save to CSV
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Convergence results saved to: {csv_path}")

    # Generate convergence plots
    print("Generating convergence plots...")
    plot_convergence_curves(df, output_dir)

    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print(f"  All results in: {output_dir}")
    print("=" * 70)

def plot_convergence_curves(df, output_dir):
    # Setup color palette
    colors = {
        'freq': '#1f77b4',  # Blue
        'RoQ': '#ff7f0e',   # Orange
        'Q': '#2ca02c'      # Green
    }
    markers = {3: 'o', 4: 's', 5: '^', 6: 'v'}

    # 1. Error vs DOFs (one adaptive refinement path per polynomial order p)
    fig, ax = plt.subplots(figsize=(8, 6))
    for p_val in sorted(df['p'].unique()):
        sub = df[df['p'] == p_val].sort_values('No of DOFs')
        m = markers.get(p_val, 'o')
        ax.plot(sub['No of DOFs'], sub['rel_err_freq'], color=colors['freq'], marker=m, ls='-', label=f"Freq (p={p_val})")
        ax.plot(sub['No of DOFs'], sub['rel_err_RoQ'], color=colors['RoQ'], marker=m, ls='--', label=f"R/Q (p={p_val})")
        ax.plot(sub['No of DOFs'], sub['rel_err_Q'], color=colors['Q'], marker=m, ls=':', label=f"Q (p={p_val})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Number of DOFs", fontsize=11)
    ax.set_ylabel("Relative Error", fontsize=11)
    ax.set_title("Cylindrical Waveguide Cavity — Adaptive Convergence vs DOFs", fontsize=12, weight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9, ncol=3, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_vs_dof.png"), dpi=200)
    plt.close(fig)

    # 2. Error vs Time
    fig, ax = plt.subplots(figsize=(8, 6))
    for p_val in sorted(df['p'].unique()):
        sub = df[df['p'] == p_val].sort_values('time [s]')
        m = markers.get(p_val, 'o')
        ax.plot(sub['time [s]'], sub['rel_err_freq'], color=colors['freq'], marker=m, ls='-', label=f"Freq (p={p_val})")
        ax.plot(sub['time [s]'], sub['rel_err_RoQ'], color=colors['RoQ'], marker=m, ls='--', label=f"R/Q (p={p_val})")
        ax.plot(sub['time [s]'], sub['rel_err_Q'], color=colors['Q'], marker=m, ls=':', label=f"Q (p={p_val})")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Solve Time [s]", fontsize=11)
    ax.set_ylabel("Relative Error", fontsize=11)
    ax.set_title("Cylindrical Waveguide Cavity — Adaptive Convergence vs Time", fontsize=12, weight='bold')
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=9, ncol=3, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "convergence_vs_time.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
