"""Data processing, file I/O, Pareto operations, and misc utilities."""
from scipy.interpolate import interp1d
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from cavsim2d.constants import *
from cavsim2d.utils.printing import *
from cavsim2d.utils.sensitivity import Sobol
from cavsim2d.utils.surrogate import PCE
from cavsim2d.utils.cubature import Data





def get_qoi_value(d, obj):
    """
    Gets the quantities of interest from simulation results

    Parameters
    ----------
    d: dict
        Dictionary containing several figures of merits from eigenmode solver
    obj: list
        List of objective functions

    Returns
    -------

    """
    # Req = d['CAVITY RADIUS'][n_cells - 1] * 10  # convert to mm
    # Freq = d['FREQUENCY'][n_cells - 1]
    # E_stored = d['STORED ENERGY'][n_cells - 1]
    # # Rsh = d['SHUNT IMPEDANCE'][n_cells-1]  # MOhm
    # Q = d['QUALITY FACTOR'][n_cells - 1]
    # Epk = d['MAXIMUM ELEC. FIELD'][n_cells - 1]  # MV/m
    # Hpk = d['MAXIMUM MAG. FIELD'][n_cells - 1]  # A/m
    # # Vacc = dict['ACCELERATION'][0]
    # # Eavg = d['AVERAGE E.FIELD ON AXIS'][n_cells-1]  # MV/m
    # Rsh_Q = d['EFFECTIVE IMPEDANCE'][n_cells - 1]  # Ohm
    #
    # Vacc = np.sqrt(
    #     2 * Rsh_Q * E_stored * 2 * np.pi * Freq * 1e6) * 1e-6
    # # factor of 2, remember circuit and accelerator definition
    # # Eacc = Vacc / (374 * 1e-3)  # factor of 2, remember circuit and accelerator definition
    # Eacc = Vacc / (norm_length * 1e-3)  # for 1 cell factor of 2, remember circuit and accelerator definition
    # Epk_Eacc = Epk / Eacc
    # Bpk_Eacc = (Hpk * 4 * np.pi * 1e-7) * 1e3 / Eacc
    #
    # d = {
    #     "Req": Req,
    #     "freq": Freq,
    #     "Q": Q,
    #     "E": E_stored,
    #     "R/Q": 2 * Rsh_Q,
    #     "Epk/Eacc": Epk_Eacc,
    #     "Bpk/Eacc": Bpk_Eacc
    # }

    objective = []

    # append objective functions
    for o in obj:
        if o in d.keys():
            objective.append(d[o])

    return objective



def interpolate_pareto(pareto, x_values):
    """Interpolate y-values for a given set of x-values based on a Pareto front."""
    f = interp1d(pareto[:, 0], pareto[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
    return f(x_values)


def extend_pareto(pareto1, pareto2):
    """Extend Pareto fronts by adding boundary points to match their x-ranges."""
    x_min1, x_max1 = pareto1[:, 0].min(), pareto1[:, 0].max()
    x_min2, x_max2 = pareto2[:, 0].min(), pareto2[:, 0].max()

    # Find boundary points to extend Pareto fronts
    extend_left1 = pareto2[pareto2[:, 0] == x_min2]
    extend_right1 = pareto2[pareto2[:, 0] == x_max2]
    extend_left2 = pareto1[pareto1[:, 0] == x_min1]
    extend_right2 = pareto1[pareto1[:, 0] == x_max1]

    if len(extend_left1) > 0 and x_min1 > x_min2:
        pareto1 = np.vstack([extend_left1, pareto1])
    if len(extend_right1) > 0 and x_max1 < x_max2:
        pareto1 = np.vstack([pareto1, extend_right1])
    if len(extend_left2) > 0 and x_min2 > x_min1:
        pareto2 = np.vstack([extend_left2, pareto2])
    if len(extend_right2) > 0 and x_max2 < x_max1:
        pareto2 = np.vstack([pareto2, extend_right2])

    return np.array(sorted(pareto1, key=lambda p: p[0])), np.array(sorted(pareto2, key=lambda p: p[0]))


def line_intersection(p1, p2):
    """Find the intersection point of two line segments (p1 and p2)."""

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (p1[0] - p1[2], p2[0] - p2[2])
    ydiff = (p1[1] - p1[3], p2[1] - p2[3])

    div = det(xdiff, ydiff)
    if div == 0:
        return None  # Parallel lines

    d = (det((p1[0], p1[1]), (p1[2], p1[3])), det((p2[0], p2[1]), (p2[2], p2[3])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)


def find_all_intersections(pareto1, pareto2):
    """Find all intersection points between the segments of two Pareto fronts."""
    intersections = []

    for i in range(len(pareto1) - 1):
        p1 = (pareto1[i, 0], pareto1[i, 1], pareto1[i + 1, 0], pareto1[i + 1, 1])
        for j in range(len(pareto2) - 1):
            p2 = (pareto2[j, 0], pareto2[j, 1], pareto2[j + 1, 0], pareto2[j + 1, 1])
            intersect_point = line_intersection(p1, p2)
            if intersect_point:
                xi, yi = intersect_point
                if min(p1[0], p1[2]) <= xi <= max(p1[0], p1[2]) and min(p2[0], p2[2]) <= xi <= max(p2[0], p2[2]):
                    intersections.append((round(xi, 6), round(yi, 6)))

    return sorted(set(intersections))


def calculate_bounded_area(x_values, y_values1, y_values2):
    """Calculate the area between two Pareto fronts."""
    return np.abs(np.trapz(np.abs(y_values1 - y_values2), x_values))


def area_pareto_fronts(pareto1, pareto2):
    """Plot the two Pareto fronts and correctly fill the area between them, handling edge cases."""
    # Extend Pareto fronts to cover the same x-range
    pareto1, pareto2 = extend_pareto(pareto1, pareto2)

    # Determine the combined range of x-values
    min_x = min(pareto1[:, 0].min(), pareto2[:, 0].min())
    max_x = max(pareto1[:, 0].max(), pareto2[:, 0].max())

    # Create a fine grid of x-values spanning the combined range
    x_values = np.linspace(min_x, max_x, 500)

    # # Interpolate both Pareto fronts at these x-values
    # y_values1 = interpolate_pareto(pareto1, x_values)
    # y_values2 = interpolate_pareto(pareto2, x_values)

    # Find intersection points and split x_values accordingly
    intersections = find_all_intersections(pareto1, pareto2)
    if intersections:
        segments = [x_values[(x_values >= min_x) & (x_values <= intersections[0][0])]]
        for i in range(len(intersections) - 1):
            segments.append(x_values[(x_values >= intersections[i][0]) & (x_values <= intersections[i + 1][0])])
        segments.append(x_values[(x_values >= intersections[-1][0]) & (x_values <= max_x)])
    else:
        segments = [x_values]

    # Calculate bounded area
    total_area = 0
    for segment in segments:
        seg_y1 = interpolate_pareto(pareto1, segment)
        seg_y2 = interpolate_pareto(pareto2, segment)
        total_area += calculate_bounded_area(segment, seg_y1, seg_y2)

    return total_area
    # print(f'Bounded Area: {total_area:.2f}')
    # if intersections:
    #     print('Intersection Points:')
    #     for x, y in intersections:
    #         print(f'X: {x:.2f}, Y: {y:.2f}')
    #
    # # Plot Pareto fronts
    # plt.plot(pareto1[:, 0], pareto1[:, 1], 'r-o', label='Pareto Front 1')
    # plt.plot(pareto2[:, 0], pareto2[:, 1], 'b-o', label='Pareto Front 2')
    #
    # # Fill the area between the two fronts in black
    # plt.fill_between(x_values, y_values1, y_values2, where=(y_values1 > y_values2), color='black', alpha=0.3)
    # plt.fill_between(x_values, y_values1, y_values2, where=(y_values1 <= y_values2), color='black', alpha=0.3)
    #
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.title('Area Between Two Pareto Fronts with Edge Cases')
    # plt.legend()
    # plt.show()


def reorder_legend(h, l, ncols):
    re_h = sum((h[ii::ncols] for ii in range(ncols)), [])
    re_l = sum((l[ii::ncols] for ii in range(ncols)), [])
    # reorder = lambda ll, nc: sum((ll[ii::nc] for ii in range(nc)), [])
    return re_h, re_l


def get_wakefield_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = {}
    current_points = []
    current_type = None
    in_electric_field_frame = False  # Flag to track "Electric Field Lines" frames
    frame_count = 0  # Counter for Electric Field Lines frames

    for line in lines:
        if "NEW FRAME" in line:
            if in_electric_field_frame:
                # Store the data for the current frame before moving to the next
                if current_type and current_points:
                    data[f"Frame_{frame_count}"].append(
                        (current_type, pd.DataFrame(current_points, columns=['X', 'Y'])))
            in_electric_field_frame = False  # Reset the flag when a new frame starts
            current_points = []
            current_type = None
        elif "Electric Field Lines" in line:
            in_electric_field_frame = True  # Set the flag for "Electric Field Lines"
            frame_count += 1
            data[f"Frame_{frame_count}"] = []  # Initialize a new entry for the frame
        elif "JOIN 1" in line and in_electric_field_frame:
            if current_type and current_points:
                data[f"Frame_{frame_count}"].append(
                    (current_type, pd.DataFrame(current_points, columns=['X', 'Y'])))
            current_type = "DOTS" if "DOTS" in line else "SOLID"
            current_points = []
        elif in_electric_field_frame and (re.match(r'\s*\d\.\d+E[+-]\d+\s+\d\.\d+E[+-]\d+', line) or re.match(
                r'\s*\d\.\d+\s+\d\.\d+E[+-]\d+', line)):
            point = [float(x) for x in re.findall(r'[-+]?\d*\.\d+E[-+]?\d+|\d+\.\d+', line)]
            current_points.append(point)

    # Add the last set of points if any
    if in_electric_field_frame and current_type and current_points:
        data[f"Frame_{frame_count}"].append((current_type, pd.DataFrame(current_points, columns=['X', 'Y'])))

    return data


import numpy as np
import copy
import re
from numpy.polynomial.legendre import leggauss

# Ordered parameter names
VAR_NAMES = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req', 'alpha']



def merge_runs_within_variable(df: pd.DataFrame, var: str,
                               rtol=1e-6, atol=1e-8):
    """
    For a given variable prefix (e.g. 'Ri' or 'Req'), find all columns
    like 'Ri1','Ri2',… sorted by index; then group any consecutive indices
    i, i+1 where df[Ri_i] ≈ df[Ri_{i+1}], producing merged names ['Ri2Ri3'], etc.
    Returns an ordered list of (merged_name, series).
    """
    # 1) find all columns for this var, extract indices
    pat = re.compile(rf'^{re.escape(var)}(\d+)$')
    cols = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            idx = int(m.group(1))
            cols.append((idx, c))
    cols.sort(key=lambda x: x[0])
    merged = []
    i = 0
    while i < len(cols):
        idx, name = cols[i]
        # look ahead
        if i+1 < len(cols):
            idx2, name2 = cols[i+1]
            # must be consecutive indices
            if idx2 == idx+1:
                # compare the arrays
                a = df[name].to_numpy()
                b = df[name2].to_numpy()
                if np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                    # merge them
                    new_name = f"{name}{name2}"
                    merged.append((new_name, df[name]))
                    i += 2
                    continue
        # else no merge
        merged.append((name, df[name]))
        i += 1
    return merged

def merge_equal_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse runs of equal columns **within** each variable (Ri, Req),
    then reassemble all merged columns in the original half‐cell order.
    """
    # 1) Determine half‐cell column order from df.columns via regex pairs
    #    We assume columns alternate Ri#, Req#, Ri#, Req#, …
    order = []
    pat = re.compile(r'^(Ri|Req)(\d+)$')
    for c in df.columns:
        if pat.match(c):
            order.append(c)

    # 2) Build merged lists
    ri_merged  = merge_runs_within_variable(df, 'Ri')
    req_merged = merge_runs_within_variable(df, 'Req')

    # 3) Map original col → merged_name
    col_to_merged = {}
    for new_name, series in ri_merged + req_merged:
        # the group names are concatenations, so split back
        # e.g. 'Ri2Ri3' → ['Ri2','Ri3']
        parts = re.findall(r'(Ri\d+|Req\d+)', new_name)
        for p in parts:
            col_to_merged[p] = new_name

    # 4) Reassemble in original order, but only add each merged_name once
    final = []
    seen = set()
    for c in order:
        m = col_to_merged.get(c)
        if m and m not in seen:
            seen.add(m)
            # pick the series from df via the first part
            first_part = re.match(r'(Ri\d+|Req\d+)', m).group(1)
            final.append((m, df[first_part]))

    # 5) Build DataFrame
    return pd.DataFrame({name: ser for name, ser in final})

def run_sa():
    folder = r'C:\Users\sosoho\Documents'
    # read nodes
    nodes = pd.read_csv(fr'{folder}/nodes.csv', sep='\t')
    nodes = nodes.loc[:, nodes.columns.str.contains('Ri|Req')]
    # merge welded dimensions at seam
    nodes = merge_equal_columns(nodes)

    results = pd.read_excel(fr'{folder}/table.xlsx', 'Sheet1')
    print(results.columns)
    print()
    data = pd.concat([nodes, results], axis=1)
    data.to_excel(fr'{folder}/data.xlsx', index=False)
    # print(data)

    names = list(nodes.columns)

    midcell = [42, 42, 12, 19, 35, 57.652, 103.3536]  # <- A, B, a, b, Ri, L, Req
    endcell_l = [40.34, 40.34, 10, 13.5, 39, 55.7251, 103.3536]
    endcell_r = [42, 42, 9, 12.8, 39, 56.8407, 103.3536]

    print(names)
    problem = {
        'names': names,
        'num_vars': len(names),
        'bounds': [[38.7, 39.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [34.7, 35.3], [103.0536, 103.6536],
                   [38.7, 39.3]]
    }
    #
    for obj in ['ff [%]']:
        # obj = 'kcc [%]'
        pce_order, pce_truncation = 2, 2

        pce_data = Data(fr'{folder}', problem)

        X_train, Y_train = pce_data.train_data(obj)
        X_test, Y_test = pce_data.test_data()

        pce_reg = PCE(pce_data, Y_train)

        pce_reg.pce_regression(pce_order, pce_truncation)

        Y_reg = pce_reg.evaluate_reg(X_test)

        # plot
        fig, axs = plt.subplot_mosaic([[0], [1]], figsize=(13, 7))
        ax = axs[0]
        ax_err = axs[1]

        ax.plot(Y_test[:50], label='actual', mec='k', lw=0, marker='o', mfc='none', ms=10)
        ax.plot(Y_reg[:50], label=f'pce reg. ({pce_order},{pce_truncation})]',
            zorder=23, lw=1, marker='^', mfc='none', ms=8)
        ax_err.plot(np.abs(Y_reg - Y_test), label='error pce reg')

        # calculate sobol
        # resample for sobol
        sobol_indices = {}
        sobol = Sobol(problem, folder)
        sobol_indices[obj] = sobol.analyse(pce_reg.evaluate_reg, 'pce reg')
        sobol.plot(obj, figsize=(10, 3))

        # # save sobol indices
        # with open(f"{folder}/{obj.replace('/', '_')}.pkl", 'wb') as f:
        #     pickle.dump(sobol.sobol_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

