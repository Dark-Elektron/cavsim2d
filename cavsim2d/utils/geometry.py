"""Geometry creation, writing, and manipulation utilities."""
from matplotlib.patches import Ellipse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import math
from cavsim2d.constants import *
from cavsim2d.utils.printing import *


def update_alpha(cell, cell_parameterisation='simplecell'):
    """
    Update geometry json file variables to include the value of alpha

    Parameters
    ----------
    cell:
        Cavity geometry parameters

    Returns
    -------
    List of cavity geometry parameters

    """
    A, B, a, b, Ri, L, Req = cell[:7]
    alpha = calculate_alpha(A, B, a, b, Ri, L, Req, 0)
    if cell_parameterisation == 'simplecell':
        cell = [A, B, a, b, Ri, L, Req, alpha[0]]
    elif cell_parameterisation == 'flattop':
        cell = [A, B, a, b, Ri, L, Req, cell[7], alpha[0]]

    return np.array(cell)


def calculate_alpha(A, B, a, b, Ri, L, Req, L_bp):
    """
    Calculates the largest angle the tangent line of two ellipses makes with the horizontal axis

    Parameters
    ----------
    A: float
    B: float
    a: float
    b: float
    Ri: float
    L: float
    Req: float
    L_bp: float

    Returns
    -------
    alpha: float
        Largest angle the tangent line of two ellipses makes with the horizontal axis
    error_msg: int
        State of the iteration, failed or successful. Refer to

    """

    df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp)
    x1, y1, x2, y2 = df[0]
    error_msg = df[-2]

    alpha = 180 - np.arctan2(y2 - y1, (x2 - x1)) * 180 / np.pi
    return alpha, error_msg


def tangent_coords(A, B, a, b, Ri, L, Req, L_bp, lft=0, tangent_check=False):
    """
    Calls to :py:func:`utils.shared_function.ellipse_tangent`

    Parameters
    ----------
    A: float
        Equator ellipse dimension
    B: float
        Equator ellipse dimension
    a: float
        Iris ellipse dimension
    b: float
        Iris ellipse dimension
    Ri: float
        Iris radius
    L: float
        Cavity half cell length
    Req: float
        Cavity equator radius
    L_bp: float
        Cavity beampipe length
    tangent_check: bool
        If set to True, the calculated tangent line as well as the ellipses are plotted and shown

    Returns
    -------
    df: pandas.Dataframe
        Pandas dataframe containing information on the results from fsolve
    """
    # data = ([0 + L_bp, Ri + b, L + L_bp, Req - B],
    #         [a, b, A, B])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    #
    # df = fsolve(ellipse_tangent,
    #             np.array([a + L_bp, Ri + f[0] * b, L - A + L_bp, Req - f[1] * B]),
    #             args=data, fprime=jac, xtol=1.49012e-12, full_output=True)
    #
    #     # ic(df)

    data = ([0 + L_bp, Ri + b, L + L_bp, Req - B], [a, b, A, B])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    # checks = {"non-reentrant": [0.5, -0.5],
    #           "reentrant": [0.85, -0.85],
    #           "expansion": [0.15, -0.01]}

    max_restart = 4
    checks = {"non-reentrant": [[0.5, -0.5], [0.75, -0.25], [0.25, -0.75], [0.9, -0.1]],
              "reentrant": [[1.1, -1.1], [1.9, -1.9], [1.5, -1.5], [1.75, -1.75], ]
              }
    msg = 4
    df = pd.DataFrame()
    for ii in range(max_restart):
        if msg != 1:
            if a + A > L:
                df = fsolve(ellipse_tangent,
                            np.array([a + L_bp, Ri + checks['reentrant'][ii][0] * b, L - A + L_bp,
                                      Req + checks['reentrant'][ii][1] * B]),
                            args=data, fprime=jac, xtol=1.49012e-12, full_output=True)
            else:
                df = fsolve(ellipse_tangent,
                            np.array([a + L_bp, Ri + checks['non-reentrant'][ii][0] * b, L - A + L_bp,
                                      Req + checks['non-reentrant'][ii][1] * B]),
                            args=data, fprime=jac, xtol=1.49012e-12, full_output=True)
            msg = df[-2]

    x1, y1, x2, y2 = df[0]
    # alpha = 180 - np.arctan2(y2 - y1, (x2 - x1)) * 180 / np.pi

    if tangent_check:
        shift_x = -L - lft
        h, k, p, q = data[0]
        a, b, A, B = data[1]
        el_ab = Ellipse((shift_x + h, k), 2 * a, 2 * b, alpha=0.5)
        el_AB = Ellipse((shift_x + p, q), 2 * A, 2 * B, alpha=0.5)

        ax = plt.gca()
        ax.add_artist(el_ab)
        ax.add_artist(el_AB)

        x1, y1, x2, y2 = df[0]
        ax.plot([shift_x + x1, shift_x + x2], [y1, y2], label=fr'{df[-2]}:: {df[-1]}')
        ax.legend()

    return df


def ellipse_tangent(z, *data):
    """
    Calculates the coordinates of the tangent line that connects two ellipses

    .. _ellipse tangent:

    .. figure:: ../images/ellipse_tangent.png
       :alt: ellipse tangent
       :align: center
       :width: 500px

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
        See :numref:`ellipse tangent`
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``


    Returns
    -------
    list of four non-linear functions

    Note
    -----
    The four returned non-linear functions are

    .. math::

       f_1 = \\frac{A^2b^2(x_1 - h)(y_2-q)}{a^2B^2(x_2-p)(y_1-k)} - 1

       f_2 = \\frac{(x_1 - h)^2}{a^2} + \\frac{(y_1-k)^2}{b^2} - 1

       f_3 = \\frac{(x_2 - p)^2}{A^2} + \\frac{(y_2-q)^2}{B^2} - 1

       f_4 = \\frac{-b^2(x_1-x_2)(x_1-h)}{a^2(y_1-y_2)(y_1-k)} - 1
    """

    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    return f1, f2, f3, f4


def jac(z, *data):
    """
    Computes the Jacobian of the non-linear system of ellipse tangent equations

    Parameters
    ----------
    z: list, array like
        Contains list of tangent points coordinate's variables ``[x1, y1, x2, y2]``.
        See :numref:`ellipse tangent`
    data: list, array like
        Contains midpoint coordinates of the two ellipses and the dimensions of the ellipses
        data = ``[coords, dim]``; ``coords`` = ``[h, k, p, q]``, ``dim`` = ``[a, b, A, B]``

    Returns
    -------
    J: array like
        Array of the Jacobian

    """
    coord, dim = data
    h, k, p, q = coord
    a, b, A, B = dim
    x1, y1, x2, y2 = z

    # f1 = A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k)) - 1
    # f2 = (x1 - h) ** 2 / a ** 2 + (y1 - k) ** 2 / b ** 2 - 1
    # f3 = (x2 - p) ** 2 / A ** 2 + (y2 - q) ** 2 / B ** 2 - 1
    # f4 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k)) - 1

    df1_dx1 = A ** 2 * b ** 2 * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))
    df1_dy1 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k) ** 2)
    df1_dx2 = - A ** 2 * b ** 2 * (x1 - h) * (y2 - q) / (a ** 2 * B ** 2 * (x2 - p) ** 2 * (y1 - k))
    df1_dy2 = A ** 2 * b ** 2 * (x1 - h) / (a ** 2 * B ** 2 * (x2 - p) * (y1 - k))

    df2_dx1 = 2 * (x1 - h) / a ** 2
    df2_dy1 = 2 * (y1 - k) / b ** 2
    df2_dx2 = 0
    df2_dy2 = 0

    df3_dx1 = 0
    df3_dy1 = 0
    df3_dx2 = 2 * (x2 - p) / A ** 2
    df3_dy2 = 2 * (y2 - q) / B ** 2

    df4_dx1 = -b ** 2 * ((x1 - x2) + (x1 - h)) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy1 = -b ** 2 * (x1 - x2) * (x1 - h) * ((y1 - y2) + (y1 - k)) / (a ** 2 * ((y1 - y2) * (y1 - k)) ** 2)
    df4_dx2 = b ** 2 * (x1 - h) / (a ** 2 * (y1 - y2) * (y1 - k))
    df4_dy2 = -b ** 2 * (x1 - x2) * (x1 - h) / (a ** 2 * (y1 - y2) ** 2 * (y1 - k))

    J = [[df1_dx1, df1_dy1, df1_dx2, df1_dy2],
         [df2_dx1, df2_dy1, df2_dx2, df2_dy2],
         [df3_dx1, df3_dy1, df3_dx2, df3_dy2],
         [df4_dx1, df4_dy1, df4_dx2, df4_dy2]]

    return J



def write_cst_paramters(key, ic_, oc_l, oc_r, projectDir, cell_type, sub_dir='', opt=False, solver='NGSolveMEVP'):
    """
    Writes cavity geometric data that can be imported into CST Studio

    Parameters
    ----------
    key: str, int
        Cavity marker
    ic_: list, array like
        Inner cavity cell geometric variables
    oc_l: list, array like
        Outer cavity cell geometric variables
    projectDir: str
        Project directory
    cell_type: str
        Single cell or multicell

    Returns
    -------

    """
    ic_ = update_alpha(ic_)
    oc_l = update_alpha(oc_l)
    if solver.lower() == 'ngsolvemevp':
        folder = 'NGSolveMEVP'
    else:
        folder = 'Optimisation'

    if cell_type is None:
        path = os.path.join(projectDir, 'Cavities', sub_dir, key, 'geometry', f'{key}.txt')

        with open(path, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'alpha', 'Aeq_e', 'Beq_e', 'ai_e', 'bi_e', 'Ri_e',
                         'L_e', 'Req', 'alpha_e', 'key']

            value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6], ic_[7],
                          oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], oc_l[6], oc_l[7], key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')

    else:
        path = os.path.join(projectDir, 'Cavities', sub_dir, key, 'geometry', f'{key}.txt')
        path_mc = os.path.join(projectDir, 'Cavities', sub_dir, key, 'geometry', f'{key}_Multicell.txt')

        with open(path, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'Aeq_e', 'Beq_e', 'ai_e', 'bi_e', 'Ri_e',
                         'L_e', 'Req_e', 'key']  # 'alpha_e', 'key']

            if cell_type == 'Mid Cell':
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req', 'key']  #'alpha', key]
            else:
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              oc_r[0], oc_r[1], oc_r[2], oc_r[3], oc_r[4], oc_r[5], ic_[6],  #oc_l[7],
                              oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], ic_[6],  #oc_l[7],
                              key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')

        with open(path_mc, 'w') as f:
            name_list = ['Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                         'Aeq_er', 'Beq_er', 'ai_er', 'bi_er', 'Ri_er', 'L_er', 'Req',  #'alpha_er',
                         'Aeq_el', 'Beq_el', 'ai_el', 'bi_el', 'Ri_el', 'L_el', 'Req', 'key']  #'alpha_el', 'key']

            if cell_type == 'Mid Cell':
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                              'Aeq', 'Beq', 'ai', 'bi', 'Ri', 'L', 'Req',  #'alpha',
                              key]
            else:
                value_list = [ic_[0], ic_[1], ic_[2], ic_[3], ic_[4], ic_[5], ic_[6],  #ic_[7],
                              oc_r[0], oc_r[1], oc_r[2], oc_r[3], oc_r[4], oc_r[5], ic_[6],  #oc_r[7],
                              oc_l[0], oc_l[1], oc_l[2], oc_l[3], oc_l[4], oc_l[5], ic_[6],  #oc_l[7],
                              key]

            for i in range(len(name_list)):
                if name_list[i] == 'key':
                    f.write(f'{name_list[i]} = "{0}" "{value_list[i]}"\n')
                else:
                    f.write(f'{name_list[i]} = "{value_list[i]}" ""\n')



def linspace(start, stop, step):
    """
    Like np.linspace but uses step instead of num
    This is inclusive to stop, so if start=1, stop=3, step=0.5
    Output is: array([1., 1.5, 2., 2.5, 3.])
    """
    if start < stop:
        ll = np.linspace(start, stop, int(np.ceil((stop - start) / abs(step) + 1)))
        if stop not in ll:
            ll = np.append(ll, stop)
        return ll
    else:
        ll = np.linspace(stop, start, int(np.ceil((start - stop) / abs(step) + 1)))
        if start not in ll:
            ll = np.append(ll, start)
        return ll


def lineTo(prevPt, nextPt, step, plot=False):
    if math.isclose(prevPt[0], nextPt[0], abs_tol=1e-6):
        # vertical line
        if prevPt[1] < nextPt[1]:
            py = linspace(prevPt[1], nextPt[1], step)
        else:
            py = linspace(nextPt[1], prevPt[1], step)
            py = py[::-1]
        px = np.ones(len(py)) * prevPt[0]

    elif math.isclose(prevPt[1], nextPt[1], abs_tol=1e-6):
        # horizontal line
        if prevPt[0] < nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step)
        else:
            px = linspace(nextPt[0], prevPt[0], step)

        py = np.ones(len(px)) * prevPt[1]
    else:
        # calculate angle to get appropriate step size for x and y
        ang = np.arctan((nextPt[1] - prevPt[1]) / (nextPt[0] - prevPt[0]))
        if prevPt[0] < nextPt[0] and prevPt[1] < nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step * np.cos(ang))
            py = linspace(prevPt[1], nextPt[1], step * np.sin(ang))
        elif prevPt[0] > nextPt[0] and prevPt[1] < nextPt[1]:
            px = linspace(nextPt[0], prevPt[0], step * np.cos(ang))
            px = px[::-1]
            py = linspace(prevPt[1], nextPt[1], step * np.sin(ang))
        elif prevPt[0] < nextPt[0] and prevPt[1] > nextPt[1]:
            px = linspace(prevPt[0], nextPt[0], step * np.cos(ang))
            py = linspace(nextPt[1], prevPt[1], step * np.sin(ang))
            py = py[::-1]
        else:
            px = linspace(nextPt[0], prevPt[0], step * np.cos(ang))
            px = px[::-1]
            py = linspace(nextPt[1], prevPt[1], step * np.sin(ang))
            py = py[::-1]
    if plot:
        plt.plot(px, py)  #, marker='x'

    return np.array([px, py]).T


def arcTo(h, k, a, b, step, start, end, plot=False):
    """

    Parameters
    ----------
    h: float, int
        x-position of the center
    k: float, int
        y-position of the center
    a float, int
        radius on the x-axis
    b float, int
        radius on the y-axis
    step: int
    start: list, ndarray
    end: list, ndarray
    plot: bool

    Returns
    -------

    """

    r_eff = (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b))) / 2  # <- Ramanujan approximate perimeter of ellipse
    # r_eff = max(a, b)
    x1, y1 = start
    x2, y2 = end
    # calculate parameter start and end points
    t1 = np.mod(np.arctan2((y1 - k) / b, (x1 - h) / a), 2 * np.pi)
    t2 = np.mod(np.arctan2((y2 - k) / b, (x2 - h) / a), 2 * np.pi)

    direction, shortest_distance = shortest_direction(t1, t2)
    if direction == 'clockwise':
        if t2 > t1:
            t1 += 2 * np.pi
        # t = np.linspace(t1, t2, int(np.ceil(C / step) * 2 * np.pi / abs(t2 - t1)))
        t = np.linspace(t1, t2, int(np.ceil(r_eff * abs(t2 - t1) / step)))
    else:
        if t1 > t2:
            t2 += 2 * np.pi
        # t = np.linspace(t1, t2, int(np.ceil(C / step) * 2 * np.pi / abs(t2 - t1)))
        t = np.linspace(t1, t2, int(np.ceil(r_eff * abs(t2 - t1) / step)))

    x = h + a * np.cos(t)
    y = k + b * np.sin(t)
    pts = np.column_stack((x, y))

    if plot:
        plt.plot(pts[:, 0], pts[:, 1], marker='x')  #

    return pts


def arcToTheta(h, k, a, b, start, end, t1, t2, step, plot=False):
    """
    Calculates the points on an arc from a start angle to an end angle

    Parameters
    ----------
    h: float, int
        x-position of the center
    k: float, int
        y-position of the center
    a float, int
        radius on the x-axis
    b float, int
        radius on the y-axis
    step: int, float
    start: list, ndarray
    end: list, ndarray
    plot: bool

    Returns
    -------

    """

    C = np.pi * (a + b)  # <- approximate perimeter of ellipse
    x1, y1 = start
    x2, y2 = end

    # plot center point
    if plot:
        plt.scatter(h, k, c='r')
    direction, shortest_distance = shortest_direction(t1, t2)
    if direction == 'clockwise':
        if t2 > t1:
            t1 += 2 * np.pi
        t = np.linspace(t1, t2, int(np.ceil(C / step) * 2 * np.pi / abs(t2 - t1)))
    else:
        if t1 > t2:
            t2 += 2 * np.pi
        t = np.linspace(t1, t2, int(np.ceil(C / step) * 2 * np.pi / abs(t2 - t1)))

    x = h + a * np.cos(t)
    y = k + b * np.sin(t)
    pts = np.column_stack((x, y))

    if plot:
        plt.plot(pts[:, 0], pts[:, 1])  #, marker='x'

    return pts


def shortest_direction(start_angle, end_angle):
    """

    Parameters
    ----------
    start_angle: float, int
        Start angle in radians
    end_angle: float, int
        End angle in radians

    Returns
    -------

    """
    # Ensure the angles are in the range [0, 2*pi]
    start_angle = np.mod(start_angle, 2 * np.pi)
    end_angle = np.mod(end_angle, 2 * np.pi)

    # Calculate the direct difference
    delta_theta = end_angle - start_angle

    # Normalize the difference to be within [-pi, pi]
    if delta_theta > np.pi:
        delta_theta -= 2 * np.pi
    elif delta_theta < -np.pi:
        delta_theta += 2 * np.pi

    # Determine the direction
    if delta_theta > 0:
        return "anticlockwise", delta_theta
    else:
        return "clockwise", -delta_theta


def add_point(cav, pt, pt_indx):
    cav.write(f"\nPoint({pt_indx}) = {{{pt[0]}, {pt[1]}, 0, {0.005}}};")
    pt_indx += 1

    return pt_indx


def add_line(cav, pt_indx, curve_indx):
    cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 2}, {pt_indx - 1}}};\n")
    curve_indx += 1
    return curve_indx

def add_bspline(cav, pt_indx, curve_indx, pts, n_cells=1):
    pts_indx_str = f'{pt_indx - 1}'

    for i in range(n_cells):
        for ii, pt in enumerate(pts):
            pts_indx_str += f', {pt_indx}'
            pt_indx = add_point(cav, pt, pt_indx)

        if i < n_cells-1:
            pts[:, 0] += pts[-1][0]

    cav.write(f"\nBSpline({curve_indx}) = {{{pts_indx_str}}};\n")

    curve_indx += 1
    return pt_indx, curve_indx

def add_bezierspline(cav, pt_indx, curve_indx, pts):
    pts_indx_str = f'{pt_indx - 1}'
    for ii, pt in enumerate(pts):
        pts_indx_str += f', {pt_indx}'
        pt_indx = add_point(cav, pt, pt_indx)

    cav.write(f"\nBezier({curve_indx}) = {{{pts_indx_str}}};\n")

    curve_indx += 1
    return pt_indx, curve_indx

def add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt):
    start_pt_tag = pt_indx - 1

    # write center point
    cav.write(f"\nPoint({pt_indx}) = {{{center_pt[0]}, {center_pt[1]}, 0, {0.005}}};")
    center_pt_tag = pt_indx
    pt_indx += 1

    # write major axis point
    cav.write(f"\nPoint({pt_indx}) = {{{majax_pt[0]}, {majax_pt[1]}, 0, {0.005}}};")
    majax_pt_tag = pt_indx
    pt_indx += 1

    # write end point
    cav.write(f"\nPoint({pt_indx}) = {{{end_pt[0]}, {end_pt[1]}, 0, {0.005}}};")
    end_pt_tag = pt_indx
    pt_indx += 1

    cav.write(
        f"\nEllipse({curve_indx}) = {{{start_pt_tag}, {center_pt_tag}, {majax_pt_tag}, {end_pt_tag}}};\n")  # Ellipse(tag) = {start_point_tag, center_point_tag, major_axis_point_tag, end_point_tag};
    curve_indx += 1

    return pt_indx, curve_indx



def write_cavity_geometry_cli(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False,
                              ignore_degenerate=False, plot=False, write=None, dimension=False,
                              contour=False, **kwargs):
    """
    Plot cavity geometry

    Parameters
    ----------
    tangent_check
    bc
    ax
    ignore_degenerate
    IC: list, ndarray
        Inner Cell geometric parameters list
    OC: list, ndarray
        Left outer Cell geometric parameters list
    OC_R: list, ndarray
        Right outer Cell geometric parameters list
    BP: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    n_cell: int
        Number of cavity cells
    scale: float
        Scale of the cavity geometry

    Returns
    -------

    """

    GEO = """
    """
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_aspect('equal')

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(IC)[:7] * scale * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(OC)[:7] * scale * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(OC_R)[:7] * scale * 1e-3

    L_bp = 4 * L_m
    if dimension or contour:
        L_bp = 1 * L_m

    if BP.lower() == 'both':
        L_bp_l = L_bp
        L_bp_r = L_bp
    elif BP.lower() == 'left':
        L_bp_l = L_bp
        L_bp_r = 0.000
    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = L_bp
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    step = 0.0005

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, tangent_check=tangent_check)
    x1el, y1el, x2el, y2el = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, tangent_check=tangent_check)
    x1, y1, x2, y2 = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, tangent_check=tangent_check)
    x1er, y1er, x2er, y2er = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    geo = []
    curve = []
    pt_indx = 1
    curve_indx = 1
    curve.append(curve_indx)
    if write:
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write(f'\nSetFactory("OpenCASCADE");\n')
            # SHIFT POINT TO START POINT
            start_point = [-shift, 0]
            pt_indx = add_point(cav, start_point, pt_indx)
            geo.append([start_point[1], start_point[0], 1])
            pts = lineTo(start_point, [-shift, Ri_el], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [-shift, Ri_el]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            if bc:
                # draw left boundary condition
                ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                        [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
                        [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)

            # ADD BEAM PIPE LENGTH
            if L_bp_l != 0:
                pts = lineTo(pt, [L_bp_l - shift, Ri_el], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0]])
                pt = [L_bp_l - shift, Ri_el]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 0])

            for n in range(1, n_cell + 1):
                if n == 1:
                    # DRAW ARC:
                    if plot and dimension:
                        ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                        ellipse = Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                          height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                          facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                                    xytext=(L_bp_l - shift, Ri_el + b_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))
                        ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                                    xytext=(L_bp_l - shift, Ri_el + b_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', ha='center', va='center')
                        ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_el + b_el]
                    majax_pt = [L_bp_l - shift + a_el, Ri_el + b_el]
                    end_pt = [-shift + x1el, y1el]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                    pt = [-shift + x1el, y1el]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2el, y2el], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2el, y2el]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    if plot and dimension:
                        ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                                 height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                                    xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                    xytext=(L_el + L_bp_l - shift, Req - B_el),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                                va='center', ha='center', rotation=90)

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

                    start_pt = pt
                    center_pt = [L_el + L_bp_l - shift, Req - B_el]
                    majax_pt = [L_el + L_bp_l - shift - A_el, Req - B_el]
                    end_pt = [L_bp_l + L_el - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                    pt = [L_bp_l + L_el - shift, Req]

                    for pp in pts:
                        geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    if n_cell == 1:
                        if L_bp_r > 0:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper

                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            if plot and dimension:
                                ax.scatter(L_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_er),
                                                                         width=2 * A_er,
                                                                         height=2 * B_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_er),
                                            xytext=(L_el + L_bp_l - shift + A_er, Req - B_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                            xytext=(L_el + L_bp_l - shift, Req - B_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n', ha='center',
                                        va='center')
                                ax.text(L_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                        va='center', ha='left', rotation=90)

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])

                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                  width=2 * a_er,
                                                  height=2 * b_er, angle=0, edgecolor='gray',
                                                  ls='--',
                                                  facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])

                            geo.append([pt[1], pt[0], 0])

                            # calculate new shift
                            shift = shift - (L_el + L_er)
                        else:
                            # EQUATOR ARC TO NEXT POINT
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            start_pt = pt
                            center_pt = [L_el + L_bp_l - shift, Req - B_er]
                            majax_pt = [L_el + L_bp_l - shift + A_er, Req - B_er]
                            end_pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                        [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                            pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                            # STRAIGHT LINE TO NEXT POINT
                            pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                            # for pp in pts:
                            #     geo.append([pp[1], pp[0], 0])
                            pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                            pt_indx = add_point(cav, pt, pt_indx)
                            curve_indx = add_line(cav, pt_indx, curve_indx)
                            curve.append(curve_indx)

                            geo.append([pt[1], pt[0], 0])

                            # ARC
                            # half of bounding box is required,
                            # start is the lower coordinate of the bounding box and end is the upper
                            if plot and dimension:
                                ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                         width=2 * a_er,
                                                                         height=2 * b_er, angle=0, edgecolor='gray',
                                                                         ls='--',
                                                                         facecolor='none')
                                ax.add_patch(ellipse)
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='<-', color='black'))
                                ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                            xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                            arrowprops=dict(arrowstyle='->', color='black'))

                                ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                        ha='center', va='center')
                                ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                        va='center', ha='center', rotation=90)

                            start_pt = pt
                            center_pt = [L_el + L_er + L_bp_l - shift, Ri_er + b_er]
                            majax_pt = [L_el + L_er + L_bp_l - shift + a_er - shift, Ri_er + b_er]
                            end_pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt,
                                                              end_pt)
                            curve.append(curve_indx)

                            pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                        [L_bp_l + L_el + L_er - shift, Ri_er])
                            pt = [L_bp_l + L_el + L_er - shift, Ri_er]

                            cav.write(f"\nPoint({pt_indx}) = {{{pt[0]}, {pt[1]}, 0, {0.1}}};")
                            pt_indx += 1

                            for pp in pts:
                                if (np.around(pp, 12) != np.around(pt, 12)).all():
                                    geo.append([pp[1], pp[0], 0])
                            geo.append([pt[1], pt[0], 0])

                    else:
                        # EQUATOR ARC TO NEXT POINT
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper

                        start_pt = pt
                        center_pt = [L_el + L_bp_l - shift, Req - B_m]
                        majax_pt = [L_el + L_bp_l - shift + a_m, Req - B_m]
                        end_pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                    [L_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                        pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # STRAIGHT LINE TO NEXT POINT
                        pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                        # for pp in pts:
                        #     geo.append([pp[1], pp[0], 0])
                        pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]

                        pt_indx = add_point(cav, pt, pt_indx)
                        curve_indx = add_line(cav, pt_indx, curve_indx)
                        curve.append(curve_indx)

                        geo.append([pt[1], pt[0], 0])

                        # ARC
                        # half of bounding box is required,
                        # start is the lower coordinate of the bounding box and end is the upper
                        start_pt = pt
                        center_pt = [L_el + L_m + L_bp_l - shift, Ri_m + b_m]
                        majax_pt = [L_el + L_m + L_bp_l - shift - A_m, Ri_m + b_m]
                        end_pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                        curve.append(curve_indx)

                        pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                    [L_bp_l + L_el + L_m - shift, Ri_m])
                        pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                        for pp in pts:
                            if (np.around(pp, 12) != np.around(pt, 12)).all():
                                geo.append([pp[1], pp[0], 0])
                        geo.append([pt[1], pt[0], 0])

                        # calculate new shift
                        shift = shift - (L_el + L_m)
                        # ic(shift)

                elif n > 1 and n != n_cell:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_m, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_m, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift + A_m, Req - B_m]
                    end_pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_m + L_m - x2 + 2 * L_bp_l - shift, y2])
                    pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_m + L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_m + L_m + L_bp_l - shift - a_m, Ri_m + b_m]
                    end_pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_m + L_m - shift, Ri_m])
                    pt = [L_bp_l + L_m + L_m - shift, Ri_m]

                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # calculate new shift
                    shift = shift - 2 * L_m
                else:
                    # DRAW ARC:
                    start_pt = pt
                    center_pt = [L_bp_l - shift, Ri_m + b_m]
                    majax_pt = [L_bp_l - shift + a_er, Ri_m + b_m]
                    end_pt = [-shift + x1, y1]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                    pt = [-shift + x1, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # DRAW LINE CONNECTING ARCS
                    pts = lineTo(pt, [-shift + x2, y2], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [-shift + x2, y2]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_m]
                    majax_pt = [L_m + L_bp_l - shift - A_er, Req - B_m]
                    end_pt = [L_bp_l + L_m - shift, Req]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                    pt = [L_bp_l + L_m - shift, Req]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_bp_l - shift, Req - B_er]
                    majax_pt = [L_m + L_bp_l - shift + A_er, Req - B_er]
                    end_pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0]])
                    pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]

                    pt_indx = add_point(cav, pt, pt_indx)
                    curve_indx = add_line(cav, pt_indx, curve_indx)
                    curve.append(curve_indx)

                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    start_pt = pt
                    center_pt = [L_m + L_er + L_bp_l - shift, Ri_er + b_er]
                    majax_pt = [L_m + L_er + L_bp_l - shift - a_er, Ri_er + b_er]
                    end_pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
                    curve.append(curve_indx)

                    pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_m + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    if L_bp_r > 0:
                        geo.append([pt[1], pt[0], 0])
                    else:
                        geo.append([pt[1], pt[0], 1])

            # BEAM PIPE
            # reset shift
            shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

            if L_bp_r > 0:  # if there's a problem, check here.
                pts = lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0], 0])
                pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]

                pt_indx = add_point(cav, pt, pt_indx)
                curve_indx = add_line(cav, pt_indx, curve_indx)
                curve.append(curve_indx)

                geo.append([pt[1], pt[0], 1])

            # END PATH
            pts = lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0],
                         step)  # to add beam pipe to right
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            # closing line
            cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

            # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
            # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
            geo.append([pt[1], pt[0], 2])

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

    # write geometry
    if write:
        try:
            df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
            # change point data precision
            df['r'] = df['r'].round(8)
            df['z'] = df['z'].round(8)
            # drop duplicates
            df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
            df.to_csv(write, sep='\t', index=False)
        except FileNotFoundError as e:
            error('Check file path:: ', e)

    # append start point
    # geo.append([start_point[1], start_point[0], 0])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0], 0])
    geo = np.array(geo)

    # if plot:
    #
    #     if dimension:
    #         top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
    #     else:
    #         # recenter asymmetric cavity to center
    #         shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n_cell - 1) * L_m) / 2
    #         if n_cell == 1:
    #             shift_to_center = L_er + L_bp_r
    #         else:
    #             shift_to_center = n_cell * L_m + L_bp_r
    #
    #         top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
    #         bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3, c=top[0].get_color(),
    #                          **kwargs)
    #
    #     # plot legend without duplicates
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())

    return ax


def write_cavity_geometry_cli_multicell(n_cell, multicell, BP, scale=1, ax=None, bc=None, tangent_check=False,
                                        ignore_degenerate=False, plot=False, write=None, dimension=False,
                                        contour=False, **kwargs):
    """
    Plot cavity geometry

    Parameters
    ----------
    tangent_check
    bc
    ax
    ignore_degenerate
    IC: list, ndarray
        Inner Cell geometric parameters list
    OC: list, ndarray
        Left outer Cell geometric parameters list
    OC_R: list, ndarray
        Right outer Cell geometric parameters list
    BP: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    n_cell: int
        Number of cavity cells
    scale: float
        Scale of the cavity geometry

    Returns
    -------

    """

    GEO = """
    """
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_aspect('equal')

    multicell_m = multicell * scale * 1e-3

    n_cell = len(multicell_m) // 16

    Ri_el = multicell_m[4]
    L_el = multicell_m[5]
    Ri_er = multicell_m[4 + 8 * (2 * n_cell - 1)]
    L_er = multicell_m[5 + 8 * (2 * n_cell - 1)]

    L_bp = 4 * L_el
    if dimension or contour:
        L_bp = 1 * L_el

    if BP.lower() == 'both':
        L_bp_l = L_bp
        L_bp_r = L_bp
    elif BP.lower() == 'left':
        L_bp_l = L_bp
        L_bp_r = 0.000
    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = L_bp
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    step = 0.0005

    # calculate shift
    shift = (L_bp_r + L_bp_l + sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)])) / 2

    geo = []
    curve = []
    pt_indx = 1
    curve_indx = 1
    curve.append(curve_indx)
    with open(write.replace('.n', '.geo'), 'w') as cav:
        cav.write(f'\nSetFactory("OpenCASCADE");\n')
        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        pt_indx = add_point(cav, start_point, pt_indx)
        geo.append([start_point[1], start_point[0], 1])
        pts = lineTo(start_point, [-shift, Ri_el], step)
        # for pp in pts:
        #     geo.append([pp[1], pp[0]])
        pt = [-shift, Ri_el]

        pt_indx = add_point(cav, pt, pt_indx)
        curve_indx = add_line(cav, pt_indx, curve_indx)
        curve.append(curve_indx)

        geo.append([pt[1], pt[0], 0])

        if bc:
            # draw left boundary condition
            ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                    [-shift - 0.2 * multicell_m[4 + 8], -shift - 0.2 * multicell_m[4 + 8]], [-0.5 * Ri_el, 0.5 * Ri_el],
                    [-shift - 0.4 * multicell_m[4 + 8], -shift - 0.4 * multicell_m[4 + 8]], [-0.1 * Ri_el, 0.1 * Ri_el],
                    c='b', lw=4, zorder=100)

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            pts = lineTo(pt, [L_bp_l - shift, Ri_el], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [L_bp_l - shift, Ri_el]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

        # calculate ncell
        # ncell = int(len(multicell_m)/8)

        for n in range(0, 2 * n_cell - 1, 2):
            A, B, a, b, Ri, L, Req, alpha = multicell_m[8 * n:8 * (n + 1)]

            df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp, tangent_check=tangent_check)
            x1, y1, x2, y2 = df[0]
            if not ignore_degenerate:
                msg = df[-2]
                if msg != 1:
                    error('Parameter set leads to degenerate geometry.')
                    # save figure of error
                    return

            # DRAW ARC:
            if plot and dimension:
                ax.scatter(L_bp_l - shift, Ri + b, c='r', ec='k', s=20)
                ellipse = Ellipse((L_bp_l - shift, Ri + b), width=2 * a,
                                  height=2 * b, angle=0, edgecolor='gray', ls='--',
                                  facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L_bp_l - shift + a, Ri + b),
                            xytext=(L_bp_l - shift, Ri + b),
                            arrowprops=dict(arrowstyle='->', color='black'))
                ax.annotate('', xy=(L_bp_l - shift, Ri),
                            xytext=(L_bp_l - shift, Ri + b),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L_bp_l - shift + a / 2, (Ri + b), f'{round(a, 2)}\n', ha='center', va='center')
                ax.text(L_bp_l - shift, (Ri + b / 2), f'{round(b, 2)}\n',
                        va='center', ha='center', rotation=90)

            start_pt = pt
            center_pt = [L_bp_l - shift, Ri + b]
            majax_pt = [L_bp_l - shift + a, Ri + b]
            end_pt = [-shift + x1, y1]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            pts = arcTo(L_bp_l - shift, Ri + b, a, b, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]

            for pp in pts:
                geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [-shift + x2, y2]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            if plot and dimension:
                ax.scatter(L + L_bp_l - shift, Req - B, c='r', ec='k', s=20)
                ellipse = Ellipse((L + L_bp_l - shift, Req - B), width=2 * A,
                                  height=2 * B, angle=0, edgecolor='gray', ls='--',
                                  facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L + L_bp_l - shift, Req - B),
                            xytext=(L + L_bp_l - shift - A, Req - B),
                            arrowprops=dict(arrowstyle='<-', color='black'))
                ax.annotate('', xy=(L + L_bp_l - shift, Req),
                            xytext=(L + L_bp_l - shift, Req - B),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L + L_bp_l - shift - A / 2, (Req - B), f'{round(A, 2)}\n', ha='center', va='center')
                ax.text(L + L_bp_l - shift, (Req - B / 2), f'{round(B, 2)}\n',
                        va='center', ha='center', rotation=90)

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

            start_pt = pt
            center_pt = [L + L_bp_l - shift, Req - B]
            majax_pt = [L + L_bp_l - shift - A, Req - B]
            end_pt = [L_bp_l + L - shift, Req]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            pts = arcTo(L + L_bp_l - shift, Req - B, A, B, step, pt, [L_bp_l + L - shift, Req])
            pt = [L_bp_l + L - shift, Req]

            for pp in pts:
                geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            A_left, B_left, a_left, b_left, Ri_left, L_left, Req_left, alpha_left = A, B, a, b, Ri, L, Req, alpha
            ###################################################
            A, B, a, b, Ri, L, Req, alpha = multicell_m[8 * (n + 1):8 * (n + 2)]
            df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp, tangent_check=tangent_check)
            x1, y1, x2, y2 = df[0]
            if not ignore_degenerate:
                msg = df[-2]
                if msg != 1:
                    error('Parameter set leads to degenerate geometry.')
                    # save figure of error
                    return

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper

            start_pt = pt
            center_pt = [L_left + L_bp_l - shift, Req - B]
            majax_pt = [L_left + L_bp_l - shift + A, Req - B]
            end_pt = [L_left + L - x2 + 2 * L_bp_l - shift, y2]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            pts = arcTo(L_left + L_bp_l - shift, Req - B, A, B, step, pt,
                        [L_left + L - x2 + 2 * L_bp_l - shift, y2])
            pt = [L_left + L - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_left + L - x1 + 2 * L_bp_l - shift, y1], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [L_left + L - x1 + 2 * L_bp_l - shift, y1]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            start_pt = pt
            center_pt = [L_left + L + L_bp_l - shift, Ri + b]
            majax_pt = [L_left + L + L_bp_l - shift - a, Ri + b]
            end_pt = [L_bp_l + L_left + L - shift, Ri]
            pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
            curve.append(curve_indx)

            pts = arcTo(L_left + L + L_bp_l - shift, Ri + b, a, b, step, pt,
                        [L_bp_l + L_left + L - shift, Ri])
            pt = [L_bp_l + L_left + L - shift, Ri]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # calculate new shift
            shift = shift - (L_left + L)
            # ic(shift)

        # BEAM PIPE
        # reset shift
        shift = (L_bp_r + L_bp_l + sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)])) / 2

        if L_bp_r > 0:  # if there's a problem, check here.
            pts = lineTo(pt, [L_bp_r + L_bp_l + sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)]) - shift, Ri_er],
                         step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)]) + L_bp_l + L_bp_r - shift, Ri_er]

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 1])

        # END PATH
        pts = lineTo(pt, [sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)]) + L_bp_l + L_bp_r - shift, 0],
                     step)  # to add beam pipe to right
        # for pp in pts:
        #     geo.append([pp[1], pp[0], 0])
        pt = [sum([multicell_m[5 + 8 * n] for n in range(2 * n_cell)]) + L_bp_l + L_bp_r - shift, 0]

        pt_indx = add_point(cav, pt, pt_indx)
        curve_indx = add_line(cav, pt_indx, curve_indx)
        curve.append(curve_indx)

        # closing line
        cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

        geo.append([pt[1], pt[0], 2])

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

    # write geometry
    if write:
        try:
            df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
            # change point data precision
            df['r'] = df['r'].round(8)
            df['z'] = df['z'].round(8)
            # drop duplicates
            df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
            df.to_csv(write, sep='\t', index=False)
        except FileNotFoundError as e:
            error('Check file path:: ', e)

    # append start point
    # geo.append([start_point[1], start_point[0], 0])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L, shift + 0.2 * L], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L, shift + 0.4 * L], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0], 0])
    geo = np.array(geo)

    if plot:

        if dimension:
            top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
        else:
            # recenter asymmetric cavity to center
            shift_left = (L_bp_l + L_bp_r + L + L_er + 2 * (n - 1) * L) / 2
            if n_cell == 1:
                shift_to_center = L_er + L_bp_r
            else:
                shift_to_center = n_cell * L + L_bp_r

            top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
            bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3, c=top[0].get_color(),
                             **kwargs)

        # plot legend without duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    return ax
def write_cavity_geometry_cli_quarter(cell, bp=False, scale=1, ax=None, bc=None, tangent_check=False,
                                        ignore_degenerate=False, plot=False, write=None, dimension=False,
                                        contour=False, **kwargs):
    """
    Plot cavity geometry

    Parameters
    ----------
    tangent_check
    bc
    ax
    ignore_degenerate
    IC: list, ndarray
        Inner Cell geometric parameters list
    OC: list, ndarray
        Left outer Cell geometric parameters list
    OC_R: list, ndarray
        Right outer Cell geometric parameters list
    BP: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    n_cell: int
        Number of cavity cells
    scale: float
        Scale of the cavity geometry

    Returns
    -------

    """

    GEO = """
    """
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_aspect('equal')

    cell_m = np.array(cell) * scale * 1e-3
    A, B, a, b, Ri, L, Req = cell_m[:7]


    L_bp = 4 * L
    if dimension or contour:
        L_bp = 1 * L

    if bp:
        L_bp_l = L_bp
    else:
        L_bp_l = 0

    step = 0.0005

    # calculate shift
    shift = (L_bp_l + cell_m[5]) / 2

    geo = []
    curve = []
    pt_indx = 1
    curve_indx = 1
    curve.append(curve_indx)
    with open(write.replace('.n', '.geo'), 'w') as cav:
        cav.write(f'\nSetFactory("OpenCASCADE");\n')

        # define parameters
        cav.write(f'\nA = DefineNumber[{A}, Name "Parameters/Equator ellipse major axis"];')
        cav.write(f'\nB = DefineNumber[{B}, Name "Parameters/Equator ellipse minor axis"];')
        cav.write(f'\na = DefineNumber[{a}, Name "Parameters/Iris ellipse major axis"];')
        cav.write(f'\nb = DefineNumber[{b}, Name "Parameters/Iris ellipse minor axis"];')
        cav.write(f'\nRi = DefineNumber[{Ri}, Name "Parameters/Iris radius"];')
        cav.write(f'\nL = DefineNumber[{L}, Name "Parameters/Half cell length"];')
        cav.write(f'\nReq = DefineNumber[{Req}, Name "Parameters/Equator radius"];\n')

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        pt_indx = add_point(cav, start_point, pt_indx)
        geo.append([start_point[1], start_point[0], 1])

        pt = [-shift, 'Ri']

        pt_indx = add_point(cav, pt, pt_indx)
        curve_indx = add_line(cav, pt_indx, curve_indx)
        curve.append(curve_indx)

        geo.append([pt[1], pt[0], 0])

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            pt = [L_bp_l - shift, 'Ri']

            pt_indx = add_point(cav, pt, pt_indx)
            curve_indx = add_line(cav, pt_indx, curve_indx)
            curve.append(curve_indx)

            geo.append([pt[1], pt[0], 0])

        df = tangent_coords(A, B, a, b, Ri, L, Req, L_bp_l, tangent_check=tangent_check)
        x1, y1, x2, y2 = df[0]
        if not ignore_degenerate:
            msg = df[-2]
            if msg != 1:
                error('Parameter set leads to degenerate geometry.')
                # save figure of error
                return

        start_pt = pt
        center_pt = [L_bp_l - shift, 'Ri + b']
        majax_pt = [f'{L_bp_l - shift} + a', 'Ri + b']
        end_pt = [-shift + x1, y1]
        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
        curve.append(curve_indx)

        # pts = arcTo(L_bp_l - shift, Ri + b, a, b, step, pt, [-shift + x1, y1])
        pt = [-shift + x1, y1]

        # for pp in pts:
        #     geo.append([pp[1], pp[0], 0])
        # geo.append([pt[1], pt[0], 0])

        # DRAW LINE CONNECTING ARCS
        pt = [-shift + x2, y2]

        pt_indx = add_point(cav, pt, pt_indx)
        curve_indx = add_line(cav, pt_indx, curve_indx)
        curve.append(curve_indx)

        geo.append([pt[1], pt[0], 0])

        # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT

        start_pt = pt
        center_pt = [f'L + {L_bp_l - shift}', 'Req - B']
        majax_pt = [f'L + {L_bp_l - shift} - A', 'Req - B']
        end_pt = [f'{L_bp_l - shift} + L', 'Req']
        pt_indx, curve_indx = add_ellipse(cav, pt_indx, curve_indx, start_pt, center_pt, majax_pt, end_pt)
        curve.append(curve_indx)

        # pts = arcTo(L + L_bp_l - shift, Req - B, A, B, step, pt, [L_bp_l + L - shift, Req])
        pt = [f'{L_bp_l - shift} + L', 'Req']

        # for pp in pts:
        #     geo.append([pp[1], pp[0], 0])
        geo.append([pt[1], pt[0], 0])

        # BEAM PIPE
        # reset shift
        shift = (L_bp_l +cell_m[5]) / 2

        # END PATH
        # pts = lineTo(pt, [cell_m[5]+ L_bp_l + - shift, 0],
        #              step)  # to add beam pipe to right

        # for pp in pts:
        #     geo.append([pp[1], pp[0], 0])
        pt = [cell_m[5] + L_bp_l - shift, 0]

        pt_indx = add_point(cav, pt, pt_indx)
        curve_indx = add_line(cav, pt_indx, curve_indx)
        curve.append(curve_indx)

        # closing line
        cav.write(f"\nLine({curve_indx}) = {{{pt_indx - 1}, {1}}};\n")

        geo.append([pt[1], pt[0], 2])

        pmcs = [1]
        axis = [curve[-1]]
        pecs = [x for x in curve if (x not in pmcs and x not in axis)]

        cav.write(f'\nPhysical Line("PEC") = {pecs};'.replace('[', '{').replace(']', '}'))
        cav.write(f'\nPhysical Line("PMC") = {pmcs};'.replace('[', '{').replace(']', '}'))
        cav.write(f'\nPhysical Line("AXI") = {axis};'.replace('[', '{').replace(']', '}'))

        cav.write(f"\n\nCurve Loop(1) = {curve};".replace('[', '{').replace(']', '}'))
        cav.write(f"\nPlane Surface(1) = {{{1}}};")
        cav.write(f"\nReverse Surface {1};")
        cav.write(f'\nPhysical Surface("Domain") = {1};')

    # write geometry
    if write:
        try:
            df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
            # change point data precision
            df['r'] = df['r'].round(8)
            df['z'] = df['z'].round(8)
            # drop duplicates
            df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
            df.to_csv(write, sep='\t', index=False)
        except FileNotFoundError as e:
            error('Check file path:: ', e)

    # append start point
    # geo.append([start_point[1], start_point[0], 0])

    # if bc:
    #     # draw right boundary condition
    #     ax.plot([shift, shift], [-Ri_er, Ri_er],
    #             [shift + 0.2 * L, shift + 0.2 * L], [-0.5 * Ri_er, 0.5 * Ri_er],
    #             [shift + 0.4 * L, shift + 0.4 * L], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0], 0])
    geo = np.array(geo)

    # if plot:
    #
    #     if dimension:
    #         top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
    #     else:
    #         # recenter asymmetric cavity to center
    #         shift_left = (L_bp_l + L_bp_r + L + L_er + 2 * (n - 1) * L) / 2
    #         if n_cell == 1:
    #             shift_to_center = L_er + L_bp_r
    #         else:
    #             shift_to_center = n_cell * L + L_bp_r
    #
    #         top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
    #         bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3, c=top[0].get_color(),
    #                          **kwargs)
    #
    #     # plot legend without duplicates
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = dict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())

    return ax


def write_cavity_geometry_cli_wo_gmsh(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False,
                                      ignore_degenerate=False, plot=False, write=None, dimension=False,
                                      contour=False, **kwargs):
    """
    Plot cavity geometry

    Parameters
    ----------
    tangent_check
    bc
    ax
    ignore_degenerate
    IC: list, ndarray
        Inner Cell geometric parameters list
    OC: list, ndarray
        Left outer Cell geometric parameters list
    OC_R: list, ndarray
        Right outer Cell geometric parameters list
    BP: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    n_cell: int
        Number of cavity cells
    scale: float
        Scale of the cavity geometry

    Returns
    -------

    """

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_aspect('equal')

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(IC)[:7] * scale * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(OC)[:7] * scale * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(OC_R)[:7] * scale * 1e-3

    L_bp = 4 * L_m
    if dimension or contour:
        L_bp = 1 * L_m

    if BP.lower() == 'both':
        L_bp_l = L_bp
        L_bp_r = L_bp
    elif BP.lower() == 'left':
        L_bp_l = L_bp
        L_bp_r = 0.000
    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = L_bp
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    step = 0.0005

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, tangent_check=tangent_check)
    x1el, y1el, x2el, y2el = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, tangent_check=tangent_check)
    x1, y1, x2, y2 = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, tangent_check=tangent_check)
    x1er, y1er, x2er, y2er = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    geo = []

    # SHIFT POINT TO START POINT
    start_point = [-shift, 0]
    geo.append([start_point[1], start_point[0], 1])

    pts = lineTo(start_point, [-shift, Ri_el], step)
    # for pp in pts:
    #     geo.append([pp[1], pp[0]])
    pt = [-shift, Ri_el]
    geo.append([pt[1], pt[0], 0])

    if bc:
        # draw left boundary condition
        ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
                [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)

    # ADD BEAM PIPE LENGTH
    if L_bp_l != 0:
        pts = lineTo(pt, [L_bp_l - shift, Ri_el], step)
        # for pp in pts:
        #     geo.append([pp[1], pp[0]])
        pt = [L_bp_l - shift, Ri_el]

        geo.append([pt[1], pt[0], 0])

    for n in range(1, n_cell + 1):
        if n == 1:
            # DRAW ARC:
            if plot and dimension:
                ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                ellipse = plt.matplotlib.patches.Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                                         height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                                         facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                            xytext=(L_bp_l - shift, Ri_el + b_el),
                            arrowprops=dict(arrowstyle='->', color='black'))
                ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                            xytext=(L_bp_l - shift, Ri_el + b_el),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', ha='center', va='center')
                ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                        va='center', ha='center', rotation=90)

            pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
            pt = [-shift + x1el, y1el]
            for pp in pts:
                geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2el, y2el], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [-shift + x2el, y2el]
            geo.append([pt[1], pt[0], 0])

            if plot and dimension:
                ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                         height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                         facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                            xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                            arrowprops=dict(arrowstyle='<-', color='black'))
                ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                            xytext=(L_el + L_bp_l - shift, Req - B_el),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', ha='center', va='center')
                ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                        va='center', ha='center', rotation=90)

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
            pt = [L_bp_l + L_el - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            if n_cell == 1:
                if L_bp_r > 0:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    if plot and dimension:
                        ax.scatter(L_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_er), width=2 * A_er,
                                                                 height=2 * B_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_er),
                                    xytext=(L_el + L_bp_l - shift + A_er, Req - B_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                                    xytext=(L_el + L_bp_l - shift, Req - B_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n', ha='center',
                                va='center')
                        ax.text(L_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                va='center', ha='left', rotation=90)

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er])

                    if plot and dimension:
                        ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                 width=2 * a_er,
                                                                 height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                    xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                ha='center', va='center')
                        ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                va='center', ha='center', rotation=90)

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])

                    geo.append([pt[1], pt[0], 0])

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    # for pp in pts:
                    #     geo.append([pp[1], pp[0], 0])
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0], 0])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    if plot and dimension:
                        ax.scatter(L_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                 width=2 * a_er,
                                                                 height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    xytext=(L_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + L_er + L_bp_l - shift, Ri_er),
                                    xytext=(L_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                ha='center', va='center')
                        ax.text(L_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                va='center', ha='center', rotation=90)

                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 0])
                    geo.append([pt[1], pt[0], 0])

            else:
                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 0])
                geo.append([pt[1], pt[0], 0])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                # for pp in pts:
                #     geo.append([pp[1], pp[0], 0])
                pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                geo.append([pt[1], pt[0], 0])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_el + L_m - shift, Ri_m])
                pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 0])
                geo.append([pt[1], pt[0], 0])

                # calculate new shift
                shift = shift - (L_el + L_m)
                # ic(shift)

        elif n > 1 and n != n_cell:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])

            geo.append([pt[1], pt[0], 0])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                        [L_m + L_m - x2 + 2 * L_bp_l - shift, y2])
            pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])

            geo.append([pt[1], pt[0], 0])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
            geo.append([pt[1], pt[0], 0])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                        [L_bp_l + L_m + L_m - shift, Ri_m])
            pt = [L_bp_l + L_m + L_m - shift, Ri_m]

            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # calculate new shift
            shift = shift - 2 * L_m
        else:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0], 0])
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0], 0])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                        [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
            pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            geo.append([pt[1], pt[0], 0])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
            # for pp in pts:
            #     geo.append([pp[1], pp[0]])
            pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
            geo.append([pt[1], pt[0], 0])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                        [L_bp_l + L_m + L_er - shift, Ri_er])
            pt = [L_bp_l + L_m + L_er - shift, Ri_er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 0])
            if L_bp_r > 0:
                geo.append([pt[1], pt[0], 0])
            else:
                geo.append([pt[1], pt[0], 1])

    # BEAM PIPE
    # reset shift
    shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

    if L_bp_r > 0:  # if there's a problem, check here.
        pts = lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
        # for pp in pts:
        #     geo.append([pp[1], pp[0], 0])
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
        geo.append([pt[1], pt[0], 1])

    # END PATH
    pts = lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0],
                 step)  # to add beam pipe to right
    # for pp in pts:
    #     geo.append([pp[1], pp[0], 0])
    pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
    # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
    # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
    geo.append([pt[1], pt[0], 2])

    # write geometry
    if write:
        try:
            df = pd.DataFrame(geo, columns=['r', 'z', 'bc'])
            # change point data precision
            df['r'] = df['r'].round(8)
            df['z'] = df['z'].round(8)
            # drop duplicates
            df.drop_duplicates(subset=['r', 'z'], inplace=True, keep='last')
            df.to_csv(write, sep='\t', index=False)
        except FileNotFoundError as e:
            error('Check file path:: ', e)

    # append start point
    # geo.append([start_point[1], start_point[0], 0])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0], 0])
    geo = np.array(geo)

    if plot:

        if dimension:
            top = ax.plot(geo[:, 1] * 1e3, geo[:, 0] * 1e3, **kwargs)
        else:
            # recenter asymmetric cavity to center
            shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
            if n_cell == 1:
                shift_to_center = L_er + L_bp_r
            else:
                shift_to_center = n_cell * L_m + L_bp_r

            top = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, geo[:, 0] * 1e3, **kwargs)
            bottom = ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3, c=top[0].get_color(),
                             **kwargs)

        # plot legend without duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    return ax


def write_cavity_geometry_cli_flattop(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False,
                                      ignore_degenerate=False, plot=False, write=None, dimension=False,
                                      contour=False, **kwargs):
    """
    Write cavity geometry

    Parameters
    ----------
    BP
    OC_R
    OC
    ignore_degenerate
    file_path: str
        File path to write geometry to
    n_cell: int
        Number of cavity cells
    mid_cell: list, ndarray
        Array of cavity middle cells' geometric parameters
    end_cell_left: list, ndarray
        Array of cavity left end cell's geometric parameters
    end_cell_right: list, ndarray
        Array of cavity left end cell's geometric parameters
    beampipe: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    plot: bool
        If True, the cavity geometry is plotted for viewing

    Returns
    -------

    """

    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_aspect('equal')

    if OC is None:
        end_cell_left = IC

    if OC_R is None:
        if OC is None:
            OC_R = IC
        else:
            OC_R = OC

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req, lft = np.array(IC[:8]) * scale * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req, lft_el = np.array(OC[:8]) * scale * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req, lft_er = np.array(OC_R[:8]) * scale * 1e-3

    step = 0.005

    L_bp = 4 * L_m
    if dimension or contour:
        L_bp = 1 * L_m

    if BP.lower() == 'both':
        L_bp_l = L_bp
        L_bp_r = L_bp
    elif BP.lower() == 'left':
        L_bp_l = L_bp
        L_bp_r = 0.000
    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = L_bp
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er + (n_cell - 2) * lft + lft_el + lft_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el
    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, lft_el / 2, tangent_check=tangent_check)
    x1el, y1el, x2el, y2el = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, lft / 2, tangent_check=tangent_check)
    x1, y1, x2, y2 = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, lft_er / 2, tangent_check=tangent_check)
    x1er, y1er, x2er, y2er = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    geo = []

    # SHIFT POINT TO START POINT
    start_point = [-shift, 0]
    geo.append([start_point[1], start_point[0], 3])

    lineTo(start_point, [-shift, Ri_el], step)
    pt = [-shift, Ri_el]
    geo.append([pt[1], pt[0], 2])

    # ADD BEAM PIPE LENGTH
    if L_bp_l != 0:
        lineTo(pt, [L_bp_l - shift, Ri_el], step)
        pt = [L_bp_l - shift, Ri_el]

        geo.append([pt[1], pt[0], 2])

    for n in range(1, n_cell + 1):
        if n == 1:
            # DRAW ARC:
            if plot and dimension:
                ax.scatter(L_bp_l - shift, Ri_el + b_el, c='r', ec='k', s=20)
                ellipse = plt.matplotlib.patches.Ellipse((L_bp_l - shift, Ri_el + b_el), width=2 * a_el,
                                                         height=2 * b_el, angle=0, edgecolor='gray', ls='--',
                                                         facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L_bp_l - shift + a_el, Ri_el + b_el),
                            xytext=(L_bp_l - shift, Ri_el + b_el),
                            arrowprops=dict(arrowstyle='->', color='black'))
                ax.annotate('', xy=(L_bp_l - shift, Ri_el),
                            xytext=(L_bp_l - shift, Ri_el + b_el),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L_bp_l - shift + a_el / 2, (Ri_el + b_el), f'{round(a_el, 2)}\n', va='center', ha='center')
                ax.text(L_bp_l - shift, (Ri_el + b_el / 2), f'{round(b_el, 2)}\n',
                        va='center', ha='center', rotation=90)

            pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
            pt = [-shift + x1el, y1el]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2el, y2el], step)
            pt = [-shift + x2el, y2el]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            if plot and dimension:
                ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                ellipse = plt.matplotlib.patches.Ellipse((L_el + L_bp_l - shift, Req - B_el), width=2 * A_el,
                                                         height=2 * B_el, angle=0, edgecolor='gray', ls='--',
                                                         facecolor='none')
                ax.add_patch(ellipse)
                ax.annotate('', xy=(L_el + L_bp_l - shift, Req - B_el),
                            xytext=(L_el + L_bp_l - shift - A_el, Req - B_el),
                            arrowprops=dict(arrowstyle='<-', color='black'))
                ax.annotate('', xy=(L_el + L_bp_l - shift, Req),
                            xytext=(L_el + L_bp_l - shift, Req - B_el),
                            arrowprops=dict(arrowstyle='->', color='black'))

                ax.text(L_el + L_bp_l - shift - A_el / 2, (Req - B_el), f'{round(A_el, 2)}\n', va='center', ha='center')
                ax.text(L_el + L_bp_l - shift, (Req - B_el / 2), f'{round(B_el, 2)}\n',
                        va='center', ha='center', rotation=90)

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
            pt = [L_bp_l + L_el - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # flat top
            pts = lineTo(pt, [L_bp_l + L_el + lft_el - shift, Req], step)
            pt = [L_bp_l + L_el + lft_el - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            if plot and dimension:
                ax.scatter(L_el + L_bp_l - shift, Req - B_el, c='r', ec='k', s=20)
                # Plot the straight line
                line_start = [L_bp_l + L_el - shift, Req]
                line_end = pt
                ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r', zorder=200)

                ax.annotate('', xy=(line_start[0], line_start[1] + 0.5), xytext=(line_end[0], line_end[1] + 0.5),
                            arrowprops=dict(arrowstyle='<->', color='black'))
                ax.text((line_start[0] + line_end[0]) / 2, line_start[1] + 0.7,
                        f'{round(lft_el, 2)}\n', va='center', ha='center')

            if n_cell == 1:
                if L_bp_r > 0:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + lft_el + L_er - x2er + + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    if plot and dimension:
                        ax.scatter(L_el + lft_el + L_bp_l - shift, Req - B_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + lft_el + L_bp_l - shift, Req - B_er),
                                                                 width=2 * A_er,
                                                                 height=2 * B_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + lft_el + L_bp_l - shift, Req - B_er),
                                    xytext=(L_el + lft_el + L_bp_l - shift + A_er, Req - B_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + lft_el + L_bp_l - shift, Req),
                                    xytext=(L_el + lft_el + L_bp_l - shift, Req - B_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + lft_el + L_bp_l - shift + A_er / 2, (Req - B_er), f'{round(A_er, 2)}\n',
                                va='center', ha='center')
                        ax.text(L_el + lft_el + L_bp_l - shift, (Req - B_er / 2), f'{round(B_er, 2)}\n',
                                va='center', ha='left', rotation=90)

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0], 2])

                    if plot and dimension:
                        ax.scatter(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                 width=2 * a_er,
                                                                 height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    xytext=(L_el + lft_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er),
                                    xytext=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + lft_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                va='center', ha='center')
                        ax.text(L_el + lft_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                va='center', ha='center', rotation=90)

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_el + lft_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])

                    geo.append([pt[1], pt[0], 2])

                    # calculate new shift
                    shift = shift - (L_el + lft_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0], 2])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    if plot and dimension:
                        ax.scatter(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, c='r', ec='k', s=20)
                        ellipse = plt.matplotlib.patches.Ellipse((L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                                                 width=2 * a_er,
                                                                 height=2 * b_er, angle=0, edgecolor='gray', ls='--',
                                                                 facecolor='none')
                        ax.add_patch(ellipse)
                        ax.annotate('', xy=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    xytext=(L_el + lft_el + L_er + L_bp_l - shift - a_er, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='<-', color='black'))
                        ax.annotate('', xy=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er),
                                    xytext=(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er),
                                    arrowprops=dict(arrowstyle='->', color='black'))

                        ax.text(L_el + lft_el + L_er + L_bp_l - shift - a_er / 2, (Ri_er + b_er), f'{round(a_er, 2)}\n',
                                va='center', ha='center')
                        ax.text(L_el + lft_el + L_er + L_bp_l - shift, (Ri_er + b_er / 2), f'{round(b_er, 2)}\n',
                                va='center', ha='center', rotation=90)

                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, Ri_er])

                    pt = [L_bp_l + L_el + lft_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])

                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_el + lft_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0], 2])
                    geo.append([pt[1], pt[0], 2])
            else:
                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_bp_l + L_el + lft_el - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                pt = [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + lft_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_el + lft_el + L_m - shift, Ri_m])
                pt = [L_bp_l + L_el + lft_el + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0], 2])
                geo.append([pt[1], pt[0], 2])

                # calculate new shift
                shift = shift - (L_el + L_m + lft_el)
                # ic(shift)

        elif n > 1 and n != n_cell:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            for pp in pts:
                geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # flat top
            pts = lineTo(pt, [L_bp_l + L_m + lft - shift, Req], step)
            pt = [L_bp_l + L_m + lft - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l + lft - shift, Req - B_m, A_m, B_m, step, pt,
                        [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2])
            pt = [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])

            geo.append([pt[1], pt[0], 2])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1], step)
            pt = [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1]
            for pp in pts:
                geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_m + lft + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                        [L_bp_l + L_m + L_m + lft - shift, Ri_m])
            pt = [L_bp_l + L_m + L_m + lft - shift, Ri_m]

            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # calculate new shift
            shift = shift - 2 * L_m - lft
        else:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # flat top
            pts = lineTo(pt, [L_bp_l + L_m + lft_er - shift, Req], step)
            pt = [L_bp_l + L_m + lft_er - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + lft_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                        [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er])
            pt = [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
            pt = [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_er + lft_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                        [L_bp_l + L_m + L_er + lft_er - shift, Ri_er])
            pt = [L_bp_l + L_m + L_er + lft_er - shift, Ri_er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0], 2])
            geo.append([pt[1], pt[0], 2])

    # BEAM PIPE
    # reset shift

    shift = (L_bp_r + L_bp_l + L_el + lft_el + (n_cell - 1) * 2 * L_m + (n_cell - 2) * lft + L_er + lft_er) / 2
    pts = lineTo(pt, [
        L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + (n_cell - 2) * lft + lft_el + lft_er + L_el + L_er - shift,
        Ri_er], step)

    if L_bp_r != 0:
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cell - 2) * lft + lft_el + lft_er - shift,
              Ri_er]
        for pp in pts:
            geo.append([pp[1], pp[0], 2])
        geo.append([pt[1], pt[0], 2])

    # END PATH
    pts = lineTo(pt, [
        2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0],
                 step)  # to add beam pipe to right
    pt = [2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0]
    # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
    geo.append([pt[1], pt[0], 2])

    # CLOSE PATH
    lineTo(pt, start_point, step)
    geo.append([pt[1], pt[0], 3])

    # write geometry
    if write:
        try:
            df = pd.DataFrame(geo, columns=['r', 'z'])
            df.to_csv(write, sep='\t', index=False)
        except FileNotFoundError as e:
            error('Check file path:: ', e)

    # append start point
    geo.append([start_point[1], start_point[0], 3])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0], 3])

    geo = np.array(geo)

    if plot:
        if dimension:
            top = ax.plot(geo[:, 1], geo[:, 0], **kwargs)
        else:
            # recenter asymmetric cavity to center
            shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2 * (n - 1) * L_m) / 2
            if n_cell == 1:
                shift_to_center = L_er + L_bp_r
            else:
                shift_to_center = n_cell * L_m + L_bp_r

            top = ax.plot(geo[:, 1] - shift_left + shift_to_center, geo[:, 0], **kwargs)
            # bottom = ax.plot(geo[:, 1] - shift_left + shift_to_center, -geo[:, 0], c=top[0].get_color(), **kwargs)

        # plot legend wthout duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        return ax



def writeCavityForMultipac(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None, beampipe='none',
                           plot=True, unit=1e-3, scale=1):
    """
    Write cavity geometry

    Parameters
    ----------
    file_path: str
        File path to write geometry to
    n_cell: int
        Number of cavity cells
    mid_cell: list, ndarray
        Array of cavity middle cells' geometric parameters
    end_cell_left: list, ndarray
        Array of cavity left end cell's geometric parameters
    end_cell_right: list, ndarray
        Array of cavity left end cell's geometric parameters
    beampipe: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    plot: bool
        If True, the cavity geometry is plotted for viewing

    scale
    unit

    Returns
    -------

    """

    if plot:
        plt.rcParams["figure.figsize"] = (12, 2)

    if end_cell_left is None:
        end_cell_left = mid_cell

    if end_cell_right is None:
        if end_cell_left is None:
            end_cell_right = mid_cell
        else:
            end_cell_right = end_cell_left

    us = unit * scale
    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(mid_cell[:7]) * us
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(end_cell_left[:7]) * us
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(end_cell_right[:7]) * us

    step = 0.005

    if beampipe.lower() == 'both':
        L_bp_l = 4 * L_m
        L_bp_r = 4 * L_m
    elif beampipe.lower() == 'none':
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #
    elif beampipe.lower() == 'left':
        L_bp_l = 4 * L_m
        L_bp_r = 0.000
    elif beampipe.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4 * L_m
    else:
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l)
    x1el, y1el, x2el, y2el = df[0]
    msg = df[-2]
    if msg != 1:
        error('Parameter set leads to degenerate geometry.')
        # save figure of error
        return

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l)
    x1, y1, x2, y2 = df[0]
    msg = df[-2]
    if msg != 1:
        error('Parameter set leads to degenerate geometry.')
        # save figure of error
        return

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r)
    x1er, y1er, x2er, y2er = df[0]
    msg = df[-2]
    if msg != 1:
        error('Parameter set leads to degenerate geometry.')
        # save figure of error
        return

    with open(file_path, 'w') as fil:
        fil.write("   2.0000000e-03   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")
        fil.write("   1.25000000e-02   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")  # a point inside the structure
        fil.write("  -3.1415927e+00  -2.7182818e+00   0.0000000e+00   0.0000000e+00\n")  # a point outside the structure

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

        pts = lineTo(start_point, [-shift, Ri_el], step, plot)
        pt = [-shift, Ri_el]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # ADD BEAM PIPE LENGTH
        pts = lineTo(pt, [L_bp_l - shift, Ri_el], step, plot)
        pt = [L_bp_l - shift, Ri_el]
        for pp in pts:
            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el], plot)
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step, plot)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req],
                            plot)
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er], plot)
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er], plot)

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_el + L_m - x2 + 2 * L_bp_l - shift, y2], plot)
                    pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                    pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_el + L_m - shift, Ri_m], plot)
                    pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m)

            elif n > 1 and n != n_cell:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_m + L_m - x2 + 2 * L_bp_l - shift, y2], plot)
                pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_m + L_m - shift, Ri_m], plot)
                pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # calculate new shift
                shift = shift - 2 * L_m
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                            [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er], plot)
                pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er - shift, Ri_er], plot)
                pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

        # BEAM PIPE
        # reset shift
        shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2
        pts = lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step, plot)
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
        for pp in pts:
            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

        # END PATH
        pts = lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step,
                     plot)  # to add beam pipe to right
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

        # CLOSE PATH
        pts = lineTo(pt, start_point, step, plot)
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

    if plot:
        plt.tight_layout()
        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def writeCavityForMultipac_multicell(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None,
                                     beampipe='none',
                                     plot=True, unit=1e-3, scale=1):
    """
    Write cavity geometry

    Parameters
    ----------
    file_path: str
        File path to write geometry to
    n_cell: int
        Number of cavity cells
    mid_cell: list, ndarray
        Array of cavity middle cells' geometric parameters
    end_cell_left: list, ndarray
        Array of cavity left end cell's geometric parameters
    end_cell_right: list, ndarray
        Array of cavity left end cell's geometric parameters
    beampipe: str {"left", "right", "both", "none"}
        Specify if beam pipe is on one or both ends or at no end at all
    plot: bool
        If True, the cavity geometry is plotted for viewing

    Returns
    -------

    """
    if plot:
        plt.rcParams["figure.figsize"] = (12, 2)

    if end_cell_left is None:
        end_cell_left = mid_cell

    if end_cell_right is None:
        if end_cell_left is None:
            end_cell_right = mid_cell
        else:
            end_cell_right = end_cell_left

    us = unit * scale
    A_m_array, B_m_array, a_m_array, b_m_array, Ri_m_array, L_m_array, Req_array = np.array(mid_cell[:7]) * us
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el = np.array(end_cell_left[:7]) * us
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req_er = np.array(end_cell_right[:7]) * us
    # ic(np.array(mid_cell[:7]), np.array(mid_cell[:7]).shape)

    step = 0.005

    if beampipe.lower() == 'both':
        L_bp_l = 4 * L_el
        L_bp_r = 4 * L_el
    elif beampipe.lower() == 'none':
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #
    elif beampipe.lower() == 'left':
        L_bp_l = 4 * L_el
        L_bp_r = 0.000
    elif beampipe.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4 * L_el
    else:
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #

    # calculate shift
    if n_cell == 1:
        shift = (L_bp_r + L_bp_l + L_el + L_er) / 2
    else:
        shift = (L_bp_r + L_bp_l + L_el + np.sum(L_m_array) + L_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    # enforce welding by average equator radius between adjacent cells
    if n_cell > 1:
        # ic(Req_array)
        Req_el = (Req_el + Req_array[0][0]) / 2
    else:
        Req_el = (Req_el + Req_er) / 2

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req_el, L_bp_l)
    x1el, y1el, x2el, y2el = df[0]

    # # CALCULATE x1, y1, x2, y2
    # A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = A_m_array[0], B_m_array[0], a_m_array[0], b_m_array[0], Ri_m_array[0], L_m_array[0], Req_array[0]
    # df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req_el, L_bp_l)
    # x1, y1, x2, y2 = df[0]

    # # CALCULATE x1_er, y1_er, x2_er, y2_er
    # df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req_m, L_bp_r)
    # x1er, y1er, x2er, y2er = df[0]

    with open(file_path, 'w') as fil:
        fil.write("   2.0000000e-03   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")
        fil.write("   1.25000000e-02   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")  # a point inside the structure
        fil.write("  -3.1415927e+00  -2.7182818e+00   0.0000000e+00   0.0000000e+00\n")  # a point outside the structure

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

        pts = lineTo(start_point, [-shift, Ri_el], step, plot)
        pt = [-shift, Ri_el]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # ADD BEAM PIPE LENGTH
        pts = lineTo(pt, [L_bp_l - shift, Ri_el], step, plot)
        pt = [L_bp_l - shift, Ri_el]
        for pp in pts:
            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el], plot)
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step, plot)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req_el - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req_el],
                            plot)
                pt = [L_bp_l + L_el - shift, Req_el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper

                    # change parameters to right end cell parameters
                    # CALCULATE x1_er, y1_er, x2_er, y2_er
                    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req_el, L_bp_r)
                    x1er, y1er, x2er, y2er = df[0]

                    pts = arcTo(L_el + L_bp_l - shift, Req_el - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er], plot)
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, y1er], plot)

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper

                    # change midcell parameters
                    A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m = A_m_array[n - 1][0], B_m_array[n - 1][0], a_m_array[n - 1][
                        0], \
                        b_m_array[n - 1][0], Ri_m_array[n - 1][0], L_m_array[n - 1][0], Req_array[n - 1][0]

                    # enforce welding by average iris radius between adjacent cells
                    Ri_m = (Ri_m + Ri_m_array[n - 1][1]) / 2

                    # CALCULATE x1, y1, x2, y2
                    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req_el, L_bp_l)
                    x1, y1, x2, y2 = df[0]

                    pts = arcTo(L_el + L_bp_l - shift, Req_el - B_m, A_m, B_m, step, pt,
                                [L_el + L_m - x2 + 2 * L_bp_l - shift, y2], plot)
                    pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                    pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_el + L_m - shift, Ri_m], plot)
                    pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m_array[n - 1][0])

            elif n > 1 and n != n_cell:
                # ic(n)
                # change mid cell parameters
                A_m, B_m, a_m, b_m, _, L_m, Req_m = A_m_array[n - 2][1], B_m_array[n - 2][1], a_m_array[n - 2][1], \
                    b_m_array[n - 2][1], Ri_m_array[n - 2][1], L_m_array[n - 2][1], Req_array[n - 2][1]

                # enforce welding by average equator radius between adjacent cells
                Req_m = (Req_m + Req_array[n - 1][0]) / 2

                # CALCULATE x1, y1, x2, y2
                df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, L_bp_l)
                x1, y1, x2, y2 = df[0]

                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m], plot)
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # change midcell parameters
                A_m, B_m, a_m, b_m, Ri_m, L_m, _ = A_m_array[n - 1][0], B_m_array[n - 1][0], \
                    a_m_array[n - 1][0], \
                    b_m_array[n - 1][0], Ri_m_array[n - 1][0], \
                    L_m_array[n - 1][0], Req_array[n - 1][0]

                # enforce welding by average iris radius between adjacent cells
                Ri_m = (Ri_m + Ri_m_array[n - 1][1]) / 2

                # CALCULATE x1, y1, x2, y2
                df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, L_bp_l)
                x1, y1, x2, y2 = df[0]

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m_array[n - 2][1] + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt,
                            [L_m_array[n - 2][1] + L_m - x2 + 2 * L_bp_l - shift, y2], plot)
                pt = [L_m_array[n - 2][1] + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m_array[n - 2][1] + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                pt = [L_m_array[n - 2][1] + L_m - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m_array[n - 2][1] + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_m_array[n - 2][1] + L_m - shift, Ri_m], plot)
                pt = [L_bp_l + L_m_array[n - 2][1] + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # calculate new shift
                shift = shift - (L_m_array[n - 2][1] + L_m_array[n - 1][0])
            else:
                # change midcell parameters
                A_m, B_m, a_m, b_m, _, L_m, Req_m = A_m_array[n - 2][1], B_m_array[n - 2][1], a_m_array[n - 2][1], \
                    b_m_array[n - 2][1], Ri_m_array[n - 2][1], L_m_array[n - 2][1], Req_array[n - 2][1]

                # enforce welding by average equator radius between adjacent cells
                Req_m = (Req_m + Req_er) / 2

                # CALCULATE x1, y1, x2, y2
                df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req_m, L_bp_l)
                x1, y1, x2, y2 = df[0]

                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # change parameters to right end cell parameters
                # CALCULATE x1_er, y1_er, x2_er, y2_er
                df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req_m, L_bp_r)
                x1er, y1er, x2er, y2er = df[0]

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req_m], plot)
                pt = [L_bp_l + L_m - shift, Req_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req_m - B_er, A_er, B_er, step, pt,
                            [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er], plot)
                pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er - shift, Ri_er], plot)
                pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

        # BEAM PIPE
        # reset shift
        if n_cell == 1:
            shift = (L_bp_r + L_bp_l + L_el + L_er) / 2
            pts = lineTo(pt, [L_bp_r + L_bp_l + L_el + L_er - shift, Ri_er], step, plot)
            pt = [L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
            for pp in pts:
                fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

            # END PATH
            pts = lineTo(pt, [L_el + L_er + L_bp_l + L_bp_r - shift, 0], step,
                         plot)  # to add beam pipe to right
            pt = [L_el + L_er + L_bp_l + L_bp_r - shift, 0]
            # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
            # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
            # for pp in pts:
            #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

            # CLOSE PATH
            pts = lineTo(pt, start_point, step, plot)
            # for pp in pts:
            #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
        else:
            shift = (L_bp_r + L_bp_l + np.sum(L_m_array) + L_el + L_er) / 2
            pts = lineTo(pt, [L_bp_r + L_bp_l + np.sum(L_m_array) + L_el + L_er - shift, Ri_er], step, plot)
            pt = [np.sum(L_m_array) + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
            for pp in pts:
                fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

            # END PATH
            pts = lineTo(pt, [np.sum(L_m_array) + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step,
                         plot)  # to add beam pipe to right
            pt = [np.sum(L_m_array) + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
            # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
            # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
            # for pp in pts:
            #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

            # CLOSE PATH
            pts = lineTo(pt, start_point, step, plot)
            # for pp in pts:
            #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

    if plot:
        plt.tight_layout()
        plt.show()

        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


def plot_pillbox_geometry(n_cell, L, Req, Ri, S, L_bp, beampipe='none', plot=False, **kwargs):
    """

    Parameters
    ----------
    n_cell
    L
    Req
    Ri
    S
    L_bp
    beampipe
    plot
    kwargs

    Returns
    -------

    """
    L, Req, Ri, S, L_bp = np.array([L, Req, Ri, S, L_bp]) * 1e-3

    step = 0.001

    if beampipe.lower() == 'both':
        L_bp_l = L_bp
        L_bp_r = L_bp
    elif beampipe.lower() == 'none':
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #
    elif beampipe.lower() == 'left':
        L_bp_l = L_bp
        L_bp_r = 0.000
    elif beampipe.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = L_bp
    else:
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #

    geo = []
    shift = (L_bp_l + L_bp_r + n_cell * L + (n_cell - 1) * S) / 2

    # SHIFT POINT TO START POINT
    start_point = [-shift, 0]
    geo.append([start_point[1], start_point[0]])

    lineTo(start_point, [-shift, Ri], step)
    pt = [-shift, Ri]
    geo.append([pt[1], pt[0]])

    # add beampipe
    if L_bp_l > 0:
        lineTo(pt, [-shift + L_bp_l, Ri], step)
        pt = [-shift + L_bp_l, Ri]
        geo.append([pt[1], pt[0]])

    for n in range(1, n_cell + 1):
        if n == 1:
            lineTo(pt, [-shift + L_bp_l, Req], step)
            pt = [-shift + L_bp_l, Req]
            geo.append([pt[1], pt[0]])

            lineTo(pt, [-shift + L_bp_l + L, Req], step)
            pt = [-shift + L_bp_l + L, Req]
            geo.append([pt[1], pt[0]])

            lineTo(pt, [-shift + L_bp_l + L, Ri], step)
            pt = [-shift + L_bp_l + L, Ri]
            geo.append([pt[1], pt[0]])

            shift -= L
        elif n > 1:
            lineTo(pt, [-shift + L_bp_l + S, Ri], step)
            pt = [-shift + L_bp_l + S, Ri]
            geo.append([pt[1], pt[0]])

            lineTo(pt, [-shift + L_bp_l + S, Req], step)
            pt = [-shift + L_bp_l + S, Req]
            geo.append([pt[1], pt[0]])

            lineTo(pt, [-shift + L_bp_l + S + L, Req], step)
            pt = [-shift + L_bp_l + S + L, Req]
            geo.append([pt[1], pt[0]])

            lineTo(pt, [-shift + L_bp_l + S + L, Ri], step)
            pt = [-shift + L_bp_l + S + L, Ri]
            geo.append([pt[1], pt[0]])

            shift -= L

    if L_bp_r > 0:
        lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri], step)
        pt = [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri]
        geo.append([pt[1], pt[0]])

    # END PATH
    lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0], step)
    pt = [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0]
    geo.append([pt[1], pt[0]])

    # if bc:
    #     # draw right boundary condition
    #     ax.plot([shift, shift], [-Ri_er, Ri_er],
    #             [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
    #             [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0]])
    geo = np.array(geo)

    top = plt.gca().plot(geo[:, 1], geo[:, 0], **kwargs)
    bottom = plt.gca().plot(geo[:, 1], -geo[:, 0], c=top[0].get_color(), **kwargs)

    # plot legend without duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    return plt.gca()


def write_pillbox_geometry(file_path, n_cell, cell_par, beampipe='none', plot=False, **kwargs):
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
    L, Req, Ri, S, L_bp = np.array(cell_par) * 1e-3

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

    with open(file_path, 'w') as fil:
        shift = (L_bp_l + L_bp_r + n_cell * L + (n_cell - 1) * S) / 2

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        lineTo(start_point, [-shift, Ri], step)
        pt = [-shift, Ri]
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # add beampipe
        if L_bp_l > 0:
            lineTo(pt, [-shift + L_bp_l, Ri], step)
            pt = [-shift + L_bp_l, Ri]
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                lineTo(pt, [-shift + L_bp_l, Req], step)
                pt = [-shift + L_bp_l, Req]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                lineTo(pt, [-shift + L_bp_l + L, Req], step)
                pt = [-shift + L_bp_l + L, Req]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                lineTo(pt, [-shift + L_bp_l + L, Ri], step)
                pt = [-shift + L_bp_l + L, Ri]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                shift -= L
            elif n > 1:
                lineTo(pt, [-shift + L_bp_l + S, Ri], step)
                pt = [-shift + L_bp_l + S, Ri]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                lineTo(pt, [-shift + L_bp_l + S, Req], step)
                pt = [-shift + L_bp_l + S, Req]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                lineTo(pt, [-shift + L_bp_l + S + L, Req], step)
                pt = [-shift + L_bp_l + S + L, Req]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                lineTo(pt, [-shift + L_bp_l + S + L, Ri], step)
                pt = [-shift + L_bp_l + S + L, Ri]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                shift -= L

        if L_bp_r > 0:
            lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri], step)
            pt = [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, Ri]
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # END PATH
        lineTo(pt, [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0], step)
        pt = [-shift + L_bp_l + (n_cell - 1) * S + L_bp_r, 0]
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # if bc:
        #     # draw right boundary condition
        #     ax.plot([shift, shift], [-Ri_er, Ri_er],
        #             [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
        #             [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

        # CLOSE PATH
        # lineTo(pt, start_point, step)
        # geo.append([start_point[1], start_point[0]])

    # top = plt.gca().plot(geo[:, 1], geo[:, 0], **kwargs)
    # bottom = plt.gca().plot(geo[:, 1], -geo[:, 0], c=top[0].get_color(), **kwargs)
    #
    # # plot legend without duplicates
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())



def enforce_Req_continuity(par_mid, par_end_l, par_end_r, cell_type=None):
    """
    Enforce continuity at iris and equator of cavities

    Parameters
    ----------
    par_mid
    par_end_l
    par_end_r
    cell_type

    Returns
    -------

    """

    if cell_type:
        if cell_type.lower() == 'mid cell' or cell_type.lower() == 'mid-cell' or cell_type.lower() == 'mid_cell':
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
        elif cell_type.lower() == 'mid-end cell' or cell_type.lower() == 'mid-end-cell' or cell_type.lower() == 'mid_end_cell':
            par_end_l[6] = par_mid[6]
            par_end_r[6] = par_mid[6]
        elif (cell_type.lower() == 'end-end cell' or cell_type.lower() == 'end-end-cell'
              or cell_type.lower() == 'end_end_cell') or cell_type.lower() == 'end end cell':
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
        else:
            par_mid[6] = par_end_r[6]
            par_end_l[6] = par_end_r[6]
    else:
        Req_avg = (par_mid[6] + par_end_l[6] + par_end_r[6]) / 3
        par_mid[6] = Req_avg
        par_end_l[6] = Req_avg
        par_end_r[6] = Req_avg


def save_tune_result(d, folder, filename):
    with open(os.path.join(folder, 'eigenmode', filename), 'w') as file:
        file.write(json.dumps(d, indent=4, separators=(',', ': ')))



def to_multicell(n_cells, shape):
    shape_multicell = {}
    mid_cell = shape['IC']
    mid_cell_multi = np.array([[[a, a] for _ in range(n_cells - 1)] for a in mid_cell])

    shape_multicell['OC'] = shape['OC']
    shape_multicell['OC_R'] = shape['OC_R']
    shape_multicell['IC'] = mid_cell_multi
    # shape_multicell['BP'] = shape['BP']
    shape_multicell['n_cells'] = shape['n_cells']
    shape_multicell['CELL TYPE'] = 'multicell'

    return shape_multicell


from scipy.interpolate import interp1d



def expand_cells(cav: dict, cells):
    """
    Given shape['n_cells'], expand 'all' to every half-cell:
      ['cell1_l','cell1_r',...,'cellN_l','cellN_r']
    Or accept a single string or list.
    """
    N = cav.n_cells
    if isinstance(cells, str) and cells == 'all':
        return [f'cell{i}_{side}' for i in range(1, N + 1) for side in ('l', 'r')]
    if isinstance(cells, str):
        return [cells]
    return list(cells)

def apply_perturbation(base,
                       deltas: list,
                       perturbed_vars: list,
                       mode: str):
    """
    mode='add': x_new = x + δ
    mode='mul': x_new = x * (1 + δ)
    """
    P     = len(VAR_NAMES)
    N     = base.n_cells
    out   = {}
    for ii, delta in enumerate(deltas):
        cav    = copy.deepcopy(base)
        # one slot for the very left half-cell

        # choose apply function
        for pvar, d in zip(perturbed_vars, delta):
            if mode == 'add':
                cav.parameters[pvar] += d
            else:
                cav.parameters[pvar] *= (1 + d)

        # rename
        new_name = f'{cav.name}_Q{ii}'
        cav.name = new_name
        cav.projectDir = cav.uq_dir
        out[new_name] = cav
    return out


def generate_perturbed_shapes(shape: dict,
                              cells,
                              variables: list,
                              mode: list,
                              node_type: list):
    """
    High-level API: returns (shapes, weights).

    - cells: 'all' or list of 'cellX_l'/'cellX_r'
    - variables: subset of VAR_NAMES
    - bound: absolute delta bound
    - mode: 'add' or 'mul'
    - n: nodes per dimension (ignored by stroud3)
    - node_type: 'uniform','gauss_legendre','stroud3'
    """
    cell_list = expand_cells(shape, cells)
    cell_vars = [(c, v) for c in cell_list for v in variables]
    k = len(cell_vars)

    deltas, weights = generate_nodes(k, mode[1], node_type)

    shapes = apply_perturbation(shape, deltas, cell_vars, mode[0])

    return shapes, np.atleast_2d(weights).T



def perturb_geometry(cav, eigenmode_config):

    uq_config = eigenmode_config['uq_config']
    uq_vars = uq_config['variables']
    # which_cell = uq_config['cell']

    method = uq_config['method']
    if 'perturbation_mode' not in uq_config:
        uq_config['perturbation_mode'] = ['add']  # default: additive perturbation with bound 0.01

    perturbation_mode = uq_config['perturbation_mode']
    if not isinstance(perturbation_mode[1], list):
        perturbation_mode[1] = [perturbation_mode[1]] * len(uq_vars)

    # cells = which_cell
    variables = uq_vars
    mode = perturbation_mode
    node_type = method

    # get perturbed variables
    uq_parameters = uq_config['variables']
    if isinstance(uq_parameters, str):
        uq_parameters = [uq_parameters]

    if uq_config['cell'] == 'all':
        perturbed_vars = [par for par in cav.parameters if any(k in par for k in uq_parameters)]
    else:
        perturbed_vars = uq_parameters

    k = len(perturbed_vars)

    deltas, weights = generate_nodes(k, mode[1], node_type)
    perturbed_cavs_dict = apply_perturbation(cav, deltas, perturbed_vars, mode[0])

    return perturbed_cavs_dict, np.atleast_2d(weights).T


def enforce_continuity_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce:
      Req1==Req2, Req3==Req4, Req5==Req6, ...
      Ri2==Ri3, Ri4==Ri5, Ri6==Ri7, ...
    in a DataFrame with columns 'Req1'...'ReqN' and 'Ri1'...'RiN'.
    """
    df2 = df.copy()
    pat = re.compile(r'^(Req|Ri)(\d+|_[a-zA-Z0-9]+)$')

    # collect column names by var and index
    req = {}  # idx -> col
    ri  = {}
    for col in df2.columns:
        m = pat.match(col)
        print(m)
        if not m: continue
        var, idx = m.group(1), int(m.group(2))
        (req if 'Req' in var else ri)[idx] = col

    max_idx = max(req.keys() | ri.keys())

    # Equator: Req at odd indices paired with next
    for i in range(1, max_idx+1, 2):
        c1 = req.get(i)
        c2 = req.get(i+1)
        if c1 and c2:
            avg = 0.5*(df2[c1] + df2[c2])
            df2[c1] = avg
            df2[c2] = avg

    # Iris: Ri at even indices paired with next
    for i in range(2, max_idx+1, 2):
        c1 = ri.get(i)
        c2 = ri.get(i+1)
        if c1 and c2:
            avg = 0.5*(df2[c1] + df2[c2])
            df2[c1] = avg
            df2[c2] = avg

    return df2


def shapes_to_dataframe(cavs_dict):
    """
    Convert a list of perturbed-shape dicts into a DataFrame.

    Columns are named A1, B1, a1, ..., A2, B2, a2, ... etc.,
    where each half-cell (left then right) across all cells
    is assigned an increasing index.
    """
    if not cavs_dict:
        return pd.DataFrame()

    # Build a list of rows from each object's `.parameter` dict
    data = []
    for name, cav in cavs_dict.items():
        row = cav.parameters.copy()  # extract the parameter dictionary
        row['name'] = name  # optionally include the cavity name as a column
        data.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Optionally set the name as index
    df.set_index('name', inplace=True)

    return df




def make_dirs_from_dict(d, current_dir):
    for key, val in d.items():
        if not os.path.exists(os.path.join(current_dir, key)):
            os.mkdir(os.path.join(current_dir, key))
            if type(val) == dict:
                make_dirs_from_dict(val, os.path.join(current_dir, key))
        elif val:
            make_dirs_from_dict(val, os.path.join(current_dir, key))
