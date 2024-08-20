import json
import math
import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import fsolve
import numpy as np
from cavsim2d.utils.printing import *


def update_alpha(cell, cell_type='normal'):
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
    if cell_type == 'normal':
        cell = [A, B, a, b, Ri, L, Req, alpha[0]]
    else:
        cell = [A, B, a, b, Ri, L, Req, cell[7], alpha[0]]

    return cell


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
    checks = {"non-reentrant": [0.5, -0.5],
              "reentrant": [0.85, -0.85],
              "expansion": [0.15, -0.01]}

    max_restart = 4
    checks = {"non-reentrant": [[0.5, -0.5], [0.75, -0.25], [0.25, -0.75], [0.9, -0.1]],
              "reentrant": [[1.1, -1.1], [1.9, -1.9], [1.5, -1.5], [1.75, -1.75], ]
              }
    msg = 4
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
    alpha = 180 - np.arctan2(y2 - y1, (x2 - x1)) * 180 / np.pi

    if tangent_check:
        shift_x = -L-lft
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


def write_cst_paramters(key, ic_, oc_l, oc_r, projectDir, cell_type, opt=False, solver='slans'):
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
        path = fr'{projectDir}/SimulationData/{folder}/{key}/{key}.txt'

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
        path = fr'{projectDir}/SimulationData/{folder}/{key}/{key}.txt'
        path_mc = fr'{projectDir}/SimulationData/{folder}/{key}/{key}_Multicell.txt'

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


def stroud(p):
    """
    Stroud-3 method

    Parameters
    ----------
    p: int
        Dimension

    Returns
    -------
    Nodes of quadrature rule in [0,1]**p (column-wise)
    """
    # Stroud-3 method
    #
    # Input parameters:
    #  p   number of dimensions
    # Output parameters:
    #  nodes   nodes of quadrature rule in [0,1]**p (column-wise)
    #

    nodes = np.zeros((p, 2 * p))
    coeff = np.pi / p
    fac = np.sqrt(2 / 3)

    for i in range(2 * p):
        for r in range(int(np.floor(0.5 * p))):
            k = 2 * r
            nodes[k, i] = fac * np.cos((k + 1) * (i + 1) * coeff)
            nodes[k + 1, i] = fac * np.sin((k + 1) * (i + 1) * coeff)

        if 0.5 * p != np.floor(0.5 * p):
            nodes[-1, i] = ((-1) ** (i + 1)) / np.sqrt(3)

    # transform nodes from [-1,+1]**p to [0,1]**p
    nodes = 0.5 * nodes + 0.5

    return nodes


def quad_stroud3(rdim, degree):
    """
    Stroud-3 quadrature in :math:`[0,1]^k`

    Parameters
    ----------
    rdim: int
        Dimension of variables
    degree: int
        Degree

    Returns
    -------
    Nodes and corresponding weights
    """
    # data for Stroud-3 quadrature in [0,1]**k
    # nodes and weights
    nodes = stroud(rdim)
    nodestr = 2. * nodes - 1.
    weights = (1 / (2 * rdim)) * np.ones((2 * rdim, 1))

    # evaluation of Legendre polynomials
    bpoly = np.zeros((degree + 1, rdim, 2 * rdim))
    for ll in range(rdim):
        for j in range(2 * rdim):
            bpoly[0, ll, j] = 1
            bpoly[1, ll, j] = nodestr[ll, j]
            for i in range(1, degree):
                bpoly[i + 1, ll, j] = ((2 * (i + 1) - 1) * nodestr[ll, j] * bpoly[i, ll, j] - i * bpoly[
                    i - 1, ll, j]) / (i + 1)

    # standardisation of Legendre polynomials
    for i in range(1, degree + 1):
        bpoly[i, :, :] = bpoly[i, :, :] * np.sqrt(2 * (i + 1) - 1)

    return nodes, weights, bpoly


def c1_leg_monomial_integral(expon):
    if expon < 0:
        error("\n")
        error("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")
        error("EXPON < 0.")
        raise ValueError("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")

    if expon % 2 == 1:
        return 0.0

    value = 2.0 / (expon + 1)
    return value


def cn_leg_03_xiu(n):
    o = 2 * n

    x = np.zeros((n, o))
    w = np.zeros(o)

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    for j in range(1, o + 1):

        i = 0
        for r in range(1, math.floor(n / 2) + 1):
            arg = (2 * r - 1) * j * np.pi / n
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.cos(arg) / np.sqrt(3.0)
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.sin(arg) / np.sqrt(3.0)

        if i < n:
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * (-1) ** j / np.sqrt(3.0)
            if n == 1:
                x[i - 1, j - 1] = x[i - 1, j - 1] / np.sqrt(2.0)

    w[0:o] = volume / o

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_leg_03_1(n):
    o = 2 * n

    w = np.zeros(o)
    x = np.zeros((n, o))

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    for j in range(1, o + 1):

        i = 0

        for r in range(1, math.floor(n / 2) + 1):
            arg = (2 * r - 1) * j * np.pi / n
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.cos(arg) / np.sqrt(3.0)
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.sin(arg) / np.sqrt(3.0)

        if i < n:
            i += 1
            if n == 1:
                x[i - 1, j - 1] = r8_mop(j) / np.sqrt(3.0)
            else:
                x[i - 1, j - 1] = np.sqrt(2.0) * r8_mop(j) / np.sqrt(3.0)

    w[0:o] = volume / o

    return x, np.atleast_2d(w).T / np.sum(w)


def r8_mop(i):
    if i % 2 == 0:
        value = 1.0
    else:
        value = -1.0

    return value


def cn_leg_05_1(n, option=1):
    # Check if the value of n is 4, 5, or 6
    if n not in [4, 5, 6]:
        error("\n")
        error("CN_LEG_05_1 - Fatal error!")
        error("The value of N must be 4, 5, or 6.")
        raise ValueError("CN_LEG_05_1 - Fatal error!")

    # Check for valid option when n = 4 or 5
    if n in [4, 5] and option not in [1, 2]:
        error("\n")
        error("CN_LEG_05_1 - Fatal error!")
        error("When N = 4 or 5, OPTION must be 1 or 2.")
        raise ValueError("CN_LEG_05_1 - Fatal error!")

    o = n ** 2 + n + 2
    w = np.zeros(o)
    x = np.zeros((n, o))

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    if (n == 4 and option == 1):
        eta = 0.778984505799815
        lmbda = 1.284565137874656
        xsi = -0.713647298819253
        mu = -0.715669761974162
        gamma = 0.217089151000943
        a = 0.206186096875899e-01 * volume
        b = 0.975705820221664e-02 * volume
        c = 0.733921929172573e-01 * volume
    elif (n == 4 and option == 2):
        eta = 0.546190755827425E+00
        lmbda = 0.745069130115661E+00
        xsi = - 0.413927294508700E+00
        mu = - 0.343989637454535E+00
        gamma = 1.134017894600344E+00
        a = 0.853094758323323E-01 * volume
        b = 0.862099000096395E-01 * volume
        c = 0.116418206881849E-01 * volume
    elif (n == 5 and option == 1):
        eta = 0.522478547481276E+00
        lmbda = 0.936135175985774E+00
        xsi = - 0.246351362101519E+00
        mu = - 0.496308106093758E+00
        gamma = 0.827180176822930E+00
        a = 0.631976901960153E-01 * volume
        b = 0.511464127430166E-01 * volume
        c = 0.181070246088902E-01 * volume
    elif (n == 5 and option == 2):
        eta = 0.798317301388741E+00
        lmbda = 0.637344273885728E+00
        xsi = - 0.455245909918377E+00
        mu = - 1.063446229997311E+00
        gamma = 0.354482076665770E+00
        a = 0.116952384292206E-01 * volume
        b = 0.701731258612708E-01 * volume
        c = 0.137439132264426E-01 * volume
    else:
        eta = 0.660225291773525E+00
        lmbda = 1.064581294844754E+00
        xsi = 0.000000000000000E+00
        mu = - 0.660225291773525E+00
        gamma = 0.660225291773525E+00
        a = 0.182742214532872E-01 * volume
        b = 0.346020761245675E-01 * volume
        c = 0.182742214532872E-01 * volume

    # Set x and w based on parameters
    k = 0
    # k += 1
    for i in range(n):
        x[i, k] = eta
    w[k] = a

    # k += 1
    for i in range(n):
        x[i, k] = -eta
    w[k] = a

    for i1 in range(n):
        for i in range(1, n):
            x[i, k] = xsi
        x[i1, k] = lmbda
        w[k] = b
        k = k + 1

    for i1 in range(n):
        for i in range(n):
            x[i, k] = - xsi
        x[i1, k] = - lmbda
        w[k] = b
        k = k + 1

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for i in range(n):
                x[i, k] = gamma
            x[i1, k] = mu
            x[i2, k] = mu
            w[k] = c
            k = k + 1

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for i in range(n):
                x[i, k] = - gamma
            x[i1, k] = - mu
            x[i2, k] = - mu
            w[k] = c
            k = k + 1

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_leg_05_2(n):
    if n < 2:
        print("\n")
        print("CN_LEG_05_2 - Fatal error!")
        print("N must be at least 2.")
        raise ValueError("CN_LEG_05_2 - Fatal error!")

    o = 2 * n ** 2 + 1
    w = np.zeros(o, dtype=np.float64)
    x = np.zeros((n, o), dtype=np.float64)

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    b0 = (25 * n * n - 115 * n + 162) * volume / 162.0
    b1 = (70 - 25 * n) * volume / 162.0
    b2 = 25.0 * volume / 324.0

    r = np.sqrt(3.0 / 5.0)

    k = 0

    k += 1
    for i in range(n):
        x[i, k - 1] = 0.0
    w[k - 1] = b0

    for i1 in range(1, n + 1):
        k += 1
        for i in range(n):
            x[i, k - 1] = 0.0
        x[i1 - 1, k - 1] = +r
        w[k - 1] = b1

        k += 1
        for i in range(n):
            x[i, k - 1] = 0.0
        x[i1 - 1, k - 1] = -r
        w[k - 1] = b1

    for i1 in range(1, n):
        for i2 in range(i1 + 1, n + 1):
            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = +r
            x[i2 - 1, k - 1] = +r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = +r
            x[i2 - 1, k - 1] = -r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = -r
            x[i2 - 1, k - 1] = +r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = -r
            x[i2 - 1, k - 1] = -r
            w[k - 1] = b2

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_gauss(rdim, degree):
    x, w = np.polynomial.legendre.leggauss(degree)

    X = [x for _ in range(rdim)]

    nodes = np.array(np.meshgrid(*X, indexing='ij')).reshape(rdim, -1)
    weights = np.ones(degree ** rdim) / (degree ** rdim)

    return nodes, np.atleast_2d(weights).T / np.sum(weights)


def weighted_mean_obj(tab_var, weights):
    rows_sims_no, cols = np.shape(tab_var)
    no_weights, dummy = np.shape(weights)

    if rows_sims_no == no_weights:
        expe = np.zeros((cols, 1))
        outvar = np.zeros((cols, 1))
        for i in range(cols):
            expe[i, 0] = np.dot(tab_var[:, i], weights)[0]
            outvar[i, 0] = np.dot(tab_var[:, i] ** 2, weights)[0]

        stdDev = np.sqrt(abs(outvar - expe ** 2))
    else:
        expe = 0
        stdDev = 0
        error('Cols_sims_no != No_weights')

    return list(expe.T[0]), list(stdDev.T[0])


def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def linspace(start, stop, step=1.):
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

    C = np.pi * (a + b)  # <- approximate perimeter of ellipse
    x1, y1 = start
    x2, y2 = end
    # calculate parameter start and end points
    t1 = np.mod(np.arctan2((y1 - k) / b, (x1 - h) / a), 2 * np.pi)
    t2 = np.mod(np.arctan2((y2 - k) / b, (x2 - h) / a), 2 * np.pi)

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


def plot_cavity_geometry(IC, OC, OC_R, BP, n_cell, bc, scale=1, plot=None):
    """
    Plot cavity geometry

    Parameters
    ----------
    plot: Plot object
        Matplotlib plot object from plotter
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
    bc: list, ndarray
        List specifying the left and right boundary conditions' color
    scale: float
        Scale of the cavity geometry

    Returns
    -------

    """
    if plot:
        fig = plot.fig
        ax = plot.ax
    else:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_aspect('equal')

    # 21578127116
    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(IC)[:7] * scale * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(OC)[:7] * scale * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(OC_R)[:7] * scale * 1e-3

    if BP.lower() == 'both':
        L_bp_l = 4 * L_m
        L_bp_r = 4 * L_m

    elif BP.lower() == 'left':
        L_bp_l = 4 * L_m
        L_bp_r = 0.000

    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4 * L_m
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    step = 0.0005

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2
    # shift = 0
    # shift = L_m  # for end cell

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el
    data = ([0 + L_bp_l, Ri_el + b_el, L_el + L_bp_l, Req - B_el],
            [a_el, b_el, A_el, B_el])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])

    x1el, y1el, x2el, y2el = fsolve(ellipse_tangent, np.array(
        [a_el + L_bp_l, Ri_el + 0.85 * b_el, L_el - A_el + L_bp_l, Req - 0.85 * B_el]),
                                    args=data,
                                    xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess

    # CALCULATE x1, y1, x2, y2
    data = ([0 + L_bp_l, Ri_m + b_m, L_m + L_bp_l, Req - B_m],
            [a_m, b_m, A_m, B_m])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1, y1, x2, y2 = fsolve(ellipse_tangent,
                            np.array([a_m + L_bp_l, Ri_m + 0.85 * b_m, L_m - A_m + L_bp_l, Req - 0.85 * B_m]),
                            args=data, xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    data = ([0 + L_bp_r, Ri_er + b_er, L_er + L_bp_r, Req - B_er],
            [a_er, b_er, A_er, B_er])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
    x1er, y1er, x2er, y2er = fsolve(ellipse_tangent, np.array(
        [a_er + L_bp_r, Ri_er + 0.85 * b_er, L_er - A_er + L_bp_r, Req - 0.85 * B_er]),
                                    args=data,
                                    xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess
    geo = []

    # SHIFT POINT TO START POINT
    start_point = [-shift, 0]
    geo.append([start_point[1], start_point[0]])

    lineTo(start_point, [-shift, Ri_el], step)
    pt = [-shift, Ri_el]
    geo.append([pt[1], pt[0]])

    # draw left boundary condition
    ax.plot([-shift, -shift], [-Ri_el, Ri_el],
            [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
            [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c=bc[0], lw=4, zorder=100)

    # ADD BEAM PIPE LENGTH
    if L_bp_l != 0:
        lineTo(pt, [L_bp_l - shift, Ri_el], step)
        pt = [L_bp_l - shift, Ri_el]

        geo.append([pt[1], pt[0]])

    for n in range(1, n_cell + 1):
        if n == 1:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
            pt = [-shift + x1el, y1el]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2el, y2el], step)
            pt = [-shift + x2el, y2el]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
            pt = [L_bp_l + L_el - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            if n_cell == 1:
                if L_bp_r > 0:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                                [L_el + L_er - x2er + + L_bp_l + L_bp_r - shift, Req])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, y1er])

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])

                    geo.append([pt[1], pt[0]])

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, Req])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, y1er])
                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

            else:
                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                            [L_el + L_m - x2 + 2 * L_bp_l - shift, Req])
                pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                geo.append([pt[1], pt[0]])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_el + L_m - shift, y1])
                pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # calculate new shift
                shift = shift - (L_el + L_m)
                # ic(shift)

        elif n > 1 and n != n_cell:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])

            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                        [L_m + L_m - x2 + 2 * L_bp_l - shift, Req])
            pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])

            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
            pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                        [L_bp_l + L_m + L_m - shift, y1])
            pt = [L_bp_l + L_m + L_m - shift, Ri_m]

            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # calculate new shift
            shift = shift - 2 * L_m
        else:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                        [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, Req])
            pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
            pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                        [L_bp_l + L_m + L_er - shift, y1er])
            pt = [L_bp_l + L_m + L_er - shift, Ri_er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

    # BEAM PIPE
    # reset shift
    shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

    if L_bp_r > 0:  # if there's a problem, check here.
        lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
        geo.append([pt[1], pt[0]])

    # END PATH
    lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step)  # to add beam pipe to right
    pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
    # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
    # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
    geo.append([pt[1], pt[0]])

    # draw rightt boundary condition
    ax.plot([shift, shift], [-Ri_er, Ri_er],
            [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
            [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c=bc[1], lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0]])
    geo = np.array(geo)

    top = ax.plot(geo[:, 1], geo[:, 0], lw=4)
    bottom = ax.plot(geo[:, 1], -geo[:, 0], lw=4, c=top[0].get_color())
    fig.canvas.draw()


#
# def plot_cavity_geometry_cli(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False):
#     """
#     Plot cavity geometry for display in w_GeometryView
#     Parameters
#     ----------
#     plot: Plot object
#         Matplotlib plot object from plotter
#     IC: list, ndarray
#         Inner Cell geometric parameters list
#     OC: list, ndarray
#         Left outer Cell geometric parameters list
#     OC_R: list, ndarray
#         Right outer Cell geometric parameters list
#     BP: str {"left", "right", "both", "none"}
#         Specify if beam pipe is on one or both ends or at no end at all
#     n_cell: int
#         Number of cavity cells
#     scale: float
#         Scale of the cavity geometry
#
#     Returns
#     -------
#
#     """
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(12,4))
#         ax.set_aspect('equal')
#
#     # 21578127116
#     A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(IC)[:7] * scale * 1e-3
#     A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(OC)[:7] * scale * 1e-3
#     A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(OC_R)[:7] * scale * 1e-3
#
#     if BP.lower() == 'both':
#         L_bp_l = 4 * L_m
#         L_bp_r = 4 * L_m
#
#     elif BP.lower() == 'left':
#         L_bp_l = 4 * L_m
#         L_bp_r = 0.000
#
#     elif BP.lower() == 'right':
#         L_bp_l = 0.000
#         L_bp_r = 4 * L_m
#     else:
#         L_bp_l = 0.000
#         L_bp_r = 0.000
#
#     step = 0.0005
#
#     # calculate shift
#     shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2
#
#     # shift = 0
#     # shift = L_m  # for end cell
#
#     # calculate angles outside loop
#     # CALCULATE x1_el, y1_el, x2_el, y2_el
#     data = ([0 + L_bp_l, Ri_el + b_el, L_el + L_bp_l, Req - B_el],
#             [a_el, b_el, A_el, B_el])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
#
#     x1el, y1el, x2el, y2el = fsolve(ellipse_tangent, np.array(
#         [a_el + L_bp_l, Ri_el + 0.85 * b_el, L_el - A_el + L_bp_l, Req - 0.85 * B_el]),
#                                     args=data,
#                                     xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess
#
#     # CALCULATE x1, y1, x2, y2
#     data = ([0 + L_bp_l, Ri_m + b_m, L_m + L_bp_l, Req - B_m],
#             [a_m, b_m, A_m, B_m])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
#     x1, y1, x2, y2 = fsolve(ellipse_tangent,
#                             np.array([a_m + L_bp_l, Ri_m + 0.85 * b_m, L_m - A_m + L_bp_l, Req - 0.85 * B_m]),
#                             args=data, xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess
#
#     # CALCULATE x1_er, y1_er, x2_er, y2_er
#     data = ([0 + L_bp_r, Ri_er + b_er, L_er + L_bp_r, Req - B_er],
#             [a_er, b_er, A_er, B_er])  # data = ([h, k, p, q], [a_m, b_m, A_m, B_m])
#     x1er, y1er, x2er, y2er = fsolve(ellipse_tangent, np.array(
#         [a_er + L_bp_r, Ri_er + 0.85 * b_er, L_er - A_er + L_bp_r, Req - 0.85 * B_er]),
#                                     args=data,
#                                     xtol=1.49012e-12)  # [a_m, b_m-0.3*b_m, L_m-A_m, Req-0.7*B_m] initial guess
#     geo = []
#
#     # SHIFT POINT TO START POINT
#     start_point = [-shift, 0]
#     geo.append([start_point[1], start_point[0]])
#
#     lineTo(start_point, [-shift, Ri_el], step)
#     pt = [-shift, Ri_el]
#     geo.append([pt[1], pt[0]])
#
#     if bc:
#         # draw left boundary condition
#         ax.plot([-shift, -shift], [-Ri_el, Ri_el],
#                 [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
#                 [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)
#
#     # ADD BEAM PIPE LENGTH
#     if L_bp_l != 0:
#         lineTo(pt, [L_bp_l - shift, Ri_el], step)
#         pt = [L_bp_l - shift, Ri_el]
#
#         geo.append([pt[1], pt[0]])
#
#     for n in range(1, n_cell + 1):
#         if n == 1:
#             # DRAW ARC:
#             pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
#             pt = [-shift + x1el, y1el]
#             for pp in pts:
#                 geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # DRAW LINE CONNECTING ARCS
#             lineTo(pt, [-shift + x2el, y2el], step)
#             pt = [-shift + x2el, y2el]
#             geo.append([pt[1], pt[0]])
#
#             # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
#             pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
#             pt = [L_bp_l + L_el - shift, Req]
#             for pp in pts:
#                 geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             if n_cell == 1:
#                 if L_bp_r > 0:
#                     # EQUATOR ARC TO NEXT POINT
#                     # half of bounding box is required,
#                     # start is the lower coordinate of the bounding box and end is the upper
#                     pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
#                                 [L_el + L_er - x2er + + L_bp_l + L_bp_r - shift, Req])
#                     pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
#                     for pp in pts:
#                         if (np.around(pp, 12) != np.around(pt, 12)).all():
#                             geo.append([pp[1], pp[0]])
#                     geo.append([pt[1], pt[0]])
#
#                     # STRAIGHT LINE TO NEXT POINT
#                     lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
#                     pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
#                     geo.append([pt[1], pt[0]])
#
#                     # ARC
#                     # half of bounding box is required,
#                     # start is the lower coordinate of the bounding box and end is the upper
#                     pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
#                                 [L_bp_l + L_el + L_er - shift, y1er])
#
#                     pt = [L_bp_l + L_el + L_er - shift, Ri_er]
#                     for pp in pts:
#                         if (np.around(pp, 12) != np.around(pt, 12)).all():
#                             geo.append([pp[1], pp[0]])
#
#                     geo.append([pt[1], pt[0]])
#
#                     # calculate new shift
#                     shift = shift - (L_el + L_er)
#                 else:
#                     # EQUATOR ARC TO NEXT POINT
#                     # half of bounding box is required,
#                     # start is the lower coordinate of the bounding box and end is the upper
#                     pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
#                                 [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, Req])
#                     pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
#                     for pp in pts:
#                         if (np.around(pp, 12) != np.around(pt, 12)).all():
#                             geo.append([pp[1], pp[0]])
#                     geo.append([pt[1], pt[0]])
#
#                     # STRAIGHT LINE TO NEXT POINT
#                     lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
#                     pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
#                     geo.append([pt[1], pt[0]])
#
#                     # ARC
#                     # half of bounding box is required,
#                     # start is the lower coordinate of the bounding box and end is the upper
#                     pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
#                                 [L_bp_l + L_el + L_er - shift, y1er])
#                     pt = [L_bp_l + L_el + L_er - shift, Ri_er]
#                     for pp in pts:
#                         if (np.around(pp, 12) != np.around(pt, 12)).all():
#                             geo.append([pp[1], pp[0]])
#                     geo.append([pt[1], pt[0]])
#
#             else:
#                 # EQUATOR ARC TO NEXT POINT
#                 # half of bounding box is required,
#                 # start is the lower coordinate of the bounding box and end is the upper
#                 pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
#                             [L_el + L_m - x2 + 2 * L_bp_l - shift, Req])
#                 pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
#                 for pp in pts:
#                     if (np.around(pp, 12) != np.around(pt, 12)).all():
#                         geo.append([pp[1], pp[0]])
#                 geo.append([pt[1], pt[0]])
#
#                 # STRAIGHT LINE TO NEXT POINT
#                 lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
#                 pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
#                 geo.append([pt[1], pt[0]])
#
#                 # ARC
#                 # half of bounding box is required,
#                 # start is the lower coordinate of the bounding box and end is the upper
#                 pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
#                             [L_bp_l + L_el + L_m - shift, y1])
#                 pt = [L_bp_l + L_el + L_m - shift, Ri_m]
#                 for pp in pts:
#                     if (np.around(pp, 12) != np.around(pt, 12)).all():
#                         geo.append([pp[1], pp[0]])
#                 geo.append([pt[1], pt[0]])
#
#                 # calculate new shift
#                 shift = shift - (L_el + L_m)
#                 # ic(shift)
#
#         elif n > 1 and n != n_cell:
#             # DRAW ARC:
#             pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
#             pt = [-shift + x1, y1]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # DRAW LINE CONNECTING ARCS
#             lineTo(pt, [-shift + x2, y2], step)
#             pt = [-shift + x2, y2]
#             geo.append([pt[1], pt[0]])
#
#             # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
#             pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
#             pt = [L_bp_l + L_m - shift, Req]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#
#             geo.append([pt[1], pt[0]])
#
#             # EQUATOR ARC TO NEXT POINT
#             # half of bounding box is required,
#             # start is the lower coordinate of the bounding box and end is the upper
#             pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
#                         [L_m + L_m - x2 + 2 * L_bp_l - shift, Req])
#             pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#
#             geo.append([pt[1], pt[0]])
#
#             # STRAIGHT LINE TO NEXT POINT
#             lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
#             pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
#             geo.append([pt[1], pt[0]])
#
#             # ARC
#             # half of bounding box is required,
#             # start is the lower coordinate of the bounding box and end is the upper
#             pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
#                         [L_bp_l + L_m + L_m - shift, y1])
#             pt = [L_bp_l + L_m + L_m - shift, Ri_m]
#
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # calculate new shift
#             shift = shift - 2 * L_m
#         else:
#             # DRAW ARC:
#             pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
#             pt = [-shift + x1, y1]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # DRAW LINE CONNECTING ARCS
#             lineTo(pt, [-shift + x2, y2], step)
#             pt = [-shift + x2, y2]
#             geo.append([pt[1], pt[0]])
#
#             # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
#             pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
#             pt = [L_bp_l + L_m - shift, Req]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # EQUATOR ARC TO NEXT POINT
#             # half of bounding box is required,
#             # start is the lower coordinate of the bounding box and end is the upper
#             pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
#                         [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, Req])
#             pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#             # STRAIGHT LINE TO NEXT POINT
#             lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
#             pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
#             geo.append([pt[1], pt[0]])
#
#             # ARC
#             # half of bounding box is required,
#             # start is the lower coordinate of the bounding box and end is the upper
#             pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, [pt[0], Ri_er],
#                         [L_bp_l + L_m + L_er - shift, y1er])
#             pt = [L_bp_l + L_m + L_er - shift, Ri_er]
#             for pp in pts:
#                 if (np.around(pp, 12) != np.around(pt, 12)).all():
#                     geo.append([pp[1], pp[0]])
#             geo.append([pt[1], pt[0]])
#
#     # BEAM PIPE
#     # reset shift
#     shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2
#
#     if L_bp_r > 0:  # if there's a problem, check here.
#         lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
#         pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
#         geo.append([pt[1], pt[0]])
#
#     # END PATH
#     lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step)  # to add beam pipe to right
#     pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
#     # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
#     # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
#     geo.append([pt[1], pt[0]])
#
#     if bc:
#         # draw right boundary condition
#         ax.plot([shift, shift], [-Ri_er, Ri_er],
#                 [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
#                 [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)
#
#     # CLOSE PATH
#     # lineTo(pt, start_point, step)
#     # geo.append([start_point[1], start_point[0]])
#     geo = np.array(geo)
#
#     # recenter asymmetric cavity to center
#     shift_left = (L_bp_l + L_bp_r + L_el + L_er + 2*(n-1)*L_m)/2
#     if n_cell == 1:
#         shift_to_center = L_er + L_bp_r
#     else:
#         shift_to_center = n_cell*L_m + L_bp_r
#
#     top = ax.plot(geo[:, 1]-shift_left + shift_to_center, geo[:, 0], lw=4)
#     bottom = ax.plot(geo[:, 1]-shift_left + shift_to_center, -geo[:, 0], lw=4, c=top[0].get_color())
#
#     return ax


def plot_cavity_geometry_cli(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False,
                             ignore_degenerate=False, **kwargs):
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_aspect('equal')

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(IC)[:7] * scale * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(OC)[:7] * scale * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(OC_R)[:7] * scale * 1e-3

    if BP.lower() == 'both':
        L_bp_l = 4 * L_m
        L_bp_r = 4 * L_m

    elif BP.lower() == 'left':
        L_bp_l = 4 * L_m
        L_bp_r = 0.000

    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4 * L_m
    else:
        L_bp_l = 0.000
        L_bp_r = 0.000

    step = 0.005

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
    geo.append([start_point[1], start_point[0]])

    lineTo(start_point, [-shift, Ri_el], step)
    pt = [-shift, Ri_el]
    geo.append([pt[1], pt[0]])

    if bc:
        # draw left boundary condition
        ax.plot([-shift, -shift], [-Ri_el, Ri_el],
                [-shift - 0.2 * L_m, -shift - 0.2 * L_m], [-0.5 * Ri_el, 0.5 * Ri_el],
                [-shift - 0.4 * L_m, -shift - 0.4 * L_m], [-0.1 * Ri_el, 0.1 * Ri_el], c='b', lw=4, zorder=100)

    # ADD BEAM PIPE LENGTH
    if L_bp_l != 0:
        lineTo(pt, [L_bp_l - shift, Ri_el], step)
        pt = [L_bp_l - shift, Ri_el]

        geo.append([pt[1], pt[0]])

    for n in range(1, n_cell + 1):
        if n == 1:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
            pt = [-shift + x1el, y1el]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2el, y2el], step)
            pt = [-shift + x2el, y2el]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
            pt = [L_bp_l + L_el - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            if n_cell == 1:
                if L_bp_r > 0:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er])

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])

                    geo.append([pt[1], pt[0]])

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
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

            else:
                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_el + L_m - x2 + 2 * L_bp_l - shift, y2])
                pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                geo.append([pt[1], pt[0]])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_el + L_m - shift, Ri_m])
                pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # calculate new shift
                shift = shift - (L_el + L_m)
                # ic(shift)

        elif n > 1 and n != n_cell:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])

            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt,
                        [L_m + L_m - x2 + 2 * L_bp_l - shift, y2])
            pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])

            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
            pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                        [L_bp_l + L_m + L_m - shift, Ri_m])
            pt = [L_bp_l + L_m + L_m - shift, Ri_m]

            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # calculate new shift
            shift = shift - 2 * L_m
        else:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                        [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
            pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
            pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                        [L_bp_l + L_m + L_er - shift, Ri_er])
            pt = [L_bp_l + L_m + L_er - shift, Ri_er]
            for pp in pts:
                if (np.around(pp, 12) != np.around(pt, 12)).all():
                    geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

    # BEAM PIPE
    # reset shift
    shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

    if L_bp_r > 0:  # if there's a problem, check here.
        lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
        geo.append([pt[1], pt[0]])

    # END PATH
    lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0], step)  # to add beam pipe to right
    pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
    # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
    # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
    geo.append([pt[1], pt[0]])

    # append start point
    geo.append([start_point[1], start_point[0]])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0]])
    geo = np.array(geo)

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


def plot_cavity_geometry_cli_flattop(IC, OC, OC_R, BP, n_cell, scale=1, ax=None, bc=None, tangent_check=False,
                             ignore_degenerate=False, **kwargs):
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

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req, lft = np.array(IC[:8]) * scale
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req, lft_el = np.array(OC[:8]) * scale
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req, lft_er = np.array(OC_R[:8]) * scale

    step = 0.005

    if BP.lower() == 'both':
        L_bp_l = 4 * L_m
        L_bp_r = 4 * L_m
    elif BP.lower() == 'none':
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #
    elif BP.lower() == 'left':
        L_bp_l = 4 * L_m
        L_bp_r = 0.000
    elif BP.lower() == 'right':
        L_bp_l = 0.000
        L_bp_r = 4 * L_m
    else:
        L_bp_l = 0.000  # 4 * L_m  #
        L_bp_r = 0.000  # 4 * L_m  #

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er + (n_cell - 2) * lft + lft_el + lft_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el
    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l, lft_el/2, tangent_check=tangent_check)
    x1el, y1el, x2el, y2el = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l, lft/2, tangent_check=tangent_check)
    x1, y1, x2, y2 = df[0]
    if not ignore_degenerate:
        msg = df[-2]
        if msg != 1:
            error('Parameter set leads to degenerate geometry.')
            # save figure of error
            return

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r, lft_er/2, tangent_check=tangent_check)
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
    geo.append([start_point[1], start_point[0]])

    lineTo(start_point, [-shift, Ri_el], step)
    pt = [-shift, Ri_el]
    geo.append([pt[1], pt[0]])

    # ADD BEAM PIPE LENGTH
    if L_bp_l != 0:
        lineTo(pt, [L_bp_l - shift, Ri_el], step)
        pt = [L_bp_l - shift, Ri_el]

        geo.append([pt[1], pt[0]])

    for n in range(1, n_cell + 1):
        if n == 1:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
            pt = [-shift + x1el, y1el]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2el, y2el], step)
            pt = [-shift + x2el, y2el]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
            pt = [L_bp_l + L_el - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # flat top
            ax.scatter(pt[0], pt[1], c='g', ec='k')
            pts = lineTo(pt, [L_bp_l + L_el + lft_el - shift, Req], step)
            pt = [L_bp_l + L_el + lft_el - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])
            ax.scatter(pt[0], pt[1], c='r', ec='k')

            if n_cell == 1:
                if L_bp_r > 0:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + L_er - x2er + + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, Ri_er])

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])

                    geo.append([pt[1], pt[0]])

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er])
                    pt = [L_el + lft_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                    pt = [L_el + lft_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    geo.append([pt[1], pt[0]])

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, Ri_er])
                    pt = [L_bp_l + L_el + lft_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            geo.append([pp[1], pp[0]])
                    geo.append([pt[1], pt[0]])
            else:
                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                            [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, Req])
                pt = [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_el + lft_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_el + lft_el + L_m - shift, y1])
                pt = [L_bp_l + L_el + lft_el + L_m - shift, Ri_m]
                for pp in pts:
                    geo.append([pp[1], pp[0]])
                geo.append([pt[1], pt[0]])

                # calculate new shift
                shift = shift - (L_el + L_m + lft_el)
                # ic(shift)

        elif n > 1 and n != n_cell:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # flat top
            pts = lineTo(pt, [L_bp_l + L_m + lft - shift, Req], step)
            pt = [L_bp_l + L_el + lft - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_bp_l + lft - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                        [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, Req])
            pt = [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1], step)
            pt = [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_m + lft + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                        [L_bp_l + L_m + L_m + lft - shift, y1])
            pt = [L_bp_l + L_m + L_m + lft - shift, Ri_m]

            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # calculate new shift
            shift = shift - 2 * L_m - (lft_el + lft)
        else:
            # DRAW ARC:
            pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
            pt = [-shift + x1, y1]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW LINE CONNECTING ARCS
            pts = lineTo(pt, [-shift + x2, y2], step)
            pt = [-shift + x2, y2]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
            pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
            pt = [L_bp_l + L_m - shift, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # flat top
            pts = lineTo(pt, [L_bp_l + L_m + lft_er - shift, Req], step)
            pt = [L_bp_l + L_m + lft_er - shift, Req, Req]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # EQUATOR ARC TO NEXT POINT
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + lft_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                        [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, Req])
            pt = [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # STRAIGHT LINE TO NEXT POINT
            pts = lineTo(pt, [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
            pt = [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

            # ARC
            # half of bounding box is required,
            # start is the lower coordinate of the bounding box and end is the upper
            pts = arcTo(L_m + L_er + lft_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                        [L_bp_l + L_m + L_er + lft_er - shift, y1er])
            pt = [L_bp_l + L_m + L_er + lft_er - shift, Ri_er]
            for pp in pts:
                geo.append([pp[1], pp[0]])
            geo.append([pt[1], pt[0]])

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
            geo.append([pp[1], pp[0]])
        geo.append([pt[1], pt[0]])

    # END PATH
    pts = lineTo(pt, [
        2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0],
                 step)  # to add beam pipe to right
    pt = [2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0]
    # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
    geo.append([pt[1], pt[0]])

    # CLOSE PATH
    lineTo(pt, start_point, step)
    geo.append([pt[1], pt[0]])

    # append start point
    geo.append([start_point[1], start_point[0]])

    if bc:
        # draw right boundary condition
        ax.plot([shift, shift], [-Ri_er, Ri_er],
                [shift + 0.2 * L_m, shift + 0.2 * L_m], [-0.5 * Ri_er, 0.5 * Ri_er],
                [shift + 0.4 * L_m, shift + 0.4 * L_m], [-0.1 * Ri_er, 0.1 * Ri_er], c='b', lw=4, zorder=100)

    # CLOSE PATH
    # lineTo(pt, start_point, step)
    # geo.append([start_point[1], start_point[0]])
    geo = np.array(geo)

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


def orig_writeCavityForMultipac(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None, beampipe='none',
                                plot=False, unit=1e-3, scale=1):
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

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(mid_cell[:7]) * unit * scale
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(end_cell_left[:7]) * unit * scale
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(end_cell_right[:7]) * unit * scale

    step = 0.001

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

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l)
    x1, y1, x2, y2 = df[0]

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r)
    x1er, y1er, x2er, y2er = df[0]

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
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step, plot)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req],
                            plot)
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                                [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, Req], plot)
                    pt = [L_el + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                    pt = [L_el + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, y1er], plot)

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                                [L_el + L_m - x2 + 2 * L_bp_l - shift, Req], plot)
                    pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                    pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                                [L_bp_l + L_el + L_m - shift, y1], plot)
                    pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m)

            elif n > 1 and n != n_cell:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                            [L_m + L_m - x2 + 2 * L_bp_l - shift, Req], plot)
                pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_m + L_m - shift, y1], plot)
                pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # calculate new shift
                shift = shift - 2 * L_m
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                            [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, Req], plot)
                pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er - shift, y1er], plot)
                pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

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
        plt.show()

        plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]


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
                                     plot=False, unit=1e-3, scale=1):
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


def orig_writeCavityForMultipac_flat_top(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None,
                                         beampipe='none',
                                         plot=False, unit=1e-3, scale=1):
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

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req, lft = np.array(mid_cell[:8]) * unit * scale
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req, lft_el = np.array(end_cell_left[:8]) * unit * scale
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req, lft_er = np.array(end_cell_right[:8]) * unit * scale

    step = 0.001

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
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er + (n_cell - 2) * lft + lft_el + lft_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l)
    x1el, y1el, x2el, y2el = df[0]

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l)
    x1, y1, x2, y2 = df[0]

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r)
    x1er, y1er, x2er, y2er = df[0]

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
        if L_bp_l != 0:
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
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step, plot)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req], plot)
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # flat top
                pts = lineTo(pt, [L_bp_l + L_el + lft_el - shift, Req], step, plot)
                pt = [L_bp_l + L_el + lft_el - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_er, A_er, B_er, step, pt,
                                [L_el + lft_el + L_er - x2er + 2 * L_bp_l - shift, y2er], plot)
                    pt = [L_el + lft_el + L_er - x2er + 2 * L_bp_l - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + lft_el + L_er - x1er + 2 * L_bp_l - shift, y1er], step, plot)
                    pt = [L_el + lft_el + L_er - x1er + 2 * L_bp_l - shift, y1er]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, Ri_er], plot)

                    pt = [L_bp_l + lft_el + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er + lft_el)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_m, A_m, B_m, step, pt,
                                [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2], plot)
                    pt = [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                    pt = [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                                [L_bp_l + L_el + lft_el + L_m - shift, y1], plot)
                    pt = [L_bp_l + L_el + lft_el + L_m - shift, y1]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m + lft_el)
                    # ic(shift)

            elif n > 1 and n != n_cell:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + lft - shift, Req], step, plot)
                pt = [L_bp_l + L_el + lft - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l + lft - shift, Req - B_m, A_m, B_m, step, pt,
                            [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2], plot)
                pt = [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1], step, plot)
                pt = [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + lft + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt,
                            [L_bp_l + L_m + L_m + lft - shift, Ri_m], plot)
                pt = [L_bp_l + L_m + L_m + lft - shift, Ri_m]
                ic(pt)
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # calculate new shift
                shift = shift - 2 * L_m - (lft_el + lft)
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1], plot)
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2, y2], step, plot)
                pt = [-shift + x2, y2]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req], plot)
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + lft_er - shift, Req], step, plot)
                pt = [L_bp_l + L_m + lft_er - shift, Req, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + lft_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, pt,
                            [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er], plot)
                pt = [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                pt = [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + lft_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er + lft_er - shift, y1er], plot)
                pt = [L_bp_l + L_m + L_er + lft_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # BEAM PIPE
        # reset shift

        shift = (L_bp_r + L_bp_l + L_el + lft_el + (n_cell - 1) * 2 * L_m + (n_cell - 2) * lft + L_er + lft_er) / 2
        pts = lineTo(pt, [
            L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + (n_cell - 2) * lft + lft_el + lft_er + L_el + L_er - shift,
            Ri_er], step, plot)

        if L_bp_r != 0:
            pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cell - 2) * lft + lft_el + lft_er - shift,
                  Ri_er]
            for pp in pts:
                fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # END PATH
        pts = lineTo(pt, [
            2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0],
                     step, plot)  # to add beam pipe to right
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   1.0000000e+00\n")

        # CLOSE PATH
        pts = lineTo(pt, start_point, step, plot)
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

    print(' plot state', plot)
    if plot:
        print(' in here plotting')
        plt.show()


def writeCavityForMultipac_flat_top(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None,
                                    beampipe='none',
                                    plot=False, unit=1e-3, scale=1):
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

    A_m, B_m, a_m, b_m, Ri_m, L_m, Req, lft = np.array(mid_cell[:8]) * unit * scale
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req, lft_el = np.array(end_cell_left[:8]) * unit * scale
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req, lft_er = np.array(end_cell_right[:8]) * unit * scale

    step = 0.001

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
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er + (n_cell - 2) * lft + lft_el + lft_er) / 2

    # calculate angles outside loop
    # CALCULATE x1_el, y1_el, x2_el, y2_el

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l)
    x1el, y1el, x2el, y2el = df[0]

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l)
    x1, y1, x2, y2 = df[0]

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r)
    x1er, y1er, x2er, y2er = df[0]

    with open(file_path, 'w') as fil:
        fil.write("   2.0000000e-03   0.0000000e+00   0.0000001e+00   0.0000000e+00\n")
        fil.write("   1.25000000e-02   0.0000000e+00   0.0000001e+00   0.0000000e+00\n")  # a point inside the structure
        fil.write("  -3.1415927e+00  -2.7182818e+00   0.0000001e+00   0.0000000e+00\n")  # a point outside the structure

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   3.0000001e+00   0.0000000e+00\n")

        pts = lineTo(start_point, [-shift, Ri_el], step, plot)
        pt = [-shift, Ri_el]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000001e+00   1.0000000e+00\n")

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            pts = lineTo(pt, [L_bp_l - shift, Ri_el], step, plot)
            pt = [L_bp_l - shift, Ri_el]
            for pp in pts:
                fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000001e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000001e+00   1.0000000e+00\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el], plot)
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                pts = lineTo(pt, [-shift + x2el, y2el], step, plot)
                pt = [-shift + x2el, y2el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req], plot)
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # flat top
                pts = lineTo(pt, [L_bp_l + L_el + lft_el - shift, Req], step, plot)
                pt = [L_bp_l + L_el + lft_el - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                                [L_el + lft_el + L_er - x2er + 2 * L_bp_l - shift, Req], plot)
                    pt = [L_el + lft_el + L_er - x2er + 2 * L_bp_l - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}0   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + lft_el + L_er - x1er + 2 * L_bp_l - shift, y1er], step, plot)
                    pt = [L_el + lft_el + L_er - x1er + 2 * L_bp_l - shift, y1er]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + lft_el + L_er - shift, y1er], plot)

                    pt = [L_bp_l + lft_el + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er + lft_el)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l + lft_el - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                                [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, Req], plot)
                    pt = [L_el + lft_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    pts = lineTo(pt, [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1], step, plot)
                    pt = [L_el + lft_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    for pp in pts:
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + lft_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                                [L_bp_l + L_el + lft_el + L_m - shift, y1], plot)
                    pt = [L_bp_l + L_el + lft_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m + lft_el)
                    # ic(shift)

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

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + lft - shift, Req], step, plot)
                pt = [L_bp_l + L_el + lft - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l + lft - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                            [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, Req], plot)
                pt = [L_m + L_m + lft - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1], step, plot)
                pt = [L_m + L_m + lft - x1 + 2 * L_bp_l - shift, y1]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + lft + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_m + L_m + lft - shift, y1], plot)
                pt = [L_bp_l + L_m + L_m + lft - shift, Ri_m]
                ic(pt)
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # calculate new shift
                shift = shift - 2 * L_m - (lft_el + lft)
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

                # flat top
                pts = lineTo(pt, [L_bp_l + L_m + lft_er - shift, Req], step, plot)
                pt = [L_bp_l + L_m + lft_er - shift, Req, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + lft_er + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                            [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, Req], plot)
                pt = [L_m + L_er + lft_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                pts = lineTo(pt, [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er], step, plot)
                pt = [L_m + L_er + lft_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + lft_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er + lft_er - shift, y1er], plot)
                pt = [L_bp_l + L_m + L_er + lft_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   {n}   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   {n}   1.0000000e+00\n")

        # BEAM PIPE
        # reset shift

        shift = (L_bp_r + L_bp_l + L_el + lft_el + (n_cell - 1) * 2 * L_m + (n_cell - 2) * lft + L_er + lft_er) / 2
        pts = lineTo(pt, [
            L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + (n_cell - 2) * lft + lft_el + lft_er + L_el + L_er - shift,
            Ri_er], step, plot)

        if L_bp_r != 0:
            pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r + (n_cell - 2) * lft + lft_el + lft_er - shift,
                  Ri_er]
            for pp in pts:
                fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000001e+00   1.0000000e+00\n")
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000001e+00   1.0000000e+00\n")

        # END PATH
        pts = lineTo(pt, [
            2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0],
                     step, plot)  # to add beam pipe to right
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + (n_cell - 2) * lft + lft_el + lft_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000001e+00   1.0000000e+00\n")

        # CLOSE PATH
        pts = lineTo(pt, start_point, step, plot)
        # for pp in pts:
        #     fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000001e+00   0.0000000e+00\n")

    print(' plot state', plot)
    if plot:
        print(' in here plotting')
        plt.show()


def write_geometry_ngsolve(file_path, n_cell, mid_cell, end_cell_left=None, end_cell_right=None, beampipe='none',
                           plot=False):
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

    # # TESLA end cell 2
    A_m, B_m, a_m, b_m, Ri_m, L_m, Req = np.array(mid_cell[:7]) * 1e-3
    A_el, B_el, a_el, b_el, Ri_el, L_el, Req = np.array(end_cell_left[:7]) * 1e-3
    A_er, B_er, a_er, b_er, Ri_er, L_er, Req = np.array(end_cell_right[:7]) * 1e-3

    step = 0.0005

    if beampipe.lower() == 'both':
        L_bp_l = 4 * L_m
        L_bp_r = 4 * L_m
    elif beampipe.lower() == 'none':
        L_bp_l = 0.0001  # 4 * L_m  #
        L_bp_r = 0.0001  # 4 * L_m  #
    elif beampipe.lower() == 'left':
        L_bp_l = 4 * L_m
        L_bp_r = 0.0001
    elif beampipe.lower() == 'right':
        L_bp_l = 0.0001
        L_bp_r = 4 * L_m
    else:
        L_bp_l = 0.0001  # 4 * L_m  #
        L_bp_r = 0.0001  # 4 * L_m  #

    # calculate shift
    shift = (L_bp_r + L_bp_l + L_el + (n_cell - 1) * 2 * L_m + L_er) / 2
    # shift = 0
    # shift = L_m  # for end cell

    df = tangent_coords(A_el, B_el, a_el, b_el, Ri_el, L_el, Req, L_bp_l)
    x1el, y1el, x2el, y2el = df[0]

    # CALCULATE x1, y1, x2, y2
    df = tangent_coords(A_m, B_m, a_m, b_m, Ri_m, L_m, Req, L_bp_l)
    x1, y1, x2, y2 = df[0]

    # CALCULATE x1_er, y1_er, x2_er, y2_er
    df = tangent_coords(A_er, B_er, a_er, b_er, Ri_er, L_er, Req, L_bp_r)
    x1er, y1er, x2er, y2er = df[0]
    default_folder = "D:\Dropbox\multipacting\MPGUI21"

    if file_path is None:
        file_path = default_folder

    with open(fr'{file_path}\geodata.n', 'w') as fil:
        fil.write("   2.0000000e-03   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")
        fil.write("   1.25000000e-02   0.0000000e+00   0.0000000e+00   0.0000000e+00\n")  # a point inside the structure
        fil.write("  -3.1415927e+00  -2.7182818e+00   0.0000000e+00   0.0000000e+00\n")  # a point outside the structure

        # SHIFT POINT TO START POINT
        start_point = [-shift, 0]
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

        lineTo(start_point, [-shift, Ri_el], step)
        pt = [-shift, Ri_el]
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # ADD BEAM PIPE LENGTH
        if L_bp_l != 0:
            lineTo(pt, [L_bp_l - shift, Ri_el], step)
            pt = [L_bp_l - shift, Ri_el]
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        for n in range(1, n_cell + 1):
            if n == 1:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_el + b_el, a_el, b_el, step, pt, [-shift + x1el, y1el])
                pt = [-shift + x1el, y1el]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2el, y2el], step)
                pt = [-shift + x2el, y2el]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_el + L_bp_l - shift, Req - B_el, A_el, B_el, step, pt, [L_bp_l + L_el - shift, Req])
                pt = [L_bp_l + L_el - shift, Req]
                for pp in pts:
                    fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                if n_cell == 1:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                                [L_el + L_er - x2er + 2 * L_bp_l - shift, Req])
                    pt = [L_el + L_er - x2er + 2 * L_bp_l - shift, y2er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_er - x1er + 2 * L_bp_l - shift, y1er], step)
                    pt = [L_el + L_er - x1er + 2 * L_bp_l - shift, y1er]
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                                [L_bp_l + L_el + L_er - shift, y1er])

                    pt = [L_bp_l + L_el + L_er - shift, Ri_er]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_er)
                    # ic(shift)
                else:
                    # EQUATOR ARC TO NEXT POINT
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                                [L_el + L_m - x2 + 2 * L_bp_l - shift, Req])
                    pt = [L_el + L_m - x2 + 2 * L_bp_l - shift, y2]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # STRAIGHT LINE TO NEXT POINT
                    lineTo(pt, [L_el + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                    pt = [L_el + L_m - x1 + 2 * L_bp_l - shift, y1]
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # ARC
                    # half of bounding box is required,
                    # start is the lower coordinate of the bounding box and end is the upper
                    pts = arcTo(L_el + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                                [L_bp_l + L_el + L_m - shift, y1])
                    pt = [L_bp_l + L_el + L_m - shift, Ri_m]
                    for pp in pts:
                        if (np.around(pp, 12) != np.around(pt, 12)).all():
                            fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                    fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                    # calculate new shift
                    shift = shift - (L_el + L_m)
                    # ic(shift)

            elif n > 1 and n != n_cell:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, [pt[0], Req - B_m],
                            [L_m + L_m - x2 + 2 * L_bp_l - shift, Req])
                pt = [L_m + L_m - x2 + 2 * L_bp_l - shift, y2]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_m - x1 + 2 * L_bp_l - shift, y1], step)
                pt = [L_m + L_m - x1 + 2 * L_bp_l - shift, y1]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_m + L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, [pt[0], Ri_m],
                            [L_bp_l + L_m + L_m - shift, y1])
                pt = [L_bp_l + L_m + L_m - shift, Ri_m]
                ic(pt)
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # calculate new shift
                shift = shift - 2 * L_m
            else:
                # DRAW ARC:
                pts = arcTo(L_bp_l - shift, Ri_m + b_m, a_m, b_m, step, pt, [-shift + x1, y1])
                pt = [-shift + x1, y1]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW LINE CONNECTING ARCS
                lineTo(pt, [-shift + x2, y2], step)
                pt = [-shift + x2, y2]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # DRAW ARC, FIRST EQUATOR ARC TO NEXT POINT
                pts = arcTo(L_m + L_bp_l - shift, Req - B_m, A_m, B_m, step, pt, [L_bp_l + L_m - shift, Req])
                pt = [L_bp_l + L_m - shift, Req]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # EQUATOR ARC TO NEXT POINT
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_bp_l - shift, Req - B_er, A_er, B_er, step, [pt[0], Req - B_er],
                            [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, Req])
                pt = [L_m + L_er - x2er + L_bp_l + L_bp_r - shift, y2er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # STRAIGHT LINE TO NEXT POINT
                lineTo(pt, [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er], step)
                pt = [L_m + L_er - x1er + L_bp_l + L_bp_r - shift, y1er]
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

                # ARC
                # half of bounding box is required,
                # start is the lower coordinate of the bounding box and end is the upper
                pts = arcTo(L_m + L_er + L_bp_l - shift, Ri_er + b_er, a_er, b_er, step, pt,
                            [L_bp_l + L_m + L_er - shift, y1er])
                pt = [L_bp_l + L_m + L_er - shift, Ri_er]
                for pp in pts:
                    if (np.around(pp, 12) != np.around(pt, 12)).all():
                        fil.write(f"  {pp[1]:.16E}  {pp[0]:.16E}   1.0000000e+00   1.0000000e+00\n")
                fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   1.0000000e+00   1.0000000e+00\n")

        # BEAM PIPE
        # reset shift
        shift = (L_bp_r + L_bp_l + (n_cell - 1) * 2 * L_m + L_el + L_er) / 2

        if L_bp_r != 0:  # if there's a problem, check here.
            lineTo(pt, [L_bp_r + L_bp_l + 2 * (n_cell - 1) * L_m + L_el + L_er - shift, Ri_er], step)
            pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, Ri_er]
            fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   3.0000000e+00   0.0000000e+00\n")

        # END PATH
        lineTo(pt, [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0],
               step)  # to add beam pipe to right
        pt = [2 * (n_cell - 1) * L_m + L_el + L_er + L_bp_l + L_bp_r - shift, 0]
        # lineTo(pt, [2 * n_cell * L_er + L_bp_l - shift, 0], step)
        # pt = [2 * n_cell * L_er + L_bp_l - shift, 0]
        fil.write(f"  {pt[1]:.16E}  {pt[0]:.16E}   0.0000000e+00   0.0000000e+00\n")

        # CLOSE PATH
        lineTo(pt, start_point, step)
        fil.write(f"  {start_point[1]:.16E}  {start_point[0]:.16E}   0.0000000e+00   0.0000000e+00\n")
    plt.gca().set_aspect('equal', 'box')
    plt.show()


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


def plot_gun():
    """

    Returns
    -------

    """
    y1, R2, T2, L3, R4, L5, R6, L7, R8, T9, R10, T10, L11, R12, L13, R14, x = \
        [1.5, 3, np.deg2rad(45), 24, 5, 11, 6, 19, 4, np.deg2rad(8), 3, np.deg2rad(40), 5, 3, 3, 3, 1]

    # calcualte R9
    R9 = (((y1 + R2 * np.sin(T2) + L3 * np.cos(T2) + R4 * np.sin(T2) + L5 + R6) -
           (R14 + L13 + R12 * np.sin(T10) + L11 * np.cos(T10) + R10 * np.sin(T10) + x + R8 * (
                       1 - np.sin(T9))))) / np.sin(T9)
    print(R9)

    step = 0.1
    geo = []

    start_pt = [0, 0]
    geo.append([start_pt[1], start_pt[0]])
    # DRAW LINE CONNECTING ARCS
    lineTo(start_pt, [0, y1], step)
    pt = [0, y1]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(-R2, y1, R2, R2, pt, [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)], 0, T2, step)
    pt = [R2 * np.cos(T2) - R2, y1 + R2 * np.sin(T2)]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)], step)
    pt = [R2 * np.cos(T2) - R2 - L3 * np.cos(T2), y1 + R2 * np.sin(T2) + L3 * np.sin(T2)]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] + R4 * np.cos(T2), pt[1] + R4 * np.sin(T2), R4, R4,
                     pt, [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)], -(np.pi - T2), np.pi, step)
    pt = [pt[0] - (R4 - R4 * np.cos(T2)), pt[1] + R4 * np.sin(T2)]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0], pt[1] + L5], step)
    pt = [pt[0], pt[1] + L5]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] + R6, pt[1], R6, R6, pt, [pt[0] + R6, pt[1] + R6], np.pi, np.pi / 2, step)
    pt = [pt[0] + R6, pt[1] + R6]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0] + L7, pt[1]], step)
    pt = [pt[0] + L7, pt[1]]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0], pt[1] - R8, R8, R8, pt, [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))],
                     np.pi / 2, T9, step)
    pt = [pt[0] + R8 * np.cos(T9), pt[1] - (R8 - R8 * np.sin(T9))]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] - R9 * np.cos(T9), pt[1] - R9 * np.sin(T9), R9, R9,
                     pt, [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)], T9, 0, step)
    pt = [pt[0] + (R9 - R9 * np.cos(T9)), pt[1] - R9 * np.sin(T9)]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] - R10, pt[1], R10, R10,
                     pt, [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)], 0, -T10, step)
    pt = [pt[0] - (R10 - R10 * np.cos(T10)), pt[1] - R10 * np.sin(T10)]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)], step)
    pt = [pt[0] - L11 * np.sin(T10), pt[1] - L11 * np.cos(T10)]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] + R12 * np.cos(T10), pt[1] - R12 * np.sin(T10), R12, R12,
                     pt, [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)], (np.pi - T10), np.pi, step)
    pt = [pt[0] - (R12 - R12 * np.cos(T10)), pt[1] - R12 * np.sin(T10)]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0], pt[1] - L13], step)
    pt = [pt[0], pt[1] - L13]
    geo.append([pt[1], pt[0]])

    # DRAW ARC:
    pts = arcToTheta(pt[0] + R14, pt[1], R14, R14, pt, [pt[0] + R14, pt[1] - R14], np.pi, -np.pi / 2, step)
    pt = [pt[0] + R14, pt[1] - R14]
    for pp in pts:
        geo.append([pp[1], pp[0]])
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0] + 10 * y1, pt[1]], step)
    pt = [pt[0] + 10 * y1, pt[1]]
    geo.append([pt[1], pt[0]])

    # line
    lineTo(pt, [pt[0], pt[1] - y1], step)
    pt = [pt[0], pt[1] - x]
    geo.append([pt[1], pt[0]])

    geo = np.array(geo)

    top = plt.plot(geo[:, 1], geo[:, 0])
    bottom = plt.plot(geo[:, 1], -geo[:, 0], c=top[0].get_color())

    # plot legend wthout duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.gca().set_aspect('equal')
    plt.xlim(left=-30)

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


def f2b_slashes(path):
    """
    Replaces forward slashes with backward slashes for windows OS

    Parameters
    ----------
    path: str
        Directory path

    Returns
    -------

    """

    if os.name == 'nt':
        path = path.replace("/", "\\")
    else:
        path = path.replace('\\', '/')
    return Path(path)


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


def enforce_Req_continuity(par_mid, par_end_l, par_end_r, cell_type):
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


def save_tune_result(d, filename, projectDir, key, sim_folder='SLAN_Opt'):
    with open(fr"{projectDir}\SimulationData\{sim_folder}\{key}\{filename}", 'w') as file:
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

# def error(*arg):
#     print(colored(f'{arg[0]}', 'red'))
#
#
# def warning(*arg):
#     print(colored(f'{arg[0]}', 'yellow'))
#
#
# def running(*arg):
#     print(colored(f'{arg[0]}', 'cyan'))
#
#
# def info(*arg):
#     print(colored(f'{arg[0]}', 'blue'))
#
#
# def done(*arg):
#     print(colored(f'{arg[0]}', 'green'))
