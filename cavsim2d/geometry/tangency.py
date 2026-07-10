"""Tangency and alpha-angle solvers for elliptical cell contours."""
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fsolve


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
