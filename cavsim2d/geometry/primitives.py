"""Low-level meridian primitives: sampling lines and elliptic arcs into points."""
import matplotlib.pyplot as plt
import numpy as np
import math


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
