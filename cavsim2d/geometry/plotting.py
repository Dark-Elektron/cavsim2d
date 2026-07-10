"""Matplotlib renderers for cavity meridian contours (no gmsh required)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cavsim2d.constants import *
from cavsim2d.utils.printing import *
from cavsim2d.geometry.tangency import tangent_coords
from cavsim2d.geometry.primitives import arcTo, lineTo


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
            # Only the top half gets the label — bottom is a visual mirror.
            bottom_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
            if 'c' in bottom_kwargs.keys():
                ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3,
                        **bottom_kwargs)
            else:
                ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3,
                        c=top[0].get_color(), **bottom_kwargs)

        # plot legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    return ax


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
