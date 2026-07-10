"""Writers that emit a cavity meridian contour as a gmsh ``.geo`` script."""
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cavsim2d.constants import *
from cavsim2d.utils.printing import *
from cavsim2d.geometry.tangency import tangent_coords
from cavsim2d.geometry.primitives import arcTo, lineTo


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

    # NOTE: plotting is handled by `write_cavity_geometry_cli_wo_gmsh`
    # (callers pass plot=True there). This gmsh-oriented variant only
    # populates `geo` when `write` is set, so there is nothing to draw
    # here. Keeping the function return stable for callers.

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
            # Only the top half gets the label — bottom is a visual mirror.
            bottom_kwargs = {k: v for k, v in kwargs.items() if k != 'label'}
            ax.plot((geo[:, 1] - shift_left + shift_to_center) * 1e3, -geo[:, 0] * 1e3,
                    c=top[0].get_color(), **bottom_kwargs)

        # plot legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
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
