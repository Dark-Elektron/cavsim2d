"""Slope triangles for log-log convergence plots.

A slope triangle marks the rate at which an error falls as unknowns are added: a right
triangle beside the curve, its hypotenuse parallel to a straight-line fit on log-log axes,
labelled with the rate. It is only drawn where a rate exists. A figure of merit whose error
changes sign or size irregularly from mesh to mesh does not follow a power law, and a slope
fitted through it would be a number with no meaning. :func:`fit_convergence_rate` therefore
refuses a curve that is not straight on log-log axes or does not fall.
"""
import matplotlib.pyplot as plt
import numpy as np

# Minimum R^2 of the log-log fit, with three or more points, for a rate to count.
R2_MIN = 0.95


def fit_convergence_rate(x, y, floor=0.0, r2_min=R2_MIN):
    """Fit ``y ~ x**-rate`` on log-log axes; the rate if the data follow a power law.

    Points at or below *floor* (a round-off level) are left out. Returns
    ``(rate, intercept, log10_x_min, log10_x_max)`` for the kept points, or ``None`` when
    fewer than two remain, when three or more are not straight on log-log axes
    (R^2 < *r2_min*), or when the fitted error does not fall.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > max(floor, 0.0))
    if keep.sum() < 2:
        return None
    lx, ly = np.log10(x[keep]), np.log10(y[keep])
    if np.ptp(lx) == 0:
        return None
    k, c = np.polyfit(lx, ly, 1)
    if keep.sum() >= 3:
        ss_tot = np.sum((ly - ly.mean()) ** 2)
        ss_res = np.sum((ly - (k * lx + c)) ** 2)
        if ss_tot <= 0 or 1 - ss_res / ss_tot < r2_min:
            return None
    if k >= 0:
        return None
    return -k, c, lx.min(), lx.max()


def _triangle(fit, where, side, offset, own):
    """Triangle vertices and bounds, in log10 coordinates.

    The hypotenuse has the fitted slope, but it is placed against the curve itself
    (*own*: its log10 polyline) at both ends, *offset* times clear of it. A curve that
    bends away from its straight fit would otherwise cut through a triangle placed
    against the fit.
    """
    rate, c, lo, hi = fit
    # At most 0.3 decades wide, and no taller than ~0.9 decades, so a steep rate does
    # not produce a triangle several times the height of a shallow one.
    width = min(0.3, max(0.08, 0.4 * (hi - lo)), 0.9 / rate)
    lx0 = lo + where * max(hi - lo - width, 0.0)
    lx1 = lx0 + width
    order = np.argsort(own[0])
    yc0, yc1 = np.interp([lx0, lx1], own[0][order], own[1][order])
    d = np.log10(offset)
    if side == 'above':
        ly0 = max(yc0, yc1 + rate * width) + d
    else:
        ly0 = min(yc0, yc1 + rate * width) - d
    ly1 = ly0 - rate * width
    if side == 'above':      # corner top-right: "1" along the top, the rate on the right
        verts = [(lx0, ly0), (lx1, ly0), (lx1, ly1)]
    else:                    # corner bottom-left: the rate on the left, "1" along the bottom
        verts = [(lx0, ly0), (lx1, ly1), (lx0, ly1)]
    return verts, (lx0, lx1, ly1, ly0)


def _box(bounds, side, lab_w, lab_h):
    """The triangle's bounding box grown by its two labels."""
    lx0, lx1, ly1, ly0 = bounds
    if side == 'above':
        return lx0, lx1 + lab_w, ly1, ly0 + lab_h
    return lx0 - lab_w, lx1, ly1 - lab_h, ly0


def _hits_curve(box, lx, ly, margin_y):
    """Whether a (log10) polyline passes through the box."""
    x0, x1, y0, y1 = box
    inside = (lx >= x0) & (lx <= x1)
    if inside.any() and np.any((ly[inside] >= y0 - margin_y) & (ly[inside] <= y1 + margin_y)):
        return True
    xs = np.linspace(x0, x1, 12)
    order = np.argsort(lx)
    ok = (xs >= lx[order][0]) & (xs <= lx[order][-1])
    if not ok.any():
        return False
    ys = np.interp(xs[ok], lx[order], ly[order])
    return bool(np.any((ys >= y0 - margin_y) & (ys <= y1 + margin_y)))


def _overlaps(a, b):
    return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])


def add_slope_triangles(ax, curves, colors=None, floor=0.0, r2_min=R2_MIN):
    """Draw a slope triangle for every curve that follows a power law.

    *curves* is a list of ``(x, y)`` already plotted on the log-log axes *ax*, and *colors*
    their colours. Each triangle sits beside its curve, above or below its fit, at the
    position nearest the middle of the curve that clears every curve, label and triangle.
    Where nothing clears the curves (a crowded plot), it settles for clearing the other
    triangles. A curve that does not follow a power law (see :func:`fit_convergence_rate`)
    gets no triangle. Returns the fitted rates, ``None`` where none was drawn.
    """
    colors = colors or [None] * len(curves)
    logs = []
    for x, y in curves:
        x, y = np.asarray(x, float), np.asarray(y, float)
        ok = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        logs.append((np.log10(x[ok]), np.log10(y[ok])))
    # Label size in decades, from the axes' size on the page: a label is about 1.2 font
    # sizes tall and three characters wide.
    xl, yl = np.log10(ax.get_xlim()), np.log10(ax.get_ylim())
    font_px = plt.rcParams['font.size'] * ax.figure.dpi / 72
    px_per_dec_x = ax.bbox.width / max(xl[1] - xl[0], 1e-12)
    px_per_dec_y = ax.bbox.height / max(yl[1] - yl[0], 1e-12)
    lab_w = 1.8 * font_px / px_per_dec_x
    lab_h = 1.2 * font_px / px_per_dec_y
    margin_y = 0.3 * font_px / px_per_dec_y

    placed, rates = [], []
    for (x, y), color, own in zip(curves, colors, logs):
        fit = fit_convergence_rate(x, y, floor=floor, r2_min=r2_min)
        rates.append(None if fit is None else fit[0])
        if fit is None:
            continue
        # Positions along the curve, nearest the middle first, each side, two offsets.
        candidates = [(where, side, offset)
                      for offset in (4.0, 10.0)
                      for where in sorted(np.linspace(0, 1, 11), key=lambda w: abs(w - 0.5))
                      for side in ('below', 'above')]
        choice = None
        for avoid_curves in (True, False):
            for where, side, offset in candidates:
                verts, bounds = _triangle(fit, where, side, offset, own)
                box = _box(bounds, side, lab_w, lab_h)
                if avoid_curves and any(_hits_curve(box, lx, ly, margin_y)
                                        for lx, ly in logs if len(lx)):
                    continue
                if any(_overlaps(box, other) for other in placed):
                    continue
                choice = (verts, bounds, box, side)
                break
            if choice:
                break
        if choice is None:
            verts, bounds = _triangle(fit, 0.5, 'below', 4.0, own)
            choice = (verts, bounds, _box(bounds, 'below', lab_w, lab_h), 'below')
        verts, (lx0, lx1, ly1, ly0), box, side = choice
        placed.append(box)
        px, py = zip(*[(10 ** vx, 10 ** vy) for vx, vy in verts + [verts[0]]])
        ax.plot(px, py, color=color, lw=1)
        x0, x1, y1, y0 = 10 ** lx0, 10 ** lx1, 10 ** ly1, 10 ** ly0
        label = f'{fit[0]:.1f}'
        if side == 'above':
            ax.text(np.sqrt(x0 * x1), y0 * 1.5, '1', color=color, ha='center', va='bottom')
            ax.text(x1 * 1.06, np.sqrt(y0 * y1), label, color=color, ha='left', va='center')
        else:
            ax.text(x0 / 1.06, np.sqrt(y0 * y1), label, color=color, ha='right', va='center')
            ax.text(np.sqrt(x0 * x1), y1 / 1.5, '1', color=color, ha='center', va='top')
    return rates
