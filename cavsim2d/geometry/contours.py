"""Meridian contours for the standard cavity families, built on :class:`Profile`.

An elliptical cavity contour is the same shape repeated: every cell is a
*forward half* (iris arc -> tangent line -> equator arc) followed by an optional
flat top and a *backward half* (equator arc -> tangent line -> iris arc). One
builder therefore covers single-cell, multicell, asymmetric end cells, one-sided
beampipes, and the flat-top parameterisation.

Coordinates are in metres.
"""
import numpy as np
from cavsim2d.geometry.profile import Profile
from cavsim2d.geometry.tangency import tangent_coords


class DegenerateGeometry(ValueError):
    """The parameter set admits no tangent line from the iris to the equator ellipse."""


def tangent_offsets(cell, Req):
    """Tangent points of one half-cell, as offsets from its iris plane.

    ``tangent_coords`` returns absolute coordinates in a frame whose iris ellipse
    sits at ``z = L_bp``; the offsets ``x - L_bp`` are invariant under that
    translation, so solving at ``L_bp = 0`` gives offsets valid for any beampipe.
    """
    A, B, a, b, Ri, L = cell[:6]
    df = tangent_coords(A, B, a, b, Ri, L, Req, 0.0)
    if df[-2] != 1:
        raise DegenerateGeometry('Parameter set leads to degenerate geometry.')
    return df[0]        # (dx1, y1, dx2, y2)


def _beampipe_lengths(beampipe, L_bp):
    bp = beampipe.lower()
    return (L_bp if bp in ('both', 'left') else 0.0,
            L_bp if bp in ('both', 'right') else 0.0)


def _flat_lengths(mid, end_l, end_r, n_cells, flattop):
    """Flat-top length of each cell, indexed 1..n_cells."""
    if not flattop:
        return {k: 0.0 for k in range(1, n_cells + 1)}
    l_m, l_el, l_er = mid[7], end_l[7], end_r[7]
    if n_cells == 1:
        # matches the writer's shift, which counts (n_cells - 2) * l_m = -l_m here
        return {1: l_el + l_er - l_m}
    flats = {k: l_m for k in range(1, n_cells + 1)}
    flats[1] = l_el
    flats[n_cells] = l_er
    return flats


def half_cell_sequence(mid, end_l, end_r, n_cells):
    """The ``2 * n_cells`` half-cells of a cavity, left to right.

    ``[end_l, mid, mid, ..., mid, end_r]`` — cell *k* is the pair
    ``(half_cells[2k], half_cells[2k+1])``: its forward (left) and backward
    (right) half. This is the canonical ordering everything else indexes with.
    """
    if n_cells == 1:
        halves = [end_l, end_r]
    else:
        halves = [end_l] + [mid] * (2 * n_cells - 2) + [end_r]
    return [list(h) for h in halves]


def continuity_violations(half_cells, atol=1e-9):
    """Physical constraints joining adjacent half-cells.

    Two halves of the same cell share an equator (``Req``); the right half of one
    cell and the left half of the next share an iris (``Ri``). Returns a list of
    human-readable violations (empty when the geometry is consistent).
    """
    h = np.asarray(half_cells, dtype=float)
    n_cells = len(h) // 2
    bad = []
    for k in range(n_cells):
        if abs(h[2 * k][6] - h[2 * k + 1][6]) > atol:
            bad.append(f'cell{k + 1}: Req mismatch across the equator '
                       f'({h[2 * k][6]:.6g} vs {h[2 * k + 1][6]:.6g})')
    for k in range(n_cells - 1):
        if abs(h[2 * k + 1][4] - h[2 * k + 2][4]) > atol:
            bad.append(f'cell{k + 1}/cell{k + 2}: Ri mismatch across the iris '
                       f'({h[2 * k + 1][4]:.6g} vs {h[2 * k + 2][4]:.6g})')
    return bad


def elliptical_profile_from_half_cells(half_cells, beampipe, L_bp, flats=None,
                                       name='elliptical'):
    """Contour of an elliptical cavity whose half-cells vary independently.

    ``half_cells`` is a ``(2 * n_cells, >=7)`` sequence in **metres**, ordered by
    :func:`half_cell_sequence`. Each half carries its own ``(A, B, a, b, Ri, L, Req)``,
    so every cell may differ — which is what the multicell UQ representation needs.
    ``flats`` is an optional per-cell flat-top length.

    Raises :class:`DegenerateGeometry` if a half-cell has no tangent solution.
    """
    halves = [list(h) for h in half_cells]
    n_cells = len(halves) // 2
    if len(halves) != 2 * n_cells or n_cells < 1:
        raise ValueError('half_cells must hold an even number of half-cells')
    flats = [0.0] * n_cells if flats is None else list(flats)

    L_bp_l, L_bp_r = _beampipe_lengths(beampipe, L_bp)
    shift = (L_bp_l + L_bp_r + sum(h[5] for h in halves) + sum(flats)) / 2.0

    p = Profile(name)
    z = -shift
    p.start(z, 0.0)
    p.line_to(z, halves[0][4], 'PMC')                   # left aperture
    if L_bp_l > 0:
        z += L_bp_l
        p.line_to(z, halves[0][4], 'PEC')               # left beampipe

    for k in range(n_cells):
        left, right = halves[2 * k], halves[2 * k + 1]

        # forward half: iris plane at z -> equator
        A, B, a, b, Ri, L, Req = left[:7]
        dx1, y1, dx2, y2 = tangent_offsets(left, Req)
        p.ellipse_arc_to(z + dx1, y1, center=(z, Ri + b), semi_z=a, semi_r=b, boundary='PEC')
        p.line_to(z + dx2, y2, 'PEC')
        z_eq = z + L
        p.ellipse_arc_to(z_eq, Req, center=(z_eq, Req - B), semi_z=A, semi_r=B, boundary='PEC')

        if flats[k] > 0:                                # flat top across the equator
            z_eq += flats[k]
            p.line_to(z_eq, Req, 'PEC')

        # backward half: equator -> next iris plane
        A, B, a, b, Ri, L, Req = right[:7]
        dx1, y1, dx2, y2 = tangent_offsets(right, Req)
        z_next = z_eq + L
        p.ellipse_arc_to(z_next - dx2, y2, center=(z_eq, Req - B), semi_z=A, semi_r=B, boundary='PEC')
        p.line_to(z_next - dx1, y1, 'PEC')
        p.ellipse_arc_to(z_next, Ri, center=(z_next, Ri + b), semi_z=a, semi_r=b, boundary='PEC')
        z = z_next

    if L_bp_r > 0:
        z += L_bp_r
        p.line_to(z, halves[-1][4], 'PEC')              # right beampipe
    p.line_to(z, 0.0, 'PMC')                            # right aperture
    p.close('AXI')
    return p


def elliptical_profile(mid, end_l, end_r, n_cells, beampipe,
                       flattop=False, name='elliptical'):
    """Build the meridian :class:`Profile` of an elliptical (or flat-top) cavity.

    ``mid`` / ``end_l`` / ``end_r`` are the per-cell parameter sequences in
    **metres**: ``(A, B, a, b, Ri, L, Req)``, plus a trailing flat-top length
    ``l`` when ``flattop`` is set. ``Req`` is taken from ``mid`` for every cell,
    matching the writers (which force ``Req = Req_m``).

    A thin wrapper over :func:`elliptical_profile_from_half_cells`: it expands the
    three cell types into the ``2 * n_cells`` half-cells.

    Raises :class:`DegenerateGeometry` if any half-cell has no tangent solution.
    """
    halves = half_cell_sequence(mid, end_l, end_r, n_cells)
    for h in halves:
        h[6] = mid[6]                                   # writers force Req = Req_m
    flat_map = _flat_lengths(mid, end_l, end_r, n_cells, flattop)
    flats = [flat_map[k] for k in range(1, n_cells + 1)]
    return elliptical_profile_from_half_cells(halves, beampipe, 4 * mid[5],
                                              flats=flats, name=name)
