"""Perturbation models: how a fabrication tolerance maps onto random variables.

A tolerance can be posed in several ways, and they are genuinely different models of
the manufacturing route rather than coarse and fine versions of one model:

``'base'``
    One tolerance per cell **type** - the mid-cell group and the end-cell group - so the
    variable count does not grow with cell number. This is what a drawing tolerance
    usually states. Both end cells share a draw when they are the same design; when they
    differ they get their own, so ``k`` is 14 or 21 for the 7 elliptical variables.

``'multicup'``
    Every half-cell formed independently: ``14n`` variables for *n* cells. Both the iris
    and the equator are welds, and each averages two independent draws.

``'dumbbell'``
    Dumb-bells formed as units, as in the usual assembly sequence. Each half keeps its own
    ``A B a b L Req`` but the two halves of a dumb-bell share **one iris radius**, because
    that joint is machined rather than welded: ``13n + 1`` variables. Only the equators
    remain welds.

A model is expressed as a list of *independent variables*, each owning the
``(half-cell row, parameter)`` slots that receive its single draw. Sampling and cubature
then consume the same definition, so the two cannot drift apart.

Note on ``Req``: the halves meeting at an equator name the same physical radius
(cell *k* is ``half_cells[2k], half_cells[2k+1]``, so a cavity's first cell is
``end_l`` + a *mid* half). Independent draws there are reconciled by the seam weld, which
averages them - see :func:`cavsim2d.utils.shapes.perturb_half_cells_independent`.
"""
import numpy as np

from cavsim2d.utils.quadrature import generate_nodes

__all__ = ['perturbation_slots', 'perturbation_nodes', 'HALF_CELL_VARS']

#: Elliptical half-cell parameters, in the order :meth:`Cavity.half_cells` returns them.
HALF_CELL_VARS = ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')
_I_RI = HALF_CELL_VARS.index('Ri')


def _ends_are_identical(cav, atol=1e-12):
    """True when the two end half-cells carry the same geometry."""
    hc = np.asarray(cav.half_cells(), dtype=float)
    return bool(np.allclose(hc[0], hc[-1], atol=atol))


def perturbation_slots(cav, kind, split_ends=None):
    """Independent variables of a perturbation model, as lists of slots.

    Parameters
    ----------
    cav : Cavity
        Used for its cell count and, for ``'base'``, to decide whether the two end
        cells are the same design.
    kind : {'base', 'multicup', 'dumbbell'}
    split_ends : bool, optional
        ``'base'`` only. Give the two end cells their own random variables (k = 21)
        rather than one shared draw (k = 14). Default: decide from the geometry -
        split when the end half-cells differ, share when they are identical. Pass
        ``True`` to split even for identical ends, which models two separately formed
        cups with independent errors rather than one common tooling error.

    Returns
    -------
    list of list of (row, var) tuples - one entry per independent random variable.
    """
    n_cells = int(cav.n_cells)
    n_rows = 2 * n_cells
    mid_rows = list(range(1, n_rows - 1))
    nv = len(HALF_CELL_VARS)

    if kind == 'base':
        if split_ends is None:
            split_ends = not _ends_are_identical(cav)
        slots = [[(r, v) for r in mid_rows] for v in range(nv)]
        if split_ends:
            slots += [[(0, v)] for v in range(nv)]
            slots += [[(n_rows - 1, v)] for v in range(nv)]
        else:
            slots += [[(r, v) for r in (0, n_rows - 1)] for v in range(nv)]
        return slots

    if kind == 'multicup':
        return [[(r, v)] for r in range(n_rows) for v in range(nv)]

    if kind == 'dumbbell':
        # parts: the left end cup, each dumb-bell straddling an iris, the right end cup
        parts = ([[0]] + [[2 * i - 1, 2 * i] for i in range(1, n_cells)]
                 + [[n_rows - 1]])
        slots = []
        for p in parts:
            if len(p) == 1:
                slots += [[(p[0], v)] for v in range(nv)]
            else:
                a, b = p
                slots += [[(a, v)] for v in range(nv) if v != _I_RI]
                slots += [[(b, v)] for v in range(nv) if v != _I_RI]
                slots += [[(a, _I_RI), (b, _I_RI)]]       # one iris, machined not welded
        return slots

    raise ValueError(f"kind must be 'base', 'multicup' or 'dumbbell', got {kind!r}")


def perturbation_nodes(cav, kind, method, half_width, n=None, seed=None,
                       split_ends=None):
    """Node table for a perturbation model, laid out for ``method=['from file', ...]``.

    The solver's multicell path reads a ``(n_nodes, 7 * 2 * n_cells)`` table whose column
    index is ``variable * n_rows + row``. This expands a model's independent draws into
    that layout.

    Parameters
    ----------
    method : list
        ``['Uniform', N]`` / ``['Normal', N]`` for sampling, or
        ``['Quadrature', 'Stroud3']`` / ``['Stroud5']`` for a cubature rule - anything
        :func:`cavsim2d.utils.quadrature.generate_nodes` accepts.
    half_width : float
        Perturbation scale in mm, in whatever sense the rule uses it: the half-width of a
        uniform measure for the cubature rules and ``'Uniform'``, the sigma for
        ``'Normal'``. To make a cubature rule represent a normal of sigma s, pass
        ``s * sqrt(3)``.
    n, seed
        Sample count and seed for the sampling rules; ignored by cubature.
    """
    slots = perturbation_slots(cav, kind, split_ends=split_ends)
    k = len(slots)
    n_rows = 2 * int(cav.n_cells)
    rule = list(method)
    if n is not None and len(rule) > 1:
        rule[1] = n
    nodes, _ = generate_nodes(k, [half_width] * k, rule, **({'seed': seed}
                                                            if seed is not None else {}))
    out = np.zeros((len(nodes), len(HALF_CELL_VARS) * n_rows))
    for i, node in enumerate(np.asarray(nodes)):
        for val, group in zip(node, slots):
            for (r, v) in group:
                out[i, v * n_rows + r] = val
    return out
