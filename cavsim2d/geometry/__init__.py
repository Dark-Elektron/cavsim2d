"""Geometry for axisymmetric structures — the single home for cavity shapes.

A cavity type describes its 2D meridian boundary as a :class:`Profile` — an
ordered list of tagged segments (lines / arcs / exact ellipse arcs) — and the
shared meshing backend turns any profile into a boundary-tagged NGSolve mesh.
This is what lets a new axisymmetric structure "just work": implement
``profile()`` and nothing else.

The package is layered:

``profile``
    The :class:`Profile` blueprint. Meshes natively via ``netgen.occ``; a cavity
    that exposes ``profile()`` needs no ``.geo`` file.
``tangency``
    Alpha-angle / tangent-point solvers shared by every elliptical contour.
``primitives``
    Low-level sampling of lines and elliptic arcs into point lists.
``plotting``
    Matplotlib renderers for a contour (no gmsh required).
``writers``
    Exporters to external formats: gmsh ``.geo``, ABCI, Multipac, CST.

``writers`` is deliberately not re-exported here — import its submodules
directly (see ``writers/__init__.py`` for the reason).
"""
from cavsim2d.geometry.profile import Profile, mesh_from_profile
from cavsim2d.geometry.tangency import (update_alpha, calculate_alpha, tangent_coords,
                                        ellipse_tangent, jac)
from cavsim2d.geometry.primitives import linspace, lineTo, arcTo, shortest_direction

__all__ = ['Profile', 'mesh_from_profile',
           'update_alpha', 'calculate_alpha', 'tangent_coords', 'ellipse_tangent', 'jac',
           'linspace', 'lineTo', 'arcTo', 'shortest_direction']
