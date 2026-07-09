"""Unified, backend-agnostic geometry layer for axisymmetric structures.

A cavity type describes its 2D meridian boundary as a :class:`Profile` — an
ordered list of tagged segments (lines / arcs) — and the shared meshing backend
turns any profile into a boundary-tagged NGSolve mesh. This is what lets a new
axisymmetric structure "just work": implement ``profile()`` and nothing else.
"""
from cavsim2d.geometry.profile import Profile, mesh_from_profile

__all__ = ['Profile', 'mesh_from_profile']
