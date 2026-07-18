"""Multipacting analysis for 2D-axisymmetric RF cavities.

Relativistic electron tracking in the cavity's eigenmode field with secondary
emission, following the PyMultipact method (the user's validated physics). The
EM field is read from a cavsim2d eigenmode result and the whole analysis is
driven through the ``cav.multipacting`` solver object; this package holds the
pieces:

- :mod:`.particles` / :mod:`.integrators` -- particle state + RK4 Lorentz push,
- :mod:`.sey` -- the secondary-emission-yield table,
- :mod:`.fields` -- the eigenmode-field adapter (in-plane E + reconstructed H),
- :mod:`.driver` -- the peak-field sweep,
- :mod:`.metrics` -- counter / final-energy / distance-function metrics.

Only the ngsolve-free pieces (``SEY``) are re-exported here; ``driver``/``fields``
pull in ngsolve, so import them from their submodules where the field is needed —
keeping ``import cavsim2d`` soft on the ngsolve dependency.
"""
from cavsim2d.analysis.multipacting.sey import SEY, DEFAULT_SEY

__all__ = ['SEY', 'DEFAULT_SEY']
