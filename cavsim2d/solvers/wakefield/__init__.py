"""Solver-agnostic wakefield backends.

``cav.wakefield`` and the wakefield objectives read every result through a
:class:`WakefieldBackend`, selected by ``wakefield_config['solver']`` (default
``'abci'``). Register a new backend here to swap ABCI for another solver — e.g.
the planned in-house NGSolve time-domain solver — without touching callers.
"""
from cavsim2d.solvers.wakefield.base import (WakefieldBackend, WakefieldResult,
                                             POL_QOI_KEYS)
from cavsim2d.solvers.wakefield.abci import ABCIWakefield

BACKENDS = {}


def register_backend(backend):
    """Register a :class:`WakefieldBackend` instance under its ``name``."""
    BACKENDS[backend.name] = backend
    return backend


def get_backend(name='abci'):
    """The wakefield backend registered under *name* (default ``'abci'``)."""
    key = (name or 'abci').lower()
    if key not in BACKENDS:
        raise ValueError(f"Unknown wakefield solver {name!r}. Registered: "
                         f"{sorted(BACKENDS)}.")
    return BACKENDS[key]


register_backend(ABCIWakefield())

__all__ = ['WakefieldBackend', 'WakefieldResult', 'POL_QOI_KEYS',
           'ABCIWakefield', 'BACKENDS', 'register_backend', 'get_backend']
