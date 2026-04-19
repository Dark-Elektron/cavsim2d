
from contextlib import contextmanager
from termcolor import colored

# Substrings that should silence matching error() calls. Mutated by
# ``suppress_errors()`` — shared across all modules because they all
# call the same function object (even under ``from ... import *``).
_suppress_substrings = []


def error(*arg):
    msg = str(arg[0]) if arg else ''
    for s in _suppress_substrings:
        if s in msg:
            return
    print(f'\x1b[31mERROR:: {msg}\x1b[0m')


@contextmanager
def suppress_errors(*substrings):
    """Silence any ``error(...)`` whose message contains any of ``substrings``.

    Typical use: suppress expected noise from inner-loop probing (e.g.
    the secant tuner's transient degenerate-geometry hits) without
    silencing unrelated errors.
    """
    _suppress_substrings.extend(substrings)
    try:
        yield
    finally:
        for s in substrings:
            try:
                _suppress_substrings.remove(s)
            except ValueError:
                pass


def warning(*arg):
    print(f'\x1b[33mWARNING:: {arg[0]}\x1b[0m')


def running(*arg):
    print(f'\x1b[36m{arg[0]}\x1b[0m')


def info(*arg):
    print(f'\x1b[34mINFO:: {arg[0]}\x1b[0m')


def done(*arg):
    print(f'\x1b[32mDONE:: {arg[0]}\x1b[0m')