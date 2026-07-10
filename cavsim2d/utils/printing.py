
import sys
from contextlib import contextmanager
from termcolor import colored

# Substrings that should silence matching error() calls. Mutated by
# ``suppress_errors()`` — shared across all modules because they all
# call the same function object (even under ``from ... import *``).
_suppress_substrings = []

# Global chattiness switch. When False (the default for all analyses),
# info / running / done / warning calls are silenced — leaving only
# errors and external progress bars (tqdm) visible. Call ``set_verbose(True)``
# or use the ``verbose()`` context manager to re-enable output.
_verbose = False


def set_verbose(flag):
    """Enable or disable chatty info/running/done/warning output globally."""
    global _verbose
    _verbose = bool(flag)


def is_verbose():
    return _verbose


@contextmanager
def verbose(flag=True):
    """Temporarily toggle verbose output within a ``with`` block."""
    global _verbose
    prev = _verbose
    _verbose = bool(flag)
    try:
        yield
    finally:
        _verbose = prev


def _printable(msg):
    """Downgrade characters the console cannot encode, instead of crashing.

    The Windows console is cp1252 by default, which cannot encode e.g. U+2248
    ('almost equal') or U+2192 ('right arrow'). Printing one raised
    ``UnicodeEncodeError`` from inside an *error message*, so a tuning failure
    crashed the run instead of reporting itself.
    """
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    try:
        msg.encode(encoding)
        return msg
    except (UnicodeEncodeError, LookupError):
        return msg.encode(encoding, errors='replace').decode(encoding, errors='replace')


def error(*arg):
    if not arg:
        return
    msg = ' '.join(str(a) for a in arg)
    for s in _suppress_substrings:
        if s in msg:
            return
    print(f'\x1b[31mERROR:: {_printable(msg)}\x1b[0m')


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
    if not _verbose:
        return
    print(f'\x1b[33mWARNING:: {_printable(str(arg[0]))}\x1b[0m')


def running(*arg):
    if not _verbose:
        return
    print(f'\x1b[36m{_printable(str(arg[0]))}\x1b[0m')


def info(*arg):
    if not _verbose:
        return
    print(f'\x1b[34mINFO:: {_printable(str(arg[0]))}\x1b[0m')


def done(*arg):
    if not _verbose:
        return
    print(f'\x1b[32mDONE:: {_printable(str(arg[0]))}\x1b[0m')
