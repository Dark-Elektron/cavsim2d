"""Temporary folders for throwaway solves.

Family scans, the QoI Newton and the staged tune solve dozens of short-lived
cavities, each in its own folder with a mesh and fields. They used to go to
``tempfile.mkdtemp()`` and stay in the system temp directory for good.
"""
import tempfile
from contextlib import contextmanager


@contextmanager
def scratch_folders(prefix='cavsim2d_'):
    """Yield a function that makes a new empty folder; every folder it made, and
    everything solved in them, is removed when the block ends."""
    with tempfile.TemporaryDirectory(prefix=prefix, ignore_cleanup_errors=True) as root:
        yield lambda: tempfile.mkdtemp(dir=root)
