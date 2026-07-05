"""Shared pytest fixtures and capability guards for the cavsim2d test suite.

The suite is designed to stay green everywhere:
- Tests needing the eigenmode solver ``pytest.importorskip("ngsolve")`` /
  ``"gmsh"`` at module import, so they skip cleanly where those aren't
  installed (e.g. minimal CI) instead of erroring at collection.
- Wakefield tests are gated on ``requires_abci`` (Windows + bundled ABCI.exe).
All tests write into pytest's ``tmp_path`` — never a personal directory.
"""
import os
import sys

import pytest

# Make the in-repo package importable without an editable install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Headless plotting for any test that touches matplotlib.
import matplotlib  # noqa: E402
matplotlib.use('Agg')


def _abci_available():
    exe = os.path.join(_REPO_ROOT, 'cavsim2d', 'solvers', 'ABCI', 'ABCI.exe')
    return sys.platform.startswith('win') and os.path.exists(exe)


requires_abci = pytest.mark.skipif(
    not _abci_available(),
    reason="wakefield needs the bundled ABCI.exe on Windows")


# A TESLA-like single mid-cell (A, B, a, b, Ri, L, Req) that meshes and solves
# cleanly and quickly — the canonical geometry for fast solver tests.
MIDCELL = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20]


@pytest.fixture
def project_dir(tmp_path):
    """A fresh, empty project directory for one test."""
    return str(tmp_path / "sim")
