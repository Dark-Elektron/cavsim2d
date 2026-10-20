"""Shared pytest fixtures and capability guards for the cavsim2d test suite.

The suite is designed to stay green everywhere:
- Tests needing the eigenmode solver ``pytest.importorskip("ngsolve")`` /
  ``"gmsh"`` at module import, so they skip cleanly where those aren't
  installed (e.g. minimal CI) instead of erroring at collection.
- Wakefield tests are gated on ``requires_abci``: Windows, plus ABCI downloaded
  from KEK and found through ``CAVSIM2D_ABCI`` (or a local copy in the checkout).
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

try:
    from cavsim2d.solvers.ABCI.abci import abci_exe_path  # noqa: E402
except ImportError:  # no NGSolve: the solver tests skip themselves anyway
    abci_exe_path = None


def _abci_available():
    return (abci_exe_path is not None and sys.platform.startswith('win')
            and os.path.isfile(abci_exe_path()))


requires_abci = pytest.mark.skipif(
    not _abci_available(),
    reason="wakefield needs ABCI on Windows: download it from "
           "https://abci.kek.jp/abci.htm and set CAVSIM2D_ABCI")


# A TESLA-like single mid-cell (A, B, a, b, Ri, L, Req) that meshes and solves
# cleanly and quickly — the canonical geometry for fast solver tests.
MIDCELL = [62.22, 66.13, 30.22, 23.11, 80, 93.5, 171.20]


@pytest.fixture
def project_dir(tmp_path):
    """A fresh, empty project directory for one test."""
    return str(tmp_path / "sim")
