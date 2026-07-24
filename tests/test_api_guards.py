"""API-surface guards: removed legacy methods stay removed, and methods that
were only guarded (not ported) raise a clear NotImplementedError."""
import ast
import pathlib

import pytest

pytest.importorskip("ngsolve")
pytest.importorskip("gmsh")

from conftest import MIDCELL
from cavsim2d import Study, EllipticalCavity
from cavsim2d.models.base import Cavity


@pytest.mark.parametrize("name", ["run_abci", "run_multipacting", "check_uq_config"])
def test_dead_cavities_wrappers_removed(name):
    assert not hasattr(Study, name), f"{name} should have been deleted"


def test_sweep_raises_clear_error(project_dir):
    cavs = Study(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    cavs.add_cavity([cav], ['S'])
    with pytest.raises(NotImplementedError):
        cav.sweep({'A': [40, 45, 3]})


def test_imports_live_at_module_top():
    """Imports belong in the module header.

    Only two exemptions are allowed, and each must be preceded by a `# Deferred:`
    comment saying why: an optional heavy dependency (netgen / ngsolve / IPython)
    or a genuine import cycle.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / 'cavsim2d'
    offenders = []
    for f in sorted(root.rglob('*.py')):
        if 'cavity_legacy' in f.name:
            continue
        src = f.read_text(encoding='utf-8')
        lines = src.split('\n')
        tree = ast.parse(src)
        top = {id(n) for n in tree.body}
        # (a) imports nested inside a function/class body. Scan upward past comment,
        # blank and sibling-import lines, so one comment can justify a grouped block.
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top:
                i, justified = node.lineno - 2, False
                while i >= 0:
                    prev = lines[i].strip()
                    if prev.startswith('# Deferred:'):
                        justified = True
                        break
                    # step over comments, blanks, sibling imports, and the `try:`
                    # of an optional-dependency guard
                    if (not prev or prev == 'try:' or prev.startswith(')')
                            or prev.startswith(('#', 'from ', 'import '))):
                        i -= 1
                        continue
                    break
                if not justified:
                    offenders.append(f'{f.name}:{node.lineno} nested, no "# Deferred:" rationale')
        # (b) top-level imports stranded below the first def/class
        firstdef = next((i for i, n in enumerate(tree.body)
                         if isinstance(n, (ast.FunctionDef, ast.ClassDef))), None)
        if firstdef is not None:
            for i, n in enumerate(tree.body):
                if i > firstdef and isinstance(n, (ast.Import, ast.ImportFrom)):
                    offenders.append(f'{f.name}:{n.lineno} top-level import below code')
    assert not offenders, 'imports not at top of file:\n  ' + '\n  '.join(offenders)


def test_no_asserts_in_the_package():
    """User-input validation must raise, not assert.

    `assert cond, error(msg)` produced `AssertionError: None` (because error()
    prints and returns None), and `python -O` strips asserts entirely — so the
    check silently vanished in optimised runs. All of them are now `require()`,
    which raises a ValueError carrying the message.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / 'cavsim2d'
    offenders = []
    for f in sorted(root.rglob('*.py')):
        if 'cavity_legacy' in f.name:
            continue
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f'{f.name}:{node.lineno}')
    assert not offenders, 'assert used for validation:\n  ' + '\n  '.join(offenders)


def test_require_raises_with_a_message_and_survives_O():
    from cavsim2d.utils.config_validation import require
    require(True, 'no-op')
    with pytest.raises(ValueError, match='number of names'):
        require(False, 'Number of cavities does not correspond to number of names.')


def test_config_errors_reach_the_user(project_dir):
    """A user mistake raises ValueError with the real message, not AssertionError: None."""
    from cavsim2d import Study, EllipticalCavity
    cavs = Study(project_dir)
    cav = EllipticalCavity(1, MIDCELL, MIDCELL, MIDCELL, beampipe='both')
    with pytest.raises(ValueError, match='number of names'):
        cavs.add_cavity([cav], ['a', 'b'])
