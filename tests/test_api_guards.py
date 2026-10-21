"""API-surface guards: removed legacy methods stay removed, and methods that
were only guarded (not ported) raise a clear NotImplementedError."""
import ast
import importlib.metadata
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


def test_sweep_is_study_only():
    """A parameter sweep saves several cavities with their own simulations, so it
    is a Study-level operation. ``Study.sweep`` is the real feature; the old
    per-cavity ``Cavity.sweep`` stub stays removed."""
    assert not hasattr(Cavity, 'sweep'), "Cavity.sweep should stay removed (sweep is Study-only)"
    assert callable(getattr(Study, 'sweep', None)), "Study.sweep should exist"


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


def test_module_headers_import_only_declared_dependencies():
    """A clean `pip install cavsim2d` must import, and quickly.

    IPython and ipywidgets come with the viewer's widget packages now, but they
    stay out of module headers: importing them is slow and only a notebook needs
    them. distutils left the standard library in Python 3.12.
    """
    root = pathlib.Path(__file__).resolve().parent.parent / 'cavsim2d'
    banned = {'IPython', 'ipywidgets', 'distutils'}
    offenders = []
    for f in sorted(root.rglob('*.py')):
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split('.')[0] in banned:
                    offenders.append(f'{f.relative_to(root)}:{node.lineno} {name}')
    assert not offenders, 'unguarded optional/removed imports:\n  ' + '\n  '.join(offenders)


def test_interactive_viewer_dependencies_are_declared():
    """`pip install cavsim2d` must bring what NGSolve's webgui viewer needs.

    ngsolve does not depend on its own widget package, so on a clean install
    `cav.show_fields()` drew nothing and said nothing. NGSolve 6.2.26xx renders
    through anywidget, earlier releases through webgui_jupyter_widgets.
    """
    pyproject = pathlib.Path(__file__).resolve().parent.parent / 'pyproject.toml'
    if pyproject.exists():                       # a source checkout
        text = pyproject.read_text(encoding='utf-8')
        core = text.split('\ndependencies = [', 1)[1].split('\n]', 1)[0]
        declared = core
    else:                                        # the tests run against an installed wheel
        declared = '\n'.join(f'"{r}' for r in importlib.metadata.requires('cavsim2d')
                             if 'extra ==' not in r)
    for package in ('anywidget', 'webgui_jupyter_widgets', 'ipywidgets'):
        assert f'"{package}' in declared, f'{package} is not a core dependency'


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
