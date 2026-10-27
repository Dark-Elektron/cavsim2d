"""Strip local file paths from the outputs of executed example notebooks.

A warning prints the file it was raised from, so an executed notebook carries
the absolute paths of the machine that ran it (``C:\\Users\\<name>\\...``).
Run this after re-executing a notebook and before committing it::

    python docs/scrub_notebook_paths.py docs/source/examples/eigenmode/pillbox.ipynb

With no arguments it scrubs every notebook under ``docs/source/examples/``.
Only text outputs change; cell sources and images are left alone.
"""
import re
import sys
from pathlib import Path

import nbformat

DOCS = Path(__file__).resolve().parent
REPO = DOCS.parent
EXAMPLES = DOCS / 'source' / 'examples'

_SEP = r'[\\/]'
# Most specific first, so what is left of a path stays readable:
# 'cavsim2d\solvers\...', 'scipy\optimize\...', 'ipykernel_123\456.py'.
_PREFIXES = [
    (re.compile(re.escape(str(REPO)) + _SEP, re.IGNORECASE), ''),
    (re.compile(r'(?:[A-Za-z]:)?' + _SEP + r'(?:[^\\/\s:]+' + _SEP + r')*?site-packages' + _SEP), ''),
    (re.compile(r'[A-Za-z]:' + _SEP + r'Users' + _SEP + r'[^\\/\s]+' + _SEP
                + r'AppData' + _SEP + r'Local' + _SEP + r'Temp' + _SEP, re.IGNORECASE), ''),
    (re.compile(r'[A-Za-z]:' + _SEP + r'Users' + _SEP + r'[^\\/\s]+', re.IGNORECASE), '~'),
    (re.compile(r'/(?:home|Users)/[^/\s]+'), '~'),
]
_TEXT_MIME = ('text/plain', 'text/html', 'text/markdown')


def scrub_text(text):
    """Return ``text`` with every local path prefix removed."""
    for pattern, replacement in _PREFIXES:
        text = pattern.sub(replacement, text)
    return text


def scrub_notebook(path):
    """Scrub one notebook in place. Returns True if anything changed."""
    nb = nbformat.read(path, as_version=4)
    changed = False

    def clean(value):
        nonlocal changed
        if isinstance(value, list):
            return [clean(v) for v in value]
        new = scrub_text(value)
        changed |= new != value
        return new

    for cell in nb.cells:
        for out in cell.get('outputs', []):
            if 'text' in out:
                out['text'] = clean(out['text'])
            if 'traceback' in out:
                out['traceback'] = clean(out['traceback'])
            if 'evalue' in out:
                out['evalue'] = clean(out['evalue'])
            for mime in _TEXT_MIME:
                if mime in out.get('data', {}):
                    out['data'][mime] = clean(out['data'][mime])
    if changed:
        nbformat.write(nb, path)
    return changed


def main(paths):
    paths = [Path(p) for p in paths] or sorted(EXAMPLES.rglob('*.ipynb'))
    for path in paths:
        if scrub_notebook(path):
            print('scrubbed', path)


if __name__ == '__main__':
    main(sys.argv[1:])
