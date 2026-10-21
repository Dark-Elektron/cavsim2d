"""Folder integrity: every block of results records the geometry it was computed for.

A cavity's folder collects the output of several analyses (eigenmode per
polarisation, wakefield, tuning, multipacting, UQ), each written at a different
time. With nothing recording which geometry produced which block, changing the
cavity, or giving a different cavity the same name, leaves a folder that mixes
results for two geometries with no way to tell them apart.

So every analysis writes ``geometry_stamp.json`` beside its results when it runs:
a fingerprint of the geometry it solved. Comparing the stamps with the cavity's
current fingerprint finds the results that no longer belong to it. Three things
are built on that:

- :func:`guard_runs` asks once before a run replaces anything, and refuses to
  leave a folder holding results for two geometries;
- :func:`warn_if_stale` warns when results computed for another geometry are read;
- :func:`current_tuned` takes a tuned version as the reference only while it was
  tuned from the geometry its parent has now;
- :func:`is_cavity_workspace` / :func:`project_contents_ok` stop ``overwrite=True``
  from deleting a folder cavsim2d did not create.
"""
import datetime
import hashlib
import json
import shutil
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np

from cavsim2d.utils.printing import CavsimWarning

#: Written beside every block of results.
STAMP_NAME = 'geometry_stamp.json'

#: One self-contained block of results. Deleting ``path`` removes all of it, and
#: ``stamp_dir`` holds the stamp naming the geometry it was computed for.
ResultUnit = namedtuple('ResultUnit', 'analysis label path stamp_dir')

#: Result blocks that sit directly in a cavity folder, as
#: ``(analysis, folder name, stamp folder relative to the cavity folder)``.
#: The per-polarisation ``eigenmode/<pol>/`` blocks are listed separately.
_TOP_LEVEL_UNITS = (
    ('eigenmode', 'uq', 'uq'),
    ('wakefield', 'wakefield', 'wakefield'),
    ('tune', 'tuned', 'tuned/tune_info'),
    ('multipacting', 'multipacting', 'multipacting'),
)

#: Folders a cavity workspace may hold besides its results.
_WORKSPACE_EXTRAS = {'geometry', 'eigenmode', 'PostProcessingData', 'PostprocessingData'}

#: Entries a study project folder may hold besides its cavity folders.
_PROJECT_EXTRAS = {'optimisation', '_optimisation', '_tune_temp', 'PostProcessingData',
                   'PostprocessingData', '.ipynb_checkpoints'}

_warned = set()
_noticed = set()


def _plain(value):
    """*value* as plain JSON data with floats rounded to 10 significant digits,
    so two equal geometries always give the same fingerprint."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(f'{float(value):.10g}')
    return value


def geometry_fingerprint(cav):
    """What fixes *cav*'s solved domain, as plain JSON data: the model, its
    parameters, cell count, beam pipes, chaining and dielectric regions."""
    signature = getattr(cav, '_dielectric_signature', None)
    fingerprint = {
        'model': type(cav).__name__,
        'n_cells': getattr(cav, 'n_cells', None),
        'beampipe': getattr(cav, 'beampipe', None),
        'beampipe_length': getattr(cav, 'beampipe_length', None),
        'cell_parameterisation': getattr(cav, 'cell_parameterisation', None),
        'chain': getattr(cav, 'chain', 1),
        'spacing': getattr(cav, 'spacing', None),
        'parameters': dict(getattr(cav, 'parameters', None) or {}),
        'dielectrics': signature() if callable(signature) else {},
    }
    half_cells = getattr(cav, '_half_cells', None)
    if half_cells is not None:
        fingerprint['half_cells'] = half_cells
    return _plain(fingerprint)


def fingerprint_hash(fingerprint):
    """A short, stable hash of a :func:`geometry_fingerprint`."""
    blob = json.dumps(fingerprint, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode('utf-8')).hexdigest()[:16]


def write_stamp(folder, cav):
    """Record in *folder* the geometry *cav* has now.

    Call it after an analysis has written its results to *folder*. Like the
    geometry snapshot, the stamp is advisory: a failed write never fails the run.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return
    fingerprint = geometry_fingerprint(cav)
    stamp = {
        'hash': fingerprint_hash(fingerprint),
        'cavity': getattr(cav, 'name', None),
        'model': type(cav).__name__,
        'module': type(cav).__module__,
        'written': datetime.datetime.now().isoformat(timespec='seconds'),
        'geometry': fingerprint,
    }
    try:
        with open(folder / STAMP_NAME, 'w') as f:
            json.dump(stamp, f, indent=2, default=str)
    except OSError:
        pass


def read_stamp(folder):
    """The stamp in *folder*, or None when there is none (or it is unreadable)."""
    try:
        with open(Path(folder) / STAMP_NAME) as f:
            stamp = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return stamp if isinstance(stamp, dict) and 'hash' in stamp else None


def stamp_matches(folder, cav):
    """True if the stamp in *folder* names *cav*'s current geometry, False if it
    names another one, and None when *folder* has no stamp."""
    stamp = read_stamp(folder)
    if stamp is None:
        return None
    return stamp['hash'] == fingerprint_hash(geometry_fingerprint(cav))


def result_units(cav):
    """Every block of results in *cav*'s folder (empty when it has no folder)."""
    if not getattr(cav, 'self_dir', None):
        return []
    root = Path(cav.self_dir)
    units = []
    eigenmode = root / 'eigenmode'
    if eigenmode.is_dir():
        for sub in sorted(eigenmode.iterdir()):
            if sub.is_dir() and any(sub.iterdir()):
                units.append(ResultUnit('eigenmode', f'eigenmode/{sub.name}', sub, sub))
    for analysis, name, stamp_rel in _TOP_LEVEL_UNITS:
        path = root / name
        if path.is_dir() and any(path.iterdir()):
            units.append(ResultUnit(analysis, name, path, root / stamp_rel))
    return units


def stale_units(cav, units=None):
    """``[(unit, stamp)]`` for every block computed for a different geometry.

    A block without a stamp (written before stamps existed) is never reported:
    nothing records which geometry it belongs to.
    """
    if units is None:
        units = result_units(cav)
    if not units:
        return []
    current = fingerprint_hash(geometry_fingerprint(cav))
    stale = []
    for unit in units:
        stamp = read_stamp(unit.stamp_dir)
        if stamp is not None and stamp['hash'] != current:
            stale.append((unit, stamp))
    return stale


def _fmt(value):
    if value is None:
        return 'unset'
    if isinstance(value, float):
        return f'{value:g}'
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def describe_change(old, new, limit=4):
    """What differs between two fingerprints, e.g. ``Req_m 103.3 -> 104``."""
    old, new = old or {}, new or {}
    changes = []
    for key in ('model', 'n_cells', 'beampipe', 'beampipe_length',
                'cell_parameterisation', 'chain', 'spacing'):
        if old.get(key) != new.get(key):
            changes.append(f'{key} {_fmt(old.get(key))} -> {_fmt(new.get(key))}')
    for group in ('parameters', 'dielectrics'):
        before, after = old.get(group) or {}, new.get(group) or {}
        for key in sorted(set(before) | set(after)):
            if before.get(key) != after.get(key):
                changes.append(f'{key} {_fmt(before.get(key))} -> {_fmt(after.get(key))}')
    if old.get('half_cells') != new.get('half_cells'):
        changes.append('half-cell dimensions')
    if not changes:
        return 'the geometry changed'
    text = ', '.join(changes[:limit])
    if len(changes) > limit:
        text += f' (and {len(changes) - limit} more)'
    return text


def warn_if_stale(cav, stamp_dir, what):
    """Warn, once per session, when the results in *stamp_dir* were computed for
    a geometry other than *cav*'s current one. Returns True if they were."""
    if stamp_dir is None:
        return False
    stamp = read_stamp(stamp_dir)
    if stamp is None:
        return False
    fingerprint = geometry_fingerprint(cav)
    current = fingerprint_hash(fingerprint)
    if stamp['hash'] == current:
        return False
    key = (str(stamp_dir), stamp['hash'], current)
    if key not in _warned:
        _warned.add(key)
        warnings.warn(
            f"{what} in {stamp_dir} were computed for a different geometry than "
            f"cavity {getattr(cav, 'name', '')!r} has now "
            f"({describe_change(stamp.get('geometry'), fingerprint)}). They do not "
            f"describe this cavity; rerun the analysis before using them.",
            CavsimWarning, stacklevel=3)
    return True


def current_tuned(cav):
    """*cav*'s tuned version if it was tuned from the geometry *cav* has now, else None.

    A tuned folder left from before the cavity changed is not this cavity's tuned
    version: ``cav.tuned`` still returns it, with a warning, but it is never taken
    as the reference. A tuned folder older than the stamps counts as current.
    """
    folder = getattr(cav, 'self_dir', None)
    if not folder:
        return None
    tune_info = Path(folder) / 'tuned' / 'tune_info'
    if not (tune_info / 'tune_res.json').is_file() or stamp_matches(tune_info, cav) is False:
        return None
    return getattr(cav, 'tuned', None)


def notice_tuned_versions(cavs, analysis, study=False, stacklevel=3):
    """Say, once per session and cavity, that a run on a cavity with a tuned version
    solves the untuned geometry and saves into the untuned cavity's folder."""
    tuned = []
    for cav in cavs:
        version = current_tuned(cav)
        if version is None:
            continue
        key = (str(cav.self_dir), analysis, fingerprint_hash(geometry_fingerprint(version)))
        if key not in _noticed:
            _noticed.add(key)
            tuned.append((cav, version))
    if not tuned:
        return
    if study:
        names = ', '.join(repr(cav.name) for cav, _ in tuned)
        has = 'has a tuned version' if len(tuned) == 1 else 'have tuned versions'
        message = (f"{names} {has}. This {analysis} run solves the untuned geometries "
                   f"and saves the results in their own folders; run "
                   f"study.tuned.run_{analysis}(...) to solve the tuned ones.")
    else:
        cav, version = tuned[0]
        change = describe_change(geometry_fingerprint(cav), geometry_fingerprint(version))
        message = (f"Cavity {cav.name!r} has a tuned version ({change}) in "
                   f"{version.self_dir}. This {analysis} run solves the untuned geometry "
                   f"and saves its results in {cav.self_dir}; run "
                   f"cav.tuned.{analysis}.run(...) to solve the tuned one.")
    warnings.warn(message, CavsimWarning, stacklevel=stacklevel)


def _belongs_to(unit, analysis, pol_labels):
    """Does a run of *analysis* (over *pol_labels*, for eigenmode) write *unit*?"""
    if unit.analysis != analysis:
        return False
    if analysis != 'eigenmode' or pol_labels is None:
        return True
    if unit.label == 'uq':
        # A monopole rerun removes the eigenmode UQ, which is computed from it.
        return 'monopole' in pol_labels
    return unit.label.split('/', 1)[-1] in pol_labels


def planned_changes(cav, analysis, rerun=True, pol_labels=None, replaces=None):
    """What a run of *analysis* on *cav* would change in its folder.

    Returns ``(replaced, stale)``. *replaced* lists the blocks this run
    overwrites: with ``rerun`` every existing block it writes, otherwise only
    those computed for another geometry (they are recomputed). *stale* lists the
    blocks it would leave behind computed for another geometry. Each entry is
    ``(unit, stamp)``, the stamp None when the block matches the geometry.

    *pol_labels* names the eigenmode polarisation folders the run writes
    (``{'monopole', 'dipole'}``); *replaces* overrides the decision for one
    block, e.g. multipacting only overwrites a finished sweep.
    """
    units = result_units(cav)
    stale_by_path = {unit.path: stamp for unit, stamp in stale_units(cav, units)}
    replaced, stale = [], []
    for unit in units:
        stamp = stale_by_path.get(unit.path)
        if _belongs_to(unit, analysis, pol_labels):
            overwrite = rerun if replaces is None else replaces(unit)
            if overwrite or stamp is not None:
                replaced.append((unit, stamp))
                continue
        if stamp is not None:
            stale.append((unit, stamp))
    return replaced, stale


def _describe_plans(plans, title):
    lines = []
    shown = plans[:10]
    for cav, replaced, stale in shown:
        current = geometry_fingerprint(cav)
        lines.append(f"{title} for {cav.name!r} ({cav.self_dir}):")
        for unit, stamp in replaced:
            note = ''
            if stamp is not None:
                note = (f"  [computed for a different geometry: "
                        f"{describe_change(stamp.get('geometry'), current)}]")
            lines.append(f"  overwrites {unit.label}{note}")
        for unit, stamp in stale:
            lines.append(f"  deletes {unit.label}  [computed for a different geometry: "
                         f"{describe_change(stamp.get('geometry'), current)}]")
    if len(plans) > len(shown):
        lines.append(f"... and {len(plans) - len(shown)} more cavities.")
    if any(stale for _, _, stale in plans):
        lines.append(
            "Results marked 'deletes' were computed for a different geometry: the "
            "cavity changed since they were run, or another cavity was given the same "
            "name. Keeping them would leave one folder holding results for two "
            "geometries. To keep them, answer N and give this cavity a new name "
            "(cav.set_name(...)) or a new folder.")
    return '\n'.join(lines)


def _delete(units):
    # No ignore_errors: a block that cannot be removed (a file held open on
    # Windows) must stop the run, not leave two geometries side by side.
    for unit in units:
        if unit.path.exists():
            shutil.rmtree(unit.path)


def guard_runs(cavs, analysis, config, pol_labels=None, title=None, replaces=None):
    """Check the folders of *cavs* before *analysis* runs on them; True to go ahead.

    Nothing is asked when the run overwrites nothing and no folder holds results
    computed for another geometry. Otherwise one question covers every cavity:
    on yes, results computed for another geometry are deleted (this run's own
    are overwritten by the run itself); on no, nothing is run or deleted.

    ``config['force']`` answers yes without asking. When nobody can be asked (a
    script, ``nbconvert`` or a test), a run that only overwrites its own results
    goes ahead, as it always has, but one that would leave results for two
    geometries in a folder raises instead, so nothing is deleted unseen.
    """
    rerun = config.get('rerun', True)
    title = title or analysis.capitalize()
    plans = []
    for cav in cavs:
        replaced, stale = planned_changes(cav, analysis, rerun=rerun,
                                          pol_labels=pol_labels, replaces=replaces)
        if replaced or stale:
            plans.append((cav, replaced, stale))
    if not plans:
        return True
    to_delete = [unit for _, _, stale in plans for unit, _ in stale]
    message = _describe_plans(plans, title)
    if config.get('force'):
        if to_delete:
            warnings.warn(f"force=True: deleting results computed for a different "
                          f"geometry:\n{message}", CavsimWarning, stacklevel=3)
        _delete(to_delete)
        return True
    try:
        answer = input(f"{message}\nContinue? [y/N] ")
    except (EOFError, OSError, RuntimeError):
        # Nobody to ask: input() raises EOFError on a closed stdin and
        # StdinNotImplementedError (a RuntimeError) under nbconvert.
        if to_delete:
            raise RuntimeError(
                f"{message}\nThis session cannot ask for confirmation, so nothing was "
                f"run or deleted. Pass force=True to delete those results and run, or "
                f"give the cavity a new name.") from None
        return True
    if answer.strip().lower() in ('y', 'yes'):
        _delete(to_delete)
        return True
    print(f"{title}: nothing was run; the existing results are kept.")
    return False


def confirm(question, force=False):
    """Ask *question* (a yes/no prompt); True to go ahead.

    ``force`` answers yes. When nobody can be asked the answer is also yes, the
    policy :func:`guard_runs` applies to a run that only replaces its own results.
    """
    if force:
        return True
    try:
        answer = input(f"{question} [y/N] ")
    except (EOFError, OSError, RuntimeError):
        return True
    return answer.strip().lower() in ('y', 'yes')


def is_cavity_workspace(path):
    """True if *path* is empty or looks like a folder cavsim2d made for one
    cavity: it has a ``geometry/`` folder, or only result folders."""
    path = Path(path)
    if not path.is_dir():
        return False
    entries = list(path.iterdir())
    if not entries:
        return True
    if (path / 'geometry').is_dir():
        return True
    known = _WORKSPACE_EXTRAS | {name for _, name, _ in _TOP_LEVEL_UNITS}
    return all(e.is_dir() and e.name in known for e in entries)


def project_contents_ok(path):
    """``(ok, unexpected)``: *ok* when every entry of the study folder *path* is a
    cavity workspace, a sweep of them, or a folder cavsim2d writes there."""
    unexpected = []
    for entry in Path(path).iterdir():
        if entry.name in _PROJECT_EXTRAS and entry.is_dir():
            continue
        if entry.is_dir() and (is_cavity_workspace(entry)
                               or (entry.name.startswith('sweep')
                                   and all(is_cavity_workspace(e) for e in entry.iterdir()
                                           if e.is_dir()))):
            continue
        unexpected.append(entry.name)
    return not unexpected, unexpected
