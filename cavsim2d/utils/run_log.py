"""Per-analysis run-timing logs.

Every analysis — eigenmode, wakefield, multipacting, tune — writes a single
``run_log.json`` into its own results folder recording when it ran, a short
config summary, per-step timings, and the total wall-clock time. A long UQ or
multi-cell run is then observable after the fact, not only in live stdout.
"""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime

# Config keys that are verbose or not worth echoing into the log.
_SKIP_KEYS = ('target', 'uq_config', 'operating_points')


def _summarise(config):
    """Compact summary of a config dict (drops verbose/callable entries)."""
    if not isinstance(config, dict):
        return config
    return {k: v for k, v in config.items() if k not in _SKIP_KEYS}


class RunTimer:
    """Accumulate per-step timings for one analysis and write them to a single
    ``<save_dir>/run_log.json``. Use as a context manager::

        with RunTimer(cav.eigenmode_dir, 'eigenmode', name=cav.name, config=cfg) as t:
            with t.step('solve'):
                ...

    The total is written on exit (even if the body raised)."""

    def __init__(self, save_dir, analysis, name=None, config=None):
        self.save_dir = str(save_dir)
        self.analysis = analysis
        self.name = name
        self.config = config
        self.steps = []
        self._t0 = time.time()
        self._start_iso = datetime.now().isoformat(timespec='seconds')

    @contextmanager
    def step(self, label):
        t = time.time()
        try:
            yield
        finally:
            self.steps.append((label, time.time() - t))

    def add(self, label, seconds):
        """Record a step whose duration was timed elsewhere."""
        self.steps.append((label, float(seconds)))

    def write(self):
        total = time.time() - self._t0
        os.makedirs(self.save_dir, exist_ok=True)
        record = {'analysis': self.analysis, 'name': self.name,
                  'started': self._start_iso,
                  'config': _summarise(self.config) if self.config is not None else None,
                  'steps': {label: round(sec, 2) for label, sec in self.steps},
                  'total_s': round(total, 2)}
        try:
            with open(os.path.join(self.save_dir, 'run_log.json'), 'w') as f:
                json.dump(record, f, indent=2, default=str)
        except Exception:
            pass                              # logging must never break a run
        return total

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.write()
        return False
