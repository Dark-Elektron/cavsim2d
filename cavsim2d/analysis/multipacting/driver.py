"""Multipacting field-level sweep driver.

Ported from PyMultipact's ``Domain.analyse_multipacting`` / ``_analyse_multipacting``,
reworked so the EM field comes from a cavsim2d eigenmode result folder (the
adapter in :mod:`.fields`) and everything is written under ``<cav>/multipacting/``.

For each peak-field value in the sweep, an electron cloud is launched from the
wall over a grid of RF launch phases and tracked with the RK4 integrator; the
counter function ``c20/c0`` is the fraction that survives to 20 impacts. The
sweep is embarrassingly parallel over field levels (one worker per chunk).
"""
import itertools
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from ngsolve import Norm
from tqdm.auto import tqdm

from cavsim2d.analysis.multipacting.fields import build_emfield, load_eigenmode_fields
from cavsim2d.analysis.multipacting.integrators import Integrators
from cavsim2d.analysis.multipacting.metrics import distance_function
from cavsim2d.analysis.multipacting.particles import Particles

q0 = 1.60217663e-19
m0 = 9.1093837e-31
c0 = 299792458


def _surface_points(mesh, bc='PEC'):
    """Sorted wall (PEC) vertices of the eigenmode mesh -- the emission sites and
    the collision polyline. cavsim2d names the cavity wall 'PEC' (PyMultipact
    used 'default')."""
    pec = mesh.Boundaries(bc)
    bel = [el.vertices for el in pec.Elements()]
    bel_unique = list(set(itertools.chain(*bel)))
    return np.array(sorted([mesh.vertices[v.nr].point for v in bel_unique]))


def _bounding_rect(xsurf):
    z, r = xsurf[:, 0], xsurf[:, 1]
    return [float(z.min()), float(z.max()), float(r.min()), float(r.max())]


def default_xrange(xsurf):
    """Default emission band: a few wall points around the equator (the
    max-radius point), where elliptical-cavity multipacting concentrates. Spans
    the nearest wall points so the band always contains emission sites (an
    empty band makes the ported Particles call exit())."""
    z = xsurf[:, 0]
    z_eq = float(z[np.argmax(xsurf[:, 1])])
    near = np.sort(np.abs(z - z_eq))[:min(7, len(z))]
    half = max(float(near.max()), 0.5e-3) + 1e-9
    return [z_eq - half, z_eq + half]


def _peak_surface_field(em, mesh, xsurf):
    """Peak in-plane |E| over the interior wall points (drives the Epk
    normalisation)."""
    zmin, zmax = xsurf[:, 0].min(), xsurf[:, 0].max()
    interior = xsurf[(xsurf[:, 1] > 0) & (xsurf[:, 0] > zmin) & (xsurf[:, 0] < zmax)]
    esurf = [Norm(em.e)(mesh(float(zi), float(ri))) for (zi, ri) in interior]
    return float(max(esurf))


def _worker(proc_id, folder, fields_dir, freq_mode, mode, xrange, procs_epks,
            phis, v_init, sey, Epk, step, bounding_rect, loss_model, t_max):
    """One process's slice of the Epk sweep. Reloads the mesh + eigenmode field
    from *fields_dir*, builds the multipacting EM field, and tracks the cloud for
    each assigned peak-field value; writes ``mresults_{proc_id}``.

    Any failure is written to ``mresults_{proc_id}.err`` — a spawned child's
    stderr is invisible in notebooks, so the traceback must reach a file the
    parent can read back. BaseException also catches the ``exit()`` the ported
    Particles code calls on an empty emission set (SystemExit)."""
    try:
        _worker_impl(proc_id, folder, fields_dir, freq_mode, mode, xrange,
                     procs_epks, phis, v_init, sey, Epk, step, bounding_rect,
                     loss_model, t_max)
    except BaseException:
        with open(os.path.join(folder, f"mresults_{proc_id}.err"), 'w') as f:
            f.write(traceback.format_exc())
        raise


def _worker_impl(proc_id, folder, fields_dir, freq_mode, mode, xrange, procs_epks,
                 phis, v_init, sey, Epk, step, bounding_rect, loss_model, t_max):
    mesh, gfu_E, _ = load_eigenmode_fields(fields_dir)
    em = build_emfield(gfu_E, mode, freq_mode)
    xsurf = _surface_points(mesh)

    dt = 1 / (freq_mode * 1e6 * 20 * 6)
    w = 2 * np.pi * freq_mode * 1e6
    integrator = Integrators(mesh, w, bounding_rect=bounding_rect, loss_model=loss_model)

    n_init_particles = 1
    particles_objects = []
    particles_left = []
    start = time.time()
    for epk in procs_epks:
        sub_start = time.time()
        t = 0.0
        particles = Particles(xrange, v_init, xsurf, phis, cmap='jet', step=step)
        n_init_particles = len(particles.x)

        scale = epk
        while t < t_max:
            if particles.len != 0:
                particles.save_old()
                integrator.rk4(particles, t, dt, em, scale, sey)
                particles.update_record()
            t += dt

        particles_objects.append(particles)
        particles_left.append(len(particles.bright_set))
        print(f"\tEpk: {epk * Epk * 1e-6:.2f} MV/m, bright set: "
              f"{len(particles.bright_set)}, {time.time() - sub_start:.1f}s")
        # Publish progress (field levels finished by this worker) so the parent
        # can aggregate a live bar; a small file the parent polls, robust across
        # the subprocess boundary.
        try:
            with open(os.path.join(folder, f"mprogress_{proc_id}"), 'w') as pf:
                pf.write(str(len(particles_objects)))
        except OSError:
            pass

    cn_c0 = np.array(particles_left) / n_init_particles
    mresult = {'cn/c0': cn_c0, 'particles_objects': particles_objects,
               'n_init_particles': n_init_particles, 'epks': procs_epks,
               'phis_v': phis}
    with open(os.path.join(folder, f"mresults_{proc_id}"), "wb") as file:
        pickle.dump(mresult, file)
    print(f"\tProc {proc_id} done ({time.time() - start:.1f}s).")


def _read_progress(folder, proc_count):
    """Total field levels finished so far, summed over the workers' progress
    files (each holds the count that worker has completed)."""
    done = 0
    for p in range(proc_count):
        try:
            with open(os.path.join(folder, f"mprogress_{p}")) as f:
                done += int(f.read() or 0)
        except (OSError, ValueError):
            pass
    return done


class _SweepProgress:
    """Live tqdm bar over the whole sweep, driven by a daemon thread that polls
    the workers' progress files — so it aggregates across the subprocess workers
    (and the in-process one) without them writing to the same bar."""

    def __init__(self, folder, proc_count, total, enabled=True):
        self.folder, self.proc_count, self.total = folder, proc_count, total
        self.enabled = enabled
        self._stop = threading.Event()
        self._thread = None
        self._bar = None

    def __enter__(self):
        if self.enabled:
            self._bar = tqdm(total=self.total, desc='Multipacting sweep',
                             unit='field')
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def _poll(self):
        while not self._stop.is_set():
            self._bar.n = min(_read_progress(self.folder, self.proc_count), self.total)
            self._bar.refresh()
            self._stop.wait(0.3)

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._bar is not None:
            self._bar.n = self.total
            self._bar.refresh()
            self._bar.close()


def run_sweep(folder, fields_dir, freqs, sey, *, mode=1, xrange=None, epks=None,
              phis=None, v_init=2, step=None, proc_count=None, loss_model='field',
              t_max=1000e-10, progress=True):
    """Run the peak-field sweep and return the assembled results dict.

    Parameters mirror PyMultipact's ``analyse_multipacting``. *fields_dir* is the
    eigenmode polarisation folder (``mesh.pkl`` + ``gfu_EH.pkl``); *freqs* is the
    mode-frequency list [MHz]; *sey* is a :class:`~.sey.SEY`. ``progress`` shows a
    live tqdm bar over the field levels as the workers finish them.
    """
    freq_mode = freqs[mode]
    lmbda = c0 / (freq_mode * 1e6)

    # peak field + geometry from the eigenmode mesh
    mesh, gfu_E, _ = load_eigenmode_fields(fields_dir)
    em = build_emfield(gfu_E, mode, freq_mode)
    xsurf = _surface_points(mesh)
    bounding_rect = _bounding_rect(xsurf)
    Epk = _peak_surface_field(em, mesh, xsurf)

    if epks is None:
        epks_v = 1 / Epk * 1e6 * np.linspace(0, 80, 192)
    else:
        epks_v = 1 / Epk * np.asarray(epks, dtype=float)   # user epks in V/m
    phi_v = np.linspace(0, 2 * np.pi, 72) if phis is None else np.asarray(phis)
    if xrange is None:
        # PyMultipact used a fixed [-0.25, 0] mm band tied to its own z-origin;
        # cavsim2d's profile has a different origin, so anchor to the geometry.
        xrange = default_xrange(xsurf)

    if proc_count is None:
        proc_count = max(1, min(mp.cpu_count() - 1, len(epks_v)))
    proc_count = max(1, int(proc_count))
    print(f"Multipacting Epk sweep: {len(epks_v)} points on {proc_count} "
          f"process{'es' if proc_count > 1 else ''}.")
    sweep_start = time.time()

    # round-robin split, remembering original indices for re-assembly
    divided, divided_idx = [[] for _ in range(proc_count)], [[] for _ in range(proc_count)]
    for idx, value in enumerate(epks_v):
        divided[idx % proc_count].append(value)
        divided_idx[idx % proc_count].append(idx)
    proc_epks = [np.array(lst) for lst in divided]

    for p in range(proc_count):                    # clear stale worker outputs
        for stale in (os.path.join(folder, f"mresults_{p}"),
                      os.path.join(folder, f"mresults_{p}.err"),
                      os.path.join(folder, f"mworker_{p}.log"),
                      os.path.join(folder, f"mprogress_{p}")):
            if os.path.exists(stale):
                try:
                    os.remove(stale)
                except OSError:
                    pass

    args = lambda p: (p, folder, fields_dir, freq_mode, mode, xrange, proc_epks[p],
                      phi_v, v_init, sey, Epk, step, bounding_rect, loss_model, t_max)
    with _SweepProgress(folder, proc_count, len(epks_v), enabled=progress):
        if proc_count == 1:
            _worker(*args(0))
        else:
            # Workers are plain subprocesses (NOT multiprocessing.Process): mp's
            # Windows spawn bootstrap re-executes the parent's __main__ in every
            # child, which requires an `if __name__ == '__main__':` guard in
            # scripts and breaks outright in notebooks/IPython. A fresh interpreter
            # running `-m ...driver <argsfile>` imports only this module, so the
            # sweep works the same from a notebook, an unguarded script, or a test.
            # Worker stdout/stderr goes to mworker_{p}.log (a child's console is
            # invisible in notebooks); failures also land in mresults_{p}.err.
            env = dict(os.environ)
            pkg_root = str(Path(__file__).resolve().parents[3])  # dir with cavsim2d/
            env['PYTHONPATH'] = pkg_root + os.pathsep + env.get('PYTHONPATH', '')
            procs = []
            for p in range(proc_count):
                args_path = os.path.join(folder, f"mworker_{p}.args")
                with open(args_path, 'wb') as f:
                    pickle.dump(args(p), f)
                log = open(os.path.join(folder, f"mworker_{p}.log"), 'w')
                proc = subprocess.Popen(
                    [sys.executable, '-m', 'cavsim2d.analysis.multipacting.driver',
                     args_path],
                    stdout=log, stderr=subprocess.STDOUT, env=env)
                procs.append((proc, log, args_path))
            for proc, log, args_path in procs:
                proc.wait()
                log.close()
                try:
                    os.remove(args_path)
                except OSError:
                    pass

    # re-assemble in epks_v order
    cf_by_idx, po_by_idx = {}, {}
    n_init = 1
    for p in range(proc_count):
        rf = os.path.join(folder, f"mresults_{p}")
        if not os.path.exists(rf):
            detail = ''
            for extra in (f"{rf}.err", os.path.join(folder, f"mworker_{p}.log")):
                if os.path.exists(extra):
                    with open(extra) as ef:
                        tail = ef.read()[-3000:]
                    if tail.strip():
                        detail += f"\n--- {os.path.basename(extra)} ---\n{tail}"
            raise RuntimeError(f"Multipacting worker {p} produced no result "
                               f"({rf}) -- it crashed.{detail}")
        with open(rf, "rb") as file:
            m = pickle.load(file)
        for local_i, global_i in enumerate(divided_idx[p]):
            cf_by_idx[global_i] = m['cn/c0'][local_i]
            po_by_idx[global_i] = m['particles_objects'][local_i]
        if p == 0:
            n_init = m['n_init_particles']

    cn_c0 = np.array([cf_by_idx[i] for i in range(len(epks_v))])
    particles_objects = [po_by_idx[i] for i in range(len(epks_v))]

    for po in particles_objects:
        distance_function(po, lmbda)

    elapsed = time.time() - sweep_start
    result = {'cn/c0': cn_c0, 'particles_objects': particles_objects,
              'n_init_particles': n_init, 'Epk': Epk, 'epks': epks_v,
              'phis_v': phi_v, 'freq [MHz]': freq_mode, 'mode': mode,
              'fields_dir': str(fields_dir), 'sweep_time [s]': elapsed}
    with open(os.path.join(folder, "mresults.pkl"), "wb") as file:
        pickle.dump(result, file)
    print(f"Multipacting sweep done: {len(epks_v)} field levels, "
          f"{n_init} launch sites x phases, {elapsed:.1f}s "
          f"({elapsed / max(len(epks_v), 1):.2f}s per field level).")
    return result


def _worker_main(args_path):
    """Subprocess entry point: load the pickled worker arguments and run."""
    with open(args_path, 'rb') as f:
        worker_args = pickle.load(f)
    _worker(*worker_args)


if __name__ == '__main__':
    # Invoked by run_sweep as `python -m cavsim2d.analysis.multipacting.driver
    # <argsfile>` — one sweep worker per interpreter (see run_sweep for why
    # this is a subprocess, not a multiprocessing.Process).
    _worker_main(sys.argv[1])
