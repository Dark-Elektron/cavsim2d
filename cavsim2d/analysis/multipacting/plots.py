"""Heavier multipacting plots: the distance map, the interactive trajectory
viewer and the trajectory animation. Ported from PyMultipact's ``Domain.plot_df``
/ ``Domain.plot_trajectories``, adapted to read from a
:class:`~cavsim2d.solvers.solver_objects.MultipactingSolver`.
Kept out of ``solver_objects`` so its ipywidgets/heavier deps stay lazy.
"""
import os
import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

from cavsim2d.analysis.multipacting.metrics import distance_function
from cavsim2d.utils.printing import info
from cavsim2d.utils.style import house_style

c0 = 299792458
q0 = 1.60217663e-19
m0 = 9.1093837e-31

#: Continuous colormap for velocity/energy-coded trajectories.
TRAJ_CMAP = 'inferno'


def _step_dt(solver):
    """The tracker's RK4 time step [s] — one ``paths`` row is recorded per step
    (``dt = 1/(120 f)``, see the sweep driver), so per-segment velocities follow
    from position differences."""
    return 1.0 / (solver.results['freq [MHz]'] * 1e6 * 20 * 6)


def _traj_values(path, dt, color_by):
    """Per-segment colour values along a trajectory: 'velocity' (speed [m/s])
    or 'energy' (relativistic kinetic energy [eV])."""
    d = np.diff(np.asarray(path)[:, :2], axis=0)
    v = np.linalg.norm(d, axis=1) / dt
    if color_by == 'velocity':
        return v, 'speed [m/s]'
    if color_by == 'energy':
        beta = np.clip(v / c0, 0.0, 1 - 1e-12)
        gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
        return (gamma - 1.0) * m0 * c0 ** 2 / q0, 'kinetic energy [eV]'
    raise ValueError(f"color_by must be 'velocity' or 'energy', got {color_by!r}")


def _traj_norm(values):
    """Log colour scale for trajectory values. The dynamic range is real and
    huge: wall-hugging multipacting electrons sit at ~10^2 eV while electrons
    crossing the cavity gap reach 10^5 eV — a linear scale washes the
    multipacting band into black."""
    v = np.asarray(values, dtype=float)
    v = v[v > 0]
    if not len(v):
        return plt.Normalize(0, 1)
    vmax = float(v.max())
    vmin = max(float(v.min()), vmax * 1e-5)
    if vmax <= vmin:
        vmax = vmin * 10
    return LogNorm(vmin, vmax)


def _phase_index(particles, k, path):
    """Launch-phase grid index of the *k*-th bright trajectory of *particles*."""
    phis_v = np.asarray(particles.phis_v)
    if hasattr(particles, 'bright_init_phi') and len(particles.bright_init_phi) > k:
        phi0 = float(particles.bright_init_phi[k])
    else:
        phi0 = float(np.asarray(path)[0, 2])
    return int(np.argmin(np.abs(phis_v - (phi0 % (2 * np.pi)))))


def _select_trajectories(solver, epk_i=None, phi_i=None, traj=None):
    """The bright (20-hit) trajectories matching a selection.

    ``epk_i`` / ``phi_i`` / ``traj`` are indices (int or list) into the swept
    field levels, the launch-phase grid and the per-level bright set; ``None``
    selects all. ``epk_i`` restricts the field levels; **within** those levels,
    ``traj`` (by bright-set index) and ``phi_i`` (by launch phase) each pick
    trajectories and their selections are **combined** (union) — so adding a
    selector only ever adds trajectories, it never intersects them away. Out-of
    -range field/trajectory/phase indices are skipped with a warning.

    Returns ``[(epk_index, traj_index, path), ...]`` with *path* the
    ``(n_steps, 3)`` ``[z, r, wt+phi]`` history.
    """
    particles_objects = solver.particles
    n_epk = len(particles_objects)
    if epk_i is None:
        epk_sel = list(range(n_epk))
    else:
        want = np.atleast_1d(epk_i).astype(int).tolist()
        epk_sel = [e for e in want if 0 <= e < n_epk]
        bad = [e for e in want if not (0 <= e < n_epk)]
        if bad:
            warnings.warn(f"animate/select: field-level index(es) {bad} out of "
                          f"range 0..{n_epk - 1}; ignored.", UserWarning, stacklevel=3)
    traj_sel = None if traj is None else set(np.atleast_1d(traj).astype(int).tolist())
    phi_sel = None if phi_i is None else set(np.atleast_1d(phi_i).astype(int).tolist())
    within = traj_sel is not None or phi_sel is not None   # any within-level filter?

    out = []
    for e in epk_sel:
        particles = particles_objects[e]
        for k, path in enumerate(particles.bright_set):
            if within:
                hit_traj = traj_sel is not None and k in traj_sel
                hit_phi = phi_sel is not None and _phase_index(particles, k, path) in phi_sel
                if not (hit_traj or hit_phi):          # union of the given filters
                    continue
            out.append((int(e), int(k), np.asarray(path)))

    if not out and epk_sel:
        # Say why nothing matched, with the valid ranges, rather than a bare None.
        parts = []
        for e in epk_sel[:6]:
            p = particles_objects[e]
            nb = len(p.bright_set)
            phases = sorted({_phase_index(p, k, pp) for k, pp in enumerate(p.bright_set)})
            parts.append(f"level {e}: {nb} bright"
                         + (f" (traj 0..{nb - 1}, phases {phases})" if nb else ""))
        info("No bright (20-hit) trajectories matched the selection. "
             + "; ".join(parts)
             + ". traj indexes the bright set; phi_i indexes the launch-phase grid.")
    return out


def _wall_polyline(solver):
    """The cavity wall polyline (z, r) in metres, from the cavity profile."""
    prof = solver.cavity.profile()
    return np.asarray(prof.contour_points(1e-3, skip=('AXI',)), dtype=float)


def distance_map(solver, epk_i, metric='d20', vmax=None, show=True):
    """MultiPac-style distance map over (emission site, launch phase) for the
    ``epk_i``-th field level. Grey cells = no electron survived to 20 impacts.
    ``metric``: 'd20' (parity-robust, default), 'd20_strict', or 'closure'."""
    r = solver.results
    particles_objects = solver.particles
    if not particles_objects:
        info("No multipacting results — run cav.multipacting.run(...) first.")
        return None
    particles = particles_objects[epk_i]
    if not hasattr(particles, 'bright_init_x') or not hasattr(particles, 'sites_init'):
        info("Result predates the bright-identity archive; re-run the analysis.")
        return None

    lmbda = c0 / (r['freq [MHz]'] * 1e6)
    kappa = lmbda / (2 * np.pi)
    if not hasattr(particles, 'df20'):
        distance_function(particles, lmbda)
    epk_axis = solver.epk
    boundary = _wall_polyline(solver)

    def _dist(x_a, phi_a, x_b, phi_b):
        return float(np.sqrt(np.linalg.norm(np.asarray(x_a) - np.asarray(x_b)) ** 2
                             + kappa * abs(np.exp(1j * phi_a) - np.exp(1j * phi_b)) ** 2))

    if metric == 'closure':
        values = [(_dist(xs[-1], ps[-1], xs[-3], ps[-3]) if len(xs) >= 3 else np.nan)
                  for xs, ps in zip(particles.bright_impact_x, particles.bright_impact_phi)]
        label = r'closure $d(x_{20}, x_{18})$'
    elif metric == 'd20' and not hasattr(particles, 'bright_impact_x'):
        values, label = particles.df20, r'$d_{20}$'
    elif metric == 'd20':
        values = []
        for x0, p0, xs, ps in zip(particles.bright_init_x, particles.bright_init_phi,
                                  particles.bright_impact_x, particles.bright_impact_phi):
            if len(xs) >= 2:
                values.append(min(_dist(x0, p0, xs[-1], ps[-1]),
                                  _dist(x0, p0, xs[-2], ps[-2])))
            elif len(xs) == 1:
                values.append(_dist(x0, p0, xs[-1], ps[-1]))
            else:
                values.append(np.nan)
        label = r'$d_{20}$'
        if vmax is None:
            vmax = 'kappa'
    elif metric == 'd20_strict':
        values, label = particles.df20, r'$d_{20}$ (strict)'
    else:
        raise ValueError(f"metric must be 'd20', 'd20_strict' or 'closure', got {metric!r}")

    sites = np.asarray(particles.sites_init)
    phis_v = np.asarray(particles.phis_v)
    dmap = np.full((len(phis_v), len(sites)), np.nan)
    for bx, bphi, df in zip(particles.bright_init_x, particles.bright_init_phi, values):
        si = int(np.argmin(np.linalg.norm(sites - np.asarray(bx), axis=1)))
        pi = int(np.argmin(np.abs(phis_v - bphi)))
        dmap[pi, si] = df

    with house_style():
        fig, axs = plt.subplots(2, 1, figsize=(8, 7), height_ratios=[2, 1.2])
        cmap = plt.get_cmap('hot').copy()
        cmap.set_bad('0.85')
        if vmax == 'kappa':
            vmax = kappa
        im = axs[0].pcolormesh(np.arange(1, len(sites) + 1), np.degrees(phis_v),
                               dmap, cmap=cmap, shading='nearest', vmin=0.0, vmax=vmax)
        fig.colorbar(im, ax=axs[0], label=label,
                     extend='max' if vmax is not None else 'neither')
        axs[0].set_xlabel('emission site (see below)')
        axs[0].set_ylabel('launch phase [deg]')
        axs[0].set_title(f'Distance map ({metric})   '
                         f'$E_{{\\mathrm{{pk}}}}$ = {epk_axis[epk_i]:.1f} MV/m')
        axs[1].plot(boundary[:, 0], boundary[:, 1], color='0.4', lw=1)
        axs[1].plot(sites[:, 0], sites[:, 1], 'o', mfc='none', mec='b', ms=5)
        for k, (sz, sr) in enumerate(sites):
            axs[1].annotate(str(k + 1), (sz, sr), fontsize=7)
        axs[1].set_xlabel('z [m]')
        axs[1].set_ylabel('r [m]')
        axs[1].set_aspect('equal', 'box')
        axs[1].set_title('Emission sites')
        fig.tight_layout()
    if show:
        plt.show()
    return axs


def _draw_trajectory(ax, boundary, path, dt, color_by, cmap, label):
    """Draw one trajectory (plain or gradient-coloured) onto *ax*, over the
    cavity wall. Returns the LineCollection when colour-coded (for a colorbar)."""
    ax.plot(boundary[:, 0] * 1e3, boundary[:, 1] * 1e3, lw=3, color='0.5')
    # trajectories are in the same (z, r) frame as the mesh/boundary --
    # PyMultipact's viewer negated z for its own convention, which mirrored the
    # path on asymmetric cavities
    z, r = path[:, 0] * 1e3, path[:, 1] * 1e3
    lc = None
    if color_by:
        vals, _ = _traj_values(path, dt, color_by)
        pts = np.column_stack([z, r])
        lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1),
                            cmap=cmap, linewidths=1.8, norm=_traj_norm(vals))
        lc.set_array(vals)
        ax.add_collection(lc)
    else:
        ax.plot(z, r, c='k', lw=1.5)
    # matplotlib requires sequences here -- passing the scalars raised
    # 'x must be a sequence' (the bug inherited from PyMultipact)
    ax.plot([z[0]], [r[0]], c='k', marker='o', ls='none', zorder=10)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('r [mm]')
    ax.set_aspect('equal', 'box')     # geometry must not be distorted
    ax.set_title(label)
    ax.relim()
    ax.autoscale_view()
    return lc


def trajectory_viewer(solver, color_by=None):
    """Interactive (ipywidgets) viewer of the surviving 20-hit trajectories.
    Notebook-only; needs ipywidgets (the ``[jupyter]`` extra).

    ``color_by``: ``None`` (plain line, default), ``'velocity'`` or ``'energy'``
    — colour-codes the path with a gradient colormap along the trajectory.

    Two sliders select the field level (``epk_i``) and the trajectory within it
    (``traj``); the plot redraws on every slider move.
    """
    # Deferred: ipywidgets is a notebook-only optional dependency ([jupyter] extra).
    from ipywidgets import IntSlider, interactive_output, HBox, VBox, Layout

    particles_objects = solver.particles
    if not particles_objects:
        info("No multipacting results — run cav.multipacting.run(...) first.")
        return None
    boundary = _wall_polyline(solver)
    epk_axis = solver.epk
    dt = _step_dt(solver)
    cmap = plt.get_cmap(TRAJ_CMAP)

    n_epks = len(particles_objects)
    # open at the field level with the most survivors (the counter peak), with
    # its first trajectory shown -- PyMultipact hardcoded indices 83/28, which
    # start the viewer blank whenever that level has no bright set
    epk0 = int(np.argmax([len(p.bright_set) for p in particles_objects]))
    epk_slider = IntSlider(min=0, max=n_epks - 1, step=1, description='epk_i:',
                           layout=Layout(width='50%'), value=epk0)
    n_bright0 = len(particles_objects[epk0].bright_set)
    w_slider = IntSlider(min=-1, max=max(n_bright0 - 1, -1), description='traj:',
                         layout=Layout(width='50%'), value=0 if n_bright0 else -1)

    def update_w_max(change):
        w_slider.max = len(particles_objects[change['new']].bright_set) - 1
    epk_slider.observe(update_w_max, names='value')

    def update(epk_i, traj):
        # The figure is (re)built inside the callback, not mutated in place: with
        # the notebook's inline backend, ipywidgets only re-displays figures
        # created *during* the callback -- mutating a figure made once outside
        # (the previous approach) left the plot frozen on every slider move.
        particles = particles_objects[epk_i]
        with house_style():
            fig, ax = plt.subplots(figsize=(6, 4))
            if len(particles.bright_set) and 0 <= traj < len(particles.bright_set):
                path = np.asarray(particles.bright_set[traj])
                lc = _draw_trajectory(
                    ax, boundary, path, dt, color_by, cmap,
                    f'{epk_axis[epk_i]:.1f} MV/m — trajectory {traj}')
                if lc is not None:
                    _, vlabel = _traj_values(path, dt, color_by)
                    fig.colorbar(lc, ax=ax, label=vlabel)
            else:
                ax.plot(boundary[:, 0] * 1e3, boundary[:, 1] * 1e3, lw=3, color='0.5')
                ax.set_aspect('equal', 'box')
                ax.set_title(f'{epk_axis[epk_i]:.1f} MV/m — no surviving trajectory')
            plt.show()

    out = interactive_output(update, {'epk_i': epk_slider, 'traj': w_slider})
    # Deferred: display is an IPython (notebook) API.
    from IPython.display import display
    display(VBox([HBox([epk_slider, w_slider]), out]))
    return out


def _is_notebook():
    """True inside a Jupyter/IPython kernel (so we can embed the animation)."""
    try:
        # Deferred: IPython is optional; absent in a plain interpreter.
        from IPython import get_ipython
        return type(get_ipython()).__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


def trajectory_animation(solver, epk_i=None, phi_i=None, traj=None,
                         color_by='energy', trail=40, step=1, fps=30,
                         save=None, dpi=120, zoom='auto', progress=True,
                         embed='auto'):
    """Animate the surviving trajectories: moving heads with a short fading
    trace, colour-coded by ``'energy'`` (default) or ``'velocity'``.

    ``zoom='auto'`` (default) frames the selected trajectories — multipacting
    orbits are sub-millimetre, invisible at full-cavity scale; ``zoom=None``
    shows the whole cavity.

    Selection (all combinable; ``None`` = all, the default):

    - ``epk_i``  — field-level index (int or list) into the swept ``epks``;
    - ``phi_i``  — launch-phase index (int or list) into the phase grid;
    - ``traj``   — bright-trajectory index (int or list) within a field level.

    ``trail`` is the trace length in recorded steps, ``step`` subsamples the
    record (speeds up long tracks), ``fps`` the playback rate. ``save`` writes
    the animation to a file (``.gif`` via Pillow, anything else via ffmpeg,
    e.g. ``.mp4``).

    ``progress=True`` shows a progress bar while frames render (both saving and
    embedding traverse every frame, so a many-particle animation is slow — the
    bar and a final timing make the wait legible).

    Always returns the :class:`~matplotlib.animation.FuncAnimation`, so the
    two-step ``anim = cav.multipacting.animate_trajectories(); anim.save(...)``
    still works. With ``embed='auto'`` (the default), in a notebook the animation
    is *also* played **inline** immediately (unless ``save`` was given, so a
    save-to-file call does not pay for a second render); ``embed=True``/``False``
    forces it on/off.
    """
    # Deferred: matplotlib.animation pulls in writer machinery only needed here.
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

    if not solver.particles:
        info("No multipacting results — run cav.multipacting.run(...) first.")
        return None
    sel = _select_trajectories(solver, epk_i=epk_i, phi_i=phi_i, traj=traj)
    if not sel:
        info("No bright (20-hit) trajectories match the selection.")
        return None

    step = max(1, int(step))
    dt = _step_dt(solver) * step
    paths = [p[::step] for (_, _, p) in sel]
    values = []
    vlabel = ''
    for p in paths:
        v, vlabel = _traj_values(p, dt, color_by)
        values.append(v)
    pooled = np.concatenate([v for v in values if len(v)])
    norm = _traj_norm(pooled)
    cmap = plt.get_cmap(TRAJ_CMAP)

    boundary = _wall_polyline(solver)
    epk_axis = solver.epk
    epks_shown = sorted({e for (e, _, _) in sel})
    title = (rf'$E_\mathrm{{pk}}$ = {epk_axis[epks_shown[0]]:.1f} MV/m'
             if len(epks_shown) == 1 else
             f'{len(sel)} trajectories, {len(epks_shown)} field levels')

    with house_style():
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(boundary[:, 0] * 1e3, boundary[:, 1] * 1e3, lw=2, color='0.45')
        lc = LineCollection([], linewidths=1.6)
        ax.add_collection(lc)
        heads = ax.scatter([], [], s=16, c=[], cmap=cmap, norm=norm, zorder=10)
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                     label=vlabel)
        ax.set_xlabel('z [mm]')
        ax.set_ylabel('r [mm]')
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
        if zoom == 'auto':
            # frame the trajectories, not the cavity: the orbits are sub-mm
            allp = np.vstack([p[:, :2] for p in paths]) * 1e3
            pad_z = max(0.2 * (allp[:, 0].max() - allp[:, 0].min()), 1.0)
            pad_r = max(0.2 * (allp[:, 1].max() - allp[:, 1].min()), 1.0)
            ax.set_xlim(allp[:, 0].min() - pad_z, allp[:, 0].max() + pad_z)
            ax.set_ylim(allp[:, 1].min() - pad_r, allp[:, 1].max() + pad_r)
        else:
            pad_z = 0.05 * (boundary[:, 0].max() - boundary[:, 0].min()) * 1e3
            pad_r = 0.05 * boundary[:, 1].max() * 1e3
            ax.set_xlim(boundary[:, 0].min() * 1e3 - pad_z,
                        boundary[:, 0].max() * 1e3 + pad_z)
            ax.set_ylim(0 - pad_r, boundary[:, 1].max() * 1e3 + pad_r)

    n_frames = max(len(p) for p in paths)

    def frame(f):
        segs, colors = [], []
        hx, hy, hv = [], [], []
        for p, v in zip(paths, values):
            i1 = min(f, len(p) - 1)
            i0 = max(0, i1 - trail)
            if i1 >= 1 and len(v):
                pts = p[i0:i1 + 1, :2] * 1e3
                seg = np.stack([pts[:-1], pts[1:]], axis=1)
                rgba = cmap(norm(v[i0:i1]))
                rgba[:, 3] = np.linspace(0.12, 1.0, len(rgba))   # fading trace
                segs.extend(seg)
                colors.extend(rgba)
            hx.append(p[i1, 0] * 1e3)
            hy.append(p[i1, 1] * 1e3)
            hv.append(v[min(i1, len(v) - 1)] if len(v) else norm.vmin)
        lc.set_segments(segs)
        lc.set_color(colors if colors else 'none')
        heads.set_offsets(np.column_stack([hx, hy]))
        heads.set_array(np.asarray(hv))
        return lc, heads

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=1000 / fps,
                         blit=False)

    def _bar(desc):
        """A frame-render progress callback (i, n) -> None, backed by tqdm."""
        # Deferred: tqdm is a core dependency but keep the import local to the
        # (optional) plotting path.
        from tqdm.auto import tqdm
        bar = tqdm(total=n_frames, desc=desc, disable=not progress,
                   unit='frame', leave=True)
        return lambda i, n: (bar.update(1), bar.close() if i + 1 >= n else None)

    if save:
        writer = (PillowWriter(fps=fps) if str(save).lower().endswith('.gif')
                  else FFMpegWriter(fps=fps))
        t0 = time.time()
        anim.save(save, writer=writer, dpi=dpi,
                  progress_callback=_bar(f'Saving {os.path.basename(str(save))}'))
        info(f"Trajectory animation saved to {save} "
             f"({n_frames} frames, {len(sel)} trajectories, "
             f"{time.time() - t0:.1f}s).")

    # embed='auto': play inline in a notebook, but not when saving (a
    # save-to-file call would otherwise render every frame a second time).
    want_embed = (_is_notebook() and not save) if embed == 'auto' else bool(embed)
    if want_embed:
        # Build the inline JS player (same per-frame cost as a save), display it,
        # and close the still frame so the notebook shows only the player, not a
        # duplicate static axes. The FuncAnimation is still returned so .save()
        # remains available.
        # Deferred: IPython display is a notebook API.
        from IPython.display import HTML, display
        t0 = time.time()
        html = (_to_jshtml_with_progress(anim, fps, n_frames) if progress
                else anim.to_jshtml(fps=fps, embed_frames=True))
        info(f"Trajectory animation rendered inline "
             f"({n_frames} frames, {len(sel)} trajectories, "
             f"{time.time() - t0:.1f}s).")
        plt.close(fig)
        display(HTML(html))
    return anim


def _to_jshtml_with_progress(anim, fps, n_frames):
    """``anim.to_jshtml`` with a tqdm bar over frame rendering (to_jshtml itself
    takes no progress callback, so drive the HTMLWriter directly)."""
    # Deferred: matplotlib.animation writer machinery only needed on this path.
    import tempfile
    # Deferred: same animation-only writer machinery.
    from matplotlib.animation import HTMLWriter
    # Deferred: tqdm core dep, kept local to the plotting path.
    from tqdm.auto import tqdm

    bar = tqdm(total=n_frames, desc='Rendering inline', unit='frame', leave=True)

    def cb(i, n):
        bar.update(1)
        if i + 1 >= n:
            bar.close()

    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
        path = f.name
    writer = HTMLWriter(fps=fps, embed_frames=True, default_mode='loop')
    anim.save(path, writer=writer, progress_callback=cb)
    with open(path) as f:
        html = f.read()
    try:
        os.remove(path)
    except OSError:
        pass
    return html
