"""Multipacting metrics computed from tracked particles.

Ported from PyMultipact's ``Domain.calculate_*`` / ``plot_*`` helpers: the
distance function ``d20`` (locates resonant fixed points), the mean final impact
energy ``Ef20``, the enhanced counter ``e20/c0`` (secondary-yield weighted), and
the launchable fraction (MultiPac-compatible counter normalisation).
"""
import numpy as np


def distance_function(particles, lmbda):
    """Distance ``d20`` of each 20-hit (bright) trajectory: the (position, phase)
    distance between the launch point and the 20th impact (Yla-Oijala Eq. 3.2.3).
    Minima locate the fixed points of resonant multipacting orbits. Stored on
    ``particles.df20``, aligned with ``bright_set``."""
    kappa = lmbda / (2 * np.pi)
    particles.df20 = []
    use_exact = (hasattr(particles, 'bright_impact_x')
                 and len(getattr(particles, 'bright_impact_x', []))
                 == len(particles.bright_set))
    for path_i in range(len(particles.bright_set)):
        if use_exact and len(particles.bright_impact_x[path_i]) > 0:
            x_0 = np.asarray(particles.bright_init_x[path_i])
            phi_0 = particles.bright_init_phi[path_i]
            x_n = np.asarray(particles.bright_impact_x[path_i][-1])
            phi_n = particles.bright_impact_phi[path_i][-1]
        else:
            path = particles.bright_set[path_i]
            x_0, x_n = path[0, 0:2], path[-1, 0:2]
            phi_0, phi_n = path[0, 2], path[-1, 2]
        df = np.sqrt(np.linalg.norm(x_n - x_0) ** 2
                     + kappa * abs(np.exp(1j * phi_n) - np.exp(1j * phi_0)) ** 2)
        particles.df20.append(float(df))
    return particles.df20


def final_energy(particles_objects):
    """Mean FINAL impact energy [eV] of the electrons that reached 20 hits, per
    field level -- zero where nothing survived (the paper's ``Ef_20``)."""
    Ef = []
    for particles in particles_objects:
        bright_E = getattr(particles, 'bright_E', None)
        if bright_E is None:
            Ef_p = [pe[-1] for pe in particles.E if len(pe) != 0]
        else:
            Ef_p = [be[-1] for be in bright_E if len(be) != 0]
        Ef.append(np.sum(Ef_p) / len(Ef_p) if len(Ef_p) > 0 else 0)
    return Ef


def enhanced_counter(particles_objects, n_init_particles):
    """Enhanced counter ``e20/c0`` per field level: the secondary-yield product
    over the 20-hit trajectories. Uses the archived (aligned) bright histories,
    each with at most ~20 entries."""
    secondaries = [
        sum(np.prod(nn) for nn in getattr(particles, 'bright_n_secondaries',
                                          particles.n_secondaries))
        for particles in (particles_objects or [])]
    if not secondaries:
        return np.array([])
    return 2 * (np.array(secondaries) + 1) / n_init_particles


def launchable_fraction(particles0, em, mesh):
    """Fraction of launched (site, phase) pairs whose surface field lets the
    electron leave the wall (E.n >= 0). ~0.5 for a sinusoidal field. MultiPac's
    counter effectively counts only launchable electrons in c0, so dividing the
    counter by this fraction matches its normalisation."""
    if particles0 is None or em is None or mesh is None \
            or not hasattr(particles0, 'sites_init'):
        return 0.5
    sites = np.asarray(particles0.sites_init)
    normals = np.asarray(particles0.pt_normals[:particles0.n_sites])
    phis = np.asarray(particles0.phis_v)
    fav, tot = 0, 0
    for sx, nrm in zip(sites, normals):
        ec = np.asarray(em.e(mesh(float(sx[0]), float(sx[1]))),
                        dtype=complex).ravel()[:2]
        e_at_phis = np.real(np.outer(np.exp(1j * phis), ec))   # (n_phis, 2)
        fav += int(np.sum(e_at_phis @ nrm >= 0))
        tot += len(phis)
    return fav / tot if tot else 0.5
