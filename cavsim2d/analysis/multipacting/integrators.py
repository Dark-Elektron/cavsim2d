"""Relativistic particle tracking for multipacting analysis.

Ported near-verbatim from PyMultipact (the user's validated physics): RK4
relativistic Lorentz push, exception-driven wall detection, secondary emission
via the SEY table, and the ``loss_model`` handling. Only the plumbing (imports,
field source) is adapted; the dynamics are unchanged.
"""
import copy
import os
import time

import numpy as np
import matplotlib.pyplot as plt

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458

# Number of nearest surface points fetched per boundary-check to reconstruct
# the local wall segments. The paper code used 1000 (nearly the whole surface);
# only a small local neighbourhood is needed to bracket the crossed segment, and
# the KD-tree query cost scales with this number. 100 was validated on the TESLA
# mid-cell: identical counter function and impact energies to 1000 across the
# resonance band, ~3-5x faster. A too-small value can MISS a crossing so the
# particle escapes the mesh, but that is now handled gracefully (see the k1
# guard in rk4 / _inside_mask) rather than crashing. Override via env var.
_HIT_NEIGHBOURS = int(os.environ.get('HIT_NEIGHBOURS', '100'))


class _ParticleDummy:
    """Lightweight stand-in for the deepcopy of Particles used inside rk4.

    rk4/hit_bound only read/write x, u, phi, x_old, u_old, u_temp (and call
    save_old()/distance()) on the dummy; the heavy per-particle lists
    (E, n_secondaries, colours, path buffer, ...) are never touched. Copying
    only the three state arrays -- instead of deepcopy'ing the whole object --
    is numerically identical but avoids the dominant runtime cost. ``bounds``
    is shared by reference (read-only in distance()).
    """

    __slots__ = ['x', 'u', 'phi', 'x_old', 'u_old', 'phi_old',
                 'x_temp', 'u_temp', 'phi_temp', 'bounds', '_bounds_tree']

    def __init__(self, particles):
        self.x = particles.x.copy()
        self.u = particles.u.copy()
        self.phi = particles.phi.copy()
        self.bounds = particles.bounds
        self._bounds_tree = particles._bounds_tree  # shared, read-only

    def save_old(self):
        self.x_old = self.x.copy()
        self.u_old = self.u.copy()
        self.phi_old = self.phi.copy()
        self.x_temp = self.x.copy()
        self.u_temp = self.u.copy()
        self.phi_temp = self.phi.copy()

    def distance(self, n):
        # identical semantics to Particles.distance (KD-tree nearest-surface)
        n = min(n, len(self.bounds))
        dists, idxs = self._bounds_tree.query(self.x, k=n)
        if n == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        return np.atleast_2d(dists[:, 0]).T, idxs.tolist()


class Integrators:
    def __init__(self, mesh, w, bounding_rect, loss_model='field'):
        self.mesh = mesh
        self.w = w
        self.fig, self.ax = plt.subplots()

        self.zmin, self.zmax, self.rmin, self.rmax = bounding_rect

        # What happens to an electron that impacts the wall while the surface
        # electric field points the wrong way (E.n < 0, a secondary could not
        # leave):
        #   'field'  -- absorb it (paper behaviour, default)
        #   'wait'   -- re-emit it uncounted; it is pushed back to the wall and
        #               retries until the RF phase turns favourable
        #               (approximates MultiPac-style delayed re-emission)
        #   'always' -- re-emit AND count the impact (upper bound)
        if loss_model not in ('field', 'wait', 'always'):
            raise ValueError(f"loss_model must be 'field', 'wait' or 'always', "
                             f"got {loss_model!r}")
        self.loss_model = loss_model

    def forward_euler(self, particles, tn, h, em, scale, sey):
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        particles.u = particles.u + ku1
        particles.x = particles.x + h * particles.u
        self.plot_path(particles)

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)

        if len(rpi) != 0:
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            particles.remove(lpi)

    def implicit_euler(self, particles, tn, h, em, scale):
        pass

    def rk2(self, particles, tn, h, em, scale):

        # k1
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        kx1 = h * particles.u

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku1[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx1[rp] = particles_dummy.x[rp] - particles.x[rp]

        if len(lpi) != 0:
            [ku1], [kx1] = self.rk_update_k([ku1], [kx1], lpi)
            particles_dummy.remove(lpi)
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        # k2=================================================
        ku2 = h * self.lorentz_force(particles_dummy, tn + 2 * h / 3, em,
                                     scale)  # <- particles dummy = particles.u + kn
        kx2 = h * (particles.u + 2 * ku1 / 3)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += 2 * ku2 / 3
        particles_dummy.x += 2 * kx2 / 3

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + 2 * h / 3, 2 * h / 3, em, scale)

        for rp in rpi:
            ku2[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx2[rp] = particles_dummy.x[rp] - particles.x[rp]

        if len(lpi) != 0:
            particles_dummy.remove(lpi)
            [ku1, ku2], [kx1, kx2] = self.rk_update_k([ku1, ku2], [kx1, kx2], lpi)
            particles.remove(lpi)
            em.remove(lpi)

        particles.u = particles.u + ku2
        particles.x = particles.x + kx1

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale)
        if len(lpi) != 0:
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles)

    def rk2_23(self, particles, tn, h, em, scale):
        # k1
        ku1 = h * self.lorentz_force(particles, tn, em, scale)
        kx1 = h * particles.u

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku1[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx1[rp] = particles_dummy.x[rp] - particles.x[rp]
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            [ku1], [kx1] = self.rk_update_k([ku1], [kx1], lpi)
            particles_dummy.remove(lpi)
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        # k2=================================================
        ku2 = h * self.lorentz_force(particles_dummy, tn + h / 2, em, scale)  # <- particles dummy = particles.u + kn
        kx2 = h * (particles.u + ku1 / 2)

        particles_dummy = copy.deepcopy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku2 / 2
        particles_dummy.x += kx2 / 2

        # check for lost particles
        lpi, rpi = self.hit_bound(particles_dummy, tn + h / 2, h / 2, em, scale)

        for rp in rpi:
            ku2[rp] = particles_dummy.u[rp] - particles.u[rp]
            kx2[rp] = particles_dummy.x[rp] - particles.x[rp]
            particles.update_hit_count(rpi)

        if len(lpi) != 0:
            particles_dummy.remove(lpi)
            [ku1, ku2], [kx1, kx2] = self.rk_update_k([ku1, ku2], [kx1, kx2], lpi)
            particles.remove(lpi)
            em.remove(lpi)

        particles.u = particles.u + (1 / 4 * ku1 + 3 / 4 * ku2)
        particles.x = particles.x + (1 / 4 * kx1 + 3 / 4 * kx2)

        # check for lost particles
        lpi, rpi = self.hit_bound(particles, tn, h, em, scale)
        if len(lpi) != 0:
            particles.remove(lpi)
            em.remove(lpi)

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles)

    # def rk4(self, particles, tn, h, em, scale, sey):
    #     #         start = time.time()
    #     # k1
    #     ku1 = h * self.lorentz_force(particles, tn, em, scale)
    #     kx1 = h * particles.u
    #     print('Strat')
    #     #         ss = time.time()
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku1 / 2
    #     particles_dummy.x += kx1 / 2
    #
    #     try:
    #         print('In here 1')
    #         ku2 = h * self.lorentz_force(particles_dummy, tn + h / 2, em,
    #                                      scale)  # <- particles dummy = particles.u + kn
    #         kx2 = h * (particles.u + ku1 / 2)
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 2')
    #         return
    #
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku2 / 2
    #     particles_dummy.x += kx2 / 2
    #
    #     try:
    #         print('In here 3')
    #         ku3 = h * self.lorentz_force(particles_dummy, tn + h / 2, em,
    #                                      scale)  # <- particles dummy = particles.u + kn
    #         kx3 = h * (particles.u + ku2 / 2)
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 2')
    #
    #         return
    #
    #     particles_dummy = copy.deepcopy(particles)
    #
    #     particles_dummy.save_old()
    #     particles_dummy.u += ku3
    #     particles_dummy.x += kx3
    #
    #     try:
    #         print('In here 4')
    #         ku4 = h * self.lorentz_force(particles_dummy, tn + h, em, scale)  # <- particles dummy = particles.u + kn
    #         kx4 = h * particles_dummy.u
    #     except:
    #         # take full step - euler
    #         particles.u = particles.u + ku1
    #         particles.x = particles.x + kx1
    #
    #         lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #         if len(rpi) != 0:
    #             particles.update_hit_count(list(set(rpi)))
    #
    #         if len(lpi) != 0:
    #             particles.remove(lpi)
    #
    #         self.plot_path(particles, tn)
    #         print('Return 3')
    #
    #         return
    #
    #     particles.u = particles.u + 1 / 6 * (ku1 + 2 * ku2 + 2 * ku3 + ku4)
    #     particles.x = particles.x + 1 / 6 * (kx1 + 2 * kx2 + 2 * kx3 + kx4)
    #
    #     # check for lost particles
    #     lpi, rpi = self.hit_bound(particles, tn, h, em, scale, sey)
    #
    #     if len(rpi) != 0:
    #         particles.update_hit_count(list(set(rpi)))
    #
    #     if len(lpi) != 0:
    #         particles.remove(lpi)
    #
    #     # check if all particles are lost
    #     if particles.len == 0:
    #         print('Return 54')
    #         return False
    #     #         print("rk4 exec time: ", time.time() - start)
    #     #         print('='*80)
    #
    #     self.plot_path(particles, tn)
    #     print('Finish')

    def rk4(self, particles, tn, h, em, scale, sey):
        lpi_all = []
        rpi_all = []
        spi_all = []   # soft-reflected ('wait' model): masked but uncounted
        lpi_rpi_all = []

        mask = np.ones(len(particles.x), dtype=bool)

        # k1
        try:
            ku1 = h * self.lorentz_force(particles, mask, tn, em, scale)
        except Exception:
            # A particle is already outside the meshed domain at the start of the
            # step -- a reflected/secondary was placed just outside, or a wall
            # crossing was missed on the previous step. It cannot be field-
            # evaluated, so drop it as lost and retry. This guards the k1 stage
            # the same way k2-k4 are already guarded; without it a single stray
            # escapee raises "Meshpoint not in mesh" and kills the whole run.
            # On well-behaved steps the field eval succeeds and this never fires,
            # so results are unchanged.
            escaped = mask & ~self._inside_mask(particles.x, mask, em)
            lpi_all.extend(np.nonzero(escaped)[0].tolist())
            mask[escaped] = False
            ku1 = np.zeros_like(particles.x)
            ku1[mask] = h * self.lorentz_force(particles, mask, tn, em, scale)
        kx1 = h * particles.u
        particles_dummy = _ParticleDummy(particles)
        particles_dummy.save_old()
        particles_dummy.u += ku1 / 2
        particles_dummy.x += kx1 / 2

        try:
            # start = time.time()
            ku2, kx2 = np.zeros_like(particles.x), np.zeros_like(particles.x)

            ku2[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em, scale)  # <- particles dummy = particles.u + kn
            kx2[mask] += h * (particles_dummy.u[mask])
            # print('check time:: ', time.time() - start)

        except Exception as e:
            # print('EXCEPTION1:: ')
            lpi, rpi, spi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)
            spi_all.extend(spi)
            lpi_rpi_all = lpi_all + rpi_all + spi_all

            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = _ParticleDummy(particles)
            particles_dummy.save_old()

            ku2, kx2 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku2[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx2[mask] += h * (particles_dummy.u[mask] + ku1[mask] / 2)

        particles_dummy = _ParticleDummy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += ku2[mask] / 2
        particles_dummy.x[mask] += kx2[mask] / 2

        try:
            ku3, kx3 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku3[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx3[mask] += h * (particles_dummy.u[mask])
        except Exception as e:
            # print('EXCEPTION2:: ')
            lpi, rpi, spi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)
            spi_all.extend(spi)

            lpi_rpi_all = lpi_all + rpi_all + spi_all
            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = _ParticleDummy(particles)
            particles_dummy.save_old()

            ku3, kx3 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku3[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h / 2, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx3[mask] += h * (particles_dummy.u[mask] + ku1[mask] / 2)

        particles_dummy = _ParticleDummy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += ku3[mask]
        particles_dummy.x[mask] += kx3[mask]

        try:
            ku4, kx4 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku4[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h, em,
                                                     scale)  # <- particles dummy = particles.u + kn
            kx4[mask] += h * (particles_dummy.u[mask] + ku3[mask])
        except Exception as e:
            # print('EXCEPTION3:: ', mask, len(particles.x), lpi_rpi_all)
            # print(particles_dummy.x, len(particles_dummy.x))
            lpi, rpi, spi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
            lpi_all.extend(lpi)
            rpi_all.extend(rpi)
            spi_all.extend(spi)

            lpi_rpi_all = lpi_all + rpi_all + spi_all
            # print(mask, lpi, rpi, lpi_all, rpi_all, lpi_rpi_all)
            if len(lpi_rpi_all) != 0:
                mask[np.sort(lpi_rpi_all)] = False

            particles_dummy = _ParticleDummy(particles)
            particles_dummy.save_old()

            ku4, kx4 = np.zeros_like(particles.x), np.zeros_like(particles.x)
            ku4[mask] += h * self.lorentz_force(particles_dummy, mask, tn + h, em,
                                                scale)  # <- particles dummy = particles.u + kn
            kx4[mask] += h * (particles_dummy.u[mask] + ku3[mask])

        particles_dummy = _ParticleDummy(particles)
        particles_dummy.save_old()
        particles_dummy.u[mask] += 1 / 6 * (ku1[mask] + 2 * ku2[mask] + 2 * ku3[mask] + ku4[mask])
        particles_dummy.x[mask] += 1 / 6 * (kx1[mask] + 2 * kx2[mask] + 2 * kx3[mask] + kx4[mask])

        # print("dummy particle", len(particles_dummy.x), '\n', particles_dummy.x)
        # check for lost particles
        lpi, rpi, spi = self.hit_bound(particles, particles_dummy, mask, tn, h, em, scale, sey)
        lpi_all.extend(lpi)
        rpi_all.extend(rpi)
        spi_all.extend(spi)

        lpi_rpi_all = lpi_all + rpi_all + spi_all
        if len(lpi_rpi_all) != 0:
            mask[np.sort(lpi_rpi_all)] = False

        # modify to only update particles not reflected or lost
        particles.u[mask] += 1 / 6 * (ku1[mask] + 2 * ku2[mask] + 2 * ku3[mask] + ku4[mask])
        particles.x[mask] += 1 / 6 * (kx1[mask] + 2 * kx2[mask] + 2 * kx3[mask] + kx4[mask])

        removed_inds = np.array([])
        if len(rpi_all) != 0:
            removed_inds = particles.update_hit_count(list(set(rpi_all)))

        if len(lpi_all) != 0:
            particles.remove(self.update_lpi(lpi_all, removed_inds))

        # check if all particles are lost
        if particles.len == 0:
            return False
        self.plot_path(particles, tn)

        # print("Done rk4", len(particles.x), '\n', particles.x)
        # self.trace(particles)
        # print('=='*50)

    def rkf45(self):
        pass

    def rk_update_k(self, ku_list, kx_list, lpi):
        ku_list_new, kx_list_new = [], []
        for ku in ku_list:
            ku = np.delete(ku, lpi, axis=0)
            ku_list_new.append(ku)
        for kx in kx_list:
            kx = np.delete(kx, lpi, axis=0)
            kx_list_new.append(kx)

        return ku_list_new, kx_list_new

    def adams_bashforth(self):
        pass

    def leapfrog(self):
        pass

    def lorentz_force(self, particles, mask, tn, em, scale):
        x, u, phi = particles.x[mask], particles.u[mask], particles.phi[mask]
        mps = self.mesh(x[:, 0], x[:, 1])          # build mesh points once (was twice)
        phase = np.exp(1j * (self.w * tn + phi))   # phase factor once (was twice)
        e = scale * em.e(mps) * phase
        b = mu0 * scale * em.h(mps) * phase

        k = q0 / m0 * np.sqrt(1 - (self.norm(u) / c0) ** 2) * (
                e.real + self.cross(u, b.real) - (1 / (c0 ** 2)) * (self.dot(u, e.real) * u))  # <- relativistic

        return k

    def _inside_mask(self, x, mask, em):
        """Per-particle test of whether each masked point can be field-evaluated
        (i.e. lies in the meshed domain). Note self.mesh(z, r) does NOT raise for
        an outside point -- it returns an invalid mesh point and only the field
        evaluation raises -- so we must actually evaluate the field here. Only
        invoked on a rare field-evaluation failure, so the loop cost is
        negligible."""
        inside = np.zeros(len(x), dtype=bool)
        for i in np.nonzero(mask)[0]:
            try:
                em.e(self.mesh(float(x[i, 0]), float(x[i, 1])))
                inside[i] = True
            except Exception:
                inside[i] = False
        return inside

    def plot_path(self, particles, tn=None):
        if tn is None:
            tn = 1 / self.w
        #     print('before', particles.paths.shape, particles.paths_count)
        #     print(particles.x.shape, particles.phi.shape, particles.paths.shape)
        particles.paths = np.vstack((particles.paths, np.hstack((particles.x, self.w * tn + particles.phi))))
        particles.paths_count += 1

    def hit_bound(self, particles, particles_dummy, mask, t, dt, em, scale, sey):
        xsurf = particles_dummy.bounds
        #     # check if particle close to boundary
        res, indx = particles_dummy.distance(_HIT_NEIGHBOURS)
        ind_ = np.where(res <= c0 * dt)
        res, indx = res[ind_[0], ind_[1]], np.array(indx)[ind_[0], :]

        lost_particles_indx = []
        reflected_particles_indx = []
        soft_reflected_indx = []   # 'wait' model: re-launched, not counted
        for ind, r, idx in zip(ind_[0], res, indx):
            #             if r < 5e-2 and mask[ind]:  # point at boundary, calculate new field value
            if r < c0 * dt and mask[ind]:  # point at boundary, calculate new field value
                # check if point is inside or outside of region
                # get surface points neighbours
                surf_pts_neigs = self.get_neighbours(xsurf, idx)

                # check for intersection
                # get intersection with old point. loop through points again.
                # the surface edge a line between an outside point and the origin intersects
                # might be different from that with which the line between old and new point intersects

                line11 = (particles_dummy.x[ind],
                          particles_dummy.x_old[ind])  # <- straight line btw current and previous points
                line22 = surf_pts_neigs[1:], surf_pts_neigs[:-1]

                bool_intc_p, x_intc_p, intc_indx = self.segment_intersection(line11, line22)
                # self.ax.plot(np.array(line11).T[0], np.array(line11).T[1], c='b', marker='o', zorder=2000)
                # self.ax.plot(np.array(line22).T[0], np.array(line22).T[1], c='r', marker='o')

                if bool_intc_p:
                    dt_frac = np.linalg.norm(x_intc_p - particles_dummy.x_old[ind]) / np.linalg.norm(
                        particles_dummy.x[ind] - particles_dummy.x_old[ind])
                    t_frac = t - dt * (1 - dt_frac)

                    # Advance particle to surface
                    #  calculate field values at this time which is a (fraction of dt) + t
                    e = scale * np.array([em.e(self.mesh(*x_intc_p))]) * np.exp(
                        1j * (self.w * t_frac + particles_dummy.phi[ind]))
                    b = mu0 * scale * np.array([em.h(self.mesh(*x_intc_p))]) * np.exp(
                        1j * (self.w * t_frac + particles_dummy.phi[ind]))

                    # check if the e-field surface normal is close to zero indicating a possible change in field
                    line22 = np.array(line22)[:, intc_indx]
                    line22 = line22[line22[:, 0].argsort()]
                    line22_normal = -np.array([-(line22[1][1] - line22[0][1]), line22[1][0] - line22[0][0]])
                    line22_normal = line22_normal / np.linalg.norm(line22_normal)
                    # self.ax.plot(np.array(line11).T[0], np.array(line11).T[1], c='k', marker='o', zorder=2000)
                    # self.ax.plot(np.array(line22).T[0], np.array(line22).T[1], c='g', marker='o', zorder=20000)

                    e_dot_surf_norm = np.dot(e.real, line22_normal)

                    # impact energy from the velocity advanced to the wall
                    # (needed by every loss model)
                    particles_dummy.u_temp[ind] = (particles_dummy.u_old[ind] +
                                                   q0 / m0 * np.sqrt(1 - (self.norm([particles_dummy.u_old[ind]]) / c0) ** 2) *
                                                   (e.real + self.cross([particles_dummy.u_old[ind]], b.real) -
                                                    (1 / c0 ** 2) * (self.dot([particles_dummy.u_old[ind]], e.real) *
                                                                     particles_dummy.u_old[ind])) * dt * dt_frac)
                    umag = np.linalg.norm(particles_dummy.u_temp[ind])
                    gamma = 1 / (np.sqrt(1 - (umag / c0) ** 2))
                    Eq = (gamma - 1) * m0 * c0 ** 2 * 6.241509e18  # 6.241509e18 Joules to eV factor

                    if e_dot_surf_norm >= 0:
                        # favourable surface field: secondary leaves the wall.
                        # update main particles array
                        particles.E[ind].append(Eq)
                        # exact impact point and RF phase for the distance
                        # function d_n
                        particles.impact_x[ind].append(np.asarray(x_intc_p, dtype=float).copy())
                        particles.impact_phi[ind].append(
                            float(self.w * t_frac + particles_dummy.phi[ind, 0]))

                        # calculate number of secondary electrons
                        if sey.Emin < Eq < sey.Emax:
                            particles.n_secondaries[ind].append(float(sey.sey(Eq)))
                        else:
                            particles.n_secondaries[ind].append(0)

                        # calculate new position using 1-dt_frac, u_temp at intersection and x_temp
                        # u_emission = line22_normal * np.sqrt(2 * particles.init_v * q0 / m0)
                        u_emission = line22_normal * particles.init_v

                        # use impact energy to calculate velocity of secondary particles
                        # to be implemented
                        particles.u[ind] = (u_emission + q0 / m0 * np.sqrt(1 - (self.norm([u_emission]) / c0) ** 2) *
                                            (e.real + self.cross([u_emission], b.real) - (1 / c0 ** 2) *
                                             (self.dot([u_emission], e.real) * u_emission)) * dt * (1 - dt_frac))

                        particles.x[ind] = x_intc_p + particles.u[ind] * dt * (1 - dt_frac)
                        reflected_particles_indx.append(ind)
                    elif self.loss_model == 'field':
                        # unfavourable surface field: absorb (paper behaviour)
                        lost_particles_indx.append(ind)
                    else:
                        # 'wait' / 'always': do not absorb. Re-launch from the
                        # impact point along the inward normal (no field kick --
                        # the unfavourable field simply pushes it back to the
                        # wall until the RF phase turns favourable).
                        u_emission = line22_normal * particles.init_v
                        particles.u[ind] = u_emission
                        particles.x[ind] = x_intc_p + u_emission * dt * (1 - dt_frac)
                        if self.loss_model == 'always':
                            # count the impact and record its energy
                            particles.E[ind].append(Eq)
                            particles.impact_x[ind].append(np.asarray(x_intc_p, dtype=float).copy())
                            particles.impact_phi[ind].append(
                                float(self.w * t_frac + particles_dummy.phi[ind, 0]))
                            if sey.Emin < Eq < sey.Emax:
                                particles.n_secondaries[ind].append(float(sey.sey(Eq)))
                            else:
                                particles.n_secondaries[ind].append(0)
                            reflected_particles_indx.append(ind)
                        else:  # 'wait': uncounted retry
                            soft_reflected_indx.append(ind)

        # finally check if particle is at the other boundaries not the wall surface
        # (vectorised; same set of indices as the previous per-particle loop --
        # bottom rotation axis, then left/right z-edges. Order within the list is
        # irrelevant: callers dedup via set() before use.)
        px = particles_dummy.x
        lost_particles_indx.extend(np.nonzero(px[:, 1] <= self.rmin)[0].tolist())
        lost_particles_indx.extend(
            np.nonzero((px[:, 0] <= self.zmin) | (px[:, 0] >= self.zmax))[0].tolist())

        return lost_particles_indx, reflected_particles_indx, soft_reflected_indx

    @staticmethod
    def cross(a, b):
        c1 = np.array(a)[:, 1] * np.array(b)[:, 0]
        c2 = -np.array(a)[:, 0] * np.array(b)[:, 0]
        return np.array([c1, c2]).T

    @staticmethod
    def dot(a, b):
        return np.atleast_2d(np.sum(a * b, axis=1)).T

    @staticmethod
    def norm(a):
        return np.atleast_2d(np.linalg.norm(a, axis=1)).T

    @staticmethod
    def segment_intersection(line1, line2):
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0][:, 0], line2[0][:, 1]
        x4, y4 = line2[1][:, 0], line2[1][:, 1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        #     if denom == 0:
        #        return False, (0, 0)

        tt = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        uu = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom

        # get index of where condition is true. condition should be true at only one point but for the case that
        #  a line has one point on a surface node intersects two edges
        condition = np.where((tt >= 0) * (tt <= 1) * (uu >= 0) * (uu <= 1))[0]
        # print('condition', condition)

        if len(condition) > 0:  # review for more complex geometry. a line that has one point on a surface node intersects two points.
            px, py = x1 + tt[condition[0]] * (x2 - x1), y1 + tt[condition[0]] * (y2 - y1)
            return True, np.array([px, py]), condition[0]
        else:
            return False, np.array([0, 0]), 0

    @staticmethod
    def get_neighbours(surf_pts, idx):
        surf_pts_neigs = np.array(surf_pts[idx])
        return surf_pts_neigs[surf_pts_neigs[:, 0].argsort()]

    def collision(self, active_interval):
        pass

    def indices_corrector(self, ind1, ind2):
        array_comp = np.array(ind2)
        for ii in ind1:
            array_comp += (array_comp >= ii) * 1
        return array_comp

    def compose_indices(self, indices_array):
        composed_indices = np.array([])
        for i in range(len(indices_array) - 1):
            if i == 0:
                xx = np.array(indices_array[i])
            else:
                xx = composed_indices

            array_comp = np.array(indices_array[i + 1])
            for ii in xx:
                array_comp += (array_comp >= ii) * 1
            composed_indices = np.concatenate((xx, array_comp))

        return composed_indices

    def update_lpi(self, lpi, removed_inds):
        # update lpi with removed indices from rpi
        counts = np.array([sum(1 for num in removed_inds if num < x) for x in lpi])
        lpi = np.array(lpi) - counts
        return lpi

    def update_rpi(self, rpi, removed_inds):
        # update lpi with removed indices from rpi
        counts = np.array([sum(1 for num in removed_inds if num < xx) for xx in rpi])
        rpi = np.array(rpi) - counts
        return rpi

    def trace(self, particles):
        for xx_old, xx in zip(particles.x_old, particles.x):
            self.ax.plot([xx[0], xx_old[0]], [xx[1], xx_old[1]], color='r', marker='o', ms=1, zorder=10000)
