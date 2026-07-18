"""Multipacting particle state (ported near-verbatim from PyMultipact).

Tracks the emitted electrons: position/velocity/phase, the per-particle impact
archive, and the 20-hit "bright set" of resonant survivors. Only the imports are
trimmed; the dynamics are unchanged.
"""
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

q0 = 1.60217663e-19
m0 = 9.1093837e-31
mu0 = 4 * np.pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458


class Particles:
    def __init__(self, xrange, init_v, bounds, phi, cmap='jet', step=None):
        # self.fig, self.ax = plt.subplots()

        self.cmap = cmap
        M = len(phi)

        self.bounds = np.array(bounds)
        # KD-tree over the fixed surface points; nearest-surface queries used to
        # be an O(N_particles x N_surf) distance matrix + full argsort per call.
        self._bounds_tree = cKDTree(self.bounds)
        # self.show_initial_points(xrange, step)

        self.x = self.bounds[(self.bounds[:, 0] > xrange[0]) & (self.bounds[:, 0] < xrange[1])]
        if step:
            self.x = self._select_values_with_step(self.x, step)

        # check if any particle in range
        if len(self.x) == 0:
            print('No surface emission point in selected range.')
            exit()

        shape = self.x.shape
        self.len = len(self.x)
        # number of emission sites (before tiling over phases); particle index
        # i corresponds to site i % n_sites and phase i // n_sites
        self.n_sites = self.len
        self.phis_v = np.asarray(phi)
        self.sites_init = self.x.copy()   # emission sites, immune to removals

        # get normal pointing inwards of emission points
        self.pt_normals = np.ones(self.x.shape)
        # get neighbouring points
        res, idxs = self.distance(1)
        for nn, idx in enumerate(idxs):
            n12 = self.get_point_normal(idx)
            self.pt_normals[nn] = n12

        # repeat into multidimensional array
        self.x = np.array(self.x.tolist() * M)
        self.pt_normals = np.array(self.pt_normals.tolist() * M)

        # convert velocity from eV to m/s
        self.init_v = np.sqrt(2 * init_v * q0 / m0)
        self.u = self.pt_normals * self.init_v

        self.phi = np.atleast_2d(np.repeat(phi, self.len)).T

        cmap = self.get_cmap(len(self.x), self.cmap)
        self.colors = np.array([cmap(i) for i in range(len(self.x))])

        #         self.len = self.x[M,S,:]

        self.x_old, self.u_old, self.phi_old = np.zeros(self.x.shape), np.zeros(self.u.shape), np.zeros(
            self.phi.shape)  # <- hold solution from prev time step
        self.x_temp, self.u_temp, self.phi_temp = np.zeros(self.x.shape), np.zeros(self.u.shape), np.zeros(
            self.phi.shape)  # <- hold tentative change in position

        # record initial particles state
        self.x_init = copy.deepcopy(self.x)
        self.u_init = copy.deepcopy(self.u)
        self.phi_init = copy.deepcopy(self.phi)

        # particles path
        self.paths = copy.deepcopy(np.hstack((self.x, self.phi)))
        self.paths_count = 1

        # hit energy
        self.E = [[] for ii in range(len(self.x))]
        self.n_secondaries = [[] for ii in range(len(self.x))]
        self.df_n = [[] for ii in range(len(self.x))]
        # exact impact point and RF phase (w*t + phi) of every counted hit;
        # the distance function d_n compares the n-th of these with the
        # initial (site, phase)
        self.impact_x = [[] for ii in range(len(self.x))]
        self.impact_phi = [[] for ii in range(len(self.x))]

        self.record = [self.x]
        self.lost_particles = []
        self.nhit = np.zeros(len(self.x))

        self.bright_set = []
        self.shadow_set = []
        # per-bright-particle impact history, archived when a particle reaches
        # 20 hits (aligned with bright_set). Ef20 / e20 metrics are computed
        # from these, matching the paper's semantics (zero outside the band).
        self.bright_E = []
        self.bright_n_secondaries = []
        self.bright_impact_x = []
        self.bright_impact_phi = []
        # initial position and phase of each bright particle (aligned with
        # bright_set) -- identity is otherwise lost at removal; needed for the
        # distance-function (d20) map and per-site statistics
        self.bright_init_x = []
        self.bright_init_phi = []

    def save_old(self):
        self.x_old = copy.deepcopy(self.x)
        self.u_old = copy.deepcopy(self.u)
        self.phi_old = copy.deepcopy(self.phi)

        self.x_temp = copy.deepcopy(self.x)
        self.u_temp = copy.deepcopy(self.u)
        self.phi_temp = copy.deepcopy(self.phi)

    def distance(self, n):
        # KD-tree nearest-neighbour query. Returns the same n nearest surface
        # indices (ascending by distance) as the previous brute-force
        # argsort, but avoids materialising the full distance matrix.
        n = min(n, len(self.bounds))
        dists, idxs = self._bounds_tree.query(self.x, k=n)
        if n == 1:
            dists = dists[:, None]
            idxs = idxs[:, None]
        return np.atleast_2d(dists[:, 0]).T, idxs.tolist()

    def remove(self, ind, bright='no'):
        ind = list(set(ind))
        if bright != 'yes':
            # add to shadow set before removal from main set
            self.shadow_set.append(self.paths[[ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)]])

        self.paths = np.delete(self.paths, [ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)], axis=0)

        self.x = np.delete(self.x, ind, axis=0)
        self.u = np.delete(self.u, ind, axis=0)

        self.x_old = np.delete(self.x_old, ind, axis=0)
        self.u_old = np.delete(self.u_old, ind, axis=0)

        self.phi = np.delete(self.phi, ind, axis=0)

        self.x_temp = np.delete(self.x_temp, ind, axis=0)
        self.u_temp = np.delete(self.u_temp, ind, axis=0)

        self.x_init = np.delete(self.x_init, ind, axis=0)
        self.u_init = np.delete(self.u_init, ind, axis=0)
        self.phi_init = np.delete(self.phi_init, ind, axis=0)

        self.colors = np.delete(self.colors, ind, axis=0)

        self.len = len(self.x)

        # print number of hits of particle before deleting
        self.nhit = np.delete(self.nhit, ind, axis=0)

        # The impact-history lists MUST shrink together with the particle
        # arrays for every removal (bright or lost) -- otherwise the indices of
        # all surviving particles shift and subsequent E[ind].append() writes
        # into the wrong particle's history. That misalignment accumulated
        # impacts from many different particles into single lists, exploding
        # the e20/c0 product metric and misattributing final impact energies.
        # (Bright particles' histories are archived in update_hit_count before
        # this is called.)
        indicesList = sorted(ind, reverse=True)
        for indx in indicesList:
            if indx < len(self.E):
                # removing element by index using pop() function
                self.E.pop(indx)
                self.n_secondaries.pop(indx)
                self.impact_x.pop(indx)
                self.impact_phi.pop(indx)

    def colors(self):
        return self.colors

    def set_cmap(self, cmap):
        self.cmap = cmap
        cmap = self.get_cmap(len(self.x), cmap)
        self.colors = np.array([cmap(i) for i in range(len(self.x))])

    def get_point_normal(self, idx):
        # calculate normal as average of connecting edge normals
        # assumpution is that the surface points are ordered in increasing x
        x0, y0 = self.bounds[idx[0] - 1]
        x1, y1 = self.bounds[idx[0]]
        x2, y2 = self.bounds[idx[0] + 1]

        dx1, dy1 = x1 - x0, y1 - y0
        dx2, dy2 = x2 - x1, y2 - y1

        n1 = -np.array([-dy1, dx1])
        n2 = -np.array([-dy2, dx2])
        n12 = n1 + n2

        return n12 / np.linalg.norm(n12)

    def update_record(self):
        self.record.append(self.x)

    def update_hit_count(self, inds):
        removed_inds = []
        self.nhit[inds] = self.nhit[inds] + 1
        # sort inds to start deleting from largest index
        inds.sort(reverse=True)
        # check if nhit = 20 and remove from main set and add to bright set
        for ind in inds:
            if self.nhit[ind] == 20:
                self.bright_set.append(self.paths[[ii * len(self.x) + np.array(ind) for ii in range(self.paths_count)]])
                # archive this particle's impact history and initial identity
                # (aligned with bright_set) before it is removed from the live
                # arrays
                self.bright_E.append(self.E[ind])
                self.bright_n_secondaries.append(self.n_secondaries[ind])
                self.bright_impact_x.append(self.impact_x[ind])
                self.bright_impact_phi.append(self.impact_phi[ind])
                self.bright_init_x.append(self.x_init[ind].copy())
                self.bright_init_phi.append(float(self.phi_init[ind, 0]))
                # remove the index from main set
                self.remove([ind], bright='yes')
                removed_inds.append(ind)
        return removed_inds

    @staticmethod
    def get_cmap(n, name='jet'):
        """
        Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.

        Parameters
        ----------
        n: int
            Number of colors to be returned
        name: str
            Name of color map

        Returns
        -------

        """
        return plt.get_cmap(name, n)

    @staticmethod
    def _select_values_with_step(values, step):
        selected_values = []
        last_value = values[0][0] - step
        for value in values:
            if value[0] >= last_value + step:
                selected_values.append(value)
                last_value = value[0]

        return np.array(selected_values)

    # def show_initial_points(self, xrange, step=None):
    #     """
    #
    #     Parameters
    #     ----------
    #     xrange: list, ndarray
    #         Interval of initial surface points
    #     step: int
    #         Minimum distance between initial surface points
    #
    #     Returns
    #     -------
    #
    #     """
    #     pts = self.bounds[(self.bounds[:, 0] > xrange[0]) & (self.bounds[:, 0] < xrange[1])]
    #
    #     if step:
    #         pts = self._select_values_with_step(pts, step)
    #
    #     self.ax.plot(self.bounds[:, 0], self.bounds[:, 1])
    #     self.ax.scatter(pts[:, 0], pts[:, 1], fc='None', ec='k', s=50)
    #     plt.show()
