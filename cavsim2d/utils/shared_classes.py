import itertools
import json
import math
import os
import pickle
import time
import warnings
from collections import defaultdict
from itertools import product
from SALib.analyze import sobol
from scipy.special import roots_legendre
import matplotlib
import numpy as np
import scipy as sp
from scipy.special import legendre
# from SALib.analyze import sobol, delta
# from SALib.test_functions import Ishigami
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
from typing import Dict
from scipy.stats import qmc
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor

from cavsim2d.utils.printing import *


class SampleGenerator:
    def __init__(self):
        pass

    def random_sample(self, problem, N, calc_second_order=False):
        D = problem["num_vars"]
        np.random.seed(3794)
        sampler = np.random.rand
        X = self.cross_A_B(N, problem, sampler, kind='numpy')

        return X

    def halton_sample(self, problem, N, calc_second_order=False):
        D = problem["num_vars"]
        sampler = qmc.Halton(d=2 * D)
        _ = sampler.reset()
        # base_sequence = np.hstack((sampler.random(N), sampler.random(N)))
        # base_sequence = np.hstack((latin.sample(problem, N), latin.sample(problem, N)))
        X = self.cross_A_B(N, problem, sampler)

        return X

    def lhs_sample(self, problem, N, calc_second_order=False, seed=3794):
        D = problem["num_vars"]
        sampler = qmc.LatinHypercube(d=2 * D, seed=seed)
        _ = sampler.reset()
        # base_sequence = np.hstack((sampler.random(N), sampler.random(N)))
        # base_sequence = np.hstack((latin.sample(problem, N), latin.sample(problem, N)))
        X = self.cross_A_B(N, problem, sampler)

        return X

    def sobol_sample(self, problem, N, calc_second_order=False):
        D = problem["num_vars"]
        sampler = qmc.Sobol(d=2 * D, scramble=False)
        _ = sampler.reset()

        # base_sequence = np.hstack((sampler.random(N+16), sampler.random(N+16)))

        X = self.cross_A_B(N, problem, sampler)
        return X

    def _nonuniform_scale_samples(self, params, bounds, dists):
        """Rescale samples in 0-to-1 range to other distributions

        Parameters
        ----------
        params : numpy.ndarray
            numpy array of dimensions num_params-by-N,
            where N is the number of samples
        dists : list
            list of distributions, one for each parameter
                unif: uniform with lower and upper bounds
                logunif: logarithmic uniform with lower and upper bounds
                triang: triangular with lower and upper bounds, as well as
                        location of peak
                        The location of peak is in percentage of width
                        e.g. :code:`[1.0, 3.0, 0.5]` indicates 1.0 to 3.0 with a
                        peak at 2.0

                        A soon-to-be deprecated two-value format assumes the lower
                        bound to be 0
                        e.g. :code:`[3, 0.5]` assumes 0 to 3, with a peak at 1.5
                norm: normal distribution with mean and standard deviation
                truncnorm: truncated normal distribution with upper and lower
                        bounds, mean and standard deviation
                lognorm: lognormal with ln-space mean and standard deviation
        """
        b = np.array(bounds, dtype=object)

        # initializing matrix for converted values
        conv_params = np.empty_like(params)

        # loop over the parameters
        for i in range(conv_params.shape[1]):
            # setting first and second arguments for distributions
            b1 = b[i][0]  # ending
            b2 = b[i][1]  # 0-1

            if dists[i] == "triang":
                if len(b[i]) == 3:
                    loc_start = b[i][0]  # loc start
                    b1 = b[i][1]  # triangular distribution end
                    b2 = b[i][2]  # 0-1 aka c (the peak)
                elif len(b[i]) == 2:
                    msg = (
                        "Two-value format for triangular distributions detected.\n"
                        "To remove this message, specify the distribution start, "
                        "end, and peak (three values) "
                        "instead of the current two-value format "
                        "(distribution end and peak, with start assumed to be 0)\n"
                        "The two-value format will be deprecated in SALib v1.5.1"
                    )
                    warnings.warn(msg, DeprecationWarning, stacklevel=2)

                    loc_start = 0
                    b1 = b[i][0]
                    b2 = b[i][1]
                else:
                    raise ValueError(
                        "Unknown triangular distribution specification. Check"
                        " problem specification."
                    )

                # checking for correct parameters
                if b1 < 0 or b2 < 0 or b2 >= 1 or loc_start > b1:
                    raise ValueError(
                        """Triangular distribution bound error: Scale must be
                        greater than zero; peak on interval [0,1], triangular
                        start value must be smaller than end value"""
                    )
                else:
                    conv_params[:, i] = sp.stats.triang.ppf(
                        params[:, i], c=b2, scale=b1 - loc_start, loc=loc_start
                    )

            elif dists[i] == "unif":
                if b1 >= b2:
                    raise ValueError(
                        """Uniform distribution: lower bound
                        must be less than upper bound"""
                    )
                else:
                    conv_params[:, i] = params[:, i] * (b2 - b1) + b1

            elif dists[i] == "logunif":
                conv_params[:, i] = sp.stats.loguniform.ppf(params[:, i], a=b1, b=b2)

            elif dists[i] == "norm":
                if b2 <= 0:
                    raise ValueError("""Normal distribution: stdev must be > 0""")
                else:
                    conv_params[:, i] = sp.stats.norm.ppf(params[:, i], loc=b1, scale=b2)

            # Truncated normal distribution
            # parameters are lower bound and upper bound, mean and stdev
            elif dists[i] == "truncnorm":
                b3 = b[i][2]
                b4 = b[i][3]
                if b4 <= 0:
                    raise ValueError(
                        """Truncated normal distribution: stdev must
                        be > 0"""
                    )
                if b1 >= b2:
                    raise ValueError(
                        """Truncated normal distribution: lower bound
                        must be less than upper bound"""
                    )
                else:
                    conv_params[:, i] = sp.stats.truncnorm.ppf(
                        params[:, i], (b1 - b3) / b4, (b2 - b3) / b4, loc=b3, scale=b4
                    )

            # lognormal distribution (ln-space, not base-10)
            # parameters are ln-space mean and standard deviation
            elif dists[i] == "lognorm":
                # checking for valid parameters
                if b2 <= 0:
                    raise ValueError("""Lognormal distribution: stdev must be > 0""")
                else:
                    conv_params[:, i] = np.exp(
                        sp.stats.norm.ppf(params[:, i], loc=b1, scale=b2)
                    )

            else:
                valid_dists = ["unif", "triang", "norm", "truncnorm", "lognorm"]
                raise ValueError("Distributions: choose one of %s" % ", ".join(valid_dists))

        return conv_params

    @staticmethod
    def compute_groups_matrix(groups: List):
        """Generate matrix which notes factor membership of groups

        Computes a k-by-g matrix which notes factor membership of groups
        where:
            k is the number of variables (factors)
            g is the number of groups
        Also returns a g-length list of unique group_names whose positions
        correspond to the order of groups in the k-by-g matrix

        Parameters
        ----------
        groups : List
            Group names corresponding to each variable

        Returns
        -------
        tuple
            containing group matrix assigning parameters to
            groups and a list of unique group names
        """
        num_vars = len(groups)
        unique_group_names = pd.unique(np.array(groups))
        number_of_groups = len(unique_group_names)

        indices = dict([(x, i) for (i, x) in enumerate(unique_group_names)])

        output = np.zeros((num_vars, number_of_groups), dtype=int)

        for parameter_row, group_membership in enumerate(groups):
            group_index = indices[group_membership]
            output[parameter_row, group_index] = 1

        return output, unique_group_names

    @staticmethod
    def _check_bounds(bounds):
        """Check user supplied distribution bounds for validity.

        Parameters
        ----------
        problem : dict
            The problem definition

        Returns
        -------
        tuple : containing upper and lower bounds
        """
        b = np.array(bounds)

        lower_bounds = b[:, 0]
        upper_bounds = b[:, 1]

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError("Bounds are not legal")

        return lower_bounds, upper_bounds

    def _scale_samples(self, params: np.ndarray, bounds: List):
        """Rescale samples in 0-to-1 range to arbitrary bounds

        Parameters
        ----------
        params : numpy.ndarray
            numpy array of dimensions `num_params`-by-:math:`N`,
            where :math:`N` is the number of samples

        bounds : list
            list of lists of dimensions `num_params`-by-2
        """
        # Check bounds are legal (upper bound is greater than lower bound)
        lower_bounds, upper_bounds = self._check_bounds(bounds)

        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Bounds are not legal (upper bound must be greater than lower bound)"
            )

        # This scales the samples in-place, by using the optional output
        # argument for the numpy ufunctions
        # The calculation is equivalent to:
        #   sample * (upper_bound - lower_bound) + lower_bound
        np.add(
            np.multiply(params, (upper_bounds - lower_bounds), out=params),
            lower_bounds,
            out=params,
        )

    def scale_samples(self, params: np.ndarray, problem: Dict):
        """Scale samples based on specified distribution (defaulting to uniform).

        Adds an entry to the problem specification to indicate samples have been
        scaled to maintain backwards compatibility (`sample_scaled`).

        Parameters
        ----------
        params : np.ndarray,
            numpy array of dimensions `num_params`-by-:math:`N`,
            where :math:`N` is the number of samples
        problem : dictionary,
            SALib problem specification

        Returns
        -------
        np.ndarray, scaled samples
        """
        bounds = problem["bounds"]
        dists = problem.get("dists")

        if dists is None:
            self._scale_samples(params, bounds)
        else:
            if params.shape[1] != len(dists):
                msg = "Mismatch in number of parameters and distributions.\n"
                msg += "Num parameters: {}".format(params.shape[1])
                msg += "Num distributions: {}".format(len(dists))
                raise ValueError(msg)

            params = self._nonuniform_scale_samples(params, bounds, dists)

        problem["sample_scaled"] = True

        return params
        # limited_params = limit_samples(params, upper_bound, lower_bound, dists)

    @staticmethod
    def _check_groups(problem):
        """Check if there is more than 1 group."""
        groups = problem.get("groups")
        if not groups:
            return False
        if groups == problem["names"]:
            return False

        if len(set(groups)) == 1:
            return False

        return groups

    def cross_A_B(self, N, problem, sampler, skip_values=None, kind=None):
        if skip_values is None:
            # If not specified, set skip_values to next largest power of 2
            skip_values = int(2 ** math.ceil(math.log(N) / math.log(2)))

            # 16 is arbitrarily selected here to avoid initial points
            # for very low sample sizes
            skip_values = max(skip_values, 16)

        elif skip_values > 0:
            M = skip_values
            if not ((M & (M - 1) == 0) and (M != 0 and M - 1 != 0)):
                msg = f"""
                Convergence properties of the Sobol' sequence is only valid if
                `skip_values` ({M}) is a power of 2.
                """
                print(msg)

            # warning when N > skip_values
            # see https://github.com/scipy/scipy/pull/10844#issuecomment-673029539
            n_exp = int(math.log(N, 2))
            m_exp = int(math.log(M, 2))
            if n_exp > m_exp:
                msg = (
                    "Convergence may not be valid as the number of "
                    "requested samples is"
                    f" > `skip_values` ({N} > {M})."
                )
                print(msg)
        elif skip_values == 0:
            print("Duplicate samples will be taken as no points are skipped.")
        else:
            assert (
                    isinstance(skip_values, int) and skip_values >= 0
            ), "`skip_values` must be a positive integer."

        D = problem["num_vars"]
        groups = self._check_groups(problem)

        if not groups:
            Dg = problem["num_vars"]
        else:
            G, group_names = self.compute_groups_matrix(groups)
            Dg = len(set(group_names))

        if kind is not None:
            base_sequence = sampler(N + skip_values, 2 * D)
        else:
            base_sequence = sampler.random(N + skip_values)

        saltelli_sequence = np.zeros([(D + 2) * N, D])
        index = 0

        for i in range(skip_values, N + skip_values):
            # Copy matrix "A"
            for j in range(D):
                # print('\t', saltelli_sequence.shape, i, j, base_sequence.shape)
                saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

            # Cross-sample elements of "B" into "A"
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j + D]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1

            # Copy matrix "B"
            for j in range(D):
                saltelli_sequence[index, j] = base_sequence[i, j + D]

            index += 1

        saltelli_sequence = self.scale_samples(saltelli_sequence, problem)
        return saltelli_sequence


class Cubature:
    def __init__(self, problem):
        self.unscaled_nodes = None
        self.nodes = np.array([])
        self.weights = np.array([])
        self.physical_nodes = np.array([])

        self.problem = problem
        self.dim = problem['num_vars']
        self.names = problem['names']
        self.bounds = problem['bounds']

        self.interval = [-1, 1]
        self.method = ''

    def stroud3(self, degree=1):
        """
        Stroud-3 quadrature in :math:`[0,1]^k`

        .. note::

            Dimensional Threshold Limitation: In practice, the Stroud 3 quadrature rule may be effective in dimensions
            up to around 3 to 6, depending on the specific problem and the function being integrated. Beyond this,
            the accuracy of the rule typically degrades, and higher-order quadrature rules or
            Monte Carlo methods might be more appropriate.

        Parameters
        ----------
        self.dim: int
            Dimension of variables
        degree: int
            Degree

        Returns
        -------
        Nodes and corresponding weights
        """

        self.method = 'stroud3'
        # data for Stroud-3 quadrature in [0,1]**k
        # nodes and weights
        nodes = self.stroud(self.dim)

        # weights = (1 / (2 * self.dim)) * np.ones((2 * self.dim, 1))
        weights = np.ones((2 * self.dim)) * (2 ** self.dim / (2 * self.dim))

        self.nodes = nodes
        self.weights = weights
        self.unscaled_nodes = self.nodes
        self.scale()
        # print('SCALED NODES')
        # print(self.nodes)
        # print()
        return self.nodes, self.weights

    def stroud5(self):
        self.method = 'stroud5'
        self.nodes, self.weights = self.cn_leg_05_2(self.dim)

        self.unscaled_nodes = self.nodes
        print(self.nodes.shape, self.weights.shape)
        self.scale()
        # print('SCALED NODES')
        # print(self.nodes)
        # print()
        return self.nodes, self.weights

    def gauss(self, degree=2):
        self.method = f'gauss{degree}'
        self.nodes, self.weights = self.cn_gauss(degree)
        # transform nodes from [-1,+1]**p to [0,1]**p
        self.unscaled_nodes = self.nodes
        # print(self.nodes.shape, self.weights.shape)
        self.scale()
        # print('SCALED NODES')
        # print(self.nodes)
        # print()
        return self.nodes, self.weights

    @staticmethod
    def stroud(p):
        """
        Stroud-3 method

        Parameters
        ----------
        p: int
            Dimension

        Returns
        -------
        Nodes of quadrature rule in [0,1]**p (column-wise)
        """
        # Stroud-3 method
        #
        # Input parameters:
        #  p   number of dimensions
        # Output parameters:
        #  nodes   nodes of quadrature rule in [0,1]**p (column-wise)
        #

        nodes = np.zeros((p, 2 * p))
        coeff = np.pi / p
        fac = np.sqrt(2 / 3)

        for i in range(2 * p):
            for r in range(int(np.floor(0.5 * p))):
                k = 2 * r
                nodes[k, i] = fac * np.cos((k + 1) * (i + 1) * coeff)
                nodes[k + 1, i] = fac * np.sin((k + 1) * (i + 1) * coeff)

            if 0.5 * p != np.floor(0.5 * p):
                nodes[-1, i] = ((-1) ** (i + 1)) / np.sqrt(3)

        # transform nodes from [-1,+1]**p to [0,1]**p
        # nodes = 0.5 * nodes + 0.5
        # print('UNSCALED NODES')
        # print(nodes.T)
        # print()
        return nodes.T

    def scale(self):
        """
        Scales each column of `nodes` from [0,1] to the specified range in `scales`.

        Parameters
        ----------
        nodes : np.ndarray
            The output array from `stroud(p)`, shape (n_samples, self.dim).
        scales : dict
            Dictionary where keys are parameter names and values are [lower, upper] bounds.

        Returns
        -------
        np.ndarray
            Scaled array with values in the specified ranges.
        """

        scaled_nodes = np.zeros_like(self.nodes)
        scale_values = self.bounds

        for i in range(self.nodes.shape[1]):  # Iterate over columns
            a, b = scale_values[i]  # Get corresponding scaling range
            scaled_nodes[:, i] = a + (self.nodes[:, i] + 1) / 2 * (b - a)  # Scale to [a, b]

        self.nodes = scaled_nodes
        # return scaled_nodes

    @staticmethod
    def c1_leg_monomial_integral(expon):
        if expon < 0:
            error("\n")
            error("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")
            error("EXPON < 0.")
            raise ValueError("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")

        if expon % 2 == 1:
            return 0.0

        value = 2.0 / (expon + 1)
        return value

    def cn_leg_05_1(self, n, option=1):
        """
        The rule has order

        O = 2 N^2 + N + 2.

        The rule has precision P = 5.

        CN_LEG is the cube [-1,+1]^N with the Legendre weight function

        w(x) = 1.

        .. note::

            Dimensional Threshold Limitation: In practice, the Stroud 3 quadrature rule may be effective in dimensions
            up to around 3 to 6, depending on the specific problem and the function being integrated. Beyond this,
            the accuracy of the rule typically degrades, and higher-order quadrature rules or
            Monte Carlo methods might be more appropriate.

        Parameters
        ----------
        n
        option

        Returns
        -------

        """
        # Check if the value of n is 4, 5, or 6
        if n not in [4, 5, 6]:
            error("\n")
            error("CN_LEG_05_1 - Fatal error!")
            error("The value of N must be 4, 5, or 6.")
            raise ValueError("CN_LEG_05_1 - Fatal error!")

        # Check for valid option when n = 4 or 5
        if n in [4, 5] and option not in [1, 2]:
            error("\n")
            error("CN_LEG_05_1 - Fatal error!")
            error("When N = 4 or 5, OPTION must be 1 or 2.")
            raise ValueError("CN_LEG_05_1 - Fatal error!")

        o = n ** 2 + n + 2
        w = np.zeros(o)
        x = np.zeros((n, o))

        expon = 0
        volume = self.c1_leg_monomial_integral(expon)
        volume = volume ** n

        if (n == 4 and option == 1):
            eta = 0.778984505799815
            lmbda = 1.284565137874656
            xsi = -0.713647298819253
            mu = -0.715669761974162
            gamma = 0.217089151000943
            a = 0.206186096875899e-01 * volume
            b = 0.975705820221664e-02 * volume
            c = 0.733921929172573e-01 * volume
        elif (n == 4 and option == 2):
            eta = 0.546190755827425E+00
            lmbda = 0.745069130115661E+00
            xsi = - 0.413927294508700E+00
            mu = - 0.343989637454535E+00
            gamma = 1.134017894600344E+00
            a = 0.853094758323323E-01 * volume
            b = 0.862099000096395E-01 * volume
            c = 0.116418206881849E-01 * volume
        elif (n == 5 and option == 1):
            eta = 0.522478547481276E+00
            lmbda = 0.936135175985774E+00
            xsi = - 0.246351362101519E+00
            mu = - 0.496308106093758E+00
            gamma = 0.827180176822930E+00
            a = 0.631976901960153E-01 * volume
            b = 0.511464127430166E-01 * volume
            c = 0.181070246088902E-01 * volume
        elif (n == 5 and option == 2):
            eta = 0.798317301388741E+00
            lmbda = 0.637344273885728E+00
            xsi = - 0.455245909918377E+00
            mu = - 1.063446229997311E+00
            gamma = 0.354482076665770E+00
            a = 0.116952384292206E-01 * volume
            b = 0.701731258612708E-01 * volume
            c = 0.137439132264426E-01 * volume
        else:
            eta = 0.660225291773525E+00
            lmbda = 1.064581294844754E+00
            xsi = 0.000000000000000E+00
            mu = - 0.660225291773525E+00
            gamma = 0.660225291773525E+00
            a = 0.182742214532872E-01 * volume
            b = 0.346020761245675E-01 * volume
            c = 0.182742214532872E-01 * volume

        # Set x and w based on parameters
        k = 0
        # k += 1
        for i in range(n):
            x[i, k] = eta
        w[k] = a

        # k += 1
        for i in range(n):
            x[i, k] = -eta
        w[k] = a

        for i1 in range(n):
            for i in range(1, n):
                x[i, k] = xsi
            x[i1, k] = lmbda
            w[k] = b
            k = k + 1

        for i1 in range(n):
            for i in range(n):
                x[i, k] = - xsi
            x[i1, k] = - lmbda
            w[k] = b
            k = k + 1

        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                for i in range(n):
                    x[i, k] = gamma
                x[i1, k] = mu
                x[i2, k] = mu
                w[k] = c
                k = k + 1

        for i1 in range(n - 1):
            for i2 in range(i1 + 1, n):
                for i in range(n):
                    x[i, k] = - gamma
                x[i1, k] = - mu
                x[i2, k] = - mu
                w[k] = c
                k = k + 1

        return x, np.atleast_2d(w).T / np.sum(w)

    def cn_leg_05_2(self, n):
        """
        The rule has order

        O = 2 N^2 + 1.

        The rule has precision P = 5.

        CN_LEG is the cube [-1,+1]^N with the Legendre weight function

        w(x) = 1.

        .. note::

            Dimensional Threshold Limitation: In practice, the Stroud 5 quadrature rule may be effective in dimensions
            up to around 5 to 10, depending on the specific problem and the function being integrated. Beyond this,
            the accuracy of the rule typically degrades, and higher-order quadrature rules or
            Monte Carlo methods might be more appropriate.

        Parameters
        ----------
        n
        option

        Returns
        -------

        """
        if n < 2:
            error("CN_LEG_05_2 - Fatal error!")
            error("N must be at least 2.")
            raise ValueError("CN_LEG_05_2 - Fatal error!")

        o = 2 * n ** 2 + 1
        w = np.zeros(o, dtype=np.float64)
        x = np.zeros((n, o), dtype=np.float64)

        expon = 0
        volume = self.c1_leg_monomial_integral(expon)
        volume = volume ** n

        b0 = (25 * n * n - 115 * n + 162) * volume / 162.0
        b1 = (70 - 25 * n) * volume / 162.0
        b2 = 25.0 * volume / 324.0

        r = np.sqrt(3.0 / 5.0)

        k = 0
        for i in range(n):
            x[i, k] = 0.0
        w[k] = b0

        for i1 in range(1, n + 1):
            k += 1
            for i in range(n):
                x[i, k] = 0.0
            x[i1 - 1, k] = +r
            w[k] = b1

            k += 1
            for i in range(n):
                x[i, k] = 0.0
            x[i1 - 1, k] = -r
            w[k] = b1

        for i1 in range(1, n):
            for i2 in range(i1 + 1, n + 1):
                k += 1
                for i in range(n):
                    x[i, k] = 0.0
                x[i1 - 1, k] = +r
                x[i2 - 1, k] = +r
                w[k] = b2

                k += 1
                for i in range(n):
                    x[i, k] = 0.0
                x[i1 - 1, k] = +r
                x[i2 - 1, k] = -r
                w[k] = b2

                k += 1
                for i in range(n):
                    x[i, k] = 0.0
                x[i1 - 1, k] = -r
                x[i2 - 1, k] = +r
                w[k] = b2

                k += 1
                for i in range(n):
                    x[i, k] = 0.0
                x[i1 - 1, k] = -r
                x[i2 - 1, k] = -r
                w[k] = b2

        # print('UNSCALED NODES')
        # print(x.T)
        # print()
        return x.T, w

    def cn_gauss(self, n_list):
        """
        Builds a tensor-product Gauss-Legendre rule in p dimensions,
        allowing different number of Gauss points and different bounds per dimension.

        Parameters
        ----------
        n_list : list of int
            Number of Gauss-Legendre points for each dimension.
            len(n_list) = p (number of dimensions).
        bounds : list of (a_i, b_i) pairs
            The integration bounds for each dimension.
            len(bounds) must also be p.

        Returns
        -------
        nodes : np.ndarray of shape (N, p)
            The tensor-product nodes in the multi-dimensional domain.
            N = n1 * n2 * ... * np (total number of nodes).
        weights : np.ndarray of shape (N,)
            The corresponding weights for each node, including Jacobian factors.
        """

        if isinstance(n_list, int):
            n_list = [n_list] * self.dim

        if len(n_list) != self.dim:
            raise ValueError("n_list and bounds must have the same length (p dimensions).")

        # Store transformed nodes and weights for each dimension
        dim_nodes = []
        dim_weights = []

        for i, (a_i, b_i) in enumerate(self.bounds):
            n_i = n_list[i]  # Number of nodes in this dimension
            s_1D, w_1D = roots_legendre(n_i)  # Gauss-Legendre nodes/weights on [-1,1]

            # # Transform nodes to [a_i, b_i]
            # x_i = 0.5 * (b_i - a_i) * s_1D + 0.5 * (a_i + b_i)
            #
            # # Adjust weights with Jacobian factor
            # w_i = 0.5 * (b_i - a_i) * w_1D

            # dim_nodes.append(x_i)
            # dim_weights.append(w_i)

            dim_nodes.append(s_1D)
            dim_weights.append(w_1D)

        # Compute the total number of nodes in the tensor-product grid
        N = np.prod(n_list)
        nodes = np.zeros((N, self.dim))
        weights = np.zeros(N)

        index = 0
        # Iterate over the Cartesian product of indices (i_1, i_2, ..., i_p)
        for combo in product(*[range(n_i) for n_i in n_list]):
            coord = []
            wprod = 1.0
            for dim_idx, i_k in enumerate(combo):
                coord.append(dim_nodes[dim_idx][i_k])  # Pick node for this dimension
                wprod *= dim_weights[dim_idx][i_k]  # Multiply corresponding weight

            nodes[index, :] = coord
            weights[index] = wprod
            index += 1
        # print('UNSCALED NODES')
        # print(nodes)
        # print()
        return nodes, weights

    def estimate_sobol_indices(self, Y):
        """
        Estimates first-order Sobol indices using the Stroud-3 cubature rule.

        Parameters
        ----------
        Y : np.ndarray
            Function evaluations at quadrature nodes.
            Shape: (n_nodes, num_objectives), where n_nodes is the number of nodes and
            num_objectives is the number of objective functions.

        Returns
        -------
        S : np.ndarray
            First-order Sobol indices for each objective function,
            shape (p, num_objectives), where p is the number of input dimensions.
        """
        # Ensure weights is a 1D array. If self.weights is not 1D, reshape it.
        weights = self.weights
        if weights.ndim != 1:
            weights = weights.reshape(-1)  # now shape (n_nodes,)

        # Compute total mean and variance for each objective function (averaging over nodes)
        mean = np.average(Y, weights=weights, axis=0)  # shape: (num_objectives,)
        var = np.average((Y - mean) ** 2, weights=weights, axis=0)  # shape: (num_objectives,)

        print("Mean:", mean)
        print("Variance:", var)

        # If Y is one-dimensional (a single objective), convert to 2D (n_nodes, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        num_objectives = Y.shape[1]

        # Number of input dimensions is taken from self.nodes
        p = self.nodes.shape[1]
        # Initialize Sobol indices array for each input (p) and each objective
        S = np.zeros((p, num_objectives))

        # Loop over each input dimension
        for i in range(p):
            # Find the unique values in the i-th column of nodes
            unique_xi = np.unique(self.nodes[:, i])
            n_unique = len(unique_xi)
            # To store the conditional means and the aggregated weights for each unique value
            conditional_means = np.zeros((n_unique, num_objectives))
            weight_xi = np.zeros(n_unique)

            # For each unique value of X_i, compute the conditional mean of Y and sum of weights
            for j, xi in enumerate(unique_xi):
                mask = np.isclose(self.nodes[:, i], xi)
                weight_xi[j] = np.sum(weights[mask])
                for k in range(num_objectives):
                    conditional_means[j, k] = np.sum(weights[mask] * Y[mask, k]) / weight_xi[j]

            # Compute Sobol indices for each objective:
            # variance of the conditional expectation divided by total variance.
            for k in range(num_objectives):
                S[i, k] = (np.sum(weight_xi * conditional_means[:, k] ** 2) - mean[k] ** 2) / var[k]

        return S

    def weighted_mean_obj(self, tab_var, weights):
        print(weights, weights.shape)
        print(tab_var, tab_var.shape)
        rows_sims_no, cols = np.shape(tab_var)
        no_weights, dummy = np.shape(weights)
        if rows_sims_no == no_weights:
            # expe = np.zeros((cols, 1))
            # outvar = np.zeros((cols, 1))
            # for i in range(cols):
            #     expe[i, 0] = np.dot(tab_var[:, i], weights)[0]
            #     outvar[i, 0] = np.dot(tab_var[:, i] ** 2, weights)[0]
            #
            # stdDev = np.sqrt(abs(outvar - expe ** 2))

            mean = self.weighted_mean(tab_var, weights.T[0])
            std = np.sqrt(self.weighted_variance(tab_var, weights.T[0], mean))
            skew = self.weighted_skew(tab_var, weights.T[0], mean, std)
            kurtosis = self.weighted_kurtosis(tab_var, weights.T[0], mean, std)
            print('statistical moments')
            print(mean.reshape(-1, 1), std.reshape(-1, 1), skew, kurtosis)
        else:
            mean = 0
            std = 0
            skew = 0
            kurtosis = 0
            error('Cols_sims_no != No_weights')
        return list(mean), list(std), list(skew), list(kurtosis)

    @staticmethod
    def weighted_mean(var, wts):
        """Calculates the weighted mean"""
        return np.average(var, weights=wts, axis=0)

    @staticmethod
    def weighted_variance(var, wts, mean):
        """Calculates the weighted variance"""
        return np.average((var - mean) ** 2, weights=wts, axis=0)

    @staticmethod
    def weighted_skew(var, wts, mean, std):
        """Calculates the weighted skewness"""
        return (np.average((var - mean) ** 3, weights=wts, axis=0) /
                std ** 3)

    @staticmethod
    def weighted_kurtosis(var, wts, mean, std):
        """Calculates the weighted skewness"""
        return (np.average((var - mean) ** 4, weights=wts, axis=0) /
                std ** 4)


class PCE:
    def __init__(self, cub, Y):
        """
        Parameters
        ----------
        cub : object with attributes:
            - nodes  (n_nodes, p): quadrature nodes, e.g. shape (6,3)
            - weights (n_nodes,1): quadrature weights
            - Y       (n_nodes,1): function values at each node
            - bounds: a list or dict of (a, b) for each dimension
        """
        # Store nodes as (n_nodes, p)
        self.expanded_coeffs = defaultdict(float)
        self.cub = cub

        self.nodes = cub.unscaled_nodes  # e.g. shape (6,3)

        # Flatten weights from (n_nodes,1) to (n_nodes,)
        if len(cub.weights.shape) == 2 and cub.weights.shape[1] == 1:
            self.weights = cub.weights.flatten()
        else:
            self.weights = cub.weights
        self.Y = Y

        self.problem = cub.problem

        # If your bounds are a dict, convert to list for indexing
        if isinstance(cub.bounds, dict):
            # Sort by key if needed, or assume an order
            self.bounds = [cub.bounds[k] for k in sorted(cub.bounds.keys())]
        else:
            self.bounds = cub.bounds

        self.dim = self.nodes.shape[1]  # number of variables
        self.pce_coeffs = {}
        self.alphas = []
        self.truncation = None

    def le(self, n, var_index):
        """
        Returns a Legendre polynomial of degree n for dimension var_index,
        scaled from [a,b] to [-1,1].
        """
        Pn = legendre(n)
        factor = np.sqrt((2 * n + 1) / 2)  # Orthonormal scaling

        # factor = 1

        # a, b = self.bounds[var_index]  # e.g. (2,4)

        def poly(x):
            # Map x from [a,b] to z in [-1,1]
            return factor * Pn(x)

        return poly

    def pce(self, degree, truncation=None):
        """
        Builds the PCE coefficients up to 'degree' for each dimension.
        """
        # Create multi-indices alpha = (alpha_1, alpha_2, ..., alpha_dim)

        if isinstance(degree, int):
            var_order = [degree] * self.dim
        else:
            var_order = degree

        if truncation is None:
            if isinstance(degree, int):
                self.truncation = self.dim * degree
            else:
                self.truncation = self.dim * np.sum(degree)
        else:
            self.truncation = truncation

        ranges = [range(d + 1) for d in var_order]
        self.alphas = list(itertools.product(*ranges))
        # e.g. for dim=3, degree=2 => (0,0,0), (0,0,1), (0,0,2), (0,1,0), ...

        # Compute coefficients
        # print(self.alphas)
        for alpha in self.alphas:
            if sum(alpha) <= self.truncation:
                coeff_num, coeff_den = 0.0, 0.0
                # Sum over nodes
                for node, w, y in zip(self.nodes, self.weights, self.Y):
                    psi_val = 1.0
                    # Multiply polynomials for each dimension
                    for var_idx, deg in enumerate(alpha):
                        psi_val *= self.le(deg, var_idx)(node[var_idx])
                    coeff_num += w * y * psi_val
                    coeff_den += w * psi_val ** 2
                    # print(coeff_num, coeff_den, psi_val**2)

                # if psi_val != 0:
                # self.pce_coeffs[alpha] = coeff_num/coeff_den
                self.pce_coeffs[f'{alpha}'] = coeff_num
                # else:
                #     self.pce_coeffs[f'{alpha}'] = 0

        self.expand_pce()
        return self

    def transform(self, x, indx):
        """
        Transform x from [a,b] to z in [-1,1].
        """
        a, b = self.bounds[indx]
        return 2 * (x - a) / (b - a) - 1

    def expand_pce(self):
        """
        Converts the PCE model into a standard polynomial representation.
        Returns a dictionary where keys are multi-indices (tuples) of exponents,
        and values are the corresponding coefficients.
        """
        for alpha in self.alphas:
            if sum(alpha) <= self.truncation:
                poly_coeffs = {tuple([0] * len(alpha)): 1.0}  # Start with constant 1

                for var_idx, deg in enumerate(alpha):
                    # Get the Legendre polynomial coefficients for the given degree
                    legendre_poly = np.sqrt((2 * deg + 1) / 2) * legendre(deg).coef[::-1]
                    # Transform polynomial representation (coefficients -> monomial terms)
                    new_poly = {}
                    for exp_tuple, coeff in poly_coeffs.items():
                        for i, c in enumerate(legendre_poly):
                            new_exp = list(exp_tuple)
                            new_exp[var_idx] += i  # Add exponent of variable index
                            new_poly[tuple(new_exp)] = new_poly.get(tuple(new_exp), 0) + coeff * c

                    poly_coeffs = new_poly

                # Multiply by PCE coefficient and accumulate
                for exp_tuple, coeff in poly_coeffs.items():
                    self.expanded_coeffs[exp_tuple] += self.pce_coeffs[f'{alpha}'] * coeff

    def evaluate(self, X):
        """
        Evaluates the expanded PCE polynomial at given input points X.

        Parameters
        ----------
        X : np.ndarray of shape (m, p)
            Input points where the polynomial should be evaluated.

        Returns
        -------
        Y : np.ndarray of shape (m,)
            Evaluated values of the polynomial.
        """
        X = np.atleast_2d(X)  # Ensure X is a 2D array
        m, p = X.shape
        Y_pred = np.zeros(m)

        # Apply the transformation
        X_transformed = np.zeros_like(X)
        for var_idx in range(p):
            X_transformed[:, var_idx] = self.transform(X[:, var_idx], var_idx)

        for exp_tuple, coeff in self.expanded_coeffs.items():
            term_values = np.ones(m) * coeff  # Start with coefficient
            for var_idx, exponent in enumerate(exp_tuple):
                term_values *= X_transformed[:, var_idx] ** exponent  # Multiply by x^exp
            Y_pred += term_values  # Sum contributions

        return np.atleast_2d(Y_pred).T

    def basis_eval(self, nodes, degree):
        """Evaluate the basis polynomials at given nodes up to a given degree."""

        num_nodes, num_vars = nodes.shape
        basis_terms = []
        alphas = []

        # Generate multi-index up to given degree
        # print(nodes.shape)
        for alpha in product(range(self.degree + 1), repeat=num_vars):
            if sum(alpha) <= self.truncation:
                alphas.append(alpha)
                term = np.prod([self.le(a, j)(nodes[:, j]) for j, a in enumerate(alpha)], axis=0)
                basis_terms.append(term)

        return np.array(basis_terms).T, alphas  # Shape: (num_nodes, num_basis)

    def pce_regression(self, degree, truncation=None, nodes=None, Y=None):
        """Compute PCE coefficients using least squares regression."""

        if truncation is None:
            self.truncation = degree
        else:
            self.truncation = truncation
        # print(nodes)
        if nodes is None:
            nodes = self.cub.nodes  # Quadrature nodes

        self.degree = degree
        start = time.time()
        basis_matrix, alphas = self.basis_eval(nodes, degree)  # Evaluate basis polynomials
        # print('time: ', time.time() - start)
        # print(basis_matrix, basis_matrix.shape)
        start = time.time()

        if Y is None:
            print('condition number', np.linalg.cond(basis_matrix))
            # print(basis_matrix, basis_matrix.shape)
            coeffs = np.linalg.pinv(basis_matrix) @ self.Y
            # print(coeffs.T)
            # print()

            poly = PolynomialFeatures(2)
            poly.fit_transform(self.nodes)
            poly.fit(self.nodes, self.Y)

            print('====' * 50)
        else:
            # print('second', Y)
            print(np.linalg.pinv(basis_matrix).shape, Y.shape)
            coeffs = np.linalg.pinv(basis_matrix) @ Y

        # print('time: ', time.time() - start)
        # coeffs, _, _, _ = np.linalg.lstsq(basis_matrix, self.Y)

        self.alphas = alphas
        # Store coefficients
        start = time.time()
        self.pce_coeffs = {f'{alphas[i]}': coeffs[i] for i in range(len(alphas))}
        # print('time: ', time.time() - start)
        # print(self.pce_coeffs)
        self.expand_pce()

    def evaluate_reg(self, test_nodes):
        """Evaluate PCE at test nodes using computed coefficients."""
        basis_matrix, _ = self.basis_eval(test_nodes, max(self.pce_coeffs.keys()))

        # print(basis_matrix.shape, np.array([self.pce_coeffs[alpha] for alpha in self.pce_coeffs]).shape)
        return basis_matrix @ np.array([self.pce_coeffs[alpha] for alpha in self.pce_coeffs])

    def metrics(self, Y=None, folder=None):
        if Y is None:
            Y_model = self.evaluate_reg(self.cub.nodes)
            Y_actual = self.Y
        else:
            Y_actual = Y

        mae = np.mean(np.abs(Y_model - Y_actual))
        mse = np.mean((Y_model - Y_actual) ** 2)
        rmse = np.sqrt(mse)
        ss_total = np.sum((Y_model - np.mean(Y_actual)) ** 2)
        ss_residual = np.sum((Y_model - Y_actual) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        stat_dict = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }
        if folder:
            with open(folder, 'w') as f:
                json.dump(fr'{folder}/stat.json', f, indent=4)

        return stat_dict

    def sobol(self, N=10000, f=None):
        sampgen = SampleGenerator()
        sampler = sampgen.lhs_sample
        X = sampler(self.problem, N, calc_second_order=False, seed=12347)

        if f is None:
            Y = self.evaluate(X)
        else:
            Y = f(X)

        num_objectives = Y.shape[1]
        print(num_objectives)
        Si_D = {}
        for obj in range(num_objectives):
            Si = sobol.analyze(self.problem, Y[:, obj], calc_second_order=False)
            Si_D[obj] = Si

        return Si_D[0], X, Y


class NeuralNet:
    def __init__(self, data, problem, obj):
        self.model = None
        self.scaler = None
        self.data = data

        self.train_size = int(len(self.data) * 0.8)
        self.test_size = len(self.data) - self.train_size

        if isinstance(problem, str):
            self.names, self.bounds = read_dakota_input_file(os.path.join(problem, 'dakota.in'))
            self.bounds = [b for _, b in self.bounds.items()]
            self.dim = len(self.names)

            self.problem = {
                'num_vars': self.dim,
                'names': self.names,
                'bounds': self.bounds
            }
        else:
            self.names = problem['names']
            self.bounds = problem['bounds']
            self.dim = len(self.names)

            self.problem = problem

        self.X_train, self.Y_train = self._train_data(obj)
        self.X_test, self.Y_test = self._test_data()

    def _train_data(self, obj):
        self.obj = obj
        # X
        self.nodes = self.data.iloc[:self.train_size, 0:self.dim].to_numpy()
        self.unscaled_nodes = self.nodes
        self.weights = np.ones(self.train_size)

        self.Y = self.data.iloc[:self.train_size][obj].to_numpy()

        return self.nodes, np.atleast_2d(self.Y).T

    def _test_data(self):
        test_data = self.data[self.train_size:]
        test_nodes = test_data.iloc[:, 0:self.dim].to_numpy()
        test_nodes_Y = np.atleast_2d(test_data[self.obj].to_numpy()).T

        return [test_nodes, test_nodes_Y]

    def train(self):
        # Step 3: Standardize Data (important for neural networks)
        self.scaler = StandardScaler()
        X_train_fit = self.scaler.fit_transform(self.X_train)

        # Step 4: Define and Train the Neural Network
        self.model = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32),  # Two hidden layers
                                  activation='tanh',  # Activation function
                                  solver='adam',  # Optimizer
                                  alpha=0.001,  # L2 regularization (prevents overfitting)
                                  max_iter=10000,  # Number of iterations
                                  random_state=42)

        self.model.fit(X_train_fit, self.Y_train.ravel())

        return self

    def test(self, x_test=None):
        if x_test is None:
            x_test = self.X_test

        X_test_fit = self.scaler.transform(x_test)
        Y_pred = self.model.predict(X_test_fit)
        # mse = mean_squared_error(self.Y_test, Y_pred)

        return Y_pred

    def func(self, X):
        return np.atleast_2d(self.model.predict(self.scaler.transform(X))).T


class CubData:
    def __init__(self, folder, problem):
        df = pd.read_excel(os.path.join(folder, 'data.xlsx'), sheet_name='Sheet1')
        # shuffle data
        self.data = df.sample(n=len(df))
        self.problem = problem
        self.bounds = problem['bounds']

        self.nvar = problem['num_vars']
        self.nodes = self.data.iloc[:, 0:self.nvar].to_numpy()

        self.weights = np.ones(len(self.nodes))
        self.unscaled_nodes = self.unscale_samples(self.nodes, problem['bounds'])
        # print(self.unscaled_nodes)

    def unscale_samples(self, samples, bounds):
        """
        Convert samples from [lower, upper] back to [0,1].

        Parameters
        ----------
        samples : ndarray or pandas.DataFrame, shape (n_samples, n_vars)
            The scaled samples in the original variable bounds.
        bounds : array‑like, shape (n_vars, 2)
            A list or array of [lower, upper] pairs for each variable.

        Returns
        -------
        unit_samples : ndarray or pandas.DataFrame, shape (n_samples, n_vars)
            The samples mapped back to the unit hypercube [0,1]^d.
        """
        lb = np.array([b[0] for b in bounds])
        ub = np.array([b[1] for b in bounds])
        span = ub - lb

        if isinstance(samples, pd.DataFrame):
            arr = samples.values
            unit = (arr - lb) / span
            return pd.DataFrame(unit, columns=samples.columns, index=samples.index)
        else:
            arr = np.asarray(samples)
            return (arr - lb) / span


class Data:
    def __init__(self, folder, problem=None):
        self.Y = None
        self.unscaled_nodes = None
        self.nodes = None
        self.obj = None
        df = pd.read_excel(os.path.join(folder, 'data.xlsx'), sheet_name='Sheet1')
        # shuffle data
        self.data = df.sample(n=len(df))

        self.train_size = int(len(self.data) * 0.9)
        self.test_size = len(self.data) - self.train_size
        # self.train_size = int(len(self.data) * 1)
        # self.test_size = len(self.data)

        if problem is None:
            self.names, self.bounds = read_dakota_input_file(os.path.join(folder, 'dakota.in'))
            self.bounds = [b for _, b in self.bounds.items()]
            self.dim = len(self.names)

            self.problem = {
                'num_vars': self.dim,
                'names': self.names,
                'bounds': self.bounds
            }
        else:
            self.names = problem['names']
            self.bounds = problem['bounds']
            self.dim = len(self.names)

            self.problem = problem

        self.count = 0
        self.weights = np.array([])

    def train_data(self, obj):
        self.obj = obj
        self.nodes = self.data.iloc[:self.train_size, 0:self.dim].to_numpy()
        self.unscaled_nodes = self.nodes
        self.weights = np.ones(self.train_size)

        self.Y = self.data.iloc[:self.train_size][obj].to_numpy()

        return self.nodes, np.atleast_2d(self.Y).T

    def test_data(self):
        test_data = self.data[self.train_size:]
        # test_data = self.data[:]
        test_nodes = test_data.iloc[:, 0:self.dim].to_numpy()
        test_nodes_Y = np.atleast_2d(test_data[self.obj].to_numpy()).T

        return [test_nodes, test_nodes_Y]


class Sobol:
    def __init__(self, problem, folder):
        self.problem = problem
        self.folder = folder
        self.sobol_indices = {}

    def analyse(self, f, label, N=10000):
        sampgen = SampleGenerator()
        sampler = sampgen.lhs_sample
        X = sampler(self.problem, N, calc_second_order=False, seed=12347)
        Y = f(X)
        plt.savefig
        Si = sobol.analyze(self.problem, Y[:, 0], calc_second_order=False)

        self.sobol_indices[label] = Si
        return Si, X, Y

    def _dict_to_dataframe(self, data: dict, key: str, normalize: bool = False, indx=[]) -> pd.DataFrame:
        """
        Converts nested dictionary to a DataFrame using the selected inner key.
        Values less than 1 are set to 0.
        Optionally normalises each column to sum to 1.

        Parameters:
        - data (dict): The nested dictionary.
        - key (str): The key to extract values for (e.g., 'S1').
        - normalize (bool): Whether to normalise each column to sum to 1.

        Returns:
        - pd.DataFrame: DataFrame with outer keys as columns and processed values as rows.
        """
        df_dict = {}
        for outer_key, inner_dict in data.items():
            if key in inner_dict:
                values = np.array(inner_dict[key], dtype=float)
                # Zero out values less than 1
                values = np.where(values < 0, 0, values)
                if normalize and values.sum() > 0:
                    values = values / values.sum()
                df_dict[outer_key] = values
        return pd.DataFrame(df_dict, index=indx)

    def _plot_horizontal_stacked_bars_by_column(self, df: pd.DataFrame, title: str = '', figsize=(10, 6), cmap='tab20'):
        """
        Plots horizontal stacked bars where each column in the DataFrame is a bar.
        Each row's value contributes to a segment in the bar.

        Parameters:
        - df (pd.DataFrame): DataFrame with rows as categories and columns as the bars.
        - title (str): Plot title.
        - figsize (tuple): Size of the figure.
        - cmap (str): Matplotlib colormap name.
        """
        fig, axs = plt.subplot_mosaic([[0]], layout='constrained', figsize=figsize)
        ax = axs[0]
        colors = plt.get_cmap(cmap).colors
        left = np.zeros(len(df.columns))

        row_labels = df.index.tolist()
        for i, row_label in enumerate(row_labels):
            values = df.loc[row_label].values
            ax.barh(df.columns, values, left=left, label=str(row_label), color=colors[i % len(colors)], ec='k',
                    alpha=0.7)
            left += values

        ax.set_yticks(df.columns)
        ax.set_yticklabels([f'{key}' for key in df.columns])
        ax.set_xlabel("Sobol' main indices")
        ax.legend(ncols=min(9, len(df)), bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower center',
                  mode="expand", borderaxespad=0)
        return fig

    def plot(self, obj, figsize=(10, 8)):
        if not os.path.exists(f'{self.folder}/sobol_indices'):
            os.mkdir(f'{self.folder}/sobol_indices')
        # # raw data
        # df_s1 = self._dict_to_dataframe(self.sobol_indices, 'S1', False, self.problem['names'])
        # fig = self._plot_horizontal_stacked_bars_by_column(df_s1)
        # fig.savefig(f"{self.folder}/sobol_indices/Regression_NN_Test_data_Sobol_{obj.replace('/', '_')}.png", dpi=300)

        # normalised data
        df_s1 = self._dict_to_dataframe(self.sobol_indices, 'S1', True, self.problem['names'])
        fig = self._plot_horizontal_stacked_bars_by_column(df_s1, title=obj, figsize=figsize)
        fig.savefig(f"{self.folder}/sobol_indices//Regression_NN_Test_data_Sobol_{obj.replace('/', '_')}_norm.png",
                    dpi=300)
        plt.close(fig)


class Ishigami:
    def __init__(self):
        problem = self.problem()
        self.nvars = problem['num_vars']
        self.names = problem['names']
        self.bounds = problem['bounds']

        # cubature node order
        self.node_order = 10
        self.sobol_indices = np.array([0.31391, 0.44241, 0])

    def function(self, x, a=7, b=0.1):
        return np.atleast_2d(np.sin(x[:, 0]) + a * (np.sin(x[:, 1])) ** 2 + b * x[:, 2] ** 4 * np.sin(x[:, 0])).T
        # return np.array([np.sin(x[:, 0]) + a*(np.sin(x[:, 1]))**2 + b*x[:, 2]**4*np.sin(x[:, 0]),
        #         np.sin(x[:, 0]) + a*(np.sin(x[:, 1]))**2 + b*x[:, 2]**4*np.sin(x[:, 0])]).T

    def problem(self):
        names = ['x1', 'x2', 'x3']
        BOUNDS = {'x1': [-np.pi, np.pi],
                  'x2': [-np.pi, np.pi],
                  'x3': [-np.pi, np.pi]}

        problem = {
            'num_vars': len(names),
            'names': names,
            'bounds': [BOUNDS[name] for name in names]
        }
        return problem

    def train_data(self):
        X = generate_training_nodes(n_variables=self.nvars, bounds=self.bounds, n_samples=1000, method="lhs")
        Y = self.function(X)
        return X, Y

    def test_data(self):
        # bounds = [(-np.pi, np.pi)] * len(self.names)
        X = generate_training_nodes(n_variables=self.nvars, bounds=self.bounds, n_samples=50, method="random")
        Y = self.function(X)
        return X, Y


class TestFunction:
    def __init__(self):
        problem = self.problem()
        self.nvars = problem['num_vars']
        self.names = problem['names']
        self.bounds = problem['bounds']

        # cubature node order
        self.node_order = 4

        self.sobol_indices = np.array([0.8458, 0.0304, 0.1217, 0.0000507])

    def function(self, x):
        # return np.atleast_2d(3*x[:, 0]**3).T
        return np.atleast_2d(-x[:, 0] ** 3 + 3 * x[:, 1] ** 2 + 6 * x[:, 2] ** 2 + x[:, 3]).T

        # names = ['x1']
        # BOUNDS = {'x1': [0, 4]}

    def problem(self):
        a, b = -15, 10
        names = ['x1', 'x2', 'x3', 'x4']
        BOUNDS = {'x1': [a, b],
                  'x2': [a, b],
                  'x3': [a, b],
                  'x4': [a, b]
                  }

        problem = {
            'num_vars': len(names),
            'names': names,
            'bounds': [BOUNDS[name] for name in names]
        }
        return problem

    def train_data(self):
        # bounds = [(-15, 10)] * len(self.names)
        X = generate_training_nodes(n_variables=self.nvars, bounds=self.bounds, n_samples=1000, method="lhs")
        Y = self.function(X)
        return X, Y

    def test_data(self):
        # bounds = [(-15, 10)] * len(self.names)
        X = generate_training_nodes(n_variables=self.nvars, bounds=self.bounds, n_samples=50, method="random")
        Y = self.function(X)
        return X, Y


def metrics(Y_model, Y):
    if not isinstance(Y_model, dict):
        Y_model = {'model': Y_model}

    stats_dict = {}

    for kk, Y_m in Y_model.items():
        mae = np.mean(np.abs(Y_m - Y))
        mse = np.mean((Y_m - Y) ** 2)
        rmse = np.sqrt(mse)
        ss_total = np.sum((Y_m - np.mean(Y)) ** 2)
        ss_residual = np.sum((Y_m - Y) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        stats_dict[kk] = {
            "data_mean": np.mean(Y),
            "data_std": np.std(Y),
            "pce_mean": np.mean(Y_m),
            "pce_std": np.std(Y_m),
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2
        }

    return stats_dict


def metrics_from_cubature(cub, Y):
    # Compute total mean and variance for each objective function (averaging over nodes)
    mean = np.average(Y, weights=cub.weights, axis=0)  # shape: (num_objectives,)
    var = np.average((Y - mean) ** 2, weights=cub.weights, axis=0)  # shape: (num_objectives,)
    stats_dict = {
        "cub_mean": mean,
        "cub_std": np.sqrt(var)
    }
    return stats_dict


def sobol_g(X, D, asterix=2, return_analytical=False):
    if asterix == 1:
        alpha_d = 1.0
        a = [0, 0, 9, 9, 9, 9, 9, 9, 9, 9]

    if asterix == 2:
        alpha_d = 1.0
        a = [0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4]

    if asterix == 3:
        alpha_d = 0.5
        a = [0, 0, 9, 9, 9, 9, 9, 9, 9, 9]

    if asterix == 4:
        alpha_d = 0.5
        a = [0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4]

    if asterix == 5:
        alpha_d = 2
        a = [0, 0, 9, 9, 9, 9, 9, 9, 9, 9]

    if asterix == 6:
        alpha_d = 2
        a = [0, 0.1, 0.2, 0.3, 0.4, 0.8, 1, 2, 3, 4]

    def V_c(c):
        Vc = alpha_d ** 2 / ((1 + 2 * alpha_d) * (1 + a[c % len(a)]) ** 2)
        return Vc

    S_analytical = {'S1': [], 'ST': []}
    np.random.seed(seed=3794)
    delta = np.random.uniform(0, 1, size=D)

    G = 1
    for d in range(D):
        g = ((1 + alpha_d) * (np.abs(2 * (X[:, d] + delta[d] - (X[:, d] + delta[d]).astype(int)) - 1)) ** alpha_d + a[
            d % len(a)]) / (1 + a[d % len(a)])
        G *= g

        # calculate analytic solutions
        Vd = V_c(d)
        VTd = Vd * np.prod([1.0 + V_c(c) for c in range(D) if c != d])
        V = np.prod([1 + V_c(c) for c in range(D)]) - 1

        if return_analytical:
            S_analytical['S1'].append(Vd / V)
            S_analytical['ST'].append(VTd / V)
    # print('Analytical solution', S_analytical['ST'])

    if return_analytical:
        return G, S_analytical
    else:
        return G


def unique_rank(a):
    seen = {}
    ranks = []
    sorted_a = sorted(a)  # Sort the list for ranking

    for idx, value in enumerate(a):
        # Assign rank based on first occurrence in sorted list + number of previous occurrences in original list
        if value not in seen:
            seen[value] = sorted_a.index(value)

        rank = seen[value] + list(a[:idx]).count(value)
        ranks.append(rank)

    return ranks


def plot_rank(Si_D, Si_D_analytical=None, which='S1', figsize=None, subcategories=True, transpose=True):
    if subcategories:
        # form grid
        ddims = list(Si_D.keys())  # dimension of the input parameters
        indims = list(Si_D[ddims[0]].keys())  # inner dimension, exponent m for sobol sequence
        subdims = list(Si_D[ddims[0]][indims[0]].keys())
        grid = [[f"{d}_{v}" for v in subdims] for d in ddims]

        vsize = len(list(Si_D.keys()))
        if figsize is None:
            figsize = (2 * len(subdims), 2 * vsize)

        fig, axs = plt.subplot_mosaic(grid, layout='constrained',
                                      sharex=True, figsize=figsize)

        # flatten dictionary
        flat = {(outerKey, innerKey1, innerKey2, innermostKey): values
                for outerKey, innerDict1 in Si_D.items()
                for innerKey1, innerDict2 in innerDict1.items()
                for innerKey2, innermostDict in innerDict2.items()
                for innermostKey, values in innermostDict.items()}

        df = pd.DataFrame(list(flat.keys()), columns=['d', 'm', 'obj', 'sensitivity_vars'])
        df['values'] = list(flat.values())

        for d in ddims:
            for ii, subdim in enumerate(subdims):
                S1s = np.vstack(
                    df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == which)]['values'].to_numpy())
                rank_array = []
                for idx, row in enumerate(S1s):
                    rank_array.append(unique_rank(row))
                for rank in np.array(rank_array).T:
                    m = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == which)]['m'].reset_index(
                        drop=True)
                    axs[f'{d}_{subdim}'].plot(m, rank, marker='o', mfc='None', label='')

                # axs[f'{d}_{subdim}'].set_yscale('log')
                axs[f'{d}_{subdim}'].set_xlabel('m')
                # axs[f'{d}_{subdim}'].legend()

                if ii == 0:
                    axs[f'{d}_{subdim}'].set_ylabel('rank')
                else:
                    axs[f'{d}_{subdim}'].set_yticks([])
                    axs[f'{d}_{subdim}'].yaxis.set_tick_params(labelleft=False)

    else:
        vsize = len(list(Si_D.keys()))
        if figsize is None:
            figsize = (6, 2 * vsize)

        if transpose:
            fig, axs = plt.subplot_mosaic(np.atleast_2d([d for d in Si_D.keys()]).T, layout='constrained',
                                          sharex=True, figsize=figsize)
        else:
            fig, axs = plt.subplot_mosaic(np.atleast_2d([d for d in Si_D.keys()]), layout='constrained',
                                          sharex=True, figsize=figsize)

        for d, sid in Si_D.items():
            sid_df = pd.DataFrame(sid.values())

            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            colors = matplotlib.colormaps['Set2'].colors

            if which == 'S1':
                data = pd.DataFrame(sid_df['S1'].tolist())
            elif which == 'delta':
                data = pd.DataFrame(sid_df['delta'].tolist())
            else:
                data = pd.DataFrame(sid_df['S2'].tolist())

            if Si_D_analytical:
                sid_analytical_df = pd.DataFrame(Si_D_analytical[d])
                data_analytical = pd.DataFrame(sid_analytical_df['S1'])

            rank_array = []
            rank_array_analytical = []
            for (idx, row) in data.iterrows():
                rank_array.append(unique_rank(row))
                if Si_D_analytical:
                    rank_array_analytical.append(unique_rank(data_analytical['S1']))

            for i, row_rank in enumerate(np.array(rank_array).T):
                axs[d].plot(row_rank, marker='o', mec='k', mfc='None', zorder=500)
                if Si_D_analytical:
                    axs[d].plot(np.array(rank_array_analytical).T[i], marker='o', mfc='None',
                                lw=10, alpha=0.5)

            axs[d].set_xlabel('m', fontsize=20)
            axs[d].set_title(f'd={d}', fontsize=24)
            axs[d].tick_params(axis='x', labelsize=18)
            axs[d].tick_params(axis='y', labelsize=18)
            axs[d].set_yscale('log')
            axs[d].legend(fontsize=15)

    return axs


def plot_Si_D(Si_D, which=None, conf_intervals=True, ax=None, figsize=None, subcategories=True, rowtitle=None,
              coltitle=None):
    if which is None:
        which = 'S1'
    if subcategories:
        # form grid
        ddims = list(Si_D.keys())  # dimension of the input parameters
        indims = list(Si_D[ddims[0]].keys())  # inner dimension, exponent m for sobol sequence
        subdims = list(Si_D[ddims[0]][indims[0]].keys())
        grid = [[f"{d}_{v}" for v in subdims] for d in ddims]

        vsize = len(list(Si_D.keys()))
        if figsize is None:
            figsize = (2.5 * len(subdims), 3 * vsize)

        fig, axs = plt.subplot_mosaic(np.array(grid).T, layout='constrained', figsize=figsize)

        # flatten dictionary
        flat = {(outerKey, innerKey1, innerKey2, innermostKey): values
                for outerKey, innerDict1 in Si_D.items()
                for innerKey1, innerDict2 in innerDict1.items()
                for innerKey2, innermostDict in innerDict2.items()
                for innermostKey, values in innermostDict.items()}

        df = pd.DataFrame(list(flat.keys()), columns=['d', 'm', 'obj', 'sensitivity_vars'])
        df['values'] = list(flat.values())

        for d in ddims:
            for ii, subdim in enumerate(subdims):
                S1s = np.vstack(df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == which)][
                                    'values'].to_numpy()).T
                if conf_intervals:
                    confs = np.vstack(
                        df[(df['obj'] == subdim) & (df['sensitivity_vars'] == 'S1_conf')]['values'].to_numpy()).T
                    for S1, conf in zip(S1s, confs):
                        m = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == 'S1')][
                            'm'].reset_index(drop=True)
                        axs[f'{d}_{subdim}'].plot(m, S1, marker='o', mfc='None', label='')
                        axs[f'{d}_{subdim}'].errorbar(m, S1, yerr=conf,
                                                      capsize=10, lw=2, mfc='none', ms=10)
                else:
                    for S1 in S1s:
                        m = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == which)][
                            'm'].reset_index(drop=True)
                        axs[f'{d}_{subdim}'].plot(m, S1, marker='o', mfc='None', label='')

                axs[f'{d}_{subdim}'].set_yscale('log')
                axs[f'{d}_{subdim}'].set_xlabel('m')
                # axs[f'{d}_{subdim}'].legend()

                if ii == 0:
                    axs[f'{d}_{subdim}'].set_ylabel(which)
                    if rowtitle:
                        axs[f'{d}_{subdim}'].annotate(rowtitle[ii], xy=(-0.25, 0.5), xycoords='axes fraction',
                                                      fontsize=14, ha='center', va='center', rotation=90)
                else:
                    axs[f'{d}_{subdim}'].set_yticks([])
                    axs[f'{d}_{subdim}'].yaxis.set_tick_params(labelleft=False)
    else:
        # form grid
        ddims = list(Si_D.keys())  # dimension of the input parameters
        indims = list(Si_D[ddims[0]].keys())  # inner dimension, exponent m for sobol sequence
        grid = [[f"{d}" for d in ddims]]

        vsize = len(list(Si_D.keys()))
        if figsize is None:
            figsize = (3 * vsize, 3)

        if ax is None:
            fig, axs = plt.subplot_mosaic(grid, layout='constrained', figsize=figsize)
        else:
            axs = ax

        # flatten dictionary
        flat = {(outerKey, innerKey1, innermostKey): values
                for outerKey, innerDict1 in Si_D.items()
                for innerKey1, innermostDict in innerDict1.items()
                for innermostKey, values in innermostDict.items()}

        df = pd.DataFrame(list(flat.keys()), columns=['d', 'm', 'sensitivity_vars'])
        df['values'] = list(flat.values())

        for ii, d in enumerate(ddims):
            S1s = np.vstack(df[(df['sensitivity_vars'] == which) & (df['d'] == d)]['values'].to_numpy()).T
            if conf_intervals:
                confs = np.vstack(df[(df['sensitivity_vars'] == 'S1_conf')]['values'].to_numpy()).T
                for S1, conf in zip(S1s, confs):
                    m = df[(df['d'] == d) & (df['sensitivity_vars'] == 'S1')][
                        'm'].reset_index(drop=True)
                    axs[f'{d}'].plot(m, S1, marker='o', mfc='None', label='')
                    axs[f'{d}'].errorbar(m, S1, yerr=conf, capsize=10, lw=2, mfc='none', ms=10)
            else:
                for i1, S1 in enumerate(S1s):
                    m = df[(df['d'] == d) & (df['sensitivity_vars'] == which)]['m'].reset_index(drop=True)
                    axs[f'{d}'].plot(m, S1, marker='o', mfc='None', label='$x_{' + f'{i1}' + '}$')

            if ii == 0:
                if which == 'delta':
                    axs[f'{d}'].set_ylabel('$\delta$', fontsize=20)
                else:
                    axs[f'{d}'].set_ylabel(f'{which}', fontsize=20)
            else:
                pass
                # axs[f'{d}'].set_yticks([])
                # axs[f'{d}'].yaxis.set_tick_params(labelleft=False)

            axs[f'{d}'].set_xlabel('m', fontsize=20)
            axs[f'{d}'].set_title(f'd={d}', fontsize=24)
            axs[f'{d}'].tick_params(axis='x', labelsize=18)
            axs[f'{d}'].tick_params(axis='y', labelsize=18)
            axs[f'{d}'].set_yscale('log')
            # axs[f'{d}'].set_xticks(range(0, 21, 5))
            if d <= 10:
                ncols = 4
            else:
                ncols = 3
            axs[f'{d}'].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=18, ncols=ncols,
                               handletextpad=0.0, labelspacing=0.1, borderpad=0.2, columnspacing=0.8)

    return axs


def plot_error(Si_D, which=None, ax=None, figsize=None, subcategories=True, label='$||\mathbf{\epsilon}||_2$',
               transpose=True, rowtitle=None, coltitle=None, oneplot=False):
    if subcategories:
        # form grid
        ddims = list(Si_D.keys())  # dimension of the input parameters
        indims = list(Si_D[ddims[0]].keys())  # inner dimension, exponent m for sobol sequence
        subdims = list(Si_D[ddims[0]][indims[0]].keys())
        grid = [[f"{d}_{v}" for v in subdims] for d in ddims]

        vsize = len(list(Si_D.keys()))
        if oneplot:
            fig, axs = plt.subplot_mosaic([[0]], layout='constrained', figsize=figsize, sharey=True)
        else:
            if ax is None:
                if figsize is None:
                    figsize = (2 * len(subdims), 2 * vsize)

                fig, axs = plt.subplot_mosaic(np.array(grid).T, layout='constrained', figsize=figsize, sharey=True)
            else:
                axs = ax
        # flatten dictionary
        flat = {(outerKey, innerKey1, innerKey2, innermostKey): values
                for outerKey, innerDict1 in Si_D.items()
                for innerKey1, innerDict2 in innerDict1.items()
                for innerKey2, innermostDict in innerDict2.items()
                for innermostKey, values in innermostDict.items()}

        df = pd.DataFrame(list(flat.keys()), columns=['d', 'm', 'obj', 'sensitivity_vars'])
        df['values'] = list(flat.values())

        for i1, d in enumerate(ddims):
            for i2, subdim in enumerate(subdims):
                m = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == 'l2_error')][
                    'm'].reset_index(drop=True)
                l2_err = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == 'l2_error')][
                    'values'].reset_index(drop=True)
                # max_err = df[(df['d'] == d) & (df['obj'] == subdim) & (df['sensitivity_vars'] == 'max_error')]['values'].reset_index(drop=True)

                if oneplot:
                    axs[0].plot(m, l2_err, marker='o', mfc='None', label=subdim)
                    axs[0].set_yscale('log')
                    axs[0].set_xlabel('m')
                    axs[0].set_xlabel('S1')
                    axs[0].legend()
                else:
                    axs[f'{d}_{subdim}'].plot(m, l2_err, marker='o', mfc='None', label=label)

                    if i1 == 0:
                        axs[f'{d}_{subdim}'].set_ylabel('$L_2$ error')

                        if rowtitle:
                            axs[f'{d}_{subdim}'].annotate(rowtitle[i2], xy=(-0.25, 0.5), xycoords='axes fraction',
                                                          fontsize=14, ha='center', va='center', rotation=90)

                    axs[f'{d}_{subdim}'].set_yscale('log')
                    axs[f'{d}_{subdim}'].set_xlabel('m')
                    axs[f'{d}_{subdim}'].legend()

                    if i2 == 0 and coltitle is not None:
                        axs[f'{d}_{subdim}'].set_title(f'd={d}', fontsize=12)

    else:
        vsize = len(list(Si_D.keys()))

        if ax is None:
            if transpose:
                if figsize is None:
                    figsize = (6, 2 * vsize)
                fig, axs = plt.subplot_mosaic(np.atleast_2d([d for d in Si_D.keys()]).T, layout='constrained',
                                              figsize=figsize)
            else:
                if figsize is None:
                    figsize = (2 * vsize, 6)
                fig, axs = plt.subplot_mosaic(np.atleast_2d([d for d in Si_D.keys()]), layout='constrained',
                                              figsize=figsize)
        else:
            axs = ax

        for i1, (d, sid) in enumerate(Si_D.items()):
            sid_df = pd.DataFrame(sid.values())

            opt_format = ['o', '^', 's', 'D', 'P', 'v']
            colors = matplotlib.colormaps['Set2'].colors

            data_l2_err = pd.DataFrame(sid_df['l2_error'].tolist()).T
            # data_max_err = pd.DataFrame(sid_df['max_error'].tolist()).T

            # Plot each "position" across all rows
            for (idx, l2_err) in data_l2_err.iterrows():
                axs[d].plot(range(1, len(l2_err) + 1), l2_err, marker='o', mfc='None', label=label)
                # axs[d].plot(max_err, marker='^', mfc='None')

            if i1 == 0:
                axs[d].set_ylabel('$L_2$ error', fontsize=20)
            axs[d].set_xlabel('m', fontsize=20)
            axs[d].set_title(f'd={d}', fontsize=24)
            axs[d].tick_params(axis='x', labelsize=18)
            axs[d].tick_params(axis='y', labelsize=18)
            axs[d].set_yscale('log')
            axs[d].legend(fontsize=15)
    return axs


def plot_sensitivity_indices(sensitivity_indices, which=None, confidence_intervals=True, ax=None):
    """
    Plot Sobol indices with optional confidence intervals as error bars.

    Parameters:
    sobol_indices (dict): Dictionary with Sobol indices.
        Should contain keys like 'S1', 'ST', and optionally 'S2'.
        Each value should be an array-like object with Sobol index values.
    confidence_intervals (dict, optional): Dictionary with confidence intervals for Sobol indices.
        Keys should match those in `sobol_indices`. Each entry should be an array-like
        of the same length as the corresponding Sobol index array, giving the error values.
    ax (matplotlib.axes.Axes, optional): Existing matplotlib Axes object.
        If None, a new Axes will be created.

    Returns:
    matplotlib.axes.Axes: The Axes object with the plot.
    """

    keys_list = ['S1', 'ST', 'delta']
    colors = matplotlib.colormaps['Set2'].colors
    opt_format = ['o', '^', 's', 'D', 'P', 'v']

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if which is None:
        which = sensitivity_indices.keys()

    for i, (key, values) in enumerate(sensitivity_indices.items()):
        if key in which and key in keys_list:
            x_pos = np.arange(len(values)) + i * 0.2  # Offset each type slightly for clarity
            # color = colours.get(key, 'grey')

            if confidence_intervals:
                ax.errorbar(
                    x_pos, values, yerr=sensitivity_indices[f'{key}_conf'], fmt=opt_format[i],
                    capsize=10, lw=2, mfc='none',
                    color=colors[i], label=f'{key} (± CI)', ms=10
                )
            else:
                ax.scatter(x_pos, values, label=key, color=colors[i], s=150,
                           marker=opt_format[i],
                           fc='none', ec=colors[i], lw=2, zorder=100)

    # Set axis labels and title
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Sobol Index Value")
    ax.set_title("Sobol Indices with Confidence Intervals")
    ax.set_xticks(np.arange(len(values)))
    ax.legend()

    return ax


def generate_training_nodes(n_variables, bounds, n_samples, method="random"):
    """
    Generate training nodes in a bounded n-dimensional space.

    Parameters:
    - n_variables: int — number of variables/dimensions
    - bounds: list of (min, max) tuples for each variable
    - n_samples: int — number of samples to generate
    - method: "random", "lhs", or "sobol"

    Returns:
    - ndarray of shape (n_samples, n_variables)
    """
    if method == "random":
        # Uniform random in [0, 1]^d
        np.random.seed(3794)
        samples = np.random.rand(n_samples, n_variables)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=n_variables, seed=3794)
        samples = sampler.random(n=n_samples)
    elif method == "sobol":
        sampler = qmc.Sobol(d=n_variables, scramble=True)
        samples = sampler.random_base2(m=int(np.log2(n_samples)))
    else:
        raise ValueError("Unsupported method. Choose from 'random', 'lhs', or 'sobol'.")

    # Scale samples to provided bounds
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(samples, lb, ub)

    return scaled_samples


# problem = {
#     'num_vars': 10,
#     'names': [f'x{i}' for i in range(D)],
#     'bounds': [[0, 1] for _ in range(D)]
# }
# print(problem)
#
# X = saltelli.sample(problem, 2**10)
# # X = latin.sample(problem, 2**10)
# # print(X, X.shape)
# # Y = Ishigami.evaluate(X)
# Y = sobol_g(X)
# Si = sobol.analyze(problem, Y)
#
# print(Si['ST'])
# print()
# # X = latin.sample(problem, 2**10)
# # Y = Ishigami.evaluate(X)
# Y = sobol_g(X)
# Si = delta.analyze(problem, X, Y)
# print(Si)
# # print(Si['ST'])

def read_dakota_input_file(filename):
    """
    Reads the specified file and extracts the descriptors as well as the lower and upper bounds.

    Returns:
        descriptors (list): List of descriptor strings.
        bounds (dict): Dictionary mapping each descriptor to a list of two floats [lower_bound, upper_bound].
    """
    descriptors = []
    lower_bounds = []
    upper_bounds = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("lower_bounds"):
                # The line format is: lower_bounds =  53.82   61.245   1.755    ...
                parts = line.split("=")
                if len(parts) > 1:
                    tokens = parts[1].split()
                    lower_bounds = [float(tok) for tok in tokens]
            elif line.startswith("upper_bounds"):
                parts = line.split("=")
                if len(parts) > 1:
                    tokens = parts[1].split()
                    upper_bounds = [float(tok) for tok in tokens]
            elif line.startswith("descriptors"):
                parts = line.split("=")
                if len(parts) > 1:
                    # The tokens are quoted strings (e.g. "'dh'"). Remove the quotes.
                    tokens = parts[1].split()
                    descriptors = [tok.strip("'\"") for tok in tokens]

    if not (len(descriptors) == len(lower_bounds) == len(upper_bounds)):
        raise ValueError("Mismatch in number of descriptors and bounds")

    # Create a dictionary mapping each descriptor to its [lower, upper] bound.
    bounds = {desc: [lower_bounds[i], upper_bounds[i]] for i, desc in enumerate(descriptors)}

    return descriptors, bounds


# Example usage:
def test_function(x):
    """ Example function: f(x) = x1 + x2^2 """
    return x[0] + x[1] ** 2


def ishigami(x):
    return np.sin(x[0]) + 7 * (np.sin(x[1])) ** 2 + 0.1 * x[2] ** 4 * np.sin(x[0])


# Example usage:
if __name__ == "__main__":
    # filename = r"C:\Users\sosoho\PycharmProjects\cst_control\files\dakota.in"  # replace with the path to your file
    # descriptors, bounds = read_dakota_input_file(filename)
    # print("Descriptors:", descriptors)
    # print("Bounds:", bounds)

    cub = Cubature()
    p = 3
    sobol_indices = cub.estimate_sobol_indices(ishigami)
    print("Estimated Sobol indices:", sobol_indices)


