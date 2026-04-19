"""Cubature-based uncertainty quantification."""
from cavsim2d.utils.sensitivity import read_dakota_input_file
from itertools import product
import itertools
import json
import os
import pickle
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import roots_legendre, legendre
from scipy.stats import qmc
from typing import List, Dict

from cavsim2d.utils.printing import *

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

        return mean, std, skew, kurtosis

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


