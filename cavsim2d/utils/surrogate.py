"""Surrogate models: Polynomial Chaos Expansion and Neural Networks."""
from cavsim2d.utils.sensitivity import read_dakota_input_file
from SALib.analyze import sobol
from cavsim2d.utils.sampling import SampleGenerator
import json
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from itertools import product
from scipy.special import legendre
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
import time
import os


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
            Y_model = self.evaluate_reg(self.cub.nodes)
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


# class NeuralNet:
#     def __init__(self, data, problem, obj):
#         self.model = None
#         self.scaler = None
#         self.data = data

#         self.train_size = int(len(self.data) * 0.8)
#         self.test_size = len(self.data) - self.train_size

#         if isinstance(problem, str):
#             self.names, self.bounds = read_dakota_input_file(os.path.join(problem, 'dakota.in'))
#             self.bounds = [b for _, b in self.bounds.items()]
#             self.dim = len(self.names)

#             self.problem = {
#                 'num_vars': self.dim,
#                 'names': self.names,
#                 'bounds': self.bounds
#             }
#         else:
#             self.names = problem['names']
#             self.bounds = problem['bounds']
#             self.dim = len(self.names)

#             self.problem = problem

#         self.X_train, self.Y_train = self._train_data(obj)
#         self.X_test, self.Y_test = self._test_data()

#     def _train_data(self, obj):
#         self.obj = obj
#         # X
#         self.nodes = self.data.iloc[:self.train_size, 0:self.dim].to_numpy()
#         self.unscaled_nodes = self.nodes
#         self.weights = np.ones(self.train_size)

#         self.Y = self.data.iloc[:self.train_size][obj].to_numpy()

#         return self.nodes, np.atleast_2d(self.Y).T

#     def _test_data(self):
#         test_data = self.data[self.train_size:]
#         test_nodes = test_data.iloc[:, 0:self.dim].to_numpy()
#         test_nodes_Y = np.atleast_2d(test_data[self.obj].to_numpy()).T

#         return [test_nodes, test_nodes_Y]

#     def train(self):
#         # Step 3: Standardize Data (important for neural networks)
#         self.scaler = StandardScaler()
#         X_train_fit = self.scaler.fit_transform(self.X_train)

#         # Step 4: Define and Train the Neural Network
#         self.model = MLPRegressor(hidden_layer_sizes=(32, 32, 32, 32),  # Two hidden layers
#                                   activation='tanh',  # Activation function
#                                   solver='adam',  # Optimizer
#                                   alpha=0.001,  # L2 regularization (prevents overfitting)
#                                   max_iter=10000,  # Number of iterations
#                                   random_state=42)

#         self.model.fit(X_train_fit, self.Y_train.ravel())

#         return self

#     def test(self, x_test=None):
#         if x_test is None:
#             x_test = self.X_test

#         X_test_fit = self.scaler.transform(x_test)
#         Y_pred = self.model.predict(X_test_fit)
#         # mse = mean_squared_error(self.Y_test, Y_pred)

#         return Y_pred

#     def func(self, X):
#         return np.atleast_2d(self.model.predict(self.scaler.transform(X))).T


