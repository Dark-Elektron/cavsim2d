"""Sensitivity analysis: Sobol indices, test functions."""
import json
import math
import os
import warnings
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from SALib.analyze import sobol
from scipy.stats import qmc
from typing import List

from cavsim2d.utils.printing import *

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
                    axs[fr'{d}'].set_ylabel('$\delta$', fontsize=20)
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


