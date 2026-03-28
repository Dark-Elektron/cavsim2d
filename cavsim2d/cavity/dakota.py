from cavsim2d.constants import *
from cavsim2d.utils.shared_functions import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess

class Dakota:
    def __init__(self, folder, name, scripts_folder=None):
        self.nodes = None
        self.sim_results = None

        assert f'/' in folder, error('Please ensure directory paths use forward slashes.')
        self.projectDir = folder
        if scripts_folder is None:
            self.scripts_folder = r'D:/Dropbox/CavityDesignHub/analysis_modules/uq/dakota_scripts'
        else:
            self.scripts_folder = scripts_folder
        self.name = name

    def write_input_file(self, **kwargs):
        keys = kwargs.keys()
        assert 'variables_config' in keys, error('please enter keyword "variables config"')
        assert 'interface_config' in keys, error('please enter keyword "interface config"')
        assert 'method_config' in keys, error('please enter keyword "method config"')

        variables_config = kwargs['variables_config']
        interface_config = kwargs['interface_config']
        method_config = kwargs['method_config']

        # check if folder exists, if not, create folder
        if not os.path.exists(os.path.join(self.projectDir, self.name)):
            try:
                os.mkdir(os.path.join(self.projectDir, self.name))
            except FileExistsError:
                error("Could not create folder. Make sure target location exists.")

        with open(os.path.join(self.projectDir, self.name, f'{self.name}.in'), 'w') as f:
            self.environment(f)
            self.method(f, **method_config)
            self.variables(f, **variables_config)
            self.interface(f, **interface_config)

    def environment(self, f):
        f.write('environment\n')
        f.write('\ttabular_data\n')
        f.write("\t\ttabular_data_file = 'sim_result_table.dat'\n")
        f.write('\tresults_output\n')
        f.write("\t\tresults_output_file = 'result_output_file.dat'\n")

        f.write('\n')

    def method(self, f, **kwargs):
        keys = kwargs.keys()
        assert 'method' in keys, error('Please enter "method" in "method config".')
        method = kwargs['method']

        f.write('method\n')
        f.write(f'\t{method}\n')
        f.write("\t\texport_expansion_file ='expansion_file.dat'\n")
        f.write("\t\tcubature_integrand = 3\n")
        f.write("\t\tsamples_on_emulator = 10000\n")
        f.write("\t\tseed = 12347\n")
        f.write("\t\tvariance_based_decomp interaction_order = 1\n")

        f.write('\n')

    def variables(self, f, **kwargs):
        """

        Parameters
        ----------
        f: File
            File
        kind: {'uniform_uncertain', 'normal_uncertain', 'beta_uncertain'}
            Type of distribution of the variables
        descriptors: list, ndarray
            Uncertain variable names
        kwargs: kwargs
            Other relevant arguments eg. {means: [], 'std_deviations': [], 'lower_bounds': [], 'upper_bounds': []}


        Returns
        -------

        """

        keys = kwargs.keys()
        assert 'kind' in keys, error('Please enter "kind"')
        assert 'lower_bounds' in keys, error('Please enter keyword "lower_bounds"')
        assert 'upper_bounds' in keys, error('Please enter keyword "upper bounds"')
        kind = kwargs['kind']
        upper_bounds = kwargs['upper_bounds']
        lower_bounds = kwargs['lower_bounds']

        assert len(upper_bounds) == len(lower_bounds), error("Length of upper and lower bounds must be equal.")

        if "descriptors" in keys:
            descriptors = kwargs['descriptors']
        else:
            info('"descriptors" not entered. Using default parameter labelling.')
            descriptors = [f'p{n}' for n in range(len(upper_bounds))]
        assert len(descriptors) == len(kwargs['upper_bounds'])

        f.write("variables\n")
        f.write(f"\t{kind} = {len(descriptors)}\n")
        f.write(
            "\tdescriptors       =   " + '\t\t\t'.join(['"' + descriptor + '"' for descriptor in descriptors]) + '\n')

        if 'means' in kwargs.keys():
            assert len(descriptors) == len(kwargs['means'])
            f.write("\tmeans      =   " + '\t\t\t'.join([str(mean) for mean in kwargs['means']]) + '\n')

        if 'std_deviations' in kwargs.keys():
            assert len(descriptors) == len(kwargs['std_deviations'])
            f.write("\tstd_deviations      =   " + '\t\t\t'.join([str(std) for std in kwargs['std_deviations']]) + '\n')

        if 'lower_bounds' in kwargs.keys():
            f.write("\tlower_bounds      =   " + '\t\t\t'.join([str(lb) for lb in kwargs['lower_bounds']]) + '\n')

        if 'upper_bounds' in kwargs.keys():
            f.write("\tupper_bounds      =   " + '\t\t\t'.join([str(ub) for ub in kwargs['upper_bounds']]) + '\n')

        f.write('\n')

    def interface(self, f, **kwargs):

        keys = kwargs.keys()
        assert 'analysis_driver' in keys, error('please enter keyword "analysis driver"')
        assert 'responses' in keys, error('Please enter "responses"')

        analysis_driver = kwargs['analysis_driver']
        responses = kwargs['responses']

        nodes_only = False
        if 'nodes_only' in keys:
            nodes_only = kwargs['nodes_only']
            responses = ['f1']

        processes = 1
        if 'processes' in keys:
            processes = kwargs['processes']

        f.write("interface\n")
        f.write("#\tcommon options\n")
        f.write("#\tfork\n")
        f.write("\tparameters_file = 'params.in'\n")
        f.write("\tresults_file    = 'results.out'\n")
        f.write(f"\tsystem asynchronous evaluation_concurrency = {processes}\n")
        f.write(f"\tanalysis_driver = '{analysis_driver} {len(responses)} {nodes_only} {self.scripts_folder}'\n")
        f.write("#\tparameters_file = 'params.in'\n")
        f.write("#\tresults_file    = 'results.out'\n")
        f.write("#\tfile_tag\n")
        f.write("#\tfile_save\n")
        f.write("#\taprepro\n")
        f.write('\n')

        self.responses(f, responses)

    def responses(self, f, responses):
        f.write("responses\n")
        f.write(f"\tresponse_functions = {len(responses)}\n")
        f.write("\tno_gradients\n")
        f.write("\tno_hessians\n")

        f.write('\n')

    def nodes_to_cst_sweep_input(self, partitions=1):
        # save parts
        row_partition = len(self.nodes.index) // partitions
        for i in range(partitions):
            if i < partitions - 1:
                df_part = self.nodes.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = self.nodes.loc[i * row_partition:]

            df_part.to_csv(fr"{self.projectDir}/{self.name}/cst_sweep_files/cst_par_in_{i + 1}.txt", sep="\t",
                           index=None)

    def run_analysis(self, write_cst=True, partitions=1):
        cwd = os.path.join(self.projectDir, self.name)
        dakota_in = f'{os.path.join(self.projectDir, self.name, f"{self.name}.in")}'
        dakota_out = f'{os.path.join(self.projectDir, self.name, f"{self.name}.out")}'

        subprocess.run(['dakota', '-i', dakota_in, '-o', dakota_out], cwd=cwd, shell=True)

        # read results
        filepath = fr"{self.projectDir}/{self.name}/sim_result_table.dat"
        self.sim_results = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        self.nodes = self.sim_results.drop(self.sim_results.filter(regex='response|interface|eval_id').columns, axis=1)
        self.sim_results.to_excel(fr"{self.projectDir}/{self.name}/nodes.xlsx", index=False)

        if write_cst:
            # check if folder exist and clear
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))
            else:
                os.mkdir(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

            # post processes
            self.nodes_to_cst_sweep_input(partitions)
        else:
            if os.path.exists(os.path.join(self.projectDir, self.name, 'cst_sweep_files')):
                shutil.rmtree(os.path.join(self.projectDir, self.name, 'cst_sweep_files'))

    def plot_sobol_indices(self, filepath, objectives, which=None, kind='stacked', orientation='vertical',
                           normalise=True,
                           group=None, reorder_index=None,
                           selection_index=None, figsize=None):

        if figsize is None:
            figsize = (8, 4)

        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall(r"\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall(r"\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []

        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=['main', 'total', 'vars'])
            df = df.astype({'main': 'float', 'total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        df_merge_interaction = pd.DataFrame(columns=['interaction', 'var1', 'var2'])
        # ic(result_interaction.items())
        for i, (k, v) in enumerate(result_interaction.items()):
            df = pd.DataFrame(v, columns=['interaction', 'var1', 'var2'])
            df = df.astype({'interaction': 'float'})
            if i == 0:
                df_merge_interaction = df
            else:
                # df_merge_interaction = pd.merge(df_merge_interaction, df, on=['var1', 'var2'])
                pass
            # ic(df_merge_interaction)

            # combine var columns
            # df_merge_interaction["vars"] = df_merge_interaction[["var1", "var2"]].agg(','.join, axis=1)
            # df.plot.bar(x='var', y='Main')

        # ic(df_merge)

        # group columns
        if group:
            df_merge_T = df_merge.T
            df_merge_T.columns = df_merge_T.loc['vars']
            df_merge_T = df_merge_T.drop('vars')
            for g in group:
                df_merge_T[','.join(g)] = df_merge_T[g].sum(axis=1)

                # drop columns
                df_merge_T = df_merge_T.drop(g, axis=1)

            # reorder index
            if reorder_index:
                df_merge_T = df_merge_T[reorder_index]
            df_merge = df_merge_T.T.reset_index()
            # ic(df_merge)

        # ic(df_merge_interaction)

        if normalise:
            # normalise dataframe columns
            for column in df_merge.columns:
                if 'main' in column or 'total' in column:
                    df_merge[column] = df_merge[column].abs() / df_merge[column].abs().sum()

        # ic(df_merge)
        # filter df
        for w in which:
            if w.lower() == 'main' or w.lower() == 'total':
                dff = df_merge.filter(regex=f'{w.lower()}|vars')
            else:
                # create new column which is a combination of the two variable names
                dff = df_merge_interaction.filter(regex=f'{w.lower()}|vars')
                if not dff.empty:
                    dff['vars'] = df_merge_interaction[['var1', 'var2']].apply(lambda x: '_'.join(x), axis=1)

            cmap = 'tab20'

            if not dff.empty:
                if kind.lower() == 'stacked':
                    dff_T = dff.set_index('vars').T
                    if orientation == 'vertical':
                        ax = dff_T.plot.bar(stacked=True, rot=0, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff_T.plot.barh(stacked=True, rot=0, edgecolor='k', figsize=(8, 4))  # , cmap=cmap
                        ax.invert_yaxis()
                        # for bars in ax.containers:
                        #     ax.bar_label(bars, fmt='%.2f', label_type='center', color='white', fontsize=5)

                        ax.set_xlim(left=0)
                        ax.set_yticklabels(objectives)
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')
                else:
                    if orientation == 'vertical':
                        ax = dff.plot.bar(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axhline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(1.04, 1), ncol=2)
                    else:
                        ax = dff.plot.barh(x='vars', stacked=True, figsize=(8, 4))  # , cmap=cmap
                        ax.set_xlim(left=0)
                        ax.axvline(0.05, c='k')
                        plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=4, loc='lower left', mode='expand')

            else:
                error(f"No {w} found.")

    def get_sobol_indices(self, filepath, objectives, which=None, selection_index=None):
        if which is None:
            which = ['Main']

        start_keyword = "Main"
        interaction_start_keyword = "Interaction"
        pattern = r'\s+(-?\d\.\d+e[+-]\d+)\s+(-?\d\.\d+e[+-]\d+)\s+(\w+)'
        pattern_interaction = r'\s*(-?\d+\.\d+e[-+]\d+)\s+(\w+)\s+(\w+)\s*'

        with open(filepath, "r") as file:
            # read the file line by line
            lines = file.readlines()

            # initialize a flag to indicate when to start and stop recording lines
            record = False
            record_interaction = False

            # initialize a list to store the lines between the keywords
            result = {}
            result_interaction = {}
            count = 0
            # loop through each line in the file
            for line in lines:
                # check if the line contains the start keyword
                if start_keyword in line:
                    # if it does, set the flag to start recording lines
                    record = True
                    result[count] = []
                    continue

                if interaction_start_keyword in line:
                    record_interaction = True
                    result_interaction[count] = []
                    continue

                # if the flag is set to record, add the line to the result list
                if record:
                    if re.match(pattern, line):
                        result[count].append(re.findall(r"\S+", line))
                    else:
                        record = False
                        count += 1

                if record_interaction:
                    if re.match(pattern_interaction, line):
                        result_interaction[count - 1].append(re.findall(r"\S+", line))
                    else:
                        record_interaction = False

        if selection_index:
            result = {i: result[key] for i, key in enumerate(selection_index)}
        # result = result_interaction
        # check if any function is empty and repeat the first one
        # result[0] = result[2]

        df_merge = pd.DataFrame(columns=['main', 'total', 'vars'])

        # print the lines between the keywords
        # df_merge_list = []
        for i, (k, v) in enumerate(result.items()):
            df = pd.DataFrame(v, columns=[f'{objectives[i].replace("$", "")}_main',
                                          f'{objectives[i].replace("$", "")}_total', 'vars'])
            df = df.astype(
                {f'{objectives[i].replace("$", "")}_main': 'float', f'{objectives[i].replace("$", "")}_total': 'float'})
            # df_merge_list.append(df)
            # ic(df)
            # ic(df_merge)
            if i == 0:
                df_merge = df
            else:
                df_merge = pd.merge(df_merge, df, on='vars', suffixes=(f'{i}', f'{i + 1}'))
            # df.plot.bar(x='var', y='Main')
        # ic(df_merge_list)
        # df_merge = pd.merge(df_merge_list, on='vars')
        # ic(df_merge)

        return df_merge

    def quadrature_nodes_to_cst_par_input(self, filefolder, n=2):
        filepath = fr"{filefolder}\sim_result_table.dat"
        df = pd.read_csv(filepath, sep='\\s+')

        # delete unnecessary columns
        df.drop(df.filter(regex='response|interface|eval_id').columns, axis=1, inplace=True)
        df.to_excel(fr"{filefolder}\cubature_nodes_pars.xlsx", index=False)

        # save parts
        row_partition = len(df.index) // n
        for i in range(n):
            if i < n - 1:
                df_part = df.loc[i * row_partition:(i + 1) * row_partition - 1]
            else:
                df_part = df.loc[i * row_partition:]

            df_part.to_csv(fr"{filefolder}\cst_par_in_{i + 1}.txt", sep="\t", index=None)

    def get_pce(self, filefolder):

        filepath = fr"{filefolder}\uq_pce_expansion.dat"
        df = pd.read_csv(filepath, sep='\\s+', header=None)

        poly = 0
        for row in df.iterrows():
            poly += 0

    def quote_to_float(self, value):
        if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
            return float(value[1:-1])
        else:
            return value

    def combine_params_output(self, folder, N):
        for i in range(N):
            if i == 0:
                df = pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)
            else:
                df = pd.concat([df, pd.read_csv(f'{folder}/m{(i + 1):02d}.csv', engine='python', skipfooter=1)])

        # rearrange column according to the reference column order
        df_reference = pd.read_excel(fr"{folder}\cubature_nodes_pars.xlsx")
        columns = list(df_reference.columns)
        # check if 3D Run ID in column and drop if yes
        if ' 3D Run ID' in list(df.columns):
            columns.append(' 3D Run ID')

        columns = list(df_reference.columns) + (df.columns.drop(columns).tolist())

        df = df[columns]

        df.to_excel(fr"{folder}\cubature_nodes.xlsx", index=False)

    def plot_sobol(self, config):
        obj = config['obj']
        group = config['group']
        reorder_index = ['reorder_index']
        filefolder = config['folder']
        kind = config['kind']
        normalise = config['normalise']
        which = config['which']
        orientation = config['orientation']
        selection_index = config['selection_index']

        # obj = [r"$Q_\mathrm{ext, FM}$", r"$\max(Q_\mathrm{ext, dip})$"]
        # obj = [r"$S_\mathrm{max}~\mathrm{[dB]}$", r"$S_\mathrm{min}~\mathrm{[dB]}$", r"$f(S_\mathrm{max})~\mathrm{[MHz]}$", r"$f(S_\mathrm{min})~\mathrm{[MHz]}$"]
        # obj =
        # obj = [r"$freq [MHz]$",	fr"$R/Q [Ohm]$", r"$Epk/Eacc []$",	r"$Bpk/Eacc [mT/MV/m]$", 'G', 'kcc', 'ff']
        # obj =
        # obj = [r"$freq [MHz]$", 'kcc']
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', group=group, reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', reorder_index=reorder_index, normalise=False)#
        # plot_sobol_indices(fr"{filefolder}\dakota_HC.out", obj, ['main', 'Total', 'Interaction'], kind='stacked', orientation='horizontal', selection_index=selection_index, normalise=False)#

        self.plot_sobol_indices(fr"{filefolder}\dakota.out", obj, which=which, kind=kind,
                                selection_index=selection_index, orientation=orientation, normalise=normalise)  #


