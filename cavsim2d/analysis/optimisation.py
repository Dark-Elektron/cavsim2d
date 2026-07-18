import datetime
import json
import operator
import os
from pathlib import Path
import random
import shutil
from distutils import dir_util

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paretoset import paretoset
from scipy.stats import qmc
from tqdm import tqdm

from cavsim2d.constants import *
from cavsim2d.processes import *
from cavsim2d.utils.printing import *
from cavsim2d.utils.shared_functions import *
from cavsim2d.utils.printing import set_verbose
from cavsim2d.processes.tune import normalize_cell_type_config, _resolve_suffixed_var
from cavsim2d.processes.tune import last_stage_result
from cavsim2d.solvers.objectives import (SEP, parse_objective, read_objective_values,
                                         canonical_objective)
from cavsim2d.utils.config_validation import require

EIGENMODE_QOIS = {"Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]", "R/Q [Ohm]", "G [Ohm]", "Q []"}

CONSTRAINT_OPS = {
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le,
    '==': operator.eq,
}


def _uq_col_name(obj):
    """Return the UQ-adjusted column name for a given objective tuple (sense, name, [target])."""
    sense, name = obj[0], obj[1]
    if sense == 'min':
        return fr'E[{name}] + 6*std[{name}]'
    elif sense == 'max':
        return fr'E[{name}] - 6*std[{name}]'
    else:
        target = obj[2]
        return fr'|E[{name}] - {target}| + std[{name}]'


def _uq_robust_value(obj, expe, std):
    """Compute the robust (6-sigma shifted) value for UQ ranking."""
    sense = obj[0]
    if sense == 'min':
        return expe + 6 * std
    elif sense == 'max':
        return expe - 6 * std
    else:
        return np.abs(expe - obj[2]) + std


def compute_hypervolume(points, ref):
    """Compute the hypervolume dominated by a set of points w.r.t. a reference point.

    All objectives are assumed to be for minimisation. For 'max' objectives, negate
    the values before calling. Points that do not dominate the reference are ignored.

    Parameters
    ----------
    points : array-like, shape (n, d)
        Objective vectors of the Pareto front.
    ref : array-like, shape (d,)
        Reference point (must be worse than all Pareto points in every objective).

    Returns
    -------
    float
        Hypervolume indicator value.
    """
    points = np.asarray(points, dtype=float)
    ref = np.asarray(ref, dtype=float)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    # Filter to points that dominate the reference in all objectives
    mask = np.all(points <= ref, axis=1)
    points = points[mask]

    if len(points) == 0:
        return 0.0

    d = points.shape[1]
    if d == 1:
        return float(ref[0] - np.min(points[:, 0]))

    if d == 2:
        return _hv_2d(points, ref)

    return _hv_recursive(points, ref)


def _hv_2d(points, ref):
    """O(n log n) hypervolume for 2 objectives."""
    # Sort by first objective ascending
    order = np.argsort(points[:, 0])
    pts = points[order]

    # Extract non-dominated subset (sweep for decreasing second objective)
    nd = []
    best_y = np.inf
    for p in pts:
        if p[1] < best_y:
            nd.append(p)
            best_y = p[1]

    # Sum vertical strips
    hv = 0.0
    for i in range(len(nd)):
        x_next = nd[i + 1][0] if i + 1 < len(nd) else ref[0]
        hv += (x_next - nd[i][0]) * (ref[1] - nd[i][1])
    return hv


def _hv_recursive(points, ref):
    """General hypervolume via recursive slicing on the last objective (HSO algorithm)."""
    n, d = points.shape

    if n == 0:
        return 0.0
    if d == 2:
        return _hv_2d(points, ref)
    if n == 1:
        return float(np.prod(ref - points[0]))

    # Sort by last objective ascending
    order = np.argsort(points[:, -1])
    pts = points[order]

    hv = 0.0
    for i in range(n):
        z_hi = pts[i + 1, -1] if i + 1 < n else ref[-1]
        height = z_hi - pts[i, -1]
        if height <= 0:
            continue
        # Points 0..i are active in this slice — project onto d-1 dimensions
        hv += height * _hv_recursive(pts[:i + 1, :-1], ref[:-1])

    return hv


class Optimisation:

    def __init__(self):
        self.poc = 0
        self.eigenmode_config = {}
        self.mid_cell = None
        self.wakefield_config = None
        self.tune_config = None
        self.hv_history = None
        self.hv_ref = None
        self.hv_tol = None
        self.hv_consecutive = None
        self.processes_count = None
        self.method = None
        self.mutation_factor = None
        self.crossover_factor = None
        self.elites_to_crossover = None
        self.chaos_factor = None
        self.tune_parameter = None
        self.uq_config = None
        self.constraints = []
        self.df = None
        self.df_global = None
        self.objs_dict = None
        self.constraints_dict = None
        self.cell_type = None
        self.bounds = None
        self.weights = None
        self.objective_vars = None
        self.objectives = None
        self.objectives_unprocessed = None
        self.ng_max = None
        self.tune_freq = None
        self.initial_points = None
        self.projectDir = None
        self.parentDir = None
        self.pareto_history = None
        self.optimisation_config = None
        self.mutation_sigma = None
        self.eta_sbx = None
        self.base_seed = None
        self.resume = False

    def run(self, cav, config, opt_solver=None):
        """Run the optimisation loop.

        Parameters
        ----------
        cav : Cavity
            Template cavity for spawning candidates.
        config : dict
            Optimisation configuration.
        opt_solver : OptimisationSolver, optional
            The solver object managing the output folder.
        """
        self.cav = cav
        self.opt_solver = opt_solver
        self.pareto_history = []
        self.optimisation_config = config
        self.parentDir = SOFTWARE_DIRECTORY
        self.projectDir = cav.projectDir

        # Global chattiness — default quiet so long optimisations only show
        # errors and the tqdm progress bar; set ``verbose: True`` to restore.
        set_verbose(config.get('verbose', False))

        # Resume support: only meaningful when an opt_solver (and therefore a
        # persistent folder) is provided. The base seed is saved inside the
        # config on disk, so a resumed run reproduces the same RNG sequence.
        self.resume = bool(config.get('resume', False))
        self.base_seed = config.get('seed', None)
        if self.base_seed is None:
            # Fall back to any sampler-level seed present in method config.
            _method_cfg = next(iter(config['method'].values()), None) \
                if isinstance(config.get('method'), dict) else None
            if isinstance(_method_cfg, dict):
                self.base_seed = _method_cfg.get('seed', None)
        if self.base_seed is not None:
            np.random.seed(int(self.base_seed))
            random.seed(int(self.base_seed))
        self.initial_points = config['initial_points']
        self.ng_max = config.get('no_of_generation', 100)
        self.hv_tol = config.get('hv_tol', 1e-9)
        self.hv_consecutive = config.get('hv_consecutive', 3)
        self.objectives_unprocessed = config['objectives']
        self.objectives, weights = process_objectives(config['objectives'])
        # Canonicalise eigenmode objective names ('1:freq [MHz]' -> 'dipole:freq [MHz]')
        # so the DataFrame columns and the uq.json keys agree. Wakefield objectives
        # (ZL/ZT) carry no polarisation and pass through untouched.
        self.objectives = [(o[0], canonical_objective(o[1]), *o[2:]) for o in self.objectives]
        self.objective_vars = [obj[1] for obj in self.objectives]
        if 'weights' in config:
            self.weights = config['weights']
            require(len(self.weights) == len(weights),
                    "Length of delta must be equal to the length of the variables. For impedance Z entries, one less than "
                    "the length of the interval list weights are needed. Eg. for ['min', 'ZL', [1, 2, 3]], two weights are required. ")
        else:
            self.weights = weights

        self.bounds = config['bounds']
        if 'constraints' in config:
            self.constraints = self.process_constraints(config['constraints'])
        self.processes_count = 1
        if 'processes' in config:
            require(config['processes'] > 0, 'Number of processes must be greater than zero!')
            require(isinstance(config['processes'], int), 'Number of processes must be integer!')
            self.processes_count = config['processes']

        self.method = config['method']
        self.mutation_factor = config['mutation_factor']
        self.crossover_factor = config['crossover_factor']
        self.elites_to_crossover = config['elites_for_crossover']
        self.chaos_factor = config['chaos_factor']

        # Mutation spread: fraction of variable range used as Gaussian sigma (default 5%)
        self.mutation_sigma = config.get('mutation_sigma', 0.05)
        # SBX distribution index: higher = offspring closer to parents (default 2)
        self.eta_sbx = config.get('eta_sbx', 2.0)

        self.tune_config = config['tune_config']
        tune_config_keys = self.tune_config.keys()
        require('freqs' in tune_config_keys, 'Please enter the target tune frequency.')
        require('cell_type' in tune_config_keys or ('parameters' in tune_config_keys and 'cell_types' in tune_config_keys),
                "Please enter 'cell_type' (dict) in tune_config, e.g. {'mid-cell': 'Req'}.")

        # Normalise to the keyed form — optimisation only supports a single
        # (cell_type, tune_variable) pair per candidate run.
        _ct_map = normalize_cell_type_config(self.tune_config)
        if len(_ct_map) != 1 or len(next(iter(_ct_map.values()))) != 1:
            error("Optimisation only supports a single cell_type/tune_variable pair; "
                  "using the first one only.")
        self.cell_type = next(iter(_ct_map.keys()))
        self.tune_parameter = next(iter(_ct_map.values()))[0]

        ct_norm = self.cell_type.lower().replace('-', ' ').replace('_', ' ')
        if ct_norm == 'end cell':
            require('mid-cell' in config, 'end-cell optimisation requires mid-cell dimensions via "mid-cell" key.')
            require(len(config['mid-cell']) >= 7, 'Incomplete mid cell dimension.')
            self.mid_cell = config['mid-cell']

        self.tune_freq = self.tune_config['freqs']

        self.wakefield_config = {}
        if (any(['ZL' in obj for obj in self.objective_vars])
                or any(['ZT' in obj for obj in self.objective_vars])
                or any([obj in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]', 'P_HOM [kW]'] for obj in
                        self.objective_vars])):
            require('wakefield_config' in config,
                    'Wakefield impedance objective detected in objectives. Please include a field for wakefield_config.')
            self.wakefield_config = config['wakefield_config']

            if 'uq_config' in self.wakefield_config:
                self.uq_config = self.wakefield_config['uq_config']
                self.wakefield_config['uq_config']['objectives'] = self.objectives
                self.wakefield_config['uq_config']['objectives_unprocessed'] = self.objectives_unprocessed

                if self.uq_config['delta']:
                    require(len(self.uq_config['delta']) == len(self.uq_config['variables']),
                            'The number of deltas must be equal to the number of variables.')

        # eigenmode_config can be at top level or nested inside tune_config
        if 'eigenmode_config' in config:
            self.eigenmode_config = config['eigenmode_config']
        elif 'eigenmode_config' in self.tune_config:
            self.eigenmode_config = self.tune_config['eigenmode_config']

        if self.eigenmode_config and 'uq_config' in self.eigenmode_config:
            self.uq_config = self.eigenmode_config['uq_config']
            if self.uq_config['delta']:
                require(len(self.uq_config['delta']) == len(self.uq_config['variables']),
                        'The number of deltas must be equal to the number of variables.')

        self.df = None

        self.df_global = pd.DataFrame()
        self.objs_dict = {}
        self.constraints_dict = {}
        self.hv_history = []
        self.hv_ref = None
        bar = tqdm(total=self.ng_max)
        # Degenerate-geometry errors AND tune-accuracy failures are expected
        # noise during the EA: random mutation/crossover/chaos generate
        # candidates that cannot tune to the target frequency, and those are
        # simply discarded (they never enter the ranked population). Silence
        # both for the whole loop so the progress bar stays readable — a real
        # failure surfaces as an empty generation, not a wall of red text.
        with suppress_errors('Parameter set leads to degenerate geometry',
                             'could not be reached'):
            self._run_ea(bar)

    def _load_resume_state(self):
        """Locate the last completed generation and reload its state.

        Returns a dict with ``df_global``, ``hv_history``, ``pareto_history``
        and ``last_n`` — or ``None`` if nothing resumable was found.
        """
        if self.opt_solver is None:
            return None
        folder = self.opt_solver.folder
        gens_dir = folder / 'generations'
        if not gens_dir.exists():
            return None

        candidates = []
        for p in gens_dir.glob('g*.xlsx'):
            stem = p.stem[1:]
            if stem.isdigit():
                candidates.append((int(stem), p))
        if not candidates:
            return None
        candidates.sort(key=lambda kv: kv[0])
        last_n, last_path = candidates[-1]

        try:
            df_global = pd.read_excel(last_path)
        except Exception as e:
            error(f'Could not read {last_path} for resume: {e}')
            return None

        hv_history = []
        hv_path = folder / 'hv_history.json'
        if hv_path.exists():
            try:
                with open(hv_path) as f:
                    hv_history = json.load(f).get('hv_history', [])
            except Exception:
                hv_history = []
        # Truncate to the completed generations only.
        hv_history = hv_history[:last_n + 1]

        pareto_history = []
        ph_path = folder / 'pareto_history.csv'
        if ph_path.exists():
            try:
                ph_df = pd.read_csv(ph_path)
                if 'generation' in ph_df.columns:
                    for g, grp in ph_df.groupby('generation'):
                        if int(g) <= last_n:
                            pareto_history.append(grp.reset_index(drop=True))
            except Exception:
                pass

        return {
            'df_global': df_global,
            'hv_history': hv_history,
            'pareto_history': pareto_history,
            'last_n': last_n,
        }

    def _run_ea(self, bar):
        """Main evolutionary algorithm loop (iterative, not recursive)."""
        # Use optimisation solver's candidates folder if available
        if self.opt_solver is not None:
            folder = self.opt_solver.candidates_folder
        else:
            folder = Path(self.cav.self_dir) / 'optimisation'

        resume_state = self._load_resume_state() if self.resume else None
        start_n = 0

        if resume_state is not None:
            self.df_global = resume_state['df_global']
            self.hv_history = resume_state['hv_history']
            self.pareto_history = resume_state['pareto_history']
            start_n = resume_state['last_n'] + 1

            # Re-seed so the offspring we're about to compute match what a
            # fresh run at this generation would have produced — then the
            # per-candidate tune/eigenmode resume can pick up any simulation
            # results left behind by the interrupted run.
            if self.base_seed is not None:
                np.random.seed(int(self.base_seed) + start_n)
                random.seed(int(self.base_seed) + start_n)

            df_prev = self.df_global
            df_cross = (self.crossover(df_prev, start_n, self.crossover_factor)
                        if len(df_prev) > 1 else pd.DataFrame())
            df_mutation = self.mutation(df_prev, start_n, self.mutation_factor)
            df_chaos = self.chaos(self.chaos_factor, start_n)
            self.df = pd.concat([df_cross, df_mutation, df_chaos])

            folder.mkdir(parents=True, exist_ok=True)
            bar.update(start_n)
            info(f'Resuming optimisation from generation {start_n} '
                 f'({len(self.df_global)} rows carried over from g{start_n - 1}.xlsx).')
        else:
            self.df = self.generate_initial_population(self.initial_points, 0)
            if folder.exists():
                for filepath in folder.iterdir():
                    try:
                        if filepath.is_dir():
                            shutil.rmtree(filepath)
                        else:
                            filepath.unlink()
                    except Exception as e:
                        error(f"Could not remove {filepath}: {e}")
            else:
                folder.mkdir(parents=True, exist_ok=True)

        converged = False
        for n in range(start_n, self.ng_max):
            df = self._evaluate_generation(n)
            if df is None:
                return

            # Check convergence: relative HV change below tolerance
            # for hv_consecutive generations in a row
            if len(self.hv_history) >= 2:
                hv_cur = self.hv_history[-1]
                hv_prev = self.hv_history[-2]
                rel_change = abs(hv_cur - hv_prev) / max(abs(hv_cur), 1e-30)

                k = self.hv_consecutive
                if len(self.hv_history) >= k + 1:
                    hv_arr = np.array(self.hv_history[-(k + 1):])
                    recent_changes = np.abs(np.diff(hv_arr)) / np.maximum(np.abs(hv_arr[1:]), 1e-30)
                    if np.all(recent_changes < self.hv_tol):
                        info(f"Converged at generation {n}: relative HV change < {self.hv_tol} "
                             f"for {k} consecutive generations.")
                        converged = True
                        bar.update(self.ng_max - n)
                        break

            # Birth next generation from current population
            # Use n+1 so offspring keys match the generation they'll be evaluated in
            if self.base_seed is not None:
                np.random.seed(int(self.base_seed) + n + 1)
                random.seed(int(self.base_seed) + n + 1)
            df_cross = self.crossover(df, n + 1, self.crossover_factor) if len(df) > 1 else pd.DataFrame()
            df_mutation = self.mutation(df, n + 1, self.mutation_factor)
            df_chaos = self.chaos(self.chaos_factor, n + 1)

            self.df = pd.concat([df_cross, df_mutation, df_chaos])

            bar.update(1)
            info("=" * 80)

        if not converged:
            info(f"Reached maximum generations ({self.ng_max}) without convergence.")

        bar.update(0)
        end = datetime.datetime.now()
        info("End time: ", end)

        # Save HV history for later retrieval via plot_convergence()
        if self.opt_solver is not None and len(self.hv_history) > 0:
            try:
                hv_path = self.opt_solver.folder / 'hv_history.json'
                with open(hv_path, 'w') as f:
                    json.dump({'hv_history': self.hv_history, 'hv_tol': self.hv_tol}, f)
            except Exception:
                pass

        if len(self.hv_history) > 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.plot(self.hv_history, marker='o')
            ax1.set_xlabel('Generation $n$')
            ax1.set_ylabel('Hypervolume')
            ax1.set_title('Hypervolume indicator')

            # Relative change: |HV_n - HV_{n-1}| / HV_n
            hv = np.array(self.hv_history)
            rel_change = np.abs(np.diff(hv)) / np.maximum(hv[1:], 1e-30)
            ax2.plot(range(1, len(hv)), rel_change, marker='s')
            ax2.axhline(y=self.hv_tol, color='r', linestyle='--', label=f'tol = {self.hv_tol:.0e}')
            ax2.set_yscale('log')
            ax2.set_xlabel('Generation $n$')
            ax2.set_ylabel(r'$|\Delta \mathrm{HV}| / \mathrm{HV}_n$')
            ax2.set_title('Relative HV change')
            ax2.legend()

            plt.tight_layout()
            plt.show()

    def _evaluate_generation(self, n):
        """Evaluate a single generation: simulate, collect results, rank, select. Returns df or None."""
        df = self.df

        # Remove duplicates already in global dataframe
        compared_cols = list(self.bounds.keys())
        if not self.df_global.empty:
            df_temp = df.set_index(compared_cols)
            df_global_temp = self.df_global.set_index(compared_cols)
            mask = ~df_temp.index.isin(df_global_temp.index)
            df = df.iloc[mask]

        # Spawn candidates into flat folder
        if self.opt_solver is not None:
            spawn_folder = str(self.opt_solver.candidates_folder)
        else:
            spawn_folder = str(Path(self.cav.self_dir) / 'optimisation')

        # Map each bounds name to *this model's* parameter key, via the same
        # resolver the tuner uses: A -> A_m (elliptical mid-cell), A -> A_el
        # (end cell), Ri -> Ri (pillbox), R6 -> R6 (gun). So optimisation is
        # driven by the model's own tune variables, not a hardcoded A..Req list.
        param_map = {v: _resolve_suffixed_var(v, self.cell_type, self.cav)
                     for v in self.bounds.keys()}
        df_spawn = df.rename(columns={k: v for k, v in param_map.items() if k in df.columns})
        cavs_object = self.cav.spawn(df_spawn, spawn_folder)
        cavs_dict = cavs_object.cavities_dict

        # Run tuning (which internally runs eigenmode after adjusting the tune parameter)
        self.run_tune_opt(cavs_dict, self.tune_config)

        # Get successfully tuned geometries. Post-refactor, tune artefacts
        # live at <self_dir>/tuned/tune_info/tune_res.json keyed by cell
        # type. Fall back to the legacy location for older on-disk runs.

        processed_keys = []
        tune_result = []
        for key, scav in cavs_dict.items():
            new_path = Path(scav.self_dir) / 'tuned' / 'tune_info' / 'tune_res.json'
            legacy_path = Path(scav.self_dir) / 'eigenmode' / 'tune_res.json'
            filename = new_path if new_path.exists() else legacy_path
            try:
                with open(filename, 'r') as file:
                    tune_res = json.load(file)

                last = last_stage_result(tune_res) or {}
                freq = last.get('FREQ')
                tuned_vars = last.get('TUNED VARIABLES') or (
                    [last['TUNED VARIABLE']] if 'TUNED VARIABLE' in last else [])
                if not tuned_vars or freq is None:
                    info(f'Incomplete tune result for {scav.self_dir}, skipping.')
                    continue
                # Read the tuned value through the model's own accessor, not by
                # indexing last['parameters'] directly: a spline's 'p3_r' is a
                # control-point coordinate (parameters stores 'p3' = [z, r]), so
                # a flat lookup KeyErrors. get_tune_value knows each model's
                # representation (elliptical 'Req_m', gun 'R6', spline 'p3_r').
                tuned_cav = scav.tuned
                if tuned_cav is not None:
                    tune_variable_value = tuned_cav.get_tune_value(tuned_vars[-1])
                else:
                    tune_variable_value = last['parameters'].get(tuned_vars[-1])

                tune_result.append([tune_variable_value, freq])
                processed_keys.append(key)
            except FileNotFoundError:
                info(f'Results not found for {scav.self_dir}, tuning probably failed.')

        df = df.loc[processed_keys]
        df.loc[:, [self.tune_parameter, 'freq [MHz]']] = tune_result

        # Eigenmode objective variables. These are polarisation-qualified
        # ('monopole:R/Q [Ohm]', 'dipole:2:freq [MHz]'); the wakefield objectives
        # (ZL/ZT) carry no polarisation and are handled further down.
        eig_objectives = [v for v in self.objective_vars if isinstance(v, str) and SEP in v]

        if eig_objectives:
            obj_result = []
            processed_keys = []
            tuned_keys = set(df.index)
            for key, scav in cavs_dict.items():
                if key not in tuned_keys:
                    continue
                # Post-refactor, eigenmode ran on the *tuned* cavity, so its results
                # live at <self_dir>/tuned/eigenmode/<pol>/. Fall back to the untuned
                # cavity's path for older runs.
                tuned_cav = scav.tuned
                candidates = [tuned_cav.eigenmode_dir] if tuned_cav is not None else []
                candidates.append(scav.eigenmode_dir)
                for eig_dir in candidates:
                    try:
                        values = read_objective_values(eig_dir, eig_objectives)
                    except (ValueError, OSError):
                        continue
                    # Order by the objective list, not by qois.json key order —
                    # otherwise the values silently misalign with their names.
                    obj_result.append([values[parse_objective(o).column] for o in eig_objectives])
                    processed_keys.append(key)
                    break

            if len(processed_keys) == 0:
                error("Unfortunately, none survived. \n"
                      "This is most likely due to all generated initial geometries being degenerate.\n"
                      "Check the variable bounds or increase the number of initial geometries.\n"
                      "Tune ended.")
                return None

            df = df.loc[processed_keys]

            # Columns are the objectives themselves, in order — obj_result rows were
            # built in the same order, so names and values cannot drift apart.
            df[eig_objectives] = obj_result

        # Wakefield objective variables
        for o in self.objectives:
            if "ZL" in o[1] or "ZT" in o[1] or o[1] in ['k_FM [V/pC]', '|k_loss| [V/pC]', '|k_kick| [V/pC/m]',
                                                        'P_HOM [kW]']:
                wake_shape_space = self.run_wakefield_opt(df, self.wakefield_config, cavs_dict)

                df_wake, processed_keys = get_wakefield_objectives_value(wake_shape_space,
                                                                         self.objectives_unprocessed,
                                                                         Path(cavs_object.projectDir),
                                                                         key_subdir='tuned')
                df = df.merge(df_wake, on='key', how='inner')
                break

        # Apply UQ
        if self.uq_config:
            uq_result_dict = {}
            # Objectives are polarisation-qualified ('monopole:Epk/Eacc []');
            # EIGENMODE_QOIS holds the bare names, so compare on the suffix.
            def _bare(name):
                return name.split(':')[-1].strip()
            # 'key' is the dataframe INDEX throughout _evaluate_generation (as in
            # the non-UQ path), not a column — iterate/join on the index.
            for key in df.index:
                # Locate this candidate's uq.json via the cavity object (robust
                # to the project-vs-candidates-folder layout, like the eigenmode
                # objectives above): eigenmode and wakefield UQ both write to the
                # tuned cavity's uq/ dir.
                scav = cavs_dict.get(key)
                uq_json = None
                if scav is not None:
                    tuned = scav.tuned
                    candidates = ([Path(tuned.self_dir) / 'uq' / 'uq.json']
                                  if tuned is not None else [])
                    candidates.append(Path(scav.self_dir) / 'uq' / 'uq.json')
                    uq_json = next((c for c in candidates if c.exists()), None)
                if uq_json is None:
                    continue
                filename_eigen = filename_abci = uq_json
                if os.path.exists(filename_eigen):
                    uq_result_dict[key] = []
                    with open(filename_eigen, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if _bare(o[1]) in EIGENMODE_QOIS:
                                expe = uq_d[o[1]]['expe'][0]
                                std = uq_d[o[1]]['stdDev'][0]
                                uq_result_dict[key].append(expe)
                                uq_result_dict[key].append(std)
                                uq_result_dict[key].append(_uq_robust_value(o, expe, std))

                if os.path.exists(filename_abci):
                    if key not in uq_result_dict:
                        uq_result_dict[key] = []
                    with open(filename_abci, "r") as infile:
                        uq_d = json.load(infile)
                        for o in self.objectives:
                            if _bare(o[1]) not in EIGENMODE_QOIS:
                                expe = uq_d[o[1]]['expe'][0]
                                std = uq_d[o[1]]['stdDev'][0]
                                uq_result_dict[key].append(expe)
                                uq_result_dict[key].append(std)
                                uq_result_dict[key].append(_uq_robust_value(o, expe, std))

            uq_column_names = []
            for o in self.objectives:
                uq_column_names.append(fr'E[{o[1]}]')
                uq_column_names.append(fr'std[{o[1]}]')
                uq_column_names.append(_uq_col_name(o))

            df_uq = pd.DataFrame.from_dict(uq_result_dict, orient='index')

            require(len(df_uq) > 0, 'Unfortunately, no geometry was returned from uq, optimisation terminated.')
            df_uq.columns = uq_column_names
            # df_uq is keyed by the same candidate key as df's index -> join on it.
            df = df.join(df_uq, how='inner')

        # Filter by constraints
        for col, op_func, val in self.constraints:
            df = df.loc[op_func(df[col], val)]

        # Merge with global (elite archive) dataframe
        if not self.df_global.empty:
            df = pd.concat([self.df_global, df], ignore_index=True)
            # Drop exact duplicates that may arise from elitism
            df = df.drop_duplicates(subset=list(self.bounds.keys()), keep='first').reset_index(drop=True)

        # Rank shapes by objectives
        df['total_rank'] = 0.0

        for i, obj in enumerate(self.objectives):
            if self.uq_config:
                col = _uq_col_name(obj)
                ascending = (obj[0] != 'max')
                df[f'rank_{col}'] = df[col].rank(ascending=ascending) * self.weights[i]
                df['total_rank'] = df['total_rank'] + df[f'rank_{col}']
            else:
                if obj[0] == "min":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank() * self.weights[i]
                elif obj[0] == "max":
                    df[f'rank_{obj[1]}'] = df[obj[1]].rank(ascending=False) * self.weights[i]
                elif obj[0] == "equal" and obj[1] != 'freq [MHz]':
                    df[f'rank_{obj[1]}'] = (df[obj[1]] - obj[2]).abs().rank() * self.weights[i]

                df['total_rank'] = df['total_rank'] + df[f'rank_{obj[1]}']

        # Normalize and sort
        tot = df.pop('total_rank')
        df['total_rank'] = tot / sum(self.weights)

        df = df.sort_values(by=['total_rank']).reset_index(drop=True)

        # Pareto front
        reorder_indx, pareto_indx_list = self.pareto_front(df)

        # Estimate convergence via hypervolume indicator
        if self.uq_config:
            obj_cols = [_uq_col_name(o) for o in self.objectives]
        else:
            obj_cols = self.objective_vars

        pareto_vals = df.loc[pareto_indx_list, obj_cols].values.copy()
        if len(pareto_vals) > 0:
            # Transform all objectives to minimisation so hypervolume is well-defined:
            #   min  → keep as-is
            #   max  → negate (minimise the negative)
            #   equal → use |value - target| (minimise distance)
            for i, obj in enumerate(self.objectives):
                if obj[0] == 'max':
                    pareto_vals[:, i] = -pareto_vals[:, i]
                elif obj[0] == 'equal':
                    pareto_vals[:, i] = np.abs(pareto_vals[:, i] - obj[2])

            # Set reference point on first generation; expand if later fronts exceed it
            obj_max = np.max(pareto_vals, axis=0)
            if self.hv_ref is None:
                obj_min = np.min(pareto_vals, axis=0)
                span = obj_max - obj_min
                margin = np.where(span > 0, 0.1 * span, 0.1 * np.abs(obj_max) + 1.0)
                self.hv_ref = obj_max + margin
            elif np.any(obj_max > self.hv_ref):
                # A Pareto point exceeds the reference — expand it so no point is lost
                new_ref = np.maximum(self.hv_ref, obj_max * 1.1 + 1.0)
                info(f"Expanding HV reference point: {self.hv_ref} -> {new_ref}")
                self.hv_ref = new_ref

            hv = compute_hypervolume(pareto_vals, self.hv_ref)
            self.hv_history.append(hv)

        df = df.loc[reorder_indx, :]
        df = df.dropna().reset_index(drop=True)

        self.df_global = df

        if self.df_global.shape[0] == 0:
            error("Unfortunately, none survived the constraints and the program has to end.")
            return None
        done(self.df_global)

        # Save dataframe
        if self.opt_solver is not None:
            save_folder = self.opt_solver.folder
        else:
            save_folder = Path(self.cav.self_dir) / 'optimisation'
        generations_folder = save_folder / 'generations'
        generations_folder.mkdir(parents=True, exist_ok=True)
        filename = generations_folder / f'g{n}.xlsx'
        self.recursive_save(self.df_global, filename, reorder_indx)

        # Save pareto front and history for the solver
        # After reorder + reset_index, Pareto-optimal rows are the first self.poc rows
        pareto_df = self.df_global.iloc[:self.poc].copy()
        pareto_df['generation'] = n
        self.pareto_history.append(pareto_df)

        if self.opt_solver is not None:
            try:
                self.df_global.to_csv(save_folder / 'history.csv', index=False)
                pareto_df.to_csv(save_folder / 'pareto_front.csv', index=False)

                # Save full pareto history (all generations)
                pareto_hist_df = pd.concat(self.pareto_history, ignore_index=True)
                pareto_hist_df.to_csv(save_folder / 'pareto_history.csv', index=False)

                # Save objective metadata so plots know which columns to use
                obj_meta = {
                    'objectives': self.objectives,
                    'objective_vars': self.objective_vars,
                    'uq_config': bool(self.uq_config),
                }
                with open(save_folder / 'objective_meta.json', 'w') as f:
                    json.dump(obj_meta, f, indent=2)
            except Exception:
                pass

        return df

    def run_tune_opt(self, cav_dict, tune_config):
        processes = tune_config.get('processes', 1)
        if processes <= 0:
            error('Number of processes must be greater than zero.')
            processes = 1

        # Integrity-based skip: any candidate whose tuned results already
        # exist on disk (from a prior interrupted run) is dropped from the
        # tune batch. Its .tuned folder will still be picked up below for
        # eigenmode + downstream objective collection.
        to_tune = cav_dict
        if self.resume:
            to_tune = {k: c for k, c in cav_dict.items()
                       if not self._candidate_tune_complete(c)}
            if len(to_tune) < len(cav_dict):
                info(f'Resume: skipping tune on '
                     f'{len(cav_dict) - len(to_tune)} / {len(cav_dict)} '
                     f'candidate(s) with existing tune_res.json.')

        if to_tune:
            run_tune_parallel(to_tune, tune_config)

        # After tuning, each candidate has a <self_dir>/tuned/ folder.
        # Force-refresh the lazy `.tuned` accessor (the parent process's
        # cavity objects were not mutated by the mp children) and then
        # run eigenmode on the *tuned* cavities so qois.json lands in
        # <self_dir>/tuned/eigenmode/.
        tuned_cavs_dict = {}
        for key, cav in cav_dict.items():
            cav._tuned_cavity = None  # force lazy reload from disk
            tuned = cav.tuned
            if tuned is not None:
                tuned_cavs_dict[key] = tuned

        if tuned_cavs_dict and self.eigenmode_config:
            eig_cfg = dict(self.eigenmode_config)
            eig_cfg['target'] = run_eigenmode_s
            # On resume, let eigenmode reuse any pre-existing qois.json.
            if self.resume:
                eig_cfg.setdefault('rerun', False)
            run_eigenmode_parallel(tuned_cavs_dict, eig_cfg, self.projectDir)

    @staticmethod
    def _candidate_tune_complete(cav):
        """Integrity check: does this candidate already have a valid tune_res.json?"""
        try:
            tune_res = Path(cav.self_dir) / 'tuned' / 'tune_info' / 'tune_res.json'
            if not tune_res.exists() or tune_res.stat().st_size == 0:
                return False
            with open(tune_res, 'r') as f:
                data = json.load(f)
            return bool(data)
        except Exception:
            return False

    def run_wakefield_opt(self, df, wakefield_config, cavs_dict):
        """Run wakefield analysis on the tuned cavities.

        Returns a shape_space dict keyed by candidate name so
        ``get_wakefield_objectives_value`` can iterate it and locate the
        ABCI output under ``<projectDir>/<key>/tuned/wakefield/``.
        """
        wakefield_config_keys = wakefield_config.keys()
        MROT = 2
        MT = 10
        NFS = 10000
        wakelength = 50
        bunch_length = 25
        DDR_SIG = 0.1
        DDZ_SIG = 0.1

        wakefield_config.setdefault('beam_config', {})
        wakefield_config.setdefault('wake_config', {})
        wakefield_config.setdefault('mesh_config', {})

        if 'bunch_length' in wakefield_config['beam_config']:
            require(not isinstance(wakefield_config['beam_config']['bunch_length'], str),
                    'Bunch length must be of type integer or float.')
        else:
            wakefield_config['beam_config']['bunch_length'] = bunch_length
        if 'wakelength' in wakefield_config['wake_config']:
            require(not isinstance(wakefield_config['wake_config']['wakelength'], str),
                    'Wakelength must be of type integer or float.')
        else:
            wakefield_config['wake_config']['wakelength'] = wakelength

        processes = wakefield_config.get('processes', 1)
        if processes <= 0:
            error('Number of processes must be greater than zero.')
            processes = 1
        wakefield_config['processes'] = processes

        if 'polarisation' in wakefield_config_keys:
            require(wakefield_config['polarisation'] in [0, 1, 2], 'Polarisation should be 0, 1, or 2.')
        else:
            wakefield_config['polarisation'] = MROT

        if 'MT' not in wakefield_config_keys:
            wakefield_config['MT'] = MT
        if 'NFS' not in wakefield_config_keys:
            wakefield_config['NFS'] = NFS
        if 'DDR_SIG' not in wakefield_config['mesh_config']:
            wakefield_config['mesh_config']['DDR_SIG'] = DDR_SIG
        if 'DDZ_SIG' not in wakefield_config['mesh_config']:
            wakefield_config['mesh_config']['DDZ_SIG'] = DDZ_SIG

        # Only run wakefield on candidates that survived tuning.
        survived_keys = df['key'].tolist() if 'key' in df.columns else list(df.index)
        tuned_cavs_dict = {}
        for key in survived_keys:
            cav = cavs_dict.get(key)
            if cav is None:
                continue
            tuned = cav.tuned
            if tuned is not None:
                tuned_cavs_dict[key] = tuned

        if tuned_cavs_dict:
            wk_cfg = dict(wakefield_config)
            wk_cfg['target'] = run_wakefield_s
            run_wakefield_parallel(tuned_cavs_dict, wk_cfg)

        # Build a shape_space dict for `get_wakefield_objectives_value`
        # (it only iterates keys — the values are unused).
        return {key: None for key in tuned_cavs_dict}

    def generate_initial_population(self, initial_points, n):
        method_name = list(self.method.keys())[0]

        if method_name == "LHS":
            seed = self.method['LHS'].get('seed') or None

            columns = list(self.bounds.keys())
            dim = len(columns)
            l_bounds = np.array(list(self.bounds.values()))[:, 0]
            u_bounds = np.array(list(self.bounds.values()))[:, 1]

            const_var = []
            for i in range(dim - 1, -1, -1):
                if l_bounds[i] == u_bounds[i]:
                    const_var.append([columns[i], l_bounds[i]])
                    del columns[i]
                    l_bounds = np.delete(l_bounds, i)
                    u_bounds = np.delete(u_bounds, i)

            reduced_dim = len(columns)
            sampler = qmc.LatinHypercube(d=reduced_dim, scramble=False, seed=seed)
            _ = sampler.reset()
            sample = sampler.random(n=initial_points)
            self.discrepancy = qmc.discrepancy(sample)

            sample = qmc.scale(sample, l_bounds, u_bounds)

            df = pd.DataFrame()
            df['key'] = [f"G{n}_C{i}_P" for i in range(initial_points)]
            df[columns] = sample

            for i in range(len(const_var) - 1, -1, -1):
                df[const_var[i][0]] = np.ones(initial_points) * const_var[i][1]

            return df.set_index('key')

        elif method_name == "Sobol Sequence":
            seed = self.method['Sobol Sequence'].get('seed') or None

            columns = list(self.bounds.keys())
            dim = len(columns)
            index = self.method["Sobol Sequence"]['index']
            l_bounds = np.array(list(self.bounds.values()))[:, 0]
            u_bounds = np.array(list(self.bounds.values()))[:, 1]

            const_var = []
            for i in range(dim - 1, -1, -1):
                if l_bounds[i] == u_bounds[i]:
                    const_var.append([columns[i], l_bounds[i]])
                    del columns[i]
                    l_bounds = np.delete(l_bounds, i)
                    u_bounds = np.delete(u_bounds, i)

            reduced_dim = len(columns)
            sampler = qmc.Sobol(d=reduced_dim, scramble=False, seed=seed)
            _ = sampler.reset()
            sample = sampler.random_base2(m=index)
            sample = qmc.scale(sample, l_bounds, u_bounds)

            df = pd.DataFrame()
            df['key'] = [f"G0_C{i}_P" for i in range(initial_points)]
            df[columns] = sample

            for i in range(len(const_var) - 1, -1, -1):
                df[const_var[i][0]] = np.ones(initial_points) * const_var[i][1]

            return df.set_index('key')

        elif method_name == "Random":
            data = {'key': [f"G{n}_C{i}_P" for i in range(initial_points)]}
            for var, bounds in self.bounds.items():
                data[var] = random.sample(
                    list(np.linspace(bounds[0], bounds[1], initial_points * 2)),
                    initial_points)
            return pd.DataFrame.from_dict(data).set_index('key')

        elif method_name == "Uniform":
            data = {'key': [f"G{n}_C{i}_P" for i in range(initial_points)]}
            for var, bounds in self.bounds.items():
                data[var] = np.linspace(bounds[0], bounds[1], initial_points)
            return pd.DataFrame.from_dict(data).set_index('key')

    @staticmethod
    def process_constraints(constraints):
        """Parse constraint dict into (column, operator_func, value) tuples."""
        processed = []
        for key, bounds in constraints.items():
            if isinstance(bounds, list):
                if len(bounds) == 2:
                    processed.append((key, operator.gt, bounds[0]))
                    processed.append((key, operator.lt, bounds[1]))
                else:
                    processed.append((key, operator.gt, bounds[0]))
            else:
                processed.append((key, operator.eq, bounds))
        return processed

    def crossover(self, df, generation, n_offspring):
        """Simulated Binary Crossover (SBX) between tournament-selected parents."""
        vars_list = list(self.bounds.keys())
        l_bounds = np.array([self.bounds[v][0] for v in vars_list])
        u_bounds = np.array([self.bounds[v][1] for v in vars_list])
        eta = self.eta_sbx
        pool_size = min(self.elites_to_crossover, len(df))

        rows = []
        for i in range(int(n_offspring)):
            # Tournament selection: pick 2 distinct parents from top-ranked pool
            idx = np.random.choice(pool_size, size=2, replace=False)
            p1 = df.iloc[idx[0]][vars_list].values.astype(float)
            p2 = df.iloc[idx[1]][vars_list].values.astype(float)

            # SBX operator (per variable)
            child = np.empty_like(p1)
            for j in range(len(vars_list)):
                if l_bounds[j] == u_bounds[j]:
                    child[j] = p1[j]
                    continue

                u = np.random.random()
                if u <= 0.5:
                    beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                else:
                    beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

                child[j] = 0.5 * ((1 + beta) * p1[j] + (1 - beta) * p2[j])

            # Clip to bounds
            child = np.clip(child, l_bounds, u_bounds)
            rows.append(child)

        df_co = pd.DataFrame(rows, columns=vars_list)
        df_co.index = [f"G{generation}_C{i}_CO" for i in range(n_offspring)]
        df_co.index.name = 'key'
        return df_co

    def mutation(self, df, n, n_mutants):
        """Gaussian mutation: perturb top individuals with per-variable Gaussian noise, clipped to bounds."""
        vars_list = list(self.bounds.keys())
        l_bounds = np.array([self.bounds[v][0] for v in vars_list])
        u_bounds = np.array([self.bounds[v][1] for v in vars_list])
        ranges = u_bounds - l_bounds
        sigma = self.mutation_sigma

        n_parents = int(min(n_mutants, len(df)))

        rows = []
        for i in range(n_parents):
            parent = df.iloc[i][vars_list].values.astype(float)
            noise = np.random.normal(0, sigma, size=len(vars_list)) * ranges
            # Don't mutate fixed variables
            noise[ranges == 0] = 0.0
            child = np.clip(parent + noise, l_bounds, u_bounds)
            rows.append(child)

        df_mut = pd.DataFrame(rows, columns=vars_list)
        df_mut.index = [f"G{n}_C{i}_M" for i in range(n_parents)]
        df_mut.index.name = 'key'
        return df_mut

    def chaos(self, f, n):
        return self.generate_initial_population(f, n)

    @staticmethod
    def remove_duplicate_values(d):
        temp = []
        res = dict()
        for key, val in d.items():
            if val not in temp:
                temp.append(val)
                res[key] = val
        return res

    @staticmethod
    def proof_filename(filepath):
        if filepath.split('.')[-1] != 'json':
            filepath = f'{filepath}.json'
        return filepath

    def recursive_save(self, df, filename, pareto_index):
        styler = self.color_pareto(df, self.poc)
        try:
            styler.to_excel(filename)
        except PermissionError:
            filename = Path(filename)
            filename = filename.with_name(f'{filename.stem}_1.xlsx')
            self.recursive_save(df, filename, pareto_index)

    def pareto_front(self, df):
        if self.uq_config:
            obj = [_uq_col_name(o) for o in self.objectives]
            datapoints = df.loc[:, obj]
        else:
            datapoints = df.loc[:, self.objective_vars]

        sense = []
        for o in self.objectives:
            if o[0] == 'min':
                sense.append('min')
            elif o[0] == "equal":
                sense.append('diff')
            elif o[0] == 'max':
                sense.append('max')

        bool_array = paretoset(datapoints, sense=sense)
        lst = np.where(bool_array)[0]
        self.poc = len(lst)

        reorder_idx = list(lst) + [i for i in range(len(df)) if i not in lst]
        return reorder_idx, lst

    @staticmethod
    def negate_list(ll, arg):
        if arg == 'max':
            return ll
        else:
            return [-x for x in ll]

    @staticmethod
    def overwriteFolder(invar, projectDir):
        path = Path(projectDir) / '_optimisation' / f'_process_{invar}'
        if path.exists():
            shutil.rmtree(path)
            # Flush any internal caches if necessary
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copyFiles(invar, parentDir, projectDir):
        src = Path(parentDir) / 'exe' / 'SLANS_exe'
        dst = Path(projectDir) / '_optimisation' / f'_process_{invar}' / 'SLANS_exe'
        shutil.copytree(src, dst, dirs_exist_ok=True)

    @staticmethod
    def color_pareto(df, no_pareto_optimal):
        def color(row):
            if row.iloc[0] in df.index.tolist()[0:no_pareto_optimal]:
                return ['background-color: #6bbcd1'] * len(row)
            return [''] * len(row)

        styler = df.style
        styler.apply(color, axis=1)
        return styler
