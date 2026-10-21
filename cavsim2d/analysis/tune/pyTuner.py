import json
import os.path
from itertools import groupby

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from cavsim2d.utils.data_utils import get_qoi_value
from cavsim2d.utils.printing import error, suppress_errors, warning
from cavsim2d.utils.quadrature import cn_gauss, cn_leg_05_2, quad_stroud3, weighted_mean_obj
from cavsim2d.utils.shapes import enforce_Req_continuity
from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP
from cavsim2d.solvers.eigenmode_result import monopole_dir

ngsolve_mevp = NGSolveMEVP()


class _TargetFreqReached(Exception):
    """Raised inside the secant to stop as soon as the FREQUENCY is within
    tolerance — decoupling frequency convergence from the tune-variable step."""

# Exceptions that mean "the code or the cavity type is wrong", not "this candidate
# geometry did not tune". They must propagate rather than be reported as a
# degenerate geometry (tune_function) or swallowed into a silent no-op
# (processes.tune), both of which left cav.tuned = None while run_tune() returned
# normally.
TUNE_DEFECTS = (NotImplementedError, TypeError, AttributeError, KeyError,
                ImportError, UnicodeError, IndexError)


def _step_eigenmode_config(tune_config, **extra):
    """The eigenmode config for one tuning solve: the caller's mesh, plus *extra*.

    Only the mesh comes from ``tune_config['eigenmode_config']``. The boundary
    conditions and polarisation belong to the stage (a mid cell is solved between
    two PMC planes, for its pi-mode), and the rest of that config is for the
    eigenmode run on the tuned cavity afterwards. Without the mesh the steps ran on
    the solver's default, and a cavity tuned "with" ``{'h': 6, 'p': 2}`` came out
    several 1e-2 MHz off target when that run re-solved it on the mesh asked for.
    """
    cfg = dict(extra)
    mesh = (tune_config.get('eigenmode_config') or {}).get('mesh_config')
    if mesh:
        cfg['mesh_config'] = dict(mesh)
    return cfg or None


VAR_TO_INDEX_DICT = {'A': 0, 'B': 1, 'a': 2, 'b': 3, 'Ri': 4, 'L': 5, 'Req': 6, 'l': 7}
TUNE_VAR_STEP_DIRECTION_DICT = {'A': -1, 'B': 1, 'a': -1, 'b': 1, 'Ri': 1, 'L': 1, 'Req': -1, 'l': 1}
MAX_TUNE_ITERATION = 10

EIGENMODE_OBJECTIVES = {"Req", "freq [MHz]", "Epk/Eacc []", "Bpk/Eacc [mT/MV/m]",
                         "R/Q [Ohm]", "G [Ohm]", "Q []", "kcc [%]", "ff [%]"}


def _resolve_cell_type(cell_type, shape, perturbed=None):
    """Map cell_type string to (mid, left, right, beampipes) tuple.

    Parameters
    ----------
    cell_type : str
        One of 'mid-cell', 'end-cell', 'single-cell'.
    shape : dict
        Shape dictionary with 'IC' and 'OC' keys.
    perturbed : np.ndarray, optional
        If provided, used as the perturbed cell node (for UQ).
        If None, returns the base cell node from shape.

    Returns
    -------
    tuple : (mid, left, right, beampipes, cell_node)
    """
    ct = cell_type.lower().replace('-', ' ').replace('_', ' ')

    if ct == 'mid cell':
        cell_node = shape['IC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'none', shape['IC']
    elif ct == 'end cell':
        base = shape['IC'] if perturbed is None else shape['IC']
        end = shape['OC'] if perturbed is None else perturbed
        return base, base, end, 'right', shape['OC']
    elif ct == 'single cell':
        cell_node = shape['OC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'both', shape['OC']
    else:
        cell_node = shape['IC'] if perturbed is None else perturbed
        return cell_node, cell_node, cell_node, 'none', shape['IC']


class PyTuneNGSolve:
    def __init__(self):
        self.plot = None
        self.beampipe = None

    def tune(self, cav, tune_config=None):
        self.cav = cav
        self.tune_config = tune_config
        self.target_freq = self.cav.shape['FREQ']
        self.tune_var = self.tune_config['parameters']

        # `tol` is the FREQUENCY tolerance [MHz] — tuning stops when the frequency is
        # within it. `var_tol` is the SEPARATE tune-variable step tolerance the secant
        # uses; make it tight so the secant keeps refining until the frequency (not the
        # variable step) is met. (Previously the single `tol` bounded both, so the
        # secant could stop with the frequency still off-target.)
        tol = tune_config.get('tol', 1e-4)
        var_tol = tune_config.get('var_tol', 1e-8)
        maxiter = tune_config.get('maxiter', 10)
        self.tol = tol

        # Whether this cavity's parameters carry per-cell suffixes (`Req_m`, `Req_el`,
        # `Req_er`). Ask the model, don't compare `kind` strings: `kind` is
        # 'elliptical cavity flat top' for the flattop, which used to fall through
        # this branch and then look up a bare 'Req' that does not exist.
        if cav.uses_cell_suffixes:
            # If tune_var is a bare name (no suffix), map it based on cell_types
            if '_m' not in self.tune_var and '_el' not in self.tune_var and '_er' not in self.tune_var:
                ct = tune_config.get('cell_types', 'mid-cell').lower().replace('-', ' ').replace('_', ' ')
                if ct in ('mid cell',):
                    self.tune_var = f'{self.tune_var}_m'
                elif ct in ('end cell r', 'end cell right'):
                    self.tune_var = f'{self.tune_var}_er'
                elif ct in ('end cell', 'end cell l', 'end cell left', 'single cell'):
                    self.tune_var = f'{self.tune_var}_el'
                else:
                    self.tune_var = f'{self.tune_var}_m'

            # Use suffix-based dispatch — substring matching breaks on names
            # like 'Req_er' (which contains '_e').
            if self.tune_var.endswith('_m'):
                self.beampipe = 'none'
            elif self.tune_var.endswith('_el'):
                self.beampipe = 'left'
            elif self.tune_var.endswith('_er'):
                self.beampipe = 'right'

        self.freq_list = []
        self.tv_list = []
        self.abs_err_list = []
        self.convergence_list = []

        x0 = cav.get_tune_value(self.tune_var)
        self.x0_initial = x0  # Store for sanity check in tuner
        # Secant method needs two starting points
        x1 = x0 * 1.05  # 5% perturbation

        self._degen_count = 0
        # Secant probing routinely lands on parameter sets that yield
        # degenerate geometry — the tuner handles these via a penalty and
        # retreats. Silence the per-iteration "Parameter set leads to
        # degenerate geometry." error so only the final stage result is
        # reported. A terminal degeneracy still surfaces via the
        # ValueError branch below.
        aborted = False
        try:
            with suppress_errors('Parameter set leads to degenerate geometry'):
                res = root_scalar(
                    self.tune_function,
                    method='secant',
                    x0=x0,
                    x1=x1,
                    xtol=var_tol,
                    rtol=var_tol,
                    maxiter=maxiter
                )
                # Final evaluation at converged root
                self.tune_function(res.root)
            converged = bool(res.converged) if hasattr(res, 'converged') else True
            root_val = res.root
        except _TargetFreqReached:
            # Frequency reached within tol before the variable step converged — success.
            converged = True
            root_val = self._converged_x
        except ValueError as e:
            # Tuning aborted mid-iteration (e.g., repeated degeneracy).
            error(f'Tune aborted: {e}')
            converged = False
            aborted = True
            # Best-effort: pick the tv with smallest |diff| from the valid points collected.
            root_val = self._best_valid_tv_or_zero()

        self.convergence_list.extend([self.tv_list, self.freq_list])
        self.conv_dict = {f'{self.tune_var}': self.convergence_list[0], 'freq [MHz]': self.convergence_list[1]}

        # Diagnose unreachable targets: if the final achievable freq is
        # far from the target, surface that clearly rather than returning
        # a spurious result.
        if self._valid_freqs():
            best_freq, best_diff = self._best_valid_result()
            freq_tol_mhz = max(1e-2, 1e-3 * self.target_freq)  # 0.001% or 10 kHz
            if abs(best_diff) > freq_tol_mhz:
                min_f = min(self._valid_freqs())
                max_f = max(self._valid_freqs())
                # A variable that does not reach the geometry gives a sampled
                # range of essentially zero while the search runs its value off
                # to absurdity. That is a different diagnosis from "too weak",
                # and pointing at the geometry would send the user the wrong way:
                # the usual cause is a SHARED parameter addressed through a
                # per-cell slot that the model ignores (Req is one equator radius
                # for the whole cavity and lives in the mid-cell slot, so
                # cell_type={'end-cell': 'Req'} moves a value nothing reads).
                if max_f - min_f < 1e-6 * max(1.0, self.target_freq):
                    error(
                        f"Tune did not reach target {self.target_freq} MHz, and "
                        f"{self.tune_var} did not move the frequency at all "
                        f"(sampled range {max_f - min_f:.3g} MHz over "
                        f"{self.tune_var} up to {root_val:.3g}). That variable does "
                        f"not reach this geometry — check the name and the cell it "
                        f"is asked for; a shared parameter such as Req belongs in the "
                        f"mid-cell slot, not a per-cell one."
                    )
                else:
                    error(
                        f"Tune did not reach target {self.target_freq} MHz. "
                        f"Best |diff| = {abs(best_diff):.3f} MHz at {self.tune_var}={root_val}. "
                        f"Achievable freq range (sampled): [{min_f:.3f}, {max_f:.3f}] MHz. "
                        f"Target may be unreachable by varying {self.tune_var} alone — "
                        f"consider adjusting other geometry parameters (e.g., Req)."
                    )
                return 0, 0, self.conv_dict, self.abs_err_list

        if converged:
            # Find closest match in recorded tv_list (float eq after re-eval may differ)
            if root_val in self.tv_list:
                idx = self.tv_list.index(root_val)
            else:
                idx = int(np.argmin(np.abs(np.array(self.tv_list) - root_val)))
            return root_val, self.freq_list[idx], self.conv_dict, self.abs_err_list
        else:
            # An abort has already said why; running out of steps close to the
            # target used to return nothing without a word.
            if not aborted and self._valid_freqs():
                _, best_diff = self._best_valid_result()
                error(f"Tune did not converge in {maxiter} secant steps: the closest "
                      f"step was {abs(best_diff):.2e} MHz from {self.target_freq} MHz, "
                      f"outside the {tol:.2e} MHz tolerance. Raise "
                      f"tune_config['maxiter'] or loosen tune_config['tol'].")
            return 0, 0, self.conv_dict, self.abs_err_list

    def _valid_freqs(self):
        """Recorded frequencies from steps that weren't flagged degenerate."""
        return [f for f, ok in zip(self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]

    def _best_valid_result(self):
        """Return (freq, diff) for the valid step closest to target."""
        valid = [(f, f - self.target_freq) for f, ok in
                 zip(self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]
        if not valid:
            return self.target_freq, 0.0
        return min(valid, key=lambda fd: abs(fd[1]))

    def _best_valid_tv_or_zero(self):
        """Return the tune variable value of the best valid step, or 0 if none."""
        valid_pairs = [(tv, f) for tv, f, ok in
                       zip(self.tv_list, self.freq_list, getattr(self, '_valid_mask', [True]*len(self.freq_list))) if ok]
        if not valid_pairs:
            return 0
        best = min(valid_pairs, key=lambda tf: abs(tf[1] - self.target_freq))
        return best[0]

    def tune_function(self, x):
        x_val = float(x if isinstance(x, (float, int, np.floating)) else x[0])
        self.cav.set_tune_value(self.tune_var, x_val)
        self.tv_list.append(x_val)
        if not hasattr(self, '_valid_mask'):
            self._valid_mask = []

        bp = self.beampipe if self.beampipe else 'none'

        # End-cell stages tune the end-cell in its real context (beampipe +
        # end-cell + adjacent mid-cell half, PMC at both outer ends). The
        # quarter-cell geometry used for mid-cell tuning mirrors the end-cell
        # to itself, which excludes the pi-mode frequency the real multicell
        # cavity has — i.e. the target is unreachable.
        endcell_tune = bp in ('left', 'right')
        tune_mode = 'tune-endcell' if endcell_tune else 'tune'

        # Geometry creation and solve both can fail on degenerate parameters.
        # Rather than aborting the entire tune, record a penalty and let the
        # root finder retreat toward the valid region.
        #
        # But only *geometry* failures mean "degenerate". A TypeError from a
        # mismatched create() signature, or an AttributeError from a missing
        # writer, is a defect: reporting it as degeneracy sent the root finder
        # wandering and then blamed the user's parameters. Let those out.
        degenerate = False
        try:
            self.cav.create(1, bp, mode=tune_mode)
        except TUNE_DEFECTS:
            raise
        except Exception:
            degenerate = True

        # UQ tuning reads its own uq.json, so it keeps the full default solve.
        # Otherwise the secant needs only the frequency: defer the QOIs and fields
        # to the solve on the tuned cavity.
        #
        # Every step uses the default (direct) preconditioner. The steps far from
        # the target used to take BDDC, and one landing near the target was solved
        # again with direct. Once PINVIT stopped on its residual, a BDDC step needed
        # several times the iterations, and that scheme made a TESLA mid-cell tune
        # 1.4-1.9x slower than direct throughout, for the same tuned Req.
        uq = bool(self.tune_config.get('uq_config'))
        if not degenerate:
            degenerate = not self._solve_step(bp, endcell_tune, freq_only=not uq)

        if degenerate:
            return self._degenerate_penalty(x_val)

        if uq:
            run_tune_uq(self.cav, self.tune_config)

            with open(os.path.join(self.cav.self_dir, 'eigenmode', 'uq.json')) as json_file:
                eigenmode_qois = json.load(json_file)
            freq = eigenmode_qois['freq [MHz]']['expe'][0]
        else:
            freq = self._read_frequency()

        self.freq_list.append(freq)
        self._valid_mask.append(True)
        self._degen_count = 0

        diff = freq - self.target_freq
        self.abs_err_list.append(abs(diff))

        # Converge on FREQUENCY: stop the secant the moment the frequency is within
        # tolerance, regardless of how small the tune-variable step is.
        if abs(diff) <= self.tol:
            self._converged_x = x_val
            raise _TargetFreqReached()

        return diff

    def _read_frequency(self):
        mono_dir = monopole_dir(os.path.join(self.cav.self_dir, 'eigenmode'))
        with open(os.path.join(mono_dir, 'qois.json')) as json_file:
            return json.load(json_file)['freq [MHz]']

    def _solve_step(self, bp, endcell_tune, freq_only):
        """Solve the stage geometry once, on the caller's mesh. True on success.

        *freq_only* solves for the frequency alone; otherwise it is the full
        default solve (QOIs and fields).
        """
        orig_n_cells = self.cav.n_cells
        orig_beampipe = getattr(self.cav, 'beampipe', None)
        self.cav.n_cells = 1
        # The eigensolver meshes cav.profile(), which reads cav.beampipe /
        # cav.n_cells — NOT the reduced .geo that create(mode=...) just
        # wrote. Without this override a mid-cell tune on a cavity built
        # with beampipe='both' would solve a single cell WITH both
        # beampipes (a different, beampipe-loaded frequency) instead of the
        # bare periodic mid-cell, converging on the wrong Req. Pin the
        # cavity's geometry context to this stage's intended beampipe for
        # the duration of the solve.
        if orig_beampipe is not None:
            self.cav.beampipe = bp

        # ...and pin the OPPOSITE end cell to the mid cell for the same
        # reason. profile() reads self.parameters live, and for n_cells=1
        # half_cells() returns [end_l, end_r] -- so without this the end-cell
        # stage solves end+end (the end cell mirrored onto itself) and never
        # sees the mid cup it actually adjoins. That is the model
        # write_endcell_tune_geometry was written to avoid, and its
        # substitution was being discarded because the solver meshes
        # profile() rather than the .geo create() just wrote.
        #
        # The result is cell 1 of the real cavity in isolation:
        #   beampipe | end-cup | equator | mid-cup | iris
        # with PMC at the outer iris plane -- both cut planes are genuine
        # pi-mode symmetry planes.
        orig_end_params = None
        if endcell_tune:
            other = '_er' if bp == 'left' else '_el'
            keys = [f'{n}{other}' for n in
                    ('A', 'B', 'a', 'b', 'Ri', 'L', 'Req')]
            orig_end_params = {k: self.cav.parameters[k]
                               for k in keys if k in self.cav.parameters}
            for k in orig_end_params:
                self.cav.parameters[k] = self.cav.parameters[
                    f'{k[:-len(other)]}_m']
        try:
            if freq_only:
                cfg = _step_eigenmode_config(self.tune_config, freq_only=True)
            else:
                cfg = _step_eigenmode_config(self.tune_config)
            res = ngsolve_mevp.solve(self.cav, eigenmode_config=cfg)
        except Exception:
            res = False
        finally:
            self.cav.n_cells = orig_n_cells
            if orig_beampipe is not None:
                self.cav.beampipe = orig_beampipe
            if orig_end_params:
                self.cav.parameters.update(orig_end_params)
        return bool(res)

    def _degenerate_penalty(self, x_val):
        """Handle degenerate geometry step by returning a penalty that
        nudges the secant method back toward the valid region.

        Strategy: keep the sign of the last valid diff and inflate magnitude
        so the secant extrapolation retreats. After too many consecutive
        failures, raise to terminate tuning.
        """
        self._degen_count = getattr(self, '_degen_count', 0) + 1

        # Find the last valid freq (if any)
        last_valid_freq = None
        for f, ok in zip(reversed(self.freq_list), reversed(self._valid_mask)):
            if ok:
                last_valid_freq = f
                break

        if last_valid_freq is not None:
            last_diff = last_valid_freq - self.target_freq
            # Keep sign, amplify magnitude so secant retreats toward last valid x.
            sign = 1.0 if last_diff >= 0 else -1.0
            penalty = sign * max(abs(last_diff) * 10.0, 1e3)
            placeholder_freq = last_valid_freq
        else:
            # No valid anchor yet — default to a large positive penalty.
            penalty = 1e3
            placeholder_freq = self.target_freq + penalty

        # Record the failed step in all tracking lists to keep them aligned.
        self.freq_list.append(placeholder_freq)
        self._valid_mask.append(False)
        self.abs_err_list.append(abs(penalty))

        if self._degen_count >= 3:
            raise ValueError(
                f'Geometry degenerated {self._degen_count} consecutive times at '
                f'{self.tune_var}~={x_val:.4g}. Target frequency may be unreachable.'
            )
        return penalty

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    @staticmethod
    def write_output(tv_list, freq_list, fid, projectDir):
        dd = {"tv": tv_list, "freq": freq_list}
        with open(os.path.join(projectDir, '_tune_temp', fid, "convergence_output.json"), "w") as outfile:
            json.dump(dd, outfile, indent=4, separators=(',', ': '))


def run_tune_uq(cav, tune_config):
    """Run eigenmode UQ at the current tune iteration.

    Perturbs the requested variables around ``cav.parameters`` over a
    quadrature rule, solves eigenmode at each node, and writes the
    weighted mean/std/skew/kurtosis of ``freq [MHz]`` to
    ``<cav.self_dir>/eigenmode/uq.json``. The tune loop then reads the
    expected frequency from that file in place of a single deterministic
    solve.
    """
    uq_config = tune_config['uq_config']

    uq_cell_complexity = uq_config.get('cell_complexity',
                                       uq_config.get('cell complexity', 'simplecell'))
    if uq_cell_complexity == 'multicell':
        # Not wired up in the tuning flow. This used to skip the UQ step with a
        # message nobody saw (warning() was silent by default), and the tune then
        # failed looking for the uq.json the step never wrote.
        raise NotImplementedError(
            "UQ-averaged tuning supports uq_config['cell_complexity'] = 'simplecell' "
            "only. Tune without uq_config, then run a multicell UQ on cav.tuned.")

    uq_vars = uq_config['variables']
    delta = uq_config.get('delta', [0.05] * len(uq_vars))
    method = uq_config.get('method', ['Quadrature', 'Stroud3'])

    # Resolve bare variable names (A, B, ...) to the suffixed parameters
    # that actually live on cav.parameters (A_m, A_el, ...). During a
    # multi-stage tune the per-stage config passes ``cell_types`` (singular
    # value) to pyTuner; UQ overrides via ``uq_config['cell_type']``.
    cell_type = uq_config.get('cell_type', tune_config.get('cell_types', 'mid-cell'))
    ct = cell_type.lower().replace('-', ' ').replace('_', ' ')
    if ct in ('end cell r', 'end cell right'):
        suffix = '_er'
    elif ct in ('end cell', 'end cell l', 'end cell left', 'single cell'):
        suffix = '_el'
    else:
        suffix = '_m'

    def _resolve(v):
        if v.endswith(('_m', '_el', '_er')):
            return v
        return f'{v}{suffix}'

    resolved_vars = [_resolve(v) for v in uq_vars]

    # Beampipe setting matching the suffix of the perturbed cell.
    bp_map = {'_m': 'none', '_el': 'left', '_er': 'right'}
    bp = bp_map.get(suffix, 'none')

    # Quadrature nodes / weights.
    rdim = len(resolved_vars)
    method_name = method[1].lower() if len(method) > 1 else 'stroud3'
    if method_name == 'stroud3':
        nodes_, weights_, _ = quad_stroud3(rdim, 1)
        nodes_ = 2. * nodes_ - 1.
    elif method_name == 'stroud5':
        nodes_, weights_ = cn_leg_05_2(rdim)
    elif method_name == 'gaussian':
        nodes_, weights_ = cn_gauss(rdim, 2)
    else:
        warning(f'UQ integration method {method!r} not recognised; '
                'defaulting to Stroud3.')
        nodes_, weights_, _ = quad_stroud3(rdim, 1)
        nodes_ = 2. * nodes_ - 1.

    n_sims = nodes_.shape[1]
    base_params = dict(cav.parameters)
    freqs = []

    eigenmode_dir = os.path.join(cav.self_dir, 'eigenmode')
    os.makedirs(eigenmode_dir, exist_ok=True)

    # Read through the model's accessor: a variable need not be a plain scalar
    # slot in `parameters` (a spline's 'p3_r' lives inside a control point).
    base_values = {rvar: cav.get_tune_value(rvar) for rvar in resolved_vars}

    with suppress_errors('Parameter set leads to degenerate geometry'):
        for j in range(n_sims):
            # Perturb in-place; cav is restored after the loop.
            for i, rvar in enumerate(resolved_vars):
                cav.set_tune_value(
                    rvar, base_values[rvar] * (1 + delta[i] * nodes_[i, j]))

            ok = False
            orig_n_cells = cav.n_cells
            cav.n_cells = 1
            # End-cell UQ uses the hybrid (end-cell + adjacent mid-cell half)
            # geometry for the same reason the non-UQ tune path does.
            uq_tune_mode = 'tune-endcell' if bp in ('left', 'right') else 'tune'
            try:
                cav.create(1, bp, mode=uq_tune_mode)
                ok = bool(ngsolve_mevp.solve(
                    cav, eigenmode_config=_step_eigenmode_config(tune_config)))
            except Exception as e:
                warning(f'UQ node {j} failed to solve ({e!r}); dropping from quadrature.')
                ok = False
            finally:
                cav.n_cells = orig_n_cells

            if ok:
                with open(os.path.join(monopole_dir(eigenmode_dir), 'qois.json')) as f:
                    q = json.load(f)
                freqs.append(q['freq [MHz]'])
            else:
                freqs.append(np.nan)

    # Restore the base parameters so the outer tune loop continues from
    # the same state it entered with.
    cav.parameters.update(base_params)

    freqs_arr = np.array(freqs, dtype=float).reshape(-1, 1)
    if np.isnan(freqs_arr).all():
        raise ValueError('All UQ eigenmode solves failed — cannot aggregate.')
    valid_mask = ~np.isnan(freqs_arr[:, 0])
    valid_freqs = freqs_arr[valid_mask]
    valid_weights = np.asarray(weights_)[valid_mask]
    # Renormalise weights over surviving nodes.
    valid_weights = valid_weights / valid_weights.sum()

    mean_obj, std_obj, skew_obj, kurtosis_obj = weighted_mean_obj(valid_freqs, valid_weights)

    result = {
        'freq [MHz]': {
            'expe': [float(mean_obj[0])],
            'stdDev': [float(std_obj[0])],
            'skew': [float(skew_obj[0])],
            'kurtosis': [float(kurtosis_obj[0])],
        }
    }
    with open(os.path.join(eigenmode_dir, 'uq.json'), 'w') as f:
        json.dump(result, f, indent=4)

    # Diagnostic dump: per-node freqs + nodes + weights for this tune iter.
    debug_rows = {
        'node': list(range(n_sims)),
        'freq [MHz]': [float(v) if np.isfinite(v) else None for v in freqs],
        'weight': [float(w) for w in np.asarray(weights_).ravel()],
    }
    for i, rv in enumerate(resolved_vars):
        debug_rows[rv] = [float(base_params[rv] * (1 + delta[i] * nodes_[i, j]))
                          for j in range(n_sims)]
    debug_rows['Req_m (base)'] = [float(base_params.get('Req_m', np.nan))] * n_sims
    debug_rows['mean freq [MHz]'] = [float(mean_obj[0])] * n_sims
    pd.DataFrame(debug_rows).to_csv(os.path.join(eigenmode_dir, 'uq_nodes.csv'),
                                    index=False, sep='\t', float_format='%.6f')


def uq_tuner(key, objectives, uq_config, uq_path, solver_args_dict, sub_dir,
             proc_keys_list, processor_nodes, proc_num, cell_node, solver):
    """Run UQ eigenmode solves for a subset of quadrature points.

    Parameters
    ----------
    key : str
        Cavity geometry identifier.
    objectives : list
        QOIs for UQ.
    uq_config : dict
        UQ configuration.
    uq_path : Path
        Output directory.
    solver_args_dict : dict
        Solver arguments.
    sub_dir : str
        Subdirectory for results.
    proc_keys_list : np.ndarray
        Indices of quadrature points for this process.
    processor_nodes : np.ndarray
        Quadrature nodes for this process.
    proc_num : int
        Process number.
    cell_node : np.ndarray
        Base cell parameters.
    solver : str
        Solver type.
    """
    if solver != 'eigenmode':
        return

    parentDir = solver_args_dict['parentDir']
    projectDir = solver_args_dict['projectDir']
    cell_type = uq_config['cell_type']
    analysis_folder = solver_args_dict['analysis folder']
    opt = solver_args_dict['optimisation']
    delta = uq_config['delta']
    uq_vars = uq_config['variables']

    eigen_obj_list = [o for o in objectives if o in EIGENMODE_OBJECTIVES]
    Ttab_val_f = []

    perturbed_cell_node = np.array(cell_node)
    for i1, proc_key in enumerate(proc_keys_list):
        for j, uq_var in enumerate(uq_vars):
            uq_var_indx = VAR_TO_INDEX_DICT[uq_var]
            perturbed_cell_node[uq_var_indx] = cell_node[uq_var_indx] * (1 + delta[j] * processor_nodes[j, i1])

        ct = cell_type.lower().replace('-', ' ').replace('_', ' ')
        if ct == 'mid cell':
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'none'
        elif ct == 'end cell':
            mid, left, right, beampipes = cell_node, cell_node, perturbed_cell_node, 'right'
        elif ct == 'single cell':
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'both'
        else:
            mid, left, right, beampipes = perturbed_cell_node, perturbed_cell_node, perturbed_cell_node, 'none'

        enforce_Req_continuity(mid, left, right, cell_type)

        fid = fr'{key}_Q{proc_key}'
        ngsolve_mevp.createFolder(fid, projectDir, subdir=sub_dir, opt=opt)
        ngsolve_mevp.cavity(1, 1, mid, left, right, f_shift=0, bc=33, beampipes=beampipes,
                            fid=fid, sim_folder=analysis_folder, parentDir=parentDir,
                            projectDir=projectDir, subdir=sub_dir)

        filename = uq_path / f'{fid}/monopole/qois.json'
        if os.path.exists(filename):
            with open(filename) as json_file:
                qois_result_dict = json.load(json_file)
            qois_result = get_qoi_value(qois_result_dict, eigen_obj_list)
            Ttab_val_f.append(qois_result)

    data_table = pd.DataFrame(Ttab_val_f, columns=list(eigen_obj_list))
    data_table.to_csv(uq_path / f'table_{proc_num}.csv', index=False, sep='\t', float_format='%.32f')
