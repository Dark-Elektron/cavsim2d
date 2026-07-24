"""Clean variance-based (Sobol') sensitivity analysis for cavsim2d UQ samples.

Fits a low-order polynomial surrogate (a polynomial-chaos-style expansion) to the
uncertainty-quantification samples — ``uq/nodes.csv`` inputs -> ``uq/table.csv``
figure-of-merit outputs — then evaluates Sobol' **main** (S1) and **total** (ST)
indices on a Saltelli quasi-Monte-Carlo design of the surrogate. This is the
method of Corno et al., *Nucl. Instrum. Methods A* **971** (2020) 164135 and the
user's JACoW WEPB015: a second-order polynomial adequately approximates the
figures of merit, so the expensive eigenvalue solves are replaced by the cheap
surrogate for the many-sample Sobol' estimate.

Built on SALib (sampling + analysis) and scikit-learn (the surrogate).
"""
from math import comb

import numpy as np
import pandas as pd
from SALib.sample import sobol as _sobol_sampler
from SALib.analyze import sobol as _sobol
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score


def independent_inputs(X, atol=1e-9):
    """Names of the columns of ``X`` that actually vary, with exact/near-duplicate
    columns removed. After welding (continuity) the two half-cells of a shared
    iris/equator hold identical values, so their node columns are duplicates; the
    Saltelli design needs each input to be independent, so keep one per group."""
    keep = []
    for c in X.columns:
        col = X[c].to_numpy(float)
        if np.ptp(col) < 1e-12:
            continue  # constant — not a random input
        if any(np.allclose(col, X[k].to_numpy(float), rtol=0, atol=atol)
               for k in keep):
            continue  # duplicate of an already-kept (welded) column
        keep.append(c)
    return keep


def analyse(X, Y, objectives=None, N=1024, degree=2, cv=5, seed=12347):
    """Fit the surrogate once, then return **both** the Sobol' indices and the
    surrogate's goodness-of-fit for every figure of merit.

    Parameters
    ----------
    X : DataFrame
        UQ input samples (perturbed node values), one row per geometry.
    Y : DataFrame
        The matching figure-of-merit outputs, one row per geometry.
    objectives : list of str, optional
        Which ``Y`` columns to analyse (default: every numeric column).
    N : int
        Base size of the Saltelli design (a power of two is recommended); the
        surrogate is evaluated at ``N * (num_vars + 2)`` points.
    degree : int
        Polynomial-surrogate degree (2 reproduces the paper).
    cv : int
        Folds for the cross-validated goodness-of-fit (clamped to the sample count).

    Returns
    -------
    (sobol, surrogate) : tuple of dict
        ``sobol``     — ``{objective: {input_name: {'S1', 'ST'}}}``.
        ``surrogate`` — ``{objective: {'r2', 'cv_r2', 'rmse', 'max_abs_err',
        'n_samples', 'n_terms', 'actual': [...], 'predicted': [...]}}`` where the
        cross-validated ``predicted`` vs ``actual`` are the honest, out-of-sample
        fit used for the parity plot; ``rmse``/``max_abs_err`` are in the FM's own
        units (cf. the paper's per-FM cross-validation errors).
    """
    names = independent_inputs(X)
    if not names:
        raise ValueError("No varying input columns in the UQ nodes — a "
                         "sensitivity analysis needs perturbed inputs.")
    Xi = X[names].to_numpy(float)
    # Sobol' indices are defined w.r.t. the INPUT distribution. The UQ perturbations
    # are Gaussian (uq_config perturbation_mode 'add', sigma), so sample the Saltelli
    # design from a normal per input (mean, std estimated from the samples) via
    # SALib's 'norm' dist — NOT a uniform over [min, max], which would compute the
    # indices for the wrong distribution and distort the totals/interactions.
    # (Welded/averaged inputs are still Gaussian with a narrower std; the per-column
    # std captures that.) A degenerate zero-std input can't happen — independent_inputs
    # already dropped constant columns.
    problem = {
        'num_vars': len(names),
        'names': list(names),
        'bounds': [[float(Xi[:, j].mean()), float(Xi[:, j].std(ddof=1))]
                   for j in range(len(names))],
        'dists': ['norm'] * len(names),
    }
    # Saltelli/Sobol' cross-sampling; seed numpy for reproducibility
    np.random.seed(seed)
    design = _sobol_sampler.sample(problem, N, calc_second_order=False)

    if objectives is None:
        objectives = [c for c in Y.columns if pd.api.types.is_numeric_dtype(Y[c])]

    # A degree-`degree` polynomial over `p` inputs has C(p+degree, degree) terms;
    # the least-squares fit is only well-posed with at least that many samples.
    n_terms_full = comb(len(names) + degree, degree)
    n_rows = len(Y)
    if n_rows < n_terms_full:
        raise ValueError(
            f"Sensitivity analysis: {n_rows} UQ samples is too few for a "
            f"degree-{degree} surrogate over {len(names)} independent inputs "
            f"({', '.join(names[:6])}{', ...' if len(names) > 6 else ''}), which has "
            f"{n_terms_full} terms. Increase the sample count "
            f"(uq_config['method']=['normal', N] with N >= {n_terms_full}; the "
            f"WEPB015 paper used ~900), reduce the perturbed variables, or lower "
            f"`degree` (degree=1 needs {len(names) + 1} samples).")

    sobol, surrogate = {}, {}
    for obj in objectives:
        y = pd.to_numeric(Y[obj], errors='coerce').to_numpy(float)
        mask = np.isfinite(y)
        n_ok = int(mask.sum())
        if n_ok < n_terms_full:
            continue  # too few finite samples to fit this objective's surrogate
        Xm, ym = Xi[mask], y[mask]
        # StandardScaler first: the raw inputs (Ri~35, Req~103 mm) raised to degree
        # 2 span ~10^4, so the unscaled normal equations are badly conditioned;
        # centring/scaling makes the least-squares fit numerically robust without
        # changing the model. (The Saltelli design is transformed by the same fitted
        # scaler inside the pipeline, so predict() stays consistent.)
        model = make_pipeline(StandardScaler(), PolynomialFeatures(degree),
                              LinearRegression())
        model.fit(Xm, ym)

        # goodness-of-fit: in-sample R^2 + honest k-fold cross-validated prediction
        k = int(min(cv, n_ok))
        if k >= 2:
            y_cv = cross_val_predict(clone(model), Xm, ym, cv=k)
            cv_r2 = float(r2_score(ym, y_cv))
        else:
            y_cv, cv_r2 = model.predict(Xm), float('nan')
        resid = y_cv - ym
        # Read the fitted attribute — do NOT re-fit/transform any pipeline step
        # here: calling fit_transform on model's own StandardScaler would refit it
        # (on one row) and corrupt the fitted model before model.score() below.
        n_terms = int(model.named_steps['polynomialfeatures'].n_output_features_)
        # A figure of merit with (near-)zero spread carries no signal for the
        # surrogate to learn — e.g. the frequency after per-sample TUNING is driven
        # to the target, so its variance is tuning-residual + numerical noise, not a
        # function of the geometry. R^2 is then meaningless (often negative on CV).
        # Flag it so the indices/parity plot aren't read as a poor fit.
        y_mean, y_std = float(np.mean(ym)), float(np.std(ym))
        rel_std = y_std / abs(y_mean) if y_mean else y_std
        near_constant = bool(rel_std < 1e-5)
        surrogate[obj] = {
            'r2': float(model.score(Xm, ym)),
            'cv_r2': cv_r2,
            'rmse': float(np.sqrt(np.mean(resid ** 2))),
            'max_abs_err': float(np.max(np.abs(resid))),
            'n_samples': n_ok,
            'n_terms': n_terms,
            'y_mean': y_mean,
            'y_std': y_std,
            'near_constant': near_constant,
            'actual': ym.tolist(),
            'predicted': [float(v) for v in y_cv],
        }

        Si = _sobol.analyze(problem, model.predict(design),
                            calc_second_order=False, print_to_console=False)
        sobol[obj] = {name: {'S1': float(Si['S1'][j]), 'ST': float(Si['ST'][j])}
                      for j, name in enumerate(names)}
    return sobol, surrogate


def sobol_indices(X, Y, **kwargs):
    """Sobol' main/total indices per FM — ``analyse(...)[0]``. See :func:`analyse`."""
    return analyse(X, Y, **kwargs)[0]
