#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIG model fitter for ANY program CSV
--------------------------------------------------
- Loads program CSV
- Fits PAIG parameters by unweighted nonlinear least squares
- y0 modes:
    * "zeros"     -> start from [0,0,0,0] at the first selected year
    * "estimated" -> fit y0 = [P0, A0, I0, G0] as 4 extra unknowns
- Provides per-series metrics + a global metrics table
- Provides a Pearson/Poisson χ² p-value (useful hypothesis-style sanity check)

Author: Manuel Guillén and ChatGPT
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist  # for p-values


# ------------------------------
# Robust CSV loading
# ------------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")

def find_col(df: pd.DataFrame, substrings: List[str]) -> Optional[str]:
    cols = [c.strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    for s in substrings:
        for i, lc in enumerate(lower):
            if s in lc:
                return cols[i]
    return None

def load_program_csv(path: Path) -> pd.DataFrame:
    """
    Standardize to columns:
      year, P, A, I, G, (optional) enrolled_in_year, I_in_year, G_in_year
    """
    df = read_csv_auto(path)
    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    year_col = find_col(df, ["year"])
    if year_col is None:
        raise ValueError(f"No 'year' column in {path}. Found: {original_cols}")

    P_col = find_col(df, ["passive"])
    A_col = find_col(df, ["active"])
    Icum_col = find_col(df, ["i cumulative", "inactive cumulative", "cum inactive"])
    Gcum_col = find_col(df, ["g cumulative", "grad cumulative", "graduated cumulative", "cumulative grad"])

    if any(x is None for x in [P_col, A_col, Icum_col, Gcum_col]):
        raise ValueError(
            f"Could not infer P/A/I/G columns in {path}.\n"
            f"Found: {original_cols}\n"
            f"Need: '...Passive', '...Active', 'I Cumulative', 'G Cumulative'."
        )

    enrolled_col  = find_col(df, ["c in year", "enrolled in year", "cohort in year"])
    I_in_year_col = find_col(df, ["i in year", "inactive in year"])
    G_in_year_col = find_col(df, ["g in year", "graduates in year"])

    out = pd.DataFrame({
        "year": df[year_col].astype(float),
        "P": df[P_col].astype(float),
        "A": df[A_col].astype(float),
        "I": df[Icum_col].astype(float),
        "G": df[Gcum_col].astype(float),
    })

    if enrolled_col is not None:
        out["enrolled_in_year"] = df[enrolled_col].astype(float)
    if I_in_year_col is not None:
        out["I_in_year"] = df[I_in_year_col].astype(float)
    if G_in_year_col is not None:
        out["G_in_year"] = df[G_in_year_col].astype(float)

    return out.sort_values("year").dropna().reset_index(drop=True)


# ------------------------------
# PAIG ODEs and integrator
# ------------------------------
def paig_rhs(t, y, rho, alpha, delta, nu, gamma):
    P, A, I, G = y
    dP = - (alpha + nu) * P + delta * A + rho
    dA = alpha * P - (delta + gamma) * A
    dI = nu * P
    dG = gamma * A
    return [dP, dA, dI, dG]

def integrate_model(t_eval: np.ndarray, y0: np.ndarray, pars: np.ndarray):
    rho, alpha, delta, nu, gamma = pars
    sol = solve_ivp(
        fun=lambda t, y: paig_rhs(t, y, rho, alpha, delta, nu, gamma),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        vectorized=False,
        rtol=1e-7,
        atol=1e-9
    )
    return sol


# ------------------------------
# Residuals (UNWEIGHTED)
# ------------------------------
def residuals_unconstrained(pars, t_eval, y0, data_mat):
    """
    Fit only the 5 PAIG parameters; initial condition y0 is fixed (given).
    pars = [rho, alpha, delta, nu, gamma]
    """
    pars = np.clip(np.asarray(pars, float), 1e-12, None)
    sol = integrate_model(t_eval, y0, pars)
    if not sol.success:
        return np.ones(data_mat.size) * 1e6
    res = sol.y - data_mat  # (4, T)
    return res.ravel()

def residuals_unconstrained_y0free(pars_ext, t_eval, data_mat):
    """
    Fit the 5 PAIG parameters **and** the initial condition y0 (4 unknowns).
    pars_ext = [rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0]
    """
    pars_ext = np.asarray(pars_ext, float)
    rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0 = pars_ext
    pars = np.clip([rho, alpha, delta, nu, gamma], 1e-12, None)
    y0   = np.clip([yP0, yA0, yI0, yG0], 0.0, None)

    sol = integrate_model(t_eval, y0, np.array(pars, float))
    if not sol.success:
        return np.ones(data_mat.size) * 1e6

    res = sol.y - data_mat
    return res.ravel()

def residuals_zeros_tminus1(pars, t_eval_data, data_mat):
    """
    Mode 'zeros@year-1':
      Integrate from t=-1 with y0=(0,0,0,0) to t_eval_data (which starts at 0),
      and compare ONLY against data at t>=0 (discard the column at t=-1).
    """
    pars = np.clip(np.asarray(pars, float), 1e-12, None)
    y0 = np.zeros(4, dtype=float)

    # Integrate one year earlier:
    t_aug = np.r_[-1.0, t_eval_data]                 # [-1, 0, 1, 2, ...]
    sol = integrate_model(t_aug, y0, pars)
    if not sol.success:
        return np.ones(data_mat.size) * 1e6

    pred_at_data = sol.y[:, 1:]                      # discard the column at t=-1
    res = pred_at_data - data_mat                    # data_mat is (4, T)
    return res.ravel()

# ------------------------------
# Derived helpers & metrics
# ------------------------------
def eigenvalues(alpha, delta, nu, gamma):
    a = (alpha + nu) + (delta + gamma)
    b = (alpha + nu) * (delta + gamma) - alpha * delta
    disc = a**2 - 4*b
    sqrt_disc = np.sqrt(disc) if disc >= 0 else 1j * np.sqrt(-disc)
    lam1 = (-a + sqrt_disc) / 2
    lam2 = (-a - sqrt_disc) / 2
    return lam1, lam2

def steady_state(rho, alpha, delta, nu, gamma):
    denom = (alpha + nu) * (delta + gamma) - alpha * delta
    P_star = rho * (delta + gamma) / denom
    A_star = rho * alpha / denom
    return P_star, A_star

def series_metrics(y_true, y_pred):
    """
    Return MAE, RMSE, R^2 for a single series (unweighted).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid  = y_true - y_pred
    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2

def _chi2_poisson(e: np.ndarray, mu: np.ndarray, eps: float = 1e-9) -> float:
    """
    Pearson chi-square for count-like data: sum (e^2 / mu).
    Used only to compute a p-value; your fit itself remains unweighted.
    """
    mu_safe = np.maximum(np.asarray(mu, float), eps)
    return float(np.sum((np.asarray(e, float) ** 2) / mu_safe))

def metrics_table(df: pd.DataFrame, sol, p: int = 5) -> pd.DataFrame:
    """
    Build the table of statistics (UNWEIGHTED), with rows:
      - Global (overall): uses 4D Euclidean residual per year
      - P, A, I, G: standard 1D formulas

    Columns:
      'Mean Absolute Error (MAE)',
      'Root Mean Square Error (RMSE)',
      'Coefficient of Determination (R^2)',
      'Adjusted R^2',
      'Chi-squared'              (unweighted SSE)
      'Reduced Chi-squared'      (unweighted SSE / (n - p))
      'p-value (χ², Poisson)'    (Pearson/Poisson χ² tail prob)

    Notes
    -----
    * Global row: n = T (years) for adj-R² and reduced χ²; its SSE is based on
      the 4D Euclidean residual per year.
    * p-values are computed via a Pearson/Poisson χ²:
        per-series:     χ² = Σ (e_t^2 / ŷ_t), df = max(T - p, 1)
        global (all 4): χ² = Σ_ij (e_ij^2 / ŷ_ij), df = max(4T - p, 1)
      This keeps the UI fully populated without changing the *fit*, which is
      still pure unweighted NLS.
    """
    # Data / prediction (T, 4)
    Y_true = df[["P", "A", "I", "G"]].values.astype(float)   # (T, 4)
    Y_pred = sol.y.T.astype(float)                           # (T, 4)
    ERR    = Y_pred - Y_true                                 # (T, 4)
    T = Y_true.shape[0]

    # ---------- per-series stats ----------
    rows = []
    for j, label in enumerate(["P", "A", "I", "G"]):
        y_true = Y_true[:, j]
        y_pred = Y_pred[:, j]
        e      = y_pred - y_true
        n      = T

        mae, rmse, r2 = series_metrics(y_true, y_pred)

        # adjusted R² (guard small n)
        if np.isfinite(r2) and (n - p - 1) > 0:
            r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
        else:
            r2_adj = np.nan

        # "χ²" and "reduced χ²" shown in table are UNWEIGHTED SSE versions
        chi2_unw = float(np.sum(e**2))
        red_chi2 = chi2_unw / max(n - p, 1)

        # p-value computed from Pearson/Poisson χ² (keeps table populated)
        chi2_poi = _chi2_poisson(e, y_pred)
        df_dof   = max(n - p, 1)
        p_value  = float(1.0 - chi2_dist.cdf(chi2_poi, df=df_dof))

        rows.append((label, mae, rmse, r2, r2_adj, chi2_unw, red_chi2, p_value))

    # ---------- global row (4D Euclidean per year) ----------
    # Euclidean residual per year: r_t = ||ERR[t, :]||_2
    r_norm = np.linalg.norm(ERR, axis=1)            # (T,)
    n_g    = T

    mae_g  = float(np.mean(r_norm))
    rmse_g = float(np.sqrt(np.mean(r_norm**2)))
    sse_g  = float(np.sum(r_norm**2))

    # total 4D variance around the 4D mean (centroid)
    mu = np.mean(Y_true, axis=0, keepdims=True)     # (1,4)
    ss_tot_g = float(np.sum(np.sum((Y_true - mu)**2, axis=1)))
    r2_g = (1.0 - sse_g / ss_tot_g) if ss_tot_g > 0 else np.nan
    if np.isfinite(r2_g) and (n_g - p - 1) > 0:
        r2_adj_g = 1.0 - (1.0 - r2_g) * (n_g - 1) / (n_g - p - 1)
    else:
        r2_adj_g = np.nan

    red_chi2_g = sse_g / max(n_g - p, 1)

    # p-value for global row via Pearson/Poisson across ALL 4 series
    chi2_poi_global = _chi2_poisson(ERR, Y_pred)         # sum over all coords
    dof_global_poi  = max(4 * T - p, 1)                  # 4T minus params
    p_value_global  = float(1.0 - chi2_dist.cdf(chi2_poi_global, df=dof_global_poi))

    # assemble table
    table = pd.DataFrame(
        [("Global (overall)", mae_g, rmse_g, r2_g, r2_adj_g, sse_g, red_chi2_g, p_value_global)] +
        rows,
        columns=[
            "Series",
            "Mean Absolute Error (MAE)",
            "Root Mean Square Error (RMSE)",
            "Coefficient of Determination (R^2)",
            "Adjusted R^2",
            "Chi-squared",
            "Reduced Chi-squared",
            "p-value (χ², Poisson)",
        ],
    )
    return table

def global_gof_metrics(df: pd.DataFrame, sol, p: int = 5):
    """
    Overall (unweighted) metrics across the 4 series.
    - R2_global: computed with data centered per series, then summed
    - MSE_reduced: SS_res / (n - p) with n = 4*T
    - RMSE_global: sqrt(SS_res / n)
    """
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)  # (4, T)
    pred_mat = sol.y.astype(float)
    res      = pred_mat - data_mat

    SS_res = float(np.sum(res**2))
    mean_  = np.mean(data_mat, axis=1, keepdims=True)
    SS_tot = float(np.sum((data_mat - mean_)**2))
    R2_global = 1.0 - SS_res / SS_tot if SS_tot > 0 else np.nan

    n = data_mat.size                     # 4*T points if flattened
    dof = max(n - p, 1)
    MSE_reduced = SS_res / dof
    RMSE_global = np.sqrt(SS_res / n)

    return R2_global, MSE_reduced, RMSE_global


# ------------------------------
# Fitting routine for one program, with y0 mode
# ------------------------------
def fit_program(df: pd.DataFrame,
                program_name: str,
                y0_mode: str = "estimated",   # "estimated" | "zeros" | "data"
                init_guess: Optional[Dict[str, float]] = None,
                max_nfev: int = 500,
                loss: str = "soft_l1",         # "soft_l1" or "linear"
                normalize: bool = False,
                ) -> Tuple[Dict, np.ndarray, pd.DataFrame, object]:
    """
    Fit PAIG parameters to one dataset.

    y0_mode:
      - "estimated": fit y0 = [P0, A0, I0, G0] as 4 extra parameters
      - "zeros":     start at t = -1 with y0=(0,0,0,0), integrate one year,
                     then compare only from t=0 to the data (replaces old zeros mode)
      - "data":      sets y0 to the first observed point of the data table without adjustments

    loss:
      - "soft_l1": robust loss
      - "linear":  pure LS
    """
    # 1) Ensure chronological order and build a shifted time axis t with t[0] = 0.
    df = df.sort_values("year").reset_index(drop=True)
    t_years = df["year"].values                  # e.g., [2013, 2014, ...]
    t = t_years - t_years[0]                     # numerical integration prefers small times

    # 2) Stack the observed series into a 4 x T matrix in the order [P; A; I; G].
    #    Transpose so rows are series and columns are time points.
    data_raw = df[["P","A","I","G"]].T.values.astype(float)
    # Optional normalization of the data
    if normalize:
        scales = data_raw.max(axis=1).astype(float)
        scales[scales <= 0] = 1.0      # avoid /0
    else:
        scales = np.ones(4, dtype=float)

    # 3) Initial condition y(0) comes from the first observation.
    #    y0 = [P0, A0, I0, G0]^T
    data_mat = data_raw / scales[:, None]
    y0_data  = data_mat[:, 0]  # only a starting guess when y0 is free

    # 4) Initial guesses (heuristics). Better guesses -> faster/robuster convergence.
    if "enrolled_in_year" in df.columns:
        # Use the typical annual enrolment as a first guess for rho (>= 1 to avoid exact zero).
        rho0 = max(1.0, float(np.nanmedian(df["enrolled_in_year"].values)))
    else:
        # Fallback: approximate rho as the trend in (P + A) over the window.
        PA = df["P"].values + df["A"].values
        if len(PA) >= 2:
            rho0 = max(1.0, float((PA[-1] - PA[0]) / max(t[-1], 1.0)))
        else:
            rho0 = 50.0  # extremely short series: just pick a modest positive inflow
    # Reasonable per-year starting rates for slow educational dynamics:
    alpha0 = 0.4
    delta0 = 0.6
    nu0    = 0.8
    gamma0 = 0.02
    # Allow external initial guesses from UI sliders
    if init_guess is not None:
        rho0   = float(init_guess.get("rho",    rho0))
        alpha0 = float(init_guess.get("alpha",  alpha0))
        delta0 = float(init_guess.get("delta",  delta0))
        nu0    = float(init_guess.get("nu",     nu0))
        gamma0 = float(init_guess.get("gamma",  gamma0))

    # 5) build optimization problem
    if y0_mode == "zeros":
        # “previous year with zero stocks”
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0])
        fun = lambda x: residuals_zeros_tminus1(x, t, data_mat)
        p_count = 5

    elif y0_mode == "data":
        # y0 fixed: same as first row of the data
        y0_fixed = y0_data.copy()
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0])
        fun = lambda x: residuals_unconstrained(x, t, y0_fixed, data_mat)
        p_count = 5

    else:  # "estimated"  (y0 free, adjusted with the first 5 params)
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0,
                       y0_data[0], y0_data[1], y0_data[2], y0_data[3]], float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.0, 0.0, 0.0, 0.0])
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0,  1e9,  1e9,  1e9,  1e9])
        fun = lambda x: residuals_unconstrained_y0free(x, t, data_mat)
        p_count = 9

    res = least_squares(fun, x0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev,
                        loss=loss,               # linear or soft_l1
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)

    # ---- extract results & build solution aligned with data BACK TO ORIGINAL UNITS
    if y0_mode == "estimated":
        rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0 = res.x
        pars_hat = np.array([rho, alpha, delta, nu, gamma], float)
        y0_used_norm = np.array([yP0, yA0, yI0, yG0], float)  # Scale of the adjustment
        sol = integrate_model(t, y0_used_norm, pars_hat)
        sol.y = sol.y * scales[:, None]                      # back to original
        y0_used = y0_used_norm * scales

    elif y0_mode == "zeros":
        rho, alpha, delta, nu, gamma = res.x
        pars_hat = np.array([rho, alpha, delta, nu, gamma], float)
        t_aug = np.r_[-1.0, t]
        y0_z_norm = np.zeros(4, dtype=float)                 # 0 in used scale
        sol_full = integrate_model(t_aug, y0_z_norm, pars_hat)
        pred_norm = sol_full.y[:, 1:]                        # t >= 0 in normalized scale
        pred_orig = pred_norm * scales[:, None]
        class _Sol: pass
        sol = _Sol()
        sol.y = pred_orig
        sol.t = t
        sol.success = sol_full.success
        y0_used = y0_z_norm * scales                         # still 0, but in original

    else:  # "data"
        rho, alpha, delta, nu, gamma = res.x
        pars_hat = np.array([rho, alpha, delta, nu, gamma], dtype=float)
        y0_used_norm = y0_data.copy()
        sol = integrate_model(t, y0_used_norm, pars_hat)
        sol.y = sol.y * scales[:, None]                      # back to original
        y0_used = y0_used_norm * scales

    # ---- derived quantities & metrics (unchanged)
    P_star, A_star = steady_state(*pars_hat)
    lam1, lam2     = eigenvalues(pars_hat[1], pars_hat[2], pars_hat[3], pars_hat[4])
    ratio          = pars_hat[1] / pars_hat[2]

    metrics = {}
    for idx, comp in enumerate(["P", "A", "I", "G"]):
        mae, rmse, r2 = series_metrics(df[comp].values, sol.y[idx])
        metrics[f"MAE_{comp}"]  = mae
        metrics[f"RMSE_{comp}"] = rmse
        metrics[f"R2_{comp}"]   = r2

    R2_global, MSE_reduced, RMSE_global = global_gof_metrics(df, sol, p=p_count)
    metrics["R2_global"]   = R2_global
    metrics["MSE_reduced"] = MSE_reduced
    metrics["RMSE_global"] = RMSE_global

    ERR = (sol.y - df[["P","A","I","G"]].T.values).T
    CHI2_poi_global = _chi2_poisson(ERR, sol.y.T)
    dof_chi2        = max(4 * len(df) - p_count, 1)
    p_value_global  = float(1.0 - chi2_dist.cdf(CHI2_poi_global, df=dof_chi2))
    metrics["chi2_global"]   = CHI2_poi_global
    metrics["chi2_dof"]      = dof_chi2
    metrics["chi2_p_value"]  = p_value_global

    summary = {
        "program": program_name,
        "rho": pars_hat[0], "alpha": pars_hat[1], "delta": pars_hat[2],
        "nu": pars_hat[3],  "gamma": pars_hat[4],
        "alpha/delta": ratio, "P*": P_star, "A*": A_star,
        "lambda1": lam1, "lambda2": lam2,
        "y0_mode": y0_mode,
        "y0_P": y0_used[0], "y0_A": y0_used[1], "y0_I": y0_used[2], "y0_G": y0_used[3],
        "success": res.success, "cost": res.cost, "nfev": res.nfev, "loss": loss,
        "scales": scales.tolist()
    }

    summary.update(metrics)

    return summary, t_years, df, sol


# ------------------------------
# Plotting helpers
# ------------------------------
def save_series_plots(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    comps = [("P", 0, "Passive students (P)"),
             ("A", 1, "Active students (A)"),
             ("I", 2, "Inactive students (I, cumulative)"),
             ("G", 3, "Graduated students (G, cumulative)")]
    for comp, idx, title in comps:
        plt.figure()
        plt.plot(t_years, sol.y[idx], 'r-', label="Model")
        plt.scatter(df["year"].values, df[comp].values, label="Data")
        y_max = max(df[comp].max(), sol.y[idx].max())
        plt.ylim(0, y_max)
        plt.title(f"{name} — {title}")
        plt.xlabel("Year")
        plt.ylabel("Students")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        png_path = outdir / f"{name}_{comp}.png"
        plt.savefig(png_path, dpi=300)
        plt.close()


def save_series_grid_plot(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]
        ax.plot(t_years, sol.y[idx], 'r-', label="Model")
        ax.scatter(df["year"].values, df[comp].values, label="Data")
        y_max = max(df[comp].max(), sol.y[idx].max())
        ax.set_ylim(0, y_max)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Students")
        ax.grid(True)
        ax.legend()
    fig.suptitle(f"{name} — PAIG: Model vs Data", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png_path = outdir / f"{name}_PAIG_grid.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def save_series_3d_phase_plots(outdir: Path, name: str, df: pd.DataFrame, sol) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    Pd, Ad, Id, Gd = df["P"].values, df["A"].values, df["I"].values, df["G"].values

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot(Am, Pm, Im, 'r-', label="Model")
    ax1.scatter(Ad, Pd, Id, label="Data")
    ax1.set_xlabel("Active (A)"); ax1.set_ylabel("Passive (P)"); ax1.set_zlabel("Inactive (I)")
    ax1.set_title("3D phase: A vs P vs I"); ax1.legend(); ax1.grid(True)

    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, label="Data")
    ax2.set_xlabel("Active (A)"); ax2.set_ylabel("Passive (P)"); ax2.set_zlabel("Graduated (G)")
    ax2.set_title("3D phase: A vs P vs G"); ax2.legend(); ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png_path = outdir / f"{name}_PAIG_3D_phase.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fit the PAIG model (unweighted) to one or more CSV files.")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input CSV file(s)")
    parser.add_argument("--outdir", type=str, default="paig_results", help="Output directory")
    parser.add_argument("--save-plots", action="store_true", help="Save PNGs")
    parser.add_argument("--y0-mode", type=str, default="estimated", choices=["estimated", "zeros"],
                        help="Initial condition at start of fit: 'estimated' (fit y0) or 'zeros'")
    parser.add_argument("--no-show", action="store_true", help="Don't display plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for inp in args.inputs:
        path = Path(inp)
        name = path.stem.replace(" ", "_")
        print(f"\n=== Processing {path} ===")
        df = load_program_csv(path)
        print(f"Loaded {len(df)} rows. Years {int(df['year'].iloc[0])}–{int(df['year'].iloc[-1])}")

        summary, t_years, df_sorted, sol = fit_program(df, name, y0_mode=args.y0_mode)
        summaries.append(summary)

        rho = summary["rho"]; alpha = summary["alpha"]; delta = summary["delta"]
        nu = summary["nu"];   gamma = summary["gamma"]; ratio = summary["alpha/delta"]
        Pstar = summary["P*"]; Astar = summary["A*"]; lam1 = summary["lambda1"]; lam2 = summary["lambda2"]

        print(f"  y0_mode={summary['y0_mode']}, y0=({summary['y0_P']:.2f},{summary['y0_A']:.2f},{summary['y0_I']:.2f},{summary['y0_G']:.2f})")
        print(f"  rho={rho:.4f}, alpha={alpha:.4f}, delta={delta:.4f}, nu={nu:.4f}, gamma={gamma:.4f}")
        print(f"  alpha/delta={ratio:.4f},  P*={Pstar:.2f}, A*={Astar:.2f}")
        print(f"  eigenvalues: lambda1={lam1:.4f}, lambda2={lam2:.4f}")
        print(f"  RMSE: P={summary['RMSE_P']:.2f}, A={summary['RMSE_A']:.2f}, I={summary['RMSE_I']:.2f}, G={summary['RMSE_G']:.2f}")
        print(f"  R2:   P={summary['R2_P']:.3f}, A={summary['R2_A']:.3f}, I={summary['R2_I']:.3f}, G={summary['R2_G']:.3f}")
        print(f"  Global (unweighted): R2={summary['R2_global']:.3f}, RMSE={summary['RMSE_global']:.3f}, MSE_reduced={summary['MSE_reduced']:.3f}")
        print(f"  χ² test (Poisson): chi2={summary['chi2_global']:.4f}, dof={summary['chi2_dof']}, p={summary['chi2_p_value']:.4g}")

        if args.save_plots:
            save_series_grid_plot(outdir, name, t_years, df_sorted, sol)
            save_series_3d_phase_plots(outdir, name, df_sorted, sol)

    pd.DataFrame(summaries).to_csv(outdir / "paig_fitted_parameters_summary.csv", index=False)
    print(f"\nSaved summary to: {(outdir / 'paig_fitted_parameters_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
