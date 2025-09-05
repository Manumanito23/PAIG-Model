#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIG model fitter for ANY program CSV (UNWEIGHTED)
--------------------------------------------------
This script reads a program dataset (CSV), fits the PAIG model parameters by
pure (unweighted) nonlinear least squares, checks stability, and plots data vs model.
It also saves a one-line parameter summary per file.

USAGE (examples):
  python paig_fit_any_csv.py --in "BSc_Computing and Information Systems.csv"
  python paig_fit_any_csv.py --in BSc_Applied\ Math.csv BSc_Biological\ Sciences.csv
  python paig_fit_any_csv.py --in *.csv --outdir results --save-plots
  python paig_fit_any_csv.py --in BSc_General.csv --ratio-max 0.99  --save-plots

The --ratio-max option enforces alpha = r * delta with 0 < r <= ratio_max.
This is useful if you want to enforce alpha/delta < 1, as in some of your slides.

The script auto-detects semicolon- vs comma-separated CSVs and tries to find
columns by name (case-insensitive, using substring matches). It expects:
  - one "Year" column
  - Passive stock (P):    contains "passive"
  - Active stock (A):     contains "active"
  - Inactive cumulative (I): contains "i cumulative" or "inactive cumulative"
  - Graduated cumulative (G): contains "g cumulative" or "graduate cumulative"
Optional:
  - "C in Year" (enrolled_in_year) to seed an initial guess for rho

ODE system (PAIG):
  dP/dt = - (alpha + nu) P + delta A + rho
  dA/dt = alpha P - (delta + gamma) A
  dI/dt = nu P
  dG/dt = gamma A

Author: Manuel Guillén and chatGPT :p  (unweighted version)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
from typing import Dict, Tuple, Optional, List

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


# ------------------------------
# Utility: robust CSV loading
# ------------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    """
    Try to read a CSV with automatic delimiter detection. Falls back to semicolon, then comma.
    Returns a pandas DataFrame.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")  # sniff separators
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def find_col(df: pd.DataFrame, substrings: List[str]) -> Optional[str]:
    """
    Find the first column whose lowercase name contains any of the substrings.
    Returns the column name or None.
    """
    cols = [c.strip() for c in df.columns]
    lower = [c.lower() for c in cols]
    for s in substrings:
        for i, lc in enumerate(lower):
            if s in lc:
                return cols[i]
    return None


def load_program_csv(path: Path) -> pd.DataFrame:
    """
    Load a program CSV and standardize columns to:
      ['year','P','A','I','G','enrolled_in_year','I_in_year','G_in_year'] (some optional)
    The function tries to be permissive with column names.
    """
    df = read_csv_auto(path)

    # Keep a copy for helpful error messages
    original_cols = df.columns.tolist()

    # Normalize names for searching
    cols_norm = [c.strip() for c in df.columns]
    df.columns = cols_norm

    # --- Detect required columns ---
    year_col = find_col(df, ["year"])
    if year_col is None:
        raise ValueError(f"Could not find a 'year' column in {path}. Found: {original_cols}")

    P_col = find_col(df, ["passive"])
    A_col = find_col(df, ["active"])

    # Inactive cumulative & Graduated cumulative (prefer cumulative)
    Icum_col = find_col(df, ["i cumulative", "inactive cumulative", "cum inactive"])
    Gcum_col = find_col(df, ["g cumulative", "grad cumulative", "graduated cumulative", "cumulative grad"])

    if any(x is None for x in [P_col, A_col, Icum_col, Gcum_col]):
        raise ValueError(
            f"Could not infer P/A/I/G columns in {path}.\n"
            f"Found columns: {original_cols}\n"
            f"Need something like: 'C and Passive', 'C and Active', 'I Cumulative', 'G Cumulative'."
        )

    # Optional helpers
    enrolled_col  = find_col(df, ["c in year", "enrolled in year", "cohort in year"])
    I_in_year_col = find_col(df, ["i in year", "inactive in year"])
    G_in_year_col = find_col(df, ["g in year", "graduates in year"])

    # Build a unified frame
    out = pd.DataFrame({
        "year": df[year_col].astype(float),
        "P":    df[P_col].astype(float),
        "A":    df[A_col].astype(float),
        "I":    df[Icum_col].astype(float),  # cumulative
        "G":    df[Gcum_col].astype(float),  # cumulative
    })

    # Attach optional columns if present
    if enrolled_col is not None:
        out["enrolled_in_year"] = df[enrolled_col].astype(float)
    if I_in_year_col is not None:
        out["I_in_year"] = df[I_in_year_col].astype(float)
    if G_in_year_col is not None:
        out["G_in_year"] = df[G_in_year_col].astype(float)

    # Sort by year and drop rows with NaNs
    out = out.sort_values("year").dropna().reset_index(drop=True)
    return out


# ------------------------------
# The PAIG model ODEs
# ------------------------------
def paig_rhs(t, y, rho, alpha, delta, nu, gamma):
    """
    Right-hand side of the PAIG system.
    y = [P, A, I, G] (stocks P,A and cumulative I,G).
    """
    P, A, I, G = y
    dP = - (alpha + nu) * P + delta * A + rho
    dA = alpha * P - (delta + gamma) * A
    dI = nu * P
    dG = gamma * A
    return [dP, dA, dI, dG]


def integrate_model(t_eval: np.ndarray, y0: np.ndarray, pars: np.ndarray):
    """
    Numerically integrate the PAIG system over the requested time grid.

    Parameters
    ----------
    t_eval : array of shape (T,)
        Times (e.g., calendar years shifted so t[0]=0) at which we want y(t).
        Must be sorted; we integrate from t_eval[0] to t_eval[-1].
    y0 : array of shape (4,)
        Initial state [P0, A0, I0, G0] from the first data row.
    pars : array of shape (5,)
        Parameters [rho, alpha, delta, nu, gamma].

    Returns
    -------
    sol : OdeSolution (scipy.integrate.solve_ivp result)
        Contains sol.t == t_eval and sol.y of shape (4, T).
    """
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
# Residuals for least squares (UNWEIGHTED)
# ------------------------------
def residuals_unconstrained(pars, t_eval, y0, data_mat):
    """
    Residual vector for pure (unweighted) least-squares, direct parameterization:
      pars = [rho, alpha, delta, nu, gamma].

    Returns a 1-D array of length 4*T stacked as:
      [ (P_model - P_data)_t0, ..., (P_model - P_data)_tT-1,
        (A_model - A_data)_t0, ...,                         ,
        (I_model - I_data)_t0, ...,                         ,
        (G_model - G_data)_t0, ... ]
    """
    # Keep parameters positive (tiny floor avoids exact zeros causing numerical issues).
    pars = np.clip(pars, 1e-12, None)

    sol = integrate_model(t_eval, y0, pars)
    if not sol.success:
        # Penalize failed integrations so optimizer moves away from this region
        return np.ones(data_mat.size, dtype=float) * 1e6

    pred = sol.y                 # (4, T)
    res  = (pred - data_mat)     # (4, T) unweighted residuals
    return res.ravel()           # flattened vector for least_squares


def residuals_ratio(pars, t_eval, y0, data_mat, ratio_max=0.99):
    """
    Residual vector for pure (unweighted) least-squares with re-param:
      alpha = r * delta, enforcing 0 < r <= ratio_max.
    pars = [rho, delta, nu, gamma, r]
    """
    rho, delta, nu, gamma, r = pars
    # enforce positivity and the r upper bound internally as well
    rho   = max(rho,   1e-12)
    delta = max(delta, 1e-12)
    nu    = max(nu,    1e-12)
    gamma = max(gamma, 1e-12)
    r     = min(max(r, 1e-12), ratio_max)

    alpha = r * delta

    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma], dtype=float))
    if not sol.success:
        return np.ones(data_mat.size, dtype=float) * 1e6

    pred = sol.y
    res  = (pred - data_mat)   # unweighted residuals
    return res.ravel()


# ------------------------------
# Derived quantities & checks
# ------------------------------
def eigenvalues(alpha, delta, nu, gamma):
    """
    Eigenvalues of the (P,A) subsystem's Jacobian matrix:
      M = [[-(alpha+nu), delta],
           [alpha, -(delta+gamma)]]
    """
    a = (alpha + nu) + (delta + gamma)
    b = (alpha + nu) * (delta + gamma) - alpha * delta
    disc = a**2 - 4*b
    sqrt_disc = np.sqrt(disc) if disc >= 0 else 1j * np.sqrt(-disc)
    lam1 = (-a + sqrt_disc) / 2
    lam2 = (-a - sqrt_disc) / 2
    return lam1, lam2


def steady_state(rho, alpha, delta, nu, gamma):
    """
    Closed-form steady state for (P*,A*).
    """
    denom = (alpha + nu) * (delta + gamma) - alpha * delta
    P_star = rho * (delta + gamma) / denom
    A_star = rho * alpha / denom
    return P_star, A_star


def series_metrics(y_true, y_pred):
    """
    Per-series RMSE and R^2 (UNWEIGHTED).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid  = y_true - y_pred
    rmse   = np.sqrt(np.mean(resid**2))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2


def global_gof_metrics(df: pd.DataFrame, sol, p: int = 5):
    """
    Overall goodness-of-fit metrics across all 4 series together (UNWEIGHTED):
      - Global R^2          : 1 - SSE / TSS  (using all series stacked)
      - Reduced MSE (SSE/DOF)
      - Global RMSE per point: sqrt(SSE / N)

    Parameters
    ----------
    df : DataFrame with columns ['P','A','I','G'] (already sorted)
    sol : OdeSolution from integrate_model (sol.y shape = (4, T))
    p : number of fitted parameters (5 here: rho, alpha, delta, nu, gamma) or (rho, delta, nu, gamma, r)

    Returns
    -------
    R2_global : float
    MSE_reduced : float
    RMSE_global : float
    """
    # Stack data and predictions in the order [P, A, I, G]
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)  # (4, T)
    pred_mat = sol.y.astype(float)                               # (4, T)

    # Unweighted residuals across all series/timepoints
    res = pred_mat - data_mat                # (4, T)
    SSE = float(np.sum(res**2))              # Sum of Squared Errors (all series, all times)

    # Unweighted TSS: center each series by its mean before squaring
    mean_by_series = np.mean(data_mat, axis=1, keepdims=True)
    TSS = float(np.sum((data_mat - mean_by_series)**2))

    R2_global = 1.0 - SSE / TSS if TSS > 0 else np.nan

    # Degrees of freedom and reduced MSE (SSE per DOF)
    N   = data_mat.size                  # total number of points = 4*T
    dof = max(N - p, 1)
    MSE_reduced = SSE / dof

    # RMSE per point (all series/timepoints)
    RMSE_global = np.sqrt(SSE / N)

    return R2_global, MSE_reduced, RMSE_global

def _adj_r2_from_r2(R2: float, n: int, p: int) -> float:
    """
    Adjusted R^2 = 1 - (1 - R^2) * ((n - 1) / (n - p - 1)).
    Guard against invalid denominators or undefined R^2.
    """
    if np.isnan(R2):
        return np.nan
    denom = (n - p - 1)
    if denom <= 0:
        return np.nan
    return 1.0 - (1.0 - R2) * ((n - 1.0) / denom)


def series_metrics_full(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """
    Standard, unweighted metrics for a single series:
      - MAE
      - RMSE
      - R^2
      - Adjusted R^2 (with p = number of fitted parameters in the model)
      - Chi-squared (SSE)
      - Reduced Chi-squared = SSE / (n - p)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    resid = y_pred - y_true
    n = len(y_true)

    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))

    ss_res = float(np.sum(resid**2))                        # SSE
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))   # TSS
    R2     = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    adjR2  = _adj_r2_from_r2(R2, n=n, p=p)

    dof = max(n - p, 1)            # degrees of freedom (guard)
    chi2 = ss_res                   # with unit variance assumption
    chi2_red = ss_res / dof

    return dict(
        MAE=mae,
        RMSE=rmse,
        R2=R2,
        Adj_R2=adjR2,
        Chi2=chi2,
        Chi2_reduced=chi2_red
    )


def global_metrics_4d(df: pd.DataFrame, sol, p: int) -> dict:
    """
    GLOBAL (overall) metrics using 4D Euclidean residuals per year.

    Definitions:
      e_t = || y_model(:,t) - y_data(:,t) ||_2  (Euclidean distance in R^4)
      SSE = sum_t e_t^2 = sum_{k,t} (resid_{k,t})^2
      TSS = sum_{k,t} (y_{k,t} - mean_k)^2

      MAE_global  = mean_t e_t
      RMSE_global = sqrt( mean_t e_t^2 ) = sqrt(SSE / (4T))
      R2_global   = 1 - SSE/TSS
      Adj_R2_global with n = 4T observations, p parameters
      Chi2_global = SSE (unit variance), Chi2_reduced = SSE / (n - p)
    """
    # Data and prediction matrices (4 x T)
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)
    pred_mat = sol.y.astype(float)
    res_mat  = pred_mat - data_mat

    # Euclidean distance per time (length T)
    e = np.sqrt(np.sum(res_mat**2, axis=0))
    T = e.size
    n = 4 * T     # total scalar observations across series

    # SSE and TSS (global)
    SSE = float(np.sum(e**2))  # == np.sum(res_mat**2)
    mean_vec = np.mean(data_mat, axis=1, keepdims=True)  # per-series mean
    TSS = float(np.sum((data_mat - mean_vec)**2))

    MAE_glob  = float(np.mean(e))
    RMSE_glob = float(np.sqrt(np.mean(e**2)))
    R2_glob   = 1.0 - SSE/TSS if TSS > 0 else np.nan
    Adj_R2_glob = _adj_r2_from_r2(R2_glob, n=n, p=p)

    dof = max(n - p, 1)
    Chi2_glob = SSE
    Chi2_red_glob = SSE / dof

    return dict(
        MAE=MAE_glob,
        RMSE=RMSE_glob,
        R2=R2_glob,
        Adj_R2=Adj_R2_glob,
        Chi2=Chi2_glob,
        Chi2_reduced=Chi2_red_glob
    )


def build_metrics_table(df: pd.DataFrame, sol, p: int = 5) -> pd.DataFrame:
    """
    Create the table you showed:
    Rows: Global (overall), P, A, I, G
    Cols: MAE, RMSE, R^2, Adjusted R^2, Chi-squared, Reduced Chi-squared
    All formulas are standard, unweighted.
    """
    # Global row (4D Euclidean)
    global_row = global_metrics_4d(df, sol, p=p)

    # Per-series rows
    names = ["P", "A", "I", "G"]
    rows = []
    rows.append(("Global (overall)", global_row))

    for i, nm in enumerate(names):
        stats = series_metrics_full(df[nm].values, sol.y[i], p=p)
        rows.append((f"{nm}", stats))

    # Build DataFrame in the requested order
    columns = [
        "Mean Absolute Error (MAE)",
        "Root Mean Square Error (RMSE)",
        "Coefficient of Determination (R^2)",
        "Adjusted R^2",
        "Chi-squared",
        "Reduced Chi-squared",
    ]

    data = []
    index = []
    for label, d in rows:
        index.append(label)
        data.append([d["MAE"], d["RMSE"], d["R2"], d["Adj_R2"], d["Chi2"], d["Chi2_reduced"]])

    table = pd.DataFrame(data, index=index, columns=columns)
    return table


def save_metrics_table(outdir: Path, name: str, df: pd.DataFrame, sol, p: int = 5) -> Path:
    """
    Build the table and save it as CSV next to the plots.
    Returns the saved path.
    """
    table = build_metrics_table(df, sol, p=p)
    csv_path = outdir / f"{name}_metrics_table.csv"
    table.to_csv(csv_path, float_format="%.6g")
    return csv_path

# ------------------------------
# Fitting routine for one program
# ------------------------------
def fit_program(df: pd.DataFrame,
                program_name: str,
                ratio_max: Optional[float] = None,
                max_nfev: int = 500,
                init_guess: Optional[Dict[str, float]] = None) -> Tuple[Dict, np.ndarray, pd.DataFrame, object]:
    """
    Fit the PAIG model parameters to a single program's dataset (UNWEIGHTED).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'year', 'P', 'A', 'I', 'G'.
    program_name : str
        Label used in the returned summary row.
    ratio_max : Optional[float]
        If None, fit [rho, alpha, delta, nu, gamma] freely (unconstrained).
        If a float (e.g., 0.99), enforce alpha = r * delta with 0 < r <= ratio_max.
    max_nfev : int
        Maximum number of function evaluations for the optimizer.
    init_guess : Optional[Dict[str,float]]
        Optional override for initial guesses: keys in {"rho","alpha","delta","nu","gamma"}.

    Returns
    -------
    summary, t_years, df_sorted, sol
    """
    # 1) Chronological order and shifted time axis t with t[0] = 0.
    df = df.sort_values("year").reset_index(drop=True)
    t_years = df["year"].values
    t = t_years - t_years[0]   # shift to improve numerical integration stability

    # 2) Observed matrix (4 x T) in order [P; A; I; G].
    data_mat = df[["P", "A", "I", "G"]].T.values

    # 3) Initial condition from first observation.
    y0 = data_mat[:, 0]  # [P0, A0, I0, G0]^T

    # 4) Initial guesses (heuristics); can be overridden with init_guess.
    if "enrolled_in_year" in df.columns:
        rho0 = max(1.0, float(np.nanmedian(df["enrolled_in_year"].values)))
    else:
        PA = df["P"].values + df["A"].values
        rho0 = max(1.0, float((PA[-1] - PA[0]) / max(t[-1], 1.0))) if len(PA) >= 2 else 50.0

    alpha0 = 0.4
    delta0 = 0.6
    nu0    = 0.8
    gamma0 = 0.02

    if init_guess is not None:
        rho0   = float(init_guess.get("rho",    rho0))
        alpha0 = float(init_guess.get("alpha",  alpha0))
        delta0 = float(init_guess.get("delta",  delta0))
        nu0    = float(init_guess.get("nu",     nu0))
        gamma0 = float(init_guess.get("gamma",  gamma0))

    # 5) Parameterization and bounds.
    if ratio_max is None:
        # Unconstrained: parameters are [rho, alpha, delta, nu, gamma]
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], dtype=float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])   # positivity
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0])    # sanity caps (gamma <= 1/yr)
        fun = lambda x: residuals_unconstrained(x, t, y0, data_mat)
    else:
        # Ratio-constrained: alpha = r * delta with 0 < r <= ratio_max
        r0 = min(0.8, ratio_max)
        x0 = np.array([rho0, delta0, nu0, gamma0, r0], dtype=float)  # [rho, delta, nu, gamma, r]
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        ub = np.array([1e6,  5.0,  5.0,  1.0,  ratio_max])
        fun = lambda x: residuals_ratio(x, t, y0, data_mat, ratio_max=ratio_max)

    # 6) Nonlinear least-squares
    res = least_squares(fun, x0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev,
                        loss='soft_l1',         # <-- soft L1 loss (robust weighting)
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)

    # 7) Extract fitted parameters (map back alpha if ratio parameterization used).
    if ratio_max is None:
        rho, alpha, delta, nu, gamma = res.x
    else:
        rho, delta, nu, gamma, r = res.x
        alpha = r * delta

    pars_hat = np.array([rho, alpha, delta, nu, gamma], dtype=float)

    # 8) Derived quantities for interpretation.
    P_star, A_star = steady_state(rho, alpha, delta, nu, gamma)
    lam1, lam2     = eigenvalues(alpha, delta, nu, gamma)
    ratio          = alpha / delta

    # 9) Re-integrate with fitted parameters on the same t-grid.
    sol = integrate_model(t, y0, pars_hat)

    # 10) Per-series metrics (UNWEIGHTED)
    metrics = {}
    comps = ["P", "A", "I", "G"]
    for idx, comp in enumerate(comps):
        rmse, r2 = series_metrics(df[comp].values, sol.y[idx])
        metrics[f"RMSE_{comp}"] = rmse
        metrics[f"R2_{comp}"]   = r2

    # 10b) Global (all-series) goodness-of-fit metrics (UNWEIGHTED)
    # p = 5 free parameters in both modes (unconstrained: rho,alpha,delta,nu,gamma; ratio: rho,delta,nu,gamma,r)
    R2_global, MSE_reduced, RMSE_global = global_gof_metrics(df, sol, p=5)
    metrics["R2_global"]     = R2_global
    metrics["MSE_reduced"]   = MSE_reduced
    metrics["RMSE_global"]   = RMSE_global

    # Assemble a compact summary row
    summary = {
        "program": program_name,
        "rho": rho, "alpha": alpha, "delta": delta, "nu": nu, "gamma": gamma,
        "alpha/delta": ratio, "P*": P_star, "A*": A_star,
        "lambda1": lam1, "lambda2": lam2,
        "success": res.success,
        "cost": res.cost,
        "nfev": res.nfev
    }
    summary.update(metrics)

    return summary, t_years, df, sol


# ------------------------------
# Plotting helpers
# ------------------------------
def save_series_plots(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    """
    Save four separate plots comparing data vs model for P, A, I, G.
    """
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
    """
    Save one PNG with a 2x2 grid of subplots comparing model vs data for P, A, I, G.
    """
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
    """
    Save one PNG with two 3D subplots:
      Left:  A (x) vs P (y) vs I (z)
      Right: A (x) vs P (y) vs G (z)
    Model is a 3D line; data are 3D scatter points.
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Model trajectories
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    # Data points
    Pd = df["P"].values
    Ad = df["A"].values
    Id = df["I"].values
    Gd = df["G"].values

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # Left: A (x), P (y), I (z)
    ax1.plot(Am, Pm, Im, 'r-', label="Model")
    ax1.scatter(Ad, Pd, Id, label="Data")
    ax1.set_xlabel("Active (A)")
    ax1.set_ylabel("Passive (P)")
    ax1.set_zlabel("Inactive (I, cumulative)")
    ax1.set_title("3D phase: A vs P vs I")
    ax1.legend()
    ax1.grid(True)

    # Right: A (x), P (y), G (z)
    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, label="Data")
    ax2.set_xlabel("Active (A)")
    ax2.set_ylabel("Passive (P)")
    ax2.set_zlabel("Graduated (G, cumulative)")
    ax2.set_title("3D phase: A vs P vs G")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = outdir / f"{name}_PAIG_3D_phase.png"
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fit the PAIG model to one or more program CSV files (unweighted).")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True,
                        help="Input CSV file(s). Wildcards allowed in most shells.")
    parser.add_argument("--outdir", type=str, default="paig_results",
                        help="Output directory for summary and plots.")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save PNG plots to the output directory.")
    parser.add_argument("--ratio-max", type=float, default=None,
                        help="If set, enforces alpha = r * delta with 0 < r <= ratio_max (e.g., 0.99).")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not display plots (useful for batch runs).")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for inp in args.inputs:
        path = Path(inp)
        name = path.stem.replace(" ", "_")
        print(f"\n=== Processing {path} ===")

        # Load & standardize
        df = load_program_csv(path)
        print(f"Loaded {len(df)} rows. Years {df['year'].iloc[0]}–{df['year'].iloc[-1]}")

        # Fit (unweighted)
        summary, t_years, df_sorted, sol = fit_program(
            df, name, ratio_max=args.ratio_max
        )
        summaries.append(summary)

        # Console report
        rho = summary["rho"]; alpha = summary["alpha"]; delta = summary["delta"]
        nu = summary["nu"];   gamma = summary["gamma"]
        ratio = summary["alpha/delta"]
        Pstar = summary["P*"]; Astar = summary["A*"]
        lam1  = summary["lambda1"]; lam2 = summary["lambda2"]

        print(f"  rho={rho:.4f}, alpha={alpha:.4f}, delta={delta:.4f}, nu={nu:.4f}, gamma={gamma:.4f}")
        print(f"  alpha/delta={ratio:.4f},  P*={Pstar:.2f}, A*={Astar:.2f}")
        print(f"  eigenvalues: lambda1={lam1:.4f}, lambda2={lam2:.4f}")
        print(f"  RMSE: P={summary['RMSE_P']:.2f}, A={summary['RMSE_A']:.2f}, I={summary['RMSE_I']:.2f}, G={summary['RMSE_G']:.2f}")
        print(f"  R2:   P={summary['R2_P']:.3f}, A={summary['R2_A']:.3f}, I={summary['R2_I']:.3f}, G={summary['R2_G']:.3f}")
        print(f"  Global: R2={summary['R2_global']:.3f}, RMSE={summary['RMSE_global']:.3f}, MSE_red={summary['MSE_reduced']:.3f}")

        # Plots
        if args.save_plots:
            save_series_plots(outdir, name, t_years, df_sorted, sol)
            save_series_3d_phase_plots(outdir, name, df_sorted, sol)
        metrics_csv = save_metrics_table(outdir, name, df_sorted, sol, p=5)
        print(f"  Saved metrics table: {metrics_csv}")
        if not args.no_show and not args.save_plots:
            # Show interactive (one figure per series)
            comps  = ["P", "A", "I", "G"]
            titles = ["Passive students (P)", "Active students (A)",
                      "Inactive students (I, cumulative)", "Graduated students (G, cumulative)"]
            for i, comp in enumerate(comps):
                plt.figure()
                plt.plot(t_years, sol.y[i], label="Model")
                plt.scatter(df_sorted["year"].values, df_sorted[comp].values, label="Data")
                plt.title(f"{name} — {titles[i]}")
                plt.xlabel("Year")
                plt.ylabel("Students")
                plt.legend()
                plt.grid(True)
            plt.show()

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)

    # Ensure numeric columns are numeric (for neat CSV)
    for col in ["rho","alpha","delta","nu","gamma","alpha/delta","P*","A*",
                "RMSE_P","RMSE_A","RMSE_I","RMSE_G","R2_P","R2_A","R2_I","R2_G",
                "R2_global","MSE_reduced","RMSE_global"]:
        if col in summary_df.columns:
            summary_df[col] = pd.to_numeric(summary_df[col], errors="coerce")

    summary_csv = outdir / "paig_fitted_parameters_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
