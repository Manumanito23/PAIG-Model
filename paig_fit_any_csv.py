#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIG model fitter for ANY program CSV
-------------------------------------
This script reads a program dataset (CSV), fits the PAIG model parameters by
nonlinear least squares, checks stability, and plots data vs model. It also
saves a one-line parameter summary per file.

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

The ODE system (PAIG):
  dP/dt = - (alpha + nu) P + delta A + rho
  dA/dt = alpha P - (delta + gamma) A
  dI/dt = nu P
  dG/dt = gamma A

Author: Manuel Guillén
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
        # Python engine can sniff separators when sep=None
        return pd.read_csv(path, sep=None, engine="python")
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

    # Keep a copy of original names (for better error messages)
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

    # Inactive cumulative & Graduated cumulative (prefer cumulative columns)
    Icum_col = find_col(df, ["i cumulative", "inactive cumulative", "cum inactive"])
    Gcum_col = find_col(df, ["g cumulative", "grad cumulative", "graduated cumulative", "cumulative grad"])

    if any(x is None for x in [P_col, A_col, Icum_col, Gcum_col]):
        raise ValueError(
            f"Could not infer P/A/I/G columns in {path}.\n"
            f"Found columns: {original_cols}\n"
            f"Need something like: 'C and Passive', 'C and Active', 'I Cumulative', 'G Cumulative'."
        )

    # Optional helpers
    enrolled_col = find_col(df, ["c in year", "enrolled in year", "cohort in year"])
    I_in_year_col = find_col(df, ["i in year", "inactive in year"])
    G_in_year_col = find_col(df, ["g in year", "graduates in year"])

    # Build a unified frame
    out = pd.DataFrame({
        "year": df[year_col].astype(float),
        "P": df[P_col].astype(float),
        "A": df[A_col].astype(float),
        "I": df[Icum_col].astype(float),   # cumulative
        "G": df[Gcum_col].astype(float),   # cumulative
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
        Time stamps (e.g., years) at which we want y(t). Must be sorted and
        include the start and end times; we’ll integrate from t_eval[0] to t_eval[-1].
    y0 : array of shape (4,)
        Initial state [P0, A0, I0, G0] taken from the first data row.
    pars : array of shape (5,)
        Model parameters [rho, alpha, delta, nu, gamma].

    Returns
    -------
    sol : OdeSolution (scipy.integrate.solve_ivp result)
        Contains sol.t (the same as t_eval) and sol.y (shape (4, T)),
        plus a .success flag indicating if integration succeeded.
    """
    rho, alpha, delta, nu, gamma = pars

    # Build the time-derivative function f(t, y) with parameters closed over
    # Note: solver API is fun(t, y), so we pass the parameters via closure.
    sol = solve_ivp(
        fun=lambda t, y: paig_rhs(t, y, rho, alpha, delta, nu, gamma),
        t_span=(t_eval[0], t_eval[-1]),  # integrate across the data window
        y0=y0,                           # initial condition from data
        t_eval=t_eval,                   # return solution on the data grid
        vectorized=False,                # our RHS expects 1 state at a time
        rtol=1e-7,                       # relative tolerance (accuracy control)
        atol=1e-9                        # absolute tolerance (accuracy control)
    )
    return sol


# ------------------------------
# Residuals for least squares
# ------------------------------
def residuals_unconstrained(pars, t_eval, y0, data_mat, weights=None):
    """
    Residual vector for least-squares when we fit parameters directly:
    pars = [rho, alpha, delta, nu, gamma].

    Returns a 1-D array of length 4*T:
      [ (P_model - P_data)_t0, ..., (P_model - P_data)_tT-1,
        (A_model - A_data)_t0, ...,                         ,
        (I_model - I_data)_t0, ...,                         ,
        (G_model - G_data)_t0, ... ]
    """
    # 1) Keep parameters in a physically meaningful region (positivity).
    #    Using a tiny floor also avoids exact zeros that can cause numerical issues.
    pars = np.clip(pars, 1e-12, None)

    # 2) Simulate the model at the data time points.
    sol = integrate_model(t_eval, y0, pars)

    # If ODE integration failed (stiff corner, bad params, etc.), penalize heavily
    # so the optimizer moves away from this candidate.
    if not sol.success:
        return np.ones(data_mat.size) * 1e6

    # 3) Model prediction has shape (4, T) in the same order [P, A, I, G].
    pred = sol.y

    # 4) Residuals = model - data, series-by-series.
    res = pred - data_mat

    # 5) Optional per-series weighting to balance scales (e.g., cumulative vs stocks).
    #    weights is shape (4,), so weights[:, None] broadcasts across columns (time).
    if weights is not None:
        res = res * weights[:, None]

    # 6) Flatten to a 1-D vector because least_squares expects a 1-D residual array.
    return res.ravel()



def residuals_ratio(pars, t_eval, y0, data_mat, weights=None, ratio_max=0.99):
    """
    Residuals with re-parametrization alpha = r * delta, enforcing 0 < r <= ratio_max.
    pars = [rho, delta, nu, gamma, r]
    """
    rho, delta, nu, gamma, r = pars
    # enforce bounds internally as well
    rho = max(rho, 1e-12)
    delta = max(delta, 1e-12)
    nu = max(nu, 1e-12)
    gamma = max(gamma, 1e-12)
    r = min(max(r, 1e-12), ratio_max)

    alpha = r * delta
    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma]))
    if not sol.success:
        return np.ones(data_mat.size) * 1e6

    pred = sol.y
    res = pred - data_mat
    if weights is not None:
        res = res * weights[:, None]
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
    # allow complex if rounding causes small negative
    if disc >= 0:
        sqrt_disc = np.sqrt(disc)
    else:
        sqrt_disc = 1j * np.sqrt(-disc)
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
    RMSE and R^2 for a single series.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    resid = y_true - y_pred
    rmse = np.sqrt(np.mean(resid**2))
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2


# ------------------------------
# Fitting routine for one program
# ------------------------------
def fit_program(df: pd.DataFrame,
                program_name: str,
                ratio_max: Optional[float] = None,
                max_nfev: int = 500) -> Tuple[Dict, np.ndarray, pd.DataFrame, object]:
    """
    Fit the PAIG model parameters to a single program's dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'year', 'P', 'A', 'I', 'G'.
        (Any CSV parsing/renaming is done before calling this function.)
    program_name : str
        Label used in the returned summary row.
    ratio_max : Optional[float]
        If None, fit [rho, alpha, delta, nu, gamma] freely (unconstrained).
        If a float (e.g., 0.99), enforce alpha = r * delta with 0 < r <= ratio_max.
    max_nfev : int
        Maximum number of function evaluations for the optimizer.

    Returns
    -------
    summary : Dict
        Dictionary of fitted parameters, derived quantities, per-series metrics,
        and optimizer diagnostics (success, cost, nfev).
    t_years : np.ndarray
        Original year stamps (e.g., [2013, 2014, ...]) for plotting/labels.
    df_sorted : pd.DataFrame
        The input DataFrame sorted by year and reindexed.
    sol : OdeSolution
        SciPy solve_ivp solution for the fitted parameters on the data grid.
        Contains sol.t (shifted time) and sol.y (shape 4 x T).
    """
    # 1) Ensure chronological order and build a shifted time axis t with t[0] = 0.
    df = df.sort_values("year").reset_index(drop=True)
    t_years = df["year"].values                  # e.g., [2013, 2014, ...]
    t = t_years - t_years[0]                     # numerical integration prefers small times

    # 2) Stack the observed series into a 4 x T matrix in the order [P; A; I; G].
    #    Transpose so rows are series and columns are time points.
    data_mat = df[["P", "A", "I", "G"]].T.values

    # 3) Initial condition y(0) comes from the first observation.
    #    y0 = [P0, A0, I0, G0]^T
    y0 = data_mat[:, 0]

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

    # 5) Choose parameterization and bounds.
    #    Bounds keep the search in a physically meaningful, numerically stable region.
    if ratio_max is None:
        # Unconstrained: parameters are [rho, alpha, delta, nu, gamma]
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], dtype=float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])   # positivity
        ub = np.array([1e6,  5.0,   5.0,   5.0,   1.0]) # sanity caps (gamma <= 1/yr)
        # Residual builder with per-series magnitude balancing
        fun = lambda x: residuals_unconstrained(x, t, y0, data_mat, weights=_series_weights(df))
    else:
        # Ratio-constrained: reparameterize alpha = r * delta, with 0 < r <= ratio_max
        r0 = min(0.8, ratio_max)  # a reasonable prior near the CIS example
        x0 = np.array([rho0, delta0, nu0, gamma0, r0], dtype=float)  # [rho, delta, nu, gamma, r]
        lb = np.array([1e-6, 1e-6,   1e-6, 1e-6,   1e-6])
        ub = np.array([1e6,  5.0,    5.0,  1.0,    ratio_max])
        fun = lambda x: residuals_ratio(x, t, y0, data_mat,
                                        weights=_series_weights(df),
                                        ratio_max=ratio_max)

    # 6) Nonlinear least-squares with a robust loss (down-weights outliers).
    #    Tight tolerances are used here; relax them if runtime is long.
    res = least_squares(fun, x0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev,
                        loss='soft_l1',
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)

    # 7) Extract fitted parameters (map back alpha if we used the ratio parameterization).
    if ratio_max is None:
        rho, alpha, delta, nu, gamma = res.x
    else:
        rho, delta, nu, gamma, r = res.x
        alpha = r * delta

    pars_hat = np.array([rho, alpha, delta, nu, gamma], dtype=float)

    # 8) Compute derived quantities for interpretation.
    P_star, A_star = steady_state(rho, alpha, delta, nu, gamma)
    lam1, lam2     = eigenvalues(alpha, delta, nu, gamma)  # eigenvalues of the (P,A) block
    ratio          = alpha / delta                         # engagement vs disengagement index

    # 9) Re-integrate with fitted parameters on the same t-grid to get clean predictions.
    sol = integrate_model(t, y0, pars_hat)  # sol.y has shape (4, T) in order [P, A, I, G]

    # 10) Compute per-series metrics (RMSE, R^2) for diagnostics.
    metrics = {}
    comps = ["P", "A", "I", "G"]
    for idx, comp in enumerate(comps):
        rmse, r2 = series_metrics(df[comp].values, sol.y[idx])
        metrics[f"RMSE_{comp}"] = rmse
        metrics[f"R2_{comp}"]   = r2

    # Assemble a compact summary row for aggregation across programs.
    summary = {
        "program": program_name,
        "rho": rho, "alpha": alpha, "delta": delta, "nu": nu, "gamma": gamma,
        "alpha/delta": ratio, "P*": P_star, "A*": A_star,
        "lambda1": lam1, "lambda2": lam2,
        "success": res.success,          # True if optimizer converged
        "cost": res.cost,                # final scalar objective (robust) value
        "nfev": res.nfev                 # number of function evaluations
    }
    summary.update(metrics)

    return summary, t_years, df, sol


def _series_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Compute per-series weights to balance magnitudes across P, A, I, G.

    Rationale
    ---------
    I and G are cumulative and can dwarf P and A. If we fit raw residuals,
    the optimizer will prioritize I/G and neglect P/A. To balance this, we scale
    each series by the inverse of its max observed value so that all four series
    contribute at roughly similar scales.

    Returns
    -------
    weights : np.ndarray of shape (4,)
        weights[i] = 1 / max(df[series_i]), for series_i in [P, A, I, G].
        If a series is all zeros, its scale is set to 1.0 to avoid division by zero.
    """
    # Max per series (vector of length 4 in order P, A, I, G)
    scales = df[["P", "A", "I", "G"]].max().values.astype(float)

    # Guard against zeros/non-positive max (e.g., empty series) to avoid 1/0.
    scales[scales <= 0] = 1.0

    # Weights are the inverse scales; broadcasting will apply these row-wise to residuals.
    return 1.0 / scales


# ------------------------------
# Plotting helpers
# ------------------------------
def save_series_plots(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    """
    Save four separate plots (no specific colors set) comparing data vs model for P, A, I, G.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    comps = [("P", 0, "Passive students (P)"),
             ("A", 1, "Active students (A)"),
             ("I", 2, "Inactive students (I, cumulative)"),
             ("G", 3, "Graduated students (G, cumulative)")]

    for comp, idx, title in comps:
        plt.figure()
        plt.plot(t_years, sol.y[idx], label="Model")
        plt.scatter(df["year"].values, df[comp].values, label="Data")
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

    Parameters
    ----------
    outdir : Path
        Directory where the combined PNG will be saved. Created if missing.
    name : str
        Program name or label, used in the suptitle and file name.
    t_years : np.ndarray (shape: T,)
        Calendar years to use on the x-axis for the model curve.
        (Matches the number of columns in sol.y.)
    df : pd.DataFrame
        Must contain columns: 'year', 'P', 'A', 'I', 'G'. Used for scatter data.
    sol : OdeSolution (from scipy.integrate.solve_ivp)
        Fitted model solution. sol.y has shape (4, T) in order [P, A, I, G].

    Returns
    -------
    None (writes a single PNG, e.g., "<outdir>/<name>_PAIG_grid.png")
    """
    # Ensure the output directory exists.
    outdir.mkdir(parents=True, exist_ok=True)

    # Define the components to plot: (column name in df, row index in sol.y, title text)
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)")
    ]

    # Create a single figure with a 2x2 grid of subplots.
    # sharex=True gives all subplots the same Year axis (nice for quick reading).
    # We do NOT share y because scales differ a lot (stocks vs cumulative).
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Iterate over components and corresponding subplot axes.
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]

        # Model: continuous line (no explicit color -> matplotlib default)
        ax.plot(t_years, sol.y[idx], 'r-', label="Model")

        # Data: discrete yearly points; using scatter emphasizes discrete observations
        ax.scatter(df["year"].values, df[comp].values, label="Data")

        # Titles and axis labels
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Students")

        # Readability
        ax.grid(True)
        ax.legend()

    # Overall title for the full grid
    fig.suptitle(f"{name} — PAIG: Model vs Data", fontsize=16)

    # Tidy layout and leave room for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save one combined PNG
    png_path = outdir / f"{name}_PAIG_grid.png"
    fig.savefig(png_path, dpi=300)

    # Free memory
    plt.close(fig)


def save_series_3d_phase_plots(outdir: Path, name: str, df: pd.DataFrame, sol) -> None:
    """
    Save one PNG with two 3D subplots:
      Left:  A (x) vs P (y) vs I (z)
      Right: A (x) vs P (y) vs G (z)
    Model is plotted as a continuous line; data as scatter points.

    Parameters
    ----------
    outdir : Path
        Output directory (created if needed).
    name : str
        Program label used in title and filename.
    df : pd.DataFrame
        Requires columns 'P','A','I','G' for the data scatter.
    sol : OdeSolution
        solve_ivp solution; sol.y is shape (4, T) in order [P, A, I, G].
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # Extract model trajectories
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    # Extract data points
    Pd = df["P"].values
    Ad = df["A"].values
    Id = df["I"].values
    Gd = df["G"].values

    # Create figure with 1x2 3D subplots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # ----- Left subplot: A (x), P (y), I (z)
    ax1.plot(Am, Pm, Im, 'r-', label="Model")            # 3D line
    ax1.scatter(Ad, Pd, Id, label="Data")          # 3D scatter
    ax1.set_xlabel("Active (A)")
    ax1.set_ylabel("Passive (P)")
    ax1.set_zlabel("Inactive (I, cumulative)")
    ax1.set_title("3D phase: A vs P vs I")
    ax1.legend()
    ax1.grid(True)

    # ----- Right subplot: A (x), P (y), G (z)
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
    parser = argparse.ArgumentParser(description="Fit the PAIG model to one or more program CSV files.")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input CSV file(s). Wildcards allowed in most shells.")
    parser.add_argument("--outdir", type=str, default="paig_results", help="Output directory for summary and plots.")
    parser.add_argument("--save-plots", action="store_true", help="Save per-series PNG plots to the output directory.")
    parser.add_argument("--ratio-max", type=float, default=None,
                        help="If set, enforces alpha = r * delta with 0 < r <= ratio_max (e.g., 0.99).")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (useful for batch runs).")
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

        # Fit
        summary, t_years, df_sorted, sol = fit_program(df, name, ratio_max=args.ratio_max)
        summaries.append(summary)

        # Print a small report to console
        rho = summary["rho"]; alpha = summary["alpha"]; delta = summary["delta"]
        nu = summary["nu"]; gamma = summary["gamma"]
        ratio = summary["alpha/delta"]
        Pstar = summary["P*"]; Astar = summary["A*"]
        lam1 = summary["lambda1"]; lam2 = summary["lambda2"]

        print(f"  rho={rho:.4f}, alpha={alpha:.4f}, delta={delta:.4f}, nu={nu:.4f}, gamma={gamma:.4f}")
        print(f"  alpha/delta={ratio:.4f},  P*={Pstar:.2f}, A*={Astar:.2f}")
        print(f"  eigenvalues: lambda1={lam1:.4f}, lambda2={lam2:.4f}")
        print(f"  RMSE: P={summary['RMSE_P']:.2f}, A={summary['RMSE_A']:.2f}, I={summary['RMSE_I']:.2f}, G={summary['RMSE_G']:.2f}")
        print(f"  R2:   P={summary['R2_P']:.3f}, A={summary['R2_A']:.3f}, I={summary['R2_I']:.3f}, G={summary['R2_G']:.3f}")

        # Plots
        if args.save_plots:
            save_series_grid_plot(outdir, name, t_years, df_sorted, sol)
            save_series_3d_phase_plots(outdir, name, df_sorted, sol)
        if not args.no_show and not args.save_plots:
            # If not saving, show interactively (one figure per series)
            comps = ["P", "A", "I", "G"]
            titles = ["Passive students (P)",
                      "Active students (A)",
                      "Inactive students (I, cumulative)",
                      "Graduated students (G, cumulative)"]
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

    # Round some columns for a tidy CSV
    for col in ["rho","alpha","delta","nu","gamma","alpha/delta","P*","A*",
                "RMSE_P","RMSE_A","RMSE_I","RMSE_G","R2_P","R2_A","R2_I","R2_G"]:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].astype(float)

    summary_csv = outdir / "paig_fitted_parameters_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv.resolve()}")

if __name__ == "__main__":
    main()
