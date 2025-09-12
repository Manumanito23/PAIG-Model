#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIG model fitter for ANY program CSV (UNWEIGHTED fitting)
----------------------------------------------------------
• Reads a program dataset (CSV), fits the PAIG parameters via *unweighted* nonlinear least squares.
• Supports two initial-condition modes:
    - y0_mode = "zeros"     -> y0 = [0,0,0,0] (five parameters fitted)
    - y0_mode = "estimated" -> y0 is fitted as 4 extra nonnegative parameters (nine parameters fitted)
• Plots (2x2 grid) and 3D phase plots.
• Writes a CSV summary of fitted parameters + diagnostic metrics.

NEW (this version):
• Adds a *χ² p-value* for **overall** fit adequacy, based on a Poisson noise model
  for yearly counts (variance ≈ mean). This does NOT affect the fit itself.
  See the function docstring of `chi2_pvalue_global()` for the exact assumptions.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.stats import chi2  # for χ² p-value


# ------------------------------
# Robust CSV loading
# ------------------------------
def read_csv_auto(path: Path) -> pd.DataFrame:
    """Try sep=None (sniff), else semicolon, else comma."""
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def find_col(df: pd.DataFrame, substrings: List[str]) -> Optional[str]:
    """Return the first column whose lowercase name contains any of the substrings."""
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
      ['year','P','A','I','G','enrolled_in_year','I_in_year','G_in_year'] (some optional).
    """
    df = read_csv_auto(path)

    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    year_col = find_col(df, ["year"])
    if year_col is None:
        raise ValueError(f"Could not find a 'year' column in {path}. Found: {original_cols}")

    P_col = find_col(df, ["passive"])
    A_col = find_col(df, ["active"])
    Icum_col = find_col(df, ["i cumulative", "inactive cumulative", "cum inactive"])
    Gcum_col = find_col(df, ["g cumulative", "grad cumulative", "graduated cumulative", "cumulative grad"])

    if any(x is None for x in [P_col, A_col, Icum_col, Gcum_col]):
        raise ValueError(
            f"Could not infer P/A/I/G columns in {path}.\n"
            f"Found columns: {original_cols}\n"
            f"Need something like: 'C and Passive', 'C and Active', 'I Cumulative', 'G Cumulative'."
        )

    enrolled_col = find_col(df, ["c in year", "enrolled in year", "cohort in year"])
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

    out = out.sort_values("year").dropna().reset_index(drop=True)
    return out


# ------------------------------
# PAIG ODEs and integrator
# ------------------------------
def paig_rhs(t, y, rho, alpha, delta, nu, gamma):
    """Right-hand side for y = [P, A, I, G]."""
    P, A, I, G = y
    dP = - (alpha + nu) * P + delta * A + rho
    dA = alpha * P - (delta + gamma) * A
    dI = nu * P
    dG = gamma * A
    return [dP, dA, dI, dG]


def integrate_model(t_eval: np.ndarray, y0: np.ndarray, pars: np.ndarray):
    """
    Integrate the PAIG system on the requested time grid (unweighted, does not alter fitting).
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
# Residuals (UNWEIGHTED)
# ------------------------------
def _residuals_with_y0fixed(pars, t_eval, y0_fixed, data_mat):
    """
    pars = [rho, alpha, delta, nu, gamma], y0 is fixed (either from zeros or any chosen value).
    Returns a 1-D residual vector of length 4*T (order: [P,A,I,G]).
    """
    pars = np.clip(np.asarray(pars, float), 1e-12, None)
    sol = integrate_model(t_eval, y0_fixed, pars)
    if not sol.success:
        return np.ones(data_mat.size, dtype=float) * 1e6
    res = sol.y - data_mat
    return res.ravel()


def _residuals_with_y0free(pars_with_y0, t_eval, data_mat):
    """
    pars_with_y0 = [rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0]
    y0 is fitted (nonnegative).
    """
    x = np.asarray(pars_with_y0, float)
    # clip parameters to positive region
    x[:5] = np.clip(x[:5], 1e-12, None)
    # y0 nonnegative
    x[5:] = np.clip(x[5:], 0.0, None)

    rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0 = x
    y0 = np.array([yP0, yA0, yI0, yG0], dtype=float)

    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma], dtype=float))
    if not sol.success:
        return np.ones(data_mat.size, dtype=float) * 1e6
    res = sol.y - data_mat
    return res.ravel()


# ------------------------------
# Derived quantities & diagnostics
# ------------------------------
def eigenvalues(alpha, delta, nu, gamma):
    """
    Eigenvalues of the (P,A) Jacobian:
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
    """Closed-form steady state (P*,A*)."""
    denom = (alpha + nu) * (delta + gamma) - alpha * delta
    P_star = rho * (delta + gamma) / denom
    A_star = rho * alpha / denom
    return P_star, A_star


def series_metrics(y_true, y_pred):
    """
    Standard per-series metrics (UNWEIGHTED):
      • RMSE
      • R²
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid**2)))
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return rmse, r2


def global_gof_metrics(df: pd.DataFrame, sol, p: int):
    """
    Overall (global) UNWEIGHTED metrics using the 4D Euclidean residual per time:
      • Global RMSE (per 4D point)
      • Global R² (built from SS_res of 4D residuals vs SS_tot of 4D data)
      • Unweighted SSE and "reduced SSE" (= SSE / dof), printed for reference

    NOTE: These are the same metrics you already had, kept unchanged.
    """
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)  # (4, T)
    pred_mat = sol.y.astype(float)
    res = pred_mat - data_mat                                   # (4, T)

    # 4D Euclidean residual per year, then sum-of-squares in that 4D space
    res4 = np.sqrt(np.sum(res**2, axis=0))                      # (T,)
    sse4 = float(np.sum(res4**2))
    # For a global R², compare to centered 4D data
    data4 = np.sqrt(np.sum(data_mat**2, axis=0))
    ss_tot4 = float(np.sum((data4 - np.mean(data4))**2))
    R2_global = 1 - sse4 / ss_tot4 if ss_tot4 > 0 else np.nan

    n = int(data_mat.size)  # 4*T scalar residuals
    dof = max(n - p, 1)
    reduced_sse = sse4 / dof

    rmse_global = float(np.sqrt(np.mean(res4**2)))
    return R2_global, reduced_sse, rmse_global, sse4, dof


# ------------- NEW: χ² p-value (Poisson model) -----------------
def chi2_pvalue_global(df: pd.DataFrame, sol, p: int,
                       variance_mode: str = "poisson_model"):
    """
    Compute a *χ² goodness-of-fit test* for the **overall** model using all 4 series together.

    WHAT WE TEST
    ------------
    H0: residuals are mean-zero independent Gaussian with variances σ²_{j,t} specified below,
        i.e., the model reproduces the data within the expected counting noise.

    HOW WE BUILD χ²
    ---------------
    We use all scalar residuals in [P, A, I, G] across time:
        χ² = Σ_{j∈{P,A,I,G}} Σ_t  ( (y_{j,t} - f_{j,t})² / σ²_{j,t} )
    dof = N - p, where N=4*T is the total number of scalar observations and p is the
    number of fitted parameters (p=5 when y0="zeros"; p=9 when y0="estimated").

    VARIANCE ASSUMPTION (variance_mode)
    -----------------------------------
    "poisson_model" (default):  σ²_{j,t} ≈ max(f_{j,t}, 1)
        • Counts per year are "approximately Poisson", variance ≈ mean.
        • We use the model mean f_{j,t} as the Poisson rate λ.
        • Guarded by min variance of 1 to avoid divide-by-zero.
    This does NOT affect the fitting (which remains unweighted) — it is only used here
    to produce a p-value that your professor requested.

    RETURNS
    -------
    chi2_val : float
    dof      : int
    p_value  : float = 1 - CDF_χ²(chi2_val; dof)
    """
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)  # (4, T)
    pred_mat = sol.y.astype(float)
    resid = data_mat - pred_mat

    if variance_mode == "poisson_model":
        sigma2 = np.maximum(pred_mat, 1.0)  # var ≈ mean (model), with a floor of 1
    else:
        # Fallback: unit variance (not recommended for p-values)
        sigma2 = np.ones_like(pred_mat)

    chi2_val = float(np.sum((resid**2) / sigma2))
    n = int(data_mat.size)               # 4*T
    dof = max(n - p, 1)
    p_value = float(1.0 - chi2.cdf(chi2_val, dof))
    return chi2_val, dof, p_value


# ------------------------------
# Fitting routine for one program
# ------------------------------
def fit_program(df: pd.DataFrame,
                program_name: str,
                y0_mode: str = "estimated",
                max_nfev: int = 500,
                init_guess: Optional[Dict[str, float]] = None) -> Tuple[Dict, np.ndarray, pd.DataFrame, object]:
    """
    Fit the PAIG model parameters to a single program's dataset (UNWEIGHTED NLS).

    Parameters
    ----------
    df : pd.DataFrame  with columns ['year','P','A','I','G']
    program_name : str  used in plots/summary
    y0_mode : {"estimated","zeros"}
        - "estimated": fit y0 as 4 extra nonnegative parameters (p=9)
        - "zeros":     use y0=[0,0,0,0] (p=5)
    max_nfev : int
        Max function evals for SciPy least_squares (kept as before).
    init_guess : dict
        Optional initial guesses for rho,alpha,delta,nu,gamma.

    Returns
    -------
    summary : dict (parameters + metrics + χ² p-value + "accept_model")
    t_years : np.ndarray (original years)
    df_sorted : pd.DataFrame
    sol : OdeSolution (solve_ivp result on the data grid)
    """
    # 1) Chronological order & shifted time axis
    df = df.sort_values("year").reset_index(drop=True)
    t_years = df["year"].values
    t = t_years - t_years[0]

    # 2) Observed (4 x T)
    data_mat = df[["P", "A", "I", "G"]].T.values
    # 3) Initial guesses (heuristics)
    if "enrolled_in_year" in df.columns:
        rho0 = max(1.0, float(np.nanmedian(df["enrolled_in_year"].values)))
    else:
        PA = df["P"].values + df["A"].values
        rho0 = max(1.0, float((PA[-1] - PA[0]) / max(t[-1], 1.0))) if len(PA) >= 2 else 50.0
    alpha0, delta0, nu0, gamma0 = 0.4, 0.6, 0.8, 0.02

    if init_guess is not None:
        rho0   = float(init_guess.get("rho",    rho0))
        alpha0 = float(init_guess.get("alpha",  alpha0))
        delta0 = float(init_guess.get("delta",  delta0))
        nu0    = float(init_guess.get("nu",     nu0))
        gamma0 = float(init_guess.get("gamma",  gamma0))

    # 4) Choose parameterization and bounds based on y0_mode
    if y0_mode == "zeros":
        # Fit 5 parameters, y0 fixed to 0
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], dtype=float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0])
        y0_fixed = np.zeros(4, dtype=float)
        fun = lambda x: _residuals_with_y0fixed(x, t, y0_fixed, data_mat)
        p_count = 5
    else:
        # Fit 9 parameters: [rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0]
        # Seed y0 near the first observed point (nonnegative)
        yP0, yA0, yI0, yG0 = [max(float(v), 0.0) for v in data_mat[:, 0]]
        x0 = np.array([rho0, alpha0, delta0, nu0, gamma0, yP0, yA0, yI0, yG0], dtype=float)
        lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 0.0,  0.0,  0.0,  0.0])
        ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0,  1e7, 1e7, 1e7, 1e7])
        fun = lambda x: _residuals_with_y0free(x, t, data_mat)
        p_count = 9

    # 5) Unweighted NLS
    res = least_squares(fun, x0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev,
                        loss='linear',      # pure L2 (nonlinear least squares)
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)

    # 6) Extract parameters & y0
    if y0_mode == "zeros":
        rho, alpha, delta, nu, gamma = res.x
        y0 = np.zeros(4, dtype=float)
    else:
        rho, alpha, delta, nu, gamma, yP0, yA0, yI0, yG0 = res.x
        y0 = np.array([yP0, yA0, yI0, yG0], dtype=float)

    pars_hat = np.array([rho, alpha, delta, nu, gamma], dtype=float)

    # 7) Derived quantities & integrate for predictions
    P_star, A_star = steady_state(rho, alpha, delta, nu, gamma)
    lam1, lam2     = eigenvalues(alpha, delta, nu, gamma)
    ratio          = alpha / delta

    sol = integrate_model(t, y0, pars_hat)

    # 8) Per-series metrics
    metrics = {}
    comps = ["P", "A", "I", "G"]
    for idx, comp in enumerate(comps):
        rmse, r2 = series_metrics(df[comp].values, sol.y[idx])
        metrics[f"RMSE_{comp}"] = rmse
        metrics[f"R2_{comp}"]   = r2

    # 9) Global unweighted metrics (same as before)
    R2_global, reduced_sse, rmse_global, sse4, dof_sse = global_gof_metrics(df, sol, p=p_count)
    metrics["R2_global"]       = R2_global
    metrics["Reduced_SSE"]     = reduced_sse
    metrics["RMSE_global"]     = rmse_global
    metrics["SSE_global"]      = sse4
    metrics["SSE_dof"]         = dof_sse

    # 10) NEW: χ² p-value (Poisson variance; does not change the fit)
    chi2_val, dof_chi2, p_value = chi2_pvalue_global(df, sol, p=p_count, variance_mode="poisson_model")
    metrics["chi2_global"]     = chi2_val
    metrics["chi2_dof"]        = dof_chi2
    metrics["chi2_p_value"]    = p_value
    # simple accept/reject at alpha = 0.05
    metrics["accept_model_0.05"] = bool(p_value >= 0.05)

    # 11) Assemble summary row
    summary = {
        "program": program_name,
        "y0_mode": y0_mode,
        "rho": rho, "alpha": alpha, "delta": delta, "nu": nu, "gamma": gamma,
        "alpha/delta": ratio, "P*": P_star, "A*": A_star,
        "lambda1": lam1, "lambda2": lam2,
        "success": res.success, "cost": res.cost, "nfev": res.nfev
    }
    summary.update(metrics)

    return summary, t_years, df, sol


# ------------------------------
# Plotting helpers
# ------------------------------
def save_series_plots(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    """Save four separate plots comparing data vs model for P,A,I,G."""
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
    """Save one PNG with a 2x2 grid of subplots comparing model vs data for P,A,I,G."""
    outdir.mkdir(parents=True, exist_ok=True)
    comps = [("P", 0, "Passive students (P)"),
             ("A", 1, "Active students (A)"),
             ("I", 2, "Inactive students (I, cumulative)"),
             ("G", 3, "Graduated students (G, cumulative)")]

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
    """Save one PNG with two 3D phase subplots: (A,P,I) and (A,P,G)."""
    outdir.mkdir(parents=True, exist_ok=True)

    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    Pd = df["P"].values; Ad = df["A"].values; Id = df["I"].values; Gd = df["G"].values

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
    parser = argparse.ArgumentParser(description="Fit the PAIG model to one or more program CSV files (UNWEIGHTED).")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input CSV file(s).")
    parser.add_argument("--outdir", type=str, default="paig_results", help="Output directory.")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots to the output directory.")
    parser.add_argument("--y0-mode", choices=["estimated","zeros"], default="estimated",
                        help="Initial condition at start of fit: 'estimated' (fit y0) or 'zeros' (y0=0).")
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

        # Console report
        rho = summary["rho"]; alpha = summary["alpha"]; delta = summary["delta"]
        nu = summary["nu"];   gamma = summary["gamma"]; ratio = summary["alpha/delta"]
        Pstar = summary["P*"]; Astar = summary["A*"]
        lam1 = summary["lambda1"]; lam2 = summary["lambda2"]

        print(f"  y0_mode: {summary['y0_mode']}")
        print(f"  rho={rho:.4f}, alpha={alpha:.4f}, delta={delta:.4f}, nu={nu:.4f}, gamma={gamma:.4f}")
        print(f"  alpha/delta={ratio:.4f},  P*={Pstar:.2f}, A*={Astar:.2f}")
        print(f"  eigenvalues: lambda1={lam1:.4f}, lambda2={lam2:.4f}")
        print(f"  RMSE: P={summary['RMSE_P']:.2f}, A={summary['RMSE_A']:.2f}, I={summary['RMSE_I']:.2f}, G={summary['RMSE_G']:.2f}")
        print(f"  R2:   P={summary['R2_P']:.3f}, A={summary['R2_A']:.3f}, I={summary['R2_I']:.3f}, G={summary['R2_G']:.3f}")
        print(f"  Global (unweighted): R2={summary['R2_global']:.3f}, RMSE4D={summary['RMSE_global']:.2f}, Reduced_SSE={summary['Reduced_SSE']:.2f}")
        print(f"  Chi² test (Poisson variances): χ²={summary['chi2_global']:.1f}, dof={summary['chi2_dof']}, p={summary['chi2_p_value']:.3g}, "f"accept@0.05={'YES' if summary['accept_model_0.05'] else 'NO'}")

        if args.save_plots:
            save_series_grid_plot(outdir, name, t_years, df_sorted, sol)
            save_series_3d_phase_plots(outdir, name, df_sorted, sol)

    summary_df = pd.DataFrame(summaries)
    summary_csv = outdir / "paig_fitted_parameters_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
