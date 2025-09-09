#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIG model fitter for ANY program CSV (UNWEIGHTED)
-------------------------------------------------
- Reads a program CSV
- Fits PAIG parameters by nonlinear least squares (unweighted)
- Optionally also estimates the initial conditions y0 = [P0,A0,I0,G0]
- Plots model vs data (2x2) and optional 3D phase plots
- Writes a one-line summary CSV across inputs

USAGE (examples):
  python paig_fit_any_csv.py --in "FST UG Programs/BSc_General.csv" --save-plots
  python paig_fit_any_csv.py --in *.csv --outdir results --save-plots
  python paig_fit_any_csv.py --in BSc_General.csv --ratio-max 0.99 --fit-y0 --save-plots
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


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
      ['year','P','A','I','G','enrolled_in_year','I_in_year','G_in_year'] (some optional)
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
            f"Need something like: '... Passive', '... Active', 'I Cumulative', 'G Cumulative'."
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
# PAIG ODE
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
# Residuals (unweighted)
# ------------------------------
def residuals_unconstrained(pars, t_eval, y0, data_mat):
    """
    pars = [rho, alpha, delta, nu, gamma]; y0 is FIXED.
    """
    pars = np.clip(pars, 1e-12, None)
    sol = integrate_model(t_eval, y0, pars)
    if not sol.success:
        return np.ones(data_mat.size) * 1e6
    return (sol.y - data_mat).ravel()


def residuals_ratio(pars, t_eval, y0, data_mat, ratio_max=0.99):
    """
    pars = [rho, delta, nu, gamma, r]; alpha = r*delta, 0<r<=ratio_max; y0 is FIXED.
    """
    rho, delta, nu, gamma, r = pars
    rho   = max(rho,   1e-12)
    delta = max(delta, 1e-12)
    nu    = max(nu,    1e-12)
    gamma = max(gamma, 1e-12)
    r     = min(max(r, 1e-12), ratio_max)
    alpha = r * delta

    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma]))
    if not sol.success:
        return np.ones(data_mat.size) * 1e6
    return (sol.y - data_mat).ravel()


# --- New: residuals when we also fit y0 = [P0,A0,I0,G0] ---
def residuals_unconstrained_with_y0(pars, t_eval, data_mat):
    """
    pars = [rho, alpha, delta, nu, gamma,  P0, A0, I0, G0]
    """
    rho, alpha, delta, nu, gamma, P0, A0, I0, G0 = pars
    # keep positivity
    rho   = max(rho,   1e-12)
    alpha = max(alpha, 1e-12)
    delta = max(delta, 1e-12)
    nu    = max(nu,    1e-12)
    gamma = max(gamma, 1e-12)
    y0 = np.array([max(P0, 0.0), max(A0, 0.0), max(I0, 0.0), max(G0, 0.0)], dtype=float)

    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma]))
    if not sol.success:
        return np.ones(data_mat.size) * 1e6
    return (sol.y - data_mat).ravel()


def residuals_ratio_with_y0(pars, t_eval, data_mat, ratio_max=0.99):
    """
    pars = [rho, delta, nu, gamma, r,  P0, A0, I0, G0] with alpha = r*delta
    """
    rho, delta, nu, gamma, r, P0, A0, I0, G0 = pars
    rho   = max(rho,   1e-12)
    delta = max(delta, 1e-12)
    nu    = max(nu,    1e-12)
    gamma = max(gamma, 1e-12)
    r     = min(max(r, 1e-12), ratio_max)
    alpha = r * delta
    y0 = np.array([max(P0, 0.0), max(A0, 0.0), max(I0, 0.0), max(G0, 0.0)], dtype=float)

    sol = integrate_model(t_eval, y0, np.array([rho, alpha, delta, nu, gamma]))
    if not sol.success:
        return np.ones(data_mat.size) * 1e6
    return (sol.y - data_mat).ravel()


# ------------------------------
# Derived quantities & metrics (unweighted)
# ------------------------------
def eigenvalues(alpha, delta, nu, gamma):
    a = (alpha + nu) + (delta + gamma)
    b = (alpha + nu) * (delta + gamma) - alpha * delta
    disc = a*a - 4*b
    sqrt_disc = np.sqrt(disc) if disc >= 0 else 1j*np.sqrt(-disc)
    return (-a + sqrt_disc)/2, (-a - sqrt_disc)/2


def steady_state(rho, alpha, delta, nu, gamma):
    denom = (alpha + nu) * (delta + gamma) - alpha * delta
    return rho*(delta+gamma)/denom, rho*alpha/denom


def series_metrics(y_true, y_pred, p: int = 5):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid  = y_true - y_pred
    T      = len(y_true)

    mae  = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true))**2))
    r2  = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    adj = 1.0 - (1.0 - r2)*((T - 1.0)/max(T - p - 1.0, 1.0)) if T > p + 1 else np.nan
    return mae, rmse, r2, adj


def global_gof_metrics(df: pd.DataFrame, sol, p_model: int):
    """
    Unweighted overall metrics using 4-D Euclidean residual at each time:
      d_t = sqrt( (P-P̂)^2 + (A-Â)^2 + (I-Î)^2 + (G-Ĝ)^2 )
    Returns: MAE, RMSE, R2, AdjR2, Chi2 (=sum of squared 4D distances), Chi2_reduced
    """
    data_mat = df[["P","A","I","G"]].T.values.astype(float)  # (4,T)
    pred_mat = sol.y.astype(float)
    res = pred_mat - data_mat                                # (4,T)

    T = res.shape[1]
    n = res.size  # = 4*T

    # 4-D distances each year
    d2 = np.sum(res**2, axis=0)          # (T,)
    d  = np.sqrt(d2)                      # (T,)

    mae  = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d2)))

    # Global R2 across all series (stacked)
    y  = data_mat.ravel()
    yhat = pred_mat.ravel()
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    R2  = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    adj = 1.0 - (1.0 - R2) * ((n - 1.0)/max(n - p_model - 1.0, 1.0)) if n > p_model + 1 else np.nan

    chi2 = ss_res
    dof  = max(n - p_model, 1)
    chi2_red = chi2 / dof

    return mae, rmse, R2, adj, chi2, chi2_red


# ------------------------------
# Fitting routine for one program
# ------------------------------
def fit_program(df: pd.DataFrame,
                program_name: str,
                ratio_max: Optional[float] = None,
                fit_y0: bool = False,
                max_nfev: int = 500,
                init_guess: Optional[Dict[str, float]] = None) -> Tuple[Dict, np.ndarray, pd.DataFrame, object]:
    """
    Unweighted nonlinear least squares.
    If fit_y0=True, estimate the initial state y0 = [P0,A0,I0,G0] as parameters.
    """
    # 1) Time base
    df = df.sort_values("year").reset_index(drop=True)
    t_years = df["year"].values
    t = t_years - t_years[0]

    # 2) Data matrix and default initial condition from data (for starting guess / non-fit mode)
    data_mat = df[["P","A","I","G"]].T.values
    y0_data  = data_mat[:, 0].astype(float)  # used either as fixed y0 or as starting guess

    # 3) Initial parameter guesses
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

    # 4) Build optimizer problem
    if not fit_y0:
        # ---- original mode: y0 is fixed to the first data row ----
        p_model = 5 if ratio_max is None else 5  # still 5 free params either way in this mode
        if ratio_max is None:
            x0 = np.array([rho0, alpha0, delta0, nu0, gamma0], float)
            lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
            ub = np.array([1e6,  5.0,  5.0,  5.0,  1.0])
            fun = lambda x: residuals_unconstrained(x, t, y0_data, data_mat)
        else:
            r0 = min(0.8, ratio_max)
            x0 = np.array([rho0, delta0, nu0, gamma0, r0], float)
            lb = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
            ub = np.array([1e6,  5.0,  5.0,  1.0,  ratio_max])
            fun = lambda x: residuals_ratio(x, t, y0_data, data_mat, ratio_max=ratio_max)

    else:
        # ---- new mode: also estimate y0 = [P0,A0,I0,G0] ----
        # Upper bounds for y0: allow up to 2× the max observed in each series (broad but sane).
        Pmax, Amax, Imax, Gmax = (df["P"].max(), df["A"].max(), df["I"].max(), df["G"].max())
        y0_lb = np.array([0.0, 0.0, 0.0, 0.0], float)
        y0_ub = np.array([max(1.0, 2*Pmax), max(1.0, 2*Amax), max(1.0, 2*Imax), max(1.0, 2*Gmax)], float)

        if ratio_max is None:
            # pars = [rho, alpha, delta, nu, gamma,  P0,A0,I0,G0]
            p_model = 9
            x0 = np.array([rho0, alpha0, delta0, nu0, gamma0, *y0_data], float)
            lb = np.concatenate([np.array([1e-6,1e-6,1e-6,1e-6,1e-6]), y0_lb])
            ub = np.concatenate([np.array([1e6, 5.0, 5.0, 5.0, 1.0]),   y0_ub])
            fun = lambda x: residuals_unconstrained_with_y0(x, t, data_mat)
        else:
            # pars = [rho, delta, nu, gamma, r,  P0,A0,I0,G0]
            p_model = 10
            r0 = min(0.8, ratio_max)
            x0 = np.array([rho0, delta0, nu0, gamma0, r0, *y0_data], float)
            lb = np.concatenate([np.array([1e-6,1e-6,1e-6,1e-6,1e-6]), y0_lb])
            ub = np.concatenate([np.array([1e6, 5.0, 5.0, 1.0, ratio_max]), y0_ub])
            fun = lambda x: residuals_ratio_with_y0(x, t, data_mat, ratio_max=ratio_max)

    # 5) Solve unweighted nonlinear least squares
    res = least_squares(fun, x0,
                        bounds=(lb, ub),
                        max_nfev=max_nfev,
                        loss='soft_l1',   #soft_l1 robust but still standard NLS, linear is the classic
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)

    # 6) Parse fitted parameters
    if not fit_y0:
        if ratio_max is None:
            rho, alpha, delta, nu, gamma = res.x
        else:
            rho, delta, nu, gamma, r = res.x
            alpha = r * delta
        y0_hat = y0_data.copy()   # fixed mode
    else:
        if ratio_max is None:
            rho, alpha, delta, nu, gamma, P0, A0, I0, G0 = res.x
        else:
            rho, delta, nu, gamma, r, P0, A0, I0, G0 = res.x
            alpha = r * delta
        y0_hat = np.array([P0, A0, I0, G0], float)

    pars_hat = np.array([rho, alpha, delta, nu, gamma], float)

    # 7) Derived quantities and re-integration
    P_star, A_star = steady_state(rho, alpha, delta, nu, gamma)
    lam1, lam2     = eigenvalues(alpha, delta, nu, gamma)
    ratio          = alpha / delta

    sol = integrate_model(t, y0_hat, pars_hat)

    # 8) Per-series metrics (unweighted; adjR2 uses p_model)
    metrics = {}
    comps = ["P", "A", "I", "G"]
    for idx, comp in enumerate(comps):
        mae, rmse, r2, adj = series_metrics(df[comp].values, sol.y[idx], p=p_model)
        metrics[f"MAE_{comp}"]  = mae
        metrics[f"RMSE_{comp}"] = rmse
        metrics[f"R2_{comp}"]   = r2
        metrics[f"AdjR2_{comp}"]= adj

    # 9) Global metrics (unweighted 4-D)
    mae_g, rmse_g, R2_g, AdjR2_g, chi2, chi2_red = global_gof_metrics(df, sol, p_model=p_model)
    metrics.update({
        "MAE_global": mae_g,
        "RMSE_global": rmse_g,
        "R2_global": R2_g,
        "AdjR2_global": AdjR2_g,
        "Chi2": chi2,
        "Chi2_reduced": chi2_red,
    })

    # 10) Summary row
    summary = {
        "program": program_name,
        "rho": rho, "alpha": alpha, "delta": delta, "nu": nu, "gamma": gamma,
        "alpha/delta": ratio, "P*": P_star, "A*": A_star,
        "lambda1": lam1, "lambda2": lam2,
        "success": res.success, "cost": res.cost, "nfev": res.nfev,
        "fit_y0": bool(fit_y0),
        "P0": y0_hat[0], "A0": y0_hat[1], "I0": y0_hat[2], "G0": y0_hat[3],
    }
    summary.update(metrics)

    return summary, t_years, df, sol


# ------------------------------
# Plotting helpers (unchanged)
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
        plt.ylim(0, y_max*1.05)
        plt.title(f"{name} — {title}")
        plt.xlabel("Year"); plt.ylabel("Students")
        plt.legend(); plt.grid(True); plt.tight_layout()
        png_path = outdir / f"{name}_{comp}.png"
        plt.savefig(png_path, dpi=300); plt.close()


def save_series_grid_plot(outdir: Path, name: str, t_years: np.ndarray, df: pd.DataFrame, sol) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    comps = [("P",0,"Passive students (P)"),
             ("A",1,"Active students (A)"),
             ("I",2,"Inactive students (I, cumulative)"),
             ("G",3,"Graduated students (G, cumulative)")]
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    for i,(comp,idx,title) in enumerate(comps):
        ax = axes[i//2, i%2]
        ax.plot(t_years, sol.y[idx], 'r-', label="Model")
        ax.scatter(df["year"].values, df[comp].values, label="Data")
        y_max = max(df[comp].max(), sol.y[idx].max())
        ax.set_ylim(0, y_max*1.05)
        ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Students")
        ax.grid(True); ax.legend()
    fig.suptitle(f"{name} — PAIG: Model vs Data (range: {int(t_years[0])}-{int(t_years[-1])})", fontsize=16)
    fig.tight_layout(rect=[0,0,1,0.96])
    png_path = outdir / f"{name}_PAIG_grid.png"
    fig.savefig(png_path, dpi=300); plt.close(fig)


def save_series_3d_phase_plots(outdir: Path, name: str, df: pd.DataFrame, sol) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    Pd, Ad, Id, Gd = df["P"].values, df["A"].values, df["I"].values, df["G"].values

    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(1,2,1, projection="3d")
    ax2 = fig.add_subplot(1,2,2, projection="3d")

    ax1.plot(Am, Pm, Im, 'r-', label="Model")
    ax1.scatter(Ad, Pd, Id, label="Data")
    ax1.set_xlabel("Active (A)"); ax1.set_ylabel("Passive (P)"); ax1.set_zlabel("Inactive (I, cum)")
    ax1.set_title("3D phase: A vs P vs I"); ax1.legend(); ax1.grid(True)

    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, label="Data")
    ax2.set_xlabel("Active (A)"); ax2.set_ylabel("Passive (P)"); ax2.set_zlabel("Graduated (G, cum)")
    ax2.set_title("3D phase: A vs P vs G"); ax2.legend(); ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=16)
    png_path = outdir / f"{name}_PAIG_3D_phase.png"
    fig.savefig(png_path, dpi=300); plt.close(fig)


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fit the PAIG model to one or more program CSV files (UNWEIGHTED).")
    parser.add_argument("--in", dest="inputs", nargs="+", required=True, help="Input CSV file(s).")
    parser.add_argument("--outdir", type=str, default="paig_results", help="Output directory.")
    parser.add_argument("--save-plots", action="store_true", help="Save PNG plots.")
    parser.add_argument("--ratio-max", type=float, default=None,
                        help="If set, enforce alpha = r*delta with 0 < r <= ratio_max (e.g. 0.99).")
    parser.add_argument("--fit-y0", action="store_true",
                        help="Estimate the initial states [P0,A0,I0,G0] instead of pinning them to the first data row.")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive plots.")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for inp in args.inputs:
        path = Path(inp)
        name = path.stem.replace(" ", "_")
        print(f"\n=== Processing {path} ===")

        df = load_program_csv(path)
        print(f"Loaded {len(df)} rows. Years {df['year'].iloc[0]}–{df['year'].iloc[-1]}")

        summary, t_years, df_sorted, sol = fit_program(
            df, name,
            ratio_max=args.ratio_max,
            fit_y0=args.fit_y0,
            max_nfev=500
        )
        summaries.append(summary)

        # Console report
        print(f"  rho={summary['rho']:.4f}, alpha={summary['alpha']:.4f}, delta={summary['delta']:.4f}, "
              f"nu={summary['nu']:.4f}, gamma={summary['gamma']:.4f}, fit_y0={summary['fit_y0']}")
        print(f"  P0={summary['P0']:.2f}, A0={summary['A0']:.2f}, I0={summary['I0']:.2f}, G0={summary['G0']:.2f}")
        print(f"  alpha/delta={summary['alpha/delta']:.4f},  P*={summary['P*']:.2f}, A*={summary['A*']:.2f}")
        print(f"  R2_global={summary['R2_global']:.3f}, AdjR2_global={summary['AdjR2_global']:.3f}, "
              f"RMSE_global={summary['RMSE_global']:.2f}, Chi2_reduced={summary['Chi2_reduced']:.2f}")

        if args.save_plots:
            save_series_grid_plot(outdir, name, t_years, df_sorted, sol)
            save_series_3d_phase_plots(outdir, name, df_sorted, sol)

    # Save summary CSV
    summary_df = pd.DataFrame(summaries)
    summary_csv = outdir / "paig_fitted_parameters_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary to: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()
