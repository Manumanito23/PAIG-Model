# app.py — PAIG Explorer (unweighted NLS) with cached fit + robust Sensitivity Analysis

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import zipfile
import streamlit as st

import paig_fit_any_csv as paig

st.set_page_config(page_title="PAIG Model Explorer", layout="wide")

if "step_m" not in st.session_state:
    st.session_state.step_m = 1 # 1 = each month (no sampling)

def _tables_for_repro(lf: dict,
                      yr_lo: int,
                      yr_hi: int,
                      monthly_mode: bool,
                      step_m: int,
                      norm_choice: str,
                      loss_choice: str,
                      scenarios: list | None,
                      years_ahead: int | None) -> dict:
    """
    Build all tables needed to reproduce the experiment.
    Returns a dict of DataFrames (and small arrays) ready to export.
    """
    summary = lf["summary"]
    df_used = lf["df"].copy()              # exact data slice used for the fit
    years   = np.asarray(lf["t_years"], float)
    sol     = lf["sol"]

    # ---- core tables ----
    data_used = df_used[["year","P","A","I","G"]].reset_index(drop=True)

    model_fit = pd.DataFrame({
        "year": years,
        "P_model": np.asarray(sol.y[0], float),
        "A_model": np.asarray(sol.y[1], float),
        "I_model": np.asarray(sol.y[2], float),
        "G_model": np.asarray(sol.y[3], float),
    })

    # residuals (Data - Model) aligned by year
    resid = data_used.merge(model_fit, on="year", how="left")
    for c, m in zip(["P","A","I","G"], ["P_model","A_model","I_model","G_model"]):
        resid[f"{c}_resid"] = resid[c] - resid[m]

    # fitted params + y0 + metadata
    params = pd.DataFrame([{
        "rho": summary["rho"], "alpha": summary["alpha"], "delta": summary["delta"],
        "nu": summary["nu"],   "gamma": summary["gamma"],
        "alpha_over_delta": summary["alpha/delta"],
        "y0_mode": summary["y0_mode"],
        "y0_P": summary["y0_P"], "y0_A": summary["y0_A"], "y0_I": summary["y0_I"], "y0_G": summary["y0_G"],
        "normalized_during_fit": (norm_choice == "Fit normalized"),
        "loss": loss_choice,
        "success": summary["success"], "cost": summary["cost"], "nfev": summary["nfev"],
    }])

    # scales (if normalized)
    scales = np.array(summary.get("scales", [1.0,1.0,1.0,1.0]), dtype=float)
    scales_df = pd.DataFrame(
        [{"scale_P": float(scales[0]), "scale_A": float(scales[1]),
          "scale_I": float(scales[2]), "scale_G": float(scales[3])}]
    )

    # --- add original-units parameters to params sheet ---
    # fitted-space params (possibly normalized)
    rho = float(summary["rho"]);    alpha = float(summary["alpha"])
    delta = float(summary["delta"]); nu = float(summary["nu"])
    gamma = float(summary["gamma"])

    sP, sA, sI, sG = map(float, scales)

    # back-transform (original units) – matches your paper and paig.fit_program
    rho_orig   = rho   * sP
    alpha_orig = alpha * (sA / sP)
    delta_orig = delta * (sP / sA)
    nu_orig    = nu    * (sP / sI)
    gamma_orig = gamma * (sA / sG)

    # append columns
    params["params_space"] = "fitted"  # these five are in the space used to fit
    params["rho_orig"]     = rho_orig
    params["alpha_orig"]   = alpha_orig
    params["delta_orig"]   = delta_orig
    params["nu_orig"]      = nu_orig
    params["gamma_orig"]   = gamma_orig

    # meta sheet
    meta = pd.DataFrame([{
        "program": summary["program"],
        "year_range_lo": int(yr_lo), "year_range_hi": int(yr_hi),
        "monthly_mode": bool(monthly_mode),
        "sampling_step_months": int(step_m if monthly_mode else 0),
        "rows_used": len(data_used),
    }])

    out = {
        "meta": meta,
        "params": params,
        "scales": scales_df,
        "data_used": data_used,
        "model_fit": model_fit,
        "residuals": resid,
    }

    # ---- optional: forecast/sensitivity tables (same logic as your figure) ----
    if scenarios is not None and years_ahead is not None:
        pars_fit = np.array([
            summary["rho"], summary["alpha"], summary["delta"], summary["nu"], summary["gamma"]
        ], dtype=float)

        last_year = int(years[-1])
        t_fore    = np.arange(0, years_ahead + 1, dtype=float)
        years_fore = np.arange(last_year, last_year + years_ahead + 1)

        # detect normalized fit
        use_norm = not np.allclose(scales, 1.0)
        y0_last  = np.asarray(sol.y[:, -1], dtype=float)

        def _forecast(pars_vec):
            if use_norm:
                y0_norm = y0_last / scales
                s = paig.integrate_model(t_fore, y0_norm, pars_vec)
                return s.y * scales[:, None]
            else:
                s = paig.integrate_model(t_fore, y0_last, pars_vec)
                return s.y

        base = _forecast(pars_fit)
        fore_base = pd.DataFrame({
            "year": years_fore,
            "P": base[0], "A": base[1], "I": base[2], "G": base[3],
        })

        # pack scenarios in "long" tidy form (label, year, series, value)
        rows = []
        for s in scenarios:
            pct = np.asarray(s["pct"], dtype=float)
            pars_mod = pars_fit * (1.0 + pct / 100.0)
            Ym = _forecast(pars_mod)
            for j, lab in enumerate(["P","A","I","G"]):
                for k, y in enumerate(years_fore):
                    rows.append({
                        "label": s["label"], "year": int(y), "series": lab, "value": float(Ym[j, k]),
                        "rho": pars_mod[0], "alpha": pars_mod[1], "delta": pars_mod[2],
                        "nu": pars_mod[3], "gamma": pars_mod[4],
                        "pct_rho": pct[0], "pct_alpha": pct[1], "pct_delta": pct[2],
                        "pct_nu": pct[3], "pct_gamma": pct[4],
                    })
        fore_scenarios = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["label","year","series","value","rho","alpha","delta","nu","gamma",
                     "pct_rho","pct_alpha","pct_delta","pct_nu","pct_gamma"])

        out["forecast_baseline"]  = fore_base
        out["forecast_scenarios"] = fore_scenarios

    return out


def _excel_bundle_bytes(tables: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        for name, df in tables.items():
            # Limit sheet name length and avoid empty-writer errors
            sheet = name[:31] if name else "sheet"
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(w, sheet_name=sheet, index=False)
    buf.seek(0)
    return buf.getvalue()


def _zip_csv_bundle_bytes(tables: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # README
        readme = (
            "PAIG reproducibility bundle\n"
            "---------------------------\n"
            "Tables are CSVs with UTF-8 encoding.\n"
            "ODEs:\n"
            "  dP = -(alpha + nu) P + delta A + rho\n"
            "  dA = alpha P - (delta + gamma) A\n"
            "  dI = nu P\n"
            "  dG = gamma A\n"
            "Parameter order: [rho, alpha, delta, nu, gamma]\n"
            "Time variable is years (t = year - year0).\n"
        )
        z.writestr("README.txt", readme)

        for name, df in tables.items():
            csv_bytes = (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_csv(index=False).encode("utf-8")
            z.writestr(f"{name}.csv", csv_bytes)
    buf.seek(0)
    return buf.getvalue()

def _zip_graphs_bundle_bytes(lf: dict,
                             sens_scenarios: list | None = None,
                             years_ahead: int = 20,
                             label_shift_years: int = 0) -> bytes:
    """
    Generate a ZIP containing high-resolution PNGs of:
      - Individual fits for P, A, I, G
      - Two 3D phase plots
      - Sensitivity analysis (P, A, I, G)
    """
    name = lf["program"]
    df = lf["df"]
    sol = lf["sol"]
    t_years = lf["t_years"]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:

        # --- individual fits ---
        comps = [("P", 0, "Passive (P)"),
                 ("A", 1, "Active (A)"),
                 ("I", 2, "Inactive (I, cumulative)"),
                 ("G", 3, "Graduated (G, cumulative)")]
        for comp, idx, title in comps:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(t_years, sol.y[idx], 'r-', lw=2, label="Model")
            ax.scatter(df["year"], df[comp], s=25, label="Data")
            ax.set_title(title)
            ax.set_xlabel("Year"); ax.set_ylabel("Students")
            ax.grid(True); ax.legend()
            fig.tight_layout()
            b = io.BytesIO()
            fig.savefig(b, format="png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            z.writestr(f"fit_{comp}.png", b.getvalue())

        # --- 3D phases ---
        Pm, Am, Im, Gm = sol.y
        Pd, Ad, Id, Gd = df["P"], df["A"], df["I"], df["G"]

        # A–P–I
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(Am, Pm, Im, 'r-', label="Model")
        ax.scatter(Ad, Pd, Id, s=18, label="Data")
        ax.set_xlabel("A"); ax.set_ylabel("P"); ax.set_zlabel("I (cum)")
        ax.set_title("3D phase: A–P–I"); ax.legend(); ax.grid(True)
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
        z.writestr("phase_API.png", b.getvalue())

        # A–P–G
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(Am, Pm, Gm, 'r-', label="Model")
        ax.scatter(Ad, Pd, Gd, s=18, label="Data")
        ax.set_xlabel("A"); ax.set_ylabel("P"); ax.set_zlabel("G (cum)")
        ax.set_title("3D phase: A–P–G"); ax.legend(); ax.grid(True)
        b = io.BytesIO(); fig.savefig(b, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
        z.writestr("phase_APG.png", b.getvalue())

        # --- Sensitivity (if any) ---
        if sens_scenarios:
            pars_fit = np.array([
                lf["summary"]["rho"], lf["summary"]["alpha"], lf["summary"]["delta"],
                lf["summary"]["nu"],  lf["summary"]["gamma"]
            ], dtype=float)

            last_year = int(lf["t_years"][-1])
            t_fore    = np.arange(0, years_ahead + 1, dtype=float)
            years_fore = np.arange(last_year, last_year + years_ahead + 1, dtype=int)

            # detect normalized fit and grab last state
            scales = np.array(lf["summary"].get("scales", [1.0,1.0,1.0,1.0]), dtype=float)
            use_norm = not np.allclose(scales, 1.0)
            y0_last_orig = np.asarray(lf["sol"].y[:, -1], dtype=float)

            def _forecast(pars_vec: np.ndarray) -> np.ndarray:
                if use_norm:
                    y0_norm = y0_last_orig / scales
                    s = paig.integrate_model(t_fore, y0_norm, pars_vec)
                    return s.y * scales[:, None]     # back to original units for plotting/saving
                else:
                    s = paig.integrate_model(t_fore, y0_last_orig, pars_vec)
                    return s.y

            base = _forecast(pars_fit)

            comps = [("P", 0, "Passive (P)"),
                     ("A", 1, "Active (A)"),
                     ("I", 2, "Inactive (I, cumulative)"),
                     ("G", 3, "Graduated (G, cumulative)")]

            for comp, idx, title in comps:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(years_fore, base[idx], '-', color='k', lw=2, label="no change (fitted)")
                for s in sens_scenarios:
                    pct = np.asarray(s["pct"], float)
                    pars_mod = pars_fit * (1.0 + pct / 100.0)
                    Y_mod = _forecast(pars_mod)
                    ax.plot(years_fore, Y_mod[idx], '--', lw=1.8, label=s["label"])
                ax.set_title(title)
                ax.set_xlabel("Year"); ax.set_ylabel("Students")
                ax.grid(True); ax.legend()
                fig.tight_layout()
                b = io.BytesIO(); fig.savefig(b, format="png", dpi=300, bbox_inches="tight"); plt.close(fig)
                z.writestr(f"sensitivity_{comp}.png", b.getvalue())

        # Add a small manifest
        z.writestr("README.txt", "Individual high-resolution graphs for fit, 3D phases, and sensitivity.\n")

    buf.seek(0)
    return buf.getvalue()


# ----------------- helpers -----------------
def slice_df_by_year(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    """Return a defensive slice for [y0, y1] with basic sanity check."""
    s = df[(df["year"] >= y0) & (df["year"] <= y1)].copy().reset_index(drop=True)
    if len(s) < 3:
        raise ValueError(f"Selected range {y0}-{y1} has too few rows ({len(s)}). Please pick ≥ 3.")
    return s

# -------- frequency helpers --------
def is_monthly(df: pd.DataFrame) -> bool:
    years = np.asarray(df.get("year", []), float)
    if years.size < 3:
        return False
    diffs = np.diff(np.unique(np.round(years, 6)))
    if diffs.size == 0:
        return False
    # monthly if median step < ~0.75 years
    return np.nanmedian(diffs) < 0.75

def downsample_months(df: pd.DataFrame, step_months: int) -> pd.DataFrame:
    """Keep every k-th row starting from the first, assuming consecutive months.
    Does NOT aggregate; preserves monotone cumulative I and G.
    """
    if step_months <= 1:
        return df
    s = df.sort_values("year").reset_index(drop=True)
    keep = np.arange(0, len(s), step_months)
    return s.iloc[keep].reset_index(drop=True)

def render_fit_png(
    name: str,
    t_years: np.ndarray,
    df: pd.DataFrame,
    sol,
    dpi: int = 150,
    label_shift_years: int = 0,
) -> bytes:
    """
    Build the 2x2 'Model vs Data' figure and return PNG bytes.
    It trims to the common length in case solver returns one point less.
    """
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]

        x_model = np.asarray(t_years, float)
        y_model = np.asarray(sol.y[idx], float)
        m = min(len(x_model), len(y_model))
        ax.plot(x_model[:m], y_model[:m], 'r-', label="Model", linewidth=2)

        ax.scatter(df["year"].values, df[comp].values, s=18, label="Data")
        y_max = float(max(np.nanmax(y_model[:m]), np.nanmax(df[comp].values)))
        ax.set_ylim(0.0, max(1.0, y_max * 1.1))
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Students")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)+label_shift_years:d}"))
        ax.grid(True)
        ax.legend()

    shown_years = (int(t_years[0] + label_shift_years), int(t_years[-1] + label_shift_years))
    fig.suptitle(f"{name} — PAIG: Model vs Data (range: {shown_years[0]}-{shown_years[1]})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_phase3d_png(name: str, df: pd.DataFrame, sol, dpi: int = 150) -> bytes:
    """Build the 3D phase projections (A-P-I and A-P-G) and return PNG bytes."""
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    Pd = np.asarray(df["P"].values, float)
    Ad = np.asarray(df["A"].values, float)
    Id = np.asarray(df["I"].values, float)
    Gd = np.asarray(df["G"].values, float)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    ax1.plot(Am, Pm, Im, 'r-', label="Model")
    ax1.scatter(Ad, Pd, Id, s=18, label="Data")
    ax1.set_xlabel("Active (A)"); ax1.set_ylabel("Passive (P)"); ax1.set_zlabel("Inactive (I, cumulative)")
    ax1.set_title("3D phase: A vs P vs I"); ax1.legend(); ax1.grid(True)

    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, s=18, label="Data")
    ax2.set_xlabel("Active (A)"); ax2.set_ylabel("Passive (P)"); ax2.set_zlabel("Graduated (G, cumulative)")
    ax2.set_title("3D phase: A vs P vs G"); ax2.legend(); ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def show_residual_plots(t_years, df_sorted, sol):
    """
    Draw 5 residual plots:
      - Four time series residuals: P, A, I, G (Data - Model), each centered at 0
      - One Euclidean residual norm ||r||_2 across P,A,I,G
    """
    data = df_sorted[["P", "A", "I", "G"]].to_numpy()   # (T,4)
    pred = sol.y.T                                      # (T,4)
    resid = data - pred

    names = ["P", "A", "I", "G"]

    # 4 residual series
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        r = resid[:, i]
        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.plot(t_years, r, marker="o", lw=1.5)
        ax.set_title(f"Residuals: {names[i]}")
        ax.set_ylabel("Data − Model")
        m = float(np.nanmax(np.abs(r))) if r.size else 1.0
        ax.set_ylim(-1.05 * m, 1.05 * m)
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # Euclidean norm
    euclid = np.sqrt((resid ** 2).sum(axis=1))
    fig2, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t_years, euclid, marker="o", lw=1.5)
    ax.set_title("Euclidean residual norm across P, A, I, G")
    ax.set_xlabel("Year")
    ax.set_ylabel("‖residual‖₂")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)


def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Save upload to a temp file and reuse the loader in paig module."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name


# ---- Sensitivity helpers (simulation + figure) ----
def _simulate_future_from_last(sol, pars, years_ahead: int):
    """
    Integrate PAIG starting at the last fitted state, for `years_ahead` years ahead (inclusive).
    Returns (t_future_relative, Y_future) where t is [0,1,2,...].
    """
    y_last = np.asarray(sol.y, float)[:, -1]
    t_future = np.arange(0.0, float(years_ahead) + 1.0, 1.0)
    sol_future = paig.integrate_model(t_future, y_last, np.asarray(pars, float))
    return t_future, np.asarray(sol_future.y, float)


def render_sensitivity_png(
    name: str,
    last_calendar_year: int,
    base_pars: np.ndarray,                 # [rho, alpha, delta, nu, gamma]
    sol_fit,                               # fitted solution (to get last state)
    scenarios: list,                       # [{"label": str, "pct": {"rho":..,"alpha":..,"delta":..,"nu":..,"gamma":..}}, ...]
    years_ahead: int = 20,
    dpi: int = 150,
    label_shift_years: int = 0,
) -> bytes:
    """
    Render a 2x2 forecast figure (P, A, I, G) showing:
      - Baseline (no param change) from last fitted year to +years_ahead
      - Any number of scenario lines with percentage tweaks to parameters
    """
    # Baseline
    t0_rel, Y0 = _simulate_future_from_last(sol_fit, base_pars, years_ahead)
    years_axis = last_calendar_year + t0_rel.astype(int)

    curves = [("no change in parameters", years_axis, Y0)]

    # Scenarios
    for sc in scenarios:
        pct = sc.get("pct", {})
        m = np.array([
            1.0 + float(pct.get("rho",   0.0))/100.0,
            1.0 + float(pct.get("alpha", 0.0))/100.0,
            1.0 + float(pct.get("delta", 0.0))/100.0,
            1.0 + float(pct.get("nu",    0.0))/100.0,
            1.0 + float(pct.get("gamma", 0.0))/100.0,
        ], dtype=float)
        pars_sc = np.clip(base_pars * m, [1e-6,1e-6,1e-6,1e-6,1e-6], [1e9,5,5,5,1])
        t_rel, Ys = _simulate_future_from_last(sol_fit, pars_sc, years_ahead)
        curves.append((sc.get("label","scenario"), last_calendar_year + t_rel.astype(int), Ys))

    # Plot
    comps = [("P",0,"Passive students (P)"),
             ("A",1,"Active students (A)"),
             ("I",2,"Inactive students (I, cumulative)"),
             ("G",3,"Graduated students (G, cumulative)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

    for i,(comp,idx,title) in enumerate(comps):
        ax = axes[i//2, i%2]
        # Baseline in black
        ax.plot(curves[0][1], curves[0][2][idx], '-', color='black', linewidth=2, label=curves[0][0])
        # Scenarios
        for j,(lab,x,y) in enumerate(curves[1:], start=0):
            ax.plot(x, y[idx], linewidth=2, label=lab,
                    color=color_cycle[j % max(1,len(color_cycle))])
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Students")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)+label_shift_years:d}"))
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"Sensitivity analysis — {name} (forecast {years_ahead} years)", fontsize=14)
    fig.tight_layout(rect=[0,0,1,0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ---- Fit-cache helpers: signature + cached rendering ------------------------
def _current_fit_signature(program_name: str, yr0: int, yr1: int,
                           y0_mode: str, loss_choice: str, norm_choice: str):
    """Build a small dictionary that uniquely represents the current base-fit setup."""
    return {
        "program": program_name,
        "yr0": int(yr0), "yr1": int(yr1),
        "y0_mode": y0_mode,
        "loss": loss_choice,
        "norm": norm_choice,
    }

def _signatures_equal(a: dict, b: dict) -> bool:
    """Safe equality for the few keys we care about."""
    if not a or not b:
        return False
    return all(a.get(k) == b.get(k) for k in ("program","yr0","yr1","y0_mode","loss","norm"))

def _render_fit_from_cache(cache: dict, label_shift_on: bool):
    """
    Re-render the last fit (summary, metrics, 2x2, 3D, residuals) from the cached objects.
    This is used on reruns triggered by Sensitivity widgets so the base-fit never 'disappears'.
    """
    summary = cache["summary"]
    t_years  = cache["t_years"]
    df_sorted = cache["df"]
    sol       = cache["sol"]

    # Summary
    # Summary table
    if summary.get("scales") and not np.allclose(summary["scales"], [1,1,1,1]):
        st.markdown("#### Parameters (normalized space)")
        st.dataframe(
            pd.DataFrame([summary])[["rho","alpha","delta","nu","gamma"]],
            use_container_width=True
        )

        st.markdown("#### Parameters (back-transformed to original units)")
        st.dataframe(
            pd.DataFrame([summary])[["rho_orig","alpha_orig","delta_orig","nu_orig","gamma_orig"]],
            use_container_width=True
        )
    else:
        st.markdown("#### Parameters (original fit, real units)")
        st.dataframe(
            pd.DataFrame([summary])[["rho","alpha","delta","nu","gamma"]],
            use_container_width=True
        )

    # Metrics
    p = 9 if summary["y0_mode"] == "estimated" else 5
    metrics_df = paig.metrics_table(df_sorted, sol, p=p)
    st.markdown("#### Statistical metrics")
    st.dataframe(
        metrics_df.style.format({
            "Mean Absolute Error (MAE)": "{:.3f}",
            "Root Mean Square Error (RMSE)": "{:.3f}",
            "Coefficient of Determination (R^2)": "{:.3f}",
            "Adjusted R^2": "{:.3f}",
            "Chi-squared": "{:.3f}",
            "Reduced Chi-squared": "{:.3f}",
        }),
        use_container_width=True
    )

    # 2x2 plot (respecting your special 'zeros' y0_mode)
    if summary["y0_mode"] == "zeros":
        t_plot_years = np.r_[t_years[0] - 1, t_years]
        class _SolPlot: pass
        sol_plot = _SolPlot()
        sol_plot.y = np.hstack([np.zeros((4, 1)), np.asarray(sol.y, float)])
        sol_plot.t = t_plot_years - t_plot_years[0]
        sol_plot.success = True
    else:
        t_plot_years = t_years
        sol_plot = sol

    png = render_fit_png(
        summary["program"], t_plot_years, df_sorted, sol_plot,
        dpi=150, label_shift_years=(1 if label_shift_on else 0),
    )
    st.image(png, use_container_width=True)

    # 3D
    png3d = render_phase3d_png(summary["program"], df_sorted, sol, dpi=150)
    st.image(png3d, use_container_width=True)

    # Residuals
    with st.expander("Residual analysis (time series)", expanded=False):
        show_residual_plots(t_years, df_sorted, sol)
        st.caption(
            "Residuals are defined as Data − Model. "
            "Panels show per-series residuals (P, A, I, G) centered at 0, "
            "and the global Euclidean residual norm across the four series."
        )


# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

label_shift_on = st.sidebar.checkbox("Shift year labels by 1 (display only)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Initial guesses for the optimizer")
rho0   = st.sidebar.slider("rho (inflow)",  min_value=1.0, max_value=500.0, value=200.0, step=1.0)
alpha0 = st.sidebar.slider("alpha",         min_value=0.0, max_value=1.0,   value=0.4,   step=0.01)
delta0 = st.sidebar.slider("delta",         min_value=0.0, max_value=1.0,   value=0.6,   step=0.01)
nu0    = st.sidebar.slider("nu",            min_value=0.0, max_value=1.0,   value=0.8,   step=0.01)
gamma0 = st.sidebar.slider("gamma",         min_value=0.0, max_value=1.0,   value=0.02,  step=0.001)

st.sidebar.markdown("---")
loss_choice = st.sidebar.radio(
    "Loss function",
    options=["linear", "soft_l1"],
    index=0,
    help="‘soft_l1’ is robust to outliers; ‘linear’ is standard least squares."
)
norm_choice = st.sidebar.radio(
    "Normalize scales during fit",
    options=["Fit original", "Fit normalized"],
    index=0,
    help="Scale each series by its max during fitting; back-transform for plots/metrics."
)


# ----------------- main -----------------
st.title("PAIG Model Explorer — Nonlinear Least Squares")

if not uploaded:
    st.info("Upload a CSV to begin. Expected columns similar to your loader (Year, Passive, Active, I cumulative, G cumulative).")
    st.stop()

# Load CSV
try:
    df_full, program_name = load_program_csv_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

monthly_mode = is_monthly(df_full)
df_effective = (
    downsample_months(df_full, int(st.session_state.step_m))
    if monthly_mode else
    df_full
)

year_min = int(np.floor(df_effective["year"].min()))
year_max = int(np.ceil(df_effective["year"].max()))

# Two columns so the Preview table is not full width
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Preview")
    st.dataframe(df_effective, height=340, use_container_width=False)  # st.dataframe(df_view, height=340, use_container_width=False)     # shows sampled + year-range slice
    yr_lo, yr_hi = st.slider(
        "Year range",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
        step=1,
        help="Select the fitting window in years."
    )
    mask = (df_effective["year"] >= yr_lo) & (df_effective["year"] <= yr_hi)
    df_view = df_effective.loc[mask].copy()

    # Sampling step (months), visible only if CSV is monthly
    if monthly_mode:
        st.slider(
            "Sampling step (months)",
            min_value=1, max_value=12, step=1,
            value=int(st.session_state.step_m),
            key="step_m",
            help="Keep every k-th month from the uploaded table (1 = every month, 12 ≈ annual)."
        )
        st.caption(f"Detected monthly data. Using every {int(st.session_state.step_m)} month(s). "
                f"Rows after sampling: {len(df_effective)}")

with col_right:
    st.markdown("#### Initial condition at start of fit (y₀)")

    # 3 options: estimated / zeros / data
    default_mode = "zeros" if year_min == yr_lo else "estimated"
    labels = {
        "estimated": "estimated (fit y₀)",
        "zeros":     "zeros (P=A=I=G=0 at previous year)",
        "data":      "data (use first-year observation)"
    }
    if "y0_mode" not in st.session_state:
        st.session_state.y0_mode = default_mode
    selected_label = st.radio(
        " ",
        options=[labels["estimated"], labels["zeros"], labels["data"]],
        index=["estimated","zeros","data"].index(st.session_state.y0_mode),
        horizontal=True,
        label_visibility="collapsed"
    )
    inv = {v:k for k,v in labels.items()}
    y0_mode = inv[selected_label]
    st.session_state.y0_mode = y0_mode

    # Build the current signature BEFORE the Run button; keep it in sync on every rerun
    current_sig = _current_fit_signature(
        program_name, yr_lo, yr_hi, st.session_state.y0_mode, loss_choice, norm_choice
    )
    current_sig["step_m"] = int(st.session_state.step_m) if monthly_mode else 1
    current_sig["monthly_mode"] = bool(monthly_mode)

    # If any base-fit control changed, invalidate cache and (optionally) clear sensitivity scenarios
    if "fit_signature" in st.session_state:
        if not _signatures_equal(st.session_state["fit_signature"], current_sig):
            st.session_state.pop("last_fit", None)
            st.session_state.pop("sens_scenarios", None)  # optional: start fresh scenarios for a new base fit
    st.session_state["fit_signature"] = current_sig

    # Run button
    run = st.button("Run Fit with current initial guesses", type="primary")

    if run:
        try:
            df_slice = df_view.copy() # It can also be slice_df_by_year(df_effective, yr_lo, yr_hi)

            # Fit (unweighted); p depends on y0_mode (5 or 9)
            summary, t_years, df_sorted, sol = paig.fit_program(
                df_slice,
                program_name,
                init_guess=dict(rho=rho0, alpha=alpha0, delta=delta0, nu=nu0, gamma=gamma0),
                y0_mode=y0_mode,
                loss=loss_choice,
                normalize=(True if norm_choice == "Fit normalized" else False)
            )

            # Choose which solution to plot in the 2x2 (respecting 'zeros' mode visualization)
            if summary["y0_mode"] == "zeros":
                t_plot_years = np.r_[t_years[0] - 1, t_years]
                class _SolPlot: pass
                sol_plot = _SolPlot()
                sol_plot.y = np.hstack([np.zeros((4, 1)), np.asarray(sol.y, float)])
                sol_plot.t = t_plot_years - t_plot_years[0]
                sol_plot.success = True
            else:
                t_plot_years = t_years
                sol_plot = sol

            # Summary table
            if summary.get("scales") and not np.allclose(summary["scales"], [1,1,1,1]):
                st.markdown("#### Parameters (normalized space)")
                st.dataframe(
                    pd.DataFrame([summary])[["rho","alpha","delta","nu","gamma"]],
                    use_container_width=True
                )

                st.markdown("#### Parameters (back-transformed to original units)")
                st.dataframe(
                    pd.DataFrame([summary])[["rho_orig","alpha_orig","delta_orig","nu_orig","gamma_orig"]],
                    use_container_width=True
                )
            else:
                st.markdown("#### Parameters (original fit, real units)")
                st.dataframe(
                    pd.DataFrame([summary])[["rho","alpha","delta","nu","gamma"]],
                    use_container_width=True
                )


            # Metrics
            p = 9 if y0_mode == "estimated" else 5
            metrics_df = paig.metrics_table(df_sorted, sol, p=p)
            st.markdown("#### Statistical metrics")
            st.dataframe(
                metrics_df.style.format({
                    "Mean Absolute Error (MAE)": "{:.3f}",
                    "Root Mean Square Error (RMSE)": "{:.3f}",
                    "Coefficient of Determination (R^2)": "{:.3f}",
                    "Adjusted R^2": "{:.3f}",
                    "Chi-squared": "{:.3f}",
                    "Reduced Chi-squared": "{:.3f}",
                }),
                use_container_width=True
            )

            # 2x2 figure
            png = render_fit_png(
                program_name,
                t_plot_years,
                df_sorted,
                sol_plot,
                dpi=150,
                label_shift_years=(1 if label_shift_on else 0),
            )
            st.image(png, use_container_width=True)

            # 3D figure
            png3d = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
            st.image(png3d, use_container_width=True)

            # Residual plots
            with st.expander("Residual analysis (time series)", expanded=False):
                show_residual_plots(t_years, df_sorted, sol)
                st.caption(
                    "Residuals are defined as Data − Model. "
                    "Panels show per-series residuals (P, A, I, G) centered at 0, "
                    "and the global Euclidean residual norm across the four series."
                )

            # Cache the last fit so it persists across reruns
            st.session_state["last_fit"] = {
                "program": program_name,
                "summary": summary,   # includes rho, alpha, delta, nu, gamma, y0_mode, etc.
                "t_years": t_years,   # data years (no display shift)
                "df": df_sorted,      # slice used for the fit
                "sol": sol,           # solution aligned with t_years
            }
            st.session_state["fit_signature"] = current_sig

        except Exception as e:
            st.error(f"Fit failed: {e}")


# ---- If user didn't press Run this time, still render the cached fit if valid ----
if "last_fit" in st.session_state and _signatures_equal(st.session_state.get("fit_signature", {}), 
                                                       _current_fit_signature(program_name, yr_lo, yr_hi, st.session_state.y0_mode, loss_choice, norm_choice)):
    _render_fit_from_cache(st.session_state["last_fit"], label_shift_on)


# ====================== Sensitivity Analysis (forecast) ======================
# This expander lives OUTSIDE 'if run:' so it doesn't disappear, and
# touching its controls won't erase the base-fit (we re-render from cache above).
with st.expander("Sensitivity analysis (forecast beyond fitted period)", expanded=False):
    lf = st.session_state.get("last_fit", None)
    if lf is None:
        st.info("⚠️ Run a fit first.")
    else:
        # Keep scenarios list in session so you can add many lines without losing them.
        if "sens_scenarios" not in st.session_state:
            st.session_state["sens_scenarios"] = []   # each: {"label": str, "pct": [rho%, alpha%, delta%, nu%, gamma%]}

        st.caption(
            "Forecast continues from the **last data year** using the fitted state as y₀. "
            "Add as many lines as you want by modifying parameters **by percent** "
            "relative to the fitted values."
        )

        # Horizon + table toggle
        colh1, colh2 = st.columns([1,1])
        with colh1:
            years_ahead = st.number_input(
                "Forecast years",
                min_value=5, max_value=50, value=20, step=1,
                help="Years beyond the last observed year."
            )
        with colh2:
            show_table = st.checkbox("Show scenarios table", value=True)

        # Form to add one scenario (form prevents immediate rerender until submit)
        with st.form("sens_add_form", clear_on_submit=False):
            st.markdown("**Add a new line (percent deltas w.r.t. fitted params)**")
            c1, c2, c3 = st.columns(3)
            with c1:
                rho_pct   = st.number_input("enrolment Δ rho (%)",   value=0.0, step=1.0, format="%.2f")
                alpha_pct = st.number_input("engagement Δ alpha (%)", value=0.0, step=1.0, format="%.2f")
            with c2:
                delta_pct = st.number_input("disengagement Δ delta (%)", value=0.0, step=1.0, format="%.2f")
                nu_pct    = st.number_input("inactivation Δ nu (%)",    value=0.0, step=1.0, format="%.2f")
            with c3:
                gamma_pct = st.number_input("graduation Δ gamma (%)", value=0.0, step=0.1, format="%.3f")
                label     = st.text_input("Label", value="custom change", help="Legend text for this line")

            add_line = st.form_submit_button("➕ Add line")
            if add_line:
                st.session_state["sens_scenarios"].append({
                    "label": label.strip() if label.strip() else "custom change",
                    "pct":   [rho_pct, alpha_pct, delta_pct, nu_pct, gamma_pct],
                })

        # Buttons to manage the list
        cbtn1, cbtn2, _ = st.columns([1,1,6])
        if cbtn1.button("🗑️ Clear lines"):
            st.session_state["sens_scenarios"].clear()
        if cbtn2.button("↩️ Remove last"):
            if st.session_state["sens_scenarios"]:
                st.session_state["sens_scenarios"].pop()

        # Optional table view
        if show_table and st.session_state["sens_scenarios"]:
            _df_scen = pd.DataFrame(
                [{
                    "label": s["label"],
                    "Δ rho %": s["pct"][0], "Δ alpha %": s["pct"][1], "Δ delta %": s["pct"][2],
                    "Δ nu %": s["pct"][3], "Δ gamma %": s["pct"][4],
                } for s in st.session_state["sens_scenarios"]]
            )
            st.dataframe(_df_scen, use_container_width=True, hide_index=True)

        # ---- Build and show the Sensitivity figure ----
        pars_fit = np.array([
            lf["summary"]["rho"],
            lf["summary"]["alpha"],
            lf["summary"]["delta"],
            lf["summary"]["nu"],
            lf["summary"]["gamma"]
        ], dtype=float)

        last_year = int(lf["t_years"][-1])
        t_fore = np.arange(0, years_ahead + 1, dtype=float)

        # Detect whether the base fit was normalized
        scales = np.array(lf["summary"].get("scales", [1.0, 1.0, 1.0, 1.0]), dtype=float)
        use_normalized = not np.allclose(scales, 1.0)

        # y0 for forecasting
        y0_orig = np.asarray(lf["sol"].y[:, -1], dtype=float)

        # Helper: integrate either in normalized space (then scale back) or in original space
        def _forecast_with_params(pars_vec: np.ndarray):
            if use_normalized:
                y0_norm = y0_orig / scales
                sol_f = paig.integrate_model(t_fore, y0_norm, pars_vec)  # params are from normalized fit
                return sol_f.y * scales[:, None]  # back to original units for plotting
            else:
                sol_f = paig.integrate_model(t_fore, y0_orig, pars_vec)
                return sol_f.y

        # Baseline (no change)
        Y_base = _forecast_with_params(pars_fit)
        years_fore = np.arange(last_year, last_year + years_ahead + 1)

        comps = [("P",0,"Passive (P)"), ("A",1,"Active (A)"),
                 ("I",2,"Inactive (I, cumulative)"), ("G",3,"Graduated (G, cumulative)")]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i,(comp,idx,title) in enumerate(comps):
            ax = axes[i//2, i%2]
            ax.plot(years_fore, Y_base[idx], '-', linewidth=2, color='k', label="no change (fitted)")
            ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Students")
            ax.grid(True, alpha=0.3)

        # Scenarios (each line modifies parameters by percent relative to the fit)
        for s in st.session_state["sens_scenarios"]:
            pct = np.asarray(s["pct"], dtype=float)
            pars_mod = pars_fit * (1.0 + pct / 100.0)
            Y_mod = _forecast_with_params(pars_mod)
            for i,(_,idx,_) in enumerate(comps):
                axes[i//2, i%2].plot(years_fore, Y_mod[idx], '--', linewidth=1.8, label=s["label"])

        for ax in axes.ravel():
            ax.legend(loc="best", framealpha=0.9)
            ymin, ymax = ax.get_ylim()
            pad = 0.06
            ax.set_ylim(ymin * (1.0 - pad), max(ymax * (1.0 + pad), 1.0))

        fig.suptitle(f"Sensitivity analysis — {lf['program']} (start at {last_year}, +{years_ahead}y)", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig, clear_figure=True)
# ==================== END Sensitivity Analysis ====================


# ==================== Reproducibility / Export ====================
with st.expander("Reproducibility: download data, parameters, and forecasts", expanded=False):
    lf = st.session_state.get("last_fit", None)
    if lf is None:
        st.info("Run a fit to enable exports.")
    else:
        # use current controls for meta
        monthly_flag = bool(is_monthly(df_full))
        step_m_cur   = int(st.session_state.step_m) if monthly_flag else 0
        sens_list    = st.session_state.get("sens_scenarios", [])
        # re-use current horizon if the Sensitivity expander was used; default 20
        years_ahead  = 20

        tables = _tables_for_repro(
            lf=lf,
            yr_lo=yr_lo, yr_hi=yr_hi,
            monthly_mode=monthly_flag,
            step_m=step_m_cur,
            norm_choice=norm_choice,
            loss_choice=loss_choice,
            scenarios=sens_list if sens_list else None,
            years_ahead=years_ahead if sens_list else None,
        )

        # Buttons
        c1, c2, c3 = st.columns(3)
        with c1:
            xlsx_bytes = _excel_bundle_bytes(tables)
            st.download_button(
                "⬇️ Excel bundle (.xlsx)",
                data=xlsx_bytes,
                file_name=f"{lf['program']}_paig_repro.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with c2:
            zip_bytes = _zip_csv_bundle_bytes(tables)
            st.download_button(
                "⬇️ CSV bundle (.zip)",
                data=zip_bytes,
                file_name=f"{lf['program']}_paig_repro_csvs.zip",
                mime="application/zip",
                use_container_width=True,
            )
        with c3:
            graphs_zip = _zip_graphs_bundle_bytes(
                lf,
                sens_scenarios=sens_list if sens_list else None,
                years_ahead=years_ahead,
                label_shift_years=(1 if label_shift_on else 0),
            )
            st.download_button(
                "⬇️ Graphs bundle (.zip)",
                data=graphs_zip,
                file_name=f"{lf['program']}_paig_graphs.zip",
                mime="application/zip",
                use_container_width=True,
            )

        st.caption(
            "Excel sheets: meta, params, scales, data_used, model_fit, residuals"
            + (", forecast_baseline, forecast_scenarios" if ("forecast_baseline" in tables) else "")
            + ". CSV bundle contains the same tables plus a README.txt with ODEs and parameter order."
        )
# ==================== End of Reproducibility / Export ====================