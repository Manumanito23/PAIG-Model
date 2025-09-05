# app.py (UNWEIGHTED)
# PAIG interactive app (Streamlit)
# - Upload CSV
# - Select year range
# - Choose initial guesses for (rho, alpha, delta, nu, gamma)
# - Optional ratio constraint alpha = r * delta
# - Fit PAIG (pure unweighted NLS) and plot Model vs Data (2x2)
# - Show 3D phase plots
# - Show metrics table:
#     Global (overall) from 4D Euclidean distances
#     Per-series (P, A, I, G) with standard unweighted formulas
# - Optional: save PNGs to ./paig_results and download assets

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)
import streamlit as st

import paig_fit_any_csv as paig

st.set_page_config(page_title="PAIG Model Explorer (Unweighted)", layout="wide")

# --------------------------------------------------------------------
# Metrics helpers (UNWEIGHTED, standard formulas)
# --------------------------------------------------------------------
def _adj_r2_from_r2(R2: float, n: int, p: int) -> float:
    """
    Adjusted R^2 = 1 - (1 - R^2) * ((n - 1) / (n - p - 1)).
    If denominator <= 0 or R2 is NaN, returns NaN.
    """
    if np.isnan(R2):
        return np.nan
    denom = (n - p - 1)
    if denom <= 0:
        return np.nan
    return 1.0 - (1.0 - R2) * ((n - 1.0) / denom)


def series_metrics_full(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    """
    Standard, UNWEIGHTED metrics for a single series:
      - MAE
      - RMSE
      - R^2
      - Adjusted R^2 (p = number of fitted parameters)
      - Chi-squared (SSE, assuming unit variance)
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

    dof = max(n - p, 1)
    chi2 = ss_res
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

    At each year t, define e_t = || y_model(:,t) - y_data(:,t) ||_2.
    Then:
      SSE = sum_t e_t^2 = sum_{k,t} (resid_{k,t})^2
      TSS = sum_{k,t} (y_{k,t} - mean_k)^2  (per-series centering)

      MAE_global  = mean_t e_t
      RMSE_global = sqrt( mean_t e_t^2 ) = sqrt(SSE / (4T))
      R2_global   = 1 - SSE/TSS
      Adj_R2_global uses n = 4T and p parameters
      Chi2_global = SSE (unit variance), Chi2_reduced = SSE / (n - p)
    """
    data_mat = df[["P", "A", "I", "G"]].T.values.astype(float)  # (4, T)
    pred_mat = sol.y.astype(float)                               # (4, T)
    res_mat  = pred_mat - data_mat                               # (4, T)

    e = np.sqrt(np.sum(res_mat**2, axis=0))  # (T,)
    T = e.size
    n = 4 * T

    SSE = float(np.sum(e**2))                # == np.sum(res_mat**2)
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
    Create the requested table:
    Rows: Global (overall), P, A, I, G
    Cols: MAE, RMSE, R^2, Adjusted R^2, Chi-squared, Reduced Chi-squared
    All formulas are standard, UNWEIGHTED.
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
# --------------------------------------------------------------------


# ----------------- helpers -----------------
def slice_df_by_year(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    s = df[(df["year"] >= y0) & (df["year"] <= y1)].copy().reset_index(drop=True)
    if len(s) < 3:
        raise ValueError(f"Selected range {y0}-{y1} has too few rows ({len(s)}). Please pick ≥ 3.")
    return s


def render_fit_png(
    name: str,
    t_years: np.ndarray,
    df: pd.DataFrame,
    sol,
    dpi: int = 140,
) -> bytes:
    """Render 2x2 plot: Model (line) vs Data (scatter)."""
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]
        ax.plot(t_years, sol.y[idx], 'r-', label="Model", linewidth=2)
        ax.scatter(df["year"].values, df[comp].values, s=18, label="Data")
        y_max = float(max(np.nanmax(sol.y[idx]), np.nanmax(df[comp].values)))
        ax.set_ylim(0.0, y_max * 1.1)
        ax.set_title(title)
        ax.set_xlabel("Year")
        ax.set_ylabel("Students")
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"{name} — PAIG: Model vs Data (range: {int(t_years[0])}-{int(t_years[-1])})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_phase3d_png(
    name: str,
    df: pd.DataFrame,
    sol,
    dpi: int = 150
) -> bytes:
    """
    Render two 3D phase plots:
      Left:  A (x) vs P (y) vs I (z)
      Right: A (x) vs P (y) vs G (z)
    """
    # Model trajectories
    Pm, Am, Im, Gm = sol.y[0], sol.y[1], sol.y[2], sol.y[3]
    # Data points
    Pd = np.asarray(df["P"].values, float)
    Ad = np.asarray(df["A"].values, float)
    Id = np.asarray(df["I"].values, float)
    Gd = np.asarray(df["G"].values, float)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    # ----- Left: A (x), P (y), I (z)
    ax1.plot(Am, Pm, Im, 'r-', label="Model")
    ax1.scatter(Ad, Pd, Id, s=18, label="Data")
    ax1.set_xlabel("Active (A)")
    ax1.set_ylabel("Passive (P)")
    ax1.set_zlabel("Inactive (I, cumulative)")
    ax1.set_title("3D phase: A vs P vs I")
    ax1.legend()
    ax1.grid(True)

    # ----- Right: A (x), P (y), G (z)
    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, s=18, label="Data")
    ax2.set_xlabel("Active (A)")
    ax2.set_ylabel("Passive (P)")
    ax2.set_zlabel("Graduated (G, cumulative)")
    ax2.set_title("3D phase: A vs P vs G")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Persist upload to a temp .csv and reuse the PAIG loader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name


# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls (Unweighted)")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

ratio_toggle = st.sidebar.checkbox("Enforce α = r·δ", value=False)
ratio_max   = st.sidebar.number_input("ratio_max", value=0.99, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
max_nfev    = st.sidebar.number_input("max_nfev", value=500, min_value=100, max_value=5000, step=50)
save_pngs   = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Initial guesses used by the optimizer (pure NLS, unweighted)")

rho0   = st.sidebar.slider("rho (inflow)",  min_value=1.0, max_value=500.0, value=200.0, step=1.0)
alpha0 = st.sidebar.slider("alpha",         min_value=0.0, max_value=1.0,   value=0.4,   step=0.01)
delta0 = st.sidebar.slider("delta",         min_value=0.0, max_value=1.0,   value=0.6,   step=0.01)
nu0    = st.sidebar.slider("nu",            min_value=0.0, max_value=1.0,   value=0.8,   step=0.01)
gamma0 = st.sidebar.slider("gamma",         min_value=0.0, max_value=1.0,   value=0.02,  step=0.001)


# ----------------- main -----------------
st.title("PAIG Model Explorer — Unweighted Nonlinear Least Squares")

if not uploaded:
    st.info("Upload a CSV to begin. Expected columns similar to your PAIG loader (Year, Passive, Active, I cumulative, G cumulative).")
    st.stop()

# Load and preview
try:
    df_full, program_name = load_program_csv_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Preview")
    st.dataframe(df_full, use_container_width=True)
    y_min, y_max = int(df_full["year"].min()), int(df_full["year"].max())
    yr0, yr1 = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max), step=1)

with col_right:
    st.subheader("Fit")
    if st.button("Run Fit with current initial guesses", type="primary"):
        try:
            df_slice = slice_df_by_year(df_full, yr0, yr1)

            summary, t_years, df_sorted, sol = paig.fit_program(
                df_slice,
                program_name,
                ratio_max=(float(ratio_max) if ratio_toggle else None),
                max_nfev=int(max_nfev),
                init_guess=dict(rho=rho0, alpha=alpha0, delta=delta0, nu=nu0, gamma=gamma0),
            )

            # ---------------- Parameters + global metrics overview ----------------
            cols = [
                "program", "rho", "alpha", "delta", "nu", "gamma", "alpha/delta",
                "R2_global", "MSE_reduced", "RMSE_global",
            ]
            st.dataframe(pd.DataFrame([summary])[cols], use_container_width=True)

            # ---------------- 2x2 plots + 3D phase ----------------
            png = render_fit_png(program_name, t_years, df_sorted, sol, dpi=150)
            st.image(png, use_container_width=True)
            phase_png = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
            st.image(phase_png, use_container_width=True, caption="3D phase plots")

            # ---------------- Metrics table (global + per series) ----------------
            st.subheader("Goodness-of-fit metrics (unweighted)")
            metrics_tbl = build_metrics_table(df_sorted, sol, p=5)
            st.dataframe(metrics_tbl.style.format(precision=3), use_container_width=True)

            # Download CSV of the metrics table
            metrics_csv_bytes = metrics_tbl.to_csv().encode("utf-8")
            st.download_button(
                "⬇️ Download metrics table (CSV)",
                data=metrics_csv_bytes,
                file_name=f"{program_name}_metrics_table.csv",
                mime="text/csv",
            )

            # Optional: save PNGs to disk and offer downloads for them
            if save_pngs:
                outdir = Path("./paig_results")
                outdir.mkdir(parents=True, exist_ok=True)
                paig.save_series_plots(outdir, program_name, t_years, df_sorted, sol)
                paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
                st.success(f"Saved plots on server: {outdir.resolve()}")

                # Offer downloads for saved PNGs
                for f in sorted(outdir.glob(f"{program_name}*.png")):
                    st.download_button(f"⬇️ Download {f.name}", f.read_bytes(), f.name, "image/png")

        except Exception as e:
            st.error(f"Fit failed: {e}")
