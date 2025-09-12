# app.py — PAIG interactive app (UNWEIGHTED fitting)
# Keeps your previous layout/behavior and adds:
#   • χ² p-value (Poisson variances) on the "Global (overall)" row
#   • An "Accept @ α" flag (you can change α in the sidebar)

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st

import paig_fit_any_csv as paig

st.set_page_config(page_title="PAIG Model Explorer (Unweighted)", layout="wide")

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
    shift_labels_by_minus1: bool = False,
) -> bytes:
    """Render 2x2 plot: Model (line) vs Data (scatter)."""
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x_model = t_years - (1 if shift_labels_by_minus1 else 0)
    x_data  = df["year"].values - (1 if shift_labels_by_minus1 else 0)

    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]
        ax.plot(x_model, sol.y[idx], 'r-', label="Model", linewidth=2)
        ax.scatter(x_data, df[comp].values, s=18, label="Data")
        y_max = float(max(np.nanmax(sol.y[idx]), np.nanmax(df[comp].values)))
        ax.set_ylim(0.0, y_max * 1.1)
        ax.set_title(title)
        ax.set_xlabel("Year" + ("  (display shifted −1)" if shift_labels_by_minus1 else ""))
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
    """Render 3D (A,P,I) and (A,P,G) plots."""
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
    ax1.set_xlabel("Active (A)")
    ax1.set_ylabel("Passive (P)")
    ax1.set_zlabel("Inactive (I, cumulative)")
    ax1.set_title("3D phase: A vs P vs I")
    ax1.legend(); ax1.grid(True)

    ax2.plot(Am, Pm, Gm, 'r-', label="Model")
    ax2.scatter(Ad, Pd, Gd, s=18, label="Data")
    ax2.set_xlabel("Active (A)")
    ax2.set_ylabel("Passive (P)")
    ax2.set_zlabel("Graduated (G, cumulative)")
    ax2.set_title("3D phase: A vs P vs G")
    ax2.legend(); ax2.grid(True)

    fig.suptitle(f"{name} — PAIG 3D phase projections", fontsize=14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    return buf.getvalue()

def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Persist upload to temp and reuse the PAIG loader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name

def make_metrics_table(summary: dict) -> pd.DataFrame:
    """
    Build the metrics table for display (per-series + global).
    Adds χ² p-value in the Global row (Poisson variances).
    """
    rows = []

    # Global row (overall)
    rows.append({
        " ": "Global (overall)",
        "Mean Absolute Error (MAE)": np.nan,  # not computed globally in 4D
        "Root Mean Square Error (RMSE)": summary.get("RMSE_global", np.nan),
        "Coefficient of Determination (R^2)": summary.get("R2_global", np.nan),
        "Adjusted R^2": np.nan,  # (optional; left blank for global 4D)
        "Chi-squared": summary.get("chi2_global", np.nan),
        "Reduced Chi-squared": summary.get("chi2_global", np.nan) / max(summary.get("chi2_dof", 1), 1),
        "p-value (χ², Poisson)": summary.get("chi2_p_value", np.nan),
    })

    # Per-series rows (same metrics you had; no p-values per-series)
    for comp in ["P", "A", "I", "G"]:
        rmse = summary.get(f"RMSE_{comp}", np.nan)
        r2   = summary.get(f"R2_{comp}", np.nan)
        rows.append({
            " ": f"{comp}",
            "Mean Absolute Error (MAE)": np.nan,  # you can add MAE easily if wanted
            "Root Mean Square Error (RMSE)": rmse,
            "Coefficient of Determination (R^2)": r2,
            "Adjusted R^2": np.nan,               # typical per-series adjR² needs n & p; omitted here
            "Chi-squared": np.nan,
            "Reduced Chi-squared": np.nan,
            "p-value (χ², Poisson)": np.nan,
        })

    df_table = pd.DataFrame(rows)
    return df_table


# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls (Unweighted)")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

save_pngs   = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)
shift_year  = st.sidebar.checkbox("Shift year labels by -1 (display only)", value=False)
auto_refit  = st.sidebar.checkbox("Re-fit automatically on change", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Initial guesses used by the optimizer (pure NLS, unweighted)")
rho0   = st.sidebar.slider("rho (inflow)",  min_value=1.0, max_value=500.0, value=200.0, step=1.0)
alpha0 = st.sidebar.slider("alpha",         min_value=0.0, max_value=1.0,   value=0.4,   step=0.01)
delta0 = st.sidebar.slider("delta",         min_value=0.0, max_value=1.0,   value=0.6,   step=0.01)
nu0    = st.sidebar.slider("nu",            min_value=0.0, max_value=1.0,   value=0.8,   step=0.01)
gamma0 = st.sidebar.slider("gamma",         min_value=0.0, max_value=1.0,   value=0.02,  step=0.001)

st.sidebar.markdown("---")
alpha_sig = st.sidebar.number_input("Significance α for χ² accept/reject", value=0.05, min_value=0.001, max_value=0.2, step=0.001, format="%.3f")

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

# Year slider at the top (so the auto y0 mode can watch it)
y_min, y_max = int(df_full["year"].min()), int(df_full["year"].max())
yr0, yr1 = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max), step=1)

# Initial-condition mode: auto-select but allow user override
auto_select_text = "Initial condition at start of fit"
col_a, col_b = st.columns([1,3])
with col_a:
    st.subheader(auto_select_text)

# Auto-choice: zeros if starting at earliest year; else estimated
default_mode = "zeros" if yr0 == y_min else "estimated"
mode_options = {"estimated (fit y₀)": "estimated", "zeros (P=A=I=G=0)": "zeros"}
mode_keys = list(mode_options.keys())
default_index = 0 if default_mode == "estimated" else 1

# Render the radio (disabled index is not needed; we simply set the default)
y0_choice_label = st.radio(
    " ",
    mode_keys,
    index=default_index,
    horizontal=True,
    label_visibility="collapsed",
)

y0_mode = mode_options[y0_choice_label]

# Preview table (narrow column)
st.subheader("Preview")
st.dataframe(df_full[(df_full["year"] >= yr0) & (df_full["year"] <= yr1)], use_container_width=False, height=300)

# ---- Fit button ----
st.subheader("Fit")
run = st.button("Run Fit with current initial guesses", type="primary") or auto_refit

if run:
    try:
        df_slice = slice_df_by_year(df_full, yr0, yr1)

        summary, t_years, df_sorted, sol = paig.fit_program(
            df_slice,
            program_name,
            y0_mode=y0_mode,
            max_nfev=500,
            init_guess=dict(rho=rho0, alpha=alpha0, delta=delta0, nu=nu0, gamma=gamma0),
        )

        # Add accept/reject at chosen α (display only)
        pval = float(summary.get("chi2_p_value", np.nan))
        accept = (pval >= float(alpha_sig)) if np.isfinite(pval) else False

        # Compact parameter table
        cols = [
            "program", "y0_mode", "rho", "alpha", "delta", "nu", "gamma", "alpha/delta",
            "R2_global", "RMSE_global",
            "chi2_global", "chi2_dof", "chi2_p_value",
        ]
        st.dataframe(pd.DataFrame([summary])[cols], use_container_width=True)

        st.markdown(
            f"**χ² test (Poisson variances):**  χ² = {summary['chi2_global']:.2f}  "
            f"with dof = {summary['chi2_dof']},  p = {summary['chi2_p_value']:.4g}.  "
            f"**Decision @ α = {alpha_sig:.3f}:** {'✅ Accept (adequate)' if accept else '❌ Reject (inadequate)'}."
        )

        # Metrics table (adds p-value in the Global row)
        st.subheader("Metrics")
        metrics_df = make_metrics_table(summary)
        st.dataframe(metrics_df, use_container_width=True)

        # Plots
        png = render_fit_png(program_name, t_years, df_sorted, sol, dpi=150, shift_labels_by_minus1=shift_year)
        st.image(png, use_container_width=True)
        phase_png = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
        st.image(phase_png, use_container_width=True, caption="3D phase plots")

        if save_pngs:
            outdir = Path("./paig_results")
            paig.save_series_grid_plot(outdir, program_name, t_years, df_sorted, sol)
            paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
            st.success(f"Saved on server: {outdir.resolve()}")
            for f in sorted(outdir.glob(f"{program_name}*.png")):
                st.download_button(f"⬇️ Download {f.name}", f.read_bytes(), f.name, "image/png")

    except Exception as e:
        st.error(f"Fit failed: {e}")
