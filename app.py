# app.py
# PAIG interactive app (Streamlit)
# - Upload CSV
# - Select year range
# - Choose initial guesses for (rho, alpha, delta, nu, gamma)
# - Optional ratio constraint alpha = r * delta
# - Fit PAIG and plot Model vs Data (2x2)
# - Optional: save PNGs via helpers in paig_fit_any_csv.py

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

st.set_page_config(page_title="PAIG Model Explorer", layout="wide")

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
    n_steps: int = 6,
) -> bytes:
    """Render 2x2 plot: Model (line) vs Data (scatter). Y-axis from 0 to max with equal ticks."""
    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]
        # model and data
        ax.plot(t_years, sol.y[idx], 'r-', label="Model", linewidth=2)
        ax.scatter(df["year"].values, df[comp].values, s=18, label="Data")

        # uniform Y scale: 0..ymax with equal ticks
        y_max = float(max(np.nanmax(sol.y[idx]), np.nanmax(df[comp].values)))
        ax.set_ylim(0.0, y_max*1.1)

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
    Render the two 3D phase plots used in paig.save_series_3d_phase_plots
    into a single PNG suitable for displaying in Streamlit.
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

    fig = plt.figure(figsize=(12, 8))
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
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Persist upload to a temp .csv and reuse your paig loader."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name

# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

ratio_toggle = st.sidebar.checkbox("Enforce α = r·δ", value=False)
ratio_max   = st.sidebar.number_input("ratio_max", value=0.99, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
max_nfev    = st.sidebar.number_input("max_nfev", value=500, min_value=100, max_value=5000, step=50)
save_pngs   = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Initial guesses used by the optimizer")

rho0   = st.sidebar.slider("rho (inflow)",  min_value=1.0, max_value=500.0, value=200.0, step=1.0)
alpha0 = st.sidebar.slider("alpha",         min_value=0.0, max_value=1.0,   value=0.4,   step=0.01)
delta0 = st.sidebar.slider("delta",         min_value=0.0, max_value=1.0,   value=0.6,   step=0.01)
nu0    = st.sidebar.slider("nu",            min_value=0.0, max_value=1.0,   value=0.8,   step=0.01)
gamma0 = st.sidebar.slider("gamma",         min_value=0.0, max_value=1.0,   value=0.02,  step=0.001)

# ----------------- main -----------------
st.title("PAIG Model Explorer")

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
    st.dataframe(df_full, use_container_width=True) #st.dataframe(df_full.head(8), use_container_width=True)
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

            # Table
            cols = ["program","rho","alpha","delta","nu","gamma","alpha/delta",
                    "RMSE_P","RMSE_A","RMSE_I","RMSE_G",
                    "R2_P","R2_A","R2_I","R2_G","cost","nfev","success"]
            st.dataframe(pd.DataFrame([summary])[cols], use_container_width=True)

            # Plot
            png = render_fit_png(program_name, t_years, df_sorted, sol, dpi=150, n_steps=6)
            st.image(png, use_container_width=True)
            phase_png = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
            st.image(phase_png, use_container_width=True, caption="3D phase plots")


            # Optional: save PNGs using your original helpers
            if save_pngs:
                outdir = Path("./paig_results")
                paig.save_series_grid_plot(outdir, program_name, t_years, df_sorted, sol)
                paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
                st.success(f"Saved PNGs in {outdir.resolve()}")

        except Exception as e:
            st.error(f"Fit failed: {e}")
