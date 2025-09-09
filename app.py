# app.py (UNWEIGHTED) — classic layout restored + auto y0 + metrics table
# - Left column: Preview table + Year range slider (narrow width)
# - Right column: y0 mode (auto or manual), parameters summary, metrics table, plots
# - Pure unweighted nonlinear least squares (delegated to paig_fit_any_csv.py)

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
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
    dpi: int = 150,
    label_shift_years: int = 0,
) -> bytes:
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
        # display-only shift for x tick labels
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

def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name

# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls (Unweighted)")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

save_pngs      = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)
label_shift_on = st.sidebar.checkbox("Shift year labels by -1 (display only)", value=False)
auto_run       = st.sidebar.checkbox("Re-fit automatically on change", value=True)

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
    st.info("Upload a CSV to begin. Expected columns similar to your loader (Year, Passive, Active, I cumulative, G cumulative).")
    st.stop()

# Load
try:
    df_full, program_name = load_program_csv_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

y_min, y_max = int(df_full["year"].min()), int(df_full["year"].max())

# Use two columns so the Preview table is not full width
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Preview")
    st.dataframe(df_full, height=340, use_container_width=False)  # narrow table

    # Year range slider stays in the left column under the table
    yr0, yr1 = st.slider("Year range", min_value=y_min, max_value=y_max,
                         value=(y_min, y_max), step=1, label_visibility="visible")

with col_right:
    # y0 mode (auto or manual). We keep a small state variable so radio updates when year range changes
    if "y0_mode" not in st.session_state:
        st.session_state.y0_mode = "estimated"

    st.markdown("#### Initial condition at start of fit")
    auto_y0 = st.checkbox("Auto-select (zeros if range starts at earliest year, else estimated)",
                          value=True, help="Checked: selection follows the year range.")

    # compute the auto choice
    auto_choice = "zeros" if yr0 == y_min else "estimated"
    if auto_y0:
        st.session_state.y0_mode = auto_choice

    # radio reflects current state; user can override only when auto_y0 is off
    disabled = auto_y0
    st.radio(
        " ",
        options=[("estimated (fit y₀)", "estimated"), ("zeros (P=A=I=G=0)", "zeros")],
        format_func=lambda x: x[0],
        horizontal=True,
        key="y0_mode",
        label_visibility="collapsed",
        disabled=disabled,
    )
    y0_mode = st.session_state.y0_mode  # "estimated" or "zeros"

    # run condition: auto or button
    should_run = auto_run or st.button("Run Fit with current initial guesses", type="primary")

    if should_run:
        try:
            df_slice = slice_df_by_year(df_full, yr0, yr1)

            summary, t_years, df_sorted, sol = paig.fit_program(
                df_slice,
                program_name,
                init_guess=dict(rho=rho0, alpha=alpha0, delta=delta0, nu=nu0, gamma=gamma0),
                y0_mode=y0_mode,  # estimated vs zeros
            )

            # Parameter summary
            st.dataframe(
                pd.DataFrame([summary])[["program","y0_mode","rho","alpha","delta","nu","gamma","alpha/delta"]],
                use_container_width=True
            )

            # Metrics table (unweighted). p depends on y0_mode (estimated fits 9 params; zeros fits 5).
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
            st.download_button(
                "⬇️ Download metrics CSV",
                data=metrics_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{program_name}_metrics.csv",
                mime="text/csv",
            )

            # Plots (2x2 + 3D)
            shift = -1 if label_shift_on else 0
            png = render_fit_png(program_name, t_years, df_sorted, sol, dpi=150, label_shift_years=shift)
            st.image(png, use_container_width=True)
            phase_png = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
            st.image(phase_png, use_container_width=True, caption="3D phase plots")

            # Optional: save PNGs
            if save_pngs:
                outdir = Path("./paig_results")
                outdir.mkdir(parents=True, exist_ok=True)
                paig.save_series_grid_plot(outdir, program_name, t_years, df_sorted, sol)
                paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
                st.success(f"Saved on server: {outdir.resolve()}")

        except Exception as e:
            st.error(f"Fit failed: {e}")
