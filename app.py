# app.py — PAIG Explorer (unweighted NLS) with loss selector and 3 y0 modes
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

        # robust: recorta a longitud común por si el solver devuelve 1 punto menos
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

# --- Residual plots -----------------------------------------------------------
def show_residual_plots(t_years, df_sorted, sol):
    """
    Draw 5 residual plots:
      - Four time series residuals: P, A, I, G (Data - Model), each centered at 0
      - One Euclidean residual norm ||r||_2 across P,A,I,G
    """
    # Data matrix (T x 4) and model predictions (T x 4)
    data = df_sorted[["P", "A", "I", "G"]].to_numpy()       # shape (T,4)
    pred = sol.y.T                                          # shape (T,4)
    resid = data - pred                                     # residuals r_t = y_t - ŷ_t

    names = ["P", "A", "I", "G"]

    # ---- 4 residual time-series (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        r = resid[:, i]
        ax.axhline(0.0, color="k", lw=1, alpha=0.5)
        ax.plot(t_years, r, marker="o", lw=1.5)
        ax.set_title(f"Residuals: {names[i]}")
        ax.set_ylabel("Data − Model")
        # symmetric y-limits around zero (helps visual diagnosis)
        m = float(np.nanmax(np.abs(r))) if r.size else 1.0
        ax.set_ylim(-1.05 * m, 1.05 * m)
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # ---- Euclidean residual norm
    euclid = np.sqrt((resid ** 2).sum(axis=1))              # ||r_t||_2
    fig2, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(t_years, euclid, marker="o", lw=1.5)
    ax.set_title("Euclidean residual norm across P, A, I, G")
    ax.set_xlabel("Year")
    ax.set_ylabel("‖residual‖₂")
    ax.set_ylim(bottom=0)                                   # norma es no-negativa
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2, clear_figure=True)

def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp.name)
    df = paig.load_program_csv(tmp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name


# ----------------- sidebar -----------------
st.sidebar.title("PAIG Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

save_pngs      = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)
label_shift_on = st.sidebar.checkbox("Shift year labels by 1 (display only)", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Initial guesses for the optimizer (pure, unweighted NLS)")
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

# Two columns so the Preview table is not full width
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("Preview")
    st.dataframe(df_full, height=340, use_container_width=False)  # narrow table
    yr0, yr1 = st.slider("Year range", min_value=y_min, max_value=y_max,
                         value=(y_min, y_max), step=1, label_visibility="visible")

with col_right:
    st.markdown("#### Initial condition at start of fit (y₀)")

    # 3 options: estimated / zeros / data
    # Default practical value: if range starts from first year, "zeros"; if not, "estimated".
    default_mode = "zeros" if yr0 == y_min else "estimated"
    labels = {
        "estimated": "estimated (fit y₀)",
        "zeros":     "zeros (P=A=I=G=0 at previous year)",
        "data":      "data (use first-year observation)"
    }
    # selectiont between modes
    if "y0_mode" not in st.session_state:
        st.session_state.y0_mode = default_mode
    # radio visible
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

    run = st.button("Run Fit with current initial guesses", type="primary")

    if run:
        try:
            df_slice = slice_df_by_year(df_full, yr0, yr1)

            # Adjust (unweighted); p depends of y0_mode (5 or 9)
            summary, t_years, df_sorted, sol = paig.fit_program(
                df_slice,
                program_name,
                init_guess=dict(rho=rho0, alpha=alpha0, delta=delta0, nu=nu0, gamma=gamma0),
                y0_mode=y0_mode,
                loss=loss_choice,
            )

            # To graph: if y0_mode == "zeros", we add the previous year
            if summary["y0_mode"] == "zeros":
                t_plot_years = np.r_[t_years[0] - 1, t_years]
                t_aug = t_plot_years - t_plot_years[0]
                y0_z = np.zeros(4, dtype=float)
                pars_hat = np.array(
                    [summary["rho"], summary["alpha"], summary["delta"], summary["nu"], summary["gamma"]],
                    dtype=float
                )
                sol_plot = paig.integrate_model(t_aug, y0_z, pars_hat)
            else:
                t_plot_years = t_years
                sol_plot = sol

            # Summary
            st.dataframe(
                pd.DataFrame([summary])[["program","y0_mode","loss","rho","alpha","delta","nu","gamma","alpha/delta"]],
                use_container_width=True
            )

            # Metrics (p = 9 if estimated, p = 5 if zeros or data)
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

            # Graph 2x2
            png = render_fit_png(
                program_name,
                t_plot_years if not label_shift_on else (t_plot_years - 1),
                df_sorted,
                sol_plot,
                dpi=150,
                label_shift_years=(1 if label_shift_on else 0),
            )
            st.image(png, use_container_width=True)

            # Graph 3D (2 panels)
            png3d = render_phase3d_png(program_name, df_sorted, sol, dpi=150)
            st.image(png3d, use_container_width=True)

            # Graph residual plots
            with st.expander("Residual analysis (time series)", expanded=False):
                show_residual_plots(t_years, df_sorted, sol)
                st.caption(
                    "Residuals are defined as Data − Model. "
                    "Panels show per-series residuals (P, A, I, G) centered at 0, "
                    "and the global Euclidean residual norm across the four series."
                )

            # Saving the pngs
            if save_pngs:
                outdir = Path(".") / "paig_results"
                paig.save_series_grid_plot(outdir, program_name, t_plot_years, df_sorted, sol_plot)
                paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
                st.success(f"Saved PNGs in: {outdir.resolve()}")

        except Exception as e:
            st.error(f"Fit failed: {e}")
