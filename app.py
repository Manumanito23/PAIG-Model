# app.py
# PAIG interactive app (no Jupyter required)
# - Upload CSV
# - Select year range
# - Fit PAIG (optional alpha = r * delta)
# - What-if overlay with sliders (solid = Fit, dashed = What-if)
# - Store Fit A / Fit B and compare parameters & plots
#
# Requires paig_fit_any_csv.py in the same folder.

import io
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import paig_fit_any_csv as paig  # your module

st.set_page_config(page_title="PAIG Model Explorer", layout="wide")

# ------------- Helpers -------------
def slice_df_by_year(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    s = df[(df["year"] >= y0) & (df["year"] <= y1)].copy().reset_index(drop=True)
    if len(s) < 3:
        raise ValueError(f"Selected range {y0}-{y1} has too few rows ({len(s)}). Please pick ≥ 3.")
    return s

def render_overlay_png(
    name: str,
    t_years: np.ndarray,
    df: pd.DataFrame,
    pars_fit: np.ndarray,
    pars_whatif: np.ndarray,
    n_plot: int = 200,
    dpi: int = 140,
) -> bytes:
    """Render a 2x2 figure as PNG bytes: solid=Fit, dashed=What-if, scatter=Data."""
    years_fine = np.linspace(t_years[0], t_years[-1], n_plot)
    t_fine = years_fine - years_fine[0]
    y0 = df[["P", "A", "I", "G"]].T.values[:, 0]

    try:
        sol_fit_fine = paig.integrate_model(t_fine, y0, np.asarray(pars_fit, float))
        sol_wi_fine  = paig.integrate_model(t_fine, y0, np.asarray(pars_whatif, float))
        if not getattr(sol_fit_fine, "success", True) or not getattr(sol_wi_fine, "success", True):
            raise RuntimeError("ODE solver did not converge for current parameters.")
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 1.2))
        ax.axis("off")
        ax.text(0, 0.5, f"Render error: {e}", va="center", fontsize=11)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    comps = [
        ("P", 0, "Passive students (P)"),
        ("A", 1, "Active students (A)"),
        ("I", 2, "Inactive students (I, cumulative)"),
        ("G", 3, "Graduated students (G, cumulative)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, (comp, idx, title) in enumerate(comps):
        ax = axes[i // 2, i % 2]
        ax.plot(years_fine, sol_fit_fine.y[idx], label="Fit (model)", linewidth=2)
        ax.plot(years_fine, sol_wi_fine.y[idx], linestyle="--", label="What-if", linewidth=2)
        ax.scatter(df["year"].values, df[comp].values, s=18, label="Data")
        ax.set_title(title); ax.set_xlabel("Year"); ax.set_ylabel("Students")
        ax.grid(True); ax.legend()

    fig.suptitle(f"{name} — Fit vs What‑if (range: {int(t_years[0])}-{int(t_years[-1])})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def load_program_csv_from_upload(uploaded_file) -> Tuple[pd.DataFrame, str]:
    """Persist upload to a temp .csv and reuse your paig loader."""
    suffix = ".csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)
    df = paig.load_program_csv(temp_path)
    name = uploaded_file.name.rsplit(".", 1)[0].replace(" ", "_")
    return df, name

# ------------- Sidebar -------------
st.sidebar.title("PAIG Controls")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

ratio_toggle = st.sidebar.checkbox("Enforce α = r·δ", value=False)
ratio_max   = st.sidebar.number_input("ratio_max", value=0.99, min_value=0.01, max_value=1.0, step=0.01, format="%.2f")
max_nfev    = st.sidebar.number_input("max_nfev", value=500, min_value=100, max_value=5000, step=50)
save_pngs   = st.sidebar.checkbox("Save PNGs to ./paig_results", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("What‑if sliders (activate after Fit)")

# We’ll fill these after a fit:
if "whatif" not in st.session_state:
    st.session_state.whatif = dict(rho=200.0, alpha=0.4, delta=0.6, nu=0.8, gamma=0.02)

# ------------- Main -------------
st.title("PAIG Model Explorer")

if not uploaded:
    st.info("Upload a CSV to begin. The app expects columns similar to those used by your PAIG loader (Year, Passive, Active, I cumulative, G cumulative).")
    st.stop()

# Load CSV
try:
    df_full, program_name = load_program_csv_from_upload(uploaded)
except Exception as e:
    st.error(f"Failed to parse CSV: {e}")
    st.stop()

col_info, col_fit = st.columns([1, 2])

with col_info:
    st.subheader("Preview")
    st.dataframe(df_full.head(6), use_container_width=True)

    y_min, y_max = int(df_full["year"].min()), int(df_full["year"].max())
    yr0, yr1 = st.slider("Year range", min_value=y_min, max_value=y_max, value=(y_min, y_max), step=1)

with col_fit:
    st.subheader("Fit")
    do_fit = st.button("Run Fit on selected range", type="primary")

    if do_fit:
        try:
            df_slice = slice_df_by_year(df_full, yr0, yr1)
            summary, t_years, df_sorted, sol = paig.fit_program(
                df_slice,
                program_name,
                ratio_max=(float(ratio_max) if ratio_toggle else None),
                max_nfev=int(max_nfev),
            )
            pars_fit = np.array([summary["rho"], summary["alpha"], summary["delta"], summary["nu"], summary["gamma"]], float)

            st.session_state.last_fit = dict(
                summary=summary, t_years=t_years, df=df_sorted, pars_fit=pars_fit
            )

            # Initialize what‑if sliders to the fitted parameters
            st.session_state.whatif.update(
                rho=max(pars_fit[0], 1e-6),
                alpha=float(np.clip(pars_fit[1], 1e-3, 5.0)),
                delta=float(np.clip(pars_fit[2], 1e-3, 5.0)),
                nu=float(np.clip(pars_fit[3], 1e-3, 5.0)),
                gamma=float(np.clip(pars_fit[4], 1e-3, 1.0)),
            )

            # Summary table
            cols = ["program","rho","alpha","delta","nu","gamma","alpha/delta",
                    "RMSE_P","RMSE_A","RMSE_I","RMSE_G","R2_P","R2_A","R2_I","R2_G","cost","nfev","success"]
            st.dataframe(pd.DataFrame([summary])[cols], use_container_width=True)

            # Optional: save PNGs using your original helpers
            if save_pngs:
                outdir = Path("./paig_results")
                paig.save_series_grid_plot(outdir, program_name, t_years, df_sorted, sol)
                paig.save_series_3d_phase_plots(outdir, program_name, df_sorted, sol)
                st.success(f"Saved PNGs in {outdir.resolve()}")

        except Exception as e:
            st.error(f"Fit failed: {e}")

# If we have a fit, show What‑if overlay & A/B
if "last_fit" in st.session_state:
    lf = st.session_state.last_fit

    st.subheader("Overlay: Fit vs What‑if")

    # What‑if sliders (use columns to keep UI compact)
    c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1, 1, 1, 1, 1.1])
    with c1:
        rho = st.number_input("rho", value=float(st.session_state.whatif["rho"]), min_value=1e-6, max_value=1e6, step=1.0, format="%.6f")
    with c2:
        alpha = st.number_input("alpha", value=float(st.session_state.whatif["alpha"]), min_value=1e-3, max_value=5.0, step=0.01, format="%.3f")
    with c3:
        delta = st.number_input("delta", value=float(st.session_state.whatif["delta"]), min_value=1e-3, max_value=5.0, step=0.01, format="%.3f")
    with c4:
        nu = st.number_input("nu", value=float(st.session_state.whatif["nu"]), min_value=1e-3, max_value=5.0, step=0.01, format="%.3f")
    with c5:
        gamma = st.number_input("gamma", value=float(st.session_state.whatif["gamma"]), min_value=1e-3, max_value=1.0, step=0.001, format="%.3f")
    with c6:
        if st.button("Reset to Fit"):
            st.session_state.whatif.update(
                rho=float(lf["pars_fit"][0]),
                alpha=float(lf["pars_fit"][1]),
                delta=float(lf["pars_fit"][2]),
                nu=float(lf["pars_fit"][3]),
                gamma=float(lf["pars_fit"][4]),
            )
            rho, alpha, delta, nu, gamma = [float(v) for v in lf["pars_fit"]]

    st.session_state.whatif.update(rho=rho, alpha=alpha, delta=delta, nu=nu, gamma=gamma)
    pars_wi = np.array([rho, alpha, delta, nu, gamma], float)

    png = render_overlay_png(lf["summary"]["program"], lf["t_years"], lf["df"], lf["pars_fit"], pars_wi)
    st.image(png, use_container_width=True)

    st.markdown("---")
    st.subheader("Store & Compare (A vs B)")
    cc1, cc2, cc3 = st.columns([1, 1, 2])
    with cc1:
        if st.button("Store as Fit A"):
            st.session_state.fitA = lf.copy()
            st.session_state.fitA_png = render_overlay_png(lf["summary"]["program"], lf["t_years"], lf["df"], lf["pars_fit"], lf["pars_fit"])
            st.success("Stored Fit A.")
    with cc2:
        if st.button("Store as Fit B"):
            st.session_state.fitB = lf.copy()
            st.session_state.fitB_png = render_overlay_png(lf["summary"]["program"], lf["t_years"], lf["df"], lf["pars_fit"], lf["pars_fit"])
            st.success("Stored Fit B.")
    with cc3:
        if st.button("Compare A vs B"):
            if "fitA" not in st.session_state or "fitB" not in st.session_state:
                st.warning("Store two fits first.")
            else:
                A, B = st.session_state.fitA["summary"], st.session_state.fitB["summary"]
                common = ["rho","alpha","delta","nu","gamma","alpha/delta","P*","A*",
                          "RMSE_P","RMSE_A","RMSE_I","RMSE_G","R2_P","R2_A","R2_I","R2_G","cost","nfev"]
                cmp_df = pd.concat([pd.DataFrame([A], index=["A"])[common],
                                    pd.DataFrame([B], index=["B"])[common]])
                st.dataframe(cmp_df, use_container_width=True)
                st.image(st.session_state.fitA_png, caption="Fit A", use_container_width=True)
                st.image(st.session_state.fitB_png, caption="Fit B", use_container_width=True)
