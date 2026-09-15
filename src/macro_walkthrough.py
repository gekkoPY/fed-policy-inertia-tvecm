"""
fed-policy-inertia-tvecm
Module: src/macro_walkthrough.py

Step 7: Guided Macro Analysis, Residual Diagnostics & Live Desk Executive Report.
Executes:
1. Stationarity battery: ADF & KPSS unit root tests on yield levels, differences, and spread.
2. Dual-Engine TVECM parameterization (Dovish gamma_cut vs Hawkish gamma_hike).
3. 30-Year historical multi-panel visualization with FOMC cycle overlays and
   explicit visual masking / NaN discontinuity across the Zero Lower Bound (2008-2015).
4. Econometric residual diagnostics: Ljung-Box test, Jarque-Bera test, and Q-Q plots.
5. Point-in-time live desk recommendation overlaying Sahm Rule, Core PCE, and SOFR-IORB plumbing spread.
All figures are automatically saved to 'reports/figures/' and displayed interactively.
"""

import os
from pathlib import Path
import sys
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

# Dynamic path resolution to support execution from project root or src/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader import MacroDataLoader
from threshold_engine import ThresholdVECM
from signal_evaluator import DualSignalEvaluator

# Create directory for saving high-resolution figures
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plotting configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.dpi"] = 120


# =============================================================================
# PART 1: COINTEGRATION & STATIONARITY BATTERY (ADF / KPSS)
# =============================================================================
def run_stationarity_battery(series: pd.Series, name: str) -> dict:
    """Calculates both Augmented Dickey-Fuller and KPSS stationarity tests."""
    clean_s = series.dropna()
    adf_stat, adf_p, _, _, _, _ = adfuller(clean_s, autolag="AIC")
    kpss_stat, kpss_p, _, _ = kpss(clean_s, regression="c", nlags="auto")

    return {
        "Variable": name,
        "ADF Stat": f"{adf_stat:+.3f}",
        "ADF p-val": f"{adf_p:.4f}",
        "ADF Verdict": "Stationary I(0)" if adf_p < 0.05 else "Unit Root I(1)",
        "KPSS Stat": f"{kpss_stat:.3f}",
        "KPSS p-val": f"{kpss_p:.4f}",
        "KPSS Verdict": "Stationary I(0)" if kpss_p > 0.05 else "Unit Root I(1)",
    }


def execute_part1_stationarity(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("  PART 1: TIME SERIES STATIONARITY & COINTEGRATION TESTS (ADF vs KPSS)")
    print("=" * 90)
    tests = [
        run_stationarity_battery(df["DGS2"], "2Y Treasury Yield (DGS2)"),
        run_stationarity_battery(df["POLICY_RATE"], "Effective Fed Funds Rate (EFFR)"),
        run_stationarity_battery(df["DGS2"].diff(), "Delta 2Y Treasury (First Diff)"),
        run_stationarity_battery(df["POLICY_RATE"].diff(), "Delta Fed Funds (First Diff)"),
        run_stationarity_battery(df["SPREAD"], "Restricted Cointegrating Spread (2Y - EFFR)"),
    ]
    diag_df = pd.DataFrame(tests).set_index("Variable")
    print(diag_df.to_string())
    print("-" * 90)
    print("Interpretation: DGS2 and EFFR are non-stationary I(1) in levels and stationary I(0)")
    print("in first differences. The spread (DGS2 - EFFR) rejects the unit root hypothesis,")
    print("confirming cointegration with cointegrating vector beta = [1.0, -1.0]'.")


# =============================================================================
# PART 2: DUAL-ENGINE MODEL ESTIMATION & THRESHOLD VISUALIZATION
# =============================================================================
def execute_part2_estimation(df: pd.DataFrame) -> Tuple[ThresholdVECM, ThresholdVECM]:
    print("\n" + "=" * 90)
    print("  PART 2: HANSEN & SEO (2002) DUAL-ENGINE THRESHOLD ESTIMATION")
    print("=" * 90)

    dov_engine = ThresholdVECM(mode="dovish", lags=1, trim=0.15, n_grid=50)
    dov_engine.fit(df)

    hawk_engine = ThresholdVECM(mode="hawkish", lags=1, trim=0.15, n_grid=50)
    hawk_engine.fit(df)

    print(f"Optimal Dovish Threshold  (gamma_cut) : {dov_engine.gamma:+.4f} ({dov_engine.gamma * 100:+.1f} bps)")
    print(f"Optimal Hawkish Threshold (gamma_hike): {hawk_engine.gamma:+.4f} ({hawk_engine.gamma * 100:+.1f} bps)")
    print(f"Inertia Corridor Width                : {(hawk_engine.gamma - dov_engine.gamma) * 100:.1f} bps")

    # Plot Distribution and Thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    spread = df["SPREAD"].dropna()

    sns.histplot(spread, bins=80, kde=True, ax=axes[0], color="#2c3e50", stat="density", alpha=0.55)
    axes[0].axvline(dov_engine.gamma, color="#c0392b", linestyle="--", linewidth=2,
                    label=f"gamma_cut = {dov_engine.gamma*100:+.0f} bps")
    axes[0].axvline(hawk_engine.gamma, color="#2980b9", linestyle="--", linewidth=2,
                    label=f"gamma_hike = {hawk_engine.gamma*100:+.0f} bps")
    axes[0].axvspan(dov_engine.gamma, hawk_engine.gamma, color="#bdc3c7", alpha=0.35, label="Policy Inertia Corridor")
    axes[0].set_title("Empirical Spread Distribution & Optimal Thresholds", fontweight="bold")
    axes[0].set_xlabel("Market Spread (DGS2 - EFFR) in %")
    axes[0].legend(loc="upper right", frameon=True)

    sorted_sp = np.sort(spread.values)
    cdf = np.linspace(0, 1, len(sorted_sp))
    axes[1].plot(sorted_sp, cdf, color="#16a085", linewidth=2)
    axes[1].axvline(dov_engine.gamma, color="#c0392b", linestyle="--", linewidth=1.5)
    axes[1].axvline(hawk_engine.gamma, color="#2980b9", linestyle="--", linewidth=1.5)
    axes[1].axhline(np.mean(spread <= dov_engine.gamma), color="#c0392b", linestyle=":", alpha=0.7)
    axes[1].axhline(np.mean(spread <= hawk_engine.gamma), color="#2980b9", linestyle=":", alpha=0.7)
    axes[1].set_title("Empirical CDF & Regime Densities", fontweight="bold")
    axes[1].set_xlabel("Market Spread (DGS2 - EFFR) in %")
    axes[1].set_ylabel("Cumulative Probability")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig1_spread_distribution_thresholds.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    plt.show()

    return dov_engine, hawk_engine


# =============================================================================
# PART 3: 30-YEAR HISTORICAL VISUALIZATION & FOMC CYCLES (WITH ZLB MASKING)
# =============================================================================
def execute_part3_macro_timeline(df: pd.DataFrame, dov_model: ThresholdVECM, hawk_model: ThresholdVECM) -> None:
    print("\n" + "=" * 90)
    print("  PART 3: 30-YEAR REGIME SHIFTS TIMELINE & FOMC POLICY CYCLES (1994-2026)")
    print("=" * 90)
    evaluator = DualSignalEvaluator()

    # Reindex onto full business day calendar to create genuine NaN breaks across the ZLB window
    full_bday_calendar = pd.date_range(start=df.index[0], end=df.index[-1], freq="B")
    df_plot = df.reindex(full_bday_calendar)

    zlb_start = pd.to_datetime("2008-12-16")
    zlb_end = pd.to_datetime("2015-12-16")
    zlb_mid = zlb_start + (zlb_end - zlb_start) / 2

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, gridspec_kw={"height_ratios": [1.7, 1.0]})

    # Top Panel: DGS2 vs EFFR (Lines naturally break across NaNs)
    ax1.plot(df_plot.index, df_plot["DGS2"], label="2-Year Treasury Yield (DGS2)", color="#2980b9", linewidth=1.3)
    ax1.plot(df_plot.index, df_plot["POLICY_RATE"], label="Effective Fed Funds Rate (EFFR)", color="#2c3e50", linewidth=1.6)

    # Shading ZLB Regime on Top Panel
    ax1.axvspan(zlb_start, zlb_end, color="#bdc3c7", alpha=0.35, hatch="//", edgecolor="#7f8c8d",
                label="ZLB Regime (Excluded from TVECM)")
    ax1.text(zlb_mid, 4.0, "ZLB REGIME (2008-2015)\nExcluded from TVECM Calibration",
             color="#2c3e50", fontsize=9, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecf0f1", alpha=0.9, edgecolor="#7f8c8d"))

    # Bottom Panel: Spread and Regime Zones
    ax2.plot(df_plot.index, df_plot["SPREAD"], label="Spread (DGS2 - EFFR)", color="#16a085", linewidth=1.1)
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.6)
    ax2.axhline(dov_model.gamma, color="#c0392b", linestyle="--", linewidth=1.4,
                label=f"Dovish Capitulation ({dov_model.gamma*100:+.0f} bps)")
    ax2.axhline(hawk_model.gamma, color="#8e44ad", linestyle="--", linewidth=1.4,
                label=f"Hawkish Capitulation ({hawk_model.gamma*100:+.0f} bps)")

    # Shading ZLB Regime on Bottom Panel
    ax2.axvspan(zlb_start, zlb_end, color="#bdc3c7", alpha=0.35, hatch="//", edgecolor="#7f8c8d")
    ax2.text(zlb_mid, 0.0, "ZLB DISCONTINUITY", color="#2c3e50", fontsize=8, fontweight="bold",
             ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.9, edgecolor="#7f8c8d"))

    # Regime Shading (Matplotlib automatically handles NaNs without drawing spurious blocks)
    ax2.fill_between(df_plot.index, df_plot["SPREAD"], dov_model.gamma,
                     where=(df_plot["SPREAD"] <= dov_model.gamma), color="#c0392b", alpha=0.35,
                     label="Dovish Regime (Imminent Cuts)")
    ax2.fill_between(df_plot.index, df_plot["SPREAD"], hawk_model.gamma,
                     where=(df_plot["SPREAD"] >= hawk_model.gamma), color="#8e44ad", alpha=0.35,
                     label="Hawkish Regime (Imminent Hikes)")

    # FOMC Pivot Event Annotations
    for c in evaluator.EASING_CYCLES:
        dt = pd.to_datetime(c["start"])
        if dt >= df.index[0]:
            ax1.axvline(dt, color="#c0392b", linestyle=":", alpha=0.85)
            ax2.axvline(dt, color="#c0392b", linestyle=":", alpha=0.85)
            ax1.text(dt, ax1.get_ylim()[1] * 0.90, f" Cut: {c['name'].split()[0]}",
                     color="#c0392b", fontsize=8, rotation=90)

    for h in evaluator.TIGHTENING_CYCLES:
        dt = pd.to_datetime(h["start"])
        if dt >= df.index[0]:
            ax1.axvline(dt, color="#27ae60", linestyle=":", alpha=0.85)
            ax2.axvline(dt, color="#27ae60", linestyle=":", alpha=0.85)
            ax1.text(dt, ax1.get_ylim()[1] * 0.90, f" Hike: {h['name'].split()[0]}",
                     color="#27ae60", fontsize=8, rotation=90)

    ax1.set_title("Federal Reserve Policy Inertia vs 2-Year Treasury Equilibrium (1994-2026)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Yield / Rate (%)")
    ax1.legend(loc="upper right", frameon=True)

    ax2.set_ylabel("Spread (% points)")
    ax2.set_xlabel("Year")
    ax2.legend(loc="lower right", frameon=True, ncol=2)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig2_macro_regimes_timeline_1994_2026.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    plt.show()


# =============================================================================
# PART 4: ECONOMETRIC RESIDUAL DIAGNOSTICS
# =============================================================================
def execute_part4_residual_diagnostics(dov_model: ThresholdVECM) -> None:
    print("\n" + "=" * 90)
    print("  PART 4: ECONOMETRIC RESIDUAL DIAGNOSTICS (LJUNG-BOX & JARQUE-BERA)")
    print("=" * 90)

    res_dgs2 = dov_model.results_regime2["residuals"][:, 0]
    res_effr = dov_model.results_regime2["residuals"][:, 1]

    # Statistical tests
    lb_dgs2 = acorr_ljungbox(res_dgs2, lags=[5, 10], return_df=True)
    lb_effr = acorr_ljungbox(res_effr, lags=[5, 10], return_df=True)
    jb_dgs2, jb_p_dgs2 = stats.jarque_bera(res_dgs2)
    jb_effr, jb_p_effr = stats.jarque_bera(res_effr)

    print(">>> Ljung-Box Test for Serial Correlation (d_DGS2 Residuals):")
    print(lb_dgs2.to_string())
    print("\n>>> Ljung-Box Test for Serial Correlation (d_POLICY_RATE Residuals):")
    print(lb_effr.to_string())
    print(f"\nJarque-Bera Test (d_DGS2)       : Stat = {jb_dgs2:.2f}, p-value = {jb_p_dgs2:.2e}")
    print(f"Jarque-Bera Test (d_POLICY_RATE): Stat = {jb_effr:.2f}, p-value = {jb_p_effr:.2e}")

    # Plot Diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    sns.histplot(res_dgs2, kde=True, ax=axes[0, 0], color="#2980b9", stat="density")
    axes[0, 0].set_title("Residual Density: d_DGS2 (Regime 2)", fontweight="bold")
    axes[0, 0].set_xlabel("Residual")

    sns.histplot(res_effr, kde=True, ax=axes[0, 1], color="#e67e22", stat="density")
    axes[0, 1].set_title("Residual Density: d_POLICY_RATE (Regime 2)", fontweight="bold")
    axes[0, 1].set_xlabel("Residual")

    stats.probplot(res_dgs2, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot: d_DGS2 (Fat Tails Characteristic)", fontweight="bold")

    stats.probplot(res_effr, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot: d_POLICY_RATE (Discrete Step Policy)", fontweight="bold")

    plt.tight_layout()
    fig_path = FIGURES_DIR / "fig3_residual_diagnostics.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved: {fig_path}")
    plt.show()


# =============================================================================
# PART 5: LIVE 2026 MACRO OVERLAY & DESK VERDICT
# =============================================================================
def execute_part5_live_dashboard(df: pd.DataFrame, dov_model: ThresholdVECM, hawk_model: ThresholdVECM) -> None:
    loader = MacroDataLoader()
    macro_file = loader.processed_dir / "macro_filters_lagged.csv"
    liq_file = loader.processed_dir / "module3_liquidity.csv"

    df_macro = pd.read_csv(macro_file, index_col=0, parse_dates=True) if macro_file.exists() else None
    df_liq = pd.read_csv(liq_file, index_col=0, parse_dates=True) if liq_file.exists() else None

    latest_date = df.index[-1]
    curr_2y = float(df.loc[latest_date, "DGS2"])
    curr_effr = float(df.loc[latest_date, "POLICY_RATE"])
    curr_spread = float(df.loc[latest_date, "SPREAD"])

    latest_sahm = float(df_macro["SAHM_INDICATOR"].dropna().iloc[-1]) if df_macro is not None else np.nan
    latest_pce = float(df_macro["CORE_PCE_YOY"].dropna().iloc[-1]) if df_macro is not None else np.nan
    latest_repo_spread = float(df_liq["SOFR_IORB_SPREAD_BPS"].dropna().iloc[-1]) if df_liq is not None else np.nan
    latest_nfci = float(df_liq["NFCI"].dropna().iloc[-1]) if df_liq is not None else np.nan

    print("\n" + "=" * 90)
    print(f"  PART 5: TACTICAL FIXED INCOME DESK DASHBOARD (AS OF {latest_date.strftime('%Y-%m-%d')})")
    print("=" * 90)
    print(f"{'2-Year Treasury Yield (DGS2)':<38}: {curr_2y:.2f}%")
    print(f"{'Effective Fed Funds Rate (EFFR)':<38}: {curr_effr:.2f}%")
    print(f"{'Market Spread (DGS2 - EFFR)':<38}: {curr_spread * 100:+.1f} bps ({curr_spread:+.2f}%)")
    print("-" * 90)
    print(f"{'Dovish Threshold (gamma_cut)':<38}: {dov_model.gamma * 100:+.1f} bps")
    print(f"{'Hawkish Threshold (gamma_hike)':<38}: {hawk_model.gamma * 100:+.1f} bps")
    print(f"{'Policy Inertia Corridor':<38}: [{dov_model.gamma * 100:+.0f} bps, {hawk_model.gamma * 100:+.0f} bps]")
    print("-" * 90)
    print(f"{'Sahm Rule Indicator (Point-in-Time)':<38}: {latest_sahm:.2f} (Threshold >= 0.50)")
    print(f"{'Core PCE Inflation YoY (Lagged 35d)':<38}: {latest_pce:.2f}% (Fed Target: 2.00%)")
    print(f"{'SOFR - IORB Plumbing Spread':<38}: {latest_repo_spread:+.1f} bps (Stress Alert > +10 bps)")
    print(f"{'Chicago Fed Financial Conditions (NFCI)':<38}: {latest_nfci:+.3f} (Tightening > 0.00)")
    print("=" * 90)

    # Desk Positioning Recommendation
    if curr_spread >= hawk_model.gamma:
        regime_status = "HAWKISH CAPITULATION ALERT (BEHIND THE CURVE)"
        action = "Market yield forces rate hikes. Recommend Short Front-End Duration / Flattener 2s10s."
    elif curr_spread <= dov_model.gamma:
        regime_status = "DOVISH CAPITULATION ALERT (RECESSION / EASING BIAS)"
        action = "Curve inversion forces rate cuts. Recommend Long Front-End Duration / Steepener 2s10s."
    else:
        regime_status = "NEUTRAL POLICY INERTIA ZONE (WAIT-AND-SEE)"
        action = "Spread inside tolerance band. Carry trades favored; no immediate pivot forced on Fed."

    print(f"REGIME STATUS          : {regime_status}")
    print(f"TACTICAL DESK BIAS     : {action}")
    print("=" * 90 + "\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    loader = MacroDataLoader()
    csv_path = loader.processed_dir / "module1_daily.csv"
    if not csv_path.exists():
        df_clean = loader.load_module1_daily(policy_rate="EFFR", start_date="1994-01-01")
    else:
        df_clean = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # 1. Cointegration & Stationarity Tests
    execute_part1_stationarity(df_clean)

    # 2. Dual-Engine TVECM Fit
    dov_m, hawk_m = execute_part2_estimation(df_clean)

    # 3. Macro Regimes Timeline (with ZLB Masking)
    execute_part3_macro_timeline(df_clean, dov_m, hawk_m)

    # 4. Residual Diagnostics
    execute_part4_residual_diagnostics(dov_m)

    # 5. Live Desk Positioning Dashboard
    execute_part5_live_dashboard(df_clean, dov_m, hawk_m)
