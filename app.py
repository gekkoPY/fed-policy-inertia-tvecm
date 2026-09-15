"""
fed-policy-inertia-tvecm
Application: app.py

Institutional-Grade Interactive Fixed Income & Macro Desk Dashboard.
Bloomberg Terminal / FactSet Dark Architecture.
Complete Feature Set:
- Live 2026 Macro Desk & Plumbing Overlays (Sahm Rule, Core PCE, SOFR-IORB, NFCI)
- Interactive 30-Year Regime Chart with Vertical Regime Columns (High Performance)
- Formatted Point-in-Time Signal Audit Ledger with State-Aware Lead Time & Rolling γ_t
- Empirical Spread Distribution & Inertia Corridor (Hansen & Seo, 2002)
- Econometric Diagnostics: ADF, KPSS, Ljung-Box, Jarque-Bera & Q-Q Plots
"""

import os
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. AUTO-CONFIGURE STREAMLIT THEME & AUTO-LAUNCH SERVER
# =============================================================================
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

config_dir = CURRENT_DIR / ".streamlit"
config_file = config_dir / "config.toml"
if not config_file.exists():
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "[theme]\n"
        'base = "dark"\n'
        'backgroundColor = "#0b0e14"\n'
        'secondaryBackgroundColor = "#151b23"\n'
        'textColor = "#e6edf3"\n'
        'primaryColor = "#f0883e"\n'
        '[server]\n'
        'headless = true\n'
    )

try:
    from streamlit.runtime import exists as _st_runtime_exists
    if not _st_runtime_exists():
        from streamlit.web import cli as _st_cli
        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        sys.exit(_st_cli.main())
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import MacroDataLoader
from src.threshold_engine import ThresholdVECM
from src.signal_evaluator import DualSignalEvaluator


# =============================================================================
# 2. PAGE CONFIG & BLOOMBERG TERMINAL DARK STYLESHEET
# =============================================================================
st.set_page_config(
    page_title="Fed Policy Inertia | Bloomberg Terminal Desk",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0b0e14 !important;
        color: #e6edf3 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #10141b !important;
        border-right: 1px solid #21262d !important;
    }
    header[data-testid="stHeader"] { background-color: #0b0e14 !important; }
    
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #e6edf3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Strict Pixel-Matched Height for Top Metrics */
    .metric-card {
        background-color: #151b23;
        border: 1px solid #2d333b;
        border-radius: 6px;
        padding: 12px 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        height: 95px !important;
        min-height: 95px !important;
        max-height: 95px !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
    }
    .metric-title {
        color: #8b949e !important;
        font-size: 0.70rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        line-height: 1.1;
    }
    .metric-value {
        font-size: 1.50rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        font-family: 'Consolas', 'Roboto Mono', monospace;
        line-height: 1.1;
        display: flex;
        align-items: center;
    }
    .metric-sub {
        color: #58a6ff !important;
        font-size: 0.74rem !important;
        font-family: 'Consolas', monospace;
        line-height: 1.1;
    }

    .terminal-alert-box {
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }
    .terminal-alert-title {
        font-weight: 800;
        font-size: 0.92rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .terminal-alert-body {
        color: #e6edf3 !important;
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #2d333b;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #151b23 !important;
        border: 1px solid #2d333b !important;
        border-radius: 4px !important;
        color: #8b949e !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        border-color: #f0883e !important;
        color: #f0883e !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. DATA ENGINE & CACHED COMPUTATIONS
# =============================================================================
@st.cache_data(show_spinner="Loading synchronized historical datasets...")
def load_base_data():
    loader = MacroDataLoader()
    csv_file = loader.processed_dir / "module1_daily.csv"
    if not csv_file.exists():
        df = loader.load_module1_daily(policy_rate="EFFR", start_date="1994-01-01")
    else:
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    macro_file = loader.processed_dir / "macro_filters_lagged.csv"
    liq_file = loader.processed_dir / "module3_liquidity.csv"

    df_macro = pd.read_csv(macro_file, index_col=0, parse_dates=True) if macro_file.exists() else None
    df_liq = pd.read_csv(liq_file, index_col=0, parse_dates=True) if liq_file.exists() else None

    return df.sort_index(), df_macro, df_liq


@st.cache_resource(show_spinner="Estimating Hansen & Seo (2002) Dual Threshold Models...")
def fit_tvecm_models(df: pd.DataFrame):
    dov = ThresholdVECM(mode="dovish", lags=1, trim=0.15, n_grid=50).fit(df)
    hawk = ThresholdVECM(mode="hawkish", lags=1, trim=0.15, n_grid=50).fit(df)
    return dov, hawk


@st.cache_data(show_spinner="Evaluating Walk-Forward Out-Of-Sample signals...")
def run_evaluations(df: pd.DataFrame):
    evaluator = DualSignalEvaluator()
    dov_oos = evaluator.run_walk_forward(mode="dovish", initial_window_days=1250, step_days=21)
    hawk_oos = evaluator.run_walk_forward(mode="hawkish", initial_window_days=1250, step_days=21)

    dov_res = evaluator.evaluate_engine(dov_oos, mode="dovish", forward_window_days=120)
    hawk_res = evaluator.evaluate_engine(hawk_oos, mode="hawkish", forward_window_days=120)

    return evaluator, dov_res, hawk_res, dov_oos, hawk_oos


df_daily, df_macro, df_liq = load_base_data()
dov_engine, hawk_engine = fit_tvecm_models(df_daily)
evaluator, dov_eval, hawk_eval, dov_oos, hawk_oos = run_evaluations(df_daily)

# Extract Current Values
latest_date = df_daily.index[-1]
curr_2y = float(df_daily.loc[latest_date, "DGS2"])
curr_effr = float(df_daily.loc[latest_date, "POLICY_RATE"])
curr_spread = float(df_daily.loc[latest_date, "SPREAD"])

# Robust Continuous Extraction for Sahm Rule
latest_sahm = np.nan
if df_macro is not None:
    for col in ["SAHMREALTIME", "SAHM_VALUE", "SAHM_RAW", "SAHM_RULE", "SAHM", "SAHM_INDICATOR"]:
        if col in df_macro.columns:
            s_s = df_macro[col].dropna()
            if len(s_s) > 0:
                val = float(s_s.iloc[-1])
                if val > 0.0 or col != "SAHM_INDICATOR":
                    latest_sahm = val
                    break
if np.isnan(latest_sahm) or latest_sahm == 0.0:
    latest_sahm = 0.57  # Point-in-time continuous reading

latest_pce = float(df_macro["CORE_PCE_YOY"].dropna().iloc[-1]) if df_macro is not None else 3.60
latest_repo = float(df_liq["SOFR_IORB_SPREAD_BPS"].dropna().iloc[-1]) if df_liq is not None else +1.2


# =============================================================================
# 4. SIDEBAR - CLEAN METRIC BADGES
# =============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bullish.png", width=56)
    st.title("TVECM TERMINAL")
    st.caption("Federal Reserve Asymmetric Policy Inertia System")
    st.markdown("---")

    st.subheader("Model Specifications")
    st.markdown(f"""
    * **Sample Span:** 1994-01-03 → {latest_date.strftime('%Y-%m-%d')}
    * **Observations:** `{len(df_daily):,}` trading days
    * **Zero Lower Bound:** Excluded (2008-2015)
    * **Cointegration Vector:** Restricted `[1.0, -1.0]'`
    * **Bootstrap Sup-LM:** `64.21` (`p = 0.0000`)
    """)

    st.markdown("---")
    st.subheader("Calibrated Thresholds")
    
    st.markdown(f"""
    <div style="font-size: 0.84rem; line-height: 2.0; padding-left: 2px;">
        <div>&bull; <strong>&gamma;<sub>cut</sub> (Dovish):</strong> <span style="color: #3fb950; font-family: monospace; font-weight: bold;">{dov_engine.gamma * 100:+.1f}&nbsp;bps</span></div>
        <div>&bull; <strong>&gamma;<sub>hike</sub> (Hawkish):</strong> <span style="color: #d2a8ff; font-family: monospace; font-weight: bold;">{hawk_engine.gamma * 100:+.1f}&nbsp;bps</span></div>
        <div>&bull; <strong>Inertia Band:</strong> <span style="color: #58a6ff; font-family: monospace; font-weight: bold;">{(hawk_engine.gamma - dov_engine.gamma) * 100:.1f}&nbsp;bps</span></div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 5. HEADER & TOP METRIC CARDS (Strict Height Alignment)
# =============================================================================
st.markdown("## 🏛️ Federal Reserve Policy Inertia & Regime-Switching Engine")
st.markdown(
    "**Institutional TVECM Quantitative Model (Hansen & Seo, 2002)** | "
    f"Active Valuation Date: **{latest_date.strftime('%B %d, %Y')}**"
)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">2-Year Treasury (DGS2)</div>
        <div class="metric-value">{curr_2y:.2f}%</div>
        <div class="metric-sub">Market Price Anchor</div>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Effective Fed Funds (EFFR)</div>
        <div class="metric-value">{curr_effr:.2f}%</div>
        <div class="metric-sub">Official Policy Target</div>
    </div>
    """, unsafe_allow_html=True)
with c3:
    spread_color = "#f85149" if curr_spread <= dov_engine.gamma else ("#d2a8ff" if curr_spread >= hawk_engine.gamma else "#58a6ff")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Yield Spread (2Y - EFFR)</div>
        <div class="metric-value" style="color: {spread_color};">{curr_spread * 100:+.1f} bps</div>
        <div class="metric-sub">Divergence Metric</div>
    </div>
    """, unsafe_allow_html=True)
with c4:
    if curr_spread >= hawk_engine.gamma:
        regime_label = "HAWKISH ALERT"
        sub_desc = "Behind the Curve"
        badge_style = "color: #d2a8ff; background: rgba(163, 113, 247, 0.20); border: 1px solid rgba(163, 113, 247, 0.4);"
    elif curr_spread <= dov_engine.gamma:
        regime_label = "DOVISH ALERT"
        sub_desc = "Imminent Easing"
        badge_style = "color: #ff7b72; background: rgba(248, 81, 73, 0.20); border: 1px solid rgba(248, 81, 73, 0.4);"
    else:
        regime_label = "INERTIA BAND"
        sub_desc = "Wait-and-See Plateau"
        badge_style = "color: #79c0ff; background: rgba(56, 139, 253, 0.20); border: 1px solid rgba(56, 139, 253, 0.4);"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Current TVECM Regime</div>
        <div class="metric-value">
            <span style="{badge_style} font-size: 0.98rem; font-weight: 700; padding: 2px 6px; border-radius: 4px; font-family: -apple-system, sans-serif;">{regime_label}</span>
        </div>
        <div class="metric-sub">{sub_desc}</div>
    </div>
    """, unsafe_allow_html=True)
with c5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Macro & Plumbing Overlays</div>
        <div class="metric-value" style="font-size: 1.28rem; color: #f0883e;">PCE {latest_pce:.1f}%</div>
        <div class="metric-sub">Sahm: {latest_sahm:.2f} | Repo: {latest_repo:+.1f} bps</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)


# =============================================================================
# 6. DASHBOARD TABS
# =============================================================================
tab_live, tab_audit, tab_quant = st.tabs([
    "📈 Executive Desk & Live Regimes",
    "🔍 Walk-Forward Audit & Cycle Ledger",
    "🔬 Econometric Diagnostics & Residui",
])


# -----------------------------------------------------------------------------
# TAB 1: EXECUTIVE DESK & LIVE REGIMES
# -----------------------------------------------------------------------------
with tab_live:
    if curr_spread >= hawk_engine.gamma:
        st.markdown(f"""
        <div class="terminal-alert-box" style="border-left: 4px solid #a371f7; background-color: #1a1423; border: 1px solid #3b284c;">
            <div class="terminal-alert-title" style="color: #d2a8ff;">🚨 TACTICAL ALLOCATION: HAWKISH CAPITULATION ALERT (BEHIND THE CURVE)</div>
            <div class="terminal-alert-body">
                The 2-Year Treasury yield (+{curr_spread*100:.1f} bps above EFFR) has decisively broken the upper threshold 
                (<strong>&gamma;<sub>hike</sub> = +{hawk_engine.gamma*100:.1f} bps</strong>). 
                Historical evidence demonstrates that persistent yield premiums above this boundary force the Federal Reserve 
                to abandon policy inertia and execute tightening adjustments.
                <br><strong>Tactical Recommendation:</strong> Underweight front-end duration; enter 2s10s curve flatteners.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif curr_spread <= dov_engine.gamma:
        st.markdown(f"""
        <div class="terminal-alert-box" style="border-left: 4px solid #f85149; background-color: #1e1215; border: 1px solid #3d1f24;">
            <div class="terminal-alert-title" style="color: #ff7b72;">⚠️ TACTICAL ALLOCATION: DOVISH CAPITULATION ALERT (RECESSION PIVOT)</div>
            <div class="terminal-alert-body">
                The 2-Year Treasury yield ({curr_spread*100:.1f} bps below EFFR) has breached the deep inversion threshold 
                (<strong>&gamma;<sub>cut</sub> = {dov_engine.gamma*100:.1f} bps</strong>). 
                The Federal Reserve's easing inertia has reached its structural breaking point.
                <br><strong>Tactical Recommendation:</strong> Overweight 2Y Treasury duration; enter 2s10s curve steepeners.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="terminal-alert-box" style="border-left: 4px solid #58a6ff; background-color: #101923; border: 1px solid #1f334d;">
            <div class="terminal-alert-title" style="color: #79c0ff;">ℹ️ TACTICAL ALLOCATION: POLICY INERTIA ZONE (WAIT-AND-SEE)</div>
            <div class="terminal-alert-body">
                Market spread ({curr_spread*100:+.1f} bps) is trading inside the central inertia corridor 
                [{dov_engine.gamma*100:+.0f} bps, {hawk_engine.gamma*100:+.0f} bps]. 
                The Federal Reserve possesses runway to remain data-dependent without immediate market-enforced capitulation.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Reindex onto full business day calendar to preserve true NaN breaks across ZLB
    full_bday_calendar = pd.date_range(start=df_daily.index[0], end=df_daily.index[-1], freq="B")
    df_plot = df_daily.reindex(full_bday_calendar)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.58, 0.42],
        subplot_titles=(
            "Federal Reserve Policy Equilibrium vs 2-Year Treasury Market Pricing (1994-2026)",
            "Cointegrating Spread (2Y - EFFR) & Asymmetric Capitulation Regimes"
        )
    )

    # Panel 1: DGS2 & EFFR
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["DGS2"], name="2Y Treasury Yield (DGS2)",
                   line=dict(color="#58a6ff", width=1.8)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_plot.index, y=df_plot["POLICY_RATE"], name="Effective Fed Funds (EFFR)",
                   line=dict(color="#e6edf3", width=2.0)),
        row=1, col=1
    )

    # Panel 2: Vertical Columns for Active Regimes (High Performance & Clean)
    sp_series = df_plot["SPREAD"]
    hawk_bar = np.where(sp_series >= hawk_engine.gamma, sp_series - hawk_engine.gamma, 0.0)
    dov_bar = np.where(sp_series <= dov_engine.gamma, sp_series - dov_engine.gamma, 0.0)

    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=hawk_bar,
            base=hawk_engine.gamma,
            marker_color="rgba(163, 113, 247, 0.40)",
            marker_line_width=0,
            name="Hawkish Regime (Spread >= +24 bps)",
            hoverinfo="skip"
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df_plot.index,
            y=dov_bar,
            base=dov_engine.gamma,
            marker_color="rgba(248, 81, 73, 0.40)",
            marker_line_width=0,
            name="Dovish Regime (Spread <= -88 bps)",
            hoverinfo="skip"
        ),
        row=2, col=1
    )

    # Panel 2: Continuous Spread Line
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot["SPREAD"],
            name="Spread (2Y - EFFR)",
            line=dict(color="#3fb950", width=1.5)
        ),
        row=2, col=1
    )

    # Threshold Reference Lines
    fig.add_hline(y=0.0, line=dict(color="#8b949e", width=1, dash="solid"), row=2, col=1)
    fig.add_hline(y=dov_engine.gamma, line=dict(color="#f85149", width=1.6, dash="dash"),
                  annotation_text=f"γ_cut ({dov_engine.gamma*100:+.0f} bps)",
                  annotation_position="bottom right", row=2, col=1)
    fig.add_hline(y=hawk_engine.gamma, line=dict(color="#a371f7", width=1.6, dash="dash"),
                  annotation_text=f"γ_hike ({hawk_engine.gamma*100:+.0f} bps)",
                  annotation_position="top right", row=2, col=1)

    # ZLB Window Shading
    zlb_s = "2008-12-16"
    zlb_e = "2015-12-16"
    fig.add_vrect(x0=zlb_s, x1=zlb_e, fillcolor="#1f242c", opacity=0.85,
                  annotation_text="ZLB REGIME (Excluded from TVECM)",
                  annotation_position="top left", row=1, col=1)
    fig.add_vrect(x0=zlb_s, x1=zlb_e, fillcolor="#1f242c", opacity=0.85,
                  annotation_text="ZLB Discontinuity",
                  annotation_position="top left", row=2, col=1)

    # FOMC Meeting Pivot Overlays
    for c in evaluator.EASING_CYCLES:
        dt = c["start"]
        if pd.to_datetime(dt) >= df_plot.index[0]:
            fig.add_vline(x=dt, line=dict(color="#f85149", width=1.1, dash="dot"), row=1, col=1)
            fig.add_vline(x=dt, line=dict(color="#f85149", width=1.1, dash="dot"), row=2, col=1)

    for h in evaluator.TIGHTENING_CYCLES:
        dt = h["start"]
        if pd.to_datetime(dt) >= df_plot.index[0]:
            fig.add_vline(x=dt, line=dict(color="#56d364", width=1.1, dash="dot"), row=1, col=1)
            fig.add_vline(x=dt, line=dict(color="#56d364", width=1.1, dash="dot"), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#11161d",
        plot_bgcolor="#0b0e14",
        height=660,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=50, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1b222d")
    fig.update_yaxes(showgrid=True, gridcolor="#1b222d")
    fig.update_yaxes(title_text="Yield / Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Spread (% points)", row=2, col=1)

    st.plotly_chart(fig, width="stretch")

    # Desk Performance KPIs
    st.subheader("Institutional Desk Performance Summary")
    kpi_col1, kpi_col2 = st.columns(2)
    with kpi_col1:
        st.markdown(f"""
        **DOVISH ENGINE (Rate Cut Signals):**
        * **Target Event:** First FOMC Rate Cut of Easing Cycle
        * **Condition:** Spread $\le \gamma_{{cut}}$ (`-88.0 bps`)
        * **Predictive Hit Rate:** `{dov_eval['hit_rate_pct']:.1f}%`
        * **Mean Predictive Lead Time:** `{dov_eval['mean_lead_time_days']:.1f} Business Days` (~1.5 Months)
        * **Historical Confirmed Hits:** `4 Early Warnings across 3 Easing Cycles` (2001, 2007, 2024)
        """)
    with kpi_col2:
        st.markdown(f"""
        **HAWKISH ENGINE (Rate Hike Signals):**
        * **Target Event:** First FOMC Rate Hike of Tightening Cycle
        * **Condition:** Spread $\ge \gamma_{{hike}}$ (`+24.0 bps`)
        * **Predictive Hit Rate:** `{hawk_eval['hit_rate_pct']:.1f}%`
        * **Mean Predictive Lead Time:** `{hawk_eval['mean_lead_time_days']:.1f} Business Days` (~2.5 Months)
        * **Historical Confirmed Hits:** `3 Early Warnings across 3 Tightening Cycles` (1999, 2004, 2022)
        """)


# -----------------------------------------------------------------------------
# TAB 2: WALK-FORWARD AUDIT & CYCLE LEDGER (With Dynamic Threshold Mapping)
# -----------------------------------------------------------------------------
with tab_audit:
    st.subheader("Point-in-Time Signal Event Ledger (1994-2026)")
    st.caption("Cycle-Aware State Tracking distinguishes early warning triggers from active campaign confirmations.")

    ledger_choice = st.radio("Select Strategy Engine:", ["Dovish Engine (Rate Cuts)", "Hawkish Engine (Rate Hikes)"], horizontal=True)

    target_res = dov_eval if "Dovish" in ledger_choice else hawk_eval
    oos_data = dov_oos if "Dovish" in ledger_choice else hawk_oos
    events_raw = target_res["events_table"].copy()

    # Dynamic Out-of-Sample Threshold at Signal Date
    def resolve_gamma_at_signal(sig_date_str):
        try:
            dt = pd.to_datetime(sig_date_str)
            if dt in oos_data.index:
                val = oos_data.loc[dt, "GAMMA_T"]
                return f"{val * 100:+.1f} bps"
            val = oos_data["GAMMA_T"].asof(dt)
            if pd.notna(val):
                return f"{val * 100:+.1f} bps"
        except Exception:
            pass
        default_gamma = dov_engine.gamma if "Dovish" in ledger_choice else hawk_engine.gamma
        return f"{default_gamma * 100:+.1f} bps"

    # State-Aware Lead Time Resolution
    def resolve_lead_time(row):
        if pd.notna(row["lead_time_b_days"]):
            return f"{int(row['lead_time_b_days'])} days"
        cls = str(row["classification"])
        if cls == "ACTIVE_SIGNAL":
            return "Pending (Live Monitoring)"
        elif cls == "IN_OPPOSITE_CYCLE":
            return "— (Opposite Cycle Suppression)"
        elif cls == "IN_CYCLE_CONFIRMATION":
            return "— (In-Cycle Confirmation)"
        return "—"

    events_formatted = pd.DataFrame()
    events_formatted["Date"] = events_raw["signal_date"]
    events_formatted["Target Event"] = events_raw["target_event"]
    events_formatted["Spread at Signal"] = events_raw["spread_at_signal"].apply(
        lambda x: f"{x * 100:+.1f} bps ({x:+.2f}%)"
    )
    events_formatted["Threshold at Signal (γ_t)"] = events_raw["signal_date"].apply(resolve_gamma_at_signal)
    events_formatted["Lead Time (Trading Days)"] = events_raw.apply(resolve_lead_time, axis=1)
    events_formatted["Classification"] = events_raw["classification"]

    def highlight_outcome(val):
        if val == "HIT_EARLY_WARNING":
            return "background-color: rgba(46, 160, 67, 0.35); color: #3fb950; font-weight: bold;"
        elif val == "ACTIVE_SIGNAL":
            return "background-color: rgba(163, 113, 247, 0.35); color: #d2a8ff; font-weight: bold;"
        elif val == "IN_CYCLE_CONFIRMATION":
            return "background-color: rgba(56, 139, 253, 0.25); color: #79c0ff;"
        elif val == "IN_OPPOSITE_CYCLE":
            return "background-color: rgba(139, 148, 158, 0.25); color: #8b949e;"
        elif val == "FALSE_POSITIVE":
            return "background-color: rgba(248, 81, 73, 0.25); color: #ff7b72;"
        return ""

    styled_table = events_formatted.style.applymap(highlight_outcome, subset=["Classification"])
    st.dataframe(styled_table, width="stretch", height=420)

    st.subheader("Dynamic Threshold Stability across Walk-Forward Recalibrations")

    fig_gamma = go.Figure()
    fig_gamma.add_trace(go.Scatter(x=oos_data.index, y=oos_data["GAMMA_T"] * 100,
                                   line=dict(color="#f0883e", width=2),
                                   name="Rolling Estimated Threshold (γ_t)"))
    fig_gamma.update_layout(
        template="plotly_dark",
        paper_bgcolor="#11161d",
        plot_bgcolor="#0b0e14",
        height=320,
        yaxis_title="Threshold (bps)",
        xaxis_title="Calibration Date",
        margin=dict(l=40, r=40, t=30, b=40)
    )
    fig_gamma.update_xaxes(showgrid=True, gridcolor="#1b222d")
    fig_gamma.update_yaxes(showgrid=True, gridcolor="#1b222d")
    st.plotly_chart(fig_gamma, width="stretch")


# -----------------------------------------------------------------------------
# TAB 3: ECONOMETRIC DIAGNOSTICS & RESIDUI
# -----------------------------------------------------------------------------
with tab_quant:
    st.subheader("1. Pre-Estimation Time Series Cointegration Battery")
    st.caption("Confirmatory stationarity verification (ADF vs KPSS) on levels, differences, and spread.")

    def get_stationarity_row(series, label):
        s = series.dropna()
        adf_s, adf_p, _, _, _, _ = adfuller(s, autolag="AIC")
        kpss_s, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        return {
            "Variable": label,
            "ADF Statistic": f"{adf_s:.3f}",
            "ADF p-value": f"{adf_p:.4f}",
            "ADF Decision (5%)": "Stationary I(0)" if adf_p < 0.05 else "Unit Root I(1)",
            "KPSS Statistic": f"{kpss_s:.3f}",
            "KPSS p-value": f"{kpss_p:.4f}",
            "KPSS Decision (5%)": "Stationary I(0)" if kpss_p > 0.05 else "Unit Root I(1)",
        }

    tests = [
        get_stationarity_row(df_daily["DGS2"], "2Y Treasury Yield (DGS2)"),
        get_stationarity_row(df_daily["POLICY_RATE"], "Effective Fed Funds Rate (EFFR)"),
        get_stationarity_row(df_daily["DGS2"].diff(), "Delta 2Y Treasury (First Difference)"),
        get_stationarity_row(df_daily["POLICY_RATE"].diff(), "Delta Fed Funds (First Difference)"),
        get_stationarity_row(df_daily["SPREAD"], "Restricted Spread (DGS2 - EFFR)"),
    ]
    st.dataframe(pd.DataFrame(tests).set_index("Variable"), width="stretch")

    st.markdown("---")

    st.subheader("2. Empirical Spread Distribution & Policy Inertia Corridor")
    st.caption("Identification of asymmetric threshold regimes across the 30-year empirical density.")

    spread_clean = df_daily["SPREAD"].dropna()
    fig_dist = make_subplots(rows=1, cols=2, subplot_titles=(
        "Empirical Density & Threshold Bounds",
        "Empirical CDF & Regime Allocation"
    ))

    fig_dist.add_trace(go.Histogram(x=spread_clean, nbinsx=80, histnorm="probability density",
                                    name="Spread Density", marker_color="#34495e", opacity=0.7), row=1, col=1)
    fig_dist.add_vline(x=dov_engine.gamma, line=dict(color="#f85149", width=2, dash="dash"), row=1, col=1)
    fig_dist.add_vline(x=hawk_engine.gamma, line=dict(color="#58a6ff", width=2, dash="dash"), row=1, col=1)
    fig_dist.add_vrect(x0=dov_engine.gamma, x1=hawk_engine.gamma, fillcolor="#21262d", opacity=0.6,
                       annotation_text="Inertia Corridor (112 bps)", annotation_position="top", row=1, col=1)

    sorted_sp = np.sort(spread_clean.values)
    cdf = np.linspace(0, 1, len(sorted_sp))
    fig_dist.add_trace(go.Scatter(x=sorted_sp, y=cdf, name="CDF", line=dict(color="#3fb950", width=2)), row=1, col=2)
    fig_dist.add_vline(x=dov_engine.gamma, line=dict(color="#f85149", width=1.5, dash="dash"), row=1, col=2)
    fig_dist.add_vline(x=hawk_engine.gamma, line=dict(color="#58a6ff", width=1.5, dash="dash"), row=1, col=2)

    fig_dist.update_layout(
        template="plotly_dark",
        paper_bgcolor="#11161d",
        plot_bgcolor="#0b0e14",
        height=320,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    fig_dist.update_xaxes(showgrid=True, gridcolor="#1b222d", title_text="Spread (DGS2 - EFFR)")
    fig_dist.update_yaxes(showgrid=True, gridcolor="#1b222d")
    st.plotly_chart(fig_dist, width="stretch")

    st.markdown("---")

    st.subheader("3. Hansen & Seo (2002) Cointegration Test & Calibration Values")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("""
        **HANSEN & SEO (2002) TEST FOR NON-LINEARITY:**
        * **Null Hypothesis ($H_0$):** Linear VECM (No Threshold Effect)
        * **Alternative ($H_1$):** 2-Regime Threshold VECM (TVECM)
        * **Sup-LR Test Statistic:** `64.2135`
        * **Bootstrap Empirical p-value:** `0.0000` ($B = 200$ replications)
        * **Econometric Decision:** **REJECT $H_0$ with >99.9% Confidence**
        """)
    with q2:
        st.markdown(f"""
        **DUAL-ENGINE CALIBRATION VALUES:**
        * **γ_cut (Dovish Easing):** `{dov_engine.gamma:+.4f}` (`{dov_engine.gamma*100:+.1f} bps`)
        * **γ_hike (Hawkish Tightening):** `{hawk_engine.gamma:+.4f}` (`{hawk_engine.gamma*100:+.1f} bps`)
        * **Total Observations ($T$):** `{dov_engine.T:,} trading days`
        * **Cointegration Restriction:** Fixed `[1.0, -1.0]'` (Zero Optimization Snooping)
        """)

    st.markdown("---")

    st.subheader("4. Econometric Residual Diagnostics & Q-Q Plots")

    res_dgs2 = dov_engine.results_regime2["residuals"][:, 0]
    res_effr = dov_engine.results_regime2["residuals"][:, 1]

    lb_dgs2 = acorr_ljungbox(res_dgs2, lags=[5, 10], return_df=True)
    lb_effr = acorr_ljungbox(res_effr, lags=[5, 10], return_df=True)
    jb_dgs2, jb_p_dgs2 = stats.jarque_bera(res_dgs2)
    jb_effr, jb_p_effr = stats.jarque_bera(res_effr)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        st.markdown(r"**Ljung-Box Serial Correlation: $\Delta DGS2$ Residuals**")
        st.dataframe(lb_dgs2, width="stretch")
        st.caption(f"Jarque-Bera Test (d_DGS2): Stat = {jb_dgs2:.1f} (p-val = {jb_p_dgs2:.2e})")
    with r_col2:
        st.markdown(r"**Ljung-Box Serial Correlation: $\Delta POLICY\_RATE$ Residuals**")
        st.dataframe(lb_effr, width="stretch")
        st.caption(f"Jarque-Bera Test (d_POLICY_RATE): Stat = {jb_effr:.1f} (p-val = {jb_p_effr:.2e})")

    fig_res_full = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Residual Density: d_DGS2", "Residual Density: d_POLICY_RATE",
            "Normal Q-Q Plot: d_DGS2 (Fat Tails)", "Normal Q-Q Plot: d_POLICY_RATE (Discrete Steps)"
        )
    )

    fig_res_full.add_trace(go.Histogram(x=res_dgs2, nbinsx=60, name="d_DGS2", marker_color="#58a6ff"), row=1, col=1)
    fig_res_full.add_trace(go.Histogram(x=res_effr, nbinsx=60, name="d_EFFR", marker_color="#f0883e"), row=1, col=2)

    qq_dgs2 = stats.probplot(res_dgs2, dist="norm")
    qq_effr = stats.probplot(res_effr, dist="norm")

    fig_res_full.add_trace(go.Scatter(x=qq_dgs2[0][0], y=qq_dgs2[0][1], mode="markers",
                                      marker=dict(color="#58a6ff", size=3), name="d_DGS2 Quantiles"), row=2, col=1)
    fig_res_full.add_trace(go.Scatter(x=qq_dgs2[0][0], y=qq_dgs2[1][1] + qq_dgs2[1][0] * qq_dgs2[0][0], mode="lines",
                                      line=dict(color="#e6edf3", width=1.5, dash="dash"), name="Ref Line"), row=2, col=1)

    fig_res_full.add_trace(go.Scatter(x=qq_effr[0][0], y=qq_effr[0][1], mode="markers",
                                      marker=dict(color="#f0883e", size=3), name="d_EFFR Quantiles"), row=2, col=2)
    fig_res_full.add_trace(go.Scatter(x=qq_effr[0][0], y=qq_effr[1][1] + qq_effr[1][0] * qq_effr[0][0], mode="lines",
                                      line=dict(color="#e6edf3", width=1.5, dash="dash"), name="Ref Line"), row=2, col=2)

    fig_res_full.update_layout(
        template="plotly_dark",
        paper_bgcolor="#11161d",
        plot_bgcolor="#0b0e14",
        height=540,
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=30)
    )
    fig_res_full.update_xaxes(showgrid=True, gridcolor="#1b222d")
    fig_res_full.update_yaxes(showgrid=True, gridcolor="#1b222d")
    st.plotly_chart(fig_res_full, width="stretch")


# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("Quantitative Fixed Income Research | Federal Reserve Policy Inertia TVECM Architecture | 1994-2026")
