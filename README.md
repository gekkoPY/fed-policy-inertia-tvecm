# fed-policy-inertia-tvecm
Institutional-grade Threshold Vector Error Correction Model (TVECM) estimating asymmetric Federal Reserve policy inertia, market capitulation bounds, and front-end Treasury regime-switching (1994–2026).
# Federal Reserve Policy Inertia & Regime-Switching Engine (TVECM)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Econometric Framework](https://img.shields.io/badge/econometrics-Hansen%20%26%20Seo%20(2002)-orange.svg?style=flat-square)](https://doi.org/10.1016/S0304-4076(02)00130-7)
[![Streamlit Interface](https://img.shields.io/badge/dashboard-Streamlit%20Dark-red.svg?style=flat-square)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)

An institutional-grade quantitative research framework and interactive decision-support terminal designed to model **Federal Reserve monetary policy inertia** and predict interest rate pivot capitulations. 

By applying the **Hansen & Seo (2002)** dual-engine Threshold Vector Error Correction Model (**TVECM**) to 30+ years of high-frequency market data (1994–2026), this system isolates the non-linear boundaries at which market pressure forces the Federal Open Market Committee (FOMC) to abandon policy patience.

---

## Key Quantitative Findings & Calibrated Architecture

* **The Restricted Cointegrating Anchor:** The equilibrium spread is theoretically restricted to $s_t = y_{2Y, t} - EFFR_t$, reflecting the structural cointegrating vector $[1.0, -1.0]'$ with zero optimization snooping.
* **Significant Non-Linear Cointegration:** Hansen & Seo (2002) Sup-LM test statistic of **$64.21$** ($p = 0.0000$, bootstrapped across $B = 200$ replications), decisively rejecting the linear VECM specification in favor of a 2-regime threshold mechanism with $>99.9\%$ confidence.
* **Asymmetric Capitulation Thresholds:**
  * **Dovish Threshold ($\gamma_{\text{cut}}$):** **$-88.0\text{ bps}$** ($-0.88\%$). The Federal Reserve exhibits extreme easing inertia, requiring severe yield curve inversion and market pain before executing rate cuts.
  * **Hawkish Threshold ($\gamma_{\text{hike}}$):** **$+24.0\text{ bps}$** ($+0.24\%$). Central bank tolerance for upside inflation and growth surprises is structurally tight, causing rapid hiking reactions.
* **The Policy Inertia Corridor:** A central tolerance band of **$112.0\text{ bps}$** where the Fed retains freedom to maintain a "wait-and-see" data-dependent stance without market-enforced capitulation.
* **Zero Lower Bound (ZLB) Sterilization:** Complete exclusion of the 2008-12-16 to 2015-12-16 zero-rate regime to prevent artificial variance compression and structural break biases.

---

## Predictive Performance & Out-of-Sample Audit

Evaluated using an expanding walk-forward out-of-sample (OOS) engine ($1,250$ initial trading days, recalibrated every $21$ business days, evaluated across a $120$-day forward execution window):

| Strategy Engine | Target FOMC Pivot Event | Calibration Threshold | Hit Rate (%) | Mean Lead Time | Confirmed Pivot Captures |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Dovish Engine** | First Rate Cut of Easing Cycle | $s_t \le \gamma_{\text{cut}}\;(-88\text{ bps})$ | **57.1%** | **33.5 b-days** (~1.5 months) | 2001 Dot-Com, 2007 GFC, 2024 Normalization |
| **Hawkish Engine** | First Rate Hike of Tightening Cycle | $s_t \ge \gamma_{\text{hike}}\;(+24\text{ bps})$ | **60.0%** | **53.3 b-days** (~2.5 months) | 1999 Dot-Com, 2004 Measured Pace, 2022 Post-Inflation |

### State-Aware Signal Classification
The model implements a proprietary 5-state cycle ledger preventing look-ahead and cycle-overlap biases:
1. `HIT_EARLY_WARNING`: Out-of-sample trigger within 120 business days prior to the first rate pivot.
2. `IN_CYCLE_CONFIRMATION`: Signal triggered during an active ongoing policy campaign (tactical continuation).
3. `IN_OPPOSITE_CYCLE`: Bull steepening suppression (e.g., market rate rebounding above EFFR while Fed is rapidly easing).
4. `FALSE_POSITIVE`: Signal triggered where the central bank persisted in inertia beyond the 120-day horizon.
5. `ACTIVE_SIGNAL`: Live pending market dislocation awaiting subsequent FOMC decisions.

---

## Institutional Bloomberg Terminal Dashboard (`app.py`)

The application features a dark-mode interactive terminal built in Python/Streamlit and Plotly WebGL, designed for portfolio managers and curve trading desks.

### 1. Executive Desk & Live Regimes
* **Live Market Pricing & Macro Overlays:** Real-time monitoring of 2Y Treasury ($4.56\%$), EFFR ($3.63\%$), Spread ($+93.0\text{ bps}$), Core PCE ($3.6\%$), Real-Time Sahm Rule ($0.57$), and SOFR-IORB liquidity plumbing spread ($-3.0\text{ bps}$).
* **30-Year High-Frequency Interactive Engine:** Dynamic Plotly chart with true vertical column shading for active capitulation regimes, historical FOMC pivot markers, and ZLB discontinuity masks.
* **Tactical Allocation Signals:** Rule-based desk recommendations (duration positioning, 2s10s curve flatteners/steepeners).

### 2. Walk-Forward Audit & Cycle Ledger
* Point-in-time signal event audit displaying dynamic threshold calibration at signal date ($\gamma_t$), actual market spread, lead time in business days, and verified status classification.
* Rolling threshold parameter stability chart across expanding calibration horizons.

### 3. Econometric Diagnostics & Residual Auditing
* **Cointegration Battery:** Full stationarity verification via Augmented Dickey-Fuller (ADF) and KPSS tests on levels, first differences, and restricted spread.
* **Hansen & Seo Distribution Matrix:** Empirical density, kernel density estimation, and cumulative distribution function (CDF) illustrating regime probabilities.
* **Residual Diagnostics:** Ljung-Box autocorrelation tests confirming residual white noise ($p > 0.05$ at lags 5 and 10) and Normal Q-Q plots diagnosing leptokurtic asset returns vs discrete policy jump dynamics.

---

## Econometric Methodology

### 1. Threshold Vector Error Correction Model (Hansen & Seo, 2002)
The model specifies a two-regime TVECM(1) with a restricted cointegrating relation $w_t = \beta' X_t = y_{2Y, t} - EFFR_t$:

$$\Delta X_t = \begin{cases}  \mu^{(1)} + \alpha^{(1)} (w_{t-1} - \gamma) + \sum_{i=1}^{k} \Gamma_i^{(1)} \Delta X_{t-i} + \varepsilon_t, & w_{t-1} \le \gamma \\ \mu^{(2)} + \alpha^{(2)} (w_{t-1} - \gamma) + \sum_{i=1}^{k} \Gamma_i^{(2)} \Delta X_{t-i} + \varepsilon_t, & w_{t-1} > \gamma  \end{cases}$$

Where:
* $X_t = [y_{2Y, t}, EFFR_t]'$ represents the vector of endogenous interest rates.
* $\gamma$ is the threshold parameter estimated via grid search maximum likelihood over the empirical spread distribution trimmed at $15\%$.
* $\alpha^{(1)}, \alpha^{(2)}$ represent the regime-dependent error correction speeds.

### 2. Confirmatory Unit Root & Cointegration Properties
* **Levels ($y_{2Y}, EFFR$):** ADF fails to reject unit root ($p = 0.4730, 0.5838$), KPSS rejects stationarity ($p = 0.0100$). Both series are confirmed $I(1)$.
* **First Differences ($\Delta y_{2Y}, \Delta EFFR$):** ADF rejects unit root ($p < 0.0001$), KPSS fails to reject stationarity ($p = 0.1000$). First differences are stationary $I(0)$.
* **Restricted Spread ($y_{2Y} - EFFR$):** ADF rejects non-stationarity ($p = 0.0012$). The discrepancy with KPSS ($p = 0.0100$) formally documents non-linear threshold persistence rather than structural unit-root non-stationarity, validating the Hansen & Seo formulation.

---

## Repository Architecture

```text
fed-policy-inertia-tvecm/
├── app.py                          # Institutional Streamlit Dashboard & Terminal
├── src/
│   ├── data_loader.py              # FRED API client, frequency alignment & ZLB filter
│   ├── threshold_engine.py         # Hansen & Seo (2002) TVECM estimation & Sup-LM test
│   ├── signal_evaluator.py         # Out-of-Sample expanding walk-forward backtest
│   └── macro_walkthrough.py        # Complete execution pipeline and static analytics
├── data/
│   ├── raw/                        # Synchronized FRED high-frequency parquet/csv series
│   └── processed/                  # Cointegrated time-series and lagged macro matrices
├── reports/
│   └── figures/                    # High-resolution vector charts and diagnostic plots
├── .streamlit/
│   └── config.toml                 # Bloomberg dark palette theme configuration
├── requirements.txt                # Production environment dependencies
└── README.md                       # Comprehensive institutional documentation






## Quickstart Installation

### Prerequisites
* Python 3.10, 3.11, or 3.12 (Anaconda/Miniconda recommended)
* Optional: St. Louis Fed FRED API Key (free registration)

### Setup & Execution

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/fed-policy-inertia-tvecm.git](https://github.com/YOUR_USERNAME/fed-policy-inertia-tvecm.git)
   cd fed-policy-inertia-tvecm
   ```

2. **Create and activate environment:**
   ```bash
   conda create -n fed-tvecm python=3.11 -y
   conda activate fed-tvecm
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard:**
   You can launch the dashboard using standard Python (auto-launch wrapper enabled) or direct Streamlit:
   ```bash
   python app.py
   ```
   *Alternatively:*
   ```bash
   streamlit run app.py
   ```
   The browser terminal will automatically initialize at `http://localhost:8501`.

---

## Author & Citation

**Tommaso De Benedetti**  
*MSc in Banking & Financial Intermediaries, LUISS Guido Carli University (Rome)*  
*Focus: Quantitative Finance, Macro Research & Fixed Income Derivatives*

If utilizing this codebase or econometric methodology for academic or institutional research, please cite:

```bibtex
@misc{debenedetti2026fedtvecm,
  author = {De Benedetti, Tommaso},
  title = {Federal Reserve Policy Inertia: A Threshold Vector Error Correction Approach},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{[https://github.com/YOUR_USERNAME/fed-policy-inertia-tvecm](https://github.com/YOUR_USERNAME/fed-policy-inertia-tvecm)}}
}
```

---

## License
Distributed under the MIT License. See `LICENSE` for more information.
