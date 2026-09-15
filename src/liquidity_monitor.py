"""
fed-policy-inertia-tvecm
Module: src/liquidity_monitor.py

Ancillary Liquidity & Repo Stress Monitor.
Detects interbank funding frictions, dealer balance sheet constraints,
and exogenous liquidity shocks (e.g., September 2019 repo spike, March 2023 SVB run)
to distinguish macro-driven Fed pivots from emergency plumbing interventions.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Dynamic Path Resolution: enables direct CLI execution from root or src/
# -----------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data_loader import MacroDataLoader
except ImportError:
    from src.data_loader import MacroDataLoader


class LiquidityMonitor:
    """Monitors money market microstructure and financial conditions to provide

    an exogenous shock overlay for the TVECM reaction function.
    """

    # Institutional stress thresholds
    SOFR_IORB_SOFT_THRESHOLD_BPS: float = 0.0    # Inversion: SOFR trading above IORB floor
    SOFR_IORB_SEVERE_THRESHOLD_BPS: float = 5.0  # Acute dealer balance sheet strain
    NFCI_TIGHTENING_THRESHOLD: float = 0.0       # Conditions tighter than historical norm
    NFCI_DELTA_4W_SPIKE: float = 0.15            # Fast velocity of financial tightening

    def __init__(
        self,
        loader: Optional[MacroDataLoader] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """Initialize monitor with data loader instance."""
        if loader is not None:
            self.loader = loader
        elif data_dir is not None:
            self.loader = MacroDataLoader(data_dir=data_dir)
        else:
            self.loader = MacroDataLoader()

        self.data: Optional[pd.DataFrame] = None

    def fetch_and_build(
        self,
        start_date: str = "2018-04-01",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Loads underlying repo and financial condition series and constructs

        stress indicators and diagnostic z-scores.

        Args:
            start_date: Observation start (defaults to SOFR inception: April
              2018).
            end_date: Observation end date.

        Returns:
            pd.DataFrame containing rates, spreads, rolling metrics, and regime
            flags.
        """
        # Ingest Module 3 series via MacroDataLoader
        raw_df = self.loader.load_module3_liquidity(start_date=start_date, end_date=end_date)
        df = raw_df.copy()

        # 1. Microstructure Spread: SOFR - IORB (bps)
        # In an abundant reserve regime, SOFR trades slightly below IORB (typically -5 to -1 bps)
        df["SOFR_IORB_SPREAD_BPS"] = (df["SOFR"] - df["FED_REMUNERATION"]) * 100

        # Rolling 21-day (approx. 1 month) statistics to track regime drift
        df["SPREAD_ROLLING_MEAN_21D"] = df["SOFR_IORB_SPREAD_BPS"].rolling(window=21).mean()
        df["SPREAD_ROLLING_STD_21D"] = df["SOFR_IORB_SPREAD_BPS"].rolling(window=21).std()
        df["SPREAD_ZSCORE_21D"] = (
            (df["SOFR_IORB_SPREAD_BPS"] - df["SPREAD_ROLLING_MEAN_21D"])
            / df["SPREAD_ROLLING_STD_21D"].replace(0, np.nan)
        )

        # 2. Macro Financial Conditions: Chicago Fed NFCI
        df["NFCI_DELTA_4W"] = df["NFCI"] - df["NFCI"].shift(20)

        # 3. Diagnostic Stress Flags
        # Mild friction: SOFR trades equal to or above IORB floor
        df["FLAG_REPO_FRICTION"] = (
            df["SOFR_IORB_SPREAD_BPS"] >= self.SOFR_IORB_SOFT_THRESHOLD_BPS
        ).astype(int)

        # Acute repo crisis: SOFR - IORB >= 5 bps or rolling z-score >= 2.5
        df["FLAG_REPO_CRISIS"] = (
            (df["SOFR_IORB_SPREAD_BPS"] >= self.SOFR_IORB_SEVERE_THRESHOLD_BPS)
            | (df["SPREAD_ZSCORE_21D"] >= 2.5)
        ).astype(int)

        # Macro financial condition strain
        df["FLAG_NFCI_TIGHT"] = (df["NFCI"] > self.NFCI_TIGHTENING_THRESHOLD).astype(int)
        df["FLAG_NFCI_RAPID_TIGHTENING"] = (df["NFCI_DELTA_4W"] >= self.NFCI_DELTA_4W_SPIKE).astype(int)

        # Composite Exogenous Plumbing Stress Flag
        # Triggered if repo markets crack OR rapid NFCI deterioration occurs concurrently
        df["EXOGENOUS_LIQUIDITY_SHOCK"] = (
            (df["FLAG_REPO_CRISIS"] == 1)
            | ((df["FLAG_REPO_FRICTION"] == 1) & (df["FLAG_NFCI_RAPID_TIGHTENING"] == 1))
        ).astype(int)

        self.data = df

        # Save processed features to data/processed/
        processed_path = self.loader.processed_dir / "liquidity_monitor_features.csv"
        df.to_csv(processed_path)
        return df

    def classify_pivot_catalyst(
        self,
        event_date: str,
        lookback_days: int = 15,
    ) -> dict:
        """Determines whether a policy shift or regime break around event_date

        was driven by cyclical macro pressures or an exogenous plumbing shock.

        Args:
            event_date: Target event date (YYYY-MM-DD).
            lookback_days: Pre-event inspection window in business days.

        Returns:
            dict containing catalyst classification, peak spread, and narrative
            context.
        """
        if self.data is None:
            raise ValueError("Data not initialized. Execute .fetch_and_build() first.")

        target_dt = pd.to_datetime(event_date)
        window_df = self.data.loc[:target_dt].tail(lookback_days)

        if window_df.empty:
            return {"date": event_date, "classification": "INSUFFICIENT_DATA"}

        max_repo_spread = window_df["SOFR_IORB_SPREAD_BPS"].max()
        max_zscore = window_df["SPREAD_ZSCORE_21D"].max()
        exogenous_shock_count = window_df["EXOGENOUS_LIQUIDITY_SHOCK"].sum()
        latest_nfci = window_df["NFCI"].iloc[-1]

        if exogenous_shock_count > 0 or max_repo_spread >= self.SOFR_IORB_SEVERE_THRESHOLD_BPS:
            catalyst = "EXOGENOUS_PLUMBING_SHOCK"
            rationale = (
                f"Severe interbank funding friction detected: peak SOFR-IORB spread of {max_repo_spread:.2f} bps "
                f"(rolling z-score: {max_zscore:.2f}). Liquidity constraints forced Fed intervention."
            )
        else:
            catalyst = "MACRO_CYCLICAL_ALIGNMENT"
            rationale = (
                f"Interbank plumbing remained orderly: peak SOFR-IORB spread confined to {max_repo_spread:.2f} bps. "
                f"Policy adjustment governed by 2Y yield curve discount."
            )

        return {
            "date": str(target_dt.date()),
            "classification": catalyst,
            "max_repo_spread_bps": float(max_repo_spread),
            "max_spread_zscore": float(max_zscore) if not np.isnan(max_zscore) else 0.0,
            "latest_nfci": float(latest_nfci),
            "rationale": rationale,
        }


if __name__ == "__main__":
    print("--- Testing Module 3: Liquidity & Repo Stress Monitor ---")
    monitor = LiquidityMonitor()
    features = monitor.fetch_and_build(start_date="2018-04-01")
    print(f"Liquidity features shape: {features.shape}")
    print("\nRecent liquidity monitor readings:")
    print(features[["SOFR", "FED_REMUNERATION", "SOFR_IORB_SPREAD_BPS", "NFCI", "EXOGENOUS_LIQUIDITY_SHOCK"]].tail())

    # Historical verification: The September 2019 Repo Spike
    print("\n--- Diagnostic Check: September 2019 Repo Crisis ---")
    sept_2019_audit = monitor.classify_pivot_catalyst(event_date="2019-09-18", lookback_days=10)
    print(f"Classification  : {sept_2019_audit['classification']}")
    print(f"Peak Repo Spread: {sept_2019_audit['max_repo_spread_bps']:.2f} bps")
    print(f"Rationale       : {sept_2019_audit['rationale']}")
