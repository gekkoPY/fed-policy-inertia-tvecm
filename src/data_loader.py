"""
fed-policy-inertia-tvecm
Module: src/data_loader.py

Data ingestion, synchronization, and caching pipeline utilizing FRED API
and the Federal Reserve Bank of Atlanta. Implements three specialized modules:
1. Module 1: Daily High-Frequency Benchmark (DGS2 vs chained DFF/EFFR with ZLB filter, 1994-present)
2. Module 2: Monthly Structural Model (DGS2 vs Wu-Xia Shadow Rate, 1994-present)
3. Module 3: Ancillary Liquidity & Repo Stress Monitor (SOFR-IORB spread & NFCI)
4. Step 4: Conditional Macro Confirmation Filters (Sahm Rule & Core PCE YoY)
   with explicit 35-day reporting lag to eliminate look-ahead bias.
"""

import os
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from fredapi import Fred

# =============================================================================
# FRED API CONFIGURATION
# =============================================================================
FRED_API_KEY: str = "91839022282387b67cb8a1dadda386d0"


def _resample_month_end(series_or_df):
    """Ensures backward and forward compatibility across pandas versions."""
    try:
        return series_or_df.resample("ME").last()
    except ValueError:
        return series_or_df.resample("M").last()


class MacroDataLoader:
    """Manages raw data ingestion, disk caching, frequency synchronization,

    and feature engineering for TVECM regime-switching models.
    """

    SERIES_MAP = {
        "dgs2": "DGS2",  # 2-Year Treasury Constant Maturity Yield (Daily)
        "dtb3": "DTB3",  # 3-Month Treasury Bill Secondary Market Rate (Daily)
        "effr": "EFFR",  # Effective Federal Funds Rate (Daily, post-July 2000)
        "dff": "DFF",  # Historical Fed Funds Effective Rate (Daily, 1954-2000)
        "sofr": "SOFR",  # Secured Overnight Financing Rate (Daily)
        "iorb": "IORB",  # Interest on Reserve Balances (Daily, post-July 2021)
        "ioer": "IOER",  # Interest on Excess Reserves (Daily, historical)
        "nfci": "NFCI",  # Chicago Fed National Financial Conditions Index (Weekly)
        "unrate": "UNRATE",  # Civilian Unemployment Rate (Monthly)
        "core_pce": "PCEPILFE",  # Core PCE Price Index excluding Food & Energy (Monthly)
    }

    # Zero Lower Bound (ZLB) conventional window (GFC era)
    ZLB_START = "2008-12-16"
    ZLB_END = "2015-12-16"

    def __init__(
        self,
        fred_api_key: Optional[str] = None,
        data_dir: Optional[str] = None,
    ) -> None:
        """Initialize directory paths and authenticate FRED API client."""
        if data_dir is not None:
            self.root_dir = Path(data_dir)
        else:
            current_path = Path(__file__).resolve().parent
            project_root = (
                current_path.parent
                if current_path.name == "src"
                else current_path
            )
            self.root_dir = project_root / "data"

        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        resolved_key = (
            fred_api_key or FRED_API_KEY or os.getenv("FRED_API_KEY")
        )

        self.fred: Optional[Fred] = None
        if resolved_key:
            sanitized_key = resolved_key.strip("[] \t\n")
            if sanitized_key:
                self.fred = Fred(api_key=sanitized_key)

    def fetch_fred_series(
        self,
        series_id: str,
        start_date: str = "1990-01-01",
        end_date: Optional[str] = None,
        force_reload: bool = False,
    ) -> pd.Series:
        """Fetch historical series from FRED API with automatic local CSV

        caching.
        """
        file_path = self.raw_dir / f"{series_id}.csv"

        if file_path.exists() and not force_reload:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            if not df.empty and df.index[0] <= pd.to_datetime(start_date):
                series = df.squeeze("columns")
                series.name = series_id
                return series

        if self.fred is None:
            raise ValueError(
                f"Missing valid FRED API key and no cached file found at '{file_path}'."
            )

        series = self.fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )
        series.name = series_id
        series.to_csv(file_path, header=True)
        return series

    # =========================================================================
    # MODULE 1: Daily High-Frequency Benchmark (1994-present)
    # =========================================================================
    def load_module1_daily(
        self,
        policy_rate: str = "EFFR",
        start_date: str = "1994-01-01",
        end_date: Optional[str] = None,
        exclude_zlb: bool = True,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Constructs synchronized daily dataset for high-frequency TVECM.

        Chains modern NY Fed EFFR (post-July 2000) with historical Board DFF
        (1994-2000)
        to ensure seamless, uninterrupted daily policy rate observations.
        Cointegration relationship: Spread_t = Y(2Y)_t - PolicyRate_t
        """
        dgs2 = self.fetch_fred_series(
            "DGS2",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )

        if policy_rate.upper() == "EFFR":
            # EFFR starts in July 2000 on FRED; chain with historical DFF for 1994-2000 continuity
            effr_modern = self.fetch_fred_series(
                "EFFR",
                start_date="2000-07-03",
                end_date=end_date,
                force_reload=force_reload,
            )
            dff_hist = self.fetch_fred_series(
                "DFF",
                start_date=start_date,
                end_date="2000-07-02",
                force_reload=force_reload,
            )
            pol = effr_modern.combine_first(dff_hist)
            pol.name = "POLICY_RATE"
        else:
            pol = self.fetch_fred_series(
                policy_rate,
                start_date=start_date,
                end_date=end_date,
                force_reload=force_reload,
            )
            pol.name = "POLICY_RATE"

        df = pd.concat([dgs2, pol], axis=1).dropna()
        df.columns = ["DGS2", "POLICY_RATE"]
        df = df.sort_index().ffill().dropna()
        df["SPREAD"] = df["DGS2"] - df["POLICY_RATE"]

        if exclude_zlb:
            mask_zlb = (df.index >= self.ZLB_START) & (df.index <= self.ZLB_END)
            df = df.loc[~mask_zlb].copy()

        df.to_csv(self.processed_dir / "module1_daily.csv")
        return df

    # =========================================================================
    # MODULE 2: Monthly Structural Model (Wu-Xia Shadow Rate Extension)
    # =========================================================================
    def load_module2_monthly(
        self,
        shadow_rate_path: Optional[str] = None,
        start_date: str = "1994-01-01",
        end_date: Optional[str] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Constructs continuous monthly structural series (1994-present)

        incorporating the Atlanta Fed Wu-Xia Shadow Rate during ZLB regimes
        and seamlessly chaining with monthly EFFR post-2022.
        """
        dgs2_daily = self.fetch_fred_series(
            "DGS2",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )
        effr_daily = self.fetch_fred_series(
            "EFFR",
            start_date="2000-07-03",
            end_date=end_date,
            force_reload=force_reload,
        )
        dff_daily = self.fetch_fred_series(
            "DFF",
            start_date=start_date,
            end_date="2000-07-02",
            force_reload=force_reload,
        )
        chained_policy_daily = effr_daily.combine_first(dff_daily)

        dgs2_m = _resample_month_end(dgs2_daily).rename("DGS2_M")
        effr_m = _resample_month_end(chained_policy_daily).rename("EFFR_M")

        excel_file = (
            Path(shadow_rate_path)
            if shadow_rate_path
            else self.raw_dir / "WuXiaShadowRate.xlsx"
        )

        if not excel_file.exists():
            raise FileNotFoundError(
                f"Wu-Xia Excel file not found at '{excel_file}'. "
                f"Place 'WuXiaShadowRate.xlsx' inside data/raw/."
            )

        raw_df = pd.read_excel(excel_file, sheet_name=0)
        date_col = raw_df.columns[0]
        shadow_col = [c for c in raw_df.columns if "shadow" in str(c).lower()][
            0
        ]

        shadow_df = raw_df[[date_col, shadow_col]].dropna().copy()
        shadow_df[date_col] = pd.to_datetime(shadow_df[date_col])
        shadow_df = shadow_df.set_index(date_col)
        shadow_series = _resample_month_end(shadow_df[shadow_col]).rename(
            "WU_XIA_SHADOW"
        )

        # Combine Wu-Xia through its endpoint (Feb 2022) with actual EFFR
        combined_policy = shadow_series.combine_first(effr_m).rename(
            "SHADOW_RATE"
        )

        df_monthly = pd.concat([dgs2_m, combined_policy], axis=1).dropna()
        df_monthly["SHADOW_SPREAD"] = (
            df_monthly["DGS2_M"] - df_monthly["SHADOW_RATE"]
        )

        if start_date:
            df_monthly = df_monthly.loc[
                df_monthly.index >= pd.to_datetime(start_date)
            ]
        if end_date:
            df_monthly = df_monthly.loc[
                df_monthly.index <= pd.to_datetime(end_date)
            ]

        df_monthly.to_csv(self.processed_dir / "module2_monthly.csv")
        return df_monthly

    # =========================================================================
    # MODULE 3: Ancillary Liquidity & Repo Stress Monitor
    # =========================================================================
    def load_module3_liquidity(
        self,
        start_date: str = "2018-04-01",
        end_date: Optional[str] = None,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Constructs interbank repo funding and financial conditions monitors."""
        sofr = self.fetch_fred_series(
            "SOFR",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )

        try:
            iorb = self.fetch_fred_series(
                "IORB",
                start_date=start_date,
                end_date=end_date,
                force_reload=force_reload,
            )
        except Exception:
            iorb = pd.Series(dtype=float)

        try:
            ioer = self.fetch_fred_series(
                "IOER",
                start_date=start_date,
                end_date=end_date,
                force_reload=force_reload,
            )
        except Exception:
            ioer = pd.Series(dtype=float)

        fed_remuneration = iorb.combine_first(ioer)
        fed_remuneration.name = "FED_REMUNERATION"

        repo_df = pd.concat([sofr, fed_remuneration], axis=1).dropna()
        repo_df.columns = ["SOFR", "FED_REMUNERATION"]
        repo_df["SOFR_IORB_SPREAD_BPS"] = (
            repo_df["SOFR"] - repo_df["FED_REMUNERATION"]
        ) * 100

        nfci = self.fetch_fred_series(
            "NFCI",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )
        nfci.name = "NFCI"

        combined = repo_df.join(nfci, how="left")
        combined["NFCI"] = combined["NFCI"].ffill()

        combined.to_csv(self.processed_dir / "module3_liquidity.csv")
        return combined

    # =========================================================================
    # STEP 4: CONDITIONAL MACRO FILTERS (Anti Look-Ahead Bias Pipeline)
    # =========================================================================
    def load_conditional_macro_filters(
        self,
        start_date: str = "1994-01-01",
        end_date: Optional[str] = None,
        publication_lag_days: int = 35,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """Loads macroeconomic confirmation filters with reporting delay."""
        unrate = self.fetch_fred_series(
            "UNRATE",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )
        core_pce = self.fetch_fred_series(
            "PCEPILFE",
            start_date=start_date,
            end_date=end_date,
            force_reload=force_reload,
        )

        macro = pd.concat([unrate, core_pce], axis=1).dropna()
        macro.columns = ["UNRATE", "CORE_PCE"]

        macro["UNRATE_3MMA"] = macro["UNRATE"].rolling(window=3).mean()
        macro["UNRATE_12M_MIN"] = macro["UNRATE_3MMA"].rolling(window=12).min()
        macro["SAHM_INDICATOR"] = macro["UNRATE_3MMA"] - macro["UNRATE_12M_MIN"]
        macro["FLAG_SAHM_RECESSION"] = (
            macro["SAHM_INDICATOR"] >= 0.50
        ).astype(int)

        macro["CORE_PCE_YOY"] = macro["CORE_PCE"].pct_change(12) * 100
        macro["FLAG_INFLATION_ABOVE_2PCT"] = (
            macro["CORE_PCE_YOY"] > 2.0
        ).astype(int)

        macro_lagged = macro.copy()
        macro_lagged.index = macro_lagged.index + pd.Timedelta(
            days=publication_lag_days
        )

        macro_lagged.to_csv(self.processed_dir / "macro_filters_lagged.csv")
        return macro_lagged


if __name__ == "__main__":
    loader = MacroDataLoader()

    print(
        "--- Regenerating Module 1 (Daily Chained DFF/EFFR from 1994, Excluding ZLB) ---"
    )
    daily_data = loader.load_module1_daily(
        policy_rate="EFFR",
        start_date="1994-01-01",
        exclude_zlb=True,
        force_reload=True,
    )
    print(f"Module 1 full sample shape: {daily_data.shape}")
    print("\nHead (1994):")
    print(daily_data.head(3))
    print("\nTail (Present):")
    print(daily_data.tail(3))
