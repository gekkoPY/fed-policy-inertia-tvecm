"""
fed-policy-inertia-tvecm
Module: src/signal_evaluator.py

Cycle-Aware Dual-Engine Signal Evaluator & Walk-Forward Performance Engine.
Distinguishes between:
1. Early Warning Pivot Signals (HIT_EARLY_WARNING vs FALSE_POSITIVE).
2. Ongoing Cycle Persistence (IN_CYCLE_CONFIRMATION).
3. Active Unresolved Signals (ACTIVE_SIGNAL - Live monitoring).
"""

import os
from pathlib import Path
import sys
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name == "src" else CURRENT_DIR

if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from data_loader import MacroDataLoader
    from threshold_engine import ThresholdVECM
except ImportError:
    from src.data_loader import MacroDataLoader
    from src.threshold_engine import ThresholdVECM


class DualSignalEvaluator:
    """Institutional Walk-Forward and Signal Evaluation Engine with

    Cycle-Aware state tracking for Easing and Tightening regimes.
    """

    # Historical Easing Campaigns [start_date, end_date]
    EASING_CYCLES: List[Dict[str, str]] = [
        {"name": "1995 Mid-Cycle Easing", "start": "1995-07-06", "end": "1996-01-31"},
        {"name": "1998 LTCM Emergency", "start": "1998-09-29", "end": "1998-11-17"},
        {"name": "2001 Dot-Com Recession", "start": "2001-01-03", "end": "2003-06-25"},
        {"name": "2007-2008 GFC Easing", "start": "2007-09-18", "end": "2008-12-16"},
        {"name": "2019-2020 Mid-Cycle/Covid", "start": "2019-07-31", "end": "2020-03-15"},
        {"name": "2024 Normalization Easing", "start": "2024-09-18", "end": "2025-12-31"},
    ]

    # Historical Tightening Campaigns [start_date, end_date]
    TIGHTENING_CYCLES: List[Dict[str, str]] = [
        {"name": "1994 Greenspan Shock", "start": "1994-02-04", "end": "1995-02-01"},
        {"name": "1999-2000 Dot-Com Tightening", "start": "1999-06-30", "end": "2000-05-16"},
        {"name": "2004-2006 Measured Pace", "start": "2004-06-30", "end": "2006-06-29"},
        {"name": "2015-2018 Post-ZLB Liftoff", "start": "2015-12-16", "end": "2018-12-19"},
        {"name": "2022-2023 Post-Inflation Strenuous", "start": "2022-03-16", "end": "2023-07-26"},
    ]

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.loader = MacroDataLoader(data_dir=data_dir)
        csv_file = self.loader.processed_dir / "module1_daily.csv"

        if not csv_file.exists():
            self.df = self.loader.load_module1_daily(policy_rate="EFFR", start_date="1994-01-01")
        else:
            self.df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        self.df = self.df.sort_index()
        self.latest_sample_date = self.df.index[-1]

    def run_walk_forward(
        self,
        mode: str = "dovish",
        initial_window_days: int = 1250,
        step_days: int = 21,
        n_grid: int = 30,
        trim: float = 0.15,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """Executes Walk-Forward expanding window calibration for specified

        engine.
        """
        T_total = len(self.df)
        eval_indices = list(range(initial_window_days, T_total, step_days))
        records = []

        if verbose:
            print(f"Executing {mode.upper()} Walk-Forward across {len(eval_indices)} monthly steps...")

        for t_end in eval_indices:
            train_df = self.df.iloc[:t_end]
            next_t_end = min(t_end + step_days, T_total)
            test_df = self.df.iloc[t_end:next_t_end]

            model = ThresholdVECM(mode=mode, lags=1, trim=trim, n_grid=n_grid)
            model.fit(train_df)
            gamma_t = model.gamma

            for dt, row in test_df.iterrows():
                spread = row["SPREAD"]
                regime = 2 if (spread >= gamma_t if mode == "hawkish" else spread <= gamma_t) else 1
                records.append({
                    "DATE": dt,
                    "DGS2": row["DGS2"],
                    "POLICY_RATE": row["POLICY_RATE"],
                    "SPREAD": spread,
                    "GAMMA_T": gamma_t,
                    "REGIME_OOS": regime,
                })

        return pd.DataFrame(records).set_index("DATE")

    def evaluate_engine(
        self,
        oos_df: pd.DataFrame,
        mode: str = "dovish",
        min_persistence_days: int = 5,
        forward_window_days: int = 120,
    ) -> Dict:
        """Evaluates prediction metrics using Cycle-Aware state tracking."""
        own_cycles = self.EASING_CYCLES if mode == "dovish" else self.TIGHTENING_CYCLES
        opp_cycles = self.TIGHTENING_CYCLES if mode == "dovish" else self.EASING_CYCLES

        df = oos_df.sort_index().copy()
        is_r2 = (df["REGIME_OOS"] == 2).astype(int)
        diff = is_r2.diff().fillna(0)

        # Detect discrete signal entries
        signal_events = []
        for i in range(len(df) - min_persistence_days):
            if diff.iloc[i] == 1:
                if is_r2.iloc[i : i + min_persistence_days].sum() == min_persistence_days:
                    sig_date = df.index[i]
                    if not signal_events or (sig_date - signal_events[-1]).days > 45:
                        signal_events.append(sig_date)

        records = []
        hits = 0
        false_positives = 0
        in_cycle_count = 0
        active_signals_count = 0
        lead_times = []

        for s_date in signal_events:
            spread_val = float(df.loc[s_date, "SPREAD"])

            # 1. Check if signal occurs within an already active campaign of same direction
            in_own = next(
                (c["name"] for c in own_cycles if pd.to_datetime(c["start"]) <= s_date <= pd.to_datetime(c["end"])),
                None
            )
            if in_own:
                in_cycle_count += 1
                records.append({
                    "signal_date": s_date.strftime("%Y-%m-%d"),
                    "target_event": f"{in_own} [In Progress]",
                    "lead_time_b_days": np.nan,
                    "spread_at_signal": spread_val,
                    "classification": "IN_CYCLE_CONFIRMATION",
                })
                continue

            # 2. Check if signal occurs during an opposite campaign (e.g. curve steepening caused by rate cuts)
            in_opp = next(
                (c["name"] for c in opp_cycles if pd.to_datetime(c["start"]) <= s_date <= pd.to_datetime(c["end"])),
                None
            )
            if in_opp and mode == "hawkish":
                in_cycle_count += 1
                records.append({
                    "signal_date": s_date.strftime("%Y-%m-%d"),
                    "target_event": f"{in_opp} [Opposite Active]",
                    "lead_time_b_days": np.nan,
                    "spread_at_signal": spread_val,
                    "classification": "IN_OPPOSITE_CYCLE",
                })
                continue

            # 3. Check for impending future cycle start within forward window
            future_cycles = [c for c in own_cycles if pd.to_datetime(c["start"]) >= s_date]
            if future_cycles:
                nearest_c = future_cycles[0]
                n_start = pd.to_datetime(nearest_c["start"])
                b_days = len(self.df.loc[s_date:n_start]) - 1

                if 0 <= b_days <= forward_window_days:
                    hits += 1
                    lead_times.append(b_days)
                    records.append({
                        "signal_date": s_date.strftime("%Y-%m-%d"),
                        "target_event": nearest_c["name"],
                        "lead_time_b_days": int(b_days),
                        "spread_at_signal": spread_val,
                        "classification": "HIT_EARLY_WARNING",
                    })
                    continue
                else:
                    false_positives += 1
                    records.append({
                        "signal_date": s_date.strftime("%Y-%m-%d"),
                        "target_event": nearest_c["name"],
                        "lead_time_b_days": int(b_days),
                        "spread_at_signal": spread_val,
                        "classification": "FALSE_POSITIVE",
                    })
                    continue

            # 4. No future cycles on record -> Check if observation window is still active (Right-Censored)
            b_days_to_sample_end = len(self.df.loc[s_date:self.latest_sample_date]) - 1
            if b_days_to_sample_end <= forward_window_days:
                active_signals_count += 1
                records.append({
                    "signal_date": s_date.strftime("%Y-%m-%d"),
                    "target_event": "Pending FOMC Cycle Decision",
                    "lead_time_b_days": np.nan,
                    "spread_at_signal": spread_val,
                    "classification": "ACTIVE_SIGNAL",
                })
            else:
                false_positives += 1
                records.append({
                    "signal_date": s_date.strftime("%Y-%m-%d"),
                    "target_event": "None",
                    "lead_time_b_days": np.nan,
                    "spread_at_signal": spread_val,
                    "classification": "FALSE_POSITIVE",
                })

        # Net Early Warning Signals (excluding in-cycle and active signals)
        net_eval = hits + false_positives
        hit_rate = (hits / net_eval * 100.0) if net_eval > 0 else 0.0
        mean_lead = float(np.mean(lead_times)) if lead_times else 0.0

        return {
            "mode": mode,
            "total_triggers": len(signal_events),
            "in_cycle_confirmations": in_cycle_count,
            "active_live_signals": active_signals_count,
            "net_early_warnings_evaluated": net_eval,
            "hits": hits,
            "false_positives": false_positives,
            "hit_rate_pct": hit_rate,
            "mean_lead_time_days": mean_lead,
            "events_table": pd.DataFrame(records),
        }

    def print_executive_dashboard(self, dov_res: Dict, hawk_res: Dict) -> None:
        """Prints high-level desk dashboard comparing Dovish and Hawkish

        performance.
        """
        print("\n" + "=" * 90)
        print("          FEDERAL RESERVE ASYMMETRIC POLICY REACTION FUNCTION DASHBOARD")
        print("=" * 90)
        print(f"{'Performance Metric':<34} {'DOVISH ENGINE (Cuts)':<27} {'HAWKISH ENGINE (Hikes)':<27}")
        print("-" * 90)
        print(f"{'Target Event':<34} {'FOMC Easing Cycle Pivot':<27} {'FOMC Tightening Cycle':<27}")
        print(f"{'Regime 2 Identification':<34} {'Inversion (Spread <= g)':<27} {'Premium (Spread >= g)':<27}")
        print(f"{'Total Raw Triggers':<34} {dov_res['total_triggers']:<27} {hawk_res['total_triggers']:<27}")
        print(f"{'In-Cycle Confirmations':<34} {dov_res['in_cycle_confirmations']:<27} {hawk_res['in_cycle_confirmations']:<27}")
        print(f"{'Active Live Signals (Pending)':<34} {dov_res['active_live_signals']:<27} {hawk_res['active_live_signals']:<27}")
        print("-" * 90)
        print(f"{'Net Early Warnings Evaluated':<34} {dov_res['net_early_warnings_evaluated']:<27} {hawk_res['net_early_warnings_evaluated']:<27}")
        print(f"{'Confirmed Cycle Hits':<34} {dov_res['hits']:<27} {hawk_res['hits']:<27}")
        print(f"{'False Positives (Inertia Run)':<34} {dov_res['false_positives']:<27} {hawk_res['false_positives']:<27}")
        print(f"{'Predictive Hit Rate':<34} {dov_res['hit_rate_pct']:<26.1f}% {hawk_res['hit_rate_pct']:<26.1f}%")
        print(f"{'Mean Early Warning Lead Time':<34} {dov_res['mean_lead_time_days']:<22.1f} days {hawk_res['mean_lead_time_days']:<22.1f} days")
        print("=" * 90)

        print("\n>>> DOVISH SIGNALS AUDIT (Historical Inversion Triggers vs Rate Cuts):")
        print("-" * 90)
        print(dov_res["events_table"].to_string(index=False))

        print("\n>>> HAWKISH SIGNALS AUDIT (Historical Premium Triggers vs Rate Hikes):")
        print("-" * 90)
        print(hawk_res["events_table"].to_string(index=False))
        print("=" * 90)


if __name__ == "__main__":
    evaluator = DualSignalEvaluator()

    # 1. Run Dovish Walk-Forward
    dov_oos = evaluator.run_walk_forward(mode="dovish", initial_window_days=1250, step_days=21, verbose=True)
    dov_metrics = evaluator.evaluate_engine(dov_oos, mode="dovish", forward_window_days=120)

    # 2. Run Hawkish Walk-Forward
    hawk_oos = evaluator.run_walk_forward(mode="hawkish", initial_window_days=1250, step_days=21, verbose=True)
    hawk_metrics = evaluator.evaluate_engine(hawk_oos, mode="hawkish", forward_window_days=120)

    # 3. Print Executive Dashboard
    evaluator.print_executive_dashboard(dov_metrics, hawk_metrics)
