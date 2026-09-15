"""
fed-policy-inertia-tvecm
Module: src/threshold_engine.py

Core Econometric Engine: Hansen & Seo (2002) Threshold Vector Error Correction Model (TVECM).
Supports Directional Regimes:
- 'dovish'  : Restricted to yield curve inversion (gamma <= 0) for rate cuts.
- 'hawkish' : Restricted to yield curve premium (gamma >= 0) for rate hikes.
- 'unconstrained': Pure mathematical SSR minimization over the full empirical distribution.
"""

import os
from pathlib import Path
import sys
from typing import Optional, Union, Dict, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Ensure Python can resolve modules regardless of working directory
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


class ThresholdVECM:
    """Hansen & Seo (2002) Threshold Vector Error Correction Model (TVECM)

    with directional asymmetric regime identification.
    """

    def __init__(
        self,
        mode: str = "dovish",  # 'dovish', 'hawkish', or 'unconstrained'
        lags: int = 1,
        trim: float = 0.15,
        n_grid: int = 40,
        criterion: str = "ssr",
    ) -> None:
        self.mode = mode.lower()
        if self.mode not in ["dovish", "hawkish", "unconstrained"]:
            raise ValueError("Mode must be 'dovish', 'hawkish', or 'unconstrained'.")

        self.lags = lags
        self.trim = trim
        self.n_grid = n_grid
        self.criterion = criterion.lower()

        # Model Estimation Results
        self.gamma: Optional[float] = None
        self.gamma_quantile: Optional[float] = None
        self.sup_lr_stat: Optional[float] = None
        self.p_value_bootstrap: Optional[float] = None
        self.boot_stats: Optional[np.ndarray] = None
        self.regime_series: Optional[pd.Series] = None
        self.is_fitted: bool = False

        self.results_regime1: Optional[Dict] = None
        self.results_regime2: Optional[Dict] = None
        self.results_linear: Optional[Dict] = None

    def _build_system(
        self,
        df: pd.DataFrame,
        y1_col: str = "DGS2",
        y2_col: str = "POLICY_RATE",
        spread_col: Optional[str] = "SPREAD",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, list]:
        Y = df[[y1_col, y2_col]].values
        dY = np.diff(Y, axis=0)

        if spread_col and spread_col in df.columns:
            spread = df[spread_col].values
        else:
            spread = Y[:, 0] - Y[:, 1]

        w_lag = spread[:-1]
        T_full = len(dY)

        regressors = [np.ones((T_full - self.lags, 1)), w_lag[self.lags :, np.newaxis]]
        for i in range(1, self.lags + 1):
            regressors.append(dY[self.lags - i : -i])

        X = np.hstack(regressors)
        Y_eff = dY[self.lags :]
        w_eff = w_lag[self.lags :]
        eff_dates = df.index[1 + self.lags :]

        col_names = ["const", "ECT_Spread_lag1"]
        for i in range(1, self.lags + 1):
            col_names.extend([f"d_{y1_col}_lag{i}", f"d_{y2_col}_lag{i}"])

        return Y_eff, X, w_eff, eff_dates, col_names

    def _get_candidates(self, w_eff: np.ndarray) -> np.ndarray:
        """Determines quantile search bounds based on macro regime mode."""
        if self.mode == "dovish":
            # Focus strictly on curve inversion (spread <= 0.0)
            sub_w = w_eff[w_eff <= 0.0]
            if len(sub_w) < 40:
                sub_w = w_eff
        elif self.mode == "hawkish":
            # Focus strictly on positive yield premium (spread >= 0.0)
            sub_w = w_eff[w_eff >= 0.0]
            if len(sub_w) < 40:
                sub_w = w_eff
        else:
            sub_w = w_eff

        return np.unique(
            np.quantile(sub_w, np.linspace(self.trim, 1.0 - self.trim, self.n_grid))
        )

    def fit(
        self,
        df: pd.DataFrame,
        y1_col: str = "DGS2",
        y2_col: str = "POLICY_RATE",
        spread_col: Optional[str] = "SPREAD",
    ) -> "ThresholdVECM":
        self.df = df
        self.y1_col = y1_col
        self.y2_col = y2_col
        self.spread_col = spread_col

        Y_eff, X, w_eff, eff_dates, col_names = self._build_system(df, y1_col, y2_col, spread_col)
        self.Y_eff = Y_eff
        self.X = X
        self.w_eff = w_eff
        self.eff_dates = eff_dates
        self.col_names = col_names
        self.T, self.K_reg = X.shape

        # 1. Linear VECM baseline (H0)
        self.b_linear = np.linalg.lstsq(X, Y_eff, rcond=None)[0]
        self.res_linear = Y_eff - X @ self.b_linear
        self.cov_linear = (self.res_linear.T @ self.res_linear) / self.T
        self.log_det_linear = float(np.log(np.maximum(np.linalg.det(self.cov_linear), 1e-15)))

        # 2. Threshold Grid Search
        candidates = self._get_candidates(w_eff)
        min_obs = max(self.K_reg + 5, int(self.trim * len(w_eff if self.mode == "unconstrained" else candidates)))

        best_score = np.inf
        best_gamma = None

        for g in candidates:
            if self.mode == "hawkish":
                idx2 = w_eff >= g  # Regime 2: Hawkish Capitulation (Hikes)
                idx1 = ~idx2      # Regime 1: Inertia
            else:
                idx2 = w_eff <= g  # Regime 2: Dovish Capitulation (Cuts)
                idx1 = ~idx2      # Regime 1: Inertia

            n1, n2 = np.sum(idx1), np.sum(idx2)
            if n1 < min_obs or n2 < min_obs:
                continue

            X1, Y1 = X[idx1], Y_eff[idx1]
            X2, Y2 = X[idx2], Y_eff[idx2]

            b1 = np.linalg.lstsq(X1, Y1, rcond=None)[0]
            b2 = np.linalg.lstsq(X2, Y2, rcond=None)[0]

            e1 = Y1 - X1 @ b1
            e2 = Y2 - X2 @ b2
            cov_g = (e1.T @ e1 + e2.T @ e2) / self.T

            score = float(np.trace(cov_g)) if self.criterion == "ssr" else float(np.log(np.maximum(np.linalg.det(cov_g), 1e-15)))

            if score < best_score:
                best_score = score
                best_gamma = g

        if best_gamma is None:
            # Fallback to median if constraints are too tight
            best_gamma = float(np.median(candidates))

        self.gamma = float(best_gamma)
        self.gamma_quantile = float(np.mean(w_eff <= self.gamma))

        # 3. Parameter Estimation at optimal gamma
        if self.mode == "hawkish":
            self.idx2 = w_eff >= self.gamma
            self.idx1 = ~self.idx2
        else:
            self.idx2 = w_eff <= self.gamma
            self.idx1 = ~self.idx2

        self.results_regime1 = self._estimate_ols(X[self.idx1], Y_eff[self.idx1])
        self.results_regime2 = self._estimate_ols(X[self.idx2], Y_eff[self.idx2])
        self.results_linear = self._estimate_ols(X, Y_eff)

        e_all = np.empty_like(Y_eff)
        e_all[self.idx1] = self.results_regime1["residuals"]
        e_all[self.idx2] = self.results_regime2["residuals"]
        self.cov_tvecm = (e_all.T @ e_all) / self.T
        self.log_det_tvecm = float(np.log(np.maximum(np.linalg.det(self.cov_tvecm), 1e-15)))

        self.sup_lr_stat = max(0.0, float(self.T * (self.log_det_linear - self.log_det_tvecm)))
        regimes = np.where(self.idx2, 2, 1)
        self.regime_series = pd.Series(regimes, index=self.eff_dates, name="TVECM_REGIME")
        self.is_fitted = True
        return self

    def _estimate_ols(self, X_sub: np.ndarray, Y_sub: np.ndarray) -> Dict:
        n, k = X_sub.shape
        b = np.linalg.lstsq(X_sub, Y_sub, rcond=None)[0]
        res = Y_sub - X_sub @ b
        dof = max(n - k, 1)
        cov = (res.T @ res) / dof

        try:
            xtx_inv = np.linalg.inv(X_sub.T @ X_sub)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(X_sub.T @ X_sub)

        se = np.zeros_like(b)
        t_stat = np.zeros_like(b)
        p_val = np.zeros_like(b)

        for j in range(2):
            var_b = cov[j, j] * xtx_inv
            diag_var = np.maximum(np.diag(var_b), 1e-14)
            se[:, j] = np.sqrt(diag_var)
            t_stat[:, j] = b[:, j] / se[:, j]
            p_val[:, j] = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat[:, j]), df=dof))

        return {
            "n_obs": n,
            "coef": b,
            "se": se,
            "t_stat": t_stat,
            "p_val": p_val,
            "cov": cov,
            "residuals": res,
        }

    def bootstrap_sup_lm_test(
        self,
        n_boot: int = 200,
        random_state: int = 42,
        verbose: bool = True,
    ) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")

        if verbose:
            print(f"[{self.mode.upper()} ENGINE] Running Hansen & Seo Bootstrap Sup-LM Test ({n_boot} reps)...")

        rng = np.random.RandomState(random_state)
        u_centered = self.res_linear - np.mean(self.res_linear, axis=0)
        candidates = self._get_candidates(self.w_eff)
        min_obs = max(self.K_reg + 5, int(self.trim * len(candidates)))

        boot_stats = np.zeros(n_boot)

        for b in range(n_boot):
            boot_idx = rng.randint(0, self.T, size=self.T)
            u_boot = u_centered[boot_idx]
            Y_boot = self.X @ self.b_linear + u_boot

            b_lin_b = np.linalg.lstsq(self.X, Y_boot, rcond=None)[0]
            e_lin_b = Y_boot - self.X @ b_lin_b
            cov_lin_b = (e_lin_b.T @ e_lin_b) / self.T
            log_det_lin_b = np.log(np.maximum(np.linalg.det(cov_lin_b), 1e-15))

            best_score_b = np.inf
            best_cov_b = None

            for g in candidates:
                if self.mode == "hawkish":
                    idx2 = self.w_eff >= g
                    idx1 = ~idx2
                else:
                    idx2 = self.w_eff <= g
                    idx1 = ~idx2

                if np.sum(idx1) < min_obs or np.sum(idx2) < min_obs:
                    continue

                X1, Y1 = self.X[idx1], Y_boot[idx1]
                X2, Y2 = self.X[idx2], Y_boot[idx2]

                b1 = np.linalg.lstsq(X1, Y1, rcond=None)[0]
                b2 = np.linalg.lstsq(X2, Y2, rcond=None)[0]

                e1 = Y1 - X1 @ b1
                e2 = Y2 - X2 @ b2
                cov_g_b = (e1.T @ e1 + e2.T @ e2) / self.T

                score = float(np.trace(cov_g_b)) if self.criterion == "ssr" else float(np.log(np.maximum(np.linalg.det(cov_g_b), 1e-15)))

                if score < best_score_b:
                    best_score_b = score
                    best_cov_b = cov_g_b

            if best_cov_b is not None:
                log_det_tvecm_b = np.log(np.maximum(np.linalg.det(best_cov_b), 1e-15))
                boot_stats[b] = max(0.0, float(self.T * (log_det_lin_b - log_det_tvecm_b)))
            else:
                boot_stats[b] = 0.0

            if verbose and (b + 1) % 100 == 0:
                print(f"  > Completed {b + 1}/{n_boot} replications...")

        self.p_value_bootstrap = float(np.mean(boot_stats >= self.sup_lr_stat))
        self.boot_stats = boot_stats
        return self.p_value_bootstrap

    def predict_regime(
        self,
        current_spread: Union[float, np.ndarray, pd.Series],
    ) -> Union[int, np.ndarray, pd.Series]:
        if self.gamma is None:
            raise ValueError("Model must be fitted first.")
        if self.mode == "hawkish":
            cond = (current_spread >= self.gamma)
        else:
            cond = (current_spread <= self.gamma)

        if isinstance(current_spread, (pd.Series, np.ndarray)):
            return np.where(cond, 2, 1)
        return 2 if cond else 1

    def summary(self) -> str:
        if not self.is_fitted:
            raise ValueError("Model must be fitted.")

        lines = []
        lines.append("=" * 86)
        lines.append(f"       TVECM REPORT - {self.mode.upper()} REGIME ENGINE")
        lines.append("=" * 86)
        lines.append(f"{'Target Orientation':<30}: {'Dovish Capitulation (Rate Cuts)' if self.mode == 'dovish' else 'Hawkish Capitulation (Rate Hikes)'}")
        lines.append(f"{'Estimated Threshold (gamma)':<30}: {self.gamma:+.4f} ({self.gamma * 100:+.1f} bps)")
        lines.append(f"{'Total Sample Size (T)':<30}: {self.T} (Regime 1: {self.results_regime1['n_obs']}, Regime 2: {self.results_regime2['n_obs']})")
        lines.append("-" * 86)
        lines.append("TEST FOR NON-LINEARITY (H0: Linear VECM vs H1: TVECM):")
        lines.append(f"  Sup-LR Statistic            : {self.sup_lr_stat:.4f}")
        if self.p_value_bootstrap is not None:
            lines.append(f"  Bootstrap Empirical p-value : {self.p_value_bootstrap:.4f}")
            verdict = "REJECT H0 (p < 0.05)" if self.p_value_bootstrap < 0.05 else "FAIL TO REJECT H0"
            lines.append(f"  Econometric Verdict         : {verdict}")
        lines.append("=" * 86)
        return "\n".join(lines)


if __name__ == "__main__":
    loader = MacroDataLoader()
    df_daily = pd.read_csv(loader.processed_dir / "module1_daily.csv", index_col=0, parse_dates=True)

    print("--- 1. Testing DOVISH Engine (Rate Cut Target) ---")
    dov_model = ThresholdVECM(mode="dovish", lags=1, trim=0.15)
    dov_model.fit(df_daily)
    dov_model.bootstrap_sup_lm_test(n_boot=100, verbose=True)
    print("\n" + dov_model.summary())

    print("\n--- 2. Testing HAWKISH Engine (Rate Hike Target) ---")
    hawk_model = ThresholdVECM(mode="hawkish", lags=1, trim=0.15)
    hawk_model.fit(df_daily)
    hawk_model.bootstrap_sup_lm_test(n_boot=100, verbose=True)
    print("\n" + hawk_model.summary())
