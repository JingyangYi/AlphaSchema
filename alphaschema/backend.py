from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import MarketData


class PanelFactorBackend:
    """Reference daily cross-sectional backend for user-supplied bar data."""

    def __init__(
        self, data: MarketData, *, forward_horizon: int = 5, min_assets: int = 20,
        alpha: float = 10.0, beta: float = 1.0, lag_penalty: float = 2.0,
        leakage_rtol: float = 1e-9, leakage_atol: float = 1e-11,
    ):
        self.data = data
        self.forward_horizon = int(forward_horizon)
        self.min_assets = int(min_assets)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lag_penalty = float(lag_penalty)
        self.leakage_rtol = float(leakage_rtol)
        self.leakage_atol = float(leakage_atol)
        self._realizations: dict[Path, tuple[Callable[..., Any], int]] = {}

    @staticmethod
    def _import_factor(path: Path) -> Callable[..., Any]:
        name = f"alphaschema_factor_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]}"
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import generated factor: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        functions = [value for key, value in vars(module).items() if key.startswith("F_") and callable(value)]
        if not functions:
            raise ValueError("Generated module has no callable F_* factor function")
        return functions[0]

    def _normalize(self, value: Any, panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Factor function must return a pandas DataFrame")
        index = pd.DatetimeIndex([])
        for bars in panel.values():
            index = index.union(bars.index)
        symbols = sorted(panel)
        result = value.copy()
        result.index = pd.to_datetime(result.index)
        result = result.reindex(index=index.sort_values(), columns=symbols)
        return result.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    def materialize(self, code_path: Path, periods: list[int], output_dir: Path) -> list[Path]:
        function = self._import_factor(code_path)
        panel = self.data.load()
        paths = []
        for period in periods:
            factor = self._normalize(function(panel, period=int(period)), panel)
            path = output_dir / f"factor_p{int(period)}.pkl"
            factor.to_pickle(path)
            self._realizations[path.resolve()] = (function, int(period))
            paths.append(path)
        return paths

    def leakage_issues(self, factor_paths: list[Path]) -> list[str]:
        panel = self.data.load()
        dates = self.data.close_matrix().index
        if len(dates) < 20:
            return ["Too few dates for prefix-invariance leakage testing."]
        cutoff = dates[max(10, int(len(dates) * 0.8)) - 1]
        prefix_panel = {symbol: frame.loc[:cutoff].copy() for symbol, frame in panel.items()}
        issues = []
        for path in factor_paths:
            function, period = self._realizations[path.resolve()]
            full = pd.read_pickle(path).loc[:cutoff]
            prefix = self._normalize(function(prefix_panel, period=period), prefix_panel)
            prefix = prefix.reindex(index=full.index, columns=full.columns)
            if not full.notna().equals(prefix.notna()):
                issues.append(f"Prefix-invariance failed for {path.name}; output availability depends on future data.")
                continue
            comparable = full.notna() & prefix.notna()
            if comparable.to_numpy().any():
                left = full.where(comparable).to_numpy(dtype=float)
                right = prefix.where(comparable).to_numpy(dtype=float)
                if not np.allclose(left, right, rtol=self.leakage_rtol, atol=self.leakage_atol, equal_nan=True):
                    issues.append(f"Prefix-invariance failed for {path.name}; output may use future data.")
        return issues

    def _daily_correlation(self, signal: pd.DataFrame, target: pd.DataFrame, *, rank: bool) -> pd.Series:
        values = []
        dates = []
        for date in signal.index.intersection(target.index):
            pair = pd.concat([signal.loc[date], target.loc[date]], axis=1).dropna()
            if len(pair) < self.min_assets or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
                continue
            values.append(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman" if rank else "pearson"))
            dates.append(date)
        return pd.Series(values, index=dates, dtype=float).dropna()

    @staticmethod
    def _mean_ir(series: pd.Series) -> tuple[float, float]:
        if series.empty:
            return float("nan"), float("nan")
        average = float(series.mean())
        std = float(series.std(ddof=1))
        return average, average / std if std > 0 else float("nan")

    def backtest(self, factor_paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:
        close = self.data.close_matrix()
        forward = close.shift(-self.forward_horizon).div(close).sub(1.0)
        rows = []
        for path in factor_paths:
            signal = pd.read_pickle(path).reindex(index=close.index, columns=close.columns)
            ic, icir = self._mean_ir(self._daily_correlation(signal, forward, rank=False))
            daily_rank_ic = self._daily_correlation(signal, forward, rank=True)
            rank_ic, rank_icir = self._mean_ir(daily_rank_ic)
            lagged = signal.shift(1)
            lag_rank_ic, lag_rank_icir = self._mean_ir(self._daily_correlation(lagged, forward, rank=True))
            lag1_gap = max(0.0, rank_ic - lag_rank_ic) if np.isfinite(rank_ic) and np.isfinite(lag_rank_ic) else float("nan")
            ranks = signal.rank(axis=1, pct=True)
            turnover = float(ranks.diff().abs().mean(axis=1).mean())
            period = self._realizations[path.resolve()][1]
            rows.append({
                "period": period, "forward_horizon": self.forward_horizon,
                "ic": ic, "icir": icir, "rank_ic": rank_ic, "rank_icir": rank_icir,
                "lag1_rank_ic": lag_rank_ic, "lag1_rank_icir": lag_rank_icir,
                "lag1_gap": lag1_gap, "mean_rank_turnover": turnover,
                "observation_days": int(daily_rank_ic.size),
            })
        pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)
        return rows

    def reward(self, metrics: list[dict[str, Any]]) -> dict[str, Any]:
        scored = []
        for row in metrics:
            values = [row.get("rank_ic"), row.get("rank_icir"), row.get("lag1_gap")]
            if not all(np.isfinite(float(value)) for value in values):
                continue
            score = self.alpha * float(values[0]) + self.beta * float(values[1]) - self.lag_penalty * float(values[2])
            scored.append((score, row))
        if not scored:
            raise ValueError("No realization has sufficient finite metrics")
        score, best = max(scored, key=lambda item: item[0])
        return {
            "reward": score, "best_period": best["period"],
            "rank_ic_term": self.alpha * best["rank_ic"],
            "rank_icir_term": self.beta * best["rank_icir"],
            "lag1_penalty": -self.lag_penalty * best["lag1_gap"],
            "formula": "alpha*rank_ic + beta*rank_icir - lambda*max(0, rank_ic-lag1_rank_ic)",
        }
