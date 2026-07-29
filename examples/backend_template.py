from __future__ import annotations

from pathlib import Path
from typing import Any


class UserFactorBackend:
    """Implement these four methods for the local market-data stack."""

    def materialize(self, code_path: Path, periods: list[int], output_dir: Path) -> list[Path]:
        """Run generated code on market bars and return factor DataFrame pickle paths."""
        raise NotImplementedError

    def leakage_issues(self, factor_paths: list[Path]) -> list[str]:
        """Run prefix-invariance or another numerical look-ahead check."""
        raise NotImplementedError

    def backtest(self, factor_paths: list[Path], output_dir: Path) -> list[dict[str, Any]]:
        """Evaluate fast and slow realizations and return their metric records."""
        raise NotImplementedError

    def reward(self, metrics: list[dict[str, Any]]) -> float:
        """Convert metrics to scalar rewards and return the best realization."""
        raise NotImplementedError
