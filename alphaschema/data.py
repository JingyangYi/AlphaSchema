from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "amount")


@dataclass(frozen=True)
class DataConfig:
    path: Path
    format: str = "parquet"
    layout: str = "one_file_per_symbol"
    date_column: str = "date"
    symbol_column: str = "symbol"
    columns: dict[str, str] = field(default_factory=dict)
    start_date: str | None = None
    end_date: str | None = None
    universe_path: Path | None = None
    universe_symbol_column: str = "symbol"
    required_fields: tuple[str, ...] = REQUIRED_FIELDS

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, base_dir: Path) -> "DataConfig":
        def resolve(raw: str | None) -> Path | None:
            if not raw:
                return None
            path = Path(os.path.expanduser(os.path.expandvars(raw)))
            return path if path.is_absolute() else (base_dir / path).resolve()

        if not payload.get("path"):
            raise ValueError("data.path is required")
        universe = payload.get("universe") or {}
        return cls(
            path=resolve(payload["path"]),
            format=str(payload.get("format", "parquet")).lower(),
            layout=str(payload.get("layout", "one_file_per_symbol")),
            date_column=str(payload.get("date_column", "date")),
            symbol_column=str(payload.get("symbol_column", "symbol")),
            columns={str(key): str(value) for key, value in payload.get("columns", {}).items()},
            start_date=payload.get("start_date"),
            end_date=payload.get("end_date"),
            universe_path=resolve(universe.get("path")),
            universe_symbol_column=str(universe.get("symbol_column", "symbol")),
            required_fields=tuple(payload.get("required_fields", REQUIRED_FIELDS)),
        )


class MarketData:
    """Load user bars into the panel contract consumed by generated factors."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._panel: dict[str, pd.DataFrame] | None = None

    @staticmethod
    def _read(path: Path, format_name: str) -> pd.DataFrame:
        if format_name == "parquet":
            return pd.read_parquet(path)
        if format_name == "csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data format: {format_name}; expected parquet or csv")

    def _universe(self) -> set[str] | None:
        path = self.config.universe_path
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Universe file does not exist: {path}")
        frame = self._read(path, path.suffix.lower().lstrip("."))
        column = self.config.universe_symbol_column
        if column not in frame:
            raise ValueError(f"Universe file has no {column!r} column")
        return set(frame[column].dropna().astype(str))

    def _standardize(self, frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
        rename = {source: canonical for canonical, source in self.config.columns.items()}
        frame = frame.rename(columns=rename).copy()
        date_column = self.config.date_column
        if date_column in frame.columns:
            frame.index = pd.to_datetime(frame.pop(date_column), errors="coerce")
        else:
            frame.index = pd.to_datetime(frame.index, errors="coerce")
        frame = frame.loc[~frame.index.isna()].sort_index()
        frame = frame.loc[~frame.index.duplicated(keep="last")]
        if self.config.start_date:
            frame = frame.loc[frame.index >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            frame = frame.loc[frame.index <= pd.Timestamp(self.config.end_date)]
        if "vwap" not in frame and {"amount", "volume"}.issubset(frame.columns):
            volume = pd.to_numeric(frame["volume"], errors="coerce").replace(0, float("nan"))
            frame["vwap"] = pd.to_numeric(frame["amount"], errors="coerce") / volume
        missing = [field for field in self.config.required_fields if field not in frame]
        if missing:
            raise ValueError(f"{symbol}: missing required standardized fields: {', '.join(missing)}")
        if frame.empty:
            return frame
        for column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame

    def load(self) -> dict[str, pd.DataFrame]:
        if self._panel is not None:
            return self._panel
        path = self.config.path
        if not path.exists():
            raise FileNotFoundError(f"Market data path does not exist: {path}")
        universe = self._universe()
        panel: dict[str, pd.DataFrame] = {}
        if self.config.layout == "long_table":
            frame = self._read(path, self.config.format)
            symbol_column = self.config.symbol_column
            if symbol_column not in frame:
                raise ValueError(f"Long table has no {symbol_column!r} column")
            for value, part in frame.groupby(symbol_column, sort=True):
                symbol = str(value)
                if universe is None or symbol in universe:
                    standardized = self._standardize(part.drop(columns=[symbol_column]), symbol=symbol)
                    if not standardized.empty:
                        panel[symbol] = standardized
        elif self.config.layout == "one_file_per_symbol":
            if not path.is_dir():
                raise ValueError("one_file_per_symbol layout requires a directory")
            suffix = ".parquet" if self.config.format == "parquet" else ".csv"
            for file_path in sorted(path.glob(f"*{suffix}")):
                symbol = file_path.stem
                if universe is None or symbol in universe:
                    standardized = self._standardize(self._read(file_path, self.config.format), symbol=symbol)
                    if not standardized.empty:
                        panel[symbol] = standardized
        else:
            raise ValueError("layout must be long_table or one_file_per_symbol")
        if not panel:
            raise ValueError("No symbols were loaded; check the path, layout, and universe")
        self._panel = panel
        return panel

    def close_matrix(self) -> pd.DataFrame:
        return pd.concat({symbol: bars["close"] for symbol, bars in self.load().items()}, axis=1).sort_index()

    def manifest(self) -> dict[str, Any]:
        panel = self.load()
        dates = pd.DatetimeIndex([])
        rows = 0
        content_hash = hashlib.sha256()
        for symbol, frame in sorted(panel.items()):
            dates = dates.union(frame.index)
            rows += len(frame)
            content_hash.update(symbol.encode())
            content_hash.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
        signature = {
            "symbols": sorted(panel), "rows": rows,
            "start": str(dates.min()), "end": str(dates.max()),
            "fields": sorted(set.intersection(*(set(frame.columns) for frame in panel.values()))),
        }
        digest = content_hash.hexdigest()
        return {**signature, "symbol_count": len(panel), "fingerprint": digest}
