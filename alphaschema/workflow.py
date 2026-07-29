from __future__ import annotations

import json
import hashlib
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .code_agent import CodeAgent
from .records import EvaluationResult
from .schema_space import Plan, SchemaSpace
from .validation import static_issues


class FactorBackend(Protocol):
    """Data-specific operations intentionally kept outside the search core."""

    def materialize(self, code_path: Path, periods: list[int], output_dir: Path) -> list[Path]: ...
    def leakage_issues(self, factor_paths: list[Path]) -> list[str]: ...
    def backtest(self, factor_paths: list[Path], output_dir: Path) -> list[dict[str, Any]]: ...
    def reward(self, metrics: list[dict[str, Any]]) -> float | dict[str, Any]: ...


@dataclass
class RealizationState:
    plan_key: str
    status: str = "pending"
    repair_attempts: int = 0
    code_path: str | None = None
    code_files: list[str] = field(default_factory=list)
    code_versions: list[str] = field(default_factory=list, repr=False)
    factor_paths: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    reward_components: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float | None = None


class FactorWorkflow:
    """Generate, validate, repair, materialize, backtest, and score one plan."""

    def __init__(
        self, *, space: SchemaSpace, agent: CodeAgent, backend: FactorBackend,
        prompt_dir: str | Path, run_dir: str | Path, periods: list[int], max_repairs: int = 1,
    ):
        self.space = space
        self.agent = agent
        self.backend = backend
        self.prompt_dir = Path(prompt_dir)
        self.run_dir = Path(run_dir)
        self.periods = periods
        self.max_repairs = max_repairs

    def _save(self, state: RealizationState, workspace: Path) -> None:
        payload = asdict(state)
        payload.pop("code_versions", None)
        (workspace / "state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def __call__(self, plan: Plan, expanded_plan: dict[str, Any]) -> EvaluationResult:
        started = time.perf_counter()
        workspace = self.run_dir / f"plan_{hashlib.sha256(plan.key.encode()).hexdigest()[:16]}"
        code_dir = workspace / "code"
        factor_dir = workspace / "factors"
        backtest_dir = workspace / "backtest"
        for path in (code_dir, factor_dir, backtest_dir):
            path.mkdir(parents=True, exist_ok=True)
        state = RealizationState(plan_key=plan.key)
        try:
            code = self.agent.generate(
                expanded_plan,
                system_prompt=self.prompt_dir / "implementation_system.md",
                user_prompt=self.prompt_dir / "implementation_user.md",
            )
        except Exception as exc:
            state.status = "failed"
            state.issues = [f"Generation failed: {type(exc).__name__}: {exc}"]
            state.elapsed_seconds = time.perf_counter() - started
            self._save(state, workspace)
            return self._result(state)
        state.events.append("generated")

        while True:
            code_path = code_dir / f"factor_r{state.repair_attempts}.py"
            code_path.write_text(code, encoding="utf-8")
            relative_code_path = str(code_path.relative_to(workspace))
            state.code_path = relative_code_path
            state.code_files.append(relative_code_path)
            state.code_versions.append(code)
            issues = static_issues(code)
            if not issues:
                try:
                    paths = self.backend.materialize(code_path, self.periods, factor_dir)
                    if not paths:
                        issues.append("Materialization produced no factor files.")
                    else:
                        state.factor_paths = [self._relative_or_name(path, workspace) for path in paths]
                        for path in paths:
                            frame = pd.read_pickle(path)
                            values = frame.to_numpy(dtype=float)
                            if values.size == 0 or not np.isfinite(values).any():
                                issues.append(f"Empty or non-finite factor output: {path.name}")
                        issues.extend(self.backend.leakage_issues(paths))
                except Exception as exc:
                    issues.append(f"Materialization failed: {type(exc).__name__}: {exc}")
            if not issues:
                try:
                    state.metrics = self.backend.backtest(paths, backtest_dir)
                    reward_output = self.backend.reward(state.metrics)
                    if isinstance(reward_output, dict):
                        state.reward = float(reward_output["reward"])
                        state.reward_components = {
                            key: value for key, value in reward_output.items() if key != "reward"
                        }
                    else:
                        state.reward = float(reward_output)
                    if not np.isfinite(state.reward):
                        raise ValueError("reward is not finite")
                    state.status = "ok"
                    state.issues = []
                    state.events.append("backtested")
                    state.elapsed_seconds = time.perf_counter() - started
                    self._save(state, workspace)
                    return self._result(state)
                except Exception as exc:
                    issues.append(f"Backtest failed: {type(exc).__name__}: {exc}")
            state.issues = issues
            if state.repair_attempts >= self.max_repairs:
                state.status = "failed"
                state.reward = 0.0
                state.elapsed_seconds = time.perf_counter() - started
                self._save(state, workspace)
                return self._result(state)
            repair_record = {
                "attempt": state.repair_attempts + 1,
                "source_code": relative_code_path,
                "issues": list(issues),
            }
            state.repair_history.append(repair_record)
            self._save(state, workspace)
            try:
                code = self.agent.repair(
                    expanded_plan, code, issues,
                    system_prompt=self.prompt_dir / "repair_system.md",
                    user_prompt=self.prompt_dir / "repair_user.md",
                )
            except Exception as exc:
                message = f"Repair failed: {type(exc).__name__}: {exc}"
                repair_record["error"] = message
                state.status = "failed"
                state.issues = [*issues, message]
                state.reward = 0.0
                state.elapsed_seconds = time.perf_counter() - started
                self._save(state, workspace)
                return self._result(state)
            state.repair_attempts += 1
            state.events.append("repaired")

    @staticmethod
    def _relative_or_name(path: Path, workspace: Path) -> str:
        try:
            return str(path.resolve().relative_to(workspace.resolve()))
        except ValueError:
            return path.name

    @staticmethod
    def _result(state: RealizationState) -> EvaluationResult:
        return EvaluationResult(
            reward=state.reward,
            status=state.status,
            metrics=state.metrics,
            reward_components=state.reward_components,
            code_files=state.code_files,
            code_versions=state.code_versions,
            factor_files=state.factor_paths,
            issues=state.issues,
            events=state.events,
            repair_attempts=state.repair_attempts,
            repair_history=state.repair_history,
            elapsed_seconds=state.elapsed_seconds,
        )
