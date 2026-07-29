from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .records import EvaluationResult
from .reward_model import RewardEnsemble
from .schema_space import Plan, SchemaSpace
from .selector import SelectedPlan, quota_for, select_batch


Evaluator = Callable[[Plan, dict[str, Any]], float | dict[str, Any] | EvaluationResult | None]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class SearchPipeline:
    def __init__(self, config_path: str | Path, evaluator: Evaluator, *, resume: bool = False):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        schema_dir = Path(self.config["schema_dir"])
        if not schema_dir.is_absolute():
            schema_dir = (self.config_path.parent / schema_dir).resolve()
        self.space = SchemaSpace(schema_dir)
        self.evaluator = evaluator
        self.observations: list[tuple[Plan, float]] = []
        self.artifact_dir = Path(self.config.get("artifact_dir", "artifacts/search"))
        if not self.artifact_dir.is_absolute():
            self.artifact_dir = (self.config_path.parent / self.artifact_dir).resolve()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.plan_dir = self.artifact_dir / "plans"
        self.plan_dir.mkdir(exist_ok=True)
        self.buffer_path = self.artifact_dir / "reward_buffer.jsonl"
        self.plans_path = self.artifact_dir / "plans.jsonl"
        self.rounds_path = self.artifact_dir / "rounds.jsonl"
        self.events_path = self.artifact_dir / "events.jsonl"
        self._write_run_manifest()
        if resume and self.buffer_path.exists():
            self._load_buffer()

        batch_size = int(self.config["batch_size"])
        for stage in self.config["selector"]["schedule"]:
            total = int(stage["explore"]) + int(stage["exploit"]) + int(stage["mutate"])
            if total != batch_size:
                raise ValueError(f"Schedule stage {stage['name']} selects {total}, expected batch_size={batch_size}")

    def _write_run_manifest(self) -> None:
        path = self.artifact_dir / "run.json"
        if path.exists():
            return
        payload = {
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_name": self.config.get("run_name"),
            "batch_size": self.config["batch_size"],
            "candidate_pool_size": self.config["candidate_pool_size"],
            "seed": self.config["seed"],
            "periods": self.config.get("periods", []),
            "max_repairs": self.config.get("max_repairs"),
            "selector": self.config["selector"],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    @staticmethod
    def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=_json_default) + "\n")

    def _event(self, event: str, **fields: Any) -> None:
        self._append_jsonl(self.events_path, {
            "timestamp": datetime.now(timezone.utc).isoformat(), "event": event, **fields,
        })

    def _load_buffer(self) -> None:
        for line in self.buffer_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            plan = Plan(
                event=row["event"], context=row["context"], qualities=tuple(row["qualities"]),
                direction=row["direction"], output=row["output"],
            )
            self.observations.append((plan, float(row["reward"])))

    @staticmethod
    def _normalize_result(raw: float | dict[str, Any] | EvaluationResult | None) -> EvaluationResult:
        if isinstance(raw, EvaluationResult):
            result = raw
        elif isinstance(raw, dict):
            payload = dict(raw)
            result = EvaluationResult(**payload)
        elif raw is None:
            result = EvaluationResult(reward=0.0, status="failed", issues=["Evaluator returned None."])
        else:
            result = EvaluationResult(reward=float(raw))
        if not math.isfinite(float(result.reward)):
            return EvaluationResult(reward=0.0, status="failed", issues=["Evaluator returned a non-finite reward."])
        result.reward = float(result.reward) if result.status == "ok" else 0.0
        return result

    def _persist_plan(
        self, round_index: int, batch_index: int, selected: SelectedPlan,
        expanded: dict[str, Any], result: EvaluationResult,
    ) -> dict[str, Any]:
        plan_id = f"round_{round_index:04d}_plan_{batch_index:03d}"
        workspace = self.plan_dir / plan_id
        code_dir = workspace / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        copied_code_files = []
        for index, code in enumerate(result.code_versions):
            name = f"factor_r{index}.py"
            (code_dir / name).write_text(code, encoding="utf-8")
            copied_code_files.append(f"plans/{plan_id}/code/{name}")

        evaluation = result.to_dict()
        evaluation.pop("code_versions", None)
        if copied_code_files:
            evaluation["code_files"] = copied_code_files
        record = {
            "format_version": 1,
            "plan_id": plan_id,
            "round": round_index,
            "batch_index": batch_index,
            "plan_key": selected.plan.key,
            "plan": selected.plan.to_dict(),
            "schemas": expanded,
            "selection": selected.to_dict(),
            "evaluation": evaluation,
        }
        (workspace / "record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8"
        )
        self._append_jsonl(self.plans_path, record)
        self._append_jsonl(self.buffer_path, {
            **selected.plan.to_dict(), "round": round_index, "reward": result.reward,
        })
        return record

    def _persist_round(self, round_index: int, records: list[dict[str, Any]]) -> None:
        rewards = [float(row["evaluation"]["reward"]) for row in records]
        statuses = Counter(row["evaluation"]["status"] for row in records)
        source_rewards: dict[str, list[float]] = defaultdict(list)
        for row, reward in zip(records, rewards):
            source_rewards[row["selection"]["selection_reason"]].append(reward)
        summary = {
            "round": round_index,
            "plan_count": len(records),
            "status_counts": dict(statuses),
            "valid_rate": statuses.get("ok", 0) / len(records) if records else 0.0,
            "zero_reward_count": sum(reward == 0.0 for reward in rewards),
            "reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "reward_max": max(rewards, default=0.0),
            "cumulative_best": max(
                [reward for _, reward in self.observations] + rewards, default=0.0
            ),
            "by_selection_reason": {
                source: {"count": len(values), "reward_mean": sum(values) / len(values), "reward_max": max(values)}
                for source, values in source_rewards.items()
            },
        }
        self._append_jsonl(self.rounds_path, summary)

    def run_round(self, round_index: int) -> list[tuple[Plan, float]]:
        selector = self.config["selector"]
        quota = quota_for(len(self.observations), selector["schedule"])
        seen = {plan.key for plan, _ in self.observations}
        candidates = self.space.sample(
            self.config["candidate_pool_size"], seed=self.config["seed"] + round_index * 1009,
            quality_count_weights=selector["quality_count_weights"], excluded=seen,
        )
        model = None
        if quota.exploit and self.observations:
            model = RewardEnsemble(
                ensemble_size=selector["ensemble_size"],
                bootstrap_fraction=selector["bootstrap_fraction"],
                seed=self.config["seed"] + round_index,
            ).fit(self.observations)
        batch = select_batch(
            candidates, self.observations, quota=quota, model=model, space=self.space,
            seed=self.config["seed"] + round_index * 7919,
            mutation_weights=selector["mutation_weights"],
        )
        self._event("round_started", round=round_index, quota=vars(quota), candidate_count=len(candidates))
        results: list[tuple[Plan, float]] = []
        records = []
        for batch_index, selected in enumerate(batch):
            plan = selected.plan
            expanded = self.space.expand(plan)
            self._event(
                "plan_started", round=round_index, batch_index=batch_index,
                plan_key=plan.key, selection_reason=selected.selection_reason,
            )
            try:
                result = self._normalize_result(self.evaluator(plan, expanded))
            except Exception as exc:
                result = EvaluationResult(
                    reward=0.0, status="failed",
                    issues=[f"Evaluator failed: {type(exc).__name__}: {exc}"],
                )
            records.append(self._persist_plan(round_index, batch_index, selected, expanded, result))
            results.append((plan, result.reward))
            self._event(
                "plan_completed", round=round_index, batch_index=batch_index,
                plan_key=plan.key, status=result.status, reward=result.reward,
            )
        self.observations.extend(results)
        self._persist_round(round_index, records)
        self._event("round_completed", round=round_index, plan_count=len(records))
        return results
