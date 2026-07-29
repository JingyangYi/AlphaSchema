from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CATEGORIES = ("event", "context", "quality", "direction", "output")


@dataclass(frozen=True)
class Plan:
    event: str
    context: str
    qualities: tuple[str, ...]
    direction: str
    output: str

    @property
    def key(self) -> str:
        return "|".join((self.event, self.context, *self.qualities, self.direction, self.output))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "qualities": list(self.qualities), "plan_key": self.key}


@dataclass(frozen=True)
class MutationResult:
    plan: Plan
    operation: str
    changed_components: dict[str, dict[str, Any]]


class SchemaSpace:
    def __init__(self, schema_dir: str | Path):
        self.schema_dir = Path(schema_dir)
        self.records: dict[str, dict[str, Any]] = {}
        self.ids: dict[str, list[str]] = {}
        for category in CATEGORIES:
            payload = json.loads((self.schema_dir / f"{category}.json").read_text(encoding="utf-8"))
            enabled = [
                row for row in payload["schemas"]
                if not row.get("disabled", False) and row.get("enabled", True) is not False
            ]
            self.ids[category] = [row["schema_id"] for row in enabled]
            self.records.update({row["schema_id"]: row for row in enabled})
        if any(not self.ids[category] for category in CATEGORIES):
            raise ValueError("Every schema category must contain at least one enabled record")

    @staticmethod
    def _quality_count(rng: random.Random, weights: dict[str, float]) -> int:
        counts = [int(value) for value in weights]
        return rng.choices(counts, weights=[weights[str(value)] for value in counts], k=1)[0]

    def sample(
        self,
        count: int,
        *,
        seed: int,
        quality_count_weights: dict[str, float],
        excluded: set[str] | None = None,
    ) -> list[Plan]:
        rng = random.Random(seed)
        blocked = set(excluded or ())
        output: list[Plan] = []
        attempts = 0
        while len(output) < count and attempts < max(1000, count * 50):
            attempts += 1
            q_count = min(3, max(0, self._quality_count(rng, quality_count_weights)))
            plan = Plan(
                event=rng.choice(self.ids["event"]),
                context=rng.choice(self.ids["context"]),
                qualities=tuple(rng.sample(self.ids["quality"], q_count)),
                direction=rng.choice(self.ids["direction"]),
                output=rng.choice(self.ids["output"]),
            )
            if plan.key in blocked:
                continue
            blocked.add(plan.key)
            output.append(plan)
        if len(output) != count:
            raise RuntimeError(f"Generated only {len(output)} of {count} requested plans")
        return output

    def mutate(self, parent: Plan, *, seed: int, operation_weights: dict[str, float]) -> Plan:
        return self.mutate_with_metadata(parent, seed=seed, operation_weights=operation_weights).plan

    def mutate_with_metadata(
        self, parent: Plan, *, seed: int, operation_weights: dict[str, float]
    ) -> MutationResult:
        rng = random.Random(seed)
        valid_ops = list(operation_weights)
        for _ in range(100):
            operation = rng.choices(valid_ops, weights=[operation_weights[x] for x in valid_ops], k=1)[0]
            values = parent.to_dict()
            qualities = list(parent.qualities)
            if operation == "replace_event":
                values["event"] = rng.choice([x for x in self.ids["event"] if x != parent.event])
            elif operation == "replace_context":
                values["context"] = rng.choice([x for x in self.ids["context"] if x != parent.context])
            elif operation == "replace_quality" and qualities:
                index = rng.randrange(len(qualities))
                choices = [x for x in self.ids["quality"] if x not in qualities]
                qualities[index] = rng.choice(choices)
            elif operation == "add_quality" and len(qualities) < 3:
                qualities.append(rng.choice([x for x in self.ids["quality"] if x not in qualities]))
            elif operation == "drop_quality" and qualities:
                qualities.pop(rng.randrange(len(qualities)))
            elif operation == "replace_direction":
                values["direction"] = rng.choice([x for x in self.ids["direction"] if x != parent.direction])
            elif operation == "replace_output":
                values["output"] = rng.choice([x for x in self.ids["output"] if x != parent.output])
            else:
                continue
            candidate = Plan(
                event=values["event"], context=values["context"], qualities=tuple(qualities),
                direction=values["direction"], output=values["output"],
            )
            if candidate != parent:
                before = parent.to_dict()
                after = candidate.to_dict()
                changed = {
                    key: {"from": before[key], "to": after[key]}
                    for key in ("event", "context", "qualities", "direction", "output")
                    if before[key] != after[key]
                }
                return MutationResult(candidate, operation, changed)
        raise RuntimeError("Unable to generate a valid mutation")

    def expand(self, plan: Plan) -> dict[str, Any]:
        return {
            "plan_key": plan.key,
            "event": self.records[plan.event],
            "context": self.records[plan.context],
            "qualities": [self.records[item] for item in plan.qualities],
            "direction": self.records[plan.direction],
            "output": self.records[plan.output],
        }
