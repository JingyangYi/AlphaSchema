from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    """Serializable outcome of realizing and evaluating one semantic plan."""

    reward: float
    status: str = "ok"
    metrics: list[dict[str, Any]] = field(default_factory=list)
    reward_components: dict[str, Any] = field(default_factory=dict)
    code_files: list[str] = field(default_factory=list)
    code_versions: list[str] = field(default_factory=list, repr=False)
    factor_files: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
