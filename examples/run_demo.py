from __future__ import annotations

import hashlib
from pathlib import Path

from alphaschema import EvaluationResult, SearchPipeline
from alphaschema.report import build_report


ROOT = Path(__file__).resolve().parents[1]


def mock_evaluator(plan, expanded_plan):
    """Deterministic stand-in that exercises the complete artifact contract."""
    digest = hashlib.sha256(plan.key.encode()).digest()
    reward = int.from_bytes(digest[:4], "big") / (2**32 - 1)
    code = (
        "def factor(panel, period=20):\n"
        f"    # Demo realization of {expanded_plan['event']['schema_id']}\n"
        "    return panel.pct_change(period)\n"
    )
    return EvaluationResult(
        reward=reward,
        metrics=[
            {"period": 20, "sharpe": 2.0 * reward - 1.0, "turnover": 1.0 - reward},
            {"period": 80, "sharpe": 1.5 * reward - 0.5, "turnover": 0.8 - 0.5 * reward},
        ],
        reward_components={"mock_score": reward},
        code_versions=[code],
        events=["demo_generated", "demo_backtested"],
        metadata={"evaluator": "deterministic_demo"},
    )


if __name__ == "__main__":
    pipeline = SearchPipeline(ROOT / "configs" / "default_stock_search.json", mock_evaluator)
    for round_index in range(3):
        rows = pipeline.run_round(round_index)
        best_plan, best_reward = max(rows, key=lambda row: row[1])
        print(
            f"round={round_index + 1} batch={len(rows)} "
            f"best_reward={best_reward:.4f} best_plan={best_plan.key}"
        )
    print(f"report={build_report(pipeline.artifact_dir)}")
