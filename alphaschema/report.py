from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(artifact_dir: str | Path) -> Path:
    artifact_dir = Path(artifact_dir)
    plans = _read_jsonl(artifact_dir / "plans.jsonl")
    rounds = _read_jsonl(artifact_dir / "rounds.jsonl")
    if not plans:
        raise ValueError(f"No plan records found in {artifact_dir}")
    report_dir = artifact_dir / "report"
    report_dir.mkdir(exist_ok=True)

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in plans:
        by_source[row["selection"]["selection_reason"]].append(row)

    with (report_dir / "plans.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "plan_id", "round", "plan_key", "selection_reason", "parent_plan_key",
            "mutation_operation", "reward_pred_mean", "reward_pred_std", "status",
            "reward", "repair_attempts", "code_files",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in plans:
            selection, evaluation = row["selection"], row["evaluation"]
            writer.writerow({
                "plan_id": row["plan_id"], "round": row["round"], "plan_key": row["plan_key"],
                "selection_reason": selection["selection_reason"],
                "parent_plan_key": selection.get("parent_plan_key"),
                "mutation_operation": selection.get("mutation_operation"),
                "reward_pred_mean": selection.get("reward_pred_mean"),
                "reward_pred_std": selection.get("reward_pred_std"),
                "status": evaluation["status"], "reward": evaluation["reward"],
                "repair_attempts": evaluation.get("repair_attempts", 0),
                "code_files": ";".join(evaluation.get("code_files", [])),
            })

    with (report_dir / "reward_by_round.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["round", "plan_count", "valid_rate", "zero_reward_count", "reward_mean", "reward_max", "cumulative_best"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rounds:
            writer.writerow({key: row.get(key) for key in fields})

    source_rows = []
    for source, rows in sorted(by_source.items()):
        rewards = [float(row["evaluation"]["reward"]) for row in rows]
        source_rows.append({
            "source": source, "count": len(rows),
            "valid_rate": sum(row["evaluation"]["status"] == "ok" for row in rows) / len(rows),
            "reward_mean": mean(rewards), "reward_max": max(rewards),
        })
    with (report_dir / "reward_by_source.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    rewards = [float(row["evaluation"]["reward"]) for row in plans]
    valid = sum(row["evaluation"]["status"] == "ok" for row in plans)
    lines = [
        "# Search Run Summary", "",
        f"- Plans: {len(plans)}", f"- Rounds: {len(rounds)}",
        f"- Valid realizations: {valid} ({valid / len(plans):.1%})",
        f"- Mean reward: {mean(rewards):.6f}", f"- Best reward: {max(rewards):.6f}", "",
        "## Reward by Selection Source", "",
        "| Source | Count | Valid rate | Mean reward | Best reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['source']} | {row['count']} | {row['valid_rate']:.1%} | "
        f"{row['reward_mean']:.6f} | {row['reward_max']:.6f} |"
        for row in source_rows
    )
    lines.extend(["", "## Top Plans", "", "| Plan | Source | Reward | Code |", "| --- | --- | ---: | --- |"])
    for row in sorted(plans, key=lambda item: item["evaluation"]["reward"], reverse=True)[:10]:
        code = ", ".join(row["evaluation"].get("code_files", [])) or "n/a"
        lines.append(
            f"| {row['plan_id']} | {row['selection']['selection_reason']} | "
            f"{row['evaluation']['reward']:.6f} | {code} |"
        )
    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an AlphaSchema search artifact directory.")
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    path = build_report(args.artifact_dir)
    print(path)


if __name__ == "__main__":
    main()
