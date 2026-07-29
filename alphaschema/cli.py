from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .backend import PanelFactorBackend
from .code_agent import CodeAgent
from .data import DataConfig, MarketData
from .pipeline import SearchPipeline
from .report import build_report
from .schema_space import SchemaSpace
from .workflow import FactorWorkflow


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _data(config: dict[str, Any], config_path: Path) -> MarketData:
    if "data" not in config:
        raise ValueError("Search config has no 'data' section")
    return MarketData(DataConfig.from_dict(config["data"], base_dir=config_path.parent))


def _backend(data: MarketData, config: dict[str, Any]) -> PanelFactorBackend:
    values = config.get("evaluation", {})
    reward = values.get("reward", {})
    return PanelFactorBackend(
        data, forward_horizon=values.get("forward_horizon", 5),
        min_assets=values.get("min_assets", 20), alpha=reward.get("alpha", 10.0),
        beta=reward.get("beta", 1.0), lag_penalty=reward.get("lag1_penalty", 2.0),
    )


def _artifact_dir(config: dict[str, Any], config_path: Path) -> Path:
    path = Path(config.get("artifact_dir", "artifacts/search"))
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def validate_data(config_path: Path) -> None:
    manifest = _data(_load(config_path), config_path).manifest()
    print(json.dumps(manifest, indent=2))


def run(config_path: Path, *, rounds: int, resume: bool, model: str | None, base_url: str | None) -> None:
    config = _load(config_path)
    data = _data(config, config_path)
    manifest = data.manifest()
    agent_config = config.get("code_agent", {})
    model = model or os.environ.get("ALPHASCHEMA_MODEL") or agent_config.get("model")
    base_url = base_url or os.environ.get("ALPHASCHEMA_BASE_URL") or agent_config.get("base_url")
    if not model or not base_url:
        raise ValueError("Specify code_agent.model/base_url in config or use --model and --base-url")

    artifact_dir = _artifact_dir(config, config_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "data_manifest.json"
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != manifest["fingerprint"]:
            raise ValueError("Configured data differs from this run's data manifest; refuse to resume")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (artifact_dir / "evaluation_config.json").write_text(
        json.dumps(config.get("evaluation", {}), indent=2) + "\n", encoding="utf-8"
    )
    schema_path = Path(config["schema_dir"])
    if not schema_path.is_absolute():
        schema_path = (config_path.parent / schema_path).resolve()
    workflow = FactorWorkflow(
        space=SchemaSpace(schema_path),
        agent=CodeAgent(
            model=str(model), base_url=str(base_url),
            api_key_env=str(agent_config.get("api_key_env", "LLM_API_KEY")),
        ),
        backend=_backend(data, config),
        prompt_dir=Path(__file__).resolve().parents[1] / "prompts",
        run_dir=artifact_dir / "realizations", periods=[int(x) for x in config["periods"]],
        max_repairs=int(config.get("max_repairs", 1)),
    )
    pipeline = SearchPipeline(config_path, workflow, resume=resume)
    completed = 0
    if resume and pipeline.rounds_path.exists():
        completed = sum(bool(line.strip()) for line in pipeline.rounds_path.read_text().splitlines())
    for round_index in range(completed, completed + rounds):
        rows = pipeline.run_round(round_index)
        print(json.dumps({
            "round": round_index, "plans": len(rows),
            "best_reward": max(reward for _, reward in rows),
        }))
    print(build_report(pipeline.artifact_dir))


def main() -> None:
    parser = argparse.ArgumentParser(prog="alphaschema")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-data", help="Load and validate configured market data")
    validate.add_argument("--config", type=Path, required=True)
    mining = commands.add_parser("run", help="Run schema-guided factor mining")
    mining.add_argument("--config", type=Path, required=True)
    mining.add_argument("--rounds", type=int, default=1)
    mining.add_argument("--resume", action="store_true")
    mining.add_argument("--model")
    mining.add_argument("--base-url")
    args = parser.parse_args()
    if args.command == "validate-data":
        validate_data(args.config.resolve())
    else:
        run(args.config.resolve(), rounds=args.rounds, resume=args.resume, model=args.model, base_url=args.base_url)


if __name__ == "__main__":
    main()
