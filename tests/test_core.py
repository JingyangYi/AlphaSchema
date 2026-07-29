from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import numpy as np

from alphaschema import DataConfig, EvaluationResult, MarketData, PanelFactorBackend, SearchPipeline
from alphaschema.report import build_report
from alphaschema.reward_model import RewardPrediction
from alphaschema.workflow import FactorWorkflow
from alphaschema.schema_space import SchemaSpace
from alphaschema.selector import Quota, quota_for, select_batch


ROOT = Path(__file__).resolve().parents[1]


def temporary_config(directory: str) -> Path:
    config = json.loads((ROOT / "configs" / "default_stock_search.json").read_text())
    config["schema_dir"] = str(ROOT / "schemas" / "stock_alpha")
    config["artifact_dir"] = str(Path(directory) / "artifacts")
    config["candidate_pool_size"] = 100
    path = Path(directory) / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_schema_sampling_and_mutation():
    space = SchemaSpace(ROOT / "schemas" / "stock_alpha")
    plans = space.sample(
        20, seed=7,
        quality_count_weights={"0": 0.15, "1": 0.35, "2": 0.35, "3": 0.15},
    )
    assert len({plan.key for plan in plans}) == 20
    assert all(0 <= len(plan.qualities) <= 3 for plan in plans)
    child = space.mutate(
        plans[0], seed=9,
        operation_weights={"replace_quality": 0.22, "replace_context": 0.16,
                           "replace_output": 0.16, "add_quality": 0.14,
                           "drop_quality": 0.12, "replace_direction": 0.12,
                           "replace_event": 0.08},
    )
    assert child.key != plans[0].key


def test_selection_records_predictions_and_mutation_parent():
    class FakeModel:
        def predict(self, plans):
            return [RewardPrediction(mean=float(index), std=0.25) for index, _ in enumerate(plans)]

    space = SchemaSpace(ROOT / "schemas" / "stock_alpha")
    plans = space.sample(12, seed=11, quality_count_weights={"1": 1.0})
    observations = [(plans[0], 0.8), (plans[1], 0.4)]
    selected = select_batch(
        plans[2:], observations, quota=Quota(explore=1, exploit=1, mutate=1),
        model=FakeModel(), space=space, seed=19,
        mutation_weights={"replace_event": 1.0},
    )
    assert {row.selection_reason for row in selected} == {"explore", "exploit", "mutation"}
    assert all(row.reward_pred_mean is not None and row.reward_pred_std == 0.25 for row in selected)
    mutation = next(row for row in selected if row.selection_reason == "mutation")
    assert mutation.parent_plan_key == plans[0].key
    assert mutation.parent_reward == 0.8
    assert mutation.mutation_operation == "replace_event"
    assert set(mutation.changed_components) == {"event"}


def test_default_schedule_and_pipeline():
    with TemporaryDirectory() as directory:
        pipeline = SearchPipeline(temporary_config(directory), lambda plan, expanded: 1.0)
        schedule = pipeline.config["selector"]["schedule"]
        assert quota_for(0, schedule).explore == 16
        assert sum(vars(quota_for(400, schedule)).values()) == 16
        assert len(pipeline.run_round(0)) == 16
        records = [json.loads(line) for line in pipeline.plans_path.read_text().splitlines()]
        assert len(records) == 16
        assert records[0]["schemas"]["event"]["schema_id"].startswith("event.")
        assert records[0]["selection"]["selection_reason"] == "explore"
        assert records[0]["evaluation"]["reward"] == 1.0
        assert build_report(pipeline.artifact_dir).exists()


def test_pipeline_preserves_metrics_and_generated_code():
    def evaluate(plan, expanded):
        return EvaluationResult(
            reward=0.7, metrics=[{"period": 20, "sharpe": 1.2}],
            reward_components={"fast": 0.7}, code_versions=["def factor(panel):\n    return panel\n"],
        )

    with TemporaryDirectory() as directory:
        pipeline = SearchPipeline(temporary_config(directory), evaluate)
        pipeline.run_round(0)
        record = json.loads((pipeline.plan_dir / "round_0000_plan_000" / "record.json").read_text())
        assert record["evaluation"]["metrics"] == [{"period": 20, "sharpe": 1.2}]
        assert record["evaluation"]["reward_components"] == {"fast": 0.7}
        code_path = pipeline.artifact_dir / record["evaluation"]["code_files"][0]
        assert code_path.read_text() == "def factor(panel):\n    return panel\n"


def test_failure_is_zero_reward():
    def fail(plan, expanded):
        raise RuntimeError("implementation failed")

    with TemporaryDirectory() as directory:
        pipeline = SearchPipeline(temporary_config(directory), fail)
        results = pipeline.run_round(0)
        assert len(results) == 16
        assert all(reward == 0.0 for _, reward in results)


def test_non_ok_structured_result_is_zero_reward():
    with TemporaryDirectory() as directory:
        pipeline = SearchPipeline(
            temporary_config(directory),
            lambda plan, expanded: EvaluationResult(reward=9.9, status="failed"),
        )
        assert all(reward == 0.0 for _, reward in pipeline.run_round(0))


def test_realization_workflow_repairs_then_backtests():
    class FakeAgent:
        def generate(self, *args, **kwargs):
            return "import numpy as np\ndef factor(panel, period=60):\n    return np.roll(panel, 1)"

        def repair(self, *args, **kwargs):
            return "def factor(panel, period=60):\n    return panel"

    class FakeBackend:
        def materialize(self, code_path, periods, output_dir):
            path = output_dir / "factor.pkl"
            pd.DataFrame({"A": [0.1, 0.2]}).to_pickle(path)
            return [path]

        def leakage_issues(self, factor_paths):
            return []

        def backtest(self, factor_paths, output_dir):
            return [{"period": 20, "score": 1.25}]

        def reward(self, metrics):
            return max(row["score"] for row in metrics)

    space = SchemaSpace(ROOT / "schemas" / "stock_alpha")
    plan = space.sample(1, seed=3, quality_count_weights={"0": 1.0})[0]
    with TemporaryDirectory() as directory:
        workflow = FactorWorkflow(
            space=space, agent=FakeAgent(), backend=FakeBackend(),
            prompt_dir=ROOT / "prompts", run_dir=directory, periods=[20, 80], max_repairs=1,
        )
        result = workflow(plan, space.expand(plan))
        assert isinstance(result, EvaluationResult)
        assert result.reward == 1.25
        assert result.metrics == [{"period": 20, "score": 1.25}]
        assert len(result.code_versions) == 2
        assert result.repair_history == [{
            "attempt": 1,
            "source_code": "code/factor_r0.py",
            "issues": ["np.roll is circular and can introduce future leakage."],
        }]


def test_user_data_pipeline_materializes_and_scores():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        dates = pd.date_range("2022-01-03", periods=45, freq="B")
        rows = []
        for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC")):
            for date_index, date in enumerate(dates):
                close = (
                    10.0 + symbol_index + 0.02 * date_index
                    + 0.35 * np.sin(date_index / (2.0 + symbol_index) + symbol_index)
                )
                rows.append({
                    "trade_date": date, "ticker": symbol,
                    "adj_open": close * 0.99, "adj_high": close * 1.01,
                    "adj_low": close * 0.98, "adj_close": close,
                    "adj_volume": 1000 + 10 * date_index + symbol_index,
                    "money": close * (1000 + 10 * date_index + symbol_index),
                })
        data_path = root / "bars.csv"
        pd.DataFrame(rows).to_csv(data_path, index=False)
        config = DataConfig.from_dict({
            "path": str(data_path), "format": "csv", "layout": "long_table",
            "date_column": "trade_date", "symbol_column": "ticker",
            "columns": {
                "open": "adj_open", "high": "adj_high", "low": "adj_low",
                "close": "adj_close", "volume": "adj_volume", "amount": "money",
            },
        }, base_dir=root)
        data = MarketData(config)
        assert data.manifest()["symbol_count"] == 3
        assert "vwap" in data.load()["AAA"]

        code_path = root / "factor.py"
        code_path.write_text(
            "import pandas as pd\n"
            "def F_momentum(panel, period=5):\n"
            "    return pd.concat({s: b['close'].pct_change(period) for s, b in panel.items()}, axis=1)\n",
            encoding="utf-8",
        )
        output = root / "output"
        output.mkdir()
        backend = PanelFactorBackend(data, forward_horizon=2, min_assets=2)
        factors = backend.materialize(code_path, [3, 6], output)
        assert len(factors) == 2
        assert backend.leakage_issues(factors) == []
        metrics = backend.backtest(factors, output)
        assert all({"rank_ic", "lag1_rank_ic", "lag1_gap", "mean_rank_turnover"} <= set(row) for row in metrics)
        reward = backend.reward(metrics)
        best = next(row for row in metrics if row["period"] == reward["best_period"])
        assert np.isclose(reward["lag1_penalty"], -2.0 * best["lag1_gap"])

        future_code = root / "future_factor.py"
        future_code.write_text(
            "import pandas as pd\n"
            "def F_future(panel, period=5):\n"
            "    return pd.concat({s: b['close'].shift(-1) for s, b in panel.items()}, axis=1)\n",
            encoding="utf-8",
        )
        future_dir = root / "future_output"
        future_dir.mkdir()
        future_paths = backend.materialize(future_code, [3], future_dir)
        assert backend.leakage_issues(future_paths)


def test_per_symbol_data_skips_files_outside_date_window():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        bars = root / "bars"
        bars.mkdir()
        columns = {
            "date": ["2024-01-02"], "open": [10.0], "high": [11.0],
            "low": [9.0], "close": [10.5], "volume": [100.0], "amount": [1050.0],
        }
        pd.DataFrame(columns).to_parquet(bars / "ACTIVE.parquet")
        inactive = dict(columns)
        inactive["date"] = ["2020-01-02"]
        pd.DataFrame(inactive).to_parquet(bars / "INACTIVE.parquet")
        config = DataConfig.from_dict({
            "path": str(bars), "layout": "one_file_per_symbol",
            "start_date": "2024-01-01", "end_date": "2024-12-31",
        }, base_dir=root)
        assert set(MarketData(config).load()) == {"ACTIVE"}
