# AlphaSchema

AlphaSchema searches a structured semantic space of quantitative factor ideas.
Each plan combines one event, one context, zero to three qualities, one
direction, and one output. A quota scheduler balances structural exploration,
reward-model exploitation, and local mutation before selected plans are sent
to a code implementation agent and evaluator.

This repository contains the core search and realization workflow and schema
vocabulary. Market data, private historical runs, backtest infrastructure, and
model credentials are intentionally excluded. New runs write their own complete
plan records and generated code locally.

## Install

```bash
python -m pip install -e .
```

## Run the minimal example

```bash
python examples/run_demo.py
```

The example uses a deterministic mock evaluator so selection and reward-model
training can run without market data or an LLM account. The production path is
`alphaschema.workflow.FactorWorkflow` with the included `PanelFactorBackend`,
which executes generation, static validation, one repair attempt,
materialization, numerical leakage checking, fast/slow evaluation, and reward
calculation. `examples/backend_template.py` is only needed when replacing the
reference evaluator with another backtest stack. Returning `None`, exhausting
repair, or raising an exception records zero reward.

The default configuration is `configs/default_stock_search.json`. It uses a batch size of
16 and the same observation thresholds and relative scheduling policy as the
mining configuration. Batch size, candidate pool size, schedule, mutation
weights, and quality-count distribution are all configurable.

## Prepare market data

The reference mining backend expects daily stock bars. Prices and volumes should
be adjusted consistently across time. The first six market fields below are
required; `vwap` may be supplied or derived:

| Field | Meaning |
| --- | --- |
| `date` | Trading date or timestamp |
| `symbol` | Stable stock identifier; required only in a long table |
| `open`, `high`, `low`, `close` | Adjusted OHLC prices |
| `volume` | Adjusted trading volume |
| `amount` | Traded value, in one consistent currency unit |
| `vwap` | Adjusted VWAP; derived as `amount / volume` when omitted |

Optional columns such as `turnover_rate`, `market_value`, `neg_market_value`,
`pre_close`, `adjfactor`, `is_open`, `is_st`, `limit_pct`, `limit_up`, and
`limit_down` are passed through to generated factors when present. Do not mix
raw and adjusted OHLC fields in the standardized columns.

Two layouts are supported. A single long-table Parquet or CSV file has one row
per date and symbol:

```text
date,symbol,open,high,low,close,volume,amount,vwap
2024-01-02,000001.SZ,9.30,9.48,9.25,9.42,82103400,768310000,9.36
2024-01-02,000002.SZ,10.20,10.45,10.18,10.39,46312000,477010000,10.30
```

Alternatively, a directory may contain one Parquet or CSV file per symbol.
The filename stem is the symbol and each file contains `date` plus the market
fields:

```text
data/stock_bars/
  000001.SZ.parquet
  000002.SZ.parquet
```

Point the `data` section of the search config to the prepared data. Field
mapping is from AlphaSchema's canonical name to the user's source column:

```json
{
  "data": {
    "format": "parquet",
    "path": "${ALPHASCHEMA_DATA}",
    "layout": "one_file_per_symbol",
    "date_column": "trade_date",
    "columns": {
      "open": "adj_open",
      "high": "adj_high",
      "low": "adj_low",
      "close": "adj_close",
      "volume": "adj_volume",
      "amount": "amount",
      "vwap": "adj_vwap"
    },
    "start_date": "2016-01-01",
    "end_date": "2025-12-31",
    "universe": {
      "path": "../data/universe.csv",
      "symbol_column": "symbol"
    }
  }
}
```

Paths may be relative to the config file or supplied through environment
variables. Validate the complete dataset before spending any LLM calls:

```bash
alphaschema validate-data --config configs/default_stock_search.json
```

The command reports only a data fingerprint, symbol count, date range, row
count, and common fields. It does not copy market data or persist its absolute
path in run artifacts.

## Run mining

Set an OpenAI-compatible endpoint and API key, then run one or more rounds:

```bash
export ALPHASCHEMA_DATA=/path/to/stock_bars
export ALPHASCHEMA_MODEL=your-code-model
export ALPHASCHEMA_BASE_URL=https://your-endpoint.example/v1
export LLM_API_KEY=your-key

alphaschema run --config configs/default_stock_search.json --rounds 10
alphaschema run --config configs/default_stock_search.json --rounds 10 --resume
```

For each fast/slow realization, the reference evaluator computes 5-day IC,
ICIR, RankIC, RankICIR, mean rank turnover, and the corresponding metrics after
delaying signal execution by one bar. The default reward is

```text
10 * RankIC + RankICIR - 2 * max(0, RankIC - lag1_RankIC)
```

Thus `lag1` is an execution-delay robustness penalty: a factor is penalized
only for the performance lost when it is acted on one bar later. The plan
reward is the better reward of its fast and slow realizations. These weights,
the 5-day horizon, and the minimum cross-sectional asset count are configurable
under `evaluation`.

## Run artifacts

Every selected plan is recorded, including failed and zero-reward realizations:

```text
artifacts/<run>/
  run.json                         # search configuration and schedule
  events.jsonl                     # live round/plan lifecycle events
  rounds.jsonl                     # reward trend and source summary per round
  plans.jsonl                      # complete append-only plan records
  reward_buffer.jsonl              # compact resume buffer
  plans/round_0000_plan_000/
    record.json                    # one readable record for this plan
    code/factor_r0.py              # generated code and every repair revision
  report/
    summary.md
    plans.csv
    reward_by_round.csv
    reward_by_source.csv
```

Each plan record contains the selected schema IDs and full expanded schema JSON,
selection source (`explore`, `exploit`, `mutation`, or fallback), reward-model
mean and uncertainty when available, mutation parent and changed components,
actual reward, reward components, all backend metrics, validation issues, repair
count, timing, and paths to preserved code revisions.

Generate or refresh the compact report with:

```bash
python -m alphaschema.report artifacts/default_stock_search
```

Construct `SearchPipeline(..., resume=True)` to restore `reward_buffer.jsonl`
and continue a stopped search. The compact buffer is only for resumption;
`plans.jsonl` and the per-plan directories are the audit record.

## Code implementation agent

`alphaschema.code_agent.CodeAgent` supports OpenAI-compatible chat completion
endpoints. Set the endpoint and model explicitly in your integration and keep
the API key in `LLM_API_KEY`; no provider address or credential is stored in
this repository. The prompts in `prompts/` implement the plan-to-code contract.
