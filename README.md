# AlphaSchema

**Structured semantic search for executable quantitative factors.**

[English](README.md) | [简体中文](README_zh-CN.md)

AlphaSchema treats factor discovery as a search over structured semantic plans rather than unconstrained code generation. Each plan describes a market event, its context, optional quality conditions, a trading interpretation, and the numerical form of the resulting signal.

During mining, the system alternates between exploring new schema combinations, exploiting promising regions, and mutating previously successful plans. A code agent turns each plan into executable factor code; validation and backtesting then return rewards and diagnostics that guide subsequent search.

[![AlphaSchema method overview](docs/assets/method.png)](docs/assets/method.pdf)

This repository contains the core mining workflow, an English schema library, prompts, configuration examples, and a lightweight runnable demo. Market data, private experiment histories, and API credentials are not included.

## Quick Start

AlphaSchema requires Python 3.10 or newer.

```bash
git clone git@github.com:JingyangYi/AlphaSchema.git
cd AlphaSchema
pip install -e .
python examples/run_demo.py
```

The demo uses the included lightweight backend and does not require private data or an LLM API.

To run a mining job with your own data and an OpenAI-compatible model endpoint:

```bash
export LLM_API_KEY="your-api-key"

alphaschema validate-data --config configs/default_stock_search.json
alphaschema run --config configs/default_stock_search.json
```

Prepare adjusted daily market data, then set the data path and field mapping in [`configs/default_stock_search.json`](configs/default_stock_search.json). Search behavior, batch size, model access, validation rules, and reward settings are all controlled through this configuration.

## Repository Structure

```text
AlphaSchema/
├── alphaschema/                 # Core search, generation, validation, and logging
│   ├── workflow.py              # End-to-end mining workflow
│   ├── selector.py              # Explore, exploit, and mutation selection
│   ├── code_agent.py            # Semantic-plan-to-code realization
│   ├── validation.py            # Implementation and leakage checks
│   ├── backend.py               # Factor evaluation interface
│   ├── reward_model.py          # Reward prediction and feedback
│   └── records.py               # Per-plan experiment records
├── configs/                     # Search and data configurations
├── schemas/stock_alpha/         # Event, context, quality, direction, and output schemas
├── prompts/                     # English prompts used by the agents
├── examples/                    # Minimal demo and backend template
├── tests/                       # Core workflow tests
└── pyproject.toml               # Package metadata and dependencies
```

## Mining Outputs

Each evaluated plan is recorded with its complete schema composition, selector source, predicted reward, realized reward, evaluation metrics, generated factor code, and validation status. Run-level summaries and reward trajectories are also written to the configured output directory, making the search process inspectable without requiring a separate UI.

## Project Scope

AlphaSchema provides the core factor-mining mechanism rather than a fully packaged research environment. Users supply their own market data, execution backend, model endpoint, and experiment-specific configuration.
