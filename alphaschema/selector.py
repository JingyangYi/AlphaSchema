from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .reward_model import RewardEnsemble
from .schema_space import Plan, SchemaSpace


@dataclass(frozen=True)
class Quota:
    explore: int
    exploit: int
    mutate: int


@dataclass
class SelectedPlan:
    plan: Plan
    selection_reason: str
    candidate_source: str
    novelty_score: float | None = None
    reward_pred_mean: float | None = None
    reward_pred_std: float | None = None
    prediction_rank: int | None = None
    parent_plan_key: str | None = None
    parent_reward: float | None = None
    mutation_operation: str | None = None
    changed_components: dict[str, dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_reason": self.selection_reason,
            "candidate_source": self.candidate_source,
            "novelty_score": self.novelty_score,
            "reward_pred_mean": self.reward_pred_mean,
            "reward_pred_std": self.reward_pred_std,
            "prediction_rank": self.prediction_rank,
            "parent_plan_key": self.parent_plan_key,
            "parent_reward": self.parent_reward,
            "mutation_operation": self.mutation_operation,
            "changed_components": self.changed_components,
        }


def quota_for(observation_count: int, schedule: list[dict[str, Any]]) -> Quota:
    for stage in schedule:
        upper = stage.get("max_observations")
        if observation_count >= stage["min_observations"] and (upper is None or observation_count < upper):
            return Quota(stage["explore"], stage["exploit"], stage["mutate"])
    raise ValueError(f"No schedule stage covers observation count {observation_count}")


def novelty(plan: Plan, history: list[Plan]) -> float:
    events = Counter(item.event for item in history)
    contexts = Counter(item.context for item in history)
    pairs = Counter((item.event, item.context) for item in history)
    return (
        0.60 / (1 + pairs[(plan.event, plan.context)]) ** 0.5
        + 0.25 / (1 + events[plan.event]) ** 0.5
        + 0.15 / (1 + contexts[plan.context]) ** 0.5
    )


def select_batch(
    candidates: list[Plan], observations: list[tuple[Plan, float]], *, quota: Quota,
    model: RewardEnsemble | None, space: SchemaSpace, seed: int,
    mutation_weights: dict[str, float],
) -> list[SelectedPlan]:
    expected = quota.explore + quota.exploit + quota.mutate
    if expected <= 0:
        return []
    history = [plan for plan, _ in observations]
    seen = {plan.key for plan in history}
    available = [plan for plan in candidates if plan.key not in seen]
    explore_plans = sorted(available, key=lambda plan: novelty(plan, history), reverse=True)[:quota.explore]
    selected = [SelectedPlan(plan, "explore", "random", novelty_score=novelty(plan, history)) for plan in explore_plans]
    selected_keys = {item.plan.key for item in selected}

    if quota.exploit:
        if model is None:
            ranked = sorted((plan for plan in available if plan.key not in selected_keys), key=lambda p: novelty(p, history), reverse=True)
        else:
            pool = [plan for plan in available if plan.key not in selected_keys]
            predictions = model.predict(pool)
            ranked = [plan for plan, _ in sorted(zip(pool, predictions), key=lambda pair: pair[1].mean, reverse=True)]
        selected.extend(
            SelectedPlan(plan, "exploit", "reward_model", novelty_score=novelty(plan, history))
            for plan in ranked[:quota.exploit]
        )
        selected_keys.update(item.plan.key for item in selected)

    parents = [plan for plan, _ in sorted(observations, key=lambda row: row[1], reverse=True)[:100]]
    attempt = 0
    while len(selected) < expected and parents and attempt < 1000:
        parent = parents[attempt % len(parents)]
        mutation = space.mutate_with_metadata(parent, seed=seed + attempt, operation_weights=mutation_weights)
        child = mutation.plan
        attempt += 1
        if child.key in seen or child.key in selected_keys:
            continue
        parent_reward = next(reward for plan, reward in observations if plan.key == parent.key)
        selected.append(SelectedPlan(
            child, "mutation", "top_plan_mutation",
            novelty_score=novelty(child, history), parent_plan_key=parent.key,
            parent_reward=float(parent_reward), mutation_operation=mutation.operation,
            changed_components=mutation.changed_components,
        ))
        selected_keys.add(child.key)

    if len(selected) < expected:
        fillers = [plan for plan in available if plan.key not in selected_keys]
        selected.extend(
            SelectedPlan(plan, "fallback", "random", novelty_score=novelty(plan, history))
            for plan in fillers[: expected - len(selected)]
        )

    chosen = selected[:expected]
    if model is not None and chosen:
        predictions = model.predict([item.plan for item in chosen])
        rank_by_key = {
            item.plan.key: rank
            for rank, (item, _) in enumerate(
                sorted(zip(chosen, predictions), key=lambda pair: pair[1].mean, reverse=True), start=1
            )
        }
        for item, prediction in zip(chosen, predictions):
            item.reward_pred_mean = prediction.mean
            item.reward_pred_std = prediction.std
            item.prediction_rank = rank_by_key[item.plan.key]
    return chosen
