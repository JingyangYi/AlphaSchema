from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from .schema_space import Plan


def plan_features(plan: Plan) -> dict[str, float]:
    features = {
        f"event={plan.event}": 1.0,
        f"context={plan.context}": 1.0,
        f"direction={plan.direction}": 1.0,
        f"output={plan.output}": 1.0,
        f"event_context={plan.event}|{plan.context}": 1.0,
        f"event_output={plan.event}|{plan.output}": 1.0,
        f"event_direction={plan.event}|{plan.direction}": 1.0,
        "quality_count": float(len(plan.qualities)),
    }
    for quality in plan.qualities:
        features[f"quality={quality}"] = 1.0
        features[f"event_quality={plan.event}|{quality}"] = 1.0
        features[f"context_quality={plan.context}|{quality}"] = 1.0
    return features


@dataclass
class RewardPrediction:
    mean: float
    std: float


class RewardEnsemble:
    """Bootstrap LightGBM regression ensemble used by the mining runs."""

    def __init__(self, *, ensemble_size: int = 8, bootstrap_fraction: float = 0.7, seed: int = 42):
        self.ensemble_size = ensemble_size
        self.bootstrap_fraction = bootstrap_fraction
        self.seed = seed
        self.vectorizer = DictVectorizer(sparse=True)
        self.models: list[lgb.LGBMRegressor] = []

    def fit(self, observations: Iterable[tuple[Plan, float]]) -> "RewardEnsemble":
        rows = list(observations)
        if not rows:
            raise ValueError("At least one observation is required")
        x = self.vectorizer.fit_transform([plan_features(plan) for plan, _ in rows])
        y = np.asarray([float(reward) for _, reward in rows])
        rng = np.random.default_rng(self.seed)
        size = max(1, int(np.ceil(len(rows) * self.bootstrap_fraction)))
        self.models = []
        for index in range(self.ensemble_size):
            sample = np.arange(len(rows)) if size >= len(rows) else rng.choice(len(rows), size=size, replace=False)
            model = lgb.LGBMRegressor(
                objective="regression", n_estimators=450, learning_rate=0.035,
                num_leaves=(11, 15, 19, 15, 13, 17, 11, 21)[index % 8],
                min_child_samples=max(3, min(20, len(rows) // 10)),
                subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
                reg_lambda=1.4, random_state=self.seed + index * 1009,
                n_jobs=1, verbose=-1,
            )
            model.fit(x[sample], y[sample])
            self.models.append(model)
        return self

    def predict(self, plans: list[Plan]) -> list[RewardPrediction]:
        if not self.models:
            raise RuntimeError("RewardEnsemble must be fitted before prediction")
        x = self.vectorizer.transform([plan_features(plan) for plan in plans])
        values = np.vstack([model.predict(x) for model in self.models])
        return [RewardPrediction(float(mean), float(std)) for mean, std in zip(values.mean(0), values.std(0))]
