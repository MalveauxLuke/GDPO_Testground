from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from dapo_lab.config_schema import RewardTermConfig
from dapo_lab.trainer.state import BatchContext, Trajectory

from .base import RewardContext, RewardTerm
from .math import BoxedFormatReward, MathAccuracyReward


TERM_FACTORIES = {
    "math_accuracy": MathAccuracyReward,
    "boxed_format": BoxedFormatReward,
}


@dataclass(slots=True)
class WeightedRewardTerm:
    name: str
    weight: float
    term: RewardTerm


class RewardComposer:
    def __init__(self, weighted_terms: list[WeightedRewardTerm]) -> None:
        self.weighted_terms = weighted_terms

    @classmethod
    def from_configs(cls, configs: list[RewardTermConfig]) -> "RewardComposer":
        weighted_terms: list[WeightedRewardTerm] = []
        for config in configs:
            if not config.enabled:
                continue
            factory = TERM_FACTORIES.get(config.kind)
            if factory is None:
                raise KeyError(f"Unknown reward term kind: {config.kind}")
            weighted_terms.append(WeightedRewardTerm(name=config.name, weight=config.weight, term=factory(name=config.name)))
        return cls(weighted_terms=weighted_terms)

    def score_trajectory(self, trajectory: Trajectory, batch: BatchContext) -> dict[str, Any]:
        total = 0.0
        raw_term_scores: dict[str, float] = {}
        weighted_term_scores: dict[str, float] = {}
        detail_metrics: dict[str, Any] = {}
        for weighted_term in self.weighted_terms:
            result = weighted_term.term.compute(RewardContext(trajectory=trajectory, batch=batch))
            raw_term_scores[weighted_term.name] = result.score
            weighted_score = weighted_term.weight * result.score
            total += weighted_score
            weighted_term_scores[weighted_term.name] = weighted_score
            for key, value in result.metrics.items():
                detail_metrics[key] = value
        trajectory.set_reward_components(
            raw_components=raw_term_scores,
            weighted_components=weighted_term_scores,
            total_reward=total,
            reward_details=detail_metrics,
        )
        trajectory.metrics.update(detail_metrics)
        trajectory.metrics["score"] = total
        return {
            "reward": total,
            "raw_term_scores": raw_term_scores,
            "weighted_term_scores": weighted_term_scores,
            "details": detail_metrics,
        }

    def score_batch(self, batch: BatchContext) -> dict[str, float]:
        totals: list[float] = []
        raw_term_history: dict[str, list[float]] = {}
        weighted_term_history: dict[str, list[float]] = {}
        for trajectory in batch.iter_trajectories():
            summary = self.score_trajectory(trajectory, batch)
            totals.append(summary["reward"])
            for key, value in summary["raw_term_scores"].items():
                raw_term_history.setdefault(key, []).append(value)
            for key, value in summary["weighted_term_scores"].items():
                weighted_term_history.setdefault(key, []).append(value)
        metrics = {"reward/mean": mean(totals) if totals else 0.0}
        for key, values in raw_term_history.items():
            metrics[f"reward/raw/{key}/mean"] = mean(values)
        for key, values in weighted_term_history.items():
            metrics[f"reward/{key}/mean"] = mean(values)
        return metrics
