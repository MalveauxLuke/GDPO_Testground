from __future__ import annotations

from statistics import mean, pstdev
from typing import Callable

from dapo_lab.trainer.state import BatchContext


# ============================================================================
# ADVANTAGE SHARED MATH
# ============================================================================

ScalarGetter = Callable[[object], float]


def group_relative_normalize(values: list[float], *, normalize_by_std: bool) -> list[float]:
    group_mean = mean(values) if values else 0.0
    group_std = pstdev(values) if len(values) > 1 else 1.0
    denominator = group_std if normalize_by_std and group_std > 0 else 1.0
    return [(value - group_mean) / denominator for value in values]


def compute_group_relative_scalar_advantages(
    batch: BatchContext,
    value_fn: Callable[[object], float],
    *,
    normalize_by_std: bool,
) -> dict[int, float]:
    scalar_advantages: dict[int, float] = {}
    for group in batch.groups:
        values = [float(value_fn(trajectory)) for trajectory in group.trajectories]
        normalized_values = group_relative_normalize(values, normalize_by_std=normalize_by_std)
        for trajectory, normalized in zip(group.trajectories, normalized_values, strict=True):
            scalar_advantages[id(trajectory)] = normalized
    return scalar_advantages


def batch_whiten_scalar_advantages(values_by_trajectory: dict[int, float]) -> dict[int, float]:
    values = list(values_by_trajectory.values())
    summary = summarize_distribution(values)
    std = summary["std"] if summary["std"] > 0 else 1.0
    mean_value = summary["mean"]
    return {trajectory_id: (value - mean_value) / std for trajectory_id, value in values_by_trajectory.items()}


def assign_scalar_advantages(batch: BatchContext, scalar_advantages: dict[int, float]) -> None:
    for trajectory in batch.iter_trajectories():
        advantage = scalar_advantages[id(trajectory)]
        trajectory.seq_advantage = advantage
        token_mask = trajectory.ensure_mask()
        trajectory.token_advantages = [advantage * mask for mask in token_mask]
        trajectory.returns = list(trajectory.token_advantages)
        trajectory.metrics["seq_advantage"] = advantage


def summarize_distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize_scalar_advantages(values, *, prefix: str | None = None) -> dict[str, float]:
    values = list(values)
    metrics = {
        "advantage/mean_abs": mean([abs(value) for value in values]) if values else 0.0,
    }
    if prefix is not None:
        summary = summarize_distribution(values)
        metrics[f"{prefix}_mean"] = summary["mean"]
        metrics[f"{prefix}_std"] = summary["std"]
    return metrics
