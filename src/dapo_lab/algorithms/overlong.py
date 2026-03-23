from __future__ import annotations

from dataclasses import dataclass

from dapo_lab.config_schema import OverlongConfig
from dapo_lab.trainer.state import BatchContext


@dataclass(slots=True)
class OverlongResult:
    batch: BatchContext
    metrics: dict[str, float]


def compute_overlong_penalty(
    *,
    response_length: int,
    max_response_length: int,
    buffer_length: int,
    penalty_factor: float,
) -> float:
    expected_length = max_response_length - buffer_length
    exceed_length = response_length - expected_length
    if exceed_length <= 0:
        return 0.0
    penalty = -exceed_length / buffer_length * penalty_factor
    return min(penalty, 0.0)


def apply_overlong_policy(batch: BatchContext, config: OverlongConfig, *, max_response_length: int) -> OverlongResult:
    if not config.enabled:
        return OverlongResult(batch=batch, metrics={"overlong/penalized": 0.0, "overlong/filtered": 0.0})

    penalized = 0
    filtered = 0
    for trajectory in batch.iter_trajectories():
        penalty = compute_overlong_penalty(
            response_length=trajectory.valid_length(),
            max_response_length=max_response_length,
            buffer_length=config.buffer_length,
            penalty_factor=config.penalty_factor,
        )
        if penalty < 0:
            penalized += 1
            raw_components = dict(trajectory.raw_reward_components)
            weighted_components = dict(trajectory.effective_weighted_reward_components())
            raw_components["overlong"] = penalty
            weighted_components["overlong"] = penalty
            trajectory.set_reward_components(
                raw_components=raw_components,
                weighted_components=weighted_components,
                total_reward=trajectory.effective_total_reward() + penalty,
                reward_details=trajectory.reward_details,
            )
            trajectory.metrics["overlong_penalty"] = penalty
        should_filter = config.mode in {"filter", "shape_and_filter"} and trajectory.valid_length() >= max_response_length
        if should_filter or config.hard_filter:
            if trajectory.valid_length() >= max_response_length:
                filtered += 1
                trajectory.filtered_out = True
                trajectory.filter_reason = "overlong"
    return OverlongResult(
        batch=batch,
        metrics={"overlong/penalized": float(penalized), "overlong/filtered": float(filtered)},
    )
