from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from dapo_lab.config_schema import PolicyLossConfig
from dapo_lab.trainer.state import BatchContext


PolicyLossFn = Callable[[BatchContext, PolicyLossConfig], "LossResult"]
_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


@dataclass(slots=True)
class LossResult:
    loss: float
    metrics: dict[str, float]


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    def decorator(fn: PolicyLossFn) -> PolicyLossFn:
        _LOSS_REGISTRY[name] = fn
        return fn

    return decorator


def get_policy_loss(name: str) -> PolicyLossFn:
    if name not in _LOSS_REGISTRY:
        raise KeyError(f"Unknown policy loss mode: {name}")
    return _LOSS_REGISTRY[name]


def aggregate_losses(loss_rows: list[list[float]], mask_rows: list[list[int]], mode: str) -> float:
    masked_tokens = [loss for row, mask in zip(loss_rows, mask_rows, strict=True) for loss, keep in zip(row, mask, strict=True) if keep]
    if not masked_tokens:
        return 0.0
    if mode == "token-mean":
        return mean(masked_tokens)
    if mode == "seq-mean-token-sum":
        seq_sums = [sum(loss * keep for loss, keep in zip(row, mask, strict=True)) for row, mask in zip(loss_rows, mask_rows, strict=True)]
        return mean(seq_sums)
    if mode == "seq-mean-token-mean":
        seq_means = []
        for row, mask in zip(loss_rows, mask_rows, strict=True):
            keep_count = sum(mask)
            if keep_count == 0:
                continue
            seq_means.append(sum(loss * keep for loss, keep in zip(row, mask, strict=True)) / keep_count)
        return mean(seq_means) if seq_means else 0.0
    raise ValueError(f"Invalid loss aggregation mode: {mode}")


@register_policy_loss("clipped")
def compute_clipped_policy_loss(batch: BatchContext, config: PolicyLossConfig) -> LossResult:
    clip_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio
    clip_c = config.clip_ratio_c

    loss_rows: list[list[float]] = []
    mask_rows: list[list[int]] = []
    clip_events = 0
    lower_clip_events = 0
    total_tokens = 0
    approx_kls: list[float] = []

    for trajectory in batch.iter_trajectories():
        mask = trajectory.ensure_mask()
        if not trajectory.token_advantages:
            raise ValueError("token_advantages must be populated before computing the policy loss.")
        row_losses: list[float] = []
        for old_log_prob, new_log_prob, advantage, keep in zip(
            trajectory.old_log_probs,
            trajectory.new_log_probs,
            trajectory.token_advantages,
            mask,
            strict=True,
        ):
            if not keep:
                row_losses.append(0.0)
                continue
            negative_approx_kl = max(min(new_log_prob - old_log_prob, 20.0), -20.0)
            ratio = math.exp(negative_approx_kl)
            unclipped = -advantage * ratio
            clipped_ratio = min(max(ratio, 1 - clip_low), 1 + clip_high)
            clipped = -advantage * clipped_ratio
            primary = max(unclipped, clipped)
            if clipped > unclipped:
                clip_events += 1
            if clip_c is not None and advantage < 0:
                dual_clipped = min(-advantage * clip_c, primary)
                if dual_clipped != primary:
                    lower_clip_events += 1
                token_loss = dual_clipped
            else:
                token_loss = primary
            row_losses.append(token_loss)
            approx_kls.append(-negative_approx_kl)
            total_tokens += 1
        loss_rows.append(row_losses)
        mask_rows.append(mask)

    aggregated = aggregate_losses(loss_rows, mask_rows, config.loss_agg_mode)
    metrics = {
        "actor/loss": aggregated,
        "actor/pg_clipfrac": clip_events / total_tokens if total_tokens else 0.0,
        "actor/pg_clipfrac_lower": lower_clip_events / total_tokens if total_tokens else 0.0,
        "actor/ppo_kl": mean(approx_kls) if approx_kls else 0.0,
    }
    return LossResult(loss=aggregated, metrics=metrics)


def compute_policy_loss(batch: BatchContext, config: PolicyLossConfig) -> LossResult:
    return get_policy_loss(config.mode)(batch, config)
