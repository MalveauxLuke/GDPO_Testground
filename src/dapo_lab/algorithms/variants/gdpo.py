from __future__ import annotations

from statistics import mean

from dapo_lab.algorithms.advantages import (
    assign_scalar_advantages,
    batch_whiten_scalar_advantages,
    compute_group_relative_scalar_advantages,
    summarize_distribution,
    summarize_scalar_advantages,
)
from dapo_lab.algorithms.variant_api import AlgorithmVariantHooks, AlgorithmVariantSpec
from dapo_lab.config_schema import AlgorithmConfig
from dapo_lab.trainer.state import BatchContext


# ============================================================================
# GDPO VARIANT
# ============================================================================

SPEC = AlgorithmVariantSpec(
    name="gdpo",
    description="Group reward-Decoupled Normalization Policy Optimization with decoupled multi-reward advantages.",
    supports_group_filtering=False,
    supports_overlong_policy=False,
    supports_asymmetric_clipping=False,
)


def _component_key_weight_pairs(config: AlgorithmConfig) -> list[tuple[str, float]]:
    assert config.gdpo.component_keys is not None
    assert config.gdpo.component_weights is not None
    return list(zip(config.gdpo.component_keys, config.gdpo.component_weights, strict=True))


def apply_advantages(batch: BatchContext, config: AlgorithmConfig) -> dict[str, float]:
    component_pairs = _component_key_weight_pairs(config)
    combined_pre_whiten: dict[int, float] = {}

    for trajectory in batch.iter_trajectories():
        trajectory.component_advantages = {}

    for component_key, weight in component_pairs:
        normalized_component = compute_group_relative_scalar_advantages(
            batch,
            lambda trajectory, key=component_key: trajectory.raw_reward_components[key],
            normalize_by_std=config.gdpo.normalize_by_std,
        )
        for trajectory in batch.iter_trajectories():
            normalized_value = normalized_component[id(trajectory)]
            trajectory.component_advantages[component_key] = normalized_value
            combined_pre_whiten[id(trajectory)] = combined_pre_whiten.get(id(trajectory), 0.0) + weight * normalized_value

    for trajectory in batch.iter_trajectories():
        trajectory.combined_advantage_pre_whiten = combined_pre_whiten.get(id(trajectory), 0.0)

    if config.gdpo.batch_whiten:
        combined_post_whiten = batch_whiten_scalar_advantages(combined_pre_whiten)
    else:
        combined_post_whiten = dict(combined_pre_whiten)

    for trajectory in batch.iter_trajectories():
        trajectory.combined_advantage_post_whiten = combined_post_whiten[id(trajectory)]

    assign_scalar_advantages(batch, combined_post_whiten)

    metrics = summarize_scalar_advantages(combined_post_whiten.values(), prefix="gdpo/combined_advantage")
    pre_values = list(combined_pre_whiten.values())
    metrics["gdpo/combined_advantage_pre_whiten_mean"] = mean(pre_values) if pre_values else 0.0
    metrics["gdpo/combined_advantage_pre_whiten_std"] = summarize_distribution(pre_values)["std"]
    return metrics


def emit_variant_metrics(batch: BatchContext, config: AlgorithmConfig) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for component_key, _weight in _component_key_weight_pairs(config):
        raw_values = [trajectory.raw_reward_components[component_key] for trajectory in batch.iter_trajectories()]
        normalized_values = [trajectory.component_advantages[component_key] for trajectory in batch.iter_trajectories()]
        raw_summary = summarize_distribution(raw_values)
        advantage_summary = summarize_distribution(normalized_values)
        metrics[f"gdpo/{component_key}/mean"] = raw_summary["mean"]
        metrics[f"gdpo/{component_key}/std"] = raw_summary["std"]
        metrics[f"gdpo/{component_key}/min"] = raw_summary["min"]
        metrics[f"gdpo/{component_key}/max"] = raw_summary["max"]
        metrics[f"gdpo/{component_key}/adv_mean"] = advantage_summary["mean"]
        metrics[f"gdpo/{component_key}/adv_std"] = advantage_summary["std"]
        metrics[f"gdpo/{component_key}/adv_min"] = advantage_summary["min"]
        metrics[f"gdpo/{component_key}/adv_max"] = advantage_summary["max"]
    return metrics


HOOKS = AlgorithmVariantHooks(
    spec=SPEC,
    apply_advantages=apply_advantages,
    emit_variant_metrics=emit_variant_metrics,
)
