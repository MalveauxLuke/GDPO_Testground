from __future__ import annotations

from dapo_lab.algorithms.advantages import (
    assign_scalar_advantages,
    compute_group_relative_scalar_advantages,
    summarize_scalar_advantages,
)
from dapo_lab.algorithms.variant_api import AlgorithmVariantHooks, AlgorithmVariantSpec
from dapo_lab.config_schema import AlgorithmConfig
from dapo_lab.trainer.state import BatchContext


# ============================================================================
# GRPO VARIANT
# ============================================================================

SPEC = AlgorithmVariantSpec(
    name="grpo",
    description="Pedagogical baseline with grouped outcome advantage and symmetric clipping.",
    supports_group_filtering=False,
    supports_overlong_policy=False,
    supports_asymmetric_clipping=False,
)


def apply_advantages(batch: BatchContext, config: AlgorithmConfig) -> dict[str, float]:
    scalar_advantages = compute_group_relative_scalar_advantages(
        batch,
        lambda trajectory: trajectory.effective_total_reward(),
        normalize_by_std=config.advantage.normalize_by_std,
    )
    assign_scalar_advantages(batch, scalar_advantages)
    return summarize_scalar_advantages(scalar_advantages.values(), prefix="grpo/combined_advantage")


def emit_variant_metrics(_batch: BatchContext, _config: AlgorithmConfig) -> dict[str, float]:
    return {}


HOOKS = AlgorithmVariantHooks(
    spec=SPEC,
    apply_advantages=apply_advantages,
    emit_variant_metrics=emit_variant_metrics,
)
