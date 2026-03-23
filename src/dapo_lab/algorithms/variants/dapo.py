from __future__ import annotations

from dapo_lab.algorithms.variant_api import AlgorithmVariantHooks, AlgorithmVariantSpec
from dapo_lab.config_schema import AlgorithmConfig
from dapo_lab.trainer.state import BatchContext

from .grpo import apply_advantages as apply_grpo_advantages


# ============================================================================
# DAPO VARIANT
# ============================================================================

SPEC = AlgorithmVariantSpec(
    name="dapo",
    description="DAPO-style variant with asymmetric clipping, dynamic sampling, and overlong handling.",
    supports_group_filtering=True,
    supports_overlong_policy=True,
    supports_asymmetric_clipping=True,
)


def apply_advantages(batch: BatchContext, config: AlgorithmConfig) -> dict[str, float]:
    # DAPO in this harness keeps GRPO-style grouped outcome advantages.
    return apply_grpo_advantages(batch, config)


def emit_variant_metrics(_batch: BatchContext, _config: AlgorithmConfig) -> dict[str, float]:
    return {}


HOOKS = AlgorithmVariantHooks(
    spec=SPEC,
    apply_advantages=apply_advantages,
    emit_variant_metrics=emit_variant_metrics,
)
