from __future__ import annotations

from dapo_lab.config_schema import AlgorithmConfig

from .variant_api import AlgorithmVariantHooks, AlgorithmVariantSpec
from .variants.dapo import HOOKS as DAPO_HOOKS
from .variants.gdpo import HOOKS as GDPO_HOOKS
from .variants.grpo import HOOKS as GRPO_HOOKS


# ============================================================================
# VARIANT DISPATCH
# ============================================================================

VARIANT_HOOKS: dict[str, AlgorithmVariantHooks] = {
    "grpo": GRPO_HOOKS,
    "dapo": DAPO_HOOKS,
    "gdpo": GDPO_HOOKS,
}


def resolve_variant_hooks(config: AlgorithmConfig) -> AlgorithmVariantHooks:
    try:
        return VARIANT_HOOKS[config.variant]
    except KeyError as error:
        raise KeyError(f"Unsupported algorithm variant: {config.variant}") from error


def resolve_variant_spec(config: AlgorithmConfig) -> AlgorithmVariantSpec:
    return resolve_variant_hooks(config).spec
