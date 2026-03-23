from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from dapo_lab.config_schema import AlgorithmConfig
from dapo_lab.trainer.state import BatchContext


VariantAdvantageFn = Callable[[BatchContext, AlgorithmConfig], dict[str, float]]
VariantMetricFn = Callable[[BatchContext, AlgorithmConfig], dict[str, float]]


@dataclass(frozen=True, slots=True)
class AlgorithmVariantSpec:
    name: str
    description: str
    supports_group_filtering: bool
    supports_overlong_policy: bool
    supports_asymmetric_clipping: bool


@dataclass(frozen=True, slots=True)
class AlgorithmVariantHooks:
    spec: AlgorithmVariantSpec
    apply_advantages: VariantAdvantageFn
    emit_variant_metrics: VariantMetricFn
