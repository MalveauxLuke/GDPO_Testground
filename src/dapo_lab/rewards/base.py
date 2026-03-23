from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from dapo_lab.trainer.state import BatchContext, Trajectory


@dataclass(slots=True)
class RewardContext:
    trajectory: Trajectory
    batch: BatchContext
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RewardTermResult:
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)


class RewardTerm(Protocol):
    name: str

    def compute(self, context: RewardContext) -> RewardTermResult:
        ...
