from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Iterable


@dataclass(slots=True)
class Trajectory:
    prompt_id: str
    prompt: str
    response: str
    ground_truth: str | None = None
    response_length: int = 0
    old_log_probs: list[float] = field(default_factory=list)
    new_log_probs: list[float] = field(default_factory=list)
    ref_log_probs: list[float] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    raw_reward_components: dict[str, float] = field(default_factory=dict)
    weighted_reward_components: dict[str, float] = field(default_factory=dict)
    reward_terms: dict[str, float] = field(default_factory=dict)
    reward_details: dict[str, Any] = field(default_factory=dict)
    total_reward: float = 0.0
    reward: float = 0.0
    component_advantages: dict[str, float] = field(default_factory=dict)
    combined_advantage_pre_whiten: float = 0.0
    combined_advantage_post_whiten: float = 0.0
    seq_advantage: float = 0.0
    token_advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    metrics: dict[str, float | bool | str] = field(default_factory=dict)
    filtered_out: bool = False
    filter_reason: str | None = None
    upstream_batch_index: int | None = None
    upstream_row_index: int | None = None

    def ensure_mask(self) -> list[int]:
        if self.response_mask:
            return self.response_mask
        if self.response_length > 0:
            self.response_mask = [1] * self.response_length
        elif self.new_log_probs:
            self.response_mask = [1] * len(self.new_log_probs)
        else:
            self.response_mask = [1]
        return self.response_mask

    def set_reward_components(
        self,
        *,
        raw_components: dict[str, float],
        weighted_components: dict[str, float],
        total_reward: float,
        reward_details: dict[str, Any] | None = None,
    ) -> None:
        self.raw_reward_components = dict(raw_components)
        self.weighted_reward_components = dict(weighted_components)
        self.reward_terms = dict(weighted_components)
        self.total_reward = total_reward
        self.reward = total_reward
        if reward_details is not None:
            self.reward_details = dict(reward_details)

    def effective_total_reward(self) -> float:
        if self.raw_reward_components or self.weighted_reward_components or self.total_reward != 0.0:
            return self.total_reward
        return self.reward

    def effective_weighted_reward_components(self) -> dict[str, float]:
        return self.weighted_reward_components or self.reward_terms

    def valid_length(self) -> int:
        return sum(self.ensure_mask())

    def metric_value(self, name: str) -> float:
        if name in self.metrics:
            return float(self.metrics[name])
        if name == "score":
            return float(self.effective_total_reward())
        if name == "seq_reward":
            return float(sum(self.effective_weighted_reward_components().values()))
        if name == "seq_final_reward":
            return float(self.effective_total_reward())
        raise KeyError(f"Unknown trajectory metric: {name}")


@dataclass(slots=True)
class PromptGroup:
    prompt_id: str
    trajectories: list[Trajectory] = field(default_factory=list)

    def metric_values(self, name: str) -> list[float]:
        return [trajectory.metric_value(name) for trajectory in self.trajectories]

    def reward_mean(self) -> float:
        return mean([trajectory.effective_total_reward() for trajectory in self.trajectories]) if self.trajectories else 0.0


@dataclass(slots=True)
class BatchContext:
    groups: list[PromptGroup] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_trajectories(self) -> Iterable[Trajectory]:
        for group in self.groups:
            yield from group.trajectories

    def prompt_count(self) -> int:
        return len(self.groups)

    def trajectory_count(self) -> int:
        return sum(len(group.trajectories) for group in self.groups)

    def kept_groups(self) -> list[PromptGroup]:
        kept: list[PromptGroup] = []
        for group in self.groups:
            survivors = [trajectory for trajectory in group.trajectories if not trajectory.filtered_out]
            if survivors:
                kept.append(PromptGroup(prompt_id=group.prompt_id, trajectories=survivors))
        return kept

    def clone_with_groups(self, groups: list[PromptGroup]) -> "BatchContext":
        return BatchContext(groups=groups, metadata=dict(self.metadata))
