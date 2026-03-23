from __future__ import annotations

from dataclasses import dataclass

from dapo_lab.config_schema import DataConfig, RolloutBehaviorConfig


@dataclass(slots=True)
class RolloutRequest:
    backend: str
    num_samples: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    max_prompt_length: int
    max_response_length: int


def build_rollout_request(data: DataConfig, behavior: RolloutBehaviorConfig) -> RolloutRequest:
    return RolloutRequest(
        backend=behavior.backend,
        num_samples=data.rollout_n,
        do_sample=behavior.do_sample,
        temperature=behavior.temperature,
        top_p=behavior.top_p,
        top_k=behavior.top_k,
        max_prompt_length=data.max_prompt_length,
        max_response_length=data.max_response_length,
    )
