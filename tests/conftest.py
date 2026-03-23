from __future__ import annotations

import pytest

from dapo_lab.config_schema import ExperimentConfig
from dapo_lab.trainer.state import BatchContext, PromptGroup, Trajectory
from dapo_lab.validation import validate_experiment_config


def build_config(variant: str = "dapo") -> ExperimentConfig:
    payload = {
        "experiment": {"name": f"{variant}-test", "seed": 1, "output_dir": "outputs/test"},
        "data": {
            "train_files": ["train.jsonl"],
            "val_files": ["val.jsonl"],
            "format": "jsonl",
            "prompt_key": "prompt",
            "answer_key": "ground_truth",
            "train_batch_size": 2,
            "gen_batch_size": 4,
            "max_prompt_length": 1024,
            "max_response_length": 10,
            "rollout_n": 2,
        },
        "reward": {
            "terms": [
                {"name": "accuracy", "kind": "math_accuracy", "weight": 1.0},
                {"name": "boxed_format", "kind": "boxed_format", "weight": 0.1},
            ],
            "overlong": {
                "enabled": variant == "dapo",
                "mode": "shaping",
                "buffer_length": 2,
                "penalty_factor": 1.0,
                "hard_filter": False,
                "log": True,
            },
        },
        "algorithm": {
            "variant": variant,
            "advantage": {"mode": "grpo", "normalize_by_std": True},
            "gdpo": {
                "component_keys": ["accuracy", "boxed_format"],
                "component_weights": [1.0, 0.1],
                "normalize_by_std": True,
                "batch_whiten": True,
                "allow_single_component_ablation": False,
            },
            "policy_loss": {
                "mode": "clipped",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28 if variant == "dapo" else 0.2,
                "clip_ratio_c": 3.0 if variant == "dapo" else None,
                "loss_agg_mode": "token-mean",
            },
            "group_filtering": {
                "enabled": variant == "dapo",
                "metric": "acc",
                "max_num_gen_batches": 3,
                "require_variance": True,
            },
            "kl": {"enabled": False, "source": "reward", "penalty": "low_var_kl"},
            "trainer_order": {
                "stages": ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
            },
            "rollout_behavior": {
                "backend": "vllm",
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "accumulate_filtered_groups": variant == "dapo",
            },
        },
        "trainer": {
            "log_level": "info",
            "diagnostics": {
                "record_stage_events": True,
                "emit_group_stats": True,
                "emit_reward_breakdown": True,
            },
            "save_freq": 100,
            "test_freq": 100,
            "val_before_train": True,
        },
        "verl": {
            "required_commit": "08e030d9b0d6f3c5c2f154ec28bf2ccb37cab375",
            "runtime_stack": "fsdp_vllm",
            "strict_compatibility": False,
            "model_path": "Qwen/Qwen2.5-7B",
            "trust_remote_code": False,
            "actor": {
                "ppo_micro_batch_size_per_gpu": 1,
                "grad_clip": 1.0,
                "ppo_epochs": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "gpu_memory_utilization": 0.5,
                "enforce_eager": True,
            },
            "reference_policy": True,
            "critic": False,
        },
    }
    if variant == "grpo":
        payload["reward"]["overlong"]["enabled"] = False
    if variant == "gdpo":
        payload["reward"]["overlong"]["enabled"] = False
        payload["algorithm"]["group_filtering"]["enabled"] = False
        payload["algorithm"]["rollout_behavior"]["accumulate_filtered_groups"] = False
        payload["algorithm"]["policy_loss"]["clip_ratio_high"] = payload["algorithm"]["policy_loss"]["clip_ratio"]
    return validate_experiment_config(ExperimentConfig.from_dict(payload))


def make_trajectory(
    *,
    prompt_id: str,
    response: str,
    ground_truth: str = "4",
    old_log_probs: list[float] | None = None,
    new_log_probs: list[float] | None = None,
    response_mask: list[int] | None = None,
    metrics: dict | None = None,
) -> Trajectory:
    return Trajectory(
        prompt_id=prompt_id,
        prompt=f"Prompt {prompt_id}",
        response=response,
        ground_truth=ground_truth,
        response_length=len(response_mask or old_log_probs or new_log_probs or [1]),
        old_log_probs=list(old_log_probs or [0.0]),
        new_log_probs=list(new_log_probs or [0.0]),
        response_mask=list(response_mask or [1] * len(old_log_probs or new_log_probs or [1])),
        metrics=dict(metrics or {}),
    )


def make_group(prompt_id: str, trajectories: list[Trajectory]) -> PromptGroup:
    return PromptGroup(prompt_id=prompt_id, trajectories=trajectories)


def make_batch(groups: list[PromptGroup]) -> BatchContext:
    return BatchContext(groups=groups, metadata={})


class StaticRewardComposer:
    def score_batch(self, batch: BatchContext) -> dict[str, float]:
        rewards = []
        raw_history: dict[str, list[float]] = {}
        weighted_history: dict[str, list[float]] = {}
        for trajectory in batch.iter_trajectories():
            raw_components = dict(trajectory.metrics.get("seed_reward_components", {}))
            if raw_components:
                component_weights = dict(trajectory.metrics.get("seed_component_weights", {}))
                weighted_components = {
                    key: float(raw_value) * float(component_weights.get(key, 1.0))
                    for key, raw_value in raw_components.items()
                }
                reward = sum(weighted_components.values())
            else:
                reward = float(trajectory.metrics["seed_reward"])
                raw_components = {"seed": reward}
                weighted_components = {"seed": reward}
            rewards.append(reward)
            trajectory.set_reward_components(
                raw_components=raw_components,
                weighted_components=weighted_components,
                total_reward=reward,
                reward_details={},
            )
            trajectory.metrics["score"] = reward
            for key, value in raw_components.items():
                raw_history.setdefault(key, []).append(float(value))
            for key, value in weighted_components.items():
                weighted_history.setdefault(key, []).append(float(value))
        metrics = {"reward/mean": sum(rewards) / len(rewards) if rewards else 0.0}
        for key, values in raw_history.items():
            metrics[f"reward/raw/{key}/mean"] = sum(values) / len(values)
        for key, values in weighted_history.items():
            metrics[f"reward/{key}/mean"] = sum(values) / len(values)
        return metrics


@pytest.fixture
def dapo_config() -> ExperimentConfig:
    return build_config("dapo")


@pytest.fixture
def grpo_config() -> ExperimentConfig:
    return build_config("grpo")


@pytest.fixture
def gdpo_config() -> ExperimentConfig:
    return build_config("gdpo")
