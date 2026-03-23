import pytest

from dapo_lab.config_schema import ExperimentConfig
from dapo_lab.validation import ConfigValidationError, validate_experiment_config

from .conftest import build_config


def test_grpo_rejects_dapo_specific_filtering() -> None:
    config = build_config("grpo")
    config.algorithm.group_filtering.enabled = True
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_dapo_requires_a_dapo_surface() -> None:
    payload = {
        "experiment": {"name": "bad", "seed": 1, "output_dir": "outputs/bad"},
        "data": {
            "train_files": ["train.jsonl"],
            "val_files": ["val.jsonl"],
            "format": "jsonl",
            "prompt_key": "prompt",
            "answer_key": "ground_truth",
            "train_batch_size": 2,
            "gen_batch_size": 2,
            "max_prompt_length": 10,
            "max_response_length": 10,
            "rollout_n": 2,
        },
        "reward": {
            "terms": [{"name": "accuracy", "kind": "math_accuracy", "weight": 1.0}],
            "overlong": {"enabled": False, "mode": "disabled", "buffer_length": 0, "penalty_factor": 0.0},
        },
        "algorithm": {
            "variant": "dapo",
            "advantage": {"mode": "grpo", "normalize_by_std": True},
            "policy_loss": {
                "mode": "clipped",
                "clip_ratio": 0.2,
                "clip_ratio_low": None,
                "clip_ratio_high": None,
                "clip_ratio_c": 3.0,
                "loss_agg_mode": "token-mean",
            },
            "group_filtering": {"enabled": False, "metric": "acc", "max_num_gen_batches": 0, "require_variance": True},
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
                "accumulate_filtered_groups": False,
            },
        },
        "trainer": {"log_level": "info", "diagnostics": {}, "save_freq": 10, "test_freq": 10, "val_before_train": True},
        "verl": {
            "required_commit": "08e030d9b0d6f3c5c2f154ec28bf2ccb37cab375",
            "runtime_stack": "fsdp_vllm",
            "strict_compatibility": False,
            "model_path": "Qwen/Qwen2.5-7B",
            "trust_remote_code": False,
            "actor": {"ppo_micro_batch_size_per_gpu": 1, "grad_clip": 1.0, "ppo_epochs": 1},
            "rollout": {"tensor_model_parallel_size": 1, "gpu_memory_utilization": 0.5, "enforce_eager": True},
            "reference_policy": True,
            "critic": False,
        },
    }

    with pytest.raises(ConfigValidationError):
        validate_experiment_config(ExperimentConfig.from_dict(payload))


def test_gdpo_rejects_unknown_component_key() -> None:
    config = build_config("gdpo")
    config.algorithm.gdpo.component_keys = ["accuracy", "missing"]
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_gdpo_rejects_mismatched_component_weights() -> None:
    config = build_config("gdpo")
    config.algorithm.gdpo.component_weights = [1.0]
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_gdpo_requires_multiple_components_without_ablation_opt_in() -> None:
    config = build_config("gdpo")
    config.algorithm.gdpo.component_keys = ["accuracy"]
    config.algorithm.gdpo.component_weights = [1.0]
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_gdpo_rejects_dapo_only_surfaces() -> None:
    config = build_config("gdpo")
    config.algorithm.group_filtering.enabled = True
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)


def test_gdpo_rejects_dual_clip_ratio_c() -> None:
    config = build_config("gdpo")
    config.algorithm.policy_loss.clip_ratio_c = 3.0
    with pytest.raises(ConfigValidationError):
        validate_experiment_config(config)
