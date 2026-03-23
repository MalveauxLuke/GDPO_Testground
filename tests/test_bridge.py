from __future__ import annotations

from copy import deepcopy

from dapo_lab.verl_adapter.config_bridge import build_verl_config
from dapo_lab.verl_adapter.contract import PINNED_VERL_COMMIT, audit_bridge_config

from .conftest import build_config


def test_build_verl_config_satisfies_pinned_contract() -> None:
    config = build_config("dapo")
    bridged = build_verl_config(config)
    audit = audit_bridge_config(bridged)

    assert audit.ok, audit.to_dict()
    assert bridged["dapo_lab"]["verl"]["required_commit"] == PINNED_VERL_COMMIT
    assert bridged["algorithm"]["adv_estimator"] == "grpo"
    assert bridged["reward"]["reward_manager"]["name"] == "dapo"
    assert bridged["reward"]["reward_kwargs"]["overlong_buffer_cfg"]["enable"] is True
    assert bridged["reward_model"]["reward_manager"] is None
    assert bridged["reward"]["reward_model"]["enable"] is False
    assert bridged["ray_kwargs"]["ray_init"]["num_cpus"] == 1
    assert bridged["transfer_queue"]["enable"] is False
    assert bridged["trainer"]["n_gpus_per_node"] == 1
    assert bridged["critic"]["enable"] is False
    assert bridged["actor_rollout_ref"]["actor"]["_target_"] == "verl.workers.config.FSDPActorConfig"
    assert bridged["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
    assert bridged["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == config.data.train_batch_size
    assert bridged["actor_rollout_ref"]["rollout"]["_target_"] == "verl.workers.config.RolloutConfig"
    assert bridged["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert bridged["actor_rollout_ref"]["rollout"]["val_kwargs"]["_target_"] == "verl.workers.config.SamplingConfig"
    assert bridged["actor_rollout_ref"]["rollout"]["val_kwargs"]["do_sample"] is False


def test_build_verl_config_preserves_gdpo_overlay() -> None:
    config = build_config("gdpo")
    bridged = build_verl_config(config)
    audit = audit_bridge_config(bridged)

    assert audit.ok, audit.to_dict()
    assert bridged["algorithm"]["adv_estimator"] == "gdpo"
    assert bridged["algorithm"]["gdpo_reward_keys"] == ["accuracy", "boxed_format"]
    assert bridged["algorithm"]["gdpo_reward_weights"] == [1.0, 0.1]
    assert bridged["reward"]["reward_manager"]["name"] == "gdpo"
    assert bridged["reward"]["reward_manager"]["_target_"] == "verl.workers.config.reward_model.RewardManagerConfig"
    assert bridged["reward"]["reward_model"]["rollout"]["name"] == config.algorithm.rollout_behavior.backend
    assert bridged["reward"]["reward_model"]["rollout"]["_target_"] == "verl.workers.config.RolloutConfig"
    assert bridged["actor_rollout_ref"]["rollout"]["prompt_length"] == config.data.max_prompt_length
    assert bridged["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
    assert bridged["algorithm"]["use_kl_in_reward"] is False


def test_contract_audit_catches_missing_legacy_reward_stubs() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["reward_model"]
    del broken["sandbox_fusion"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_top_level == ["reward_model", "sandbox_fusion"]


def test_contract_audit_catches_missing_runtime_top_level_sections() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["ray_kwargs"]
    del broken["transfer_queue"]
    del broken["global_profiler"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_top_level == ["global_profiler", "ray_kwargs", "transfer_queue"]


def test_contract_audit_catches_missing_target_blocks() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["actor_rollout_ref"]["actor"]["_target_"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_target_paths == ["actor_rollout_ref.actor"]


def test_contract_audit_catches_missing_use_dynamic_bsz() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["actor_rollout_ref"]["actor"]["use_dynamic_bsz"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_paths == ["actor_rollout_ref.actor.use_dynamic_bsz"]


def test_contract_audit_catches_missing_data_sampler() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["data"]["sampler"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_paths == sorted(
        [
            "data.sampler.class_name",
            "data.sampler.class_path",
        ]
    )


def test_contract_audit_catches_bad_actor_mini_batch_size() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    broken["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = broken["data"]["train_batch_size"] + 1

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.semantic_errors == [
        "data.train_batch_size must be >= actor_rollout_ref.actor.ppo_mini_batch_size when use_dynamic_bsz is false."
    ]


def test_contract_audit_catches_missing_reference_policy_block() -> None:
    bridged = build_verl_config(build_config("grpo"))
    broken = deepcopy(bridged)
    del broken["actor_rollout_ref"]["ref"]

    audit = audit_bridge_config(broken)

    assert audit.ok is False
    assert audit.missing_paths == sorted([
        "actor_rollout_ref.ref.log_prob_micro_batch_size",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu",
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu",
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz",
    ])
    assert audit.missing_target_paths == sorted([
        "actor_rollout_ref.ref",
        "actor_rollout_ref.ref.fsdp_config",
        "actor_rollout_ref.ref.profiler",
        "actor_rollout_ref.ref.router_replay",
    ])
