from dapo_lab.verl_adapter.config_bridge import build_verl_config

from .conftest import build_config


def test_build_verl_config_keeps_bridge_small_and_direct() -> None:
    config = build_config("dapo")
    bridged = build_verl_config(config)

    assert sorted(bridged.keys()) == sorted([
        "actor_rollout_ref",
        "critic",
        "algorithm",
        "custom_reward_function",
        "dapo_lab",
        "data",
        "global_profiler",
        "ray_kwargs",
        "reward",
        "reward_model",
        "sandbox_fusion",
        "transfer_queue",
        "trainer",
    ])
    assert bridged["algorithm"]["adv_estimator"] == "grpo"
    assert bridged["algorithm"]["use_kl_in_reward"] is False
    assert bridged["actor_rollout_ref"]["actor"]["clip_ratio_high"] == 0.28
    assert bridged["reward"]["reward_manager"]["name"] == "dapo"
    assert bridged["reward"]["reward_kwargs"]["overlong_buffer_cfg"]["enable"] is True
    assert bridged["reward_model"]["reward_manager"] is None
    assert bridged["reward"]["reward_model"]["enable"] is False
    assert bridged["ray_kwargs"]["ray_init"]["num_cpus"] == 1
    assert bridged["transfer_queue"]["enable"] is False
    assert bridged["trainer"]["n_gpus_per_node"] == 1
    assert bridged["critic"]["enable"] is False
    assert bridged["actor_rollout_ref"]["actor"]["use_dynamic_bsz"] is False
    assert bridged["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] == config.data.train_batch_size * config.data.rollout_n
    assert bridged["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert bridged["actor_rollout_ref"]["rollout"]["val_kwargs"]["do_sample"] is False


def test_build_verl_config_passes_gdpo_keys_and_weights() -> None:
    config = build_config("gdpo")
    bridged = build_verl_config(config)

    assert bridged["algorithm"]["adv_estimator"] == "gdpo"
    assert bridged["algorithm"]["gdpo_reward_keys"] == ["accuracy", "boxed_format"]
    assert bridged["algorithm"]["gdpo_reward_weights"] == [1.0, 0.1]
    assert bridged["reward"]["reward_manager"]["name"] == "gdpo"
    assert bridged["reward"]["reward_model"]["rollout"]["name"] == config.algorithm.rollout_behavior.backend
    assert bridged["actor_rollout_ref"]["rollout"]["prompt_length"] == config.data.max_prompt_length
    assert bridged["actor_rollout_ref"]["actor"]["use_kl_loss"] is False
