from dapo_lab.verl_adapter.config_bridge import build_verl_config

from .conftest import build_config


def test_build_verl_config_keeps_bridge_small_and_direct() -> None:
    config = build_config("dapo")
    bridged = build_verl_config(config)

    assert sorted(bridged.keys()) == [
        "actor_rollout_ref",
        "algorithm",
        "custom_reward_function",
        "dapo_lab",
        "data",
        "reward",
        "reward_model",
        "sandbox_fusion",
        "trainer",
    ]
    assert bridged["algorithm"]["adv_estimator"] == "grpo"
    assert bridged["actor_rollout_ref"]["actor"]["clip_ratio_high"] == 0.28
    assert bridged["reward"]["reward_manager"]["name"] == "dapo"
    assert bridged["reward"]["reward_kwargs"]["overlong_buffer_cfg"]["enable"] is True
    assert bridged["reward_model"]["reward_manager"] is None
    assert bridged["reward"]["reward_model"]["enable"] is False


def test_build_verl_config_passes_gdpo_keys_and_weights() -> None:
    config = build_config("gdpo")
    bridged = build_verl_config(config)

    assert bridged["algorithm"]["adv_estimator"] == "gdpo"
    assert bridged["algorithm"]["gdpo_reward_keys"] == ["accuracy", "boxed_format"]
    assert bridged["algorithm"]["gdpo_reward_weights"] == [1.0, 0.1]
    assert bridged["reward"]["reward_manager"]["name"] == "gdpo"
    assert bridged["reward"]["reward_model"]["rollout"]["name"] == config.algorithm.rollout_behavior.backend
