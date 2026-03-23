from dapo_lab.algorithms.variants.gdpo import apply_advantages as apply_gdpo_advantages
from dapo_lab.algorithms.variants.grpo import apply_advantages as apply_grpo_advantages

from .conftest import gdpo_config, grpo_config, make_batch, make_group, make_trajectory


def test_grpo_advantage_broadcasts_group_centered_scores(grpo_config) -> None:
    batch = make_batch(
        [
            make_group(
                "p1",
                [
                    make_trajectory(prompt_id="p1", response="a", old_log_probs=[0.0, 0.0], new_log_probs=[0.0, 0.0]),
                    make_trajectory(prompt_id="p1", response="b", old_log_probs=[0.0, 0.0], new_log_probs=[0.0, 0.0]),
                ],
            )
        ]
    )
    batch.groups[0].trajectories[0].set_reward_components(
        raw_components={"accuracy": -1.0},
        weighted_components={"accuracy": -1.0},
        total_reward=-1.0,
        reward_details={},
    )
    batch.groups[0].trajectories[1].set_reward_components(
        raw_components={"accuracy": 1.0},
        weighted_components={"accuracy": 1.0},
        total_reward=1.0,
        reward_details={},
    )

    metrics = apply_grpo_advantages(batch, grpo_config.algorithm)

    first, second = batch.groups[0].trajectories
    assert round(first.seq_advantage, 5) == -1.0
    assert round(second.seq_advantage, 5) == 1.0
    assert first.token_advantages == [-1.0, -1.0]
    assert second.token_advantages == [1.0, 1.0]
    assert metrics["advantage/mean_abs"] == 1.0


def test_gdpo_decouples_reward_components_and_batch_whitens(gdpo_config) -> None:
    batch = make_batch(
        [
            make_group(
                "p1",
                [
                    make_trajectory(prompt_id="p1", response="a", old_log_probs=[0.0, 0.0], new_log_probs=[0.0, 0.0]),
                    make_trajectory(prompt_id="p1", response="b", old_log_probs=[0.0, 0.0], new_log_probs=[0.0, 0.0]),
                ],
            )
        ]
    )
    first, second = batch.groups[0].trajectories
    first.set_reward_components(
        raw_components={"accuracy": 1.0, "boxed_format": 0.0},
        weighted_components={"accuracy": 1.0, "boxed_format": 0.0},
        total_reward=1.0,
        reward_details={},
    )
    second.set_reward_components(
        raw_components={"accuracy": 0.0, "boxed_format": 10.0},
        weighted_components={"accuracy": 0.0, "boxed_format": 1.0},
        total_reward=1.0,
        reward_details={},
    )

    metrics = apply_gdpo_advantages(batch, gdpo_config.algorithm)

    assert round(first.component_advantages["accuracy"], 5) == 1.0
    assert round(second.component_advantages["accuracy"], 5) == -1.0
    assert round(first.component_advantages["boxed_format"], 5) == -1.0
    assert round(second.component_advantages["boxed_format"], 5) == 1.0
    assert round(first.combined_advantage_pre_whiten, 5) == 0.9
    assert round(second.combined_advantage_pre_whiten, 5) == -0.9
    assert round(first.combined_advantage_post_whiten, 5) == 1.0
    assert round(second.combined_advantage_post_whiten, 5) == -1.0
    assert first.token_advantages == [1.0, 1.0]
    assert second.token_advantages == [-1.0, -1.0]
    assert metrics["gdpo/combined_advantage_mean"] == 0.0
    assert metrics["gdpo/combined_advantage_std"] == 1.0


def test_grpo_vs_gdpo_on_same_multi_reward_batch(grpo_config, gdpo_config) -> None:
    grpo_batch = make_batch(
        [
            make_group(
                "p1",
                [
                    make_trajectory(prompt_id="p1", response="a"),
                    make_trajectory(prompt_id="p1", response="b"),
                ],
            )
        ]
    )
    gdpo_batch = make_batch(
        [
            make_group(
                "p1",
                [
                    make_trajectory(prompt_id="p1", response="a"),
                    make_trajectory(prompt_id="p1", response="b"),
                ],
            )
        ]
    )
    grpo_first, grpo_second = grpo_batch.groups[0].trajectories
    gdpo_first, gdpo_second = gdpo_batch.groups[0].trajectories

    for trajectory in (grpo_first, gdpo_first):
        trajectory.set_reward_components(
            raw_components={"accuracy": 1.0, "boxed_format": 0.0},
            weighted_components={"accuracy": 1.0, "boxed_format": 0.0},
            total_reward=1.0,
            reward_details={},
        )
    for trajectory in (grpo_second, gdpo_second):
        trajectory.set_reward_components(
            raw_components={"accuracy": 0.0, "boxed_format": 10.0},
            weighted_components={"accuracy": 0.0, "boxed_format": 1.0},
            total_reward=1.0,
            reward_details={},
        )

    apply_grpo_advantages(grpo_batch, grpo_config.algorithm)
    apply_gdpo_advantages(gdpo_batch, gdpo_config.algorithm)

    assert grpo_first.seq_advantage == 0.0
    assert grpo_second.seq_advantage == 0.0
    assert gdpo_first.seq_advantage == 1.0
    assert gdpo_second.seq_advantage == -1.0
