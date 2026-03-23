from dapo_lab.diagnostics import DiagnosticsRecorder
from dapo_lab.trainer.loop import TrainerLoop

from .conftest import StaticRewardComposer, dapo_config, gdpo_config, grpo_config, make_batch, make_group, make_trajectory


def _seeded_group(
    prompt_id: str,
    rewards: list[float],
    accs: list[float],
    ratio: float = 1.4,
    reward_components: list[dict[str, float]] | None = None,
):
    trajectories = []
    for index, (reward, acc) in enumerate(zip(rewards, accs, strict=True), start=1):
        metrics = {"seed_reward": reward, "acc": acc}
        if reward_components is not None:
            metrics["seed_reward_components"] = reward_components[index - 1]
            metrics["seed_component_weights"] = {"accuracy": 1.0, "boxed_format": 0.1}
        trajectory = make_trajectory(
            prompt_id=prompt_id,
            response=f"resp-{prompt_id}-{index}",
            old_log_probs=[0.0],
            new_log_probs=[0.3364722366212129 if ratio == 1.4 else 0.0],
            response_mask=[1],
            metrics=metrics,
        )
        trajectories.append(trajectory)
    return make_group(prompt_id, trajectories)


def test_grpo_loop_keeps_shared_stage_order(grpo_config) -> None:
    batch = make_batch([_seeded_group("p1", rewards=[-1.0, 1.0], accs=[0.0, 1.0])])
    loop = TrainerLoop(grpo_config, reward_composer=StaticRewardComposer(), diagnostics=DiagnosticsRecorder())

    outcome = loop.run_training_step([batch])

    assert outcome.stage_order == ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
    assert outcome.metrics["filtering/kept_prompts"] == 1.0
    assert outcome.batch.prompt_count() == 1


def test_dapo_loop_accumulates_filtered_batches(dapo_config) -> None:
    constant = make_batch([_seeded_group("drop", rewards=[1.0, 1.0], accs=[1.0, 1.0])])
    varying = make_batch(
        [
            _seeded_group("keep-a", rewards=[-1.0, 1.0], accs=[0.0, 1.0]),
            _seeded_group("keep-b", rewards=[1.0, -1.0], accs=[1.0, 0.0]),
        ]
    )
    loop = TrainerLoop(dapo_config, reward_composer=StaticRewardComposer(), diagnostics=DiagnosticsRecorder())

    outcome = loop.run_training_step([constant, varying])

    assert outcome.batch.prompt_count() == 2
    assert outcome.metrics["filtering/kept_prompts"] == 2.0
    assert outcome.stage_order[0] == "rollout"
    assert outcome.stage_order[-1] == "diagnostics"
    assert outcome.stage_order.count("reward") == 2


def test_same_batch_differs_between_grpo_and_dapo(grpo_config, dapo_config) -> None:
    seed_batch = make_batch([_seeded_group("p1", rewards=[-1.0, 1.0], accs=[0.0, 1.0])])

    grpo_outcome = TrainerLoop(
        grpo_config,
        reward_composer=StaticRewardComposer(),
        diagnostics=DiagnosticsRecorder(),
    ).run_training_step([seed_batch])

    dapo_batch = make_batch([_seeded_group("p1", rewards=[-1.0, 1.0], accs=[0.0, 1.0])])
    for trajectory in dapo_batch.iter_trajectories():
        trajectory.response_mask = [1] * 10
        trajectory.old_log_probs = [0.0] * 10
        trajectory.new_log_probs = [0.3364722366212129] * 10
        trajectory.response_length = 10
    dapo_outcome = TrainerLoop(
        dapo_config,
        reward_composer=StaticRewardComposer(),
        diagnostics=DiagnosticsRecorder(),
    ).run_training_step([dapo_batch])

    assert grpo_outcome.loss != dapo_outcome.loss
    assert dapo_outcome.metrics["overlong/penalized"] == 2.0
    assert "overlong/penalized" not in grpo_outcome.metrics


def test_gdpo_loop_uses_shared_order_and_emits_component_metrics(gdpo_config) -> None:
    reward_components = [{"accuracy": 1.0, "boxed_format": 0.0}, {"accuracy": 0.0, "boxed_format": 10.0}]
    batch = make_batch([_seeded_group("p1", rewards=[1.0, 1.0], accs=[1.0, 1.0], reward_components=reward_components)])
    loop = TrainerLoop(gdpo_config, reward_composer=StaticRewardComposer(), diagnostics=DiagnosticsRecorder())

    outcome = loop.run_training_step([batch])

    assert outcome.stage_order == ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
    assert outcome.metrics["gdpo/accuracy/mean"] == 0.5
    assert outcome.metrics["gdpo/boxed_format/max"] == 10.0
    assert outcome.metrics["gdpo/combined_advantage_mean"] == 0.0
    assert outcome.batch.prompt_count() == 1
