from dapo_lab.config_schema import RewardTermConfig
from dapo_lab.rewards.composition import RewardComposer
from dapo_lab.rewards.math import compute_math_accuracy, last_boxed_only_string

from .conftest import make_batch, make_group, make_trajectory


def test_weighted_reward_composition_and_metrics() -> None:
    trajectory = make_trajectory(prompt_id="p1", response="Work\nAnswer: \\boxed{4}", ground_truth="4")
    batch = make_batch([make_group("p1", [trajectory])])
    composer = RewardComposer.from_configs(
        [
            RewardTermConfig(name="accuracy", kind="math_accuracy", weight=1.0),
            RewardTermConfig(name="boxed_format", kind="boxed_format", weight=0.1),
        ]
    )

    metrics = composer.score_batch(batch)

    assert round(trajectory.total_reward, 5) == 1.1
    assert trajectory.raw_reward_components == {"accuracy": 1.0, "boxed_format": 1.0}
    assert trajectory.weighted_reward_components == {"accuracy": 1.0, "boxed_format": 0.1}
    assert trajectory.reward_terms == {"accuracy": 1.0, "boxed_format": 0.1}
    assert round(trajectory.reward, 5) == 1.1
    assert metrics["reward/raw/accuracy/mean"] == 1.0
    assert metrics["reward/raw/boxed_format/mean"] == 1.0
    assert metrics["reward/accuracy/mean"] == 1.0
    assert metrics["reward/boxed_format/mean"] == 0.1
    assert trajectory.metrics["acc"] == 1.0


def test_math_reward_detects_wrong_answer() -> None:
    correct, prediction = compute_math_accuracy("Answer: \\boxed{3}", "4")
    assert correct is False
    assert prediction == "3"


def test_last_boxed_only_string_uses_last_box() -> None:
    response = "Answer: \\boxed{1} ... final \\boxed{4}"
    assert last_boxed_only_string(response) == "\\boxed{4}"
