from dapo_lab.algorithms.overlong import apply_overlong_policy, compute_overlong_penalty
from dapo_lab.config_schema import OverlongConfig

from .conftest import make_batch, make_group, make_trajectory


def test_overlong_shaping_matches_dapo_formula() -> None:
    penalty = compute_overlong_penalty(
        response_length=9,
        max_response_length=10,
        buffer_length=2,
        penalty_factor=1.0,
    )
    assert penalty == -0.5


def test_overlong_policy_can_shape_and_filter() -> None:
    trajectory = make_trajectory(prompt_id="p1", response="a", response_mask=[1] * 10)
    trajectory.reward = 1.0
    batch = make_batch([make_group("p1", [trajectory])])

    result = apply_overlong_policy(
        batch,
        OverlongConfig(
            enabled=True,
            mode="shape_and_filter",
            buffer_length=2,
            penalty_factor=1.0,
            hard_filter=False,
            log=True,
        ),
        max_response_length=10,
    )

    assert trajectory.reward < 1.0
    assert trajectory.filtered_out is True
    assert result.metrics["overlong/penalized"] == 1.0
    assert result.metrics["overlong/filtered"] == 1.0
