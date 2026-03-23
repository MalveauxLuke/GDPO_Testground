import math

from dapo_lab.algorithms.losses import aggregate_losses, compute_policy_loss
from dapo_lab.config_schema import PolicyLossConfig

from .conftest import make_batch, make_group, make_trajectory


def test_grpo_symmetric_clipping_is_used() -> None:
    trajectory = make_trajectory(
        prompt_id="p1",
        response="a",
        old_log_probs=[0.0],
        new_log_probs=[math.log(1.4)],
    )
    trajectory.token_advantages = [1.0]
    batch = make_batch([make_group("p1", [trajectory])])

    result = compute_policy_loss(
        batch,
        PolicyLossConfig(
            mode="clipped",
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
            clip_ratio_c=3.0,
            loss_agg_mode="token-mean",
        ),
    )

    assert round(result.loss, 5) == -1.2
    assert result.metrics["actor/pg_clipfrac"] == 1.0
    assert result.metrics["actor/pg_clipfrac_lower"] == 0.0


def test_dapo_asymmetric_clip_and_dual_clip_lower_bound() -> None:
    positive = make_trajectory(
        prompt_id="p1",
        response="a",
        old_log_probs=[0.0],
        new_log_probs=[math.log(1.4)],
    )
    positive.token_advantages = [1.0]
    negative = make_trajectory(
        prompt_id="p2",
        response="b",
        old_log_probs=[0.0],
        new_log_probs=[math.log(5.0)],
    )
    negative.token_advantages = [-1.0]
    batch = make_batch([make_group("p1", [positive]), make_group("p2", [negative])])

    result = compute_policy_loss(
        batch,
        PolicyLossConfig(
            mode="clipped",
            clip_ratio=0.2,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            clip_ratio_c=3.0,
            loss_agg_mode="token-mean",
        ),
    )

    assert round(result.loss, 5) == round((-1.28 + 3.0) / 2.0, 5)
    assert result.metrics["actor/pg_clipfrac"] == 0.5
    assert result.metrics["actor/pg_clipfrac_lower"] == 0.5


def test_loss_aggregation_modes() -> None:
    loss_rows = [[1.0, 2.0], [3.0, 4.0]]
    mask_rows = [[1, 0], [1, 1]]

    assert aggregate_losses(loss_rows, mask_rows, "token-mean") == (1.0 + 3.0 + 4.0) / 3.0
    assert aggregate_losses(loss_rows, mask_rows, "seq-mean-token-sum") == (1.0 + 7.0) / 2.0
    assert aggregate_losses(loss_rows, mask_rows, "seq-mean-token-mean") == (1.0 + 3.5) / 2.0
