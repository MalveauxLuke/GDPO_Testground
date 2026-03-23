from dapo_lab.algorithms.filtering import accumulate_filtered_batches, filter_groups
from dapo_lab.config_schema import GroupFilteringConfig

from .conftest import make_batch, make_group, make_trajectory


def test_group_filtering_drops_constant_groups() -> None:
    kept_group = make_group(
        "keep",
        [
            make_trajectory(prompt_id="keep", response="a", metrics={"acc": 1.0}),
            make_trajectory(prompt_id="keep", response="b", metrics={"acc": 0.0}),
        ],
    )
    dropped_group = make_group(
        "drop",
        [
            make_trajectory(prompt_id="drop", response="a", metrics={"acc": 1.0}),
            make_trajectory(prompt_id="drop", response="b", metrics={"acc": 1.0}),
        ],
    )

    result = filter_groups(
        make_batch([kept_group, dropped_group]),
        GroupFilteringConfig(enabled=True, metric="acc", max_num_gen_batches=3, require_variance=True),
    )

    assert [group.prompt_id for group in result.batch.groups] == ["keep"]
    assert result.dropped_prompt_ids == ["drop"]
    assert dropped_group.trajectories[0].filter_reason == "constant_acc"


def test_dynamic_sampling_accumulates_batches_until_target() -> None:
    batch_one = make_batch(
        [
            make_group(
                "drop",
                [
                    make_trajectory(prompt_id="drop", response="a", metrics={"acc": 1.0}),
                    make_trajectory(prompt_id="drop", response="b", metrics={"acc": 1.0}),
                ],
            )
        ]
    )
    batch_two = make_batch(
        [
            make_group(
                "keep-a",
                [
                    make_trajectory(prompt_id="keep-a", response="a", metrics={"acc": 1.0}),
                    make_trajectory(prompt_id="keep-a", response="b", metrics={"acc": 0.0}),
                ],
            ),
            make_group(
                "keep-b",
                [
                    make_trajectory(prompt_id="keep-b", response="a", metrics={"acc": 0.0}),
                    make_trajectory(prompt_id="keep-b", response="b", metrics={"acc": 1.0}),
                ],
            ),
        ]
    )

    result = accumulate_filtered_batches(
        [batch_one, batch_two],
        target_prompt_count=2,
        config=GroupFilteringConfig(enabled=True, metric="acc", max_num_gen_batches=3, require_variance=True),
    )

    assert result.generation_batches_used == 2
    assert [group.prompt_id for group in result.batch.groups] == ["keep-a", "keep-b"]
