from __future__ import annotations

import json
from pathlib import Path

from dapo_lab.trainer.loop import LoopOutcome
from dapo_lab.verl_adapter.trainer import ResearchTrainer

from .conftest import build_config


class FakeUpstreamBatch:
    def __init__(self, batch: dict, non_tensor_batch: dict | None = None, meta_info: dict | None = None) -> None:
        self.batch = batch
        self.non_tensor_batch = non_tensor_batch or {}
        self.meta_info = meta_info or {}

    def __getitem__(self, indices):
        if not isinstance(indices, list):
            raise TypeError("FakeUpstreamBatch supports list indexing only.")
        batch = {key: [value[index] for index in indices] for key, value in self.batch.items()}
        non_tensor = {key: [value[index] for index in indices] for key, value in self.non_tensor_batch.items()}
        return FakeUpstreamBatch(batch=batch, non_tensor_batch=non_tensor, meta_info=dict(self.meta_info))

    @classmethod
    def concat(cls, batches: list["FakeUpstreamBatch"]) -> "FakeUpstreamBatch":
        merged_batch: dict[str, list] = {}
        merged_non_tensor: dict[str, list] = {}
        for key in batches[0].batch:
            merged_batch[key] = []
            for batch in batches:
                merged_batch[key].extend(batch.batch[key])
        for key in batches[0].non_tensor_batch:
            merged_non_tensor[key] = []
            for batch in batches:
                merged_non_tensor[key].extend(batch.non_tensor_batch[key])
        return cls(batch=merged_batch, non_tensor_batch=merged_non_tensor, meta_info=dict(batches[0].meta_info))


class FakeActorRolloutWG:
    def update_actor(self, _batch):
        return {"actor/grad_norm": 1.0}

    def compute_log_prob(self, batch):
        bumped = []
        for row in batch.batch["old_log_probs"]:
            bumped.append([value + 0.1 for value in row])
        return FakeUpstreamBatch(batch={"old_log_probs": bumped}, non_tensor_batch={})


def _make_upstream_batch() -> FakeUpstreamBatch:
    return FakeUpstreamBatch(
        batch={
            "old_log_probs": [[0.0, 0.0], [0.0, 0.0]],
            "new_log_probs": [[0.1, 0.1], [0.1, 0.1]],
            "response_mask": [[1, 1], [1, 1]],
            "responses": [[1, 2], [3, 4]],
        },
        non_tensor_batch={
            "prompt_id": ["p1", "p1"],
            "prompt": ["What is 2 + 2?", "What is 2 + 2?"],
            "ground_truth": ["4", "4"],
            "response_text": ["Scratch work.\nAnswer: \\boxed{4}", "Scratch work.\nAnswer: 5"],
        },
        meta_info={"source": "fake"},
    )


def test_build_local_batch_extracts_prompt_groups_and_source_refs() -> None:
    trainer = ResearchTrainer(experiment_config=build_config("grpo"))

    batch = trainer.build_local_batch(_make_upstream_batch(), source_batch_index=3)

    assert batch.prompt_count() == 1
    trajectories = list(batch.iter_trajectories())
    assert len(trajectories) == 2
    assert trajectories[0].prompt_id == "p1"
    assert trajectories[0].upstream_batch_index == 3
    assert trajectories[1].upstream_row_index == 1


def test_apply_outcome_to_upstream_batch_materializes_actor_batch() -> None:
    trainer = ResearchTrainer(experiment_config=build_config("grpo"))
    upstream_batch = _make_upstream_batch()

    outcome = trainer.fit_local_batches([upstream_batch])
    actor_batch = trainer.apply_outcome_to_upstream_batch(outcome, [upstream_batch])

    assert actor_batch.batch["advantages"]
    assert actor_batch.batch["returns"]
    assert actor_batch.meta_info["dapo_lab_prompt_count"] == 1
    assert actor_batch.meta_info["dapo_lab_trajectory_count"] == 2
    assert outcome.metrics["certify/adapter_prompt_count"] == 1.0


def test_update_actor_from_outcome_emits_delta_metric() -> None:
    trainer = ResearchTrainer(experiment_config=build_config("grpo"))
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    actor_batch = _make_upstream_batch()

    metrics = trainer.update_actor_from_outcome(actor_batch)

    assert metrics["trainer/actor_updates_completed"] == 1.0
    assert metrics["certify/actor_param_delta_l2"] > 0.0


def test_fit_respects_max_steps_and_writes_report(tmp_path: Path) -> None:
    config = build_config("grpo")
    config.trainer.max_steps = 1
    config.trainer.diagnostics.report_path = str(tmp_path / "runtime_report.json")
    trainer = ResearchTrainer(experiment_config=config)
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    trainer.train_dataloader = [{"seed": 1}, {"seed": 2}]
    upstream_batch = _make_upstream_batch()

    def fake_collect(_batch_dict):
        return [upstream_batch]

    trainer.collect_generated_batches = fake_collect  # type: ignore[method-assign]

    trainer.fit()

    payload = json.loads((tmp_path / "runtime_report.json").read_text())
    assert payload["metrics"]["trainer/steps_completed"] == 1.0
    assert payload["metrics"]["trainer/actor_updates_completed"] == 1.0
    assert payload["metrics"]["certify/actor_param_delta_l2"] > 0.0
