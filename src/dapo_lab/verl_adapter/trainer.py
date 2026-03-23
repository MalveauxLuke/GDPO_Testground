from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from dapo_lab.config_schema import ExperimentConfig
from dapo_lab.diagnostics import DiagnosticsRecorder
from dapo_lab.trainer.loop import TrainerLoop
from dapo_lab.verl_adapter.batch_adapter import (
    compute_behavior_delta,
    extract_local_batch,
    prepare_actor_update_batch,
)

try:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer as _VerlRayPPOTrainer  # type: ignore
except Exception:  # pragma: no cover - exercised only when verl is absent
    class _VerlRayPPOTrainer:  # type: ignore
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass


class ResearchTrainer(_VerlRayPPOTrainer):
    """Thin shim that keeps the research loop local while delegating workers to verl."""

    def __init__(
        self,
        *,
        experiment_config: ExperimentConfig,
        diagnostics: DiagnosticsRecorder | None = None,
        trainer_loop: TrainerLoop | None = None,
        **verl_kwargs: Any,
    ) -> None:
        self.experiment_config = experiment_config
        self.diagnostics = diagnostics or DiagnosticsRecorder()
        self.trainer_loop = trainer_loop or TrainerLoop(experiment_config, diagnostics=self.diagnostics)
        self.completed_steps = 0
        self.completed_actor_updates = 0
        super().__init__(**verl_kwargs)

    def fit_local_batches(self, generated_batches: list[Any]) -> Any:
        local_batches = [
            self.build_local_batch(upstream_batch, source_batch_index=batch_index)
            for batch_index, upstream_batch in enumerate(generated_batches)
        ]
        return self.trainer_loop.run_training_step(local_batches)

    def build_local_batch(self, upstream_batch: Any, *, source_batch_index: int = 0) -> Any:
        return extract_local_batch(upstream_batch, tokenizer=getattr(self, "tokenizer", None), source_batch_index=source_batch_index)

    def apply_outcome_to_upstream_batch(self, outcome: Any, upstream_batches: list[Any]) -> Any:
        prepared = prepare_actor_update_batch(outcome, upstream_batches)
        outcome.metrics["certify/adapter_prompt_count"] = float(prepared.prompt_count)
        outcome.metrics["certify/adapter_trajectory_count"] = float(prepared.trajectory_count)
        return prepared.batch

    def emit_upstream_metrics(self, outcome: Any) -> None:  # pragma: no cover
        if hasattr(self, "logger"):
            self.logger.log(data=outcome.metrics, step=getattr(self, "global_steps", 0))

    def _extract_log_prob_rows(self, upstream_batch: Any, field_name: str = "old_log_probs") -> list[list[float]]:
        batch = getattr(upstream_batch, "batch", upstream_batch if isinstance(upstream_batch, dict) else None)
        if batch is None:
            return []
        if isinstance(batch, dict):
            value = batch.get(field_name)
        else:
            value = batch[field_name]
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [list(map(float, row)) for row in value]

    def _compute_post_update_delta(self, actor_batch: Any, pre_update_log_probs: list[list[float]]) -> float:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None or not hasattr(actor_rollout_wg, "compute_log_prob"):
            return 0.0
        post_update = actor_rollout_wg.compute_log_prob(actor_batch)
        post_update_log_probs = self._extract_log_prob_rows(post_update, field_name="old_log_probs")
        if not post_update_log_probs:
            post_update_log_probs = self._extract_log_prob_rows(post_update, field_name="log_probs")
        if not post_update_log_probs:
            return 0.0
        return compute_behavior_delta(pre_update_log_probs, post_update_log_probs)

    def update_actor_from_outcome(self, actor_batch: Any) -> dict[str, float]:
        actor_rollout_wg = getattr(self, "actor_rollout_wg", None)
        if actor_rollout_wg is None or not hasattr(actor_rollout_wg, "update_actor"):
            raise RuntimeError("ResearchTrainer.update_actor_from_outcome requires actor_rollout_wg.update_actor from verl.")

        pre_update_log_probs = self._extract_log_prob_rows(actor_batch, field_name="old_log_probs")
        update_result = actor_rollout_wg.update_actor(actor_batch)
        metrics: dict[str, float] = {}
        if isinstance(update_result, dict):
            metrics.update({str(key): float(value) for key, value in update_result.items() if isinstance(value, (int, float))})
        elif hasattr(update_result, "items"):
            metrics.update({str(key): float(value) for key, value in update_result.items() if isinstance(value, (int, float))})

        actor_delta = self._compute_post_update_delta(actor_batch, pre_update_log_probs)
        metrics["certify/actor_param_delta_l2"] = actor_delta
        metrics["trainer/actor_updates_completed"] = 1.0
        return metrics

    def write_runtime_report(self, outcome: Any) -> None:
        report_path = self.experiment_config.trainer.diagnostics.report_path
        if not report_path:
            return
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "experiment": self.experiment_config.experiment.name,
            "variant": self.experiment_config.algorithm.variant,
            "stage_order": outcome.stage_order,
            "loss": outcome.loss,
            "metrics": outcome.metrics,
            "annotations": self.diagnostics.annotations,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def fit(self) -> None:  # pragma: no cover - requires a real verl installation
        if not hasattr(self, "train_dataloader"):
            raise RuntimeError("ResearchTrainer.fit requires a live verl runtime with initialized workers.")

        self.global_steps = getattr(self, "global_steps", 0)
        for batch_dict in self.train_dataloader:
            generated_batches = self.collect_generated_batches(batch_dict)
            outcome = self.fit_local_batches(generated_batches)
            actor_batch = self.apply_outcome_to_upstream_batch(outcome, generated_batches)
            actor_metrics = self.update_actor_from_outcome(actor_batch)
            outcome.metrics.update(actor_metrics)
            self.completed_actor_updates += int(actor_metrics.get("trainer/actor_updates_completed", 0.0))
            self.completed_steps += 1
            outcome.metrics["trainer/steps_completed"] = float(self.completed_steps)
            outcome.metrics["trainer/actor_updates_completed"] = float(self.completed_actor_updates)
            self.emit_upstream_metrics(outcome)
            self.write_runtime_report(outcome)
            self.global_steps += 1
            max_steps = self.experiment_config.trainer.max_steps
            if max_steps > 0 and self.completed_steps >= max_steps:
                break

    def collect_generated_batches(self, batch_dict: Any) -> list[Any]:  # pragma: no cover - requires a live verl installation
        if not hasattr(self, "_get_gen_batch") or not hasattr(self, "async_rollout_manager"):
            raise RuntimeError(
                "ResearchTrainer.collect_generated_batches expects the standard verl rollout plumbing to be initialized."
            )

        try:
            from verl import DataProto  # type: ignore
        except Exception as error:  # pragma: no cover
            raise RuntimeError("collect_generated_batches requires a live verl installation.") from error

        source_batch = DataProto.from_single_dict(batch_dict)
        generated_batches: list[Any] = []
        max_batches = self.experiment_config.algorithm.group_filtering.max_num_gen_batches or 1
        for _ in range(max_batches):
            gen_batch = self._get_gen_batch(source_batch)
            gen_batch = gen_batch.repeat(repeat_times=self.experiment_config.data.rollout_n, interleave=True)
            generated_batches.append(self.async_rollout_manager.generate_sequences(gen_batch))
            if not self.experiment_config.algorithm.rollout_behavior.accumulate_filtered_groups:
                break
        return generated_batches
