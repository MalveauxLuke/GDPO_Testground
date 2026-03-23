from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dapo_lab.algorithms.registry import resolve_variant_hooks
from dapo_lab.algorithms.filtering import FilteringResult, accumulate_filtered_batches, filter_groups
from dapo_lab.algorithms.losses import LossResult, compute_policy_loss
from dapo_lab.algorithms.overlong import OverlongResult, apply_overlong_policy
from dapo_lab.config_schema import ExperimentConfig
from dapo_lab.diagnostics import DiagnosticsRecorder
from dapo_lab.rewards.composition import RewardComposer
from dapo_lab.trainer.state import BatchContext


@dataclass(slots=True)
class LoopOutcome:
    batch: BatchContext
    metrics: dict[str, float]
    loss: float
    stage_order: list[str] = field(default_factory=list)


class TrainerLoop:
    def __init__(
        self,
        config: ExperimentConfig,
        *,
        reward_composer: RewardComposer | None = None,
        diagnostics: DiagnosticsRecorder | None = None,
    ) -> None:
        self.config = config
        self.variant_hooks = resolve_variant_hooks(config.algorithm)
        self.variant = self.variant_hooks.spec
        self.reward_composer = reward_composer or RewardComposer.from_configs(config.reward.terms)
        self.diagnostics = diagnostics or DiagnosticsRecorder()

    def _record_stage(self, stage: str, detail: str = "") -> None:
        if self.config.trainer.diagnostics.record_stage_events:
            self.diagnostics.record_stage(stage, detail=detail)

    def apply_rewards(self, batch: BatchContext) -> dict[str, float]:
        self._record_stage("reward")
        return self.reward_composer.score_batch(batch)

    def apply_kl(self, batch: BatchContext) -> dict[str, float]:
        self._record_stage("kl")
        if not self.config.algorithm.kl.enabled:
            return {"kl/enabled": 0.0}
        return {"kl/enabled": 1.0}

    def apply_filtering(self, batches: list[BatchContext]) -> FilteringResult:
        self._record_stage("filtering")
        filtering = self.config.algorithm.group_filtering
        if self.variant.supports_group_filtering and self.config.algorithm.rollout_behavior.accumulate_filtered_groups:
            return accumulate_filtered_batches(
                batches,
                target_prompt_count=self.config.data.train_batch_size,
                config=filtering,
            )
        return filter_groups(batches[0], filtering)

    def apply_overlong(self, batch: BatchContext) -> OverlongResult:
        return apply_overlong_policy(
            batch,
            self.config.reward.overlong,
            max_response_length=self.config.data.max_response_length,
        )

    def apply_advantages(self, batch: BatchContext) -> dict[str, float]:
        self._record_stage("advantage")
        return self.variant_hooks.apply_advantages(batch, self.config.algorithm)

    def apply_actor_update(self, batch: BatchContext) -> LossResult:
        self._record_stage("actor_update")
        return compute_policy_loss(batch, self.config.algorithm.policy_loss)

    def run_training_step(self, generated_batches: list[BatchContext]) -> LoopOutcome:
        metrics: dict[str, float] = {}
        self._record_stage("rollout", detail=f"generated_batches={len(generated_batches)}")

        reward_metrics = [self.apply_rewards(batch) for batch in generated_batches]
        if reward_metrics:
            metrics.update(reward_metrics[-1])

        kl_metrics = [self.apply_kl(batch) for batch in generated_batches]
        if kl_metrics:
            metrics.update(kl_metrics[-1])

        filtering_result = self.apply_filtering(generated_batches)
        metrics.update(filtering_result.metrics)
        working_batch = filtering_result.batch

        if self.variant.supports_overlong_policy:
            overlong_result = self.apply_overlong(working_batch)
            metrics.update(overlong_result.metrics)

        advantage_metrics = self.apply_advantages(working_batch)
        metrics.update(advantage_metrics)
        metrics.update(self.variant_hooks.emit_variant_metrics(working_batch, self.config.algorithm))

        loss_result = self.apply_actor_update(working_batch)
        metrics.update(loss_result.metrics)

        self._record_stage("diagnostics")
        self.diagnostics.record_metrics(metrics)
        self.diagnostics.annotate(
            prompt_count=working_batch.prompt_count(),
            trajectory_count=working_batch.trajectory_count(),
        )
        return LoopOutcome(
            batch=working_batch,
            metrics=metrics,
            loss=loss_result.loss,
            stage_order=self.diagnostics.ordered_stage_names(),
        )
