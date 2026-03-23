from __future__ import annotations

from dataclasses import dataclass
from statistics import pstdev

from dapo_lab.config_schema import GroupFilteringConfig
from dapo_lab.trainer.state import BatchContext, PromptGroup


@dataclass(slots=True)
class FilteringResult:
    batch: BatchContext
    metrics: dict[str, float]
    kept_prompt_ids: list[str]
    dropped_prompt_ids: list[str]
    generation_batches_used: int = 1


def _group_varies(group: PromptGroup, metric_name: str, require_variance: bool) -> bool:
    metric_values = group.metric_values(metric_name)
    if len(metric_values) <= 1:
        return True
    if not require_variance:
        return True
    return pstdev(metric_values) > 0


def filter_groups(batch: BatchContext, config: GroupFilteringConfig) -> FilteringResult:
    if not config.enabled:
        return FilteringResult(
            batch=batch,
            metrics={"filtering/kept_prompts": float(batch.prompt_count()), "filtering/dropped_prompts": 0.0},
            kept_prompt_ids=[group.prompt_id for group in batch.groups],
            dropped_prompt_ids=[],
        )

    kept_groups: list[PromptGroup] = []
    kept_prompt_ids: list[str] = []
    dropped_prompt_ids: list[str] = []
    for group in batch.groups:
        if _group_varies(group, config.metric, config.require_variance):
            kept_groups.append(group)
            kept_prompt_ids.append(group.prompt_id)
        else:
            dropped_prompt_ids.append(group.prompt_id)
            for trajectory in group.trajectories:
                trajectory.filtered_out = True
                trajectory.filter_reason = f"constant_{config.metric}"

    kept_batch = batch.clone_with_groups(kept_groups)
    metrics = {
        "filtering/kept_prompts": float(len(kept_prompt_ids)),
        "filtering/dropped_prompts": float(len(dropped_prompt_ids)),
    }
    return FilteringResult(
        batch=kept_batch,
        metrics=metrics,
        kept_prompt_ids=kept_prompt_ids,
        dropped_prompt_ids=dropped_prompt_ids,
    )


def accumulate_filtered_batches(
    generated_batches: list[BatchContext],
    *,
    target_prompt_count: int,
    config: GroupFilteringConfig,
) -> FilteringResult:
    accumulated_groups: list[PromptGroup] = []
    dropped_prompt_ids: list[str] = []
    batches_used = 0

    for batch in generated_batches:
        batches_used += 1
        result = filter_groups(batch, config)
        accumulated_groups.extend(result.batch.groups)
        dropped_prompt_ids.extend(result.dropped_prompt_ids)
        if len(accumulated_groups) >= target_prompt_count:
            break
        if config.max_num_gen_batches > 0 and batches_used >= config.max_num_gen_batches:
            break

    kept_groups = accumulated_groups[:target_prompt_count]
    metadata = dict(generated_batches[0].metadata) if generated_batches else {}
    metadata["generation_batches_used"] = batches_used
    final_batch = BatchContext(groups=kept_groups, metadata=metadata)
    return FilteringResult(
        batch=final_batch,
        metrics={
            "filtering/kept_prompts": float(len(kept_groups)),
            "filtering/dropped_prompts": float(len(dropped_prompt_ids)),
        },
        kept_prompt_ids=[group.prompt_id for group in kept_groups],
        dropped_prompt_ids=dropped_prompt_ids,
        generation_batches_used=batches_used,
    )
