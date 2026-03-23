from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dapo_lab.trainer.loop import LoopOutcome
from dapo_lab.trainer.state import BatchContext, PromptGroup, Trajectory


@dataclass(slots=True)
class PreparedActorBatch:
    batch: Any
    trajectory_order: list[Trajectory]
    prompt_count: int
    trajectory_count: int


def _lookup(container: Any, *names: str) -> Any:
    if container is None:
        return None
    for name in names:
        if isinstance(container, dict) and name in container:
            return container[name]
        if hasattr(container, name):
            return getattr(container, name)
    return None


def _to_python(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def _rows(value: Any) -> list[Any]:
    python_value = _to_python(value)
    if python_value is None:
        return []
    if isinstance(python_value, list):
        return python_value
    if isinstance(python_value, tuple):
        return list(python_value)
    return [python_value]


def _row_value(value: Any, row_index: int, default: Any = None) -> Any:
    rows = _rows(value)
    if row_index >= len(rows):
        return default
    return rows[row_index]


def _batch_fields(upstream_batch: Any) -> dict[str, Any]:
    batch = _lookup(upstream_batch, "batch")
    if batch is None:
        return upstream_batch if isinstance(upstream_batch, dict) else {}
    if isinstance(batch, dict):
        return batch
    return dict(batch.items()) if hasattr(batch, "items") else dict(batch)


def _non_tensor_fields(upstream_batch: Any) -> dict[str, Any]:
    non_tensor = _lookup(upstream_batch, "non_tensor_batch")
    if non_tensor is None:
        return {}
    if isinstance(non_tensor, dict):
        return non_tensor
    return dict(non_tensor.items()) if hasattr(non_tensor, "items") else dict(non_tensor)


def _meta_info(upstream_batch: Any) -> dict[str, Any]:
    meta = _lookup(upstream_batch, "meta_info")
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return dict(meta)
    return dict(meta.items()) if hasattr(meta, "items") else dict(meta)


def _infer_row_count(batch_fields: dict[str, Any], non_tensor_fields: dict[str, Any]) -> int:
    for candidate in [
        _lookup(batch_fields, "responses"),
        _lookup(batch_fields, "old_log_probs"),
        _lookup(batch_fields, "new_log_probs"),
        _lookup(batch_fields, "response_mask"),
        _lookup(batch_fields, "input_ids"),
        _lookup(non_tensor_fields, "prompt"),
        _lookup(non_tensor_fields, "raw_prompt"),
        _lookup(non_tensor_fields, "ground_truth"),
    ]:
        rows = _rows(candidate)
        if rows:
            return len(rows)
    return 0


def _decode_text_rows(tokenizer: Any, token_rows: list[Any]) -> list[str]:
    if not token_rows or tokenizer is None or not hasattr(tokenizer, "batch_decode"):
        return []
    try:
        return [str(item) for item in tokenizer.batch_decode(token_rows, skip_special_tokens=True)]
    except Exception:
        return []


def extract_local_batch(upstream_batch: Any, *, tokenizer: Any = None, source_batch_index: int = 0) -> BatchContext:
    batch_fields = _batch_fields(upstream_batch)
    non_tensor_fields = _non_tensor_fields(upstream_batch)
    meta_info = _meta_info(upstream_batch)
    row_count = _infer_row_count(batch_fields, non_tensor_fields)

    prompt_ids = _rows(
        _lookup(non_tensor_fields, "prompt_id", "prompt_ids", "uid", "uids", "raw_prompt_ids")
    )
    prompts = _rows(_lookup(non_tensor_fields, "prompt", "prompts", "raw_prompt", "raw_prompts"))
    answers = _rows(_lookup(non_tensor_fields, "ground_truth", "ground_truths", "answer", "answers"))
    response_texts = _rows(
        _lookup(
            non_tensor_fields,
            "response_text",
            "response_texts",
            "generated_text",
            "generated_texts",
            "response",
            "responses_text",
        )
    )
    response_tokens = _rows(_lookup(batch_fields, "responses", "response_ids"))
    if not response_texts:
        response_texts = _decode_text_rows(tokenizer, response_tokens)

    groups_by_prompt_id: dict[str, PromptGroup] = {}
    for row_index in range(row_count):
        prompt_id = str(
            _row_value(prompt_ids, row_index, default=_row_value(prompts, row_index, default=f"prompt-{row_index}"))
        )
        prompt = str(_row_value(prompts, row_index, default=prompt_id))
        response = str(_row_value(response_texts, row_index, default=""))
        ground_truth = _row_value(answers, row_index)
        old_log_probs = list(_row_value(_lookup(batch_fields, "old_log_probs", "log_probs"), row_index, default=[0.0]))
        new_log_probs = list(_row_value(_lookup(batch_fields, "new_log_probs"), row_index, default=old_log_probs))
        ref_log_probs = list(_row_value(_lookup(batch_fields, "ref_log_probs", "ref_log_prob"), row_index, default=[]))
        response_mask = list(_row_value(_lookup(batch_fields, "response_mask"), row_index, default=[]))
        if not response_mask:
            inferred_length = len(new_log_probs or old_log_probs or _row_value(response_tokens, row_index, default=[]))
            response_mask = [1] * max(inferred_length, 1)
        trajectory = Trajectory(
            prompt_id=prompt_id,
            prompt=prompt,
            response=response,
            ground_truth=str(ground_truth) if ground_truth is not None else None,
            response_length=sum(response_mask),
            old_log_probs=old_log_probs,
            new_log_probs=new_log_probs,
            ref_log_probs=ref_log_probs,
            response_mask=response_mask,
            metrics={},
            upstream_batch_index=source_batch_index,
            upstream_row_index=row_index,
        )
        group = groups_by_prompt_id.setdefault(prompt_id, PromptGroup(prompt_id=prompt_id, trajectories=[]))
        group.trajectories.append(trajectory)

    return BatchContext(groups=list(groups_by_prompt_id.values()), metadata={"meta_info": meta_info})


def _slice_tensor_like(value: Any, indices: list[int]) -> Any:
    python_value = _to_python(value)
    if hasattr(value, "__getitem__") and not isinstance(python_value, (list, tuple)):
        try:
            return value[indices]
        except Exception:
            pass
    rows = _rows(python_value)
    return [rows[index] for index in indices]


def _rebuild_like(template: Any, batch_fields: dict[str, Any], non_tensor_fields: dict[str, Any], meta_info: dict[str, Any]) -> Any:
    batch_container = _lookup(template, "batch")
    if batch_container is not None:
        try:
            import torch  # type: ignore
        except Exception:  # pragma: no cover - optional runtime dependency
            torch = None  # type: ignore
        rebuilt_fields: dict[str, Any] = {}
        for key, value in batch_fields.items():
            reference = _lookup(batch_container, key) if batch_container is not None else None
            if torch is not None and reference is not None and isinstance(reference, torch.Tensor):
                rebuilt_fields[key] = torch.tensor(value, dtype=reference.dtype, device=reference.device)
            else:
                rebuilt_fields[key] = value
        batch_fields = rebuilt_fields

    cls = template.__class__
    if hasattr(cls, "from_dict"):
        for kwargs in (
            {"tensors": batch_fields, "non_tensors": non_tensor_fields, "meta_info": meta_info},
            {"batch": batch_fields, "non_tensor_batch": non_tensor_fields, "meta_info": meta_info},
        ):
            try:
                return cls.from_dict(**kwargs)
            except TypeError:
                continue
    if hasattr(template, "batch"):
        rebuilt = cls.__new__(cls)
        rebuilt.batch = batch_fields
        rebuilt.non_tensor_batch = non_tensor_fields
        rebuilt.meta_info = meta_info
        return rebuilt
    return {"batch": batch_fields, "non_tensor_batch": non_tensor_fields, "meta_info": meta_info}


def _select_rows(upstream_batch: Any, indices: list[int]) -> Any:
    if hasattr(upstream_batch, "__getitem__"):
        try:
            return upstream_batch[indices]
        except Exception:
            pass

    batch_fields = _batch_fields(upstream_batch)
    non_tensor_fields = _non_tensor_fields(upstream_batch)
    meta_info = _meta_info(upstream_batch)
    selected_batch_fields = {key: _slice_tensor_like(value, indices) for key, value in batch_fields.items()}
    selected_non_tensor = {key: [_row_value(value, index) for index in indices] for key, value in non_tensor_fields.items()}
    return _rebuild_like(upstream_batch, selected_batch_fields, selected_non_tensor, meta_info)


def _concat_batches(selected_batches: list[Any]) -> Any:
    if len(selected_batches) == 1:
        return selected_batches[0]
    first = selected_batches[0]
    concat = getattr(first.__class__, "concat", None)
    if callable(concat):
        try:
            return concat(selected_batches)
        except Exception:
            pass
    concat = getattr(first, "concat", None)
    if callable(concat):
        try:
            return concat(selected_batches)
        except Exception:
            pass

    merged_batch: dict[str, Any] = {}
    merged_non_tensor: dict[str, Any] = {}
    for key in _batch_fields(first):
        merged_rows: list[Any] = []
        for batch in selected_batches:
            merged_rows.extend(_rows(_lookup(_batch_fields(batch), key)))
        merged_batch[key] = merged_rows
    for key in _non_tensor_fields(first):
        merged_rows = []
        for batch in selected_batches:
            merged_rows.extend(_rows(_lookup(_non_tensor_fields(batch), key)))
        merged_non_tensor[key] = merged_rows
    return _rebuild_like(first, merged_batch, merged_non_tensor, _meta_info(first))


def _set_batch_field(upstream_batch: Any, key: str, value: Any) -> None:
    batch = _lookup(upstream_batch, "batch")
    if batch is not None:
        batch[key] = _coerce_batch_field_value(batch, key, value)
        return
    if isinstance(upstream_batch, dict):
        batch_dict = upstream_batch.setdefault("batch", {})
        batch_dict[key] = _coerce_batch_field_value(batch_dict, key, value)
        return
    raise TypeError(f"Cannot assign batch field {key!r} on {type(upstream_batch)!r}")


def _set_meta(upstream_batch: Any, key: str, value: Any) -> None:
    meta = _lookup(upstream_batch, "meta_info")
    if meta is None:
        if isinstance(upstream_batch, dict):
            upstream_batch["meta_info"] = {key: value}
            return
        raise TypeError(f"Cannot assign meta_info on {type(upstream_batch)!r}")
    meta[key] = value


def _coerce_batch_field_value(batch: Any, key: str, value: Any) -> Any:
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - torch is optional in local tests
        torch = None  # type: ignore

    if torch is None:
        return value

    reference_names_by_key = {
        "advantages": ("advantages", "returns", "old_log_probs", "new_log_probs", "token_level_rewards"),
        "returns": ("returns", "advantages", "old_log_probs", "new_log_probs", "token_level_rewards"),
        "token_level_rewards": ("token_level_rewards", "returns", "advantages", "old_log_probs", "new_log_probs"),
        "response_mask": ("response_mask", "attention_mask"),
    }

    for reference_name in reference_names_by_key.get(key, (key,)):
        reference = _lookup(batch, reference_name)
        if isinstance(reference, torch.Tensor):
            return torch.tensor(value, dtype=reference.dtype, device=reference.device)
    return value


def prepare_actor_update_batch(outcome: LoopOutcome, upstream_batches: list[Any]) -> PreparedActorBatch:
    selected_rows_by_batch: dict[int, list[tuple[int, Trajectory]]] = {}
    for trajectory in outcome.batch.iter_trajectories():
        if trajectory.filtered_out:
            continue
        if trajectory.upstream_batch_index is None or trajectory.upstream_row_index is None:
            raise RuntimeError("Selected trajectories must keep upstream batch and row references for actor update.")
        selected_rows_by_batch.setdefault(trajectory.upstream_batch_index, []).append((trajectory.upstream_row_index, trajectory))

    selected_batches: list[Any] = []
    trajectory_order: list[Trajectory] = []
    for batch_index in sorted(selected_rows_by_batch):
        sorted_pairs = sorted(selected_rows_by_batch[batch_index], key=lambda pair: pair[0])
        row_indices = [row_index for row_index, _trajectory in sorted_pairs]
        selected_batches.append(_select_rows(upstream_batches[batch_index], row_indices))
        trajectory_order.extend([trajectory for _row_index, trajectory in sorted_pairs])

    if not selected_batches:
        raise RuntimeError("No trajectories survived filtering, so no actor-update batch can be constructed.")

    prepared = _concat_batches(selected_batches)
    advantages = [trajectory.token_advantages for trajectory in trajectory_order]
    returns = [trajectory.returns for trajectory in trajectory_order]
    response_masks = [trajectory.ensure_mask() for trajectory in trajectory_order]
    rewards = [trajectory.effective_total_reward() for trajectory in trajectory_order]

    _set_batch_field(prepared, "advantages", advantages)
    _set_batch_field(prepared, "returns", returns)
    _set_batch_field(prepared, "response_mask", response_masks)
    _set_batch_field(prepared, "token_level_rewards", returns)
    _set_meta(prepared, "dapo_lab_prompt_count", outcome.batch.prompt_count())
    _set_meta(prepared, "dapo_lab_trajectory_count", outcome.batch.trajectory_count())
    _set_meta(prepared, "dapo_lab_rewards", rewards)
    return PreparedActorBatch(
        batch=prepared,
        trajectory_order=trajectory_order,
        prompt_count=outcome.batch.prompt_count(),
        trajectory_count=outcome.batch.trajectory_count(),
    )


def compute_behavior_delta(pre_update_log_probs: list[list[float]], post_update_log_probs: list[list[float]]) -> float:
    squared_error = 0.0
    count = 0
    for before_row, after_row in zip(pre_update_log_probs, post_update_log_probs, strict=True):
        for before, after in zip(before_row, after_row, strict=True):
            squared_error += (float(after) - float(before)) ** 2
            count += 1
    return math.sqrt(squared_error) if count else 0.0
