from __future__ import annotations

from typing import Any

from .schema import PromptExample


def normalize_math_example(raw: dict[str, Any], prompt_key: str, answer_key: str, prompt_id_key: str = "prompt_id") -> PromptExample:
    prompt = str(raw[prompt_key])
    ground_truth = str(raw[answer_key])
    prompt_id = str(raw.get(prompt_id_key) or raw.get("id") or prompt[:64])
    metadata = {key: value for key, value in raw.items() if key not in {prompt_key, answer_key, prompt_id_key}}
    return PromptExample(prompt_id=prompt_id, prompt=prompt, ground_truth=ground_truth, metadata=metadata)
