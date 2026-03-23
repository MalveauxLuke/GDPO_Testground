from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .math import normalize_math_example
from .schema import PromptExample


class DataFormatError(ValueError):
    """Raised when dataset loading is requested for an unsupported format."""


def load_jsonl_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def load_examples(
    paths: list[str],
    *,
    dataset_format: str,
    prompt_key: str,
    answer_key: str,
    normalizer: Callable[[dict, str, str], PromptExample] = normalize_math_example,
) -> list[PromptExample]:
    if dataset_format != "jsonl":
        raise DataFormatError(
            "This repo ships a stdlib-first JSONL loader. Use JSONL locally or add a parquet loader in data/prep.py."
        )

    examples: list[PromptExample] = []
    for path in paths:
        for record in load_jsonl_records(path):
            examples.append(normalizer(record, prompt_key, answer_key))
    return examples
