from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PromptExample:
    prompt_id: str
    prompt: str
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)
