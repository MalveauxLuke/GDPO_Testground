from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StageEvent:
    name: str
    detail: str = ""


@dataclass(slots=True)
class DiagnosticsRecorder:
    stage_events: list[StageEvent] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)

    def record_stage(self, name: str, detail: str = "") -> None:
        self.stage_events.append(StageEvent(name=name, detail=detail))

    def record_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update(metrics)

    def annotate(self, **values: Any) -> None:
        self.annotations.update(values)

    def ordered_stage_names(self) -> list[str]:
        return [event.name for event in self.stage_events]
