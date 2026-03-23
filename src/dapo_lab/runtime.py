from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config_schema import ExperimentConfig
from .validation import load_experiment_config
from .verl_adapter.compat import CompatibilityReport, check_verl_compatibility
from .verl_adapter.task_runner import launch_with_verl


@dataclass(slots=True)
class ResearchRuntime:
    config: ExperimentConfig
    compatibility: CompatibilityReport

    @classmethod
    def from_path(cls, path: str | Path) -> "ResearchRuntime":
        config = load_experiment_config(path)
        compatibility = check_verl_compatibility(
            required_commit=config.verl.required_commit,
            strict=config.verl.strict_compatibility,
        )
        return cls(config=config, compatibility=compatibility)

    def launch(self) -> None:
        launch_with_verl(self.config, self.compatibility)
