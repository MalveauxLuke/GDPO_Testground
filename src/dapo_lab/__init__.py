"""dapo_lab: a research harness for GRPO-to-DAPO work on top of verl."""

from .config_schema import ExperimentConfig
from .validation import ConfigValidationError, load_experiment_config

__all__ = [
    "ConfigValidationError",
    "ExperimentConfig",
    "load_experiment_config",
]
