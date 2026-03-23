from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from .config_schema import ExperimentConfig


class ConfigValidationError(ValueError):
    """Raised when the experiment config is inconsistent for research use."""


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigValidationError(message)


def _ordered_stages(stages: Iterable[str]) -> list[str]:
    return [stage.strip() for stage in stages if stage.strip()]


def _resolve_path(base_dir: Path, candidate: str) -> str:
    path = Path(candidate)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def validate_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    _ensure(config.algorithm.variant in {"grpo", "dapo", "gdpo"}, "algorithm.variant must be 'grpo', 'dapo', or 'gdpo'.")
    _ensure(config.data.train_batch_size > 0, "data.train_batch_size must be positive.")
    _ensure(config.data.gen_batch_size >= config.data.train_batch_size, "data.gen_batch_size must be >= train_batch_size.")
    _ensure(config.data.rollout_n > 0, "data.rollout_n must be positive.")
    _ensure(config.data.max_prompt_length > 0, "data.max_prompt_length must be positive.")
    _ensure(config.data.max_response_length > 0, "data.max_response_length must be positive.")
    _ensure(config.trainer.max_steps >= 0, "trainer.max_steps must be >= 0.")
    _ensure(bool(config.reward.terms), "reward.terms must include at least one enabled reward term.")
    enabled_reward_terms = [term for term in config.reward.terms if term.enabled]
    enabled_reward_names = [term.name for term in enabled_reward_terms]
    enabled_reward_weights = {term.name: term.weight for term in enabled_reward_terms}

    policy = config.algorithm.policy_loss
    _ensure(policy.mode == "clipped", "Only policy_loss.mode='clipped' is implemented in v1.")
    _ensure(policy.clip_ratio > 0, "algorithm.policy_loss.clip_ratio must be positive.")
    _ensure(
        policy.loss_agg_mode in {"token-mean", "seq-mean-token-sum", "seq-mean-token-mean"},
        "algorithm.policy_loss.loss_agg_mode must be one of token-mean, seq-mean-token-sum, seq-mean-token-mean.",
    )

    if policy.clip_ratio_low is not None:
        _ensure(policy.clip_ratio_low >= 0, "algorithm.policy_loss.clip_ratio_low must be non-negative.")
    if policy.clip_ratio_high is not None:
        _ensure(policy.clip_ratio_high >= 0, "algorithm.policy_loss.clip_ratio_high must be non-negative.")
    if policy.clip_ratio_c is not None:
        _ensure(policy.clip_ratio_c > 1.0, "algorithm.policy_loss.clip_ratio_c must be > 1.0 when provided.")

    filtering = config.algorithm.group_filtering
    _ensure(filtering.max_num_gen_batches >= 0, "algorithm.group_filtering.max_num_gen_batches must be >= 0.")

    gdpo = config.algorithm.gdpo
    if gdpo.component_keys is None:
        gdpo.component_keys = list(enabled_reward_names)
    if gdpo.component_weights is None:
        gdpo.component_weights = [enabled_reward_weights[key] for key in gdpo.component_keys]
    _ensure(
        len(gdpo.component_keys) == len(set(gdpo.component_keys)),
        "algorithm.gdpo.component_keys must be unique when provided.",
    )
    _ensure(
        all(key in enabled_reward_names for key in gdpo.component_keys),
        "algorithm.gdpo.component_keys must all exist in enabled reward term names.",
    )
    _ensure(
        len(gdpo.component_weights) == len(gdpo.component_keys),
        "algorithm.gdpo.component_weights must match algorithm.gdpo.component_keys length.",
    )

    overlong = config.reward.overlong
    if overlong.enabled:
        _ensure(
            overlong.mode in {"shaping", "filter", "shape_and_filter"},
            "reward.overlong.mode must be shaping, filter, or shape_and_filter when enabled.",
        )
        _ensure(overlong.buffer_length > 0, "reward.overlong.buffer_length must be positive when enabled.")
        _ensure(
            overlong.buffer_length <= config.data.max_response_length,
            "reward.overlong.buffer_length cannot exceed data.max_response_length.",
        )
        _ensure(overlong.penalty_factor >= 0, "reward.overlong.penalty_factor must be non-negative.")

    if config.algorithm.variant == "grpo":
        _ensure(
            not filtering.enabled,
            "algorithm.group_filtering.enabled must be false for the naive GRPO baseline in this repo.",
        )
        _ensure(
            not overlong.enabled,
            "reward.overlong.enabled must be false for the naive GRPO baseline in this repo.",
        )
        if policy.clip_ratio_low is not None:
            _ensure(
                policy.clip_ratio_low == policy.clip_ratio,
                "GRPO uses symmetric clipping; clip_ratio_low must match clip_ratio when set.",
            )
        if policy.clip_ratio_high is not None:
            _ensure(
                policy.clip_ratio_high == policy.clip_ratio,
                "GRPO uses symmetric clipping; clip_ratio_high must match clip_ratio when set.",
            )
    elif config.algorithm.variant == "dapo":
        _ensure(
            filtering.enabled or overlong.enabled or policy.clip_ratio_low is not None or policy.clip_ratio_high is not None,
            "DAPO should expose at least one DAPO-specific surface: asymmetric clipping, group filtering, or overlong shaping.",
        )
    elif config.algorithm.variant == "gdpo":
        _ensure(
            len(gdpo.component_keys) >= 2 or gdpo.allow_single_component_ablation,
            "GDPO requires at least two reward components unless allow_single_component_ablation=true.",
        )
        _ensure(
            not filtering.enabled,
            "Strict GDPO v1 does not enable algorithm.group_filtering.enabled.",
        )
        _ensure(
            not config.algorithm.rollout_behavior.accumulate_filtered_groups,
            "Strict GDPO v1 does not enable rollout_behavior.accumulate_filtered_groups.",
        )
        _ensure(
            not overlong.enabled,
            "Strict GDPO v1 does not enable reward.overlong.enabled.",
        )
        if policy.clip_ratio_low is not None:
            _ensure(
                policy.clip_ratio_low == policy.clip_ratio,
                "Strict GDPO v1 requires symmetric clipping; clip_ratio_low must match clip_ratio.",
            )
        if policy.clip_ratio_high is not None:
            _ensure(
                policy.clip_ratio_high == policy.clip_ratio,
                "Strict GDPO v1 requires symmetric clipping; clip_ratio_high must match clip_ratio.",
            )

    stages = _ordered_stages(config.algorithm.trainer_order.stages)
    expected = ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
    _ensure(stages == expected, f"algorithm.trainer_order.stages must be exactly {expected}.")

    _ensure(len(enabled_reward_names) == len(set(enabled_reward_names)), "reward term names must be unique.")
    if config.source_path is not None:
        for path in [*config.data.train_files, *config.data.val_files]:
            _ensure(Path(path).exists(), f"Configured dataset path does not exist: {path}")
    return config


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    source_path = Path(path)
    payload = yaml.safe_load(source_path.read_text()) or {}
    config = ExperimentConfig.from_dict(payload, source_path=source_path)
    base_dir = source_path.resolve().parent
    config.data.train_files = [_resolve_path(base_dir, candidate) for candidate in config.data.train_files]
    config.data.val_files = [_resolve_path(base_dir, candidate) for candidate in config.data.val_files]
    config.experiment.output_dir = _resolve_path(base_dir, config.experiment.output_dir)
    report_path = config.trainer.diagnostics.report_path
    if report_path:
        config.trainer.diagnostics.report_path = _resolve_path(base_dir, report_path)
    return validate_experiment_config(config)
