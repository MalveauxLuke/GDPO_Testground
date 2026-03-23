from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _copy_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    return dict(mapping or {})


@dataclass(slots=True)
class RewardTermConfig:
    name: str
    kind: str
    weight: float = 1.0
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardTermConfig":
        return cls(
            name=str(payload["name"]),
            kind=str(payload["kind"]),
            weight=float(payload.get("weight", 1.0)),
            enabled=bool(payload.get("enabled", True)),
            params=_copy_mapping(payload.get("params")),
        )


@dataclass(slots=True)
class OverlongConfig:
    enabled: bool = False
    mode: str = "disabled"
    buffer_length: int = 0
    penalty_factor: float = 0.0
    hard_filter: bool = False
    log: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OverlongConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            mode=str(payload.get("mode", "disabled")),
            buffer_length=int(payload.get("buffer_length", 0)),
            penalty_factor=float(payload.get("penalty_factor", 0.0)),
            hard_filter=bool(payload.get("hard_filter", False)),
            log=bool(payload.get("log", False)),
        )


@dataclass(slots=True)
class RewardConfig:
    terms: list[RewardTermConfig] = field(default_factory=list)
    overlong: OverlongConfig = field(default_factory=OverlongConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewardConfig":
        return cls(
            terms=[RewardTermConfig.from_dict(item) for item in payload.get("terms", [])],
            overlong=OverlongConfig.from_dict(payload.get("overlong")),
        )


@dataclass(slots=True)
class DataConfig:
    train_files: list[str]
    val_files: list[str]
    format: str
    prompt_key: str
    answer_key: str
    train_batch_size: int
    gen_batch_size: int
    max_prompt_length: int
    max_response_length: int
    rollout_n: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataConfig":
        return cls(
            train_files=[str(item) for item in payload.get("train_files", [])],
            val_files=[str(item) for item in payload.get("val_files", [])],
            format=str(payload.get("format", "jsonl")),
            prompt_key=str(payload.get("prompt_key", "prompt")),
            answer_key=str(payload.get("answer_key", "ground_truth")),
            train_batch_size=int(payload["train_batch_size"]),
            gen_batch_size=int(payload["gen_batch_size"]),
            max_prompt_length=int(payload["max_prompt_length"]),
            max_response_length=int(payload["max_response_length"]),
            rollout_n=int(payload.get("rollout_n", 1)),
        )


@dataclass(slots=True)
class AdvantageConfig:
    mode: str = "grpo"
    normalize_by_std: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AdvantageConfig":
        payload = payload or {}
        return cls(
            mode=str(payload.get("mode", "grpo")),
            normalize_by_std=bool(payload.get("normalize_by_std", True)),
        )


@dataclass(slots=True)
class GDPOConfig:
    component_keys: list[str] | None = None
    component_weights: list[float] | None = None
    normalize_by_std: bool = True
    batch_whiten: bool = True
    allow_single_component_ablation: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GDPOConfig":
        payload = payload or {}
        component_keys = payload.get("component_keys")
        component_weights = payload.get("component_weights")
        return cls(
            component_keys=[str(item) for item in component_keys] if component_keys is not None else None,
            component_weights=[float(item) for item in component_weights] if component_weights is not None else None,
            normalize_by_std=bool(payload.get("normalize_by_std", True)),
            batch_whiten=bool(payload.get("batch_whiten", True)),
            allow_single_component_ablation=bool(payload.get("allow_single_component_ablation", False)),
        )


@dataclass(slots=True)
class PolicyLossConfig:
    mode: str = "clipped"
    clip_ratio: float = 0.2
    clip_ratio_low: float | None = None
    clip_ratio_high: float | None = None
    clip_ratio_c: float | None = None
    loss_agg_mode: str = "token-mean"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "PolicyLossConfig":
        payload = payload or {}
        clip_ratio_low = payload.get("clip_ratio_low")
        clip_ratio_high = payload.get("clip_ratio_high")
        clip_ratio_c = payload.get("clip_ratio_c")
        return cls(
            mode=str(payload.get("mode", "clipped")),
            clip_ratio=float(payload.get("clip_ratio", 0.2)),
            clip_ratio_low=float(clip_ratio_low) if clip_ratio_low is not None else None,
            clip_ratio_high=float(clip_ratio_high) if clip_ratio_high is not None else None,
            clip_ratio_c=float(clip_ratio_c) if clip_ratio_c is not None else None,
            loss_agg_mode=str(payload.get("loss_agg_mode", "token-mean")),
        )


@dataclass(slots=True)
class GroupFilteringConfig:
    enabled: bool = False
    metric: str = "acc"
    max_num_gen_batches: int = 0
    require_variance: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GroupFilteringConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            metric=str(payload.get("metric", "acc")),
            max_num_gen_batches=int(payload.get("max_num_gen_batches", 0)),
            require_variance=bool(payload.get("require_variance", True)),
        )


@dataclass(slots=True)
class KLConfig:
    enabled: bool = False
    source: str = "reward"
    penalty: str = "low_var_kl"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "KLConfig":
        payload = payload or {}
        return cls(
            enabled=bool(payload.get("enabled", False)),
            source=str(payload.get("source", "reward")),
            penalty=str(payload.get("penalty", "low_var_kl")),
        )


@dataclass(slots=True)
class TrainerOrderConfig:
    stages: list[str] = field(default_factory=lambda: [
        "rollout",
        "reward",
        "kl",
        "filtering",
        "advantage",
        "actor_update",
        "diagnostics",
    ])

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainerOrderConfig":
        payload = payload or {}
        return cls(stages=[str(item) for item in payload.get("stages", [])] or cls().stages)


@dataclass(slots=True)
class RolloutBehaviorConfig:
    backend: str = "vllm"
    do_sample: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    accumulate_filtered_groups: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "RolloutBehaviorConfig":
        payload = payload or {}
        return cls(
            backend=str(payload.get("backend", "vllm")),
            do_sample=bool(payload.get("do_sample", True)),
            temperature=float(payload.get("temperature", 1.0)),
            top_p=float(payload.get("top_p", 1.0)),
            top_k=int(payload.get("top_k", -1)),
            accumulate_filtered_groups=bool(payload.get("accumulate_filtered_groups", False)),
        )


@dataclass(slots=True)
class AlgorithmConfig:
    variant: str
    advantage: AdvantageConfig = field(default_factory=AdvantageConfig)
    gdpo: GDPOConfig = field(default_factory=GDPOConfig)
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    group_filtering: GroupFilteringConfig = field(default_factory=GroupFilteringConfig)
    kl: KLConfig = field(default_factory=KLConfig)
    trainer_order: TrainerOrderConfig = field(default_factory=TrainerOrderConfig)
    rollout_behavior: RolloutBehaviorConfig = field(default_factory=RolloutBehaviorConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AlgorithmConfig":
        return cls(
            variant=str(payload["variant"]),
            advantage=AdvantageConfig.from_dict(payload.get("advantage")),
            gdpo=GDPOConfig.from_dict(payload.get("gdpo")),
            policy_loss=PolicyLossConfig.from_dict(payload.get("policy_loss")),
            group_filtering=GroupFilteringConfig.from_dict(payload.get("group_filtering")),
            kl=KLConfig.from_dict(payload.get("kl")),
            trainer_order=TrainerOrderConfig.from_dict(payload.get("trainer_order")),
            rollout_behavior=RolloutBehaviorConfig.from_dict(payload.get("rollout_behavior")),
        )


@dataclass(slots=True)
class DiagnosticsConfig:
    record_stage_events: bool = True
    emit_group_stats: bool = True
    emit_reward_breakdown: bool = True
    report_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "DiagnosticsConfig":
        payload = payload or {}
        return cls(
            record_stage_events=bool(payload.get("record_stage_events", True)),
            emit_group_stats=bool(payload.get("emit_group_stats", True)),
            emit_reward_breakdown=bool(payload.get("emit_reward_breakdown", True)),
            report_path=str(payload["report_path"]) if payload.get("report_path") else None,
        )


@dataclass(slots=True)
class TrainerConfig:
    log_level: str = "info"
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    save_freq: int = 100
    test_freq: int = 100
    val_before_train: bool = True
    max_steps: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "TrainerConfig":
        payload = payload or {}
        return cls(
            log_level=str(payload.get("log_level", "info")),
            diagnostics=DiagnosticsConfig.from_dict(payload.get("diagnostics")),
            save_freq=int(payload.get("save_freq", 100)),
            test_freq=int(payload.get("test_freq", 100)),
            val_before_train=bool(payload.get("val_before_train", True)),
            max_steps=int(payload.get("max_steps", 0)),
        )


@dataclass(slots=True)
class VerlActorConfig:
    ppo_micro_batch_size_per_gpu: int = 1
    grad_clip: float = 1.0
    ppo_epochs: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "VerlActorConfig":
        payload = payload or {}
        return cls(
            ppo_micro_batch_size_per_gpu=int(payload.get("ppo_micro_batch_size_per_gpu", 1)),
            grad_clip=float(payload.get("grad_clip", 1.0)),
            ppo_epochs=int(payload.get("ppo_epochs", 1)),
        )


@dataclass(slots=True)
class VerlRolloutConfig:
    tensor_model_parallel_size: int = 1
    gpu_memory_utilization: float = 0.5
    enforce_eager: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "VerlRolloutConfig":
        payload = payload or {}
        return cls(
            tensor_model_parallel_size=int(payload.get("tensor_model_parallel_size", 1)),
            gpu_memory_utilization=float(payload.get("gpu_memory_utilization", 0.5)),
            enforce_eager=bool(payload.get("enforce_eager", True)),
        )


@dataclass(slots=True)
class VerlConfig:
    required_commit: str
    runtime_stack: str
    strict_compatibility: bool = False
    model_path: str = ""
    trust_remote_code: bool = False
    actor: VerlActorConfig = field(default_factory=VerlActorConfig)
    rollout: VerlRolloutConfig = field(default_factory=VerlRolloutConfig)
    reference_policy: bool = True
    critic: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VerlConfig":
        return cls(
            required_commit=str(payload["required_commit"]),
            runtime_stack=str(payload.get("runtime_stack", "fsdp_vllm")),
            strict_compatibility=bool(payload.get("strict_compatibility", False)),
            model_path=str(payload.get("model_path", "")),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            actor=VerlActorConfig.from_dict(payload.get("actor")),
            rollout=VerlRolloutConfig.from_dict(payload.get("rollout")),
            reference_policy=bool(payload.get("reference_policy", True)),
            critic=bool(payload.get("critic", False)),
        )


@dataclass(slots=True)
class ExperimentMeta:
    name: str
    seed: int
    output_dir: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentMeta":
        return cls(
            name=str(payload["name"]),
            seed=int(payload.get("seed", 42)),
            output_dir=str(payload.get("output_dir", "outputs/default")),
        )


@dataclass(slots=True)
class ExperimentConfig:
    experiment: ExperimentMeta
    data: DataConfig
    reward: RewardConfig
    algorithm: AlgorithmConfig
    trainer: TrainerConfig
    verl: VerlConfig
    source_path: Path | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], source_path: Path | None = None) -> "ExperimentConfig":
        return cls(
            experiment=ExperimentMeta.from_dict(payload["experiment"]),
            data=DataConfig.from_dict(payload["data"]),
            reward=RewardConfig.from_dict(payload["reward"]),
            algorithm=AlgorithmConfig.from_dict(payload["algorithm"]),
            trainer=TrainerConfig.from_dict(payload.get("trainer")),
            verl=VerlConfig.from_dict(payload["verl"]),
            source_path=source_path,
        )
