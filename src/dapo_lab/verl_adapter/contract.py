from __future__ import annotations

import copy
import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


PINNED_VERL_COMMIT = "7dc46e834209948cf1cdd8a04d83f82b4a7efd24"

_CONTRACT_DIR = Path(__file__).resolve().parent / "contracts"
_PINNED_CONTRACT_PATH = _CONTRACT_DIR / "verl_7dc46e8_contract.yaml"
_PINNED_SCAFFOLD_PATH = _CONTRACT_DIR / "verl_7dc46e8_scaffold.yaml"


@dataclass(slots=True)
class ContractAudit:
    pinned_commit: str
    missing_top_level: list[str] = field(default_factory=list)
    unexpected_top_level: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    missing_target_paths: list[str] = field(default_factory=list)
    semantic_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            [
                self.missing_top_level,
                self.unexpected_top_level,
                self.missing_paths,
                self.missing_target_paths,
                self.semantic_errors,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinned_commit": self.pinned_commit,
            "ok": self.ok,
            "missing_top_level": self.missing_top_level,
            "unexpected_top_level": self.unexpected_top_level,
            "missing_paths": self.missing_paths,
            "missing_target_paths": self.missing_target_paths,
            "semantic_errors": self.semantic_errors,
        }


@dataclass(slots=True)
class LiveCheckoutDrift:
    pinned_commit: str
    checkout_path: str | None
    detected_commit: str | None
    warnings: list[str] = field(default_factory=list)
    missing_top_level: list[str] = field(default_factory=list)
    extra_top_level: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    missing_target_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pinned_commit": self.pinned_commit,
            "checkout_path": self.checkout_path,
            "detected_commit": self.detected_commit,
            "warnings": self.warnings,
            "missing_top_level": self.missing_top_level,
            "extra_top_level": self.extra_top_level,
            "missing_paths": self.missing_paths,
            "missing_target_paths": self.missing_target_paths,
        }


@lru_cache(maxsize=1)
def load_pinned_contract() -> dict[str, Any]:
    return yaml.safe_load(_PINNED_CONTRACT_PATH.read_text()) or {}


@lru_cache(maxsize=1)
def _load_pinned_scaffold_template() -> dict[str, Any]:
    return yaml.safe_load(_PINNED_SCAFFOLD_PATH.read_text()) or {}


def load_pinned_scaffold() -> dict[str, Any]:
    return copy.deepcopy(_load_pinned_scaffold_template())


def _has_path(payload: dict[str, Any], dotted_path: str) -> bool:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return False
        current = current[segment]
    return True


def _get_path(payload: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = payload
    for segment in dotted_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return current


def _target_missing(payload: dict[str, Any], dotted_path: str) -> bool:
    node = _get_path(payload, dotted_path)
    return not isinstance(node, dict) or "_target_" not in node or not node["_target_"]


def _validate_semantics(payload: dict[str, Any], *, pinned_commit: str) -> list[str]:
    errors: list[str] = []

    required_commit = _get_path(payload, "dapo_lab.verl.required_commit")
    if required_commit is not None and required_commit != pinned_commit:
        errors.append(
            f"dapo_lab.verl.required_commit={required_commit!r} does not match pinned contract {pinned_commit!r}."
        )

    train_batch_size = _get_path(payload, "data.train_batch_size")
    actor_mini_batch_size = _get_path(payload, "actor_rollout_ref.actor.ppo_mini_batch_size")
    use_dynamic_bsz = bool(_get_path(payload, "actor_rollout_ref.actor.use_dynamic_bsz", False))
    if (
        not use_dynamic_bsz
        and isinstance(train_batch_size, int)
        and isinstance(actor_mini_batch_size, int)
        and train_batch_size < actor_mini_batch_size
    ):
        errors.append(
            "data.train_batch_size must be >= actor_rollout_ref.actor.ppo_mini_batch_size when use_dynamic_bsz is false."
        )

    def _check_log_prob_fields(prefix: str, *, required: bool) -> None:
        micro = _get_path(payload, f"{prefix}.log_prob_micro_batch_size")
        per_gpu = _get_path(payload, f"{prefix}.log_prob_micro_batch_size_per_gpu")
        if micro is not None and per_gpu is not None:
            errors.append(
                f"{prefix} sets both log_prob_micro_batch_size and log_prob_micro_batch_size_per_gpu; only one is allowed."
            )
        if required and micro is None and per_gpu is None:
            errors.append(
                f"{prefix} must set one of log_prob_micro_batch_size or log_prob_micro_batch_size_per_gpu."
            )

    _check_log_prob_fields("actor_rollout_ref.rollout", required=True)
    reference_needed = bool(_get_path(payload, "algorithm.use_kl_in_reward", False)) or bool(
        _get_path(payload, "actor_rollout_ref.actor.use_kl_loss", False)
    )
    _check_log_prob_fields("actor_rollout_ref.ref", required=reference_needed)

    return errors


def audit_bridge_config(payload: dict[str, Any]) -> ContractAudit:
    contract = load_pinned_contract()
    required_top_level = set(contract.get("required_top_level", []))
    allowed_extra_top_level = set(contract.get("allow_extra_top_level", []))
    required_paths = contract.get("required_paths", [])
    required_target_paths = contract.get("required_target_paths", [])

    top_level = set(payload.keys())
    unexpected_top_level = sorted(
        key for key in (top_level - required_top_level) if key not in allowed_extra_top_level
    )
    return ContractAudit(
        pinned_commit=str(contract.get("pinned_commit", PINNED_VERL_COMMIT)),
        missing_top_level=sorted(required_top_level - top_level),
        unexpected_top_level=unexpected_top_level,
        missing_paths=sorted(path for path in required_paths if not _has_path(payload, path)),
        missing_target_paths=sorted(path for path in required_target_paths if _target_missing(payload, path)),
        semantic_errors=_validate_semantics(payload, pinned_commit=str(contract.get("pinned_commit", PINNED_VERL_COMMIT))),
    )


def _git_rev_parse_head(repo_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def resolve_live_verl_checkout(override: str | os.PathLike[str] | None = None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    env_path = os.environ.get("VERL_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return None


def audit_live_checkout(override: str | os.PathLike[str] | None = None) -> LiveCheckoutDrift | None:
    checkout = resolve_live_verl_checkout(override)
    if checkout is None:
        return None

    contract = load_pinned_contract()
    generated_path = checkout / "verl" / "trainer" / "config" / "_generated_ppo_trainer.yaml"
    detected_commit = _git_rev_parse_head(checkout)
    drift = LiveCheckoutDrift(
        pinned_commit=str(contract.get("pinned_commit", PINNED_VERL_COMMIT)),
        checkout_path=str(checkout),
        detected_commit=detected_commit,
    )

    if detected_commit and detected_commit != drift.pinned_commit:
        drift.warnings.append(
            f"Live verl checkout commit {detected_commit} differs from pinned contract {drift.pinned_commit}."
        )

    if not generated_path.exists():
        drift.warnings.append(f"Live verl checkout is missing {generated_path}.")
        return drift

    live_payload = yaml.safe_load(generated_path.read_text()) or {}
    required_top_level = set(contract.get("required_top_level", []))
    drift.missing_top_level = sorted(required_top_level - set(live_payload.keys()))
    drift.extra_top_level = sorted(set(live_payload.keys()) - required_top_level)
    drift.missing_paths = sorted(
        path for path in contract.get("required_paths", []) if not _has_path(live_payload, path)
    )
    drift.missing_target_paths = sorted(
        path for path in contract.get("required_target_paths", []) if _target_missing(live_payload, path)
    )
    if drift.missing_top_level:
        drift.warnings.append("Live verl generated config is missing pinned top-level sections.")
    if drift.extra_top_level:
        drift.warnings.append("Live verl generated config has top-level sections not present in the pinned contract.")
    if drift.missing_paths:
        drift.warnings.append("Live verl generated config is missing pinned required paths.")
    if drift.missing_target_paths:
        drift.warnings.append("Live verl generated config is missing pinned _target_ blocks.")
    return drift
