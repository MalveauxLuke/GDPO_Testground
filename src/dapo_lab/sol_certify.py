from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import inspect
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from dapo_lab.validation import load_experiment_config
from dapo_lab.verl_adapter.compat import check_verl_compatibility
from dapo_lab.verl_adapter.config_bridge import build_verl_config
from dapo_lab.verl_adapter.contract import PINNED_VERL_COMMIT, audit_bridge_config, audit_live_checkout


SUITES = {"env", "preflight", "debug", "hf", "vllm", "all"}
TINY_SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class CertificationFailure(RuntimeError):
    """Raised when a certification suite fails."""


@dataclass(slots=True)
class SuiteResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    children: list["SuiteResult"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "children": [child.to_dict() for child in self.children],
        }


RuntimeRunner = Callable[[Path, Path], dict[str, Any]]


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _log(message: str) -> None:
    print(f"[sol_certify] {message}", flush=True)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CertificationFailure(message)


def _load_payload(config_path: Path) -> dict[str, Any]:
    return yaml.safe_load(config_path.read_text()) or {}


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _suite_root(config_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir).resolve()
    config = load_experiment_config(config_path)
    return Path(config.experiment.output_dir) / "certify" / _timestamp()


def _audit_lines(audit: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if audit.get("missing_top_level"):
        lines.append(f"missing top-level: {', '.join(audit['missing_top_level'])}")
    if audit.get("unexpected_top_level"):
        lines.append(f"unexpected top-level: {', '.join(audit['unexpected_top_level'])}")
    if audit.get("missing_paths"):
        lines.append(f"missing paths: {', '.join(audit['missing_paths'])}")
    if audit.get("missing_target_paths"):
        lines.append(f"missing _target_ paths: {', '.join(audit['missing_target_paths'])}")
    if audit.get("semantic_errors"):
        lines.extend(audit["semantic_errors"])
    return lines


def _check_import(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise CertificationFailure(f"Required module is not importable: {module_name}")
    module = importlib.import_module(module_name)
    return {"module": module_name, "location": getattr(module, "__file__", "built-in")}


def _check_ray() -> dict[str, Any]:
    ray = importlib.import_module("ray")
    init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
    signature = inspect.signature(ray.init)
    if "include_dashboard" in signature.parameters:
        init_kwargs["include_dashboard"] = False
    if "_temp_dir" in signature.parameters:
        init_kwargs["_temp_dir"] = os.environ.get("RAY_TMPDIR")
    if "num_cpus" in signature.parameters:
        init_kwargs["num_cpus"] = 1

    context = ray.init(**init_kwargs)
    try:
        resources = ray.cluster_resources()
    finally:
        ray.shutdown()
    return {"cluster_resources": resources}


def _check_nvidia_smi() -> dict[str, Any]:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise CertificationFailure("nvidia-smi not found. Run GPU suites on a GPU allocation.") from error
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    _require(bool(names), "nvidia-smi returned no visible GPUs.")
    return {"gpu_names": names}


def _check_torch_cuda() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    _require(torch.cuda.is_available(), "torch.cuda.is_available() is false on the allocated node.")
    return {"cuda_device_count": int(torch.cuda.device_count()), "device_name": str(torch.cuda.get_device_name(0))}


def _check_transformers_model(model_name: str, cache_dir: str | None) -> dict[str, Any]:
    transformers = importlib.import_module("transformers")
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}
    _log(f"probing transformers tokenizer for {model_name} (cache_dir={cache_dir or 'default'})")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
    _log(f"tokenizer loaded: {tokenizer.__class__.__name__}")
    config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
    _log(f"model config loaded: {getattr(config, 'model_type', 'unknown')}")
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_type": getattr(config, "model_type", "unknown"),
    }


def _run_verl_runtime_preflight(config_path: Path, bridge_payload: dict[str, Any]) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    compatibility = check_verl_compatibility(
        required_commit=config.verl.required_commit,
        strict=config.verl.strict_compatibility,
    )
    _require(compatibility.importable, compatibility.message)

    try:
        from omegaconf import OmegaConf  # type: ignore
        from verl.experimental.reward_loop import migrate_legacy_reward_impl  # type: ignore
        from verl.trainer.ppo.utils import need_critic, need_reference_policy  # type: ignore
        from verl.utils.config import validate_config  # type: ignore
        from verl.utils.device import auto_set_device  # type: ignore
    except Exception as error:
        raise CertificationFailure(
            "Pinned contract passed, but the local verl runtime stack is incomplete. "
            "Install verl, ray, hydra, and omegaconf in the active environment."
        ) from error

    _log("building OmegaConf config from pinned scaffold bridge")
    upstream_config = OmegaConf.create(bridge_payload)
    auto_set_device(upstream_config)
    _log("running legacy reward migration")
    upstream_config = migrate_legacy_reward_impl(upstream_config)
    reference_policy_required = bool(need_reference_policy(upstream_config))
    critic_required = bool(need_critic(upstream_config))
    _log("running upstream verl validate_config")
    validate_config(
        config=upstream_config,
        use_reference_policy=reference_policy_required,
        use_critic=critic_required,
    )
    return {
        "compatibility": {
            "importable": compatibility.importable,
            "detected_commit": compatibility.detected_commit,
            "required_commit": compatibility.required_commit,
            "compatible": compatibility.compatible,
            "message": compatibility.message,
        },
        "reference_policy_required": reference_policy_required,
        "critic_required": critic_required,
        "device": upstream_config.trainer.device,
    }


def _ensure_sol_cache_env() -> dict[str, str]:
    defaults = {
        "HF_HOME": f"/scratch/{os.environ['USER']}/dapo_lab/hf",
        "VLLM_CACHE_ROOT": f"/scratch/{os.environ['USER']}/dapo_lab/vllm",
        "RAY_TMPDIR": f"/scratch/{os.environ['USER']}/dapo_lab/ray",
        "TORCH_EXTENSIONS_DIR": f"/scratch/{os.environ['USER']}/dapo_lab/torch",
    }
    os.environ.setdefault("HF_HOME", defaults["HF_HOME"])
    os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HOME"])
    for key, default in defaults.items():
        os.environ.setdefault(key, default)

    realized = {
        "HF_HOME": os.environ["HF_HOME"],
        "TRANSFORMERS_CACHE": os.environ["TRANSFORMERS_CACHE"],
        "VLLM_CACHE_ROOT": os.environ["VLLM_CACHE_ROOT"],
        "RAY_TMPDIR": os.environ["RAY_TMPDIR"],
        "TORCH_EXTENSIONS_DIR": os.environ["TORCH_EXTENSIONS_DIR"],
    }
    for key, value in realized.items():
        try:
            Path(value).mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise CertificationFailure(
                f"Could not create {key} at {value!r}. The SOL env suite expects writable /scratch-backed cache paths."
            ) from error
    return realized


def run_preflight_suite(
    config_path: Path,
    report_dir: Path,
    *,
    verl_checkout: str | os.PathLike[str] | None = None,
) -> SuiteResult:
    _log("starting preflight suite")
    config = load_experiment_config(config_path)
    bridge_payload = build_verl_config(config)
    contract_audit = audit_bridge_config(bridge_payload)
    audit_details = contract_audit.to_dict()
    _require(contract_audit.ok, "Pinned verl contract audit failed:\n" + "\n".join(_audit_lines(audit_details)))

    details: dict[str, Any] = {
        "pinned_commit": PINNED_VERL_COMMIT,
        "required_commit": config.verl.required_commit,
        "bridge_top_level_keys": sorted(bridge_payload.keys()),
        "contract": audit_details,
    }

    live_drift = audit_live_checkout(verl_checkout)
    if live_drift is not None:
        details["live_checkout"] = live_drift.to_dict()
        for warning in live_drift.warnings:
            _log(f"live drift warning: {warning}")

    details["runtime_validation"] = _run_verl_runtime_preflight(config_path, bridge_payload)
    result = SuiteResult(name="preflight", passed=True, details=details)
    _write_json(report_dir / "preflight_report.json", result.to_dict())
    _log("preflight suite passed")
    return result


def run_env_suite(config_path: Path, report_dir: Path, *, require_gpu: bool) -> SuiteResult:
    _log("starting env suite")
    config = load_experiment_config(config_path)
    scratch_roots = _ensure_sol_cache_env()
    for key, value in scratch_roots.items():
        if value:
            _require("/scratch/" in value, f"{key} should point at /scratch on SOL, got {value!r}")

    _log("checking required imports")
    imports = {
        "torch": _check_import("torch"),
        "ray": _check_import("ray"),
        "omegaconf": _check_import("omegaconf"),
        "transformers": _check_import("transformers"),
        "verl": _check_import("verl"),
    }
    _log("initializing local ray")
    ray_details = _check_ray()
    _log("probing model/tokenizer download")
    model_probe = _check_transformers_model(config.verl.model_path, scratch_roots.get("HF_HOME") or None)

    details = {
        "python": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "scratch_paths": scratch_roots,
        "imports": imports,
        "ray": ray_details,
        "dataset_paths": {"train_files": config.data.train_files, "val_files": config.data.val_files},
        "model_probe": model_probe,
    }
    if require_gpu:
        _log("checking GPU visibility")
        details["nvidia_smi"] = _check_nvidia_smi()
        details["torch_cuda"] = _check_torch_cuda()

    result = SuiteResult(name="env", passed=True, details=details)
    _write_json(report_dir / "env_report.json", result.to_dict())
    _log("env suite passed")
    return result


def _variant_runtime_payload(
    base_payload: dict[str, Any],
    *,
    variant: str,
    backend: str,
    runtime_output_dir: Path,
    report_path: Path,
    resolved_train_files: list[str],
    resolved_val_files: list[str],
) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    payload["experiment"]["name"] = f"sol-{backend}-{variant}-smoke"
    payload["experiment"]["output_dir"] = str(runtime_output_dir)
    payload["data"]["train_files"] = resolved_train_files
    payload["data"]["val_files"] = resolved_val_files
    payload["data"]["train_batch_size"] = 2
    payload["data"]["gen_batch_size"] = 4
    payload["data"]["max_prompt_length"] = 256
    payload["data"]["max_response_length"] = 96
    payload["data"]["rollout_n"] = 2 if variant == "grpo" else 4
    payload["algorithm"]["variant"] = variant
    payload["algorithm"]["rollout_behavior"]["backend"] = backend
    payload["algorithm"]["advantage"]["mode"] = "grpo"
    payload["trainer"]["max_steps"] = 1
    payload["trainer"]["save_freq"] = 0
    payload["trainer"]["test_freq"] = 0
    payload["trainer"]["val_before_train"] = False
    payload["trainer"].setdefault("diagnostics", {})
    payload["trainer"]["diagnostics"]["report_path"] = str(report_path)
    payload["verl"]["model_path"] = TINY_SMOKE_MODEL
    payload["verl"]["reference_policy"] = False
    payload["verl"]["critic"] = False
    payload["verl"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    payload["verl"]["rollout"]["tensor_model_parallel_size"] = 1
    payload["verl"]["rollout"]["gpu_memory_utilization"] = 0.35
    payload["verl"]["rollout"]["enforce_eager"] = True

    if variant == "grpo":
        payload["reward"]["overlong"]["enabled"] = False
        payload["algorithm"]["group_filtering"]["enabled"] = False
        payload["algorithm"]["rollout_behavior"]["accumulate_filtered_groups"] = False
        payload["algorithm"]["policy_loss"]["clip_ratio_high"] = payload["algorithm"]["policy_loss"]["clip_ratio"]
        payload["algorithm"]["policy_loss"]["clip_ratio_low"] = payload["algorithm"]["policy_loss"]["clip_ratio"]
        payload["algorithm"]["policy_loss"]["clip_ratio_c"] = None
    elif variant == "gdpo":
        payload["reward"]["overlong"]["enabled"] = False
        payload["algorithm"]["group_filtering"]["enabled"] = False
        payload["algorithm"]["rollout_behavior"]["accumulate_filtered_groups"] = False
        payload["algorithm"]["policy_loss"]["clip_ratio_high"] = payload["algorithm"]["policy_loss"]["clip_ratio"]
        payload["algorithm"]["policy_loss"]["clip_ratio_low"] = payload["algorithm"]["policy_loss"]["clip_ratio"]
        payload["algorithm"]["policy_loss"]["clip_ratio_c"] = None
    else:
        payload["reward"]["overlong"]["enabled"] = True
        payload["algorithm"]["group_filtering"]["enabled"] = True
        payload["algorithm"]["rollout_behavior"]["accumulate_filtered_groups"] = True
    return payload


def run_training_subprocess(config_path: Path, work_dir: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "dapo_lab.train", str(config_path)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise CertificationFailure(
            f"Training command failed for {config_path.name}.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    report_path = Path(load_experiment_config(config_path).trainer.diagnostics.report_path or "")
    _require(report_path.exists(), f"Expected runtime report not written: {report_path}")
    payload = json.loads(report_path.read_text())
    payload["stdout"] = result.stdout
    payload["stderr"] = result.stderr
    return payload


def _validate_runtime_payload(payload: dict[str, Any], *, variant: str) -> None:
    metrics = payload["metrics"]
    stage_order = payload["stage_order"]
    _require("actor_update" in stage_order, "Runtime stage order is missing actor_update.")
    _require(math.isfinite(float(metrics["actor/loss"])), "actor/loss must be finite.")
    _require(float(metrics["trainer/actor_updates_completed"]) == 1.0, "trainer/actor_updates_completed must equal 1.")
    _require(float(metrics["trainer/steps_completed"]) == 1.0, "trainer/steps_completed must equal 1.")
    _require(float(metrics["certify/adapter_prompt_count"]) > 0.0, "Adapter prompt count must be positive.")
    _require(float(metrics["certify/adapter_trajectory_count"]) > 0.0, "Adapter trajectory count must be positive.")
    _require(float(metrics["certify/actor_param_delta_l2"]) > 0.0, "Actor delta metric must be > 0 to rule out a no-op.")
    if variant == "gdpo":
        _require(any(key.startswith("gdpo/") for key in metrics), "GDPO runtime smoke must emit gdpo/* metrics.")
    if variant == "dapo":
        _require("overlong/penalized" in metrics, "DAPO runtime smoke must emit overlong metrics.")
        _require("filtering/kept_prompts" in metrics, "DAPO runtime smoke must emit filtering metrics.")


def run_runtime_suite(
    config_path: Path,
    report_dir: Path,
    *,
    suite_name: str,
    backend: str,
    variants: list[str],
    runtime_runner: RuntimeRunner | None = None,
) -> SuiteResult:
    runtime_runner = runtime_runner or run_training_subprocess
    base_payload = _load_payload(config_path)
    base_config = load_experiment_config(config_path)
    children: list[SuiteResult] = []
    for variant in variants:
        variant_dir = report_dir / suite_name / variant
        runtime_output_dir = variant_dir / "runtime_output"
        runtime_report_path = variant_dir / "runtime_report.json"
        temp_config_path = variant_dir / "runtime_config.yaml"
        payload = _variant_runtime_payload(
            base_payload,
            variant=variant,
            backend=backend,
            runtime_output_dir=runtime_output_dir,
            report_path=runtime_report_path,
            resolved_train_files=base_config.data.train_files,
            resolved_val_files=base_config.data.val_files,
        )
        _write_payload(temp_config_path, payload)
        runtime_payload = runtime_runner(temp_config_path, config_path.resolve().parent)
        _validate_runtime_payload(runtime_payload, variant=variant)
        child = SuiteResult(name=f"{suite_name}:{variant}", passed=True, details=runtime_payload)
        _write_json(variant_dir / "suite_report.json", child.to_dict())
        children.append(child)

    result = SuiteResult(name=suite_name, passed=True, children=children, details={"backend": backend})
    _write_json(report_dir / f"{suite_name}_report.json", result.to_dict())
    return result


def run_certification(
    *,
    suite: str,
    config_path: Path,
    output_dir: str | None = None,
    runtime_runner: RuntimeRunner | None = None,
    verl_checkout: str | os.PathLike[str] | None = None,
    include_preflight: bool = True,
) -> SuiteResult:
    if suite not in SUITES:
        raise CertificationFailure(f"Unsupported suite {suite!r}. Expected one of {sorted(SUITES)}.")
    report_dir = _suite_root(config_path, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if suite == "env":
        result = run_env_suite(config_path, report_dir, require_gpu=False)
    elif suite == "preflight":
        result = run_preflight_suite(config_path, report_dir, verl_checkout=verl_checkout)
    elif suite == "debug":
        children: list[SuiteResult] = []
        if include_preflight:
            children.append(run_preflight_suite(config_path, report_dir, verl_checkout=verl_checkout))
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        children.append(env_result)
        debug_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="debug",
            backend="hf",
            variants=["grpo"],
            runtime_runner=runtime_runner,
        )
        children.append(debug_result)
        result = SuiteResult(name="debug", passed=True, children=children)
    elif suite == "hf":
        children = []
        if include_preflight:
            children.append(run_preflight_suite(config_path, report_dir, verl_checkout=verl_checkout))
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        children.append(env_result)
        hf_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="hf",
            backend="hf",
            variants=["grpo", "gdpo", "dapo"],
            runtime_runner=runtime_runner,
        )
        children.append(hf_result)
        result = SuiteResult(name="hf", passed=True, children=children)
    elif suite == "vllm":
        children = []
        if include_preflight:
            children.append(run_preflight_suite(config_path, report_dir, verl_checkout=verl_checkout))
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        children.append(env_result)
        vllm_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="vllm",
            backend="vllm",
            variants=["grpo", "gdpo", "dapo"],
            runtime_runner=runtime_runner,
        )
        children.append(vllm_result)
        result = SuiteResult(name="vllm", passed=True, children=children)
    else:
        env_result = run_env_suite(config_path, report_dir, require_gpu=False)
        preflight_result = run_preflight_suite(config_path, report_dir, verl_checkout=verl_checkout)
        debug_result = run_certification(
            suite="debug",
            config_path=config_path,
            output_dir=str(report_dir / "debug"),
            runtime_runner=runtime_runner,
            verl_checkout=verl_checkout,
            include_preflight=False,
        )
        hf_result = run_certification(
            suite="hf",
            config_path=config_path,
            output_dir=str(report_dir / "hf"),
            runtime_runner=runtime_runner,
            verl_checkout=verl_checkout,
            include_preflight=False,
        )
        vllm_result = run_certification(
            suite="vllm",
            config_path=config_path,
            output_dir=str(report_dir / "vllm"),
            runtime_runner=runtime_runner,
            verl_checkout=verl_checkout,
            include_preflight=False,
        )
        result = SuiteResult(name="all", passed=True, children=[env_result, preflight_result, debug_result, hf_result, vllm_result])

    _write_json(report_dir / "report.json", result.to_dict())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SOL certification suites for dapo_lab.")
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--config", default="config/sol_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--verl-checkout", default=None)
    args = parser.parse_args(argv)
    result = run_certification(
        suite=args.suite,
        config_path=Path(args.config),
        output_dir=args.output_dir,
        verl_checkout=args.verl_checkout,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
