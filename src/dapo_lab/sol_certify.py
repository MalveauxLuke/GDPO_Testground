from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from dapo_lab.validation import load_experiment_config


SUITES = {"env", "debug", "hf", "vllm", "all"}
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
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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


def _check_import(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise CertificationFailure(f"Required module is not importable: {module_name}")
    module = importlib.import_module(module_name)
    return {"module": module_name, "location": getattr(module, "__file__", "built-in")}


def _check_ray() -> dict[str, Any]:
    ray = importlib.import_module("ray")
    ray.init(local_mode=True, ignore_reinit_error=True)
    resources = ray.cluster_resources()
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
    config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
    return {
        "tokenizer_class": tokenizer.__class__.__name__,
        "model_type": getattr(config, "model_type", "unknown"),
    }


def run_env_suite(config_path: Path, report_dir: Path, *, require_gpu: bool) -> SuiteResult:
    config = load_experiment_config(config_path)
    scratch_roots = {
        "HF_HOME": os.environ.get("HF_HOME", ""),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
        "VLLM_CACHE_ROOT": os.environ.get("VLLM_CACHE_ROOT", ""),
        "RAY_TMPDIR": os.environ.get("RAY_TMPDIR", ""),
        "TORCH_EXTENSIONS_DIR": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
    }
    for key, value in scratch_roots.items():
        if value:
            _require("/scratch/" in value, f"{key} should point at /scratch on SOL, got {value!r}")

    details = {
        "python": sys.executable,
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "scratch_paths": scratch_roots,
        "imports": {
            "torch": _check_import("torch"),
            "ray": _check_import("ray"),
            "omegaconf": _check_import("omegaconf"),
            "transformers": _check_import("transformers"),
            "verl": _check_import("verl"),
        },
        "ray": _check_ray(),
        "dataset_paths": {"train_files": config.data.train_files, "val_files": config.data.val_files},
        "model_probe": _check_transformers_model(config.verl.model_path, scratch_roots.get("HF_HOME") or None),
    }
    if require_gpu:
        details["nvidia_smi"] = _check_nvidia_smi()
        details["torch_cuda"] = _check_torch_cuda()

    result = SuiteResult(name="env", passed=True, details=details)
    _write_json(report_dir / "env_report.json", result.to_dict())
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
) -> SuiteResult:
    if suite not in SUITES:
        raise CertificationFailure(f"Unsupported suite {suite!r}. Expected one of {sorted(SUITES)}.")
    report_dir = _suite_root(config_path, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if suite == "env":
        result = run_env_suite(config_path, report_dir, require_gpu=False)
    elif suite == "debug":
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        debug_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="debug",
            backend="hf",
            variants=["grpo"],
            runtime_runner=runtime_runner,
        )
        result = SuiteResult(name="debug", passed=True, children=[env_result, debug_result])
    elif suite == "hf":
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        hf_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="hf",
            backend="hf",
            variants=["grpo", "gdpo", "dapo"],
            runtime_runner=runtime_runner,
        )
        result = SuiteResult(name="hf", passed=True, children=[env_result, hf_result])
    elif suite == "vllm":
        env_result = run_env_suite(config_path, report_dir, require_gpu=True)
        vllm_result = run_runtime_suite(
            config_path,
            report_dir,
            suite_name="vllm",
            backend="vllm",
            variants=["grpo", "gdpo", "dapo"],
            runtime_runner=runtime_runner,
        )
        result = SuiteResult(name="vllm", passed=True, children=[env_result, vllm_result])
    else:
        env_result = run_env_suite(config_path, report_dir, require_gpu=False)
        debug_result = run_certification(suite="debug", config_path=config_path, output_dir=str(report_dir / "debug"), runtime_runner=runtime_runner)
        hf_result = run_certification(suite="hf", config_path=config_path, output_dir=str(report_dir / "hf"), runtime_runner=runtime_runner)
        vllm_result = run_certification(suite="vllm", config_path=config_path, output_dir=str(report_dir / "vllm"), runtime_runner=runtime_runner)
        result = SuiteResult(name="all", passed=True, children=[env_result, debug_result, hf_result, vllm_result])

    _write_json(report_dir / "report.json", result.to_dict())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SOL certification suites for dapo_lab.")
    parser.add_argument("--suite", required=True, choices=sorted(SUITES))
    parser.add_argument("--config", default="config/sol_smoke.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_certification(suite=args.suite, config_path=Path(args.config), output_dir=args.output_dir)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
