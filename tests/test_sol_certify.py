from __future__ import annotations

import json
from pathlib import Path

import yaml

from dapo_lab.sol_certify import SuiteResult, run_certification, run_env_suite


def _write_smoke_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text('{"prompt":"Compute 2 + 2.","ground_truth":"4"}\n')
    (data_dir / "val.jsonl").write_text('{"prompt":"Compute 1 + 4.","ground_truth":"5"}\n')
    payload = {
        "experiment": {"name": "sol-test", "seed": 1, "output_dir": str(tmp_path / "outputs")},
        "data": {
            "train_files": [str(data_dir / "train.jsonl")],
            "val_files": [str(data_dir / "val.jsonl")],
            "format": "jsonl",
            "prompt_key": "prompt",
            "answer_key": "ground_truth",
            "train_batch_size": 2,
            "gen_batch_size": 4,
            "max_prompt_length": 128,
            "max_response_length": 32,
            "rollout_n": 2,
        },
        "reward": {
            "terms": [
                {"name": "accuracy", "kind": "math_accuracy", "weight": 1.0},
                {"name": "boxed_format", "kind": "boxed_format", "weight": 0.1},
            ],
            "overlong": {"enabled": True, "mode": "shaping", "buffer_length": 8, "penalty_factor": 0.5, "hard_filter": False},
        },
        "algorithm": {
            "variant": "dapo",
            "advantage": {"mode": "grpo", "normalize_by_std": True},
            "gdpo": {
                "component_keys": ["accuracy", "boxed_format"],
                "component_weights": [1.0, 0.1],
                "normalize_by_std": True,
                "batch_whiten": True,
                "allow_single_component_ablation": False,
            },
            "policy_loss": {
                "mode": "clipped",
                "clip_ratio": 0.2,
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.28,
                "clip_ratio_c": 3.0,
                "loss_agg_mode": "token-mean",
            },
            "group_filtering": {"enabled": True, "metric": "acc", "max_num_gen_batches": 2, "require_variance": True},
            "kl": {"enabled": False, "source": "reward", "penalty": "low_var_kl"},
            "trainer_order": {
                "stages": ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
            },
            "rollout_behavior": {
                "backend": "hf",
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "accumulate_filtered_groups": True,
            },
        },
        "trainer": {
            "log_level": "info",
            "max_steps": 1,
            "diagnostics": {"report_path": str(tmp_path / "outputs" / "runtime_report.json")},
            "save_freq": 0,
            "test_freq": 0,
            "val_before_train": False,
        },
        "verl": {
            "required_commit": "08e030d9b0d6f3c5c2f154ec28bf2ccb37cab375",
            "runtime_stack": "fsdp_vllm",
            "strict_compatibility": False,
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "trust_remote_code": False,
            "actor": {"ppo_micro_batch_size_per_gpu": 1, "grad_clip": 1.0, "ppo_epochs": 1},
            "rollout": {"tensor_model_parallel_size": 1, "gpu_memory_utilization": 0.35, "enforce_eager": True},
            "reference_policy": False,
            "critic": False,
        },
    }
    config_path = tmp_path / "sol_smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def test_run_env_suite_writes_machine_readable_report(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_smoke_config(tmp_path)
    scratch_root = tmp_path / "scratch"
    monkeypatch.setenv("HF_HOME", str(scratch_root / "hf"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(scratch_root / "hf"))
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(scratch_root / "vllm"))
    monkeypatch.setenv("RAY_TMPDIR", str(scratch_root / "ray"))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(scratch_root / "torch"))

    monkeypatch.setattr("dapo_lab.sol_certify._check_import", lambda name: {"module": name})
    monkeypatch.setattr("dapo_lab.sol_certify._check_ray", lambda: {"cluster_resources": {"CPU": 1}})
    monkeypatch.setattr("dapo_lab.sol_certify._check_transformers_model", lambda model_name, cache_dir: {"model": model_name})

    result = run_env_suite(config_path, tmp_path / "reports", require_gpu=False)

    assert result.passed is True
    assert (tmp_path / "reports" / "env_report.json").exists()


def test_run_certification_hf_runs_all_variants(tmp_path: Path, monkeypatch) -> None:
    config_path = _write_smoke_config(tmp_path)

    monkeypatch.setattr("dapo_lab.sol_certify.run_env_suite", lambda config_path, report_dir, require_gpu: SuiteResult(name="env", passed=True))

    def fake_runtime_runner(runtime_config_path: Path, _work_dir: Path) -> dict:
        payload = yaml.safe_load(runtime_config_path.read_text())
        variant = payload["algorithm"]["variant"]
        metrics = {
            "actor/loss": -0.1,
            "trainer/steps_completed": 1.0,
            "trainer/actor_updates_completed": 1.0,
            "certify/adapter_prompt_count": 2.0,
            "certify/adapter_trajectory_count": 4.0,
            "certify/actor_param_delta_l2": 0.42,
        }
        if variant == "gdpo":
            metrics["gdpo/accuracy/mean"] = 0.5
        if variant == "dapo":
            metrics["overlong/penalized"] = 1.0
            metrics["filtering/kept_prompts"] = 2.0
        return {"stage_order": ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"], "metrics": metrics}

    result = run_certification(suite="hf", config_path=config_path, output_dir=str(tmp_path / "certify"), runtime_runner=fake_runtime_runner)

    assert result.passed is True
    report = json.loads((tmp_path / "certify" / "report.json").read_text())
    child_names = [child["name"] for child in report["children"][1]["children"]]
    assert child_names == ["hf:grpo", "hf:gdpo", "hf:dapo"]
