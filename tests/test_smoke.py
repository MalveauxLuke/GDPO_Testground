from __future__ import annotations

import json
from pathlib import Path

import yaml

from dapo_lab.smoke import run_local_smoke
from dapo_lab.verl_adapter.contract import PINNED_VERL_COMMIT


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _write_smoke_config(tmp_path: Path, *, variant: str) -> Path:
    train_path = tmp_path / "data" / "train.jsonl"
    val_path = tmp_path / "data" / "val.jsonl"
    _write_jsonl(
        train_path,
        [
            {"prompt_id": "p1", "prompt": "Compute 2 + 2.", "ground_truth": "4"},
            {"prompt_id": "p2", "prompt": "Compute 3 + 5.", "ground_truth": "8"},
        ],
    )
    _write_jsonl(
        val_path,
        [
            {"prompt_id": "v1", "prompt": "Compute 1 + 4.", "ground_truth": "5"},
        ],
    )

    payload = {
        "experiment": {"name": f"{variant}-smoke", "seed": 1, "output_dir": str(tmp_path / "outputs")},
        "data": {
            "train_files": [str(train_path)],
            "val_files": [str(val_path)],
            "format": "jsonl",
            "prompt_key": "prompt",
            "answer_key": "ground_truth",
            "train_batch_size": 2,
            "gen_batch_size": 4,
            "max_prompt_length": 128,
            "max_response_length": 24,
            "rollout_n": 4,
        },
        "reward": {
            "terms": [
                {"name": "accuracy", "kind": "math_accuracy", "weight": 1.0},
                {"name": "boxed_format", "kind": "boxed_format", "weight": 0.1},
            ],
            "overlong": {
                "enabled": variant == "dapo",
                "mode": "shaping",
                "buffer_length": 6,
                "penalty_factor": 0.5,
                "hard_filter": False,
                "log": True,
            },
        },
        "algorithm": {
            "variant": variant,
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
                "clip_ratio_high": 0.28 if variant == "dapo" else 0.2,
                "clip_ratio_c": 3.0 if variant == "dapo" else None,
                "loss_agg_mode": "token-mean",
            },
            "group_filtering": {
                "enabled": variant == "dapo",
                "metric": "acc",
                "max_num_gen_batches": 2,
                "require_variance": True,
            },
            "kl": {"enabled": False, "source": "reward", "penalty": "low_var_kl"},
            "trainer_order": {
                "stages": ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
            },
            "rollout_behavior": {
                "backend": "local-smoke",
                "do_sample": False,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "accumulate_filtered_groups": variant == "dapo",
            },
        },
        "trainer": {
            "log_level": "info",
            "diagnostics": {
                "record_stage_events": True,
                "emit_group_stats": True,
                "emit_reward_breakdown": True,
            },
            "save_freq": 0,
            "test_freq": 0,
            "val_before_train": False,
        },
        "verl": {
            "required_commit": PINNED_VERL_COMMIT,
            "runtime_stack": "local_smoke",
            "strict_compatibility": False,
            "model_path": "not-used-in-local-smoke",
            "trust_remote_code": False,
            "actor": {"ppo_micro_batch_size_per_gpu": 1, "grad_clip": 1.0, "ppo_epochs": 1},
            "rollout": {"tensor_model_parallel_size": 1, "gpu_memory_utilization": 0.0, "enforce_eager": True},
            "reference_policy": False,
            "critic": False,
        },
    }

    config_path = tmp_path / f"{variant}_smoke.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return config_path


def test_local_smoke_runner_exercises_dapo_path(tmp_path: Path) -> None:
    config_path = _write_smoke_config(tmp_path, variant="dapo")

    outcome, report_path = run_local_smoke(config_path)

    assert report_path.exists()
    assert outcome.stage_order[0] == "rollout"
    assert outcome.stage_order[-1] == "diagnostics"
    assert outcome.stage_order.count("reward") == 2
    assert outcome.metrics["filtering/kept_prompts"] >= 1.0
    assert outcome.metrics["overlong/penalized"] >= 1.0


def test_local_smoke_runner_exercises_gdpo_path(tmp_path: Path) -> None:
    config_path = _write_smoke_config(tmp_path, variant="gdpo")

    outcome, report_path = run_local_smoke(config_path)
    payload = json.loads(report_path.read_text())

    assert outcome.stage_order == ["rollout", "reward", "kl", "filtering", "advantage", "actor_update", "diagnostics"]
    assert "gdpo/accuracy/mean" in outcome.metrics
    assert payload["variant"] == "gdpo"
