from __future__ import annotations

import json
import sys
from itertools import cycle, islice
from pathlib import Path

from dapo_lab.data.prep import load_examples
from dapo_lab.diagnostics import DiagnosticsRecorder
from dapo_lab.trainer.loop import LoopOutcome, TrainerLoop
from dapo_lab.trainer.state import BatchContext, PromptGroup, Trajectory
from dapo_lab.validation import load_experiment_config


def _cycle_examples(examples, *, count: int, offset: int = 0):
    repeated = cycle(examples)
    for _ in range(offset):
        next(repeated)
    return list(islice(repeated, count))


def _wrong_answer(ground_truth: str, *, offset: int) -> str:
    normalized = ground_truth.strip()
    if normalized.lstrip("-").isdigit():
        return str(int(normalized) + offset + 1)
    return f"wrong-{offset + 1}"


def _response_text(answer: str, *, boxed: bool) -> str:
    if boxed:
        return f"Scratch work.\nAnswer: \\boxed{{{answer}}}"
    return f"Scratch work.\nAnswer: {answer}"


def _response_token_budget(
    *,
    config,
    long_response: bool,
) -> int:
    if not long_response or not config.reward.overlong.enabled:
        return min(8, config.data.max_response_length)
    expected_length = max(config.data.max_response_length - config.reward.overlong.buffer_length, 1)
    return min(expected_length + max(config.reward.overlong.buffer_length // 2, 2), config.data.max_response_length)


def _log_prob_rows(token_count: int, *, ratio: float) -> tuple[list[float], list[float]]:
    old_log_probs = [0.0] * token_count
    new_log_probs = [ratio] * token_count
    return old_log_probs, new_log_probs


def _trajectory_for_slot(example, config, *, prompt_id: str, sample_index: int, scenario: str) -> Trajectory:
    boxed = sample_index % 2 == 0
    constant_group = scenario == "constant"
    correct = True if constant_group else sample_index % 3 != 1
    long_response = config.reward.overlong.enabled and not constant_group and sample_index == config.data.rollout_n - 1
    answer = example.ground_truth if correct else _wrong_answer(example.ground_truth, offset=sample_index)
    response = _response_text(answer, boxed=boxed)
    token_count = _response_token_budget(config=config, long_response=long_response)
    old_log_probs, new_log_probs = _log_prob_rows(token_count, ratio=0.28 if sample_index % 2 == 0 else -0.12)
    response_mask = [1] * token_count
    metrics = {
        "scenario": scenario,
        "expected_correct": float(correct),
        "expected_boxed": float(boxed),
    }
    return Trajectory(
        prompt_id=prompt_id,
        prompt=example.prompt,
        response=response,
        ground_truth=example.ground_truth,
        response_length=token_count,
        old_log_probs=old_log_probs,
        new_log_probs=new_log_probs,
        response_mask=response_mask,
        metrics=metrics,
    )


def build_smoke_batches(config) -> list[BatchContext]:
    examples = load_examples(
        config.data.train_files,
        dataset_format=config.data.format,
        prompt_key=config.data.prompt_key,
        answer_key=config.data.answer_key,
    )
    if not examples:
        raise ValueError("Smoke mode requires at least one training example.")

    prompt_count = min(max(config.data.train_batch_size, 1), max(len(examples), 1))
    scenarios = ["mixed"]
    if config.algorithm.group_filtering.enabled and config.algorithm.rollout_behavior.accumulate_filtered_groups:
        scenarios = ["constant", "mixed"]

    batches: list[BatchContext] = []
    for batch_index, scenario in enumerate(scenarios):
        groups: list[PromptGroup] = []
        selected_examples = _cycle_examples(examples, count=prompt_count, offset=batch_index * prompt_count)
        for prompt_index, example in enumerate(selected_examples):
            prompt_id = f"{example.prompt_id}:b{batch_index}:p{prompt_index}"
            trajectories = [
                _trajectory_for_slot(
                    example,
                    config,
                    prompt_id=prompt_id,
                    sample_index=sample_index,
                    scenario=scenario,
                )
                for sample_index in range(config.data.rollout_n)
            ]
            groups.append(PromptGroup(prompt_id=prompt_id, trajectories=trajectories))
        batches.append(
            BatchContext(
                groups=groups,
                metadata={
                    "mode": "local_smoke",
                    "scenario": scenario,
                    "variant": config.algorithm.variant,
                    "backend": config.algorithm.rollout_behavior.backend,
                },
            )
        )
    return batches


def _report_payload(config, outcome: LoopOutcome) -> dict:
    return {
        "experiment": config.experiment.name,
        "variant": config.algorithm.variant,
        "rollout_backend": config.algorithm.rollout_behavior.backend,
        "prompt_count": outcome.batch.prompt_count(),
        "trajectory_count": outcome.batch.trajectory_count(),
        "stage_order": outcome.stage_order,
        "loss": outcome.loss,
        "metrics": outcome.metrics,
    }


def write_smoke_report(config, outcome: LoopOutcome) -> Path:
    report_path = Path(config.trainer.diagnostics.report_path or (Path(config.experiment.output_dir) / "local_smoke_report.json"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_report_payload(config, outcome), indent=2, sort_keys=True) + "\n")
    return report_path


def run_local_smoke(config_path: str | Path) -> tuple[LoopOutcome, Path]:
    config = load_experiment_config(config_path)
    loop = TrainerLoop(config, diagnostics=DiagnosticsRecorder())
    generated_batches = build_smoke_batches(config)
    outcome = loop.run_training_step(generated_batches)
    report_path = write_smoke_report(config, outcome)
    return outcome, report_path


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config_path = Path(argv[0]) if argv else Path("config/local_smoke.yaml")
    outcome, report_path = run_local_smoke(config_path)
    print(f"local smoke completed: variant={outcome.batch.metadata.get('variant', 'n/a')}")
    print(f"stage_order={' -> '.join(outcome.stage_order)}")
    print(f"loss={outcome.loss:.6f}")
    print(f"report={report_path}")
    for key in sorted(outcome.metrics):
        print(f"{key}={outcome.metrics[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
