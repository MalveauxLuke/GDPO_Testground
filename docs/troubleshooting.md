# Troubleshooting

## `verl is not installed`

The repo is intentionally light on bundled infrastructure. Install a compatible `verl` environment before launching real training.

Check:

- `src/dapo_lab/verl_adapter/compat.py`
- `config/experiment.yaml -> verl.required_commit`

## `group filtering must be false for grpo`

The naive GRPO baseline in this repo is intentionally simple. Disable:

- `algorithm.group_filtering.enabled`
- `reward.overlong.enabled`

or switch to `algorithm.variant: dapo`.

## `Strict GDPO v1 does not enable ...`

The GDPO variant in this repo is intentionally strict by default. Disable:

- `algorithm.group_filtering.enabled`
- `reward.overlong.enabled`
- asymmetric `clip_ratio_low` / `clip_ratio_high`

or switch to `algorithm.variant: dapo`.

## `clip_ratio_c must be > 1.0`

The dual-clip lower bound follows the usual PPO dual-clip constraint. Set a value greater than `1.0` or omit it.

## Adapter or certification metrics failed

The live `verl` bridge now materializes local batches and actor-update batches, but this is still the thinnest and most dangerous integration seam in the repo.

If certification fails on:

- `certify/adapter_prompt_count`
- `certify/adapter_trajectory_count`
- `certify/actor_param_delta_l2`

start here:

- `src/dapo_lab/verl_adapter/trainer.py`
- `src/dapo_lab/verl_adapter/batch_adapter.py`
- `src/dapo_lab/sol_certify.py`

## Dataset Loader Errors

The shipped loader is stdlib-first and only supports JSONL locally. If you want parquet in the local prep path, add it in `src/dapo_lab/data/prep.py` instead of spreading dataset logic across the repo.
