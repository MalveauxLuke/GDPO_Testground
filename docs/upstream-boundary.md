# Upstream Boundary

## What Stays Local

The files below are intended to be edited during research:

- `src/dapo_lab/rewards/`
- `src/dapo_lab/algorithms/variants/`
- `src/dapo_lab/algorithms/advantages.py`
- `src/dapo_lab/algorithms/losses.py`
- `src/dapo_lab/algorithms/filtering.py`
- `src/dapo_lab/algorithms/overlong.py`
- `src/dapo_lab/trainer/loop.py`
- `src/dapo_lab/trainer/rollout.py`
- `src/dapo_lab/validation.py`
- `src/dapo_lab/diagnostics.py`

## What Stays Delegated To `verl`

- worker classes
- Ray orchestration
- distributed controller plumbing
- rollout engine internals
- FSDP and Megatron worker internals
- checkpoint internals

## Adapter Files

- `src/dapo_lab/verl_adapter/config_bridge.py`
  - converts local config to a minimal upstream shape
- `src/dapo_lab/verl_adapter/compat.py`
  - checks the expected upstream commit and runtime availability
- `src/dapo_lab/verl_adapter/trainer.py`
  - contains the explicit batch-conversion seams
- `src/dapo_lab/verl_adapter/task_runner.py`
  - hooks into the standard `verl` launch path

## Smallest Boundary Principle

If a change affects worker lifecycle, cluster orchestration, checkpoint persistence, or rollout engine internals, it should stay upstream.

If a change affects reward semantics, filtering semantics, policy loss behavior, advantage behavior, or trainer ordering, it should stay local.
