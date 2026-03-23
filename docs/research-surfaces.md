# Research Surfaces

This is the map for common algorithm edits.

## Reward Logic

- File: `src/dapo_lab/rewards/math.py`
- Use this for task-specific reward terms like math accuracy or format checks.

## Multiple Rewards And Weighted Composition

- File: `src/dapo_lab/rewards/composition.py`
- Add a new reward term class, register it in `TERM_FACTORIES`, and add it to `reward.terms` in the config.
- This file now preserves raw per-term scores and weighted totals separately for GDPO.

## Advantage Logic

- Files:
  - `src/dapo_lab/algorithms/variants/grpo.py`
  - `src/dapo_lab/algorithms/variants/dapo.py`
  - `src/dapo_lab/algorithms/variants/gdpo.py`
- Shared helpers live in `src/dapo_lab/algorithms/advantages.py`.
- GDPO component normalization lives only in `src/dapo_lab/algorithms/variants/gdpo.py`.

## Clip And Policy Loss Logic

- File: `src/dapo_lab/algorithms/losses.py`
- This owns ratio computation, asymmetric clipping, dual-clip behavior, and metric emission.

## Aggregation Mode

- File: `src/dapo_lab/algorithms/losses.py`
- Edit `aggregate_losses()` to experiment with new reduction schemes.

## Dynamic Sampling And Group Filtering

- File: `src/dapo_lab/algorithms/filtering.py`
- This is where prompt-group keep/drop logic and multi-batch accumulation live.

## Overlong Shaping

- File: `src/dapo_lab/algorithms/overlong.py`
- This is where the DAPO-style shaping formula and optional filtering are applied.

## Trainer Ordering

- File: `src/dapo_lab/trainer/loop.py`
- This is the main local control surface for algorithm sequencing.

## Rollout-Related Behavior

- File: `src/dapo_lab/trainer/rollout.py`
- Use this for request-level rollout changes such as sample count and decoding behavior.

## Diagnostics

- File: `src/dapo_lab/diagnostics.py`
- File: `src/dapo_lab/trainer/loop.py`
- GDPO-specific component metrics are emitted through the variant hook in `src/dapo_lab/algorithms/variants/gdpo.py`.
