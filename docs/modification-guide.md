# Modification Guide

## Safest Files To Modify First

- `src/dapo_lab/rewards/math.py`
- `src/dapo_lab/rewards/composition.py`
- `src/dapo_lab/algorithms/variants/grpo.py`
- `src/dapo_lab/algorithms/variants/dapo.py`
- `src/dapo_lab/algorithms/variants/gdpo.py`
- `src/dapo_lab/algorithms/advantages.py`
- `src/dapo_lab/algorithms/losses.py`
- `src/dapo_lab/algorithms/filtering.py`
- `src/dapo_lab/algorithms/overlong.py`
- `src/dapo_lab/trainer/loop.py`

These files are local and covered by tests.

## More Dangerous Files

- `src/dapo_lab/verl_adapter/trainer.py`
- `src/dapo_lab/verl_adapter/task_runner.py`
- `src/dapo_lab/verl_adapter/config_bridge.py`

These are dangerous because they control the live boundary with `verl`.

## Common Changes

### Add A New Reward Term

1. Implement the term in `rewards/`
2. Register it in `rewards/composition.py`
3. Add it to `reward.terms` in the config
4. Add or update a unit test

### Add A New Advantage Estimator

1. Decide whether it belongs to `grpo`, `dapo`, or `gdpo`
2. Add it in the corresponding file under `algorithms/variants/`
3. Put only shared math helpers in `algorithms/advantages.py`
4. Add a focused unit test

### Change Clipping Behavior

1. Edit `algorithms/losses.py`
2. Keep metric emission intact so comparisons remain easy
3. Add a comparison test against the old behavior

### Change Dynamic Sampling

1. Edit `algorithms/filtering.py`
2. Keep the trainer loop unchanged unless the stage ordering itself should change

### Change Trainer Ordering

1. Edit `trainer/loop.py`
2. Update the loop tests to lock the new order

## Trace A Training Decision

Start in `trainer/loop.py`, then follow the called module:

- reward decision -> `rewards/`
- filtering decision -> `algorithms/filtering.py`
- advantage decision -> `algorithms/variants/`
- loss decision -> `algorithms/losses.py`
