# Architecture

## Top-Level Shape

The repo is organized around a thick local algorithm layer and a thin upstream boundary.

```text
src/dapo_lab/
  config_schema.py
  validation.py
  runtime.py
  train.py
  algorithms/
  rewards/
  data/
  trainer/
  verl_adapter/
```

## Local Layer

- `config_schema.py` and `validation.py`
  - own the single YAML schema and fail-fast validation
- `rewards/`
  - own reward terms and weighted reward composition
- `algorithms/registry.py`
  - owns the neutral variant dispatch
- `algorithms/variants/`
  - owns visibly separate GRPO, DAPO, and GDPO behavior
- `algorithms/advantages.py`
  - owns only shared advantage math helpers
- `algorithms/losses.py`
  - owns clipping behavior, dual-clip support, and aggregation modes
- `algorithms/filtering.py`
  - owns prompt-group filtering and multi-batch accumulation
- `algorithms/overlong.py`
  - owns overlong shaping and filtering behavior
- `trainer/loop.py`
  - owns the readable research loop ordering
- `trainer/rollout.py`
  - owns rollout request construction

## Upstream Boundary

- `verl_adapter/config_bridge.py`
  - maps the local config into a minimal `verl`-shaped config
- `verl_adapter/compat.py`
  - checks the expected upstream commit and import state
- `verl_adapter/batch_adapter.py`
  - owns the local-to-upstream batch translation helpers
- `verl_adapter/trainer.py`
  - keeps the local algorithm loop in control while delegating worker plumbing and actor updates
- `verl_adapter/task_runner.py`
  - hooks into the standard `verl` task-runner entry path

## Control Flow

1. `python -m dapo_lab.train config/experiment.yaml`
2. `train.py` loads `ResearchRuntime`
3. `runtime.py` loads and validates config, then checks `verl` compatibility
4. `verl_adapter/task_runner.py` builds the minimal upstream config
5. `ResearchTrainer` collects generated batches from `verl`
6. `TrainerLoop.run_training_step()` applies:
   - rollout
   - reward
   - KL
   - filtering
   - overlong handling when enabled
   - advantage
   - actor update
   - diagnostics

## Visible Variant Boundaries

The main algorithm split now lives here:

- `src/dapo_lab/algorithms/variants/grpo.py`
- `src/dapo_lab/algorithms/variants/dapo.py`
- `src/dapo_lab/algorithms/variants/gdpo.py`

Those files have explicit ASCII comment banners and are the intended first stop when you want to compare algorithms.

## Intentional Dangerous Spots

The main live-runtime risk now sits in the batch bridge and actor-update handoff:

- `verl_adapter/batch_adapter.py`
- `ResearchTrainer.build_local_batch()`
- `ResearchTrainer.apply_outcome_to_upstream_batch()`
- `ResearchTrainer.update_actor_from_outcome()`

Those spots are dangerous because they touch the upstream batch representation and the real actor update path. The rest of the algorithm files remain the preferred research-edit surface.
