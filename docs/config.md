# Config

The repo uses one source-of-truth file: `config/experiment.yaml`.

## Sections

- `experiment`
  - run name, seed, output directory
- `data`
  - dataset paths, dataset keys, batch sizes, rollout multiplicity, prompt/response limits
- `reward`
  - weighted reward terms and overlong settings
- `algorithm`
  - `variant`, advantage mode, GDPO controls, clip settings, aggregation mode, filtering settings, rollout behavior, loop ordering
- `trainer`
  - diagnostics and validation cadence
- `verl`
  - the smallest required upstream runtime fields

## Key Knobs

- `algorithm.variant`
  - `grpo` for the clean baseline
  - `dapo` for asymmetric clipping, dynamic sampling, and overlong handling
  - `gdpo` for decoupled multi-reward normalization
- `algorithm.gdpo.component_keys`
  - ordered reward component names used by GDPO
- `algorithm.gdpo.component_weights`
  - weights used after per-component group normalization
- `algorithm.gdpo.normalize_by_std`
  - whether each reward component is normalized by group std
- `algorithm.gdpo.batch_whiten`
  - whether the combined GDPO scalar is batch-whitened before token broadcast
- `algorithm.policy_loss.clip_ratio`
  - symmetric PPO clip used by GRPO
- `algorithm.policy_loss.clip_ratio_low`
  - lower asymmetric clip bound used by DAPO
- `algorithm.policy_loss.clip_ratio_high`
  - upper asymmetric clip bound used by DAPO
- `algorithm.policy_loss.clip_ratio_c`
  - optional dual-clip lower bound
- `algorithm.policy_loss.loss_agg_mode`
  - `token-mean`
  - `seq-mean-token-sum`
  - `seq-mean-token-mean`
- `algorithm.group_filtering.*`
  - dynamic sampling and prompt-group filtering controls
- `reward.overlong.*`
  - overlong shaping and filtering controls

## Config Flow

1. YAML is parsed in `validation.load_experiment_config()`
2. The dataclass schema is built in `config_schema.py`
3. Variant-aware checks run in `validation.validate_experiment_config()`
4. The same config object drives:
   - local algorithm modules
   - the rollout request builder
   - the minimal `verl` bridge

## Why The Config Is Flat Enough

The schema keeps names close to `verl` where that reduces translation cost, but it avoids a large Hydra inheritance tree. The main research knobs live directly under `algorithm`, `reward`, and `data`.
