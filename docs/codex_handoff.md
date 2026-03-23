# Codex Handoff: SOL `verl` Bring-Up Status

Last updated: 2026-03-23

## Snapshot

- Repo: `/Users/god/Documents/GRPO`
- Branch: `main`
- Local HEAD at handoff: `b0dc6cf` (`Add sampler to pinned verl data contract`)
- Pinned upstream `verl` contract commit: `7dc46e834209948cf1cdd8a04d83f82b4a7efd24`
- SOL repo path: `~/GDPO_Testground`
- SOL `verl` checkout path: `~/verl`
- SOL env path: `~/.conda/envs/dapo-lab`

## Main Diagnosis

The repeated SOL failures were not primarily GDPO math bugs. The main problem was that our local `verl` compatibility layer was too sparse for the current upstream `verl` runtime contract.

The fix direction is now:

1. pin one exact upstream `verl` revision
2. build the bridge from a checked-in scaffold for that revision
3. run a non-GPU preflight before any GPU job
4. only then run `debug`, `hf`, `vllm`

Do not go back to the old sparse hand-built bridge approach unless you also re-audit the full upstream contract.

## What Was Implemented

### Pinned contract + scaffold

These files are now the source of truth for the bridge:

- `src/dapo_lab/verl_adapter/contract.py`
- `src/dapo_lab/verl_adapter/contracts/verl_7dc46e8_contract.yaml`
- `src/dapo_lab/verl_adapter/contracts/verl_7dc46e8_scaffold.yaml`

They define:

- the pinned commit
- required top-level sections
- required nested paths
- required `_target_` blocks
- advisory live-checkout drift detection

### Bridge now uses scaffold + overlay

- `src/dapo_lab/verl_adapter/config_bridge.py`

This now deep-copies the pinned scaffold and overlays repo-owned values instead of building a tiny dict from scratch.

### Preflight suite added

- `src/dapo_lab/sol_certify.py`

New command:

```bash
python -m dapo_lab.sol_certify --suite preflight --config config/sol_smoke.yaml --verl-checkout "$VERL_DIR"
```

Preflight now:

1. builds the bridged config
2. audits it against the pinned contract
3. optionally compares it against the live `verl` checkout
4. runs upstream:
   - `OmegaConf.create(...)`
   - `auto_set_device(...)`
   - `migrate_legacy_reward_impl(...)`
   - `validate_config(...)`

### Runtime launch guarded

- `src/dapo_lab/verl_adapter/task_runner.py`

The runtime now re-audits the bridged config before launching `run_ppo(...)`, so contract failures are raised earlier and more clearly.

### SOL job scripts updated

- `scripts/sol/run_certify_debug.sbatch`
- `scripts/sol/run_certify_hf.sbatch`
- `scripts/sol/run_certify_vllm.sbatch`

They now pass `--verl-checkout "$VERL_DIR"` so GPU jobs also emit advisory drift warnings.

## What Is Confirmed Working

### Local repo checks

These passed locally at this checkpoint:

```bash
python -m pytest -q
python -m dapo_lab.smoke config/local_gdpo_smoke.yaml
```

Latest local result:

- `44 passed, 1 skipped`
- local GDPO smoke passed and emitted `gdpo/*` metrics

### SOL checks

These are known-good on SOL from this conversation:

```bash
$ENV_PREFIX/bin/python -m pytest -q tests
$ENV_PREFIX/bin/python -m dapo_lab.smoke config/local_gdpo_smoke.yaml
$ENV_PREFIX/bin/python -m dapo_lab.sol_certify --suite env --config config/sol_smoke.yaml
$ENV_PREFIX/bin/python -m dapo_lab.sol_certify --suite preflight --config config/sol_smoke.yaml --verl-checkout "$VERL_DIR"
```

The SOL preflight passed with:

- pinned contract audit OK
- live checkout commit matching pinned commit
- upstream `validate_config(...)` passing

At one point there was an advisory drift warning for `data.gen_batch_size`; that was removed from the hard contract because it was not an actual runtime requirement.

## Latest Runtime State

The last concrete runtime failure before the current checkpoint was:

- missing `data.sampler`

This came from upstream `verl` here:

- `verl/trainer/main_ppo.py -> create_rl_sampler(...)`

That gap has already been fixed in commit `b0dc6cf` by adding `data.sampler` into the pinned scaffold and contract.

After that fix, the user launched another `debug` job on SOL and reported:

- no error for several minutes
- no final outcome captured yet in this thread

So the exact current job result is unknown at handoff time. The next agent should first check whether the current `debug` job succeeded, is still running, or failed with a new runtime error.

## Exact Commands To Resume From Here

### 1. Check the active `debug` job on SOL

In the SOL shell:

```bash
cd ~/GDPO_Testground
source scripts/sol/session_env.sh
squeue -u "$USER"
```

If a `dapo-cert-debug` job is running, inspect the log:

```bash
tail -f dapo-cert-debug-<jobid>.out
```

Use `Ctrl+C` to leave `tail -f`.

### 2. If the current job already finished, inspect the outcome

Check for the runtime report:

```bash
find ~/GDPO_Testground/outputs -name "report.json" -o -name "*report.json"
```

For a successful `debug` run, confirm:

- `actor_update` appears in `stage_order`
- `trainer/steps_completed == 1`
- `trainer/actor_updates_completed == 1`
- `certify/actor_param_delta_l2 > 0`

### 3. If the job failed, rerun with the current contract path

```bash
cd ~/GDPO_Testground
git pull origin main
source scripts/sol/session_env.sh
$ENV_PREFIX/bin/python -m dapo_lab.sol_certify --suite preflight --config config/sol_smoke.yaml --verl-checkout "$VERL_DIR"
sbatch scripts/sol/run_certify_debug.sbatch
```

### 4. If `debug` passes, continue in this order

```bash
sbatch scripts/sol/run_certify_hf.sbatch
sbatch scripts/sol/run_certify_vllm.sbatch
```

Only after those are green:

```bash
sbatch scripts/sol/run_research.sbatch
```

## Mandatory Workflow Rules

### In every new SOL shell

Always run:

```bash
cd ~/GDPO_Testground
source scripts/sol/session_env.sh
```

This sets:

- `REPO_DIR`
- `VERL_DIR`
- `ENV_PREFIX`
- `HF_HOME`
- `TRANSFORMERS_CACHE`
- `VLLM_CACHE_ROOT`
- `RAY_TMPDIR`
- `TORCH_EXTENSIONS_DIR`

### Use direct env Python if activation is flaky

Preferred:

```bash
$ENV_PREFIX/bin/python ...
```

This is often more reliable than shell activation in fresh SOL terminals.

### Preflight is mandatory

Before any GPU runtime submission, run:

```bash
$ENV_PREFIX/bin/python -m dapo_lab.sol_certify --suite preflight --config config/sol_smoke.yaml --verl-checkout "$VERL_DIR"
```

If preflight fails, fix the bridge or contract before burning more GPU time.

## Historical Failures Already Addressed

These specific failure classes were already fixed during this thread:

- missing legacy reward config stubs:
  - `reward_model`
  - `custom_reward_function`
  - `sandbox_fusion`
- missing runtime top-level keys:
  - `ray_kwargs`
  - `transfer_queue`
  - `global_profiler`
- missing nested actor/rollout fields:
  - `use_dynamic_bsz`
  - rollout/ref log-prob settings
- missing Hydra `_target_` blocks
- bad actor mini-batch sizing:
  - `train_batch_size` vs `ppo_mini_batch_size`
- missing `data.sampler`

Future failures are now more likely to be real runtime/data/worker issues rather than simple config-shape surprises.

## Files A Future Codex Should Read First

If a new thread needs to continue debugging, start with:

- `docs/codex_handoff.md`
- `docs/sol.md`
- `src/dapo_lab/sol_certify.py`
- `src/dapo_lab/verl_adapter/contract.py`
- `src/dapo_lab/verl_adapter/config_bridge.py`
- `src/dapo_lab/verl_adapter/task_runner.py`
- `src/dapo_lab/verl_adapter/contracts/verl_7dc46e8_contract.yaml`
- `src/dapo_lab/verl_adapter/contracts/verl_7dc46e8_scaffold.yaml`

Then compare any new missing-key/runtime error against upstream `verl` source, especially:

- `verl/trainer/main_ppo.py`
- `verl/utils/config.py`
- `verl/experimental/reward_loop/reward_loop.py`
- `verl/trainer/config/_generated_ppo_trainer.yaml`

## One-Paragraph Context For A Fresh Codex Thread

We pinned `verl` compatibility to commit `7dc46e834209948cf1cdd8a04d83f82b4a7efd24`, replaced the sparse local bridge with a scaffold-plus-overlay bridge, added `sol_certify --suite preflight` to catch upstream config/runtime mismatches before GPU jobs, and updated SOL job scripts to reuse that guardrail. Local tests and local GDPO smoke are green, SOL `env` and `preflight` are green, and the latest unresolved question is whether the most recent `debug` job now completes the first real one-step runtime update after the `data.sampler` fix in commit `b0dc6cf`.
