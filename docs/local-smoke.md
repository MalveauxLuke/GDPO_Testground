# Local Smoke On A Mac

This repo now includes a CPU-friendly local smoke path for machines like a MacBook Pro with Apple silicon:

```bash
python -m dapo_lab.smoke config/local_smoke.yaml
python -m dapo_lab.smoke config/local_gdpo_smoke.yaml
```

This does **not** try to run the full `verl` stack locally. The upstream training runtime is still aimed at Linux + CUDA, while current `vLLM` docs separately document Apple silicon support. On a Mac, the useful local target is a deterministic "fake rollout, real local algorithm loop" run that exercises:

- reward composition
- GRPO vs GDPO vs DAPO variant dispatch
- filtering
- overlong shaping
- advantage computation
- clipped policy loss
- diagnostics and stage ordering

Primary references:

- [verl install docs](https://verl.readthedocs.io/en/latest/start/install.html)
- [vLLM installation docs](https://docs.vllm.ai/en/latest/getting_started/installation/index.html)

## Quick Start

Create a small virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If you also want the local test suite:

```bash
python -m pip install -e .[dev]
pytest -q
```

Run the smoke loop:

```bash
python -m dapo_lab.smoke config/local_smoke.yaml
python -m dapo_lab.smoke config/local_gdpo_smoke.yaml
```

The command prints the stage order, loss, and key metrics, and writes a JSON report to:

```text
outputs/local-mac-smoke/local_smoke_report.json
```

## What To Edit

`config/local_smoke.yaml` and `config/local_gdpo_smoke.yaml` are the supported local debug entrypoints.

- `config/local_smoke.yaml` is the DAPO-oriented local path.
- `config/local_gdpo_smoke.yaml` is the strict GDPO path with the DAPO-only surfaces disabled already.
- Switch to `grpo` if you want the simplest baseline.

If you switch away from `dapo`, also disable DAPO-only surfaces in the same file:

- `reward.overlong.enabled`
- `algorithm.group_filtering.enabled`
- `algorithm.rollout_behavior.accumulate_filtered_groups`
- asymmetric clipping via `clip_ratio_low` / `clip_ratio_high`

## What This Proves

This local smoke path is meant to answer:

- Does the config load?
- Does reward logic behave as expected?
- Do the variant-specific advantages differ the way we expect?
- Does the trainer loop ordering stay stable?
- Do filtering and overlong penalties change the batch the way we expect?
- Do we get a sane policy-loss number and useful metrics?

It does **not** prove:

- `verl` runtime compatibility
- Ray worker orchestration
- CUDA or `vLLM` behavior
- model loading or rollout generation from a real LLM

Those remain part of the Linux/SOL bring-up path.
