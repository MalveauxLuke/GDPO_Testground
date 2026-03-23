# Overview

`dapo_lab` is a research harness for GRPO, DAPO, and GDPO RL work on top of `verl`.

The repo is intentionally split into two layers:

- Local algorithm ownership in `src/dapo_lab/`
- Upstream infrastructure delegation through `src/dapo_lab/verl_adapter/`

The goal is not to reimplement `verl`. The goal is to make the research-facing parts easy to read, easy to edit, and easy to compare across algorithm variants.

## What This Repo Optimizes For

- One readable experiment config in `config/experiment.yaml`
- One shared harness with `algorithm.variant: grpo | dapo | gdpo`
- Local ownership of reward logic, loss logic, advantage logic, filtering, overlong handling, rollout request behavior, trainer ordering, and diagnostics
- Small, explicit adapter seams where `verl` remains in charge

## What This Repo Does Not Try To Own

- Ray worker classes
- distributed controller plumbing
- FSDP or Megatron internals
- checkpoint engine internals
- cluster/runtime orchestration

## Starting Points

- Read `docs/architecture.md` for control flow.
- Read `docs/research-surfaces.md` for edit locations.
- Read `docs/dapo-mapping.md` for the GRPO-to-DAPO comparison.
- Edit `config/experiment.yaml` first.
