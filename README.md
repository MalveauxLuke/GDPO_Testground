# dapo_lab

Research harness for GRPO, DAPO, and GDPO RL work on top of `verl`.

Install with `python -m pip install -e .`, start with [docs/overview.md](/Users/god/Documents/GRPO/docs/overview.md), edit [config/experiment.yaml](/Users/god/Documents/GRPO/config/experiment.yaml), and launch with `python -m dapo_lab.train config/experiment.yaml`.

For a Mac-friendly local smoke run that avoids the full `verl` runtime, see [docs/local-smoke.md](/Users/god/Documents/GRPO/docs/local-smoke.md) and run `python -m dapo_lab.smoke config/local_smoke.yaml`.

For the staged SOL bring-up and certification flow, see [docs/sol.md](/Users/god/Documents/GRPO/docs/sol.md).
