# SOL Certification Guide

This repo now has a staged SOL bring-up path built around `python -m dapo_lab.sol_certify`.

The intended order is:

1. local smoke on your Mac
2. SOL environment certification
3. SOL debug runtime certification
4. SOL HF certification matrix
5. SOL vLLM certification matrix
6. first real research run

Primary references:

- [ASU New User Guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/1905721457)
- [ASU Mamba guide](https://asurc.atlassian.net/wiki/spaces/RC/pages/1905328428)
- [ASU Partitions and QoS](https://asurc.atlassian.net/wiki/spaces/RC/pages/1908867081)
- [ASU Building Software](https://asurc.atlassian.net/wiki/spaces/RC/pages/1993932838/Building%2BSoftware%2Bon%2BASU%2BSupercomputers?atl_f=content-tree)
- [verl install docs](https://verl.readthedocs.io/en/v0.4.1/start/install.html)

## 1. Local Gate

Before SOL, pass these locally:

```bash
cd /Users/god/Documents/GRPO
python -m pytest -q
# DAPO-oriented local smoke
python -m dapo_lab.smoke config/local_smoke.yaml
# strict GDPO local smoke
python -m dapo_lab.smoke config/local_gdpo_smoke.yaml
```

## 2. Recommended SOL Layout

Persistent:

- repo: `/data/grp_XXXX/dapo_lab` or `~/GDPO_Testground`
- env: `/data/grp_XXXX/.conda/envs/dapo-lab` or `~/.conda/envs/dapo-lab`
- reports/checkpoints: `/data/grp_XXXX/runs/dapo_lab/...`

Scratch:

- `HF_HOME=/scratch/$USER/dapo_lab/hf`
- `TRANSFORMERS_CACHE=/scratch/$USER/dapo_lab/hf`
- `VLLM_CACHE_ROOT=/scratch/$USER/dapo_lab/vllm`
- `RAY_TMPDIR=/scratch/$USER/dapo_lab/ray`
- `TORCH_EXTENSIONS_DIR=/scratch/$USER/dapo_lab/torch`

## 3. Bootstrap Every New Shell

In every fresh SOL terminal, run:

```bash
cd ~/GDPO_Testground
source scripts/sol/session_env.sh
```

This exports:

- `REPO_DIR`
- `VERL_DIR`
- `ENV_PREFIX`
- scratch-backed cache paths

It also creates the scratch cache directories so `sol_certify` and the runtime jobs do not depend on stale shell state.

## 4. Build The Environment

Use a lightwork-style allocation first:

```bash
interactive -p lightwork -q public -t 0-02:00:00 -c 4 --mem=16G
```

Then run:

```bash
cd ~/GDPO_Testground
source scripts/sol/session_env.sh
bash scripts/sol/create_env.sh
```

That script uses system `mamba`, creates a Python 3.10 env, installs `verl` in editable mode, installs this repo in editable mode, and keeps caches on scratch.

## 5. Run Certification

Before the GPU suites, the non-GPU env suite should now print progress lines such as:

- `starting env suite`
- `checking required imports`
- `initializing local ray`
- `probing model/tokenizer download`
- `env suite passed`

That output is the easiest way to confirm it is actively working instead of silently hanging.

Short GPU debug check:

```bash
sbatch scripts/sol/run_certify_debug.sbatch
```

Full HF matrix:

```bash
sbatch scripts/sol/run_certify_hf.sbatch
```

Full vLLM matrix:

```bash
sbatch scripts/sol/run_certify_vllm.sbatch
```

Each job runs `python -m dapo_lab.sol_certify` and writes machine-readable reports under the configured output directory.

## 6. What The Certifier Proves

Environment suite:

- Python version
- required imports
- Ray init/shutdown
- scratch-path setup
- dataset paths exist
- tiny public model download works
- CUDA visibility when a GPU is allocated

Runtime suites:

- the real `python -m dapo_lab.train ...` entrypoint runs
- the trainer reaches `actor_update`
- exactly one update step completes
- adapter prompt/trajectory counts are non-zero
- the post-update actor behavior changes on the batch
- GRPO, GDPO, and DAPO emit the expected variant metrics

## 7. Promote To Research

Only after `debug`, `hf`, and `vllm` are green should you submit:

```bash
sbatch scripts/sol/run_research.sbatch
```

That uses `config/experiment.yaml` instead of the smoke config.
