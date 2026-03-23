#!/usr/bin/env bash
#SBATCH -J dapo-cert-debug
#SBATCH -p htc
#SBATCH -q debug
#SBATCH -t 0-00:15:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -o %x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/GRPO}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/dapo-lab}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.6.1-gcc-12.1.0}"

export HF_HOME="${HF_HOME:-/scratch/$USER/dapo_lab/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/$USER/dapo_lab/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/scratch/$USER/dapo_lab/ray}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch/$USER/dapo_lab/torch}"

module load mamba/latest
module load "$CUDA_MODULE"
source activate "$ENV_PREFIX"

cd "$REPO_DIR"
python -m dapo_lab.sol_certify --suite debug --config config/sol_smoke.yaml
