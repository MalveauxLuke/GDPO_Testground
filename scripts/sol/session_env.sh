#!/usr/bin/env bash

REPO_DIR="${REPO_DIR:-$HOME/GDPO_Testground}"
VERL_DIR="${VERL_DIR:-$HOME/verl}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/dapo-lab}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.6.1-gcc-12.1.0}"

export REPO_DIR
export VERL_DIR
export ENV_PREFIX
export CUDA_MODULE

export HF_HOME="${HF_HOME:-/scratch/$USER/dapo_lab/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/$USER/dapo_lab/vllm}"
export RAY_TMPDIR="${RAY_TMPDIR:-/scratch/$USER/dapo_lab/ray}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/scratch/$USER/dapo_lab/torch}"

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$RAY_TMPDIR" "$TORCH_EXTENSIONS_DIR"

echo "REPO_DIR=$REPO_DIR"
echo "VERL_DIR=$VERL_DIR"
echo "ENV_PREFIX=$ENV_PREFIX"
echo "HF_HOME=$HF_HOME"
