#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/GDPO_Testground}"
VERL_DIR="${VERL_DIR:-$HOME/verl}"
ENV_PREFIX="${ENV_PREFIX:-$HOME/.conda/envs/dapo-lab}"
CUDA_MODULE="${CUDA_MODULE:-cuda-12.6.1-gcc-12.1.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

source "$(dirname "$0")/session_env.sh"

module load mamba/latest
module load "$CUDA_MODULE"

if [ ! -d "$ENV_PREFIX" ]; then
  mamba create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION"
fi
source activate "$ENV_PREFIX"

python -m pip install --upgrade pip
python -m pip install ray omegaconf transformers

if [ ! -d "$VERL_DIR" ]; then
  echo "Expected a verl checkout at $VERL_DIR" >&2
  exit 1
fi

python -m pip install -e "$VERL_DIR"
python -m pip install -e "$REPO_DIR"
python -m pip install pytest

python - <<'PY'
import sys
import yaml
print("python:", sys.executable)
print("yaml:", yaml.__version__)
PY
