#!/usr/bin/env bash
#SBATCH -J dapo-cert-vllm
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 0-04:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -o %x-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/GDPO_Testground}"
source "$REPO_DIR/scripts/sol/session_env.sh"

module load mamba/latest
module load "$CUDA_MODULE"
source activate "$ENV_PREFIX"

cd "$REPO_DIR"
python -m dapo_lab.sol_certify --suite vllm --config config/sol_smoke.yaml
