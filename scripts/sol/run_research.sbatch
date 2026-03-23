#!/usr/bin/env bash
#SBATCH -J dapo-research
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 0-08:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -o %x-%j.out

set -euo pipefail

source "$(dirname "$0")/session_env.sh"

module load mamba/latest
module load "$CUDA_MODULE"
source activate "$ENV_PREFIX"

cd "$REPO_DIR"
python -m dapo_lab.train config/experiment.yaml
