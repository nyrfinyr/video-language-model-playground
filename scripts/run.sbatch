#!/bin/bash
#SBATCH --job-name=qwen-vl-inf
#SBATCH --account=sai2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

set -euo pipefail

# Diagnostica nodo + GPU assegnata (utile in caso di errori).
echo "node: $(hostname)"
echo "date: $(date -Iseconds)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

# Stack trace completo se Hydra intercetta un'eccezione.
export HYDRA_FULL_ERROR=1

# HF_HOME via Hydra (vedi conf/run/default.yaml + main.py:35-36).
# La P100 non supporta BF16 → forziamo float16.
uv run python main.py \
    run.hf_home=/work/tesi_avalenza/hf \
    model.torch_dtype=float16
