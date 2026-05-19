#!/bin/bash
#SBATCH --job-name=egoschema-eval
#SBATCH --account=sai2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

set -euo pipefail

# Diagnostica nodo + GPU assegnata (utile per linkare i fail in wandb).
echo "node: $(hostname)"
echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

export HYDRA_FULL_ERROR=1

# wandb + weave: legge la API key da ~/.netrc (popolato con `wandb login`
# sul login node — l'NFS home è visibile dal compute node). Se preferisci
# passare via env esporta WANDB_API_KEY qui sopra.

uv run python main.py \
    hf_home=/work/tesi_avalenza/hf \
    run.root=/work/tesi_avalenza/datasets/egoschema_subset_20 \
    model.torch_dtype=bfloat16 \
    wandb.tags=[eval,egoschema,sbatch] \
    wandb.name=qwen25vl3b-egoschema-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
