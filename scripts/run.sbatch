#!/bin/bash
#SBATCH --job-name=egoschema-eval
#SBATCH --account=sai2026
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:06:00
#SBATCH --output=/homes/avalenza/video-language-model-playground/logs/%x-%j.out
#SBATCH --error=/homes/avalenza/video-language-model-playground/logs/%x-%j.err

set -euo pipefail

echo "node: $(hostname)"
echo "date: $(date -Iseconds)"
echo "job_id: ${SLURM_JOB_ID}"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

cd /homes/avalenza/video-language-model-playground

export HYDRA_FULL_ERROR=1

uv run python main.py \
    hf_home=/work/tesi_avalenza/hf \
    run.root=/work/tesi_avalenza/datasets/egoschema_subset_20 \
    model.torch_dtype=bfloat16 \
    wandb.tags=[eval,egoschema,sbatch] \
    wandb.name=qwen25vl3b-egoschema-${SLURM_JOB_ID} \
    wandb.notes="slurm ${SLURM_JOB_ID} on $(hostname)"
