#!/bin/bash
#SBATCH --partition=all_serial
#SBATCH --account=sai2026
#SBATCH --nodelist=ailb-login-02
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --job-name=vllm
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load py-torch/2.8.0-gcc-11.4.0-cuda-12.6.3

source .venv/bin/activate

python main.py