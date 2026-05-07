#!/bin/bash

PROJECT_DIR="/work/tesi_avalenza/video-language-model-playground"
DEBUGPY_PORT=5678

module unload python/3.11.11-gcc-11.4.0
module load py-torch/2.8.0-gcc-11.4.0-cuda-12.6.3

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== debugpy ==="
echo "Nodo: $(hostname)"
echo "In ascolto su 0.0.0.0:${DEBUGPY_PORT} — attendo PyCharm..."
echo "Sul tuo Mac: ./tunnel-pycharm.sh -> inserisci $(hostname)"
echo "In PyCharm: Run -> Attach to Process -> Remote -> localhost:${DEBUGPY_PORT}"
echo "---------------"

python -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client main.py