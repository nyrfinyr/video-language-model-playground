#!/bin/bash
set -euo pipefail

PROJECT_DIR="/homes/avalenza/video-language-model-playground"
DEBUGPY_PORT=5678

cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== debugpy ==="
echo "Nodo: $(hostname)"
echo "In ascolto su 0.0.0.0:${DEBUGPY_PORT} — attendo PyCharm..."
echo "Sul Mac: bash scripts/tunnel-pycharm.sh -> inserisci $(hostname)"
echo "In PyCharm: Run -> Attach to Process / Attach to DAP -> localhost:${DEBUGPY_PORT}"
echo "---------------"

python -m debugpy --listen "0.0.0.0:${DEBUGPY_PORT}" --wait-for-client main.py
