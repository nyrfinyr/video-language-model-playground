#!/bin/zsh

LOGIN_NODE="ailab02"
LOCAL_PORT=5678
DEBUGPY_PORT=5678

echo "=== PyCharm AiLab Tunnel (DAP / debugpy) ==="
echo "Forward: localhost:${LOCAL_PORT} -> <compute>:${DEBUGPY_PORT} (via ${LOGIN_NODE})"
echo "Pre-requisiti:"
echo "  1. Hai già lanciato 'srun' e sai quale nodo ti è stato assegnato."
echo "  2. Sul nodo GPU: bash scripts/debug.sh (debugpy in --wait-for-client)."

printf "Nome del nodo assegnato (es. gpu01): "
read -r COMPUTE_NODE

if [ -z "$COMPUTE_NODE" ]; then
    echo "Errore: serve il nome di un nodo."
    exit 1
fi

echo ""
echo "Apertura tunnel..."
echo "In PyCharm: Run -> Attach to Process / Attach to DAP -> localhost:${LOCAL_PORT}"
echo "Premi Ctrl+C per chiudere."
echo "--------------------------------------------------------------------------------"

ssh -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L ${LOCAL_PORT}:${COMPUTE_NODE}:${DEBUGPY_PORT} \
    "${LOGIN_NODE}"
