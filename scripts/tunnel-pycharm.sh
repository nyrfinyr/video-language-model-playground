#!/bin/zsh

LOGIN_NODE="ailab02"
LOCAL_PORT=5678
DEBUGPY_PORT=5678

echo "=== PyCharm AiLab Tunnel ==="
echo "Assicurati di aver già lanciato 'srun' in un altro terminale e di sapere quale nodo ti è stato assegnato."

printf "Inserisci il nome del nodo assegnato (es. ailb-login-03, gpu01): "
read -r COMPUTE_NODE

if [ -z "$COMPUTE_NODE" ]; then
    echo "Errore: Devi inserire il nome di un nodo."
    exit 1
fi

echo ""
echo "Apertura del tunnel: localhost:${LOCAL_PORT} -> ${COMPUTE_NODE}:${DEBUGPY_PORT} (via ${LOGIN_NODE})..."
echo "Sul nodo GPU lancia: python -m debugpy --listen 0.0.0.0:${DEBUGPY_PORT} --wait-for-client main.py"
echo "In PyCharm: Run -> Attach to Process -> Remote -> localhost:${LOCAL_PORT}"
echo "Premi Ctrl+C per chiudere il tunnel."
echo "--------------------------------------------------------------------------------"

ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:${DEBUGPY_PORT} ${LOGIN_NODE}