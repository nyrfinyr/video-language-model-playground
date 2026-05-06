#!/bin/zsh

LOGIN_NODE="ailab02"
LOCAL_PORT=2222

echo "=== PyCharm AiLab Tunnel ==="
echo "Assicurati di aver già lanciato 'srun' in un altro terminale e di sapere quale nodo ti è stato assegnato."

printf "Inserisci il nome del nodo assegnato (es. node05, gpu01): "
read COMPUTE_NODE

if [ -z "$COMPUTE_NODE" ]; then
    echo "Errore: Devi inserire il nome di un nodo."
    exit 1
fi

echo ""
echo "Apertura del tunnel sulla porta locale $LOCAL_PORT verso $COMPUTE_NODE (via $LOGIN_NODE)..."
echo "⚠️ Ti verranno chieste la passphrase della chiave e la password (come al solito)."
echo "👉 Premi Ctrl+C quando hai finito di lavorare su PyCharm per chiudere il tunnel."
echo "--------------------------------------------------------------------------------"

#ssh -N -L ${LOCAL_PORT}:${COMPUTE_NODE}:22 ${LOGIN_NODE}
ssh -N -L 2222:localhost:22 ailab02