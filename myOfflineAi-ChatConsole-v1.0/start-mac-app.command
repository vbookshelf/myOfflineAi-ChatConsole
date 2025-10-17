#!/bin/bash
echo "============================================"
echo "   Starting myOfflineAi-ChatConsole"
echo "============================================"

cd "$(dirname "$0")"

# --- Check for Ollama ---
if ! command -v ollama >/dev/null 2>&1; then
    echo "[ERROR] Ollama is not installed."
    echo "Please install from: https://ollama.com/download"
    exit 1
fi

# --- Check for uv ---
if ! command -v uv >/dev/null 2>&1; then
    echo "[ERROR] 'uv' is not installed."
    echo "See: https://docs.astral.sh/uv/getting-started/installation"
    exit 1
fi

# --- Create venv if missing ---
if [ ! -d ".venv" ]; then
    echo "[INFO] Creating Python 3.10 environment via uv..."
    uv venv --python 3.10 || { echo "[ERROR] uv failed to create venv"; exit 1; }
fi

# --- Activate venv ---
source .venv/bin/activate

# --- Lock & Sync dependencies ---
echo "[INFO] Updating lockfile..."
uv lock || { echo "[ERROR] uv failed to update lockfile"; exit 1; }

echo "[INFO] Syncing dependencies..."
uv sync || { echo "[ERROR] uv failed to sync dependencies"; exit 1; }

# --- Launch app ---
echo "[INFO] Launching app..."
python app.py
