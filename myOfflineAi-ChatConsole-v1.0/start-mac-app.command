#!/bin/bash
echo "============================================"
echo "   Starting myOfflineAi-ChatConsole"
echo "============================================"

# Ensure the script runs from its own directory
cd "$(dirname "$0")"

# --- [NEW] Download Model Files if they are missing ---

# --- Configuration: SET YOUR GITHUB RELEASE URLS HERE ---
KOKORO_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


# --- File Names ---
KOKORO_FILE="kokoro-v1.0.onnx"
VOICES_FILE="voices-v1.0.bin"

# --- Check for curl ---
if ! command -v curl >/dev/null 2>&1; then
    echo "[ERROR] 'curl' is not installed. It is required to download model files."
    echo "Please install curl using your system's package manager (e.g., 'sudo apt install curl')."
    exit 1
fi

# --- Check and Download Kokoro Model ---
if [ ! -f "$KOKORO_FILE" ]; then
    echo "[INFO] Kokoro model not found. Downloading from GitHub..."
    curl -L -o "$KOKORO_FILE" "$KOKORO_URL" || {
        echo "[ERROR] Failed to download Kokoro model. Please check the URL and your connection."
        rm -f "$KOKORO_FILE" # Clean up partial download
        exit 1
    }
    echo "[INFO] Download complete."
fi

# --- Check and Download Voices Model ---
if [ ! -f "$VOICES_FILE" ]; then
    echo "[INFO] Voices file not found. Downloading from GitHub..."
    curl -L -o "$VOICES_FILE" "$VOICES_URL" || {
        echo "[ERROR] Failed to download voices file. Please check the URL and your connection."
        rm -f "$VOICES_FILE" # Clean up partial download
        exit 1
    }
    echo "[INFO] Download complete."
fi

# --- [END NEW SECTION] ---


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
    echo "[INFO] Creating Python 3.12 environment via uv..."
    # Note: Updated this to 3.12 to match your pyproject.toml
    uv venv --python 3.12 || { echo "[ERROR] uv failed to create venv"; exit 1; }
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