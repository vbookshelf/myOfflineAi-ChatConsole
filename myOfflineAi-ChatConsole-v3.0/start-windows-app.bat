@echo off
echo ============================================
echo   Starting myOfflineAi-ChatConsole
echo ============================================

REM Ensure script runs from its own directory
cd /d %~dp0

REM --- Download Model Files if they are missing ---

REM --- Configuration: GitHub Release URLs ---
set "KOKORO_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
set "VOICES_URL=https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
set "KOKORO_FILE=kokoro-v1.0.onnx"
set "VOICES_FILE=voices-v1.0.bin"


REM --- Check for curl ---
where curl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'curl.exe' was not found. It is required to download model files.
    echo It is included by default in modern Windows 10/11.
    pause
    exit /b
)

REM --- Check and Download Kokoro Model ---
IF NOT EXIST "%KOKORO_FILE%" (
    echo [INFO] Kokoro model not found. Downloading from GitHub...
    curl -L -o "%KOKORO_FILE%" "%KOKORO_URL%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to download Kokoro model. Please check the URL and your connection.
        del "%KOKORO_FILE%" >nul 2>&1
        pause
        exit /b
    )
    echo [INFO] Download complete.
)

REM --- Check and Download Voices Model ---
IF NOT EXIST "%VOICES_FILE%" (
    echo [INFO] Voices file not found. Downloading from GitHub...
    curl -L -o "%VOICES_FILE%" "%VOICES_URL%"
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to download voices file. Please check the URL and your connection.
        del "%VOICES_FILE%" >nul 2>&1
        pause
        exit /b
    )
    echo [INFO] Download complete.
)


REM --- Check for Ollama ---
where ollama >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Ollama is not installed.
    echo Please install from: https://ollama.com/download
    pause
    exit /b
)

REM --- Check for uv ---
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'uv' is not installed.
    echo See: https://docs.astral.sh/uv/getting-started/installation
    pause
    exit /b
)

REM --- Create venv if missing ---
IF NOT EXIST .venv (
    echo [INFO] Creating Python 3.12 environment via uv...
    uv venv --python 3.12
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create Python 3.12 environment via uv.
        pause
        exit /b
    )
)

REM --- Activate venv ---
call .venv\Scripts\activate

REM --- Lock & Sync dependencies ---
echo [INFO] Updating lockfile...
uv lock
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv failed to update lockfile
    pause
    exit /b
)

echo [INFO] Syncing dependencies...
uv sync
if %ERRORLEVEL% neq 0 (
    echo [ERROR] uv failed to sync dependencies
    pause
    exit /b
)

REM --- Launch app ---
echo [INFO] Launching app...
python app.py

pause