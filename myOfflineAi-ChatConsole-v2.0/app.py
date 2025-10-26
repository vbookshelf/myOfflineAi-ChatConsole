#----------------------
# myOfflineAi-ChatConsole
# Creator: vbookshelf
# GitHub: https://github.com/vbookshelf/myOfflineAi-ChatConsole
# License: MIT
# Version: 2.0 (Hybrid network architecture - HTTP and Websockets)
#----------------------

from flask import Flask, render_template_string, request, jsonify, Response, session
from flask_socketio import SocketIO, disconnect
import json
import os
import sys
import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import re
import time
from datetime import datetime, timezone
import uuid
from werkzeug.utils import secure_filename

import requests
import subprocess
import shutil

import ollama
from ollama import chat as ollama_chat
from urllib.parse import urlparse

# --- Voice-Specific Imports ---
import whisper
import soundfile as sf
from kokoro_onnx import Kokoro


# --- Enable Flash Attention and Context Caching ---

os.environ['OLLAMA_FLASH_ATTENTION'] = '1'

# Enable context quantization (K/V Cache Quantization).
# This enables the model to use less RAM and operate faster
# when large context sizes are set.
os.environ['OLLAMA_KV_CACHE_TYPE'] = 'q8_0' # q4_0, q8_0, f16 (default)

# Check if flash attention is being used.
# If this value is 1 then flash attention has been set.
flash_attention_status = os.environ.get('OLLAMA_FLASH_ATTENTION')
context_quantization_status = os.environ.get('OLLAMA_KV_CACHE_TYPE')
print()
print(f"Flash Attention: {flash_attention_status}")
print(f"Context Quantization: {context_quantization_status}")
print()


# --- In-memory store for uploaded images ---
# This dictionary will hold Base64 encoded images, keyed by a unique ID.
# This avoids writing any uploaded files to the disk.
# Ensures data privacy through ephemeral data processing.
uploaded_images_store = {}


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'webp', 'heic', 'avif'}

# Default Model Parameters
# Notes: 
# 1. Not all models support all parameters.
# 2. Qwen3 models will fail if a repeat_penalty parameter is set.
# 3. To ensure robust operation, this app does not set top_k, frequency_penalty and repeat_penalty.

DEFAULT_NUM_CTX = 16000
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.95

# The name of the last model selected
# is stored in this file
LAST_MODEL_FILE = "last_model.txt"

# All agents, including the default, are stored in this file
AGENTS_FILE = "agents.json"

# File to store conversation histories
CONVERSATIONS_FILE = "conversations.json"

# File to store user settings for voice and model parameters
SETTINGS_FILE = "user_settings.json"

# --- Voice-Specific Configuration ---
WHISPER_MODEL = "base" # Models: base, tiny.en, etc.

# TTS settings
TTS_ENABLED = "On"
TTS_LANG = "en-us"
TTS_VOICE = "af_heart"
TTS_SPEED = 1.0

# Pdf upload settings
MAX_PAGES = 15
PDF_IMAGE_RES = 1.5
MAX_UPLOAD_FILE_SIZE = 20



# -----------------------------------------
# Settings Management Functions
# -----------------------------------------

def save_settings(settings):
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
    except IOError as e:
        print(f"[ERROR] Could not save settings: {e}", file=sys.stderr)

def load_settings():
    defaults = {
        "tts_enabled": TTS_ENABLED,
        "tts_lang": TTS_LANG,
        "tts_voice": TTS_VOICE,
        "tts_speed": TTS_SPEED,
        "num_ctx": DEFAULT_NUM_CTX,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_pages": MAX_PAGES,
        "pdf_image_res": PDF_IMAGE_RES,
        "max_upload_file_size": MAX_UPLOAD_FILE_SIZE,
    }
    if not os.path.exists(SETTINGS_FILE):
        return defaults
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
            # Ensure all keys from defaults are present in the loaded settings
            for key, value in defaults.items():
                settings.setdefault(key, value)
            return settings
    except (IOError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not read settings file, using defaults: {e}", file=sys.stderr)
        return defaults


# -----------------------------------------
# Flask App Initialization
# -----------------------------------------

app = Flask(__name__)

# Load settings once at startup to configure the app
# Note: MAX_CONTENT_LENGTH requires an app restart to change.
initial_settings = load_settings()

# --- Explicitly cast the setting to an integer to prevent TypeError ---
app.config['MAX_CONTENT_LENGTH'] = int(initial_settings.get("max_upload_file_size", 20)) * 1024 * 1024
app.config['SECRET_KEY'] = os.urandom(24) # Required for SocketIO

socketio = SocketIO(app, cors_allowed_origins="*")



# ----------------------------------------------
# Privacy Feature: Enforce Localhost Connection
# ----------------------------------------------

current_ollama_host = os.environ.get("OLLAMA_HOST", "").strip()
print(f"Current Ollama host: {current_ollama_host!r}")

def is_localhost_url(url):
    if not url: return True
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname
    port = parsed.port or 11434
    return hostname in ("127.0.0.1", "localhost") and port == 11434

if not is_localhost_url(current_ollama_host):
    print(f"[SECURITY] OLLAMA_HOST is not localhost: {current_ollama_host}. Aborting start.", file=sys.stderr)
    sys.exit(1)


# -----------------------------------------
# Voice Engine Initialization
# -----------------------------------------

# Check for Kokoro (TTS) model files
KOKORO_ONNX_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"
kokoro = None

if not os.path.exists(KOKORO_ONNX_FILE) or not os.path.exists(KOKORO_VOICES_FILE):
    print(f"[WARNING] Kokoro TTS model files not found. Voice generation will be disabled.", file=sys.stderr)
else:
    try:
        print("[INFO] Loading Kokoro text-to-speech engine...")
        kokoro = Kokoro(KOKORO_ONNX_FILE, KOKORO_VOICES_FILE)
        print("[INFO] Kokoro engine loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load Kokoro engine: {e}", file=sys.stderr)
        kokoro = None # Ensure it's disabled on failure

# Load Whisper (STT) model
whisper_model = None
try:
    print(f"[INFO] Loading Whisper STT model ({WHISPER_MODEL})...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    print("[INFO] Whisper model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load Whisper model: {e}. Voice input will be disabled.", file=sys.stderr)


# -----------------------------------------
# Model and Agent Management
# -----------------------------------------

def save_last_model(model_name):
    try:
        with open(LAST_MODEL_FILE, "w") as f: f.write(model_name)
    except IOError as e:
        print(f"[ERROR] Could not save last model: {e}", file=sys.stderr)

def load_last_model():
    if not os.path.exists(LAST_MODEL_FILE): return None
    try:
        with open(LAST_MODEL_FILE, "r") as f: return f.read().strip()
    except IOError as e:
        print(f"[ERROR] Could not read last model: {e}", file=sys.stderr)
        return None

def get_ollama_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        resp.raise_for_status()
        return sorted([m["name"] for m in resp.json().get("models", [])])
    except Exception:
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
            lines = result.stdout.strip().splitlines()
            return sorted([line.split()[0] for line in lines[1:]]) if len(lines) > 1 else []
        except Exception:
            return []

model_list = get_ollama_models()
if not model_list:
    model_list = ['gemma3:4b'] # Fallback default

last_model = load_last_model()
MODEL_NAME = last_model if last_model and last_model in model_list else model_list[0]
print(f"[INFO] Using model: {MODEL_NAME}")


# --- HTML Template ---

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ai Chat Console</title>
    
    
	<script src="{{ url_for('static', filename='tailwind.js') }}"></script>
	<script src="{{ url_for('static', filename='marked.min.js') }}"></script>
	<link rel="stylesheet" href="{{ url_for('static', filename='atom-one-dark.min.css') }}">
	<script src="{{ url_for('static', filename='highlight.min.js') }}"></script>
	<link rel="shortcut icon" type="image/png" href="static/icon.png">
	<script src="{{ url_for('static', filename='socket.io.min.js') }}"></script>


	<link rel="stylesheet" href="{{ url_for('static', filename='katex.min.css') }}">

<script src="{{ url_for('static', filename='katex.min.js') }}"></script>
<script src="{{ url_for('static', filename='katex-auto-render.min.js') }}"></script>


    <style>
        /* Main page styling */
        body {
            font-family: 'Arial', sans-serif;
            background-color: white;
        }

        /* Custom scrollbar styles for better UX */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #e2e8f0; /* slate-200 */
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: #94a3b8; /* slate-400 */
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b; /* slate-500 */
        }

        /* Sidebar animation for mobile toggle */
        #agent-sidebar {
            transition: transform 0.3s ease-in-out;
        }

        /* Loading/typing animation for when AI is responding */
        .typing-dot {
            animation: typing-bounce 1.2s infinite ease-in-out;
        }
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        @keyframes typing-bounce {
            0%, 80%, 100% {
                transform: scale(0);
            }
            40% {
                transform: scale(1.0);
            }
        }

        /* Styling for agent icons in the sidebar */
        .agent-icon {
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 2rem;
            font-weight: bold;
            border-radius: 0.75rem; /* rounded-xl */
            border-width: 4px;
            border-color: #1e293b; /* slate-800 */
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); /* shadow-sm */
            width: 4rem; /* w-16 */
            height: 4rem; /* h-16 */
        }

        /* Model selector dropdown styling */
        .model-selector {
            border: 1px solid #6b7280; /* Gray-500 */
        }

        /* Scrollbar styling for message content areas */
        .message-content::-webkit-scrollbar {
            width: 6px;
        }
        .message-content::-webkit-scrollbar-track {
            background: #e2e8f0;
            border-radius: 10px;
        }
        .message-content::-webkit-scrollbar-thumb {
            background: #94a3b8;
            border-radius: 10px;
        }
        .message-content::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }

        /* --- Styles for Markdown Content in AI responses --- */
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {
            font-weight: bold;
            margin-top: 1.25rem;
            margin-bottom: 0.75rem;
        }
        .markdown-content h1 { font-size: 1.5rem; }
        .markdown-content h2 { font-size: 1.25rem; }
        .markdown-content h3 { font-size: 1.125rem; }
        .markdown-content p {
			/*font-size: 17px;*/
            margin-bottom: 0.75rem;
            line-height: 1.8; /*1.6 */
        }
        .markdown-content ul, .markdown-content ol {
            list-style-position: outside;
            margin-left: 1.5rem;
            margin-bottom: 0.75rem;
            line-height: 1.8; /* 2.0 */
        }
		
		/* Adjust this value to increase/decrease space between bullet points */
		.markdown-content li {
		    margin-bottom: 1rem; 
		}
		
		
        .markdown-content a {
            color: #4f46e5;
            text-decoration: underline;
            font-weight: 500;
        }
        .markdown-content blockquote {
            border-left: 4px solid #cbd5e1; /* slate-300 */
            padding-left: 1rem;
            margin: 1rem 0.5rem;
            font-style: italic;
            color: #64748b; /* slate-500 */
        }
        .markdown-content code {
            font-family: monospace;
            font-size: 0.875rem;
            background-color: #e2e8f0;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
        }

        /* Code block wrapper styling with copy button */
        .code-block-wrapper {
            position: relative;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            background-color: #1a202c; /* Darker code block bg */
            border-radius: 0.5rem;
            overflow: hidden;

			 width: 100%; /* For a full width code area */
			 box-sizing: border-box;  /* For a full width code area */
        }


        .code-block-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #2d3748; /* Header bg */
            padding: 0.5rem 0.75rem;
            color: #e2e8f0;
            font-size: 0.8rem;
        }
        .copy-btn {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background-color: #4a5568;
            border: none;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            cursor: pointer;
            transition: background-color 0.2s;
            font-size: 0.8rem;
        }
        .copy-btn:hover {
            background-color: #718096;
        }

        /* Override default code block styles within our wrapper */
        .markdown-content .code-block-wrapper pre {
            background-color: transparent !important;
            margin: 0;
            padding: 0.75rem;
            border-radius: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .markdown-content .code-block-wrapper pre code,
        .markdown-content .code-block-wrapper pre code.hljs {
            background: transparent !important;
            padding: 0;
            color: #e2e8f0;
        }

        /* Styling for AI "thinking" bubble (reasoning display) */
		.thinking-bubble {
		    font-size: 0.85rem;
		    opacity: 0.9;
		    max-height: 200px;
		    overflow-y: auto;
		    white-space: pre-wrap;
		    transition: background-color 0.2s;
		}
		.thinking-bubble:hover {
		    background-color: #e2e8f0; /* slate-200 hover */
		}
		.hidden-reasoning {
		    font-family: monospace;
		    line-height: 1.4;
		    white-space: pre-wrap;
		}
        .edit-agent-btn {
            opacity: 0;
            transition: opacity 0.2s ease-in-out;
        }
        .agent-item:hover .edit-agent-btn {
            opacity: 1;
        }

		/* --- Styles for Markdown Tables --- */
        .markdown-content table {
            width: 100%;
            border-collapse: collapse; /* Ensures borders are clean and single */
            margin-top: 1.25rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1); /* Optional: adds a subtle shadow */
            border-radius: 0.5rem;
            overflow: hidden; /* Important for rounded corners */
        }
        .markdown-content th, .markdown-content td {
            padding: 0.75rem 1rem; /* Adds padding inside each cell */
            border: 1px solid #cbd5e1; /* slate-300 border */
            text-align: left;
            font-size: 0.95rem; /* Slightly smaller text for tables */
        }
        .markdown-content th {
            background-color: #f1f5f9; /* slate-100 background for the header */
            font-weight: 600; /* Bolder header text */
        }
        .markdown-content tr:nth-child(even) {
            background-color: #f8fafc; /* slate-50 for zebra-striping */
        }

        /* --- START: New styles for Chat History Panel --- */
        #chat-history-panel {
            transition: transform 0.3s ease-in-out;
        }
        .history-item:hover .delete-history-btn,
        .history-item:hover .edit-history-btn {
            opacity: 1;
        }
        /* --- END: New styles for Chat History Panel --- */

		/* Prevents the cursor from changing to the text-selection I-beam */
        .agent-item {
            user-select: none;
        }

        /* Overrides the default browser behavior to remove the hand cursor from agent buttons */
        .edit-agent-btn,
        .move-up-btn,
        .move-down-btn {
            cursor: default;
        }

        /* Styles for Webcam */
        #webcam-container video {
		    width: 100%;
		    height: auto;
		    min-height: 270px; 
		    /*max-height: 250px;*/
		    border-radius: 0.5rem;
		    background-color: #1e293b;
		}
		
		
        #webcam-canvas {
            display: none;
        }

        /* --- Voice Button Styles --- */
        .action-btn {
            flex-shrink: 0;
            width: 3rem;
            height: 3rem;
            border-radius: 0.75rem; /* rounded-xl */
            border: none;
            background-color: #4f46e5; /* indigo-600 */
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background-color 0.2s, box-shadow 0.2s;
        }
        .action-btn:hover {
             background-color: #4338ca; /* indigo-700 */
        }
        .action-btn.listening {
            animation: pulse-blue 1.5s infinite;
        }
        @keyframes pulse-blue {
            0% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(79, 70, 229, 0); }
            100% { box-shadow: 0 0 0 0 rgba(79, 70, 229, 0); }
        }
        .stop-audio-btn, .stop-btn {
            background-color: #4f46e5; /* red-600: #dc2626, indigo-600: #4f46e5 */
        }
        .stop-audio-btn:hover, .stop-btn:hover {
            background-color: #4338ca; /* red-700: #b91c1c, indigo-700: #4338ca */
        }
		
		/* --- Custom Slider Styles --- */
		input[type="range"] {
		    -webkit-appearance: none;
		    width: 100%;
		    background: transparent;
		}
		
		input[type="range"]::-webkit-slider-thumb {
		    -webkit-appearance: none;
		    height: 16px;
		    width: 16px;
		    border-radius: 50%;
		    background: #4f46e5; /* This is the Indigo color for the thumb */
		    cursor: pointer;
		    margin-top: -6px; /* Adjusts thumb position vertically */
		}
		
		input[type="range"]::-webkit-slider-runnable-track {
		    width: 100%;
		    height: 4px;
		    cursor: pointer;
		    background: #475569; /* This is the dark Slate color for the track */
		    border-radius: 5px;
		}

        /* --- Styles for disabled sidebar buttons --- */
        .sidebar-toggle-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            background-color: rgba(71, 85, 105, 0.25); /* slate-700 with alpha */
        }
        .sidebar-toggle-btn:disabled:hover {
             background-color: rgba(71, 85, 105, 0.25); /* No hover effect */
        }
        
        /* --- Styles for image upload previews --- */
        .upload-progress-overlay {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.75rem;
            font-weight: bold;
            border-radius: 0.5rem; /* match the image's rounded corners */
        }

        /* --- START: Styles for Collapsible Sidebar Feature --- */
        .panel-hidden #agent-sidebar {
            display: none;
        }
        .panel-hidden .main-flex-container {
            max-width: 100%;
            background-color: #1e293b; /* slate-800 */
        }
        .panel-hidden main {
            max-width: 48rem; /* max-w-3xl */
            margin-left: auto;
            margin-right: auto;
        }
        #toggle-panel-btn {
            background-color: rgba(226, 232, 240, 0.8); /* slate-200 with transparency */
        }
		
        .panel-hidden #toggle-panel-btn {
             /* This is the same light color as the default state */
             background-color: rgba(226, 232, 240, 0.8); /* slate-200 with transparency */
             
             /* Change the icon color to be dark so it's visible on the light background */
             color: #475569; /* slate-600 */
        }
        /* --- END: Styles for Collapsible Sidebar Feature --- */

    </style>

</head>
<body class="flex h-screen overflow-hidden text-slate-800">
    <audio id="audio-player" class="hidden"></audio>

    <div class="main-flex-container flex flex-1 overflow-hidden relative max-w-7xl mx-auto transition-all duration-300">

        <aside id="agent-sidebar" class="w-full md:w-80 lg:w-96 p-4 bg-slate-800 border-r border-slate-700 text-slate-100 overflow-y-auto flex-shrink-0 absolute md:relative h-full z-20 md:z-10 transform -translate-x-full md:translate-x-0">
            <div class="flex justify-between items-center mb-4">
				<button id="close-sidebar-btn" class="md:hidden text-slate-400 hover:text-white">
				    <span class="text-xl">&times;</span>
				</button>
            </div>

            <div class="mb-4">
                <label for="model-selector" id="model-selector-label" class="block text-sm font-medium text-slate-300 mb-2">Model & Settings: <span class="text-indigo-400 font-semibold text-lg">Ai Assistant</span></label>
                <select id="model-selector" class="bg-slate-800 model-selector w-full p-3 rounded-lg text-white focus:outline-none transition-all duration-200">
                    {% for model in model_list %}
                    <option value="{{ model }}" {% if model == current_model %}selected{% endif %} class="bg-slate-800 text-white">{{ model }}</option>
                    {% endfor %}
                </select>

				
                <div id="model-status" class="hidden mt-2 text-xs text-slate-400">
                    Current: <span id="current-model-display">{{ current_model }}</span>
                </div>
            </div>
			
			
			
			<div class="mb-4">
                <button id="toggle-params-btn" class="sidebar-toggle-btn w-full text-left py-2.5 px-4 bg-slate-700/50 text-slate-200 rounded-lg font-semibold hover:bg-slate-700 transition-colors duration-200 flex items-center justify-center">
                    <span>Model Settings</span>
                </button>
                <div id="params-container" class="hidden mt-2 p-3 bg-slate-900/50 rounded-lg space-y-4">
                    <div>
                        <label for="num-ctx-slider" class="block text-sm font-medium text-slate-300 mb-1">Context Size (Tokens)</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="num-ctx-slider" min="0" max="128000" step="2000" class="w-full">
                            <span id="num-ctx-value" class="text-xs font-semibold text-slate-300 w-14 text-center">16000</span>
                        </div>
                    </div>
                    <div>
                        <label for="temperature-slider" class="block text-sm font-medium text-slate-300 mb-1">Temperature</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="temperature-slider" min="0" max="2" step="0.05" class="w-full">
                            <span id="temperature-value" class="text-xs font-semibold text-slate-300 w-10 text-center">0.40</span>
                        </div>
                    </div>
                    
                    <div>
                        <label for="top-p-slider" class="block text-sm font-medium text-slate-300 mb-1">Top P</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="top-p-slider" min="0" max="1" step="0.01" class="w-full">
                            <span id="top-p-value" class="text-xs font-semibold text-slate-300 w-10 text-center">0.95</span>
                        </div>
                    </div>
					
                    <hr class="border-slate-700">
                    <div>
                        <label for="max-pages-slider" class="block text-sm font-medium text-slate-300 mb-1">Max PDF Pages</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="max-pages-slider" min="1" max="100" step="1" class="w-full">
                            <span id="max-pages-value" class="text-xs font-semibold text-slate-300 w-10 text-center">15</span>
                        </div>
                    </div>
                    <div>
                        <label for="pdf-res-slider" class="block text-sm font-medium text-slate-300 mb-1">PDF Image Resolution</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="pdf-res-slider" min="0.5" max="3" step="0.1" class="w-full">
                            <span id="pdf-res-value" class="text-xs font-semibold text-slate-300 w-10 text-center">1.5</span>
                        </div>
                    </div>
                    <div>
                        <label for="max-upload-slider" class="block text-sm font-medium text-slate-300 mb-1">* Max Upload Size (MB)</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="max-upload-slider" min="5" max="100" step="5" class="w-full">
                            <span id="max-upload-value" class="text-xs font-semibold text-slate-300 w-10 text-center">20</span>
                        </div>
                        <p class="text-xs text-slate-400 mt-1 text-center">* Requires app restart to apply setting</p>
                    </div>
                </div>
            </div>

            
			
			
            <div class="mb-10">
                <button id="toggle-voice-btn" class="sidebar-toggle-btn w-full text-left py-2.5 px-4 bg-slate-700/50 text-slate-200 rounded-lg font-semibold hover:bg-slate-700 transition-colors duration-200 flex items-center justify-center">
                    <span>Voice Settings</span>
                </button>
                <div id="voice-container" class="hidden mt-2 p-3 bg-slate-900/50 rounded-lg space-y-4">
                    <div>
                        <label for="tts-enabled-selector" class="block text-sm font-medium text-slate-300 mb-1">Voice Output</label>
                        <select id="tts-enabled-selector" class="bg-slate-800 model-selector w-full p-2 rounded-md text-white focus:outline-none">
                            <option value="On">On</option>
                            <option value="Off">Off</option>
                        </select>
                    </div>
                     <div>
                        <label for="language-selector" class="block text-sm font-medium text-slate-300 mb-1">Language</label>
                        <select id="language-selector" class="bg-slate-800 model-selector w-full p-2 rounded-md text-white focus:outline-none"></select>
                    </div>
                     <div>
                        <label for="voice-selector" class="block text-sm font-medium text-slate-300 mb-1">Voice</label>
                        <select id="voice-selector" class="bg-slate-800 model-selector w-full p-2 rounded-md text-white focus:outline-none"></select>
                    </div>
                    <div>
                        <label for="speed-slider" class="block text-sm font-medium text-slate-300 mb-1">Speech Speed</label>
                        <div class="flex items-center space-x-3">
                            <input type="range" id="speed-slider" min="0.5" max="1.5" step="0.1" value="1.0" class="w-full">
                            <span id="speed-value" class="text-xs font-semibold text-slate-300 w-10 text-center">1.0x</span>
                        </div>
                    </div>
                </div>
            </div>
			
			
     
			
			
			
			<div class="mb-4">
                <button id="toggle-webcam-btn" class="sidebar-toggle-btn w-full text-left py-2.5 px-4 bg-slate-700/50 text-slate-200 rounded-lg font-semibold hover:bg-slate-700 transition-colors duration-200 flex items-center justify-center">
                    <span>Webcam Photo</span>
                </button>
                <div id="webcam-container" class="hidden mt-2 p-3 bg-slate-900/50 rounded-lg space-y-2">
                    <video id="webcam-feed" autoplay playsinline></video>
                    <canvas id="webcam-canvas"></canvas>
                    <div class="flex space-x-2">
                        <button id="start-stop-webcam-btn" class="flex-1 py-2 px-3 bg-slate-800 text-white rounded-md text-sm font-semibold hover:bg-slate-700 transition-colors">Start Webcam</button>
                        <button id="capture-webcam-btn" class="flex-1 py-2 px-3 bg-slate-800 text-white rounded-md text-sm font-semibold hover:bg-slate-700 transition-colors disabled:bg-slate-800 disabled:cursor-not-allowed" disabled>Take Photo</button>
                    </div>
                </div>
            </div>
			
			
			
            <div class="mb-4">
                <button id="open-create-agent-modal-btn" class="w-full py-2.5 px-4 bg-slate-700/50 text-slate-200 rounded-lg font-semibold hover:bg-slate-700 transition-colors duration-200 flex items-center justify-center space-x-2">
                    <span>+</span>
                    <span>Create Ai Tool</span>
                </button>
            </div>

            <div id="agent-list" class="w-full space-y-3">
            </div>
        </aside>

        <main class="flex-1 flex flex-col bg-slate-100 relative">
            
            <!-- Panel Toggle Button -->
            <button id="toggle-panel-btn" class="hidden md:block absolute top-14 left-3 z-30 p-2 rounded-md hover:bg-slate-300 text-slate-600 transition-colors" title="Toggle Sidebar">
                <!-- Icon will be managed by JavaScript -->
            </button>

			<header class="md:hidden flex items-center justify-between p-3 bg-white border-b border-slate-200 shadow-sm">
			    <button id="open-sidebar-btn" class="text-slate-600 hover:text-indigo-600">
			        <span class="text-xl">&#9776;</span>
			    </button>
			    <h2 id="mobile-header-title" class="font-bold text-lg">myOfflineAi</h2>
			    <div class="w-8"></div>
			</header>

            <div id="tab-header" class="hidden flex w-full flex-shrink-0 p-1 pr-3 bg-white/80 backdrop-blur-sm border-b border-slate-200 shadow-sm items-center justify-between">
                <div id="tab-buttons-container" class="flex overflow-x-auto whitespace-nowrap">
                    </div>

                <button id="global-history-btn" class="ml-4 text-sm font-semibold text-slate-700 hover:text-slate-900 p-2 rounded-lg flex items-center gap-2 flex-shrink-0">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.414-1.415L11 9.586V6z" clip-rule="evenodd"></path>
                    </svg>
                    History
                </button>
            </div>

            <div id="tab-content" class="flex-1 flex flex-col p-2 sm:p-4 overflow-hidden">
                <div id="initial-message" class="flex-1 flex items-center justify-center text-center text-slate-500">
                    <div>
                        <span class="text-7xl text-slate-300 mb-2">myOfflineAi</span>
                        <p class="text-2xl font-semibold mb-2 mt-3">Chat Console</p>
						</div>
                </div>
            </div>
        </main>


    </div>

    <div id="error-modal" class="hidden fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4">
        <div class="bg-white rounded-lg p-6 w-full max-w-sm shadow-xl">
            <h2 class="text-2xl font-bold text-red-600 mb-4 flex items-center">
                <span class="mr-3">⚠️</span>Error
            </h2>
            <p id="error-message" class="text-slate-700 mb-6"></p>
            <div class="flex justify-end">
                <button id="close-error-modal-btn" class="py-2 px-5 bg-slate-200 text-slate-800 rounded-lg hover:bg-slate-300 transition-colors duration-200 font-semibold">Close</button>
            </div>
        </div>
    </div>

    <div id="agent-editor-modal" class="hidden fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50 p-4">
        <div class="bg-white rounded-lg p-6 w-full max-w-md shadow-xl transform transition-all duration-300 scale-95 opacity-0">
            <h2 id="agent-modal-title" class="text-2xl font-bold text-slate-800 mb-6">Create an Ai Tool</h2>

            <form id="agent-editor-form">
                <input type="hidden" id="agent-id">
                <div class="space-y-4">
                    <div>
                        <label for="agent-name" class="block text-sm font-medium text-slate-700 mb-1">Agent Name</label>
                        <input autocomplete="off" type="text" id="agent-name" required class="w-full p-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. Text Summarizer">
                    </div>

                    <div>
                        <label for="agent-title" class="block text-sm font-medium text-slate-700 mb-1">Agent Title</label>
                        <input autocomplete="off" type="text" id="agent-title" required class="w-full p-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. Creates a summary">
                    </div>

                    <div>
                        <label for="agent-type" class="block text-sm font-medium text-slate-700 mb-1">Conversation Type</label>
                        <select id="agent-type" required class="w-full p-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 bg-white">
                            <option value="multi-turn">Conversational</option>
                            <option value="single-turn">Non-conversational</option>
                        </select>
                    </div>

                    <div>
						<p class="text-xs text-slate-400 mt-1">
						 This information is saved in a local file. Do not include sensitive data. Deleting the Tool will permanently delete this data from the file.
						</p>
                        <label for="agent-persona" class="block text-sm font-medium text-slate-700 mb-1">Persona / System Instruction</label>
                        <textarea autocomplete="off" id="agent-persona" rows="4" required class="w-full p-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500" placeholder="e.g. You are an expert at summarizing text..."></textarea>
                    </div>
                </div>

                <div class="flex justify-between mt-6">
                    <div>
                        <button type="button" id="delete-agent-btn" class="hidden py-2 px-5 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold">Delete Agent</button>
                    </div>
                    <div class="flex space-x-3">
                        <button type="button" id="cancel-agent-editor-btn" class="py-2 px-5 bg-slate-200 text-slate-800 rounded-lg hover:bg-slate-300 font-semibold">Cancel</button>
                        <button type="submit" id="save-agent-btn" class="py-2 px-5 bg-indigo-500 text-white rounded-lg hover:bg-indigo-700 font-semibold">Create</button>
                    </div>
                </div>
            </form>
        </div>
    </div>


    <script type="module">

        // --- Agents list is now populated dynamically from the server ---
        let agents = [];
        let savedSettings = {{ saved_settings | tojson }};

        let currentModel = '{{ current_model }}';
        const agentListEl = document.getElementById('agent-list');
        const tabHeaderEl = document.getElementById('tab-header');
        const tabContentEl = document.getElementById('tab-content');
        const initialMessageEl = document.getElementById('initial-message');
        const errorModalEl = document.getElementById('error-modal');
        const errorMessageEl = document.getElementById('error-message');
        const agentSidebar = document.getElementById('agent-sidebar');
        const openSidebarBtn = document.getElementById('open-sidebar-btn');
        const closeSidebarBtn = document.getElementById('close-sidebar-btn');
        const closeErrorModalBtn = document.getElementById('close-error-modal-btn');
        const modelSelector = document.getElementById('model-selector');
        const currentModelDisplay = document.getElementById('current-model-display');

        const agentEditorModalEl = document.getElementById('agent-editor-modal');
        const agentEditorModalContent = agentEditorModalEl.querySelector('div');
        const openCreateAgentModalBtn = document.getElementById('open-create-agent-modal-btn');
        const cancelAgentEditorBtn = document.getElementById('cancel-agent-editor-btn');
        const agentEditorForm = document.getElementById('agent-editor-form');
        const deleteAgentBtn = document.getElementById('delete-agent-btn');

        // Webcam elements
        const toggleWebcamBtn = document.getElementById('toggle-webcam-btn');
        const webcamContainer = document.getElementById('webcam-container');
        const webcamFeed = document.getElementById('webcam-feed');
        const webcamCanvas = document.getElementById('webcam-canvas');
        const startStopWebcamBtn = document.getElementById('start-stop-webcam-btn');
        const captureWebcamBtn = document.getElementById('capture-webcam-btn');
        let webcamStream = null;

        // --- Voice Elements & State ---
        const toggleVoiceBtn = document.getElementById('toggle-voice-btn');
        const voiceContainer = document.getElementById('voice-container');
        const ttsEnabledSelector = document.getElementById('tts-enabled-selector');
        const languageSelector = document.getElementById('language-selector');
        const voiceSelector = document.getElementById('voice-selector');
        const speedSlider = document.getElementById('speed-slider');
        const speedValue = document.getElementById('speed-value');
        const audioPlayer = document.getElementById('audio-player');
        
        // --- Model Parameter Elements ---
        const toggleParamsBtn = document.getElementById('toggle-params-btn');
        const paramsContainer = document.getElementById('params-container');
        const numCtxSlider = document.getElementById('num-ctx-slider');
        const numCtxValue = document.getElementById('num-ctx-value');
        const temperatureSlider = document.getElementById('temperature-slider');
        const temperatureValue = document.getElementById('temperature-value');
        const topPSlider = document.getElementById('top-p-slider');
        const topPValue = document.getElementById('top-p-value');
        const maxPagesSlider = document.getElementById('max-pages-slider');
        const maxPagesValue = document.getElementById('max-pages-value');
        const pdfResSlider = document.getElementById('pdf-res-slider');
        const pdfResValue = document.getElementById('pdf-res-value');
        const maxUploadSlider = document.getElementById('max-upload-slider');
        const maxUploadValue = document.getElementById('max-upload-value');

        // --- START: Sidebar Toggle Elements ---
        const togglePanelBtn = document.getElementById('toggle-panel-btn');
        const bodyEl = document.body;
        const iconHide = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M11 19l-7-7 7-7m8 14l-7-7 7-7" /></svg>`;
        const iconShow = `<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><path stroke-linecap="round" stroke-linejoin="round" d="M13 5l7 7-7 7M5 5l7 7-7 7" /></svg>`;
        // --- END: Sidebar Toggle Elements ---

        let isRecording = false, wasManuallyStopped = false;
        let mediaRecorder, audioStream, audioContext, audioChunks = [], silenceTimer = null;

        const ttsVoices = {
            'en-us': { name: 'American English', voices: { 'af_heart': 'Female', 'am_michael': 'Male' } },
            'en-gb': { name: 'British English', voices: { 'bf_emma': 'Female 1', 'bm_george': 'Male 1', 'if_sara': 'Female 2 (Italian)', 'im_nicola': 'Male 2 (Italian)' } },
            'zh': { name: 'Mandarin Chinese', voices: { 'zf_xiaoni': 'Female', 'zm_yunyang': 'Male' } },
            'es': { name: 'Spanish', voices: { 'ef_dora': 'Female', 'em_alex': 'Male' } },
            'fr': { name: 'French', voices: { 'ff_siwis': 'Female' } },
            'it': { name: 'Italian', voices: { 'if_sara': 'Female', 'im_nicola': 'Male' } },
            'pt-br': { name: 'Brazilian Portuguese', voices: { 'pf_dora': 'Female', 'pm_alex': 'Male' } }
        };

        let activeChats = {};
        let currentAgentId = null;
        let isTyping = false;
        let savedHistories = {};
        
        // --- WebSocket and Audio Streaming State ---
        const socket = io();
        let audioQueue = [];
        let isAudioPlaying = false;
        let isPlaybackStopped = false;
        let currentAiMessageElement = null; // Tracks the element being updated with tokens

        function showError(message) {
            errorMessageEl.textContent = message;
            errorModalEl.classList.remove('hidden');
        }


        async function handleSettingsChange() {
            // If no agent is selected, find the default agent.
            const agent = currentAgentId
                ? agents.find(a => a.id === currentAgentId)
                : agents.find(a => a.isDefault);
        
            if (!agent) return; // Exit if no agent could be determined.

            // Gather all current settings from the UI
            const settings = {
                tts_enabled: ttsEnabledSelector.value,
                tts_lang: languageSelector.value,
                tts_voice: voiceSelector.value,
                tts_speed: speedSlider.value,
                num_ctx: numCtxSlider.value,
                temperature: temperatureSlider.value,
                top_p: topPSlider.value,
                max_pages: maxPagesSlider.value,
                pdf_image_res: pdfResSlider.value,
                max_upload_file_size: maxUploadSlider.value,
                model: modelSelector.value
            };
            
            // --- Save globally if it's the default agent OR if no agent was selected. ---
            if (agent.isDefault || !currentAgentId) {
                // For default agent, save globally
                try {
                    await fetch('/change_model', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ model: settings.model })
                    });
                    await fetch('/save_settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(settings)
                    });
                    savedSettings = {...savedSettings, ...settings};
                    currentModel = settings.model;

                } catch (error) {
                    console.error('Error saving default settings:', error);
                    showError('Could not save default settings.');
                }
            } else {
                // For custom agents, save to the agent's specific config
                try {
                    const response = await fetch(`/agents/${currentAgentId}/settings`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(settings)
                    });
                    if (response.ok) {
                        // Update the agent object in the local 'agents' array to keep state synced
                        Object.assign(agent, settings);
                    } else {
                         console.error('Failed to save agent settings');
                         showError('Could not save the settings for this AI Tool.');
                    }
                } catch (error) {
                    console.error('Error saving agent settings:', error);
                    showError('Network error while saving AI Tool settings.');
                }
            }
        }


        function renderAgents() {
            agentListEl.innerHTML = '';
            agents.forEach((agent, index) => {
                const agentItem = document.createElement('div');
                agentItem.className = `agent-item group flex items-center justify-between p-3 rounded-xl transition-all duration-200 bg-slate-900/40 hover:bg-slate-700/80`;
                agentItem.dataset.id = agent.id;

                // --- FIX: Create elements programmatically to prevent XSS ---
                const mainContent = document.createElement('div');
                mainContent.className = 'flex items-center space-x-4 overflow-hidden';

                const icon = document.createElement('div');
                icon.className = 'flex-shrink-0 agent-icon';
                icon.style.backgroundColor = agent.color;
                const initialSpan = document.createElement('span');
                initialSpan.textContent = agent.name.charAt(0);
                icon.appendChild(initialSpan);

                const textContainer = document.createElement('div');
                textContainer.className = 'overflow-hidden';
                const nameH3 = document.createElement('h3');
                nameH3.className = 'font-bold text-slate-50 text-lg truncate';
                nameH3.textContent = agent.name;
                const titleP = document.createElement('p');
                titleP.className = 'text-indigo-400 text-sm font-semibold truncate';
                titleP.textContent = agent.title;
                textContainer.appendChild(nameH3);
                textContainer.appendChild(titleP);
                
                mainContent.appendChild(icon);
                mainContent.appendChild(textContainer);

                // Buttons can still use innerHTML as they are not user-controlled
                const buttonsContainer = document.createElement('div');
                buttonsContainer.className = 'flex items-center';
                buttonsContainer.innerHTML = `
                    <div class="flex flex-col mr-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <button class="move-up-btn text-slate-400 hover:text-white rounded-md px-1 text-xs">▲</button>
                        <button class="move-down-btn text-slate-400 hover:text-white rounded-md px-1 text-xs">▼</button>
                    </div>
                    ${!agent.isDefault ? `
                    <button class="edit-agent-btn flex-shrink-0 text-slate-400 hover:text-white p-2 rounded-full">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                          <path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" />
                          <path fill-rule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clip-rule="evenodd" />
                        </svg>
                    </button>` 
                    : '<div class="w-9 h-9"></div>' /* Spacer */
                    }
                `;

                agentItem.appendChild(mainContent);
                agentItem.appendChild(buttonsContainer);
                // --- END FIX ---

                agentItem.onclick = () => {
                    openChatTab(agent);
                    if (window.innerWidth < 768) agentSidebar.classList.add('-translate-x-full');
                };

                // Logic for edit button (only for non-default agents)
                if (!agent.isDefault) {
                    agentItem.querySelector('.edit-agent-btn').onclick = (e) => {
                        e.stopPropagation();
                        openEditAgentModal(agent);
                    };
                }

                // Logic for move buttons (for ALL agents)
                const moveUpBtn = agentItem.querySelector('.move-up-btn');
                const moveDownBtn = agentItem.querySelector('.move-down-btn');

                if (index === 0) {
                    moveUpBtn.classList.add('invisible'); // Hide up arrow for the very first item
                }
                if (index === agents.length - 1) {
                    moveDownBtn.classList.add('invisible'); // Hide down arrow for the very last item
                }

                moveUpBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (index > 0) { // Can move up if not the first item
                        [agents[index], agents[index - 1]] = [agents[index - 1], agents[index]]; // Swap
                        renderAgents();
                        saveAgentOrder(agents.map(a => a.id));
                    }
                };

                moveDownBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (index < agents.length - 1) { // Can move down if not the last item
                        [agents[index], agents[index + 1]] = [agents[index + 1], agents[index]]; // Swap
                        renderAgents();
                        saveAgentOrder(agents.map(a => a.id));
                    }
                };

                agentListEl.appendChild(agentItem);
            });
        }

        async function saveAgentOrder(newOrder) {
            try {
                const response = await fetch('/agents/reorder', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ order: newOrder })
                });
                if (!response.ok) {
                    const error = await response.json();
                    showError(error.error || 'Failed to save new agent order.');
                }
            } catch (err) {
                showError('Network error while saving agent order.');
            }
        }


        function updateSidebarControlsState() {
            // --- Sidebar controls are now always enabled. ---
            toggleWebcamBtn.disabled = false;
            toggleVoiceBtn.disabled = false;
            toggleParamsBtn.disabled = false;
            // The logic to hide containers or stop the webcam when disabled is removed,
            // as they can now be opened anytime. Changes will affect the default agent if no chat is open.
        }

        function openChatTab(agent) {
            initialMessageEl.classList.add('hidden');
            tabHeaderEl.classList.remove('hidden');
			
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('text-slate-700', 'bg-slate-200'));
            document.querySelectorAll('.chat-view').forEach(view => view.classList.add('hidden'));
            
            let chatView = document.getElementById(`chat-view-${agent.id}`);
            let tabBtn = document.getElementById(`tab-btn-${agent.id}`);

            if (!chatView) {
                chatView = createChatView(agent);
                tabContentEl.appendChild(chatView);
                tabBtn = createTabButton(agent);
                document.getElementById('tab-buttons-container').appendChild(tabBtn);
                activeChats[agent.id] = {
                    history: [],
                    agent: agent,
                    showFullHistory: false,
                    chatId: 'new',
                    uploadQueue: [] // NEW: Each chat now has its own upload queue
                };
            }
            
            
            const modelSelectorLabel = document.getElementById('model-selector-label');
            if (modelSelectorLabel) {
                 modelSelectorLabel.innerHTML = `Model & Settings: <span class="text-indigo-400 font-semibold text-lg">${agent.name}</span>`;
            }
            

            if (agent.isDefault) {
                const defaultAgentSettings = { ...savedSettings, model: currentModel };
                applySettingsToUI(defaultAgentSettings);
            } else {
                const agentSettings = {
                    ...savedSettings, 
                    model: currentModel, 
                    ...agent 
                };
                applySettingsToUI(agentSettings);
            }

            tabBtn.classList.add('text-slate-700', 'bg-slate-200');
			
            chatView.classList.remove('hidden');
            currentAgentId = agent.id;
            renderChatHistory(agent.id);
            updateSidebarControlsState();
        }


        async function closeChatTab(agentId) {
            document.getElementById(`tab-btn-${agentId}`)?.remove();
            document.getElementById(`chat-view-${agentId}`)?.remove();
            delete activeChats[agentId];
            const remainingTabKeys = Object.keys(activeChats);
            if (remainingTabKeys.length > 0) {
                const lastAgentId = remainingTabKeys[remainingTabKeys.length - 1];
                openChatTab(activeChats[lastAgentId].agent);
            } else {
                tabHeaderEl.classList.add('hidden');
                initialMessageEl.classList.remove('hidden');
                currentAgentId = null;
                updateSidebarControlsState();

                
                const modelSelectorLabel = document.getElementById('model-selector-label');
                if (modelSelectorLabel) {
                     modelSelectorLabel.innerHTML = `Model & Settings: <span class="text-indigo-400 text-lg font-semibold">Ai Assistant</span>`;
                }
                // When closing the last tab, reset UI settings to the global defaults
                applySettingsToUI({ ...savedSettings, model: currentModel });
                
            }
        }


        function createTabButton(agent) {
            const btn = document.createElement('button');
            btn.id = `tab-btn-${agent.id}`;
            btn.className = `tab-btn flex-shrink-0 flex items-center px-4 py-2 rounded-lg text-sm font-medium mr-2 transition-colors duration-200 hover:bg-indigo-100`;
            
            // --- FIX: Create elements programmatically to prevent XSS ---
            const nameSpan = document.createElement('span');
            nameSpan.textContent = agent.name;
            const closeSpan = document.createElement('span');
            closeSpan.className = 'close-tab-btn ml-2 text-xs text-slate-400 hover:text-slate-800 p-1';
            closeSpan.dataset.agentId = agent.id;
            closeSpan.innerHTML = '&times;'; // Safe, not user-controlled
            btn.appendChild(nameSpan);
            btn.appendChild(closeSpan);
            // --- END FIX ---

            btn.onclick = (e) => {
                if (!e.target.classList.contains('close-tab-btn')) {
                    openChatTab(agent);
                }
            };
            return btn;
        }



		function createChatView(agent) {
		    const chatView = document.createElement('div');
		    chatView.id = `chat-view-${agent.id}`;
		    chatView.className = `chat-view flex flex-col flex-1 hidden overflow-hidden relative`;
		    chatView.innerHTML = `
                <div id="chat-history-panel-${agent.id}" class="absolute top-0 right-0 bottom-0 w-80 bg-slate-100 border-l border-slate-300 z-30 p-4 transform translate-x-full overflow-y-auto">
                    <div class="flex justify-between items-center mb-4">
                        <h3 class="font-bold text-lg text-slate-700">Chat History</h3>
                        <button class="close-history-panel-btn text-slate-500 hover:text-slate-800 text-2xl" data-agent-id="${agent.id}">&times;</button>
                    </div>
                    <div id="chat-history-list-${agent.id}" class="space-y-2"></div>
                </div>
                <div class="dropzone-overlay absolute inset-0 bg-slate-900/60 backdrop-blur-sm z-40 flex items-center justify-center opacity-0 pointer-events-none transition-opacity duration-300">
                    <div class="text-center text-white border-4 border-dashed border-white rounded-2xl p-8">
                        <svg class="mx-auto h-16 w-16" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
                          <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                        <p class="text-xl font-semibold mt-4">Drop Files Here</p>
                    </div>
                </div>
                <div class="flex-1 overflow-y-auto p-4" id="chat-messages-container-${agent.id}">
                    <div id="history-toggle-container-${agent.id}" class="text-center">
		                <button id="history-toggle-btn-${agent.id}" class="mx-auto text-slate-500 rounded-lg transition-colors text-sm font-medium hidden mb-4 hover:text-indigo-600">
		                    Show Full History
		                </button>
                    </div>
		            <div id="chat-messages-${agent.id}" class="space-y-6"></div>
		        </div>
		        <div class="p-4 pt-0">
		            <div id="loading-indicator-${agent.id}" class="hidden flex items-center space-x-2 text-sm text-slate-500 mb-2">
		                 <div class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></div>
		                 <div class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></div>
		                 <div class="typing-dot w-2 h-2 bg-slate-400 rounded-full"></div>
		                 <span id="loading-text-${agent.id}">${agent.name} is processing...</span>
		            </div>
		            <form class="chat-form flex flex-col" data-agent-id="${agent.id}">
		                <div id="image-preview-container-${agent.id}" class="mb-2 hidden flex flex-wrap gap-2"></div>
		                <div class="flex space-x-3">
		                    <input type="file" id="file-input-${agent.id}" class="hidden file-input" accept="image/*,.pdf" multiple>
                            <div class="relative group">
                                <button type="button" class="attach-file-btn flex-shrink-0 w-12 h-12 flex items-center justify-center bg-slate-200 text-slate-600 rounded-xl hover:bg-slate-300 transition-colors">
                                    <span style="font-size: 1.5rem;">+</span>
                                </button>
                                <div class="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-slate-800 text-white text-xs rounded-md whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                                    jpg, png<br>webp, pdf
                                </div>
                            </div>
							
		                    <textarea autocomplete="off" placeholder="Type or talk..." rows="1" class="chat-input flex-1 p-3 border border-slate-300 rounded-xl focus:outline-none focus:border-slate-300 resize-none"></textarea>
							
		                    <div class="flex space-x-2">
                                <button type="button" class="mic-btn action-btn" title="Start Listening">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg>
                                </button>
                                <button type="button" class="stop-btn action-btn hidden" title="Stop Generating">
		                           <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4h16v16H4z"/></svg>
		                        </button>
                                <button type="button" class="stop-audio-btn action-btn hidden" title="Stop Playback">
                                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M4 4h16v16H4z"></path></svg>
                                </button>
		                    </div>
		                </div>
		            </form>
		        </div>`;

		    const attachBtn = chatView.querySelector('.attach-file-btn');
		    const fileInput = chatView.querySelector('.file-input');
		    const textInput = chatView.querySelector('.chat-input');
		    const form = chatView.querySelector('.chat-form');
		    const historyToggleBtn = chatView.querySelector(`#history-toggle-btn-${agent.id}`);
            const closeHistoryPanelBtn = chatView.querySelector('.close-history-panel-btn');
            const historyPanel = chatView.querySelector(`#chat-history-panel-${agent.id}`);
            const dropzoneOverlay = chatView.querySelector('.dropzone-overlay');
            const micBtn = chatView.querySelector('.mic-btn');
            const stopAudioBtn = chatView.querySelector('.stop-audio-btn');
            const stopBtn = chatView.querySelector('.stop-btn');

            micBtn.onclick = () => toggleListening(agent.id);
            stopAudioBtn.onclick = stopAudioPlayback;
            stopBtn.onclick = () => socket.emit('stop_generation'); // Send stop signal
            closeHistoryPanelBtn.onclick = () => historyPanel.classList.add('translate-x-full');

		    historyToggleBtn.onclick = () => {
		        activeChats[agent.id].showFullHistory = !activeChats[agent.id].showFullHistory;
		        renderChatHistory(agent.id);
		    };

            let dragCounter = 0;

            chatView.addEventListener('dragenter', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragCounter++;
                if (dragCounter === 1) {
                    dropzoneOverlay.classList.remove('opacity-0', 'pointer-events-none');
                }
            });

            chatView.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragCounter--;
                if (dragCounter === 0) {
                    dropzoneOverlay.classList.add('opacity-0', 'pointer-events-none');
                }
            });

            chatView.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

            chatView.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                dragCounter = 0;
                dropzoneOverlay.classList.add('opacity-0', 'pointer-events-none');
                fileInput.files = e.dataTransfer.files;
                fileInput.dispatchEvent(new Event('change', { bubbles: true }));
            });

		    textInput.addEventListener('keydown', (e) => {
		        if (e.key === 'Enter' && !e.shiftKey) {
		            e.preventDefault();
		            form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
		        }
		    });

		    textInput.addEventListener('input', () => {
		        textInput.style.height = 'auto';
		        textInput.style.height = `${Math.min(textInput.scrollHeight, 200)}px`;
		    });

		    attachBtn.onclick = () => fileInput.click();

            // --- REWRITTEN FILE HANDLING ---
		    fileInput.addEventListener('change', (event) => {
                handleFileUploads(agent.id, event.target.files);
                fileInput.value = ''; // Clear the input after handling
            });

		    form.addEventListener('submit', handleFormSubmit);
		    return chatView;
		}

        // --- NEW: Centralized file upload handler with ALERT checks ---
        function handleFileUploads(agentId, files) {
            const chat = activeChats[agentId];
            if (!files || files.length === 0) return;

            const newUploads = Array.from(files).map(file => ({
                file: file,
                filename: file.name,
                preview: URL.createObjectURL(file),
                status: 'uploading',
                serverId: null,
                error: null
            }));

            chat.uploadQueue.push(...newUploads);
            updatePreviews(agentId);

            newUploads.forEach(uploadItem => {
                const formData = new FormData();
                const fileName = uploadItem.file.name || 'webcam.jpeg';
                formData.append('file', uploadItem.file, fileName);

                fetch('/upload_file', {
                    method: 'POST',
                    body: formData
                })
                .then(async response => { // Make this async to await json()
                    // --- Consolidated Error Handling ---
                    if (!response.ok) {
                         // Handle all non-OK responses (like 400, 413, 500)
                         const errorData = await response.json().catch(() => null);
                         let errorMessage;
                         if (errorData && errorData.error) {
                            // This handles JSON errors like 413 (File size) and 400 (PDF pages)
                            errorMessage = errorData.error;
                         } else {
                            // Fallback for non-JSON errors
                            errorMessage = `Server error: ${response.status}`;
                         }
                         
                         showError(`Upload Error: ${errorMessage}`);
                         uploadItem.status = 'error';
                         uploadItem.error = errorMessage;
                         updatePreviews(agentId);
                         return null; // Stop processing this item
                    }
                    
                    return response.json();
                })
                .then(data => {
                    if (data === null) return; // Skip if error handled above

                    // This block should now only run on success
                    if (data.error) {
                        // This is a safety check, but errors should be caught above
                        showError(`Upload Error: ${data.error}`);
                        uploadItem.status = 'error';
                        uploadItem.error = data.error;
                    } else {
                        uploadItem.status = 'complete';
                        uploadItem.serverId = data.file_id;
                    }
                    updatePreviews(agentId);
                })
                .catch(err => {
                    console.error("Upload fetch error:", err);
                    uploadItem.status = 'error';
                    // Show a generic alert for network or unexpected errors
                    const errorMessage = err.message.includes('Failed to fetch') ? 'Network error' : err.message;
                    showError(`Upload Failed: ${errorMessage}`);
                    updatePreviews(agentId);
                });
            });
        }
        
        // --- Update previews based on the upload queue ---
        function updatePreviews(agentId) {
            const chat = activeChats[agentId];
            const previewContainer = document.getElementById(`image-preview-container-${agentId}`);
            previewContainer.innerHTML = '';
            previewContainer.classList.toggle('hidden', chat.uploadQueue.length === 0);

            chat.uploadQueue.forEach((item, index) => {
                const wrapper = document.createElement('div');
                wrapper.className = 'relative';
                
                let content = '';
                // --- Show new PDF icon or image preview ---
                if (item.filename && item.filename.toLowerCase().endsWith('.pdf')) {
                    content = `
                        <div class="bg-slate-200 text-slate-800 rounded-lg flex items-center p-2 border border-slate-300 max-w-full">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-slate-600 flex-shrink-0 mr-2">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                            </svg>
                            <span class="text-sm font-medium whitespace-normal break-all pr-4" title="${item.filename}">${item.filename}</span>
                        </div>`;
                } else {
                     // Use item.preview which is the Object URL
                    content = `<img src="${item.preview}" class="h-24 w-24 rounded-lg object-cover border-2 border-slate-300">`;
                }


                if (item.status === 'uploading') {
                    content += `<div class="upload-progress-overlay"><span>...</span></div>`;
                } else if (item.status === 'error') {
                    // Make error overlay more visible
                    content += `<div class="upload-progress-overlay bg-red-600/80"><span>Failed</span></div>`;
                }

                wrapper.innerHTML = content + `<button type="button" data-index="${index}" class="remove-upload-btn absolute -top-2 -right-2 bg-red-500 text-white rounded-full h-6 w-6 flex items-center justify-center text-xs font-bold shadow-md hover:bg-red-600">&times;</button>`;
                previewContainer.appendChild(wrapper);
            });

            // Add event listeners for the new remove buttons
            previewContainer.querySelectorAll('.remove-upload-btn').forEach(btn => {
                btn.onclick = () => {
                    const index = parseInt(btn.dataset.index, 10);
                    const itemToRemove = chat.uploadQueue[index];
                    
                    // Revoke the object URL to free memory, especially important for large files/many uploads
                    if (itemToRemove && itemToRemove.preview) {
                         URL.revokeObjectURL(itemToRemove.preview);
                    }

                    // TODO: Optionally send a request to the server to delete the uploaded file if needed
                    // if (itemToRemove && itemToRemove.serverId) { /* fetch DELETE /delete_file/{itemToRemove.serverId} */ }

                    chat.uploadQueue.splice(index, 1);
                    updatePreviews(agentId); // Re-render previews after removal
                };
            });
        }


        function renderChatHistory(agentId) {
            const messagesEl = document.getElementById(`chat-messages-${agentId}`);
            const historyToggleContainer = document.getElementById(`history-toggle-container-${agentId}`);
            const historyToggleBtn = document.getElementById(`history-toggle-btn-${agentId}`);

            if (!messagesEl) return;
            messagesEl.innerHTML = '';
            const chat = activeChats[agentId];
            if (!chat) return;
            const { history, agent, showFullHistory } = chat;

            if (history.length === 0) {
                 const welcomeDiv = document.createElement('div');
                 welcomeDiv.className = 'text-center py-8';
                 
                 const agentIcon = document.createElement('div');
                 agentIcon.className = 'agent-icon mx-auto mb-4 border-4 border-white shadow-lg';
                 agentIcon.style.backgroundColor = agent.color;
                 const agentInitial = document.createElement('span');
                 agentInitial.textContent = agent.name.charAt(0);
                 agentIcon.appendChild(agentInitial);

                 const agentName = document.createElement('h2');
                 agentName.className = 'text-2xl font-bold';
                 agentName.textContent = agent.name;
                 
                 const agentTitle = document.createElement('p');
                 agentTitle.className = 'text-slate-500 mt-1';
                 agentTitle.textContent = agent.title;

                 welcomeDiv.appendChild(agentIcon);
                 welcomeDiv.appendChild(agentName);
                 welcomeDiv.appendChild(agentTitle);
                 messagesEl.appendChild(welcomeDiv);

                historyToggleBtn.classList.add('hidden');
            } else {
                if (history.length > 2) {
                    historyToggleBtn.classList.remove('hidden');
                    historyToggleBtn.textContent = showFullHistory ? 'Show Recent Only' : `Show Full History (${history.length} messages)`;
                } else {
                    historyToggleBtn.classList.add('hidden');
                }

                let messagesToShow;
                if (showFullHistory) {
                    messagesToShow = history;
                } else {
                    const lastMessage = history[history.length - 1];
                    if (history.length > 1 && lastMessage.role === 'assistant') {
                        messagesToShow = history.slice(-2);
                    } else {
                        messagesToShow = history.slice(-1);
                    }
                }

                messagesToShow.forEach(msg => renderMessage(agentId, msg));
                scrollToBottom(agentId);
            }
        }

		
		

		function renderMessage(agentId, msg) {
		    const messagesListEl = document.getElementById(`chat-messages-${agentId}`);
		    if (!messagesListEl) return;

		    const isUser = msg.role === 'user';
		    const part = msg.parts?.[0] || {};
		    const rawText = part.text || '';
		    const imagePreviews = part.image_previews || []; // Exist for live messages
            const imageFilenames = part.image_filenames || []; // Saved in history
		    const thinkingContent = part.thinking || '';

		    const msgEl = document.createElement('div');
		    msgEl.className = `flex items-start gap-3 ${isUser ? 'justify-end' : ''}`;

		    const contentContainer = document.createElement('div');
			contentContainer.className = isUser ? 'flex flex-col items-end' : 'w-full';

            if (isUser && imageFilenames.length > 0) {
                const attachmentsContainer = document.createElement('div');

                // --- Conditionally set layout based on whether it's a live preview or history ---
                if (imagePreviews && imagePreviews.length > 0) {
                    // LIVE: Has previews, so display images side-by-side (flex-wrap)
                    attachmentsContainer.className = 'flex flex-wrap-reverse gap-2 mb-2 justify-end';
                } else {
                    // HISTORY: No previews, so stack file icons vertically (flex-col)
                    attachmentsContainer.className = 'flex flex-col gap-2 mb-2 items-end';
                }
                
                const pdfIcon = `
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-slate-600 flex-shrink-0 mr-2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>`;
                const imageIcon = `
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-5 h-5 text-slate-600 flex-shrink-0 mr-2">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm16.5-1.5H3.75" />
                    </svg>`;

                imageFilenames.forEach((filename, index) => {
                    const fileWrapper = document.createElement('div');
                    const isPdf = filename.toLowerCase().endsWith('.pdf');

                    if (isPdf) {
                        // For PDFs, always show the icon and filename.
                        fileWrapper.className = 'max-w-xs bg-slate-200 text-indigo-700 rounded-lg flex items-center p-2 border border-slate-300 shadow-sm';
                        fileWrapper.innerHTML = `
                            ${pdfIcon}
                            <span class="text-sm font-medium whitespace-normal break-all" title="${filename}">${filename}</span>`;
                    } else if (imagePreviews && imagePreviews[index]) {
                        // This is a LIVE image message, show the actual image preview.
                        fileWrapper.innerHTML = `<img src="${imagePreviews[index]}" class="h-24 w-24 rounded-lg object-cover border-2 border-slate-200 shadow-sm">`;
                    } else {
                        // This is a HISTORICAL image message, show the icon.
                        fileWrapper.className = 'max-w-xs bg-slate-200 text-indigo-700 rounded-lg flex items-center p-2 border border-slate-300 shadow-sm';
                        fileWrapper.innerHTML = `
                            ${imageIcon}
                            <span class="text-sm font-medium whitespace-normal break-all" title="${filename}">${filename}</span>`;
                    }
                    
                    if (fileWrapper.innerHTML) {
                        attachmentsContainer.appendChild(fileWrapper);
                    }
                });

                if (attachmentsContainer.hasChildNodes()) {
                    contentContainer.appendChild(attachmentsContainer);
                }
            }


		    if (rawText.trim().length > 0 || !isUser) {
		        if (!isUser && thinkingContent) {
		            const thinkingEl = document.createElement("div");
		            thinkingEl.className = "thinking-bubble bg-slate-200 text-slate-600 italic rounded-xl p-2 mb-2 cursor-pointer";
		            thinkingEl.textContent = "View reasoning";

		            const hiddenPanel = document.createElement("div");
		            hiddenPanel.className = "hidden hidden-reasoning mt-2 text-xs bg-slate-100 p-2 rounded";
		            hiddenPanel.textContent = thinkingContent;
		            thinkingEl.appendChild(hiddenPanel);

		            thinkingEl.onclick = () => hiddenPanel.classList.toggle("hidden");
		            contentContainer.appendChild(thinkingEl);
		        }

		        const bubbleEl = document.createElement('div');
		        const bubbleClasses = isUser
		            ? 'text-[17px] bg-slate-200 text-slate-700 rounded-br-none shadow-sm'
		            : 'text-[17px] text-slate-700 rounded-bl-none border-slate-200';
		        // Add bottom margin only if there are no attachments above
		        const bottomMargin = (isUser && imageFilenames.length > 0) ? '' : 'mb-2';
		        bubbleEl.className = `message-bubble max-w-lg md:max-w-xl lg:max-w-3xl rounded-2xl p-4 ${bubbleClasses} ${bottomMargin}`;
                
                if (!isUser) {
                    bubbleEl.id = `ai-message-${agentId}`;
                }

		        if (isUser) {
		            const contentWrapper = document.createElement('div');
		            const textDiv = document.createElement('div');
		            textDiv.className = 'message-content';
		            textDiv.style.whiteSpace = 'pre-wrap';
		            textDiv.style.wordWrap = 'break-word';
		            textDiv.textContent = rawText;

		            textDiv.style.maxHeight = '200px';
		            textDiv.style.overflowY = 'auto';
		            textDiv.style.scrollbarWidth = 'thin';
		            textDiv.style.paddingRight = '8px';
		            contentWrapper.appendChild(textDiv);

		            const MAX_LENGTH = 400;
		            if (rawText.length > MAX_LENGTH) {
		                const scrollHint = document.createElement('div');
		                scrollHint.className = 'text-xs italic text-slate-500 mt-1 text-center';
		                scrollHint.textContent = '↑ scroll ↓';
		                contentWrapper.appendChild(scrollHint);
		            }
		            bubbleEl.appendChild(contentWrapper);

		        } else {
		            const markdownDiv = document.createElement('div');
		            markdownDiv.className = 'markdown-content';
		            markdownDiv.innerHTML = marked.parse(rawText);
		            enhanceCodeBlocks(markdownDiv);
		            bubbleEl.appendChild(markdownDiv);
		        }
		        contentContainer.appendChild(bubbleEl);
		    }

		    msgEl.appendChild(contentContainer);
		    messagesListEl.appendChild(msgEl);
		    return msgEl;
		}
		
		
		

        function enhanceCodeBlocks(element) {
		    element.querySelectorAll('pre > code').forEach(codeBlock => {
		        const preElement = codeBlock.parentElement;
		        if (preElement.parentElement.classList.contains('code-block-wrapper')) return;
		        const wrapper = document.createElement('div');
		        wrapper.className = 'code-block-wrapper';
		        const language = Array.from(codeBlock.classList).find(c => c.startsWith('language-'))?.replace('language-', '') || 'code';

		        wrapper.innerHTML = `
		            <div class="code-block-header">
		                <span class="font-sans">${language}</span>
		                <button class="copy-btn">Copy</button>
		            </div>`;

		        preElement.parentNode.insertBefore(wrapper, preElement);
		        wrapper.appendChild(preElement);

		        wrapper.querySelector('.copy-btn').addEventListener('click', () => {
		           navigator.clipboard.writeText(codeBlock.textContent).then(() => {
		               const button = wrapper.querySelector('.copy-btn');
		               button.textContent = 'Copied!';
		               setTimeout(() => { button.textContent = 'Copy'; }, 2000);
		           });
		        });
		        hljs.highlightElement(codeBlock);
		    });

		    if (window.renderMathInElement) {
		        renderMathInElement(element, {
		            delimiters: [
		                {left: '$$', right: '$$', display: true},
		                {left: '$', right: '$', display: false},
		                {left: '\\[', right: '\\]', display: true},
		                {left: '\\(', right: '\\)', display: false}
		            ],
		            throwOnError: false
		        });
		    }
		}


        function scrollToBottom(agentId) {
            const container = document.getElementById(`chat-messages-container-${agentId}`);
            if (container) container.scrollTop = container.scrollHeight;
        }

        function isUserAtBottom(container) {
            if (!container) return false;
            const tolerance = 25; // Set this to fix page jitter
            return container.scrollHeight - container.scrollTop - container.clientHeight < tolerance;
        }

        // --- REWRITTEN FOR WEBSOCKETS ---
        function sendChatMessageViaSocket(agentId, userMessage) {
            const chat = activeChats[agentId];
            const { history, agent } = chat;

            // Add an empty assistant message to the history locally, which will be filled by the stream
            const assistantMessage = { role: "assistant", parts: [{ text: "", thinking: "" }] };
            history.push(assistantMessage);
            const messageEl = renderMessage(agentId, assistantMessage);
            currentAiMessageElement = messageEl.querySelector('.markdown-content');

            if (!currentAiMessageElement) {
                console.error("Could not find the message element to stream to.");
                return;
            }

            const agentSettings = agent.isDefault ? { ...savedSettings, model: modelSelector.value } : { ...savedSettings, ...agent };

            const payload = {
                history: history.slice(0, -1), // Send history *before* the empty assistant message
                system_message: agent.persona,
                model: agentSettings.model,
                tts_enabled: agentSettings.tts_enabled,
                llm_options: {
                    num_ctx: agentSettings.num_ctx,
                    temperature: agentSettings.temperature,
                    top_p: agentSettings.top_p,
                },
                tts_settings: {
                    tts_lang: agentSettings.tts_lang,
                    tts_voice: agentSettings.tts_voice,
                    tts_speed: agentSettings.tts_speed,
                }
            };
            socket.emit('chat_message', payload);
        }

        async function handleFormSubmit(e) {
		    e.preventDefault();
		    const form = e.target;
		    const agentId = form.dataset.agentId;
		    const textInput = form.querySelector(".chat-input");
		    const stopBtn = form.querySelector(".stop-btn");
		    const loadingText = document.getElementById(`loading-text-${agentId}`);
		    const chat = activeChats[agentId];
            const { uploadQueue } = chat;

		    const messageText = textInput.value.trim();
            const hasPendingUploads = uploadQueue.some(item => item.status === 'uploading');
            const hasFailedUploads = uploadQueue.some(item => item.status === 'error'); // Check for failed uploads

            if (hasPendingUploads) {
                showError("Please wait for all files to finish uploading before sending a message.");
                return;
            }
            if (hasFailedUploads) { // Prevent sending if any upload failed
                 showError("Some file uploads failed. Please remove them before sending.");
                 return;
            }
            
		    if ((messageText === "" && uploadQueue.length === 0) || !agentId || isTyping) return;

		    textInput.value = "";
		    textInput.style.height = 'auto';

            // --- Capture image filenames and previews for history ---
            const imageIds = uploadQueue.map(item => item.serverId).filter(Boolean);
            const imagePreviews = uploadQueue.map(item => item.preview); // Get temporary previews
            const imageFilenames = uploadQueue.map(item => item.filename); // Get original filenames
            
		    if (chat.history.length === 0) document.getElementById(`chat-messages-${agentId}`).innerHTML = "";

		    const userMessage = {
                role: "user",
                parts: [{
                    text: messageText,
                    image_ids: imageIds, // IDs for backend processing
                    image_previews: imagePreviews, // For immediate display
                    image_filenames: imageFilenames // Saved in history for reconstruction
                }]
            };
		    chat.history.push(userMessage);
            chat.uploadQueue = []; // Clear the queue after sending
            
            // --- FIX: Ensure previews are cleared AFTER pushing to history ---
            updatePreviews(agentId); // This will now correctly hide the container

		    if (chat.agent.type === 'single-turn') {
		        chat.history = chat.history.slice(-1);
		    }

		    chat.showFullHistory = false;
		    renderChatHistory(agentId); // Render message with previews

		    isTyping = true;
		    setChatControlsEnabled(agentId, false);
            
            if(loadingText) {
                loadingText.textContent = `${chat.agent.name} is processing...`;
            }
            
            isPlaybackStopped = false;
            sendChatMessageViaSocket(agentId, userMessage);
		}



		function openAgentEditorModal() {
            agentEditorModalEl.classList.remove('hidden');
            setTimeout(() => agentEditorModalContent.classList.remove('scale-95', 'opacity-0'), 10);
        }

        function closeAgentEditorModal() {
             agentEditorModalContent.classList.add('scale-95', 'opacity-0');
             setTimeout(() => agentEditorModalEl.classList.add('hidden'), 300);
        }

		function openCreateAgentModal() {
            agentEditorForm.reset();
            document.getElementById('agent-id').value = '';
            document.getElementById('agent-modal-title').innerHTML = `Create an Ai Tool`;
            document.getElementById('save-agent-btn').textContent = 'Create';
            deleteAgentBtn.classList.add('hidden');
            openAgentEditorModal();
        }

        function openEditAgentModal(agent) {
            agentEditorForm.reset();
            document.getElementById('agent-id').value = agent.id;
            document.getElementById('agent-name').value = agent.name;
            document.getElementById('agent-title').value = agent.title;
            document.getElementById('agent-persona').value = agent.persona;
            document.getElementById('agent-type').value = agent.type;
            document.getElementById('agent-modal-title').innerHTML = `Edit Ai Tool`;
            document.getElementById('save-agent-btn').textContent = 'Save Changes';
            
            if (agent.isDefault) {
                deleteAgentBtn.classList.add('hidden');
            } else {
                deleteAgentBtn.classList.remove('hidden');
            }
            openAgentEditorModal();
        }

        async function handleSaveAgent(e) {
            e.preventDefault();
            const agentId = document.getElementById('agent-id').value;
            const name = document.getElementById('agent-name').value.trim();
            const title = document.getElementById('agent-title').value.trim();
            const persona = document.getElementById('agent-persona').value.trim();
            const type = document.getElementById('agent-type').value;

            if (!name || !title || !persona) return showError("Please fill out all fields.");

            const agentData = {
                name,
                title,
                persona,
                type,
                color: '#4f46e5'
            };

            let url = "/agents";
            let method = "POST";

            if (agentId) {
                url = `/agents/${agentId}`;
                method = "PUT";
                agentData.id = agentId;
            } else {
                agentData.id = name.toLowerCase().replace(/\s+/g, '-') + '-' + Date.now();
            }

            try {
                const res = await fetch(url, {
                    method: method,
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(agentData)
                });
                const savedAgent = await res.json();
                if (!res.ok) {
                    throw new Error(savedAgent.error || `Failed to ${agentId ? 'update' : 'create'} agent`);
                }
                
                if (agentId) {
                    const index = agents.findIndex(a => a.id === agentId);
                    if (index !== -1) agents[index] = { ...agents[index], ...savedAgent };
                } else {
                    agents.unshift(savedAgent);
                }

                renderAgents();
                closeAgentEditorModal();

            } catch (err) {
                showError(err.message);
            }
        }

		
		
        async function handleDeleteAgent() {
            const agentId = document.getElementById('agent-id').value;
            if (!agentId) return;

            // --- ADD THIS CONFIRMATION BLOCK ---
            if (confirm('Are you sure you want to permanently delete this Ai Tool? This action cannot be undone.')) {
                try {
                    const res = await fetch(`/agents/${agentId}`, { method: "DELETE" });
                    if (!res.ok) {
                        const error = await res.json();
                        throw new Error(error.error || "Failed to delete agent");
                    }

                    agents = agents.filter(a => a.id !== agentId);
                    delete savedHistories[agentId];
                    closeChatTab(agentId);
                    renderAgents();
                    closeAgentEditorModal();
                } catch (err) {
                    showError(err.message);
                }
            }
        }

		
		
		
        async function saveOrUpdateCurrentChat(agentId) {
            const chat = activeChats[agentId];
            if (!chat || !chat.history || chat.history.length === 0) return;

            // --- Create a deep copy and clean temporary keys before saving ---
            const historyToSave = JSON.parse(JSON.stringify(chat.history));
            historyToSave.forEach(msg => {
                if (msg.role === 'user' && msg.parts?.[0]) {
                    delete msg.parts[0].image_previews; // Remove temporary preview data BEFORE saving
                }
            });

            if (chat.chatId === 'new') {
                const firstUserMessage = historyToSave.find(m => m.role === 'user');
                // Ensure text exists before trying to substring
                const titleText = firstUserMessage?.parts?.[0]?.text || '';
                const title = titleText ? titleText.substring(0, 50) + (titleText.length > 50 ? '...' : '') : 'Chat with Attachments';


                const newChatSession = {
                    id: `chat-${Date.now()}`,
                    timestamp: new Date().toISOString(),
                    title: title,
                    history: historyToSave // Use the cleaned history
                };

                try {
                    const res = await fetch(`/conversations/${agentId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(newChatSession)
                    });
                    if (res.ok) {
                        if (!savedHistories[agentId]) {
                            savedHistories[agentId] = [];
                        }
                        savedHistories[agentId].unshift(newChatSession);
                        activeChats[agentId].chatId = newChatSession.id;
                    } else {
                        console.error('Failed to save new chat session.');
                    }
                } catch (err) {
                    console.error('Error saving new chat session:', err);
                }
            }
            else {
                try {
                    const res = await fetch(`/conversations/${agentId}/${chat.chatId}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ history: historyToSave }) // Use the cleaned history
                    });

                    if (res.ok) {
                        const chatIndex = (savedHistories[agentId] || []).findIndex(c => c.id === chat.chatId);
                        if (chatIndex !== -1) {
                            const updatedChat = savedHistories[agentId][chatIndex];
                            updatedChat.history = historyToSave;
                            updatedChat.timestamp = new Date().toISOString();
                            savedHistories[agentId].splice(chatIndex, 1);
                            savedHistories[agentId].unshift(updatedChat);
                        }
                    } else {
                        console.error('Failed to update chat session.');
                    }
                } catch (err) {
                    console.error('Error updating chat session:', err);
                }
            }
        }
		
		
		

        function renderSavedChatsList(agentId) {
            const listEl = document.getElementById(`chat-history-list-${agentId}`);
            listEl.innerHTML = '';
            const chats = savedHistories[agentId] || [];

            if (chats.length === 0) {
                listEl.innerHTML = `<p class="text-sm text-slate-500 italic">No saved chats for this agent.</p>`;
                return;
            }

            chats.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

            chats.forEach(chat => {
                const itemEl = document.createElement('div');
                itemEl.className = 'history-item p-3 bg-white rounded-lg cursor-pointer hover:bg-indigo-50 border border-slate-200';
                
                // --- FIX: Build structure programmatically to set title with .textContent ---
                const mainFlex = document.createElement('div');
                mainFlex.className = 'flex justify-between items-start';
                
                const textContentDiv = document.createElement('div');
                textContentDiv.className = 'flex-grow overflow-hidden';
                
                const titleContainer = document.createElement('div');
                titleContainer.id = `title-container-${chat.id}`;
                const titleP = document.createElement('p');
                titleP.className = 'font-semibold text-slate-800 truncate';
                titleP.textContent = chat.title;
                titleContainer.appendChild(titleP);
                
                const timestampP = document.createElement('p');
                timestampP.className = 'text-xs text-slate-500';
                timestampP.textContent = new Date(chat.timestamp).toLocaleString();
                
                textContentDiv.appendChild(titleContainer);
                textContentDiv.appendChild(timestampP);
                
                const buttonsDiv = document.createElement('div');
                buttonsDiv.className = 'flex items-center flex-shrink-0';
                buttonsDiv.innerHTML = `
                    <button class="edit-history-btn text-slate-500 hover:text-indigo-700 p-1 opacity-0 transition-opacity" title="Edit title">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path d="M17.414 2.586a2 2 0 00-2.828 0L7 10.172V13h2.828l7.586-7.586a2 2 0 000-2.828z" /><path fill-rule="evenodd" d="M2 6a2 2 0 012-2h4a1 1 0 010 2H4v10h10v-4a1 1 0 112 0v4a2 2 0 01-2 2H4a2 2 0 01-2-2V6z" clip-rule="evenodd" /></svg>
                    </button>
                    <button class="delete-history-btn text-red-500 hover:text-red-700 p-1 opacity-0 transition-opacity" title="Delete chat">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M9 2a1 1 0 00-.894.553L7.382 4H4a1 1 0 000 2v10a2 2 0 002 2h8a2 2 0 002-2V6a1 1 0 100-2h-3.382l-.724-1.447A1 1 0 0011 2H9zM7 8a1 1 0 012 0v6a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v6a1 1 0 102 0V8a1 1 0 00-1-1z" clip-rule="evenodd" /></svg>
                    </button>
                `;
                
                mainFlex.appendChild(textContentDiv);
                mainFlex.appendChild(buttonsDiv);
                itemEl.appendChild(mainFlex);
                // --- END FIX ---
                
                itemEl.onclick = (e) => {
                    if (e.target.closest('button') || e.target.closest('input')) {
                        return;
                    }
                    loadChatHistory(agentId, chat.id);
                };
                itemEl.querySelector('.edit-history-btn').onclick = (e) => {
                    e.stopPropagation();
                    enterEditMode(agentId, chat.id);
                };
                itemEl.querySelector('.delete-history-btn').onclick = (e) => {
                    e.stopPropagation();
                    if (confirm('Are you sure you want to delete this chat history forever?')) {
                        deleteChatHistory(agentId, chat.id);
                    }
                };
                listEl.appendChild(itemEl);
            });
        }
        
		function loadChatHistory(agentId, chatId) {
            const chats = savedHistories[agentId] || [];
            const chatToLoad = chats.find(c => c.id === chatId);

            if (chatToLoad) {
                 // --- FIX: Ensure deep copy when loading history ---
                 activeChats[agentId].history = JSON.parse(JSON.stringify(chatToLoad.history));

                activeChats[agentId].chatId = chatToLoad.id;
                activeChats[agentId].showFullHistory = true; // Show full history when loading
                renderChatHistory(agentId); // Re-render with loaded history

                const historyPanel = document.getElementById(`chat-history-panel-${agentId}`);
                if(historyPanel) historyPanel.classList.add('translate-x-full'); // Close panel after loading
            }
        }
		
        function enterEditMode(agentId, chatId) {
            const titleContainer = document.getElementById(`title-container-${chatId}`);
            if (!titleContainer || titleContainer.querySelector('input')) return;
            
            const titleEl = titleContainer.querySelector('p');
            const originalTitle = titleEl.textContent;

            const inputEl = document.createElement('input');
            inputEl.type = 'text';
            inputEl.value = originalTitle;
            inputEl.className = 'w-full p-0 font-semibold text-slate-800 bg-transparent border-b-2 border-slate-400 focus:outline-none focus:border-indigo-500';
            
            // --- FIX: Function to safely render the title paragraph ---
            const renderTitleText = (container, text) => {
                const p = document.createElement('p');
                p.className = 'font-semibold text-slate-800 truncate';
                p.textContent = text;
                container.innerHTML = '';
                container.appendChild(p);
            };

            const saveChanges = async () => {
                const newTitle = inputEl.value.trim();
                
                // Detach listeners to prevent multiple triggers
                inputEl.removeEventListener('blur', saveChanges);
                inputEl.removeEventListener('keydown', handleKeyDown);

                if (!newTitle || newTitle === originalTitle) {
                    renderTitleText(titleContainer, originalTitle);
                    return;
                }
                try {
                    await updateChatTitle(agentId, chatId, newTitle);
                    renderTitleText(titleContainer, newTitle);
                } catch (err) {
                    showError(err.message || "Failed to update title.");
                    renderTitleText(titleContainer, originalTitle);
                }
            };
            
            const handleKeyDown = (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    saveChanges();
                } else if (e.key === 'Escape') {
                    inputEl.removeEventListener('blur', saveChanges);
                    inputEl.removeEventListener('keydown', handleKeyDown);
                    renderTitleText(titleContainer, originalTitle);
                }
            };

            inputEl.addEventListener('blur', saveChanges);
            inputEl.addEventListener('keydown', handleKeyDown);

            titleContainer.innerHTML = '';
            titleContainer.appendChild(inputEl);
            inputEl.focus();
            inputEl.select();
        }

        async function updateChatTitle(agentId, chatId, newTitle) {
            const res = await fetch(`/conversations/${agentId}/${chatId}/title`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newTitle })
            });
            const result = await res.json();
            if (!res.ok) {
                throw new Error(result.error || 'Server error updating title.');
            }
            const chatIndex = (savedHistories[agentId] || []).findIndex(c => c.id === chatId);
            if (chatIndex !== -1) {
                savedHistories[agentId][chatIndex].title = result.newTitle;
            }
            return result;
        }
		
        async function deleteChatHistory(agentId, chatId) {
            try {
                const res = await fetch(`/conversations/${agentId}/${chatId}`, { method: 'DELETE' });
                if (res.ok) {
                    savedHistories[agentId] = savedHistories[agentId].filter(c => c.id !== chatId);
                    renderSavedChatsList(agentId);

                    if (activeChats[agentId] && activeChats[agentId].chatId === chatId) {
                        activeChats[agentId].history = [];
                        activeChats[agentId].chatId = 'new';
                        renderChatHistory(agentId);
                    }
                } else {
                    showError('Failed to delete chat history.');
                }
            } catch (err) {
                showError('Error deleting chat history.');
            }
        }
        
        async function startWebcam() {
            if (webcamStream) { // If stream exists, it's a stop request
                stopWebcam();
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamStream = stream;
                webcamFeed.srcObject = stream;
                startStopWebcamBtn.textContent = 'Stop Webcam';
                captureWebcamBtn.disabled = false;
            } catch (err) {
                console.error("Error accessing webcam:", err);
                showError("Could not access the webcam. Please ensure it is not in use by another application and that you have granted permission.");
            }
        }

        function stopWebcam() {
            if (webcamStream) {
                webcamStream.getTracks().forEach(track => track.stop());
                webcamStream = null;
                webcamFeed.srcObject = null;
                startStopWebcamBtn.textContent = 'Start Webcam';
                captureWebcamBtn.disabled = true;
            }
        }

        function captureWebcamImage() {
            // --- If no chat is open, open the default agent's chat first. ---
            let agentIdToUse = currentAgentId;
            if (!agentIdToUse) {
                const defaultAgent = agents.find(a => a.isDefault);
                if (defaultAgent) {
                    openChatTab(defaultAgent);
                    agentIdToUse = defaultAgent.id;
                } else {
                    showError("Default AI tool not found. Cannot start a new chat for the photo.");
                    return;
                }
            }
        
            if (!webcamStream) {
                showError("Please start the webcam first.");
                return;
            }
        
            const context = webcamCanvas.getContext('2d');
            webcamCanvas.width = webcamFeed.videoWidth;
            webcamCanvas.height = webcamFeed.videoHeight;
            context.drawImage(webcamFeed, 0, 0, webcamCanvas.width, webcamCanvas.height);
            
            webcamCanvas.toBlob(blob => {
                 const webcamFile = new File([blob], "webcam.jpeg", { type: "image/jpeg" });
                 // Use the determined agentIdToUse
                 handleFileUploads(agentIdToUse, [webcamFile]);
            }, 'image/jpeg');
        }

        // --- Voice Functions ---
        function setChatControlsEnabled(agentId, isEnabled, options = {}) {
            const chatView = document.getElementById(`chat-view-${agentId}`);
            if (!chatView) return;

            const { keepMicActive = false } = options;
            const textInput = chatView.querySelector('.chat-input');
            const micBtn = chatView.querySelector('.mic-btn');
            const attachBtn = chatView.querySelector('.attach-file-btn');
            const loadingIndicator = document.getElementById(`loading-indicator-${agentId}`);
            const stopBtn = chatView.querySelector('.stop-btn');

            textInput.disabled = !isEnabled;
            attachBtn.disabled = !isEnabled;
            micBtn.disabled = !isEnabled && !keepMicActive;

            loadingIndicator.classList.toggle('hidden', isEnabled);
            if (isEnabled) {
                micBtn.classList.remove('hidden');
                stopBtn.classList.add('hidden');
            } else if (!keepMicActive) {
                micBtn.classList.add('hidden');
                stopBtn.classList.remove('hidden');
            }
        }

        function onAiSpeechEnd() {
            isAudioPlaying = false;
            if (currentAgentId) {
                const chatView = document.getElementById(`chat-view-${currentAgentId}`);
                if (chatView) {
                    chatView.querySelector('.stop-audio-btn').classList.add('hidden');
                    const micBtn = chatView.querySelector('.mic-btn:not(.stop-audio-btn)');
                    micBtn.classList.remove('hidden');
                    setChatControlsEnabled(currentAgentId, true);
                    
                    const textInput = chatView.querySelector('.chat-input');
                    if (textInput) {
                        textInput.focus();
                    }

                    if (micBtn.classList.contains('listening')) {
                        startRecording(currentAgentId);
                    }
                }
            }
        }

        function stopAudioPlayback() {
            isPlaybackStopped = true;
            socket.emit('stop_generation');
            audioPlayer.pause();
            audioPlayer.currentTime = 0;
            audioQueue = []; // Clear the queue to prevent further playback
            isTyping = false;
            onAiSpeechEnd();
        }
        
        function playNextInQueue() {
            if (isAudioPlaying || audioQueue.length === 0 || isPlaybackStopped) return;
            isAudioPlaying = true;
            
            const chatView = document.getElementById(`chat-view-${currentAgentId}`);
            if(chatView) {
                chatView.querySelector('.mic-btn:not(.stop-audio-btn)').classList.add('hidden');
                chatView.querySelector('.stop-btn').classList.add('hidden');
                chatView.querySelector('.stop-audio-btn').classList.remove('hidden');
            }

            const audioBase64 = audioQueue.shift();
            audioPlayer.src = `data:audio/wav;base64,${audioBase64}`;
            audioPlayer.play().catch(e => {
                console.error("Audio playback error:", e);
                isAudioPlaying = false;
            });
        }


        function toggleListening(agentId) {
            const chatView = document.getElementById(`chat-view-${agentId}`);
            if (!chatView) return;
            const micBtn = chatView.querySelector('.mic-btn:not(.stop-audio-btn)');

            if (isAudioPlaying || isTyping) return;
            const isNowListening = micBtn.classList.toggle('listening');
            micBtn.title = isNowListening ? "Stop Listening" : "Start Listening";
            if (isNowListening) {
                startRecording(agentId);
            } else {
                stopRecording(true);
            }
        }

        async function startRecording(agentId) {
            if (isRecording) return;
            setChatControlsEnabled(agentId, false, { keepMicActive: true });
            
            const loadingIndicator = document.getElementById(`loading-indicator-${agentId}`);
            const loadingText = document.getElementById(`loading-text-${agentId}`);
            if(loadingText) {
                loadingText.textContent = "Listening...";
                loadingIndicator.classList.remove("hidden");
            }

            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                isRecording = true;
                wasManuallyStopped = false;
                mediaRecorder = new MediaRecorder(audioStream);
                audioChunks = [];
                mediaRecorder.start();
                mediaRecorder.addEventListener("dataavailable", e => audioChunks.push(e.data));
                mediaRecorder.addEventListener("stop", () => onRecordingStop(agentId));
                audioContext = new AudioContext();
                const source = audioContext.createMediaStreamSource(audioStream);
                const analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                const dataArray = new Uint8Array(analyser.frequencyBinCount);
                source.connect(analyser);
                detectSilence(analyser, dataArray);
            } catch (err) {
                showError("Could not access microphone. Please check permissions.");
                setChatControlsEnabled(agentId, true);
            }
        }

        function stopRecording(isManualStop = false) {
            if (!isRecording) return;
            if (isManualStop) { wasManuallyStopped = true; }
            isRecording = false;
            mediaRecorder?.stop();
            audioStream?.getTracks().forEach(track => track.stop());
            audioContext?.close();
            clearTimeout(silenceTimer);
            if (isManualStop && currentAgentId) {
                const micBtn = document.querySelector(`#chat-view-${currentAgentId} .mic-btn`);
                if (micBtn) {
                    micBtn.classList.remove('listening');
                    micBtn.title = 'Start Listening';
                }
                setChatControlsEnabled(currentAgentId, true);
            }
        }

        function detectSilence(analyser, dataArray) {
            if (!isRecording) return;
            analyser.getByteTimeDomainData(dataArray);
            let sum = 0;
            for (let i = 0; i < dataArray.length; i++) sum += Math.abs((dataArray[i] / 128.0) - 1.0);
            const SILENCE_THRESHOLD = 0.01;
            const SILENCE_TIMEOUT = 1500;
            if (sum / dataArray.length > SILENCE_THRESHOLD) {
                if (silenceTimer) clearTimeout(silenceTimer);
                silenceTimer = null;
            } else if (!silenceTimer) {
                silenceTimer = setTimeout(() => stopRecording(false), SILENCE_TIMEOUT);
            }
            requestAnimationFrame(() => detectSilence(analyser, dataArray));
        }

        
		function onRecordingStop(agentId) {
            if (wasManuallyStopped) {
                wasManuallyStopped = false; 
                return;
            }
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            audioChunks = [];
            const micBtn = document.querySelector(`#chat-view-${agentId} .mic-btn.listening`);
            if (audioBlob.size < 1000) { 
                if (micBtn) startRecording(agentId);
                else setChatControlsEnabled(agentId, true);
                return;
            };
            
            document.getElementById(`loading-text-${agentId}`).textContent = "Speech detected, processing...";
            sendAudioToServer(audioBlob, agentId, micBtn);
        }
		
		

        async function sendAudioToServer(audioBlob, agentId, micBtn) {
             const chatView = document.getElementById(`chat-view-${agentId}`);
             const textInput = chatView.querySelector('.chat-input');
             
             const agent = agents.find(a => a.id === agentId);
             const langToUse = agent && !agent.isDefault ? agent.tts_lang : languageSelector.value;

             const formData = new FormData();
             formData.append('audio_data', audioBlob, 'recording.wav');
             formData.append('language', langToUse);

             try {
                const response = await fetch('/transcribe', { method: 'POST', body: formData });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Transcription failed.");

                if (data.status === 'no_speech' && activeChats[agentId].uploadQueue.length === 0) {
                   if (micBtn) startRecording(agentId);
                   else setChatControlsEnabled(agentId, true);
                   return;
                }
                
                if (!micBtn) { // If mic was not in continuous mode, turn it off now.
                    stopRecording(true);
                }

                textInput.value = data.transcribedText || '';
                chatView.querySelector('.chat-form').dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));

             } catch (error) {
                showError(error.message);
                setChatControlsEnabled(agentId, true);
             }
        }
        
        function updateVoiceOptions() {
            const langCode = languageSelector.value;
            const voices = ttsVoices[langCode]?.voices || {};
            const currentVoice = voiceSelector.value;
            voiceSelector.innerHTML = '';
            for (const voiceCode in voices) {
                const option = new Option(voices[voiceCode], voiceCode);
                voiceSelector.add(option);
            }
            // Try to preserve the selected voice if it exists in the new language
            if (Array.from(voiceSelector.options).some(o => o.value === currentVoice)) {
                voiceSelector.value = currentVoice;
            }
        }
		
        // --- Function to manage panel state ---
        function updatePanelState(isHidden) {
            if (isHidden) {
                bodyEl.classList.add('panel-hidden');
                togglePanelBtn.innerHTML = iconShow;
                togglePanelBtn.title = "Show Sidebar";
            } else {
                bodyEl.classList.remove('panel-hidden');
                togglePanelBtn.innerHTML = iconHide;
                togglePanelBtn.title = "Hide Sidebar";
            }
        }
        

        function applySettingsToUI(settings) {
            const updateSlider = (slider, valueDisplay, value, formatFn) => {
                // Defensive check to prevent errors if an element is missing
                if (!slider || !valueDisplay) return;
                
                slider.value = value;
                // Ensure formatFn is always a function before calling it
                if (typeof formatFn === 'function') {
                    valueDisplay.textContent = formatFn(value);
                }
            };

            modelSelector.value = settings.model;

            ttsEnabledSelector.value = settings.tts_enabled;
            languageSelector.value = settings.tts_lang;
            updateVoiceOptions();
            
            const availableVoices = Array.from(voiceSelector.options).map(o => o.value);
            if (availableVoices.includes(settings.tts_voice)) {
                voiceSelector.value = settings.tts_voice;
            }

            updateSlider(speedSlider, speedValue, settings.tts_speed, v => `${parseFloat(v).toFixed(1)}x`);
            updateSlider(numCtxSlider, numCtxValue, settings.num_ctx, v => v);
            updateSlider(temperatureSlider, temperatureValue, settings.temperature, v => parseFloat(v).toFixed(2));
            updateSlider(topPSlider, topPValue, settings.top_p, v => parseFloat(v).toFixed(2));
            updateSlider(maxPagesSlider, maxPagesValue, settings.max_pages, v => v);
            updateSlider(pdfResSlider, pdfResValue, settings.pdf_image_res, v => parseFloat(v).toFixed(1));
            updateSlider(maxUploadSlider, maxUploadValue, settings.max_upload_file_size, v => v);
        }
		
		

        // --- WebSocket Event Listeners Setup ---
        function setupSocketListeners() {
            socket.on('llm_token', (data) => {
                if (isPlaybackStopped) return;
                if (currentAiMessageElement) {
                    const container = document.getElementById(`chat-messages-container-${currentAgentId}`);
                    const shouldScroll = isUserAtBottom(container);

                    const chat = activeChats[currentAgentId];
                    if (chat && chat.history.length > 0) {
                        const lastMsg = chat.history[chat.history.length - 1];
                        lastMsg.parts[0].text += data.token;
                        currentAiMessageElement.innerHTML = marked.parse(lastMsg.parts[0].text);
                        
                        
                        // This line is added to apply styling to code blocks as they are being streamed.
                        enhanceCodeBlocks(currentAiMessageElement);
                        
                    }
                    
                    if (shouldScroll) {
                        scrollToBottom(currentAgentId);
                    }
                }
            });

            socket.on('tts_audio_chunk', (data) => {
                if (isPlaybackStopped) return;
                if (data.audioData) {
                    audioQueue.push(data.audioData);
                    playNextInQueue();
                }
            });

            socket.on('chat_end', async (data) => {
                isTyping = false;
                const chat = activeChats[currentAgentId];
                if (chat && chat.history.length > 0) {
                    const lastMsg = chat.history[chat.history.length - 1];
                    lastMsg.parts[0].text = data.final_message;
                    
                    if(currentAiMessageElement) {
                        currentAiMessageElement.innerHTML = marked.parse(data.final_message);
                        enhanceCodeBlocks(currentAiMessageElement);
                    }
                    
                    await saveOrUpdateCurrentChat(currentAgentId);
                }
                currentAiMessageElement = null;
                if (audioQueue.length === 0 && !isAudioPlaying) {
                    onAiSpeechEnd();
                }
            });
            
            // --- START: New listener for context warning ---
            socket.on('context_warning', (data) => {
                if (data.message) {
                    showError(data.message); // Use the non-blocking modal
                }
            });
            // --- END: New listener ---

            socket.on('error', (data) => {
                showError(data.error);
                isTyping = false;
                onAiSpeechEnd();
            });
        }
		
		

        // --- Event Listeners Setup ---
        function setupEventListeners() {
            closeErrorModalBtn.onclick = () => errorModalEl.classList.add('hidden');
            openSidebarBtn.onclick = () => agentSidebar.classList.remove('-translate-x-full');
            closeSidebarBtn.onclick = () => agentSidebar.classList.add('-translate-x-full');
            openCreateAgentModalBtn.addEventListener('click', openCreateAgentModal);
            cancelAgentEditorBtn.addEventListener('click', closeAgentEditorModal);
            agentEditorForm.addEventListener('submit', handleSaveAgent);
            deleteAgentBtn.addEventListener('click', handleDeleteAgent);
            
            toggleWebcamBtn.addEventListener('click', () => {
                webcamContainer.classList.toggle('hidden');
                if (webcamContainer.classList.contains('hidden')) stopWebcam();
            });
            startStopWebcamBtn.addEventListener('click', startWebcam);
            captureWebcamBtn.addEventListener('click', captureWebcamImage);

            toggleVoiceBtn.addEventListener('click', () => voiceContainer.classList.toggle('hidden'));
            toggleParamsBtn.addEventListener('click', () => paramsContainer.classList.toggle('hidden'));
            audioPlayer.addEventListener('ended', () => {
                isAudioPlaying = false;
                if (!isPlaybackStopped) {
                    playNextInQueue();
                }
                if (audioQueue.length === 0 && !isTyping) {
                    onAiSpeechEnd();
                }
            });

			document.getElementById('global-history-btn').addEventListener('click', () => {
                if (currentAgentId) {
                    const historyPanel = document.getElementById(`chat-history-panel-${currentAgentId}`);
                    if (historyPanel) {
                        if (historyPanel.classList.contains('translate-x-full')) {
                            renderSavedChatsList(currentAgentId);
                            historyPanel.classList.remove('translate-x-full');
                        } else {
                            historyPanel.classList.add('translate-x-full');
                        }
                    }
                }
            });

            tabHeaderEl.addEventListener('click', e => {
                if (e.target.classList.contains('close-tab-btn')) {
                    e.stopPropagation();
                    closeChatTab(e.target.dataset.agentId);
                }
            });
            
            const setupSliderListener = (slider, valueDisplay, formatFn) => {
                slider.addEventListener('input', () => {
                    valueDisplay.textContent = formatFn(slider.value);
                });
                slider.addEventListener('change', handleSettingsChange);
            };

            // Voice and Model Selectors
            modelSelector.addEventListener('change', handleSettingsChange);
            ttsEnabledSelector.addEventListener('change', handleSettingsChange);
            languageSelector.addEventListener('change', () => {
                updateVoiceOptions();
                handleSettingsChange();
            });
            voiceSelector.addEventListener('change', handleSettingsChange);
            
            // Sliders
            setupSliderListener(speedSlider, speedValue, v => `${parseFloat(v).toFixed(1)}x`);
            setupSliderListener(numCtxSlider, numCtxValue, v => v);
            setupSliderListener(temperatureSlider, temperatureValue, v => parseFloat(v).toFixed(2));
            setupSliderListener(topPSlider, topPValue, v => parseFloat(v).toFixed(2));
            setupSliderListener(maxPagesSlider, maxPagesValue, v => v);
            setupSliderListener(pdfResSlider, pdfResValue, v => parseFloat(v).toFixed(1));
            setupSliderListener(maxUploadSlider, maxUploadValue, v => v);

            // --- Added event listener for panel toggle ---
            togglePanelBtn.addEventListener('click', () => {
                const isHidden = bodyEl.classList.toggle('panel-hidden');
                // --- The following line is removed to stop saving the state ---
                // localStorage.setItem('panelHidden', isHidden); 
                updatePanelState(isHidden);
            });
            // --- END: Added event listener ---
        }


		async function loadAgents() {
			try {
				const res = await fetch("/agents");
				if (!res.ok) throw new Error("Failed to load agents");
				agents = await res.json();
				renderAgents();
			} catch (err) {
				console.error("Error loading agents:", err);
                showError("Could not load the list of AI Tools.");
			}
		}

		document.addEventListener('DOMContentLoaded', async () => {
			try {
				const res = await fetch("/conversations");
				if (!res.ok) throw new Error("Failed to load histories");
				savedHistories = await res.json();
			} catch (err) {
				console.error("Could not load saved conversations:", err);
				showError("Could not load saved conversations. They may be lost.");
			}
			await loadAgents();
   
            for (const langCode in ttsVoices) {
                const option = new Option(ttsVoices[langCode].name, langCode);
                languageSelector.add(option);
            }
            
            const initialSettings = { ...savedSettings, model: currentModel };
            applySettingsToUI(initialSettings);

			setupEventListeners();
            setupSocketListeners();
            updateSidebarControlsState();

            // --- Restore panel state from localStorage on load ---
            // --- The following lines are removed to ensure the panel is always open on load ---
            // const savedState = localStorage.getItem('panelHidden') === 'true';
            updatePanelState(false); // Always start with the panel visible
            
		});


    </script>

</body>
</html>
"""

# --- Agent management logic is centralized on the backend ---

DEFAULT_AGENT = {
    "id": "assistant",
    "name": "Ai Assistant",
    "title": "A friendly Ai Assistant",
    "persona": "You are a friendly and helpful assistant. Do not use emojis. Use LaTeX notation for mathematical or scientific expressions only.",
    "color": "#4f46e5",
    "type": "multi-turn",
    "isDefault": True
}



def save_agents(all_agents):
    """Saves the full list of agents to agents.json."""
    with open(AGENTS_FILE, "w") as f:
        json.dump(all_agents, f, indent=2)
		
		

def initialize_agents_file():
    """Checks if agents.json exists and is valid, creating it with the default agent if not."""
    if not os.path.exists(AGENTS_FILE):
        print(f"[INFO] '{AGENTS_FILE}' not found. Creating with default agent.")
        save_agents([DEFAULT_AGENT])
    else:
        try:
            with open(AGENTS_FILE, "r") as f:
                agents_from_file = json.load(f)
                if not isinstance(agents_from_file, list) or not agents_from_file:
                    print(f"[INFO] '{AGENTS_FILE}' is empty or invalid. Re-creating with default agent.")
                    save_agents([DEFAULT_AGENT])
                elif not any(a.get('isDefault') for a in agents_from_file):
                     print(f"[INFO] Default agent not found in '{AGENTS_FILE}'. Prepending it.")
                     agents_from_file.insert(0, DEFAULT_AGENT)
                     save_agents(agents_from_file)
        except (json.JSONDecodeError, IOError):
            print(f"[ERROR] Could not read '{AGENTS_FILE}'. Re-creating with default agent.")
            save_agents([DEFAULT_AGENT])
			
			

def load_agents():
    """Loads all agents from agents.json, falling back to default if file is corrupt."""
    try:
        with open(AGENTS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return [DEFAULT_AGENT]



def load_conversations():
    if not os.path.exists(CONVERSATIONS_FILE):
        return {}
    try:
        with open(CONVERSATIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
		
		

def save_conversations(conversations):
    with open(CONVERSATIONS_FILE, "w") as f:
        json.dump(conversations, f, indent=2)
		
		
# --- Helper Text Processing Functions (from voice-app.py) ---
def clean_text(text):
    # Basic cleaning to remove markdown and emojis that TTS might read literally
    markdown_pattern = r'([*_~`#\[\]()<>])'
    text = re.sub(markdown_pattern, '', text)
    try:
        emoji_pattern = re.compile(
            "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
    except re.error:
        emoji_pattern = re.compile(u'(\ud83c[\udf00-\udfff]|\ud83d[\udc00-\ude4f\ude80-\udeff]|[\u2600-\u26FF\u2700-\u27BF])+', flags=re.UNICODE)
    return emoji_pattern.sub(r'', text).strip()
	

def split_into_sentences(text):
    # More robust sentence splitting
    abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g|Inc|Ltd|Corp|Co)'
    protected_text = re.sub(rf'({abbreviations})\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
    sentence_pattern = r'([.!?]+["\'\)]*(?:\s+|$))'
    parts = re.split(sentence_pattern, protected_text)
    parts = [p.replace('<PERIOD>', '.') for p in parts]
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else '')
        sentence = sentence.strip()
        if sentence: sentences.append(sentence)
    if len(parts) % 2 == 1 and parts[-1].strip(): sentences.append(parts[-1].strip())
    return sentences
	

# --- Garbled Text Filtering Functions ---
def has_repeated_phrases(text: str) -> bool:
    """Checks for garbled, highly repetitive text using regex."""
    pattern = r"(.{5,})(\s*\1){2,}"
    return bool(re.search(pattern, text))
	

def contains_mixed_scripts(text: str) -> bool:
    """Checks if text contains multiple scripts, indicating garbled transcription."""
    scripts = {
        "latin": re.compile(r'[a-zA-Z]'),
        "arabic": re.compile(r'[\u0600-\u06FF]'),
        "cyrillic": re.compile(r'[\u0400-\u04FF]'),
        "cjk": re.compile(r'[\u4e00-\u9fff]')
    }
    return sum(1 for script in scripts.values() if script.search(text)) > 1
		

# --- Custom Error Handler for 413 Payload Too Large ---
@app.errorhandler(413)
def request_entity_too_large(error):
    user_settings = load_settings()
    max_size = user_settings.get("max_upload_file_size", 20)
    error_message = f"File is too large. The current maximum upload size is {max_size} MB."
    return jsonify({"error": error_message}), 413


@app.route("/agents", methods=["GET"])
def get_agents():
    return jsonify(load_agents())
	
	

@app.route("/agents", methods=["POST"])
def create_agent():
    all_agents = load_agents()
    new_agent_data = request.json
    if not all(k in new_agent_data for k in ['id', 'name', 'title', 'persona', 'type']):
        return jsonify({"error": "Missing agent data"}), 400
    
    new_agent_data['isDefault'] = False
    
    current_settings = load_settings()
    current_model = load_last_model() or (model_list[0] if model_list else "")
    
    new_agent_data['model'] = current_model
    new_agent_data.update(current_settings)

    all_agents.insert(0, new_agent_data)
    save_agents(all_agents)
    return jsonify(new_agent_data), 201
	
	

@app.route("/agents/reorder", methods=["POST"])
def reorder_agents():
    data = request.json
    ordered_ids = data.get("order")
    if not isinstance(ordered_ids, list):
        return jsonify({"error": "Invalid data format"}), 400

    all_agents = load_agents()
    agent_map = {agent['id']: agent for agent in all_agents}
    
    reordered_agents = [agent_map[agent_id] for agent_id in ordered_ids if agent_id in agent_map]
    
    if len(reordered_agents) != len(all_agents):
        return jsonify({"error": "Mismatch in agent count during reordering"}), 400

    save_agents(reordered_agents)
    return jsonify({"status": "success"})
	
	

@app.route("/agents/<agent_id>", methods=["PUT"])
def update_agent(agent_id):
    all_agents = load_agents()
    agent_found = False
    for i, agent in enumerate(all_agents):
        if agent["id"] == agent_id:
            if agent.get("isDefault"):
                return jsonify({"error": "Default agent properties cannot be modified."}), 403
            
            updated_data = request.json
            updated_data.pop('id', None)
            updated_data.pop('isDefault', None)
            all_agents[i].update(updated_data)
            agent_found = True
            break
    if agent_found:
        save_agents(all_agents)
        return jsonify(all_agents[i])
    return jsonify({"error": "Agent not found"}), 404



@app.route("/agents/<agent_id>/settings", methods=["POST"])
def save_agent_settings(agent_id):
    all_agents = load_agents()
    agent_found = False
    settings_to_save = request.json
    
    for agent in all_agents:
        if agent["id"] == agent_id:
            if agent.get("isDefault"):
                return jsonify({"error": "Default agent settings are managed globally."}), 400
            
            agent.update(settings_to_save)
            agent_found = True
            break
    
    if agent_found:
        save_agents(all_agents)
        print(f"[INFO] Saved settings for agent '{agent_id}'.")
        return jsonify({"status": "success"})
    
    return jsonify({"error": "Agent not found"}), 404
	
	

@app.route("/agents/<agent_id>", methods=["DELETE"])
def delete_agent(agent_id):
    all_agents = load_agents()
    
    agent_to_delete = next((a for a in all_agents if a["id"] == agent_id), None)
    if not agent_to_delete:
        return jsonify({"error": "Agent not found"}), 404

    if agent_to_delete.get("isDefault"):
        return jsonify({"error": "The default agent cannot be deleted."}), 403

    all_agents = [a for a in all_agents if a["id"] != agent_id]
    save_agents(all_agents)
    
    conversations = load_conversations()
    if agent_id in conversations:
        del conversations[agent_id]
        save_conversations(conversations)
        
    return jsonify({"status": "deleted"})
	
	

@app.route("/conversations", methods=["GET"])
def get_conversations():
    return jsonify(load_conversations())
	
		

@app.route("/conversations/<agent_id>", methods=["POST"])
def save_conversation(agent_id):
    new_chat_session = request.json
    if not all(k in new_chat_session for k in ['id', 'timestamp', 'title', 'history']):
        return jsonify({"error": "Invalid chat session format"}), 400

    conversations = load_conversations()
    if agent_id not in conversations:
        conversations[agent_id] = []

    conversations[agent_id].insert(0, new_chat_session)
    save_conversations(conversations)
    return jsonify({"status": "saved"}), 200
	
		

@app.route("/conversations/<agent_id>/<chat_id>", methods=["PUT"])
def update_conversation(agent_id, chat_id):
    updated_data = request.json
    if 'history' not in updated_data:
        return jsonify({"error": "Invalid update format, missing history"}), 400

    conversations = load_conversations()
    if agent_id in conversations:
        chat_index = next((i for i, chat in enumerate(conversations[agent_id]) if chat.get('id') == chat_id), -1)

        if chat_index != -1:
            conversations[agent_id][chat_index]['history'] = updated_data['history']
            conversations[agent_id][chat_index]['timestamp'] = datetime.now(timezone.utc).isoformat()
            updated_chat = conversations[agent_id].pop(chat_index)
            conversations[agent_id].insert(0, updated_chat)
            save_conversations(conversations)
            return jsonify({"status": "updated"})

    return jsonify({"error": "History not found"}), 404
	
		

@app.route("/conversations/<agent_id>/<chat_id>/title", methods=["PUT"])
def update_chat_title(agent_id, chat_id):
    data = request.json
    new_title = data.get('title')

    if not new_title or not isinstance(new_title, str):
        return jsonify({"error": "Invalid or missing title"}), 400

    conversations = load_conversations()
    if agent_id in conversations:
        chat_found = False
        for chat in conversations[agent_id]:
            if chat.get('id') == chat_id:
                chat['title'] = new_title.strip()
                chat_found = True
                break
        
        if chat_found:
            save_conversations(conversations)
            return jsonify({"status": "title updated", "newTitle": new_title.strip()})

    return jsonify({"error": "Chat not found"}), 404



@app.route("/conversations/<agent_id>/<chat_id>", methods=["DELETE"])
def delete_conversation(agent_id, chat_id):
    conversations = load_conversations()
    if agent_id in conversations:
        initial_len = len(conversations[agent_id])
        conversations[agent_id] = [chat for chat in conversations[agent_id] if chat.get('id') != chat_id]
        if len(conversations[agent_id]) < initial_len:
            save_conversations(conversations)
            return jsonify({"status": "deleted"})
    return jsonify({"error": "History not found"}), 404



@app.route("/")
def index():
    user_settings = load_settings()
    response = Response(render_template_string(HTML_TEMPLATE, model_list=model_list, current_model=MODEL_NAME, saved_settings=user_settings))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
	

# --- Settings Routes ---
@app.route("/get_settings", methods=["GET"])
def get_settings():
    return jsonify(load_settings())
	

@app.route("/save_settings", methods=["POST"])
def save_user_settings():
    settings = load_settings()
    new_settings = request.json
    settings.update(new_settings)
    save_settings(settings)
    print("[INFO] Saved new user settings.")
    return jsonify({"status": "success"})


@app.route("/change_model", methods=["POST"])
def change_model():
    global MODEL_NAME
    data = request.json
    new_model = data.get("model")
    if new_model in model_list:
        MODEL_NAME = new_model
        save_last_model(new_model)
        print(f"[INFO] Model changed to: {MODEL_NAME}")
        return jsonify({"status": "success", "current_model": MODEL_NAME})
    else:
        return jsonify({"error": f"Model '{new_model}' not found in the available list."}), 400


# --- File Upload Handling processes to Base64 in memory ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	

def process_pdf_to_base64(file_stream, filename):
    """Converts all pages of a PDF to a list of Base64 encoded PNG strings."""
    try:
        user_settings = load_settings()
        max_pages = int(user_settings.get("max_pages", MAX_PAGES))
        pdf_image_res = float(user_settings.get("pdf_image_res", PDF_IMAGE_RES))

        pdf_bytes = file_stream.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if len(doc) > max_pages:
             # Return the error message directly
            return None, f"PDF '{filename}' has {len(doc)} pages. The limit is {max_pages} pages."
        
        # --- Add terminal printout ---
        print(f"[INFO] Received PDF '{filename}'. Converting {len(doc)} page(s) to images.")

        base64_pages = []
        # --- Loop through all pages ---
        for page_num, page in enumerate(doc): # Add page number for logging
             print(f"[INFO] Processing page {page_num + 1}/{len(doc)} of '{filename}'...")
             matrix = fitz.Matrix(pdf_image_res, pdf_image_res)
             pix = page.get_pixmap(matrix=matrix)
             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
             buffer = io.BytesIO()
             img.save(buffer, 'PNG', optimize=True) # Use PNG for better quality than JPEG for text/diagrams
             buffer.seek(0)
             encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
             base64_pages.append(encoded_string)
            
        doc.close()
        print(f"[INFO] Finished processing PDF '{filename}'.")
        return base64_pages, None
    except Exception as e:
        print(f"[ERROR] PDF processing error for '{filename}': {e}", file=sys.stderr)
        return None, f"Failed to process PDF file '{filename}'."


def process_image_to_base64(file_stream):
    """Sanitizes and converts an image file to a Base64 encoded PNG string."""
    try:
        img = Image.open(file_stream)
        # Optional: Add image resizing or validation here if needed
        buffer = io.BytesIO()
        img.save(buffer, 'PNG', optimize=True) # Standardize to PNG
        buffer.seek(0)
        encoded_string = base64.b64encode(buffer.read()).decode('utf-8')
        return [encoded_string], None # Return as a list for consistency
    except Exception as e:
        print(f"[ERROR] Image processing error: {e}", file=sys.stderr)
        return None, "Invalid or corrupt image file."
		

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        extension = filename.rsplit('.', 1)[1].lower()
        
        # Use session ID and UUID for uniqueness, even across restarts (UUID helps)
        unique_id = f"{session.get('sid', 'unknown_session')}_{uuid.uuid4()}"
        
        if extension == 'pdf':
            base64_data, error = process_pdf_to_base64(file.stream, filename)
        else:
            base64_data, error = process_image_to_base64(file.stream)

        if error:
             # Return the specific error message (e.g., page limit exceeded)
            return jsonify({'error': error}), 400
        
        # Store the list of Base64 strings (could be multiple for PDF)
        uploaded_images_store[unique_id] = base64_data
        
        # Add the ID to the user's session for cleanup on disconnect
        session.setdefault('uploaded_ids', []).append(unique_id)
        session.modified = True # Ensure session changes are saved
        
        # Return the unique ID ('file_id' for frontend)
        return jsonify({'file_id': unique_id})
    else: # If file extension is not allowed
         return jsonify({'error': f"File type '{file.filename.rsplit('.', 1)[1]}' not allowed."}), 400



@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if whisper_model is None:
        return jsonify({"error": "Whisper model not loaded."}), 500
    if 'audio_data' not in request.files:
        return jsonify({"error": "No audio file."}), 400
    
    temp_audio_path = "temp_recording.wav"
    try:
        audio_file = request.files['audio_data']
        audio_file.save(temp_audio_path)
        
        lang = request.form.get('language', 'en')
        lang_code = lang.split('-')[0]
        
        stt_start_time = time.time()
        result = whisper_model.transcribe(temp_audio_path, fp16=False, language=lang_code)
        stt_end_time = time.time()
        stt_duration = stt_end_time - stt_start_time
        print(f"\n[TIME] STT (Whisper) Duration: {stt_duration:.2f} seconds")
        print()

        user_transcript = result["text"].strip()

        if has_repeated_phrases(user_transcript) or contains_mixed_scripts(user_transcript):
            print(f"[INFO] Garbled text detected and discarded: '{user_transcript}'")
            user_transcript = ""

        if not user_transcript:
             return jsonify({"status": "no_speech"})
        
        return jsonify({"transcribedText": user_transcript})
    except Exception as e:
        print(f"[ERROR] /transcribe error: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error during transcription."}), 500
    finally:
        if os.path.exists(temp_audio_path): os.remove(temp_audio_path)


# --- WebSocket Event Handlers ---

def process_sentence_for_tts(sentence, tts_settings):
    """Generates audio for a sentence and emits it over WebSocket."""
    if kokoro is None: return
    sentence = clean_text(sentence)
    if not sentence: return
    
    try:
        tts_lang = tts_settings.get("tts_lang", "en-us")
        tts_voice = tts_settings.get("tts_voice", "af_heart")
        tts_speed = float(tts_settings.get("tts_speed", 1.0))

        lang_map = {"zh": "cmn", "fr": "fr-fr"}
        kokoro_lang = lang_map.get(tts_lang, tts_lang)

        samples, sample_rate = kokoro.create(text=sentence, voice=tts_voice, speed=tts_speed, lang=kokoro_lang)
        
        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV")
        buffer.seek(0)
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        socketio.emit('tts_audio_chunk', {'audioData': audio_base64}, room=request.sid)
    except Exception as e:
        print(f"[ERROR] TTS generation failed for sentence '{sentence}': {e}", file=sys.stderr)


@socketio.on('connect')
def handle_connect():
    session['sid'] = request.sid
    print(f"[INFO] Client connected with SID: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[INFO] Client disconnected: {request.sid}")
    # --- Clean up in-memory images associated with this session ---
    uploaded_ids = session.pop('uploaded_ids', []) # Safely get and remove list
    for image_id in uploaded_ids:
        if image_id in uploaded_images_store:
            try:
                del uploaded_images_store[image_id]
                print(f"[CLEANUP] Removed in-memory image on disconnect: {image_id}")
            except KeyError:
                pass # Already deleted? Fine.
    session.clear() # Clear remaining session data


@socketio.on('stop_generation')
def handle_stop_generation():
    """Flag the session to stop generating tokens."""
    print(f"[INFO] Received stop generation request for SID: {request.sid}")
    session['stop_generation'] = True


@socketio.on('chat_message')
def handle_chat_message(data):
    session['stop_generation'] = False  # Reset the flag for a new message
    history = data.get("history", [])
    system_message = data.get("system_message", "You are a helpful assistant.")
    model = data.get("model", MODEL_NAME)
    tts_enabled = data.get("tts_enabled", "On")
    llm_options = data.get("llm_options", {})
    tts_settings = data.get("tts_settings", {})
    
    messages = [{"role": "system", "content": system_message}]
    
    # --- Assemble messages with in-memory Base64 image data ---
    ids_to_delete_this_message = [] # Track IDs used in this turn for cleanup

    for msg in history:
        part = msg.get("parts", [{}])[0]
        content = part.get("text", "")
        image_ids = part.get("image_ids", []) # Get the IDs sent from frontend
        
        ollama_msg = {"role": msg["role"], "content": content}
        
        if image_ids:
            images_base64_list = []
            for image_id in image_ids:
                if image_id not in ids_to_delete_this_message: # Avoid duplicates if resent
                     ids_to_delete_this_message.append(image_id)
                
                stored_data = uploaded_images_store.get(image_id)
                if stored_data and isinstance(stored_data, list):
                     # stored_data is a list of base64 strings (pages for PDF, one for image)
                     images_base64_list.extend(stored_data) 
                else:
                    print(f"[WARNING] Image ID '{image_id}' not found or invalid in store for SID {request.sid}")
            
            if images_base64_list:
                ollama_msg["images"] = images_base64_list # Add all collected base64 strings
        
        messages.append(ollama_msg)

    options = {
        "num_ctx": int(llm_options.get("num_ctx", DEFAULT_NUM_CTX)),
        "temperature": float(llm_options.get("temperature", DEFAULT_TEMPERATURE)),
        "top_p": float(llm_options.get("top_p", DEFAULT_TOP_P)),
    }
    
    # --- START: Print parameters to terminal ---
    print("\n--- Applying Parameters ---")
    print(f"- Model: {model}")
	
    print(f"- Flash attention status: {flash_attention_status}")
    print(f"- Context quantization: {context_quantization_status}")

    print("\n[LLM Parameters]")
    for key, value in options.items():
        print(f"- {key}: {value}")

    if tts_enabled == "On":
        print("\n[VOICE] Settings:")
        print(f"- Language: {tts_settings.get('tts_lang', 'en-us')}")
        print(f"- Voice: {tts_settings.get('tts_voice', 'af_heart')}")
        print(f"- Speed: {tts_settings.get('tts_speed', 1.0)}")

    print("---------------------------\n")
    # --- END: Print parameters to terminal ---

    try:
        response_stream = ollama.chat(
									model=model, 
									messages=messages, 
									stream=True, 
									options=options
									)
		
        full_response, sentence_buffer = "", ""
        
        for chunk in response_stream:
            if session.get('stop_generation'):
                print(f"[INFO] Stopping generation for SID: {request.sid}")
                break

            if chunk.get("done"):
                 # --- START: Context and Performance Logging ---
                 print("\n[INFO] Finished streaming response.")
                 prompt_tokens = chunk.get('prompt_eval_count', 0)
                 completion_tokens = chunk.get('eval_count', 0)
                 eval_duration_ns = chunk.get('eval_duration', 0)
                 total_tokens = prompt_tokens + completion_tokens

                 print(f"   [STATS] Prompt Tokens:     {prompt_tokens}")
                 print(f"   [STATS] Completion Tokens: {completion_tokens}")
                 print(f"   [STATS] Total Tokens:      {total_tokens}")

                 # --- NEW: Calculate and print Tokens Per Second ---
                 if eval_duration_ns > 0:
                    eval_duration_s = eval_duration_ns / 1_000_000_000
                    tokens_per_second = completion_tokens / eval_duration_s
                    print(f"   [STATS] Performance:       {tokens_per_second:.2f} tokens/sec")
                 print()
                
                 current_num_ctx = options.get("num_ctx", DEFAULT_NUM_CTX)
                 if total_tokens >= (current_num_ctx * 0.9):
                    warning_msg = (
                        f"Context Warning: The chat has used {total_tokens} tokens, "
                        f"which is over 90% of the {current_num_ctx} token limit. "
                        "The AI may soon lose track of the conversation. Please start a new chat."
                    )
                    print(f"[WARNING] {warning_msg}")
                    print()
					
                    socketio.emit('context_warning', {'message': warning_msg}, room=request.sid)
                 # --- END: Context and Performance Logging ---
                
                 if sentence_buffer.strip() and tts_enabled == "On":
                    process_sentence_for_tts(sentence_buffer, tts_settings)
                 break
                
            token = chunk['message']['content']
            full_response += token
            sentence_buffer += token
            socketio.emit('llm_token', {'token': token}, room=request.sid)
            socketio.sleep(0)
            
            if tts_enabled == "On":
                complete_sentences = split_into_sentences(sentence_buffer)
                if len(complete_sentences) > 1:
                    for sentence in complete_sentences[:-1]:
                        process_sentence_for_tts(sentence, tts_settings)
                    sentence_buffer = complete_sentences[-1]
                elif len(complete_sentences) == 1 and sentence_buffer.strip().endswith(('.', '!', '?')):
                    process_sentence_for_tts(complete_sentences[0], tts_settings)
                    sentence_buffer = ""

        socketio.emit('chat_end', {'final_message': full_response.strip()}, room=request.sid)
        
    except Exception as e:
        print(f"[ERROR] Chat handler error: {e}", file=sys.stderr)
        socketio.emit('error', {'error': 'An error occurred with the AI model.'}, room=request.sid)
    finally:
        # --- Guaranteed cleanup of in-memory images after use ---
        # Ensure cleanup runs even if generation was stopped or errored
        for image_id in ids_to_delete_this_message:
            try:
                if image_id in uploaded_images_store:
                    del uploaded_images_store[image_id]
                    print(f"[CLEANUP] Actively deleted used in-memory image: {image_id}")
                
                # Also remove it from the session list if it exists
                if 'uploaded_ids' in session and image_id in session['uploaded_ids']:
                    try:
                         session['uploaded_ids'].remove(image_id)
                         session.modified = True # Flag session as modified
                    except ValueError:
                         pass # Already removed, fine
            except KeyError:
                 # This might happen if deleted on disconnect race condition? Be safe.
                 print(f"[ERROR] Attempted to delete {image_id} but it was already gone.", file=sys.stderr)

				 

if __name__ == "__main__":
	
    # --- Initialize agents.json on startup ---
    initialize_agents_file()

    import webbrowser, threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        threading.Timer(1.0, open_browser).start()

    socketio.run(app, host="127.0.0.1", port=5000, debug=False, allow_unsafe_werkzeug=True)

	
	