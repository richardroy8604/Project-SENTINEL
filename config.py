"""
ULTRON Configuration
====================
Centralized configuration for all ULTRON modules.
Modify values here — never hardcode settings in module files.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# CAMERA / VISION
# =============================================================================
CAMERA_INDEX = 0                  # Webcam device index (0 = default)
CAMERA_WIDTH = 640                # Capture resolution width
CAMERA_HEIGHT = 480               # Capture resolution height
CAMERA_FPS = 30                   # Target capture FPS

# =============================================================================
# UI / DASHBOARD
# =============================================================================
WINDOW_WIDTH = 1280               # Dashboard window width
WINDOW_HEIGHT = 720               # Dashboard window height
WINDOW_TITLE = "ULTRON — AI Security System"

# Video display dimensions inside the dashboard
VIDEO_DISPLAY_WIDTH = 640
VIDEO_DISPLAY_HEIGHT = 480

# AI Blob visualization
BLOB_RADIUS = 120                 # Base radius of the AI orb
BLOB_COLOR_MONITORING = (0.0, 0.6, 1.0, 0.8)    # Calm blue
BLOB_COLOR_ATTENTION = (1.0, 0.7, 0.0, 0.8)     # Amber
BLOB_COLOR_SUSPICIOUS = (1.0, 0.1, 0.1, 0.8)    # Red

# =============================================================================
# SECURITY STATES
# =============================================================================
class SecurityState:
    """Enumeration of security states."""
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    ATTENTION = "ATTENTION"
    SUSPICIOUS = "SUSPICIOUS"

# =============================================================================
# DETECTION / TRACKING (Stage 2)
# =============================================================================
YOLO_MODEL = "yolo11s.pt"        # Model name (auto-downloads on first run)
DETECTION_CONFIDENCE = 0.45       # Minimum confidence for person detection

# Auto-detect GPU: use CUDA if available, else CPU
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "0"  # First GPU
    except ImportError:
        pass
    return "cpu"

DETECTION_DEVICE = _detect_device()

# =============================================================================
# AUDIO / SPEECH RECOGNITION (Stage 4)
# =============================================================================
AUDIO_SAMPLE_RATE = 16000         # 16 kHz mono audio for VAD and Whisper
AUDIO_CHANNELS = 1                # Mono channel
AUDIO_BLOCK_SIZE = 512            # 32ms chunks (512 samples @ 16kHz) for Silero VAD

# Voice Activity Detection (Silero VAD v5)
VAD_MODEL_PATH = MODELS_DIR / "silero_vad.onnx"
VAD_THRESHOLD = 0.45              # Speech probability threshold (real speech is >0.85; blocks noise)
VAD_MIN_VOICE_CHUNKS = 5          # Must have at least 5 frames (~160ms) of real voice phonemes
VAD_MIN_PEAK_AMPLITUDE = 0.02     # Minimum peak amplitude to prevent transcribing faint background hum
VAD_MIN_SPEECH_DURATION_MS = 350  # Minimum total utterance duration
VAD_SILENCE_TAIL_CHUNKS = 10      # ~320ms silence marks end of sentence (crisp cutoff)
VAD_MAX_SPEECH_DURATION_S = 6.0   # Maximum utterance length before forcing transcription

# Speech-to-Text (faster-whisper)
# Run on CPU with int8 to leave GPU VRAM 100% for YOLO + Qwen 7B LLM
STT_MODEL_SIZE = "small.en"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
STT_CPU_THREADS = 4
STT_NO_SPEECH_THRESHOLD = 0.40    # Discard if model is >40% confident it's non-speech
STT_LOG_PROB_THRESHOLD = -0.80    # Discard low-confidence mumbled audio

# =============================================================================
# LLM BRAIN & PERSONALITY (Stage 5)
# =============================================================================
# Provider selection: "groq" (ultra-fast 300 t/s LPU) or "local" (Ollama / llama-server)
LLM_PROVIDER = "groq"

# Groq Cloud Configuration (100% Free, runs massive 70B models at ~300 tokens/sec)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"  # Best conversational model for Ultron persona

# Local Ollama / llama-server Configuration
LOCAL_LLM_API_URL = "http://localhost:11434/v1"
LOCAL_LLM_MODEL = "qwen2.5:7b"

# Active configuration derived from selected provider
LLM_API_URL = GROQ_API_URL if LLM_PROVIDER == "groq" else LOCAL_LLM_API_URL
LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else LOCAL_LLM_MODEL
LLM_API_KEY = GROQ_API_KEY if LLM_PROVIDER == "groq" else ""

LLM_TEMPERATURE = 0.72            # Balance between sharp wit and grounded coherence
LLM_MAX_TOKENS = 120              # Keeps responses punchy and conversational
LLM_TIMEOUT = 8.0                 # Max seconds to wait for generation

# Vision enhancement for phone detection
VISION_DETECT_PHONES = True       # Detect COCO class 67 (cell phone) held by tracked persons

# =============================================================================
# FUTURE STAGE CONFIGS (Placeholders — will be populated as we build)
# =============================================================================

# Stage 6: TTS
# TTS_VOICE = "am_onyx"
