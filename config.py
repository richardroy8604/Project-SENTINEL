"""
ULTRON Configuration
====================
Centralized configuration for all ULTRON modules.
Modify values here — never hardcode settings in module files.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS & ENVIRONMENT
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Auto-load .env file if present
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    _k = _k.strip()
                    _v = _v.strip().strip("'\"")
                    if _k and _k not in os.environ:
                        os.environ[_k] = _v
    except Exception as _e:
        print(f"[ULTRON Config] Warning: Failed to load .env: {_e}")

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

# Groq Cloud Configuration (Free, ultra-fast 27B model at ~300 tokens/sec)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "qwen/qwen3.8-27b"  # Fast, conversational, razor-sharp wit

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
# VOICE OUTPUT / TTS (Stage 6)
# =============================================================================
TTS_ENABLED = True
TTS_PROVIDER = "fish"             # "fish" (Custom Ultron Clone), "edge" (Neural Christopher), "kokoro" (offline)

# Fish Audio Configuration (Custom Voice Clone: 06cfdb3653a4496983d6ad77f98cc184)
FISH_AUDIO_API_KEY = os.getenv("FISH_AUDIO_API_KEY", "")
FISH_AUDIO_VOICE_ID = os.getenv("FISH_AUDIO_VOICE_ID", "06cfdb3653a4496983d6ad77f98cc184")
FISH_AUDIO_MODEL = "s2.1-pro-free"

# Neural Voice (Edge-TTS — Tier 2 Fallback)
TTS_EDGE_VOICE = "en-US-ChristopherNeural"  # Deep, emotional, confident male voice
TTS_EDGE_RATE = "+0%"             # Natural spoken pace
TTS_EDGE_PITCH = "-3Hz"           # Deep baritone resonance

# Offline Kokoro Fallback (Tier 3 Fallback)
TTS_MODEL_PATH = MODELS_DIR / "kokoro-v1.0.int8.onnx"
TTS_VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"
TTS_VOICE = "hybrid_baritone"      # Options: "hybrid_baritone", "am_michael", "am_echo", "am_onyx"
TTS_SPEED = 1.15                  # 1.15x = confident, conversational cadence
TTS_SENTENCE_PAUSE = 0.12         # Snappy breath between sentences
TTS_CLAUSE_PAUSE = 0.06           # Crisp comma transitions
TTS_SAMPLE_RATE = 24000           # 24 kHz audio output
TTS_REVERB_TAIL_MS = 250          # Silence tail (ms) before unmuting mic after speech
