# ULTRON — AI-Powered Interactive Security System

An intelligent, interactive security and deterrence prototype combining computer vision, speech recognition, behavioral state machines, and local AI reasoning.

---

## 🌟 Key Features

- **Live Vision & Multi-Person Tracking:** Real-time webcam feed with **YOLO11s** person detection and **ByteTrack** persistent multi-person tracking.
- **Cyberpunk AI Dashboard:** High-performance **Dear PyGui** interface with GPU-rendered video and an animated, audio-reactive AI orb.
- **Real-Time Speech Pipeline:**
  - **Silero VAD v5** Voice Activity Detection (<1 ms latency on CPU, 0 MB VRAM).
  - **faster-whisper** (`small.en`) Speech-to-Text with INT8 quantization on CPU.
  - Multi-layer audio gating and strict English speech/hallucination validation.
- **Deterministic Security State Machine:**
  - `MONITORING`: Armed, scanning area (Calm Blue orb).
  - `ATTENTION`: Person(s) detected and actively tracked (Amber orb).
  - `SUSPICIOUS`: Loitering, tampering, or acoustic anomalies detected (Alert Red orb).
- **Contextual Memory:** Tracks dwell times, enter/leave timestamps, and conversation history.

---

## 🏗️ Architecture

```
SENTINEL/
├── config.py             # Centralized configuration (thresholds, hardware, UI)
├── main.py               # Main loop and event bus orchestrator
├── setup_models.py       # Auto-downloader for local ONNX/PyTorch models
├── RUN_ULTRON.bat        # 1-click Windows launcher
├── requirements.txt      # Project dependencies
│
├── core/
│   ├── event_bus.py      # Decoupled thread-safe pub/sub event system
│   ├── state_machine.py  # Rule-based deterministic security states
│   └── context.py        # Short-term tracking & conversation memory
│
├── vision/
│   ├── camera.py         # Threaded OpenCV capture (drop-oldest buffer)
│   └── detector.py       # YOLO11s person detection + ByteTrack tracking
│
├── audio/
│   ├── capture.py        # Microphone capture & live VU meter (WASAPI)
│   ├── vad.py            # Silero VAD v5 ONNX implementation (<1ms)
│   ├── stt.py            # faster-whisper INT8 CPU transcription
│   ├── listener.py       # Pipeline orchestrator with pre-roll buffer & noise gating
│   └── validator.py      # Strict English & anti-hallucination filter
│
├── brain/                # LLM brain & personality engine (Stage 5)
├── voice/                # Text-to-Speech voice synthesis (Stage 6)
├── security/             # Threat assessment & alert generation (Stage 9)
└── ui/
    └── dashboard.py      # Dear PyGui dark cyberpunk dashboard & AI orb
```

---

## 🚀 Quick Start (Windows)

### 1. Prerequisites
- Windows 10/11
- Python 3.11+
- NVIDIA GPU (RTX series recommended for CUDA acceleration)
- Webcam & Microphone

### 2. Setup
```bash
git clone https://github.com/richardroy8604/Project-SENTINEL.git
cd Project-SENTINEL

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA 12.8 support (for GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Run
Double-click `RUN_ULTRON.bat` or run:
```bash
python main.py
```
