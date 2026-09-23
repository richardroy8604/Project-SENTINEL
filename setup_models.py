"""
ULTRON — Model Setup & Downloader
==================================
Utility script to ensure required local models (like Silero VAD ONNX)
are downloaded and placed in the models/ directory.
"""

import os
import urllib.request
from pathlib import Path

import config

SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)


def ensure_vad_model() -> Path:
    """Download silero_vad.onnx if not already present."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = config.VAD_MODEL_PATH

    if target_path.exists() and target_path.stat().st_size > 10000:
        return target_path

    print(f"[ULTRON Setup] Downloading Silero VAD v5 to {target_path}...")
    try:
        # User-agent header to prevent raw github 403
        req = urllib.request.Request(
            SILERO_VAD_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        )
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
            f.write(resp.read())
        print(f"[ULTRON Setup] Download complete ({target_path.stat().st_size} bytes).")
    except Exception as e:
        print(f"[ULTRON Setup] Failed to download Silero VAD from primary URL: {e}")
        # Secondary fallback mirror
        fallback_url = "https://huggingface.co/spaces/abidlabs/silero-vad/resolve/main/silero_vad.onnx"
        print(f"[ULTRON Setup] Trying fallback mirror: {fallback_url}...")
        req = urllib.request.Request(
            fallback_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req) as resp, open(target_path, "wb") as f:
            f.write(resp.read())
        print(f"[ULTRON Setup] Fallback download complete.")

    return target_path


if __name__ == "__main__":
    ensure_vad_model()
