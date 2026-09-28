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

KOKORO_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx"
)
KOKORO_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
)


def _download_file(url: str, target: Path, description: str):
    """Download a file with streaming progress."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    print(f"[ULTRON Setup] Downloading {description}...")
    temp_target = target.with_suffix(".download")

    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(temp_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        mb = downloaded / (1024 * 1024)
                        tot_mb = total_size / (1024 * 1024)
                        print(f"\r[ULTRON Setup]   -> {mb:.1f}MB / {tot_mb:.1f}MB ({pct:.0f}%)", end="", flush=True)

        print()
        if temp_target.exists():
            temp_target.replace(target)
        print(f"[ULTRON Setup] Downloaded {description} ({target.stat().st_size / (1024*1024):.1f} MB).")
    except Exception as e:
        if temp_target.exists():
            temp_target.unlink()
        raise RuntimeError(f"Failed to download {description} from {url}: {e}")


def ensure_vad_model() -> Path:
    """Download silero_vad.onnx if not already present."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = config.VAD_MODEL_PATH

    if target_path.exists() and target_path.stat().st_size > 10000:
        return target_path

    _download_file(SILERO_VAD_URL, target_path, "Silero VAD v5")
    return target_path


def ensure_tts_models() -> tuple[Path, Path]:
    """Download Kokoro ONNX model and voices file if not present."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.TTS_MODEL_PATH
    voices_path = config.TTS_VOICES_PATH

    if not model_path.exists() or model_path.stat().st_size < 50_000_000:
        _download_file(KOKORO_MODEL_URL, model_path, "Kokoro v1.0 INT8 Model (~88MB)")
    else:
        print(f"[ULTRON Setup] Kokoro TTS model verified ({model_path.name}).")

    if not voices_path.exists() or voices_path.stat().st_size < 10_000_000:
        _download_file(KOKORO_VOICES_URL, voices_path, "Kokoro Voices Library (~27MB)")
    else:
        print(f"[ULTRON Setup] Kokoro Voices verified ({voices_path.name}).")

    return model_path, voices_path


if __name__ == "__main__":
    ensure_vad_model()
    ensure_tts_models()
