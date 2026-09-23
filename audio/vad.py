"""
ULTRON Audio — Silero VAD v5 (Voice Activity Detector)
=======================================================
Real-time deep learning Voice Activity Detection using ONNX Runtime on CPU.
Correctly adheres to Silero VAD v5 architecture requiring a 64-sample
context prefix (576 input samples per 32ms frame).
Processes each chunk in < 1 ms on CPU with 0 MB VRAM.
"""

from pathlib import Path
import numpy as np
import onnxruntime as ort

import config


class SileroVAD:
    """
    Silero VAD v5 ONNX implementation with official 64-sample context handling.
    """

    def __init__(
        self,
        model_path: Path = config.VAD_MODEL_PATH,
        sample_rate: int = config.AUDIO_SAMPLE_RATE,
        threshold: float = config.VAD_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.threshold = threshold

        self._session: ort.InferenceSession | None = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._sr_tensor = np.array(self.sample_rate, dtype=np.int64)
        self._loaded = False

    def load(self) -> bool:
        """Initialize ONNX Runtime inference session on CPU."""
        if not self.model_path.exists():
            print(f"[ULTRON VAD] Model not found at {self.model_path}")
            return False

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.reset_state()
            self._loaded = True
            print(f"[ULTRON VAD] Silero VAD v5 loaded on CPU.")
            return True
        except Exception as e:
            print(f"[ULTRON VAD] ERROR loading model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def reset_state(self):
        """Reset internal recurrent state and context."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def process_chunk(self, chunk: np.ndarray) -> float:
        """
        Process a 512-sample chunk of 16kHz float32 audio.

        Returns:
            Speech probability (0.0 to 1.0)
        """
        if not self._loaded or self._session is None:
            return 0.0

        if chunk.ndim == 1:
            chunk = np.expand_dims(chunk, axis=0)

        # Pad or slice to exactly 512 samples
        if chunk.shape[1] < 512:
            chunk = np.pad(chunk, ((0, 0), (0, 512 - chunk.shape[1])))
        elif chunk.shape[1] > 512:
            chunk = chunk[:, :512]

        chunk = chunk.astype(np.float32)

        # Silero VAD v5 requires concatenation with 64-sample context -> (1, 576)
        inp = np.concatenate([self._context, chunk], axis=1).astype(np.float32)
        self._context = inp[:, -64:]

        try:
            ort_inputs = {
                "input": inp,
                "state": self._state,
                "sr": self._sr_tensor,
            }
            out, new_state = self._session.run(None, ort_inputs)
            self._state = new_state
            prob = float(out[0][0])
            return prob
        except Exception as e:
            print(f"[ULTRON VAD] Inference error: {e}")
            return 0.0

    def is_speech(self, chunk: np.ndarray) -> tuple[bool, float]:
        """Check if chunk exceeds speech threshold."""
        prob = self.process_chunk(chunk)
        return (prob >= self.threshold, prob)
