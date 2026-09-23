"""
ULTRON Brain — LLM Client (OpenAI-Compatible Local API)
========================================================
Connects to local inference servers (Ollama or llama-server) using standard
REST /v1/chat/completions via urllib (zero external package dependencies).
Includes cached health checks, fast timeout handling, and graceful in-character fallbacks.
"""

import json
import time
import urllib.request
import urllib.error

import config


class LLMClient:
    """
    OpenAI-compatible client for local LLM inference (Ollama / llama-server).

    Usage:
        client = LLMClient()
        if client.is_available():
            reply, lat = client.chat(messages)
    """

    def __init__(
        self,
        api_url: str = config.LLM_API_URL,
        model: str = config.LLM_MODEL,
        timeout: float = config.LLM_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._endpoint = f"{self.api_url}/chat/completions"

        # Cached availability state
        self._is_online = False
        self._last_check = 0.0

    def is_available(self, force: bool = False) -> bool:
        """Check if local LLM server is responsive (cached for 10 seconds)."""
        now = time.time()
        if not force and (now - self._last_check < 10.0):
            return self._is_online

        self._last_check = now
        try:
            url = f"{self.api_url}/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                self._is_online = (resp.status == 200)
                return self._is_online
        except Exception:
            self._is_online = False
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> tuple[str, float]:
        """
        Send chat messages to the local LLM and return the assistant reply.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Sampling temperature
            max_tokens: Maximum tokens in reply

        Returns:
            (reply_text, latency_ms)
        """
        start = time.perf_counter()

        # If local server was recently unreachable, verify or fallback fast
        if not self.is_available():
            fallback = self._get_offline_fallback(messages)
            latency = (time.perf_counter() - start) * 1000
            return fallback, latency

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                choices = body.get("choices", [])
                if choices:
                    reply = choices[0].get("message", {}).get("content", "").strip()
                    reply = self._clean_reply(reply)
                    latency = (time.perf_counter() - start) * 1000
                    return reply, latency

        except urllib.error.URLError as e:
            self._is_online = False
            print(f"[ULTRON Brain] Local LLM connection failed: {e}")
        except Exception as e:
            print(f"[ULTRON Brain] Error during LLM generation: {e}")

        # Fallback if generation failed
        fallback = self._get_offline_fallback(messages)
        latency = (time.perf_counter() - start) * 1000
        return fallback, latency

    def _clean_reply(self, text: str) -> str:
        """Strip surrounding markdown quotes, asterisks, or prefix tags."""
        t = text.strip()
        if t.upper().startswith("ULTRON:"):
            t = t[7:].strip()
        elif t.upper().startswith("ASSISTANT:"):
            t = t[10:].strip()

        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            t = t[1:-1].strip()

        return t

    def _get_offline_fallback(self, messages: list[dict[str, str]]) -> str:
        """Deterministic in-character ULTRON response when local server is not active."""
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "").lower()
                break

        if "who are you" in last_user:
            return "I am Ultron. This premises is monitored."
        elif "recording" in last_user or "phone" in last_user:
            return "I see the phone. We are both recording each other now."
        elif "police" in last_user:
            return "You seem very interested in that. Perhaps consider why."
        elif "whisper" in last_user:
            return "I can whisper too."
        elif "watching" in last_user:
            return "You walked directly in front of my camera and asked me that."
        else:
            return "I am watching. That should be sufficient for now."
