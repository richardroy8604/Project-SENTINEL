"""
ULTRON Brain — LLM Client (OpenAI & Groq Compatible)
======================================================
Connects to OpenAI-compatible endpoints:
  - Groq Cloud LPU (Free, ultra-fast ~300 tokens/sec, massive 70B models)
  - Local Ollama / llama-server (100% offline)

Supports bearer authorization, cached health checks, and graceful fallbacks.
"""

import json
import time
import requests

import config


class LLMClient:
    """
    Client for Groq or local Ollama LLM inference.

    Usage:
        client = LLMClient()
        if client.is_available():
            reply, lat = client.chat(messages)
    """

    def __init__(
        self,
        api_url: str = config.LLM_API_URL,
        model: str = config.LLM_MODEL,
        api_key: str = config.LLM_API_KEY,
        timeout: float = config.LLM_TIMEOUT,
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.api_key = api_key.strip()
        self.timeout = timeout
        self._endpoint = f"{self.api_url}/chat/completions"
        self._session = requests.Session()

        # Cached availability state
        self._is_online = False
        self._last_check = 0.0

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ULTRON-Security/1.0 (Windows NT 10.0; Win64; x64)",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def is_available(self, force: bool = False) -> bool:
        """Check if LLM provider is responsive (cached for 15 seconds)."""
        now = time.time()
        if not force and (now - self._last_check < 15.0):
            return self._is_online

        self._last_check = now

        # If using Groq but no API key is set yet
        if config.LLM_PROVIDER == "groq" and not self.api_key:
            self._is_online = False
            return False

        try:
            resp = self._session.get(
                f"{self.api_url}/models",
                headers=self._get_headers(),
                timeout=3.0,
            )
            self._is_online = (resp.status_code == 200)
            return self._is_online
        except Exception:
            # If /models endpoint is restricted or timeout, verify if key exists
            if self.api_key:
                self._is_online = True
                return True
            self._is_online = False
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
    ) -> tuple[str, float]:
        """
        Send chat messages to Groq or local LLM and return the assistant reply.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Sampling temperature
            max_tokens: Maximum tokens in reply

        Returns:
            (reply_text, latency_ms)
        """
        start = time.perf_counter()

        # If Groq is selected but no key is provided, alert and use fallback
        if config.LLM_PROVIDER == "groq" and not self.api_key:
            print("[ULTRON Brain] NOTICE: GROQ_API_KEY is not set. Add your key in .env or config.py!")
            fallback = self._get_offline_fallback(messages)
            latency = (time.perf_counter() - start) * 1000
            return fallback, latency

        if not self.is_available():
            print("[ULTRON Brain] Provider unavailable, using offline response")
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

        try:
            response = self._session.post(
                self._endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code == 200:
                body = response.json()
                choices = body.get("choices", [])
                if choices:
                    reply = choices[0].get("message", {}).get("content", "").strip()
                    reply = self._clean_reply(reply)
                    latency = (time.perf_counter() - start) * 1000
                    return reply, latency
            else:
                print(f"[ULTRON Brain] API HTTP Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"[ULTRON Brain] Connection error during generation: {e}")

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
        """Deterministic in-character ULTRON response when offline."""
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
