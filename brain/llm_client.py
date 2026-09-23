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
import urllib.request
import urllib.error

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

        # Cached availability state
        self._is_online = False
        self._last_check = 0.0

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
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
            url = f"{self.api_url}/models"
            req = urllib.request.Request(url, headers=self._get_headers(), method="GET")
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                self._is_online = (resp.status == 200)
                return self._is_online
        except Exception:
            # If /models is forbidden or blocked, try a quick ping on chat/completions endpoint
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
            print("[ULTRON Brain] NOTICE: GROQ_API_KEY is not set. Add your free key in config.py!")
            fallback = self._get_offline_fallback(messages)
            latency = (time.perf_counter() - start) * 1000
            return fallback, latency

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
            headers=self._get_headers(),
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

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass
            print(f"[ULTRON Brain] API HTTP Error {e.code}: {e.reason} - {err_body}")
        except urllib.error.URLError as e:
            self._is_online = False
            print(f"[ULTRON Brain] Connection failed: {e}")
        except Exception as e:
            print(f"[ULTRON Brain] Error during generation: {e}")

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
