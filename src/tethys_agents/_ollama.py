import os
from openai import OpenAI
from typing import Optional


class OllamaProvider:
    """Wrapper around the OpenAI client pointed at Ollama's OpenAI-compatible endpoint.

    Host precedence: constructor arg → OLLAMA_HOST env → http://localhost:11434.
    `/v1` is appended if the host doesn't already include it (Ollama's OpenAI-
    compatible endpoint lives at `/v1/...`).

    API-key precedence: constructor arg → OLLAMA_API_KEY env → "ollama"
    placeholder. The OpenAI client requires *some* api_key argument; local
    Ollama ignores its value, so any non-empty string works.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        raw_host = host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434"
        raw_host = raw_host.rstrip("/")
        self.host = raw_host if raw_host.endswith("/v1") else f"{raw_host}/v1"
        api_key = api_key or os.environ.get("OLLAMA_API_KEY") or "ollama"
        self.client = OpenAI(base_url=self.host, api_key=api_key)
