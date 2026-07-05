from __future__ import annotations

import hashlib
import json
import os
import requests
import time
from dataclasses import dataclass
from pathlib import Path

from . import config
from .utils import approx_tokens


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    input_tokens_est: int
    output_tokens_est: int
    cache_hit: bool = False


class BaseLLM:
    provider = "base"
    model = "base"

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        raise NotImplementedError


class DummyLLM(BaseLLM):
    provider = "dummy"
    model = "dry-run"

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.perf_counter()
        text = (
            "DRY_RUN_RESPONSE: Configure an API key to generate real outputs. "
            "This placeholder preserves the workflow, records, citations, and latency fields."
        )
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            latency_s=time.perf_counter() - start,
            input_tokens_est=approx_tokens(system + prompt),
            output_tokens_est=approx_tokens(text),
        )


class GeminiLLM(BaseLLM):
    provider = "gemini"

    def __init__(self) -> None:
        from google import genai

        self.model = config.env("GEMINI_MODEL", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite"
        self.client = genai.Client(api_key=config.env("GEMINI_API_KEY"))

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.perf_counter()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        response = call_with_retries(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
            )
        )
        text = getattr(response, "text", "") or ""
        return LLMResponse(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            latency_s=time.perf_counter() - start,
            input_tokens_est=approx_tokens(full_prompt),
            output_tokens_est=approx_tokens(text),
        )


class OpenAILLM(BaseLLM):
    provider = "openai"

    def __init__(self, compatible: bool = False) -> None:
        from openai import OpenAI

        if compatible:
            self.provider = "openai_compatible"
            self.model = config.env("OPENAI_COMPATIBLE_MODEL", "deepseek-v4-flash") or "deepseek-v4-flash"
            self.client = OpenAI(
                api_key=config.env("OPENAI_COMPATIBLE_API_KEY"),
                base_url=config.env("OPENAI_COMPATIBLE_BASE_URL"),
            )
        else:
            self.model = config.env("OPENAI_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
            self.client = OpenAI(api_key=config.env("OPENAI_API_KEY"))

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.perf_counter()
        response = call_with_retries(
            lambda: self.client.chat.completions.create(
                model=self.model,
                temperature=float(config.env("TEMPERATURE", "0") or "0"),
                messages=[
                    {"role": "system", "content": system or "You are a careful course-material QA assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            latency_s=time.perf_counter() - start,
            input_tokens_est=approx_tokens(system + prompt),
            output_tokens_est=approx_tokens(text),
        )


class OllamaLLM(BaseLLM):
    provider = "ollama"

    def __init__(self) -> None:
        self.model = config.env("OLLAMA_MODEL", "qwen2.5:3b") or "qwen2.5:3b"
        self.base_url = (config.env("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434").rstrip("/")

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "You are a careful course-material QA assistant.",
            "stream": False,
            "options": {
                "temperature": float(config.env("TEMPERATURE", "0") or "0"),
            },
        }
        def _post():
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            return resp

        response = call_with_retries(_post)
        data = response.json()
        text = data.get("response", "")
        return LLMResponse(
            text=text.strip(),
            provider=self.provider,
            model=self.model,
            latency_s=time.perf_counter() - start,
            input_tokens_est=approx_tokens(system + prompt),
            output_tokens_est=approx_tokens(text),
        )


class CachedLLM(BaseLLM):
    def __init__(self, inner: BaseLLM, cache_dir: Path = config.CACHE_DIR):
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, prompt: str, system: str) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "system": system,
                    "prompt": prompt,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{key}.json"

    def complete(self, prompt: str, system: str = "") -> LLMResponse:
        path = self._cache_path(prompt, system)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("cache_hit", None)
            return LLMResponse(**data, cache_hit=True)
        response = self.inner.complete(prompt, system)
        path.write_text(json.dumps(response.__dict__, ensure_ascii=True, indent=2), encoding="utf-8")
        return response


def build_llm() -> CachedLLM:
    provider = (os.getenv("LLM_PROVIDER") or "dummy").lower()
    if provider == "gemini":
        inner: BaseLLM = GeminiLLM()
    elif provider == "openai":
        inner = OpenAILLM(compatible=False)
    elif provider in {"openai_compatible", "compatible", "deepseek", "groq"}:
        inner = OpenAILLM(compatible=True)
    elif provider == "ollama":
        inner = OllamaLLM()
    else:
        inner = DummyLLM()
    return CachedLLM(inner)


def call_with_retries(fn):
    retries = int(config.env("LLM_RETRIES", "6") or "6")
    base_delay = float(config.env("LLM_RETRY_BASE_DELAY", "4") or "4")
    last_error = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # Provider SDKs expose different exception classes.
            last_error = exc
            message = str(exc).lower()
            retryable = any(
                marker in message
                for marker in [
                    "503",
                    "unavailable",
                    "rate limit",
                    "429",
                    "temporarily",
                    "timeout",
                    "connection",
                ]
            )
            if not retryable or attempt >= retries:
                raise
            delay = min(base_delay * (2**attempt), 60)
            time.sleep(delay)
    raise last_error
