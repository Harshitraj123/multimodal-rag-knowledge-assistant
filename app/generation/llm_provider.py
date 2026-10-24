"""
LLM provider abstraction.
Supports local Ollama (primary default) and Groq, with graceful degradation when not configured or unreachable.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Wrapper around local Ollama (via standard OpenAI-compatible client) and Groq for text generation.
    Provides a simple generate() interface used throughout the application.
    """

    def __init__(
        self,
        base_url: Optional[str] = "http://localhost:11434/v1",
        model: str = "llama3.2",
        api_key: Optional[str] = None,
        provider: str = "ollama",
    ):
        # Backward-compatible check: if first positional argument is an API key rather than a URL
        if base_url and not base_url.startswith("http://") and not base_url.startswith("https://"):
            api_key = base_url
            base_url = "http://localhost:11434/v1"
            if provider == "ollama":
                provider = "groq"

        self.base_url = base_url or "http://localhost:11434/v1"
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self._client = None

    @property
    def available(self) -> bool:
        if self.provider == "ollama":
            return bool(self.base_url)
        return bool(self.api_key)

    def _get_client(self):
        if not self.available:
            if self.provider == "groq":
                raise RuntimeError(
                    "Groq API key not configured. Set GROQ_API_KEY in .env to enable LLM generation."
                )
            raise RuntimeError(
                f"LLM provider '{self.provider}' is not configured properly."
            )

        if self._client is None:
            if self.provider == "ollama":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key or "ollama",
                    )
                except ImportError:
                    try:
                        import ollama  # type: ignore
                        self._client = "ollama_native"
                    except ImportError:
                        raise ImportError(
                            "Neither 'openai' nor 'ollama' is installed. "
                            "Install dependencies with: pip install openai ollama"
                        )
            elif self.provider == "groq":
                try:
                    from groq import Groq
                    self._client = Groq(api_key=self.api_key)
                except ImportError:
                    raise ImportError("groq not installed. Run: pip install groq")
            else:
                raise ValueError(f"Unsupported LLM provider: {self.provider}")
        return self._client

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.
        Raises RuntimeError if API is not configured or fails.
        """
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        t = time.time()
        try:
            if client == "ollama_native":
                import ollama  # type: ignore
                host = self.base_url.rstrip("/").removesuffix("/v1")
                ol_client = ollama.Client(host=host)
                response = ol_client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": temperature, "num_predict": max_tokens},
                )
                result = response.message.content or ""
            else:
                # Standard OpenAI-compatible API or Groq API
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                result = response.choices[0].message.content or ""

            elapsed = time.time() - t
            logger.debug(f"LLM generated {len(result)} chars in {elapsed:.2f}s")
            return result
        except Exception as e:
            # If using OpenAI client with Ollama and it failed, try ollama native as fallback
            if self.provider == "ollama" and client != "ollama_native":
                try:
                    import ollama  # type: ignore
                    host = self.base_url.rstrip("/").removesuffix("/v1")
                    ol_client = ollama.Client(host=host)
                    ol_resp = ol_client.chat(
                        model=self.model,
                        messages=messages,
                        options={"temperature": temperature, "num_predict": max_tokens},
                    )
                    result = ol_resp.message.content or ""
                    elapsed = time.time() - t
                    logger.debug(f"LLM (fallback to ollama native) generated {len(result)} chars in {elapsed:.2f}s")
                    return result
                except Exception:
                    pass

            logger.error(f"LLM generation failed ({self.provider} / {self.model}): {e}")
            if self.provider == "ollama":
                raise RuntimeError(
                    f"Local Ollama generation failed: {e}. "
                    f"Make sure Ollama is running at {self.base_url} (run 'ollama serve') "
                    f"and model '{self.model}' is downloaded (run 'ollama pull {self.model}')."
                ) from e
            raise RuntimeError(f"LLM generation failed: {e}") from e

    def __call__(self, prompt: str, **kwargs) -> str:
        """Allow using instance as a callable (used by query_processing modules)."""
        return self.generate(prompt, **kwargs)
