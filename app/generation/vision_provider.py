"""
Vision-capable model provider.
Supports local Ollama (Llava) via standard OpenAI-compatible or native client,
as well as OpenAI GPT-4o-mini.
"""
from __future__ import annotations

import base64
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class VisionProvider:
    """
    Handles image + text queries using a vision-capable model.
    Used for answering questions about charts, diagrams, and images.
    Supports local Ollama (Llava) and OpenAI vision models.
    """

    def __init__(
        self,
        provider: str = "ollama",
        api_key: Optional[str] = None,
        model: str = "llava",
        base_url: str = "http://localhost:11434/v1",
    ):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = None

    @property
    def available(self) -> bool:
        if self.provider == "ollama":
            return bool(self.base_url)
        return bool(self.api_key)

    def _get_client(self):
        if not self.available:
            if self.provider == "openai":
                raise RuntimeError(
                    f"Vision API key not configured. Set VISION_API_KEY in .env "
                    f"to enable visual analysis with {self.provider}."
                )
            raise RuntimeError(
                f"Vision provider '{self.provider}' is not configured properly."
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
            elif self.provider == "openai":
                try:
                    from openai import OpenAI
                    self._client = OpenAI(api_key=self.api_key)
                except ImportError:
                    raise ImportError("openai not installed. Run: pip install openai")
            else:
                raise ValueError(f"Unsupported vision provider: {self.provider}")
        return self._client

    def describe_image(
        self,
        image_bytes: bytes,
        question: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        """
        Describe or answer a question about an image.

        Args:
            image_bytes: Raw image data.
            question: Optional specific question about the image.
            context: Optional surrounding text context.

        Returns:
            Description or answer string.
        """
        client = self._get_client()
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = question or "Describe this image in detail, focusing on any charts, diagrams, architecture, data, or key information."
        if context:
            prompt_text = f"Context: {context[:500]}\n\nQuestion: {prompt_text}"

        t = time.time()
        try:
            if client == "ollama_native":
                import ollama  # type: ignore
                host = self.base_url.rstrip("/").removesuffix("/v1")
                ol_client = ollama.Client(host=host)
                ol_resp = ol_client.chat(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": prompt_text,
                        "images": [img_b64],
                    }],
                )
                result = ol_resp.message.content or ""
            else:
                # Standard OpenAI-compatible API (Ollama or OpenAI)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }],
                    max_tokens=512,
                )
                result = response.choices[0].message.content or ""
        except Exception as e:
            # If using OpenAI client with Ollama and it failed, try ollama native fallback
            if self.provider == "ollama" and client != "ollama_native":
                try:
                    import ollama  # type: ignore
                    host = self.base_url.rstrip("/").removesuffix("/v1")
                    ol_client = ollama.Client(host=host)
                    ol_resp = ol_client.chat(
                        model=self.model,
                        messages=[{
                            "role": "user",
                            "content": prompt_text,
                            "images": [img_b64],
                        }],
                    )
                    result = ol_resp.message.content or ""
                    elapsed = time.time() - t
                    logger.info(f"Vision analysis (ollama native fallback) completed in {elapsed:.2f}s")
                    return result
                except Exception:
                    pass

            logger.error(f"Vision analysis failed with {self.provider} ({self.model}): {e}")
            raise RuntimeError(
                f"Vision analysis failed: {e}. "
                f"Make sure local Ollama is running at {self.base_url} with vision model '{self.model}' "
                f"(e.g., 'ollama pull {self.model}')."
            ) from e

        elapsed = time.time() - t
        logger.info(f"Vision analysis completed in {elapsed:.2f}s")
        return result

    def generate_visual_description(self, image_bytes: bytes, source_info: str = "") -> str:
        """Generate a searchable description of a visual asset for indexing."""
        question = (
            "Provide a detailed description of this image for document retrieval. "
            "Include: what type of visual this is (chart/diagram/photo/table/architecture), "
            "key data or information shown, labels and axes if present, "
            "main relationships or structure depicted."
        )
        if source_info:
            question = f"This image is from: {source_info}. " + question
        try:
            return self.describe_image(image_bytes, question=question)
        except Exception as e:
            logger.warning(f"Visual description generation failed: {e}")
            return f"[Visual content from {source_info}. Vision model ({self.model}) not available or unreachable.]"
