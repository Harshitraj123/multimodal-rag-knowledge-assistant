"""Application settings loaded from environment variables and a project-root .env.

Default configuration uses local Ollama models and local Hugging Face models, so
normal operation does not require a paid LLM API key.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Assumes this file is in <project>/app/core/settings.py and .env is in <project>/.
BASE_DIR = Path(__file__).resolve().parents[2]

LLMProvider = Literal["ollama", "groq"]
VisionProvider = Literal["ollama"]  # Keep vision local; paid hosted vision APIs removed.


class Settings(BaseSettings):
    """Central configuration. Secrets should be supplied via environment/.env."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM provider ────────────────────────────────────────────────────────
    # Ollama is local and has no per-token API charge. Groq is optional and
    # subject to its current free-tier limits; it requires GROQ_API_KEY.
    llm_provider: LLMProvider = "ollama"
    # OpenAI-compatible endpoint (e.g. OpenAI Python client with api_key="ollama").
    ollama_base_url: str = "http://localhost:11434/v1"
    # Native Ollama endpoint (e.g. the LangChain ChatOllama integration).
    ollama_native_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    groq_api_key: str | None = None
    # Current Groq free-tier model option; availability/limits depend on account.
    groq_model: str = "openai/gpt-oss-20b"

    # ── Vision provider ─────────────────────────────────────────────────────
    # Local Ollama vision model; run `ollama pull llava` before using image features.
    vision_provider: VisionProvider = "ollama"
    vision_model: str = "llava"
    # Kept to avoid breaking older code that references this setting. Not used by
    # the local-only vision configuration.
    vision_api_key: str | None = None

    # ── Local embeddings and reranking ─────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = False

    # ── MMR ─────────────────────────────────────────────────────────────────
    mmr_enabled: bool = False
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)

    # ── Vector store ────────────────────────────────────────────────────────
    chroma_persist_directory: str = str(BASE_DIR / "data" / "chroma_db")

    # ── File handling ───────────────────────────────────────────────────────
    upload_directory: str = str(BASE_DIR / "data" / "uploads")
    max_upload_mb: int = Field(default=50, ge=1)

    # ── Retrieval defaults ──────────────────────────────────────────────────
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=64, ge=0)

    # ── Logging ─────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── App ─────────────────────────────────────────────────────────────────
    app_name: str = "Multimodal RAG Knowledge Assistant"
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False

    @field_validator("llm_provider", "vision_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: object) -> object:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        value = value.upper().strip()
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if value not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return value

    @model_validator(mode="after")
    def validate_retrieval_settings(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        if self.llm_provider == "groq" and not (self.groq_api_key and self.groq_api_key.strip()):
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        return self

    @property
    def llm_model(self) -> str:
        return self.groq_model if self.llm_provider == "groq" else self.ollama_model

    @property
    def llm_available(self) -> bool:
        """Whether the provider is configured; does not ping the server/model."""
        if self.llm_provider == "ollama":
            return bool(self.ollama_base_url.strip())
        return bool(self.groq_api_key and self.groq_api_key.strip())

    @property
    def vision_available(self) -> bool:
        """Whether a vision endpoint is configured; does not verify model download."""
        return bool(self.ollama_native_base_url.strip()) and bool(self.vision_model.strip())

    @property
    def upload_dir_path(self) -> Path:
        path = Path(self.upload_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def chroma_dir_path(self) -> Path:
        path = Path(self.chroma_persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def report_capabilities(self) -> dict[str, object]:
        """Return a sanitized capability/configuration report without secrets."""
        return {
            "llm_generation": self.llm_available,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model if self.llm_available else "not configured",
            "groq_model": self.groq_model if self.groq_api_key else "not configured",
            "vision_analysis": self.vision_available,
            "vision_provider": self.vision_provider,
            "vision_model": self.vision_model if self.vision_available else "not configured",
            "embedding_model": self.embedding_model,
            "reranking": self.reranker_enabled,
            "reranker_model": self.reranker_model if self.reranker_enabled else "disabled",
            "mmr": self.mmr_enabled,
        }


# Singleton instance
settings = Settings()
