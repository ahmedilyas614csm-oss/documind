"""Central configuration for DocuMind.

All tunable parameters live here. Values can be overridden via a .env file
or environment variables — nothing is hardcoded in the rest of the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (Groq) ---
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(
        default="openai/gpt-oss-120b",
        description="Groq model for answer generation",
    )
    groq_judge_model: str = Field(
        default="openai/gpt-oss-20b",
        description="Groq model for evaluation (JSON-only)",
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_answer_tokens: int = Field(default=512, ge=64, le=2048)

    # --- Embeddings (local, no API) ---
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name",
    )

    # --- Vector store ---
    chroma_dir: Path = Field(default=Path(".chroma"), description="Chroma persist dir")
    collection_name: str = Field(default="documind")

    # --- Chunking ---
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0, le=500)

    # --- Retrieval ---
    top_k: int = Field(default=5, ge=1, le=20)

    # --- Paths ---
    data_dir: Path = Field(default=Path("data"))
    sample_pdfs_dir: Path = Field(default=Path("data/sample_pdfs"))

    @property
    def has_api_key(self) -> bool:
        """True if a non-empty Groq API key is configured."""
        return bool(self.groq_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (so .env is only read once)."""
    return Settings()
