"""Centralized configuration for the LLM + RAG application."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


class PathsConfig(BaseSettings):
    """File and directory paths used by the app."""

    project_root: Path = Field(default_factory=_project_root)
    artifacts_dir: Path = Field(default_factory=lambda: _project_root() / "artifacts")
    rag_data_dir: Path = Field(default_factory=lambda: _project_root() / "rag_data")
    document_corpus_dir: Path = Field(
        default_factory=lambda: _project_root() / "rag_data" / "document_corpus"
    )
    raw_documents_dir: Path = Field(
        default_factory=lambda: _project_root() / "rag_data" / "raw_documents"
    )
    cleaned_documents_dir: Path = Field(
        default_factory=lambda: _project_root() / "rag_data" / "cleaned_documents"
    )

    pipeline_log: str = "pipeline.log"
    streamlit_log: str = "streamlit.log"
    metadata_csv: str = "metadata.csv"
    chunks_parquet: str = "chunks.parquet"
    embeddings_parquet: str = "embeddings.parquet"
    index_faiss: str = "index.faiss"

    @field_validator("artifacts_dir", "rag_data_dir", mode="after")
    @classmethod
    def ensure_dir_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_config = SettingsConfigDict(
        env_prefix="PATHS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LLMConfig(BaseSettings):
    """LLM settings."""

    provider: Literal["openai", "local"] = Field(default="openai")
    model_name: str = Field(default="gpt-5-nano")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class APIConfig(BaseSettings):
    """API credentials and OpenAI-compatible endpoint settings."""

    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    rotation: str = "10 MB"
    retention: str = "1 week"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
        "{name}:{function}:{line} | {message}"
    )

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class StreamlitConfig(BaseSettings):
    page_title: str = "Causal Analysis"
    page_icon: str = "💬"
    layout: str = "wide"

    model_config = SettingsConfigDict(
        env_prefix="STREAMLIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class AppConfig(BaseSettings):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)

    version: str = "1.0.0"

    @property
    def full_artifacts_dir(self) -> Path:
        return self.paths.artifacts_dir

    @property
    def rag_data_dir(self) -> Path:
        return self.paths.rag_data_dir

    @property
    def document_corpus_dir(self) -> Path:
        return self.paths.document_corpus_dir

    @property
    def raw_documents_dir(self) -> Path:
        return self.paths.raw_documents_dir

    @property
    def cleaned_documents_dir(self) -> Path:
        return self.paths.cleaned_documents_dir

    @property
    def chunks_path(self) -> Path:
        return self.rag_data_dir / self.paths.chunks_parquet

    @property
    def documents_metadata_path(self) -> Path:
        return self.document_corpus_dir / self.paths.metadata_csv

    @property
    def embeddings_path(self) -> Path:
        return self.rag_data_dir / self.paths.embeddings_parquet

    @property
    def faiss_index_path(self) -> Path:
        return self.rag_data_dir / self.paths.index_faiss

    @property
    def pipeline_log_path(self) -> Path:
        return self.full_artifacts_dir / self.paths.pipeline_log

    @property
    def streamlit_log_path(self) -> Path:
        return self.full_artifacts_dir / self.paths.streamlit_log

    @property
    def effective_llm_api_key(self) -> Optional[str]:
        return os.getenv("LLM_API_KEY") or self.api.llm_api_key

    @property
    def effective_openai_api_key(self) -> Optional[str]:
        return self.effective_llm_api_key

    @property
    def effective_llm_provider(self) -> str:
        return (os.getenv("LLM_PROVIDER") or self.llm.provider or "openai").lower()

    @property
    def effective_llm_base_url(self) -> Optional[str]:
        return os.getenv("LLM_BASE_URL") or self.api.llm_base_url

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = AppConfig()


def get_config() -> AppConfig:
    return config
