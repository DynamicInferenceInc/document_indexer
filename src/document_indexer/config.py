"""Typed configuration for a single indexer profile."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from document_indexer.domain.documents import resolve_index_extensions


class QdrantSettings(BaseModel):
    """Connection to one Qdrant instance and collection."""

    model_config = ConfigDict(extra="forbid")

    url: str = "http://127.0.0.1:6333"
    collection: str = "docs"
    extra_payload: dict[str, Any] = Field(default_factory=dict)
    payload_indexes: list[str] | None = None
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    index_version: str = ""

    @field_validator("extra_payload", mode="before")
    @classmethod
    def _parse_extra_payload(cls, value: Any) -> Any:
        if value is None or value == "":
            return {}
        if isinstance(value, str):
            return json.loads(value)
        return value

    @field_validator("payload_indexes", mode="before")
    @classmethod
    def _parse_payload_indexes(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


class ModelSettings(BaseModel):
    """Embedding, VLM and chunker settings. Defaults match the original reindex."""

    model_config = ConfigDict(extra="forbid")

    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_sec: float = 120.0
    extraction_model: str = ""
    extraction_timeout_sec: float = 180.0
    chunk_size: int = 1024
    picture_description_enabled: bool = True
    vlm_model: str = "qwen3-vl:8b"
    vlm_timeout_sec: float = 90.0
    vlm_concurrency: int = 2
    picture_area_threshold: float = 0.02


class ChunkingSettings(BaseModel):
    """Which chunker to use and sliding-window size for resume fallback."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["table_aware", "resume_project", "hybrid"] = "table_aware"
    merge_peers: bool = True
    repeat_table_header: bool = False
    window_chars: int = 1200
    window_overlap: int = 150


class LocalSourceSettings(BaseModel):
    """Watch a local directory with inotify/watchdog events."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["local"] = "local"
    watch_path: str = "/var/lib/document-indexer/docs"
    debounce_seconds: float = 1.0


class SmbSourceSettings(BaseModel):
    """Poll an SMB share and mirror it into a local staging directory."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["smb"] = "smb"
    server: str
    share: str
    username: str
    password: SecretStr
    staging_path: str = "/var/lib/document-indexer/staging"
    domain: str | None = None
    subpath: str = ""
    port: int = 445
    timeout_sec: float = 30.0
    poll_interval_sec: float = 15.0
    max_backoff_sec: float = 60.0


class IndexerSettings(BaseSettings):
    """One indexer profile: source, Qdrant target, models, extensions.

    Prefer a concrete profile::

        ProfileLocal(
            source=LocalSourceSettings(watch_path="/data/docs"),
            qdrant=QdrantSettings(collection="legal"),
        )
        ProfileSmb(
            source=SmbSourceSettings(server="fileserver", share="docs", ...),
            qdrant=QdrantSettings(collection="legal"),
        )

    Nested env keys use ``__`` (``QDRANT__URL``, ``SOURCE__WATCH_PATH``,
    ``MODELS__EMBEDDING_MODEL``). Passwords are ``SecretStr`` and are redacted
    by :meth:`model_dump_safe`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    source: LocalSourceSettings | SmbSourceSettings = Field(
        default_factory=LocalSourceSettings,
    )
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    index_extensions: str = Field(
        default="",
        description=(
            "Comma-separated suffixes to index. Empty means all Docling-readable "
            "types from domain.formats. Unknown types raise at settings load."
        ),
    )
    log_level: str = "INFO"
    resume_parse_only: bool = Field(
        default=False,
        description=(
            "Docling + resume parser only. No extraction LLM, embeddings, or Qdrant. "
            "Env: RESUME_PARSE_ONLY=1."
        ),
    )

    @field_validator("resume_parse_only", mode="before")
    @classmethod
    def _empty_resume_parse_only_is_false(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return False
        return value

    @field_validator("index_extensions")
    @classmethod
    def _index_extensions_must_be_docling_readable(cls, value: str) -> str:
        resolve_index_extensions(value)
        return value

    def model_dump_safe(self) -> dict[str, Any]:
        """Serialize settings without revealing secrets."""
        return self.model_dump(mode="json")


class ProfileLocal(IndexerSettings):
    """Indexer profile that watches a local directory."""

    source: LocalSourceSettings = Field(default_factory=LocalSourceSettings)


class ProfileSmb(IndexerSettings):
    """Indexer profile that polls an SMB share into local staging."""

    source: SmbSourceSettings
