"""Typed configuration for a single indexer profile."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from document_indexer.domain.documents import resolve_index_extensions


def parse_direction_map(value: Any) -> dict[str, str]:
    """Parse JSON object or ``path=direction;folder=direction`` pairs."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        parsed: dict[str, str] = {}
        for key, label in value.items():
            path = str(key).replace("\\", "/").strip().strip("/")
            dest = str(label).strip()
            if path and dest:
                parsed[path] = dest
        return parsed
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        if text.startswith("{"):
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise ValueError("DIRECTION_MAP JSON must be an object")
            return parse_direction_map(loaded)
        mapping: dict[str, str] = {}
        for part in text.split(";"):
            item = part.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(
                    "SOURCE__DIRECTION_MAP items must be path=direction, separated by ;"
                )
            path, dest = item.split("=", 1)
            path = path.replace("\\", "/").strip().strip("/")
            dest = dest.strip()
            if path and dest:
                mapping[path] = dest
        return mapping
    raise ValueError("SOURCE__DIRECTION_MAP must be a JSON object or path=direction pairs")


class SourceDirectionMixin(BaseModel):
    """Optional direction labels for Qdrant payload."""

    direction: str = Field(
        default="",
        description=(
            "If set, every file in this profile gets this payload.direction. "
            "Empty keeps the parent-folder name."
        ),
    )
    direction_map: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Relative path or folder prefix → direction. Exact file wins, then the "
            "longest folder prefix, then SOURCE__DIRECTION, then parent folder."
        ),
    )

    @field_validator("direction", mode="before")
    @classmethod
    def _empty_direction_is_blank(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return ""
        return value.strip() if isinstance(value, str) else value

    @field_validator("direction_map", mode="before")
    @classmethod
    def _parse_direction_map(cls, value: Any) -> dict[str, str]:
        return parse_direction_map(value)


class SourceIncludeMixin(BaseModel):
    """Optional filename allowlist relative to the source root."""

    include: list[str] = Field(
        default_factory=list,
        description=(
            "If set, only these filenames or relative paths are copied and indexed. "
            "Empty = every file that matches INDEX_EXTENSIONS."
        ),
    )

    @field_validator("include", mode="before")
    @classmethod
    def _parse_include(cls, value: Any) -> list[str]:
        if value is None or value == "":
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("["):
                loaded = json.loads(text)
                if not isinstance(loaded, list):
                    raise ValueError("SOURCE__INCLUDE JSON must be an array of names")
                value = loaded
            else:
                value = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, list):
            return [
                str(item).replace("\\", "/").strip().strip("/")
                for item in value
                if str(item).strip()
            ]
        raise ValueError("SOURCE__INCLUDE must be a comma-separated list or JSON array")


class QdrantSettings(BaseModel):
    """Connection to one Qdrant instance and collection."""

    model_config = ConfigDict(extra="forbid")

    url: str = "http://127.0.0.1:6333"
    collection: str = "docs"
    timeout_sec: float = 120.0
    extra_payload: dict[str, Any] = Field(default_factory=dict)
    payload_indexes: list[str] | None = None
    distance: Literal["cosine", "dot", "euclid"] = "cosine"
    index_version: str = ""
    prune_missing: bool = Field(
        default=True,
        description=(
            "Delete Qdrant points whose source_path is not in the current source. "
            "Set false to append a new folder into the same collection."
        ),
    )

    @field_validator("prune_missing", mode="before")
    @classmethod
    def _parse_prune_missing(cls, value: Any) -> Any:
        if value is None or value == "":
            return True
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"0", "false", "no", "off"}:
                return False
            if lowered in {"1", "true", "yes", "on"}:
                return True
        return value

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
    # Resume LLM on a bandwidth-bound host (DGX Spark ~10 t/s): long answers take minutes.
    extraction_timeout_sec: float = 1800.0
    extraction_num_ctx: int = 65_536
    extraction_num_predict: int = 8_192
    extraction_think: bool = False
    chunk_size: int = 1024
    picture_description_enabled: bool = True
    vlm_model: str = "qwen3-vl:8b"
    vlm_timeout_sec: float = 90.0
    vlm_concurrency: int = 2
    picture_area_threshold: float = 0.02


class ChunkingSettings(BaseModel):
    """``table_aware``, vanilla Docling ``hybrid``, or ``resume_project`` CVs."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["table_aware", "hybrid", "resume_project"] = "table_aware"
    merge_peers: bool = True
    repeat_table_header: bool = False
    window_chars: int = 1200
    window_overlap: int = 150


class ResumeSettings(BaseModel):
    """LLM steps of the ``resume_project`` strategy (``RESUME__*``)."""

    model_config = ConfigDict(extra="forbid")

    llm_projects: bool = True
    llm_refine: bool = True
    llm_experience: bool = True
    residual_min_chars: int = 1500
    evidence_min_ratio: float = 0.85
    section_max_chars: int = 120_000
    section_overlap_chars: int = 2_000


class LocalSourceSettings(SourceDirectionMixin, SourceIncludeMixin):
    """Watch a local directory with inotify/watchdog events."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["local"] = "local"
    watch_path: str = "/var/lib/document-indexer/docs"
    debounce_seconds: float = 1.0


class SmbSourceSettings(SourceDirectionMixin, SourceIncludeMixin):
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
    max_depth: int | None = Field(
        default=None,
        description=(
            "How many folder levels below SOURCE__SUBPATH to copy. "
            "1 = files in the path plus one nested folder (Проекты/Alpha/file.docx). "
            "Empty = walk the whole tree."
        ),
    )
    port: int = 445
    timeout_sec: float = 30.0
    poll_interval_sec: float = 15.0
    max_backoff_sec: float = 60.0

    @field_validator("max_depth", mode="before")
    @classmethod
    def _empty_max_depth_is_none(cls, value: Any) -> Any:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        return value

    @field_validator("max_depth")
    @classmethod
    def _max_depth_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("max_depth must be >= 0")
        return value


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
    resume: ResumeSettings = Field(default_factory=ResumeSettings)
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

    resume_llm_audit: bool = Field(
        default=False,
        description=(
            "Docling + resume parser + LLM steps, no embeddings or Qdrant. Writes the "
            "report and resume_chunks.jsonl for manual review. Env: RESUME_LLM_AUDIT=1."
        ),
    )

    @field_validator("resume_parse_only", "resume_llm_audit", mode="before")
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
