"""Typed configuration for a single indexer profile."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseModel):
    """Connection to one Qdrant instance and collection."""

    model_config = ConfigDict(extra="forbid")

    url: str = "http://127.0.0.1:6333"
    collection: str = "docs"


class ModelSettings(BaseModel):
    """Embedding, VLM and chunker settings. Defaults match the original reindex."""

    model_config = ConfigDict(extra="forbid")

    ollama_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_sec: float = 120.0
    chunk_size: int = 1024
    picture_description_enabled: bool = True
    vlm_model: str = "qwen3-vl:8b"
    vlm_timeout_sec: float = 90.0
    vlm_concurrency: int = 2
    picture_area_threshold: float = 0.02


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
    staging_path: str
    domain: str | None = None
    subpath: str = ""
    port: int = 445
    timeout_sec: float = 30.0
    poll_interval_sec: float = 15.0
    max_backoff_sec: float = 60.0


_DEFAULT_EXTENSIONS = (
    ".txt,.md,.markdown,.rst,.log,.csv,.pdf,.docx,.pptx,.xlsx,.xls,.html,.htm"
)


class IndexerSettings(BaseSettings):
    """One indexer profile: source, Qdrant target, models, extensions.

    Construct from nested objects in Python::

        IndexerSettings(
            source=SmbSourceSettings(...),
            qdrant=QdrantSettings(url="http://qdrant:6333", collection="legal"),
            models=ModelSettings(embedding_model="nomic-embed-text"),
        )

    Flat names work the same way as constructor kwargs and as env / ``.env``
    keys (``QDRANT_URL``, ``WATCH_PATH``, ``SMB_PASSWORD``, …). Passwords are
    ``SecretStr`` and are redacted by :meth:`model_dump_safe`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    source_type: Literal["local", "smb"] = Field(
        default="local",
        validation_alias=AliasChoices("source_type", "SOURCE_TYPE"),
    )
    watch_path: str = Field(
        default="/var/lib/document-indexer/docs",
        validation_alias=AliasChoices("watch_path", "WATCH_PATH"),
    )
    debounce_seconds: float = Field(
        default=1.0,
        validation_alias=AliasChoices("debounce_seconds", "DEBOUNCE_SECONDS"),
    )
    smb_server: str | None = Field(
        default=None,
        validation_alias=AliasChoices("smb_server", "SMB_SERVER"),
    )
    smb_share: str | None = Field(
        default=None,
        validation_alias=AliasChoices("smb_share", "SMB_SHARE"),
    )
    smb_username: str | None = Field(
        default=None,
        validation_alias=AliasChoices("smb_username", "SMB_USERNAME"),
    )
    smb_password: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("smb_password", "SMB_PASSWORD"),
    )
    smb_domain: str | None = Field(
        default=None,
        validation_alias=AliasChoices("smb_domain", "SMB_DOMAIN"),
    )
    smb_subpath: str = Field(
        default="",
        validation_alias=AliasChoices("smb_subpath", "SMB_SUBPATH"),
    )
    smb_port: int = Field(
        default=445,
        validation_alias=AliasChoices("smb_port", "SMB_PORT"),
    )
    smb_timeout_sec: float = Field(
        default=30.0,
        validation_alias=AliasChoices("smb_timeout_sec", "SMB_TIMEOUT_SEC"),
    )
    smb_poll_interval_sec: float = Field(
        default=15.0,
        validation_alias=AliasChoices("smb_poll_interval_sec", "SMB_POLL_INTERVAL_SEC"),
    )
    smb_staging_path: str = Field(
        default="/var/lib/document-indexer/staging",
        validation_alias=AliasChoices("smb_staging_path", "SMB_STAGING_PATH"),
    )
    smb_max_backoff_sec: float = Field(
        default=60.0,
        validation_alias=AliasChoices("smb_max_backoff_sec", "SMB_MAX_BACKOFF_SEC"),
    )
    qdrant_url: str = Field(
        default="http://127.0.0.1:6333",
        validation_alias=AliasChoices("qdrant_url", "QDRANT_URL"),
    )
    qdrant_collection: str = Field(
        default="docs",
        validation_alias=AliasChoices("qdrant_collection", "QDRANT_COLLECTION"),
    )
    ollama_base_url: str = Field(
        default="http://127.0.0.1:11434",
        validation_alias=AliasChoices("ollama_base_url", "OLLAMA_BASE_URL"),
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        validation_alias=AliasChoices("embedding_model", "EMBEDDING_MODEL"),
    )
    embedding_timeout_sec: float = Field(
        default=120.0,
        validation_alias=AliasChoices("embedding_timeout_sec", "EMBEDDING_TIMEOUT_SEC"),
    )
    chunk_size: int = Field(
        default=1024,
        validation_alias=AliasChoices("chunk_size", "CHUNK_SIZE"),
    )
    picture_description_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "picture_description_enabled",
            "PICTURE_DESCRIPTION_ENABLED",
        ),
    )
    vlm_model: str = Field(
        default="qwen3-vl:8b",
        validation_alias=AliasChoices("vlm_model", "VLM_MODEL"),
    )
    vlm_timeout_sec: float = Field(
        default=90.0,
        validation_alias=AliasChoices("vlm_timeout_sec", "VLM_TIMEOUT_SEC"),
    )
    vlm_concurrency: int = Field(
        default=2,
        validation_alias=AliasChoices("vlm_concurrency", "VLM_CONCURRENCY"),
    )
    picture_area_threshold: float = Field(
        default=0.02,
        validation_alias=AliasChoices(
            "picture_area_threshold",
            "PICTURE_AREA_THRESHOLD",
        ),
    )
    index_extensions: str = Field(
        default=_DEFAULT_EXTENSIONS,
        validation_alias=AliasChoices("index_extensions", "INDEX_EXTENSIONS"),
    )
    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("log_level", "LOG_LEVEL"),
    )

    @model_validator(mode="before")
    @classmethod
    def _expand_nested(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        expanded = dict(data)
        source = expanded.pop("source", None)
        if source is not None:
            payload = (
                source.model_dump()
                if isinstance(source, (LocalSourceSettings, SmbSourceSettings))
                else dict(source)
            )
            kind = str(payload.get("kind") or payload.get("type") or "local").lower()
            expanded.setdefault("source_type", kind)
            if kind == "smb":
                _copy_if_absent(expanded, "smb_server", payload.get("server"))
                _copy_if_absent(expanded, "smb_share", payload.get("share"))
                _copy_if_absent(expanded, "smb_username", payload.get("username"))
                _copy_if_absent(expanded, "smb_password", payload.get("password"))
                _copy_if_absent(expanded, "smb_staging_path", payload.get("staging_path"))
                _copy_if_absent(expanded, "smb_domain", payload.get("domain"))
                _copy_if_absent(expanded, "smb_subpath", payload.get("subpath"))
                _copy_if_absent(expanded, "smb_port", payload.get("port"))
                _copy_if_absent(expanded, "smb_timeout_sec", payload.get("timeout_sec"))
                _copy_if_absent(
                    expanded,
                    "smb_poll_interval_sec",
                    payload.get("poll_interval_sec"),
                )
                _copy_if_absent(
                    expanded,
                    "smb_max_backoff_sec",
                    payload.get("max_backoff_sec"),
                )
            else:
                _copy_if_absent(expanded, "watch_path", payload.get("watch_path"))
                _copy_if_absent(
                    expanded,
                    "debounce_seconds",
                    payload.get("debounce_seconds"),
                )

        qdrant = expanded.pop("qdrant", None)
        if qdrant is not None:
            payload = qdrant.model_dump() if isinstance(qdrant, QdrantSettings) else dict(qdrant)
            _copy_if_absent(expanded, "qdrant_url", payload.get("url"))
            _copy_if_absent(expanded, "qdrant_collection", payload.get("collection"))

        models = expanded.pop("models", None)
        if models is not None:
            payload = models.model_dump() if isinstance(models, ModelSettings) else dict(models)
            for key in (
                "ollama_base_url",
                "embedding_model",
                "embedding_timeout_sec",
                "chunk_size",
                "picture_description_enabled",
                "vlm_model",
                "vlm_timeout_sec",
                "vlm_concurrency",
                "picture_area_threshold",
            ):
                _copy_if_absent(expanded, key, payload.get(key))
        return expanded

    @model_validator(mode="after")
    def _require_smb_fields(self) -> IndexerSettings:
        if self.source_type != "smb":
            return self
        missing = [
            name
            for name, value in (
                ("smb_server", self.smb_server),
                ("smb_share", self.smb_share),
                ("smb_username", self.smb_username),
                ("smb_password", self.smb_password),
                ("smb_staging_path", self.smb_staging_path),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "SMB source requires smb_server, smb_share, smb_username, "
                f"smb_password and smb_staging_path; missing: {missing}"
            )
        return self

    @property
    def source(self) -> LocalSourceSettings | SmbSourceSettings:
        if self.source_type == "smb":
            assert self.smb_server and self.smb_share and self.smb_username
            assert self.smb_password is not None
            return SmbSourceSettings(
                server=self.smb_server,
                share=self.smb_share,
                username=self.smb_username,
                password=self.smb_password,
                staging_path=self.smb_staging_path,
                domain=self.smb_domain,
                subpath=self.smb_subpath,
                port=self.smb_port,
                timeout_sec=self.smb_timeout_sec,
                poll_interval_sec=self.smb_poll_interval_sec,
                max_backoff_sec=self.smb_max_backoff_sec,
            )
        return LocalSourceSettings(
            watch_path=self.watch_path,
            debounce_seconds=self.debounce_seconds,
        )

    @property
    def qdrant(self) -> QdrantSettings:
        return QdrantSettings(url=self.qdrant_url, collection=self.qdrant_collection)

    @property
    def models(self) -> ModelSettings:
        return ModelSettings(
            ollama_base_url=self.ollama_base_url,
            embedding_model=self.embedding_model,
            embedding_timeout_sec=self.embedding_timeout_sec,
            chunk_size=self.chunk_size,
            picture_description_enabled=self.picture_description_enabled,
            vlm_model=self.vlm_model,
            vlm_timeout_sec=self.vlm_timeout_sec,
            vlm_concurrency=self.vlm_concurrency,
            picture_area_threshold=self.picture_area_threshold,
        )

    @property
    def watch_root(self) -> str:
        """Local path the Qdrant indexer reads from (watch dir or SMB staging)."""
        source = self.source
        if isinstance(source, SmbSourceSettings):
            return source.staging_path
        return source.watch_path

    def model_dump_safe(self) -> dict[str, Any]:
        """Serialize settings without revealing secrets."""
        dumped = self.model_dump(mode="json")
        if dumped.get("smb_password"):
            dumped["smb_password"] = "***"
        return dumped


def _copy_if_absent(data: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if key not in data or data[key] is None:
        data[key] = value
