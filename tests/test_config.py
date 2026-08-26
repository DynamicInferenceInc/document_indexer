"""Tests for typed IndexerSettings."""

from __future__ import annotations

from pydantic import ValidationError

from document_indexer.config import (
    IndexerSettings,
    LocalSourceSettings,
    ModelSettings,
    QdrantSettings,
    SmbSourceSettings,
)


def test_default_local_profile_matches_original_models() -> None:
    settings = IndexerSettings(_env_file=None)
    assert settings.source_type == "local"
    assert settings.watch_path == "/var/lib/document-indexer/docs"
    assert settings.debounce_seconds == 1.0
    assert settings.qdrant_url == "http://127.0.0.1:6333"
    assert settings.qdrant_collection == "docs"
    assert settings.embedding_model == "nomic-embed-text"
    assert settings.chunk_size == 1024
    assert settings.picture_description_enabled is True
    assert settings.vlm_model == "qwen3-vl:8b"
    assert settings.vlm_timeout_sec == 90.0
    assert settings.vlm_concurrency == 2
    assert settings.picture_area_threshold == 0.02
    assert isinstance(settings.source, LocalSourceSettings)


def test_nested_constructor_overrides_qdrant_and_models() -> None:
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path="/data/legal", debounce_seconds=2.0),
        qdrant=QdrantSettings(url="http://qdrant:6333", collection="legal"),
        models=ModelSettings(embedding_model="bge-m3", vlm_model="qwen3-vl:8b"),
    )
    assert settings.watch_path == "/data/legal"
    assert settings.qdrant.collection == "legal"
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.models.embedding_model == "bge-m3"
    assert settings.models.vlm_model == "qwen3-vl:8b"


def test_flat_env_aliases(monkeypatch) -> None:
    monkeypatch.setenv("WATCH_PATH", "/mnt/docs")
    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:6334")
    monkeypatch.setenv("QDRANT_COLLECTION", "hr")
    monkeypatch.setenv("EMBEDDING_MODEL", "nomic-embed-text")
    settings = IndexerSettings(_env_file=None)
    assert settings.watch_path == "/mnt/docs"
    assert settings.qdrant_url == "http://127.0.0.1:6334"
    assert settings.qdrant_collection == "hr"


def test_smb_source_requires_connection_fields() -> None:
    try:
        IndexerSettings(_env_file=None, source_type="smb")
    except ValidationError as exc:
        assert "smb_server" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_smb_nested_constructor_hides_password() -> None:
    settings = IndexerSettings(
        _env_file=None,
        source=SmbSourceSettings(
            server="fileserver",
            share="docs",
            username="svc",
            password="secret-pass",
            staging_path="/var/lib/document-indexer/staging",
            domain="CORP",
            subpath="1c-docs",
        ),
    )
    assert settings.source_type == "smb"
    assert settings.source.server == "fileserver"
    assert settings.source.password.get_secret_value() == "secret-pass"
    dumped = settings.model_dump_safe()
    assert dumped["smb_password"] == "***"
    assert "secret-pass" not in str(dumped)


def test_independent_profiles_do_not_share_qdrant_targets() -> None:
    legal = IndexerSettings(
        _env_file=None,
        qdrant=QdrantSettings(url="http://qdrant:6333", collection="legal"),
    )
    hr = IndexerSettings(
        _env_file=None,
        qdrant=QdrantSettings(url="http://qdrant-b:6333", collection="hr"),
    )
    assert legal.qdrant_collection != hr.qdrant_collection
    assert legal.qdrant_url != hr.qdrant_url
    assert legal.embedding_model == hr.embedding_model
