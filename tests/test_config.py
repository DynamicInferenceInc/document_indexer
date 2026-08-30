"""Tests for typed IndexerSettings and source profiles."""

from __future__ import annotations

from pydantic import ValidationError

from document_indexer.config import (
    IndexerSettings,
    LocalSourceSettings,
    ModelSettings,
    ProfileLocal,
    ProfileSmb,
    QdrantSettings,
    SmbSourceSettings,
)


def test_default_local_profile_matches_original_models() -> None:
    settings = IndexerSettings(_env_file=None)
    assert settings.source.kind == "local"
    assert isinstance(settings.source, LocalSourceSettings)
    assert settings.source.watch_path == "/var/lib/document-indexer/docs"
    assert settings.source.debounce_seconds == 1.0
    assert settings.qdrant.url == "http://127.0.0.1:6333"
    assert settings.qdrant.collection == "docs"
    assert settings.models.embedding_model == "nomic-embed-text"
    assert settings.models.extraction_model == ""
    assert settings.models.chunk_size == 1024
    assert settings.models.picture_description_enabled is True
    assert settings.models.vlm_model == "qwen3-vl:8b"
    assert settings.models.vlm_timeout_sec == 90.0
    assert settings.models.vlm_concurrency == 2
    assert settings.models.picture_area_threshold == 0.02
    assert settings.chunking.strategy == "table_aware"
    assert settings.chunking.window_chars == 1200


def test_profile_local_is_indexer_settings() -> None:
    settings = ProfileLocal(
        _env_file=None,
        source=LocalSourceSettings(watch_path="/data/legal"),
    )
    assert isinstance(settings, IndexerSettings)
    assert isinstance(settings.source, LocalSourceSettings)
    assert settings.source.watch_path == "/data/legal"


def test_nested_constructor_overrides_qdrant_and_models() -> None:
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path="/data/legal", debounce_seconds=2.0),
        qdrant=QdrantSettings(url="http://qdrant:6333", collection="legal"),
        models=ModelSettings(embedding_model="bge-m3", vlm_model="qwen3-vl:8b"),
    )
    assert settings.source.watch_path == "/data/legal"
    assert settings.source.debounce_seconds == 2.0
    assert settings.qdrant.collection == "legal"
    assert settings.qdrant.url == "http://qdrant:6333"
    assert settings.models.embedding_model == "bge-m3"
    assert settings.models.vlm_model == "qwen3-vl:8b"


def test_nested_env(monkeypatch) -> None:
    monkeypatch.setenv("SOURCE__WATCH_PATH", "/mnt/docs")
    monkeypatch.setenv("QDRANT__URL", "http://127.0.0.1:6334")
    monkeypatch.setenv("QDRANT__COLLECTION", "hr")
    monkeypatch.setenv("MODELS__EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("MODELS__EXTRACTION_MODEL", "qwen3:8b")
    monkeypatch.setenv("QDRANT__EXTRA_PAYLOAD", '{"project":"legal"}')
    monkeypatch.setenv("QDRANT__PAYLOAD_INDEXES", "source_path,grade")
    monkeypatch.setenv("QDRANT__INDEX_VERSION", "resume-v1")
    monkeypatch.setenv("CHUNKING__STRATEGY", "resume_project")
    settings = IndexerSettings(_env_file=None)
    assert settings.source.watch_path == "/mnt/docs"
    assert settings.qdrant.url == "http://127.0.0.1:6334"
    assert settings.qdrant.collection == "hr"
    assert settings.models.embedding_model == "nomic-embed-text"
    assert settings.models.extraction_model == "qwen3:8b"
    assert settings.qdrant.extra_payload == {"project": "legal"}
    assert settings.qdrant.payload_indexes == ["source_path", "grade"]
    assert settings.qdrant.index_version == "resume-v1"
    assert settings.chunking.strategy == "resume_project"


def test_smb_source_requires_connection_fields() -> None:
    try:
        IndexerSettings(_env_file=None, source={"kind": "smb"})
    except ValidationError as exc:
        assert "server" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_profile_smb_requires_source_fields() -> None:
    try:
        ProfileSmb(_env_file=None)
    except ValidationError as exc:
        assert "source" in str(exc)
    else:
        raise AssertionError("expected ValidationError")


def test_smb_nested_constructor_hides_password() -> None:
    settings = ProfileSmb(
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
    assert settings.source.kind == "smb"
    assert isinstance(settings.source, SmbSourceSettings)
    assert settings.source.server == "fileserver"
    assert settings.source.password.get_secret_value() == "secret-pass"
    dumped = settings.model_dump_safe()
    assert dumped["source"]["password"] == "**********"
    assert "secret-pass" not in str(dumped)


def test_independent_profiles_do_not_share_qdrant_targets() -> None:
    legal = ProfileLocal(
        _env_file=None,
        qdrant=QdrantSettings(url="http://qdrant:6333", collection="legal"),
    )
    hr = ProfileLocal(
        _env_file=None,
        qdrant=QdrantSettings(url="http://qdrant-b:6333", collection="hr"),
    )
    assert legal.qdrant.collection != hr.qdrant.collection
    assert legal.qdrant.url != hr.qdrant.url
    assert legal.models.embedding_model == hr.models.embedding_model
