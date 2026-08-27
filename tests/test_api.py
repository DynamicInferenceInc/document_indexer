"""Public API tests for DocumentIndexer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from document_indexer import (
    DocumentIndexer,
    IndexerSettings,
    JsonSchemaEnricher,
    LocalSourceSettings,
    PayloadBuilder,
    ProfileLocal,
    ProfileSmb,
    QdrantSettings,
    reindex_once,
    run,
)
from document_indexer.domain.changes import FsChange
from document_indexer.ports import Indexer


class RecordingIndexer(Indexer):
    def __init__(self) -> None:
        self.reindex_calls: list[str] = []
        self.applied: list[list[FsChange]] = []

    def index(
        self,
        watch_path: str,
        changes: Sequence[FsChange] | None = None,
    ) -> None:
        if changes is None:
            self.reindex_calls.append(watch_path)
            return
        self.applied.append(list(changes))


def test_public_exports() -> None:
    assert callable(reindex_once)
    assert callable(run)
    assert callable(DocumentIndexer)
    assert issubclass(ProfileLocal, IndexerSettings)
    assert issubclass(ProfileSmb, IndexerSettings)
    assert JsonSchemaEnricher is not None
    assert PayloadBuilder is not None


def test_reindex_once_uses_watch_path(tmp_path: Path) -> None:
    docs = tmp_path / "db"
    docs.mkdir()
    indexer = RecordingIndexer()
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path=str(docs)),
    )
    DocumentIndexer(
        settings,
        indexer=indexer,
        configure_logs=False,
    ).reindex_once()
    assert indexer.reindex_calls == [str(docs)]


def test_reindex_once_fails_when_watch_path_missing(tmp_path: Path) -> None:
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path=str(tmp_path / "missing")),
    )
    with pytest.raises(FileNotFoundError, match="watch_path"):
        DocumentIndexer(settings, indexer=RecordingIndexer(), configure_logs=False).reindex_once()


def test_two_instances_keep_separate_collections(tmp_path: Path) -> None:
    docs = tmp_path / "db"
    docs.mkdir()
    first = RecordingIndexer()
    second = RecordingIndexer()
    a = DocumentIndexer(
        IndexerSettings(
            _env_file=None,
            source=LocalSourceSettings(watch_path=str(docs)),
            qdrant=QdrantSettings(collection="legal"),
        ),
        indexer=first,
        configure_logs=False,
    )
    b = DocumentIndexer(
        IndexerSettings(
            _env_file=None,
            source=LocalSourceSettings(watch_path=str(docs)),
            qdrant=QdrantSettings(collection="hr"),
        ),
        indexer=second,
        configure_logs=False,
    )
    a.reindex_once()
    b.reindex_once()
    assert a.settings.qdrant.collection == "legal"
    assert b.settings.qdrant.collection == "hr"
    assert first.reindex_calls == [str(docs)]
    assert second.reindex_calls == [str(docs)]
