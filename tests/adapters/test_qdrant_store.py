from unittest.mock import MagicMock

import pytest

from document_indexer.adapters.qdrant.store import (
    QdrantStore,
    _iter_upsert_batches,
)


def _point(size: int = 8) -> MagicMock:
    point = MagicMock()
    point.model_dump_json.return_value = "x" * size
    return point


def test_iter_upsert_batches_splits_by_point_count() -> None:
    points = [_point() for _ in range(5)]
    batches = list(_iter_upsert_batches(points, max_bytes=10_000, max_points=2))
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_iter_upsert_batches_splits_by_json_size() -> None:
    points = [_point(size=6) for _ in range(3)]
    batches = list(_iter_upsert_batches(points, max_bytes=10, max_points=10))
    assert [len(batch) for batch in batches] == [1, 1, 1]


def test_upsert_retries_timeout_then_succeeds(monkeypatch) -> None:
    client = MagicMock()
    client.upsert.side_effect = [TimeoutError("timed out"), None]
    store = QdrantStore(
        url="http://127.0.0.1:6333",
        collection="docs",
        client=client,
        max_upsert_points=10,
    )
    monkeypatch.setattr("document_indexer.adapters.qdrant.store.time.sleep", lambda _delay: None)
    store.upsert([_point(), _point()])
    assert client.upsert.call_count == 2


def test_upsert_gives_up_after_retries(monkeypatch) -> None:
    client = MagicMock()
    client.upsert.side_effect = TimeoutError("timed out")
    store = QdrantStore(
        url="http://127.0.0.1:6333",
        collection="docs",
        client=client,
        max_upsert_points=10,
    )
    monkeypatch.setattr("document_indexer.adapters.qdrant.store.time.sleep", lambda _delay: None)
    with pytest.raises(TimeoutError):
        store.upsert([_point()])
    assert client.upsert.call_count == 4
