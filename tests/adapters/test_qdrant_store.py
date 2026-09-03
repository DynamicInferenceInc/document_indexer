from unittest.mock import MagicMock

from qdrant_client.http import models as qmodels

from document_indexer.adapters.qdrant.store import QdrantStore, _encoded_point_size


def _point(point_id: int, text: str) -> qmodels.PointStruct:
    return qmodels.PointStruct(
        id=point_id,
        vector=[0.1, 0.2, 0.3],
        payload={"text": text, "source_path": "cv.docx"},
    )


def test_upsert_sends_empty_list_as_noop() -> None:
    client = MagicMock()
    store = QdrantStore(url="http://qdrant", collection="docs", client=client)
    store.upsert([])
    client.upsert.assert_not_called()


def test_upsert_keeps_small_batch_in_one_request() -> None:
    client = MagicMock()
    store = QdrantStore(url="http://qdrant", collection="docs", client=client)
    points = [_point(1, "hello")]
    store.upsert(points)
    client.upsert.assert_called_once_with(collection_name="docs", points=points)


def test_upsert_splits_when_json_exceeds_limit() -> None:
    points = [_point(index, "payload-" * 20) for index in range(6)]
    point_size = _encoded_point_size(points[0])
    client = MagicMock()
    store = QdrantStore(
        url="http://qdrant",
        collection="docs",
        client=client,
        max_upsert_json_bytes=point_size * 2 + 10,
    )
    store.upsert(points)
    assert client.upsert.call_count >= 3
    sent_ids = [
        point.id
        for call in client.upsert.call_args_list
        for point in call.kwargs["points"]
    ]
    assert sent_ids == [point.id for point in points]
    assert all(len(call.kwargs["points"]) <= 2 for call in client.upsert.call_args_list)
