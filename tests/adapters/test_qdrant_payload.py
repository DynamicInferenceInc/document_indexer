"""Payload merge, extra_payload, and reserved identity keys."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from document_indexer.adapters.qdrant.payload import DefaultPayloadBuilder, IndexRecord, merge_payload
from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.domain.models import DocumentChunk


class FakeEmbedder:
    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return [1.0, 0.0]
        return [[1.0, 0.0] for _ in text]


class FakeReader:
    def read(self, path: Path) -> list[DocumentChunk]:
        return [DocumentChunk(text=path.read_text(encoding="utf-8"), headings=("H",))]


class OverwriteBuilder:
    def build(self, record: IndexRecord) -> dict:
        return {
            "text": record.chunk.text,
            "source_path": "hijacked",
            "file_hash": "nope",
            "chunk_index": 99,
            "index_version": "evil",
            "grade": "Senior",
        }

    def payload_indexes(self):
        return ("source_path", "grade")


def _mock_client(*, exists: bool = False) -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = exists
    client.scroll.return_value = ([], None)
    return client


def test_merge_payload_reserved_keys_win() -> None:
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="hi"),
        file_path=Path("cv.md"),
        document_fields={"grade": "Senior"},
        index_version="resume-v1",
    )
    merged = merge_payload(
        {"source_path": "x", "text": "hi", "grade": "Senior"},
        {"project": "cv"},
        record,
    )
    assert merged["source_path"] == "cv.md"
    assert merged["chunk_index"] == 0
    assert merged["file_hash"] == "abc"
    assert merged["index_version"] == "resume-v1"
    assert merged["text"] == "hi"
    assert merged["grade"] == "Senior"
    assert merged["project"] == "cv"


def test_default_builder_matches_table_aware_fields() -> None:
    record = IndexRecord(
        source_path="guide.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="body", headings=("Intro",), chunk_type="prose"),
        file_path=Path("guide.md"),
    )
    built = DefaultPayloadBuilder().build(record)
    payload = merge_payload(built, {}, record)
    assert payload["text"] == "body"
    assert payload["headings"] == ["Intro"]
    assert payload["chunk_type"] == "prose"
    assert payload["source_path"] == "guide.md"
    assert "grade" not in payload


def test_extra_payload_and_custom_builder_keep_reserved(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cv.md").write_text("Звание старший", encoding="utf-8")
    client = _mock_client()
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs-cv",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
        payload_builder=OverwriteBuilder(),
        extra_payload={"kb": "resume"},
        index_version="resume-v1",
    )
    indexer._client = client
    indexer.index(str(docs))
    points = client.upsert.call_args.kwargs["points"]
    payload = points[0].payload
    assert payload["source_path"] == "cv.md"
    assert payload["chunk_index"] == 0
    assert payload["index_version"] == "resume-v1"
    assert payload["file_hash"] != "nope"
    assert len(payload["file_hash"]) == 64
    assert payload["grade"] == "Senior"
    assert payload["kb"] == "resume"
    assert payload["text"] == "Звание старший"


def test_custom_payload_indexes_are_created(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cv.md").write_text("text", encoding="utf-8")
    client = _mock_client()
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs-cv",
        embedder=FakeEmbedder(),
        document_reader=FakeReader(),
        payload_builder=OverwriteBuilder(),
    )
    indexer._client = client
    indexer.index(str(docs))
    names = [
        call.kwargs["field_name"]
        for call in client.create_payload_index.call_args_list
    ]
    assert "grade" in names
    assert "source_path" in names
