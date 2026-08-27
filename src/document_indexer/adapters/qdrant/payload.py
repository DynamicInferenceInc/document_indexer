"""Payload contract: project fields plus reserved indexer identity keys."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from document_indexer.domain.models import DocumentChunk

DEFAULT_INDEX_VERSION = "table-aware-v2"
RESERVED_PAYLOAD_KEYS = (
    "source_path",
    "chunk_index",
    "file_hash",
    "index_version",
)
DEFAULT_PAYLOAD_INDEXES = (
    "source_path",
    "file_hash",
    "chunk_type",
    "table_ref",
)


@dataclass(frozen=True, slots=True)
class IndexRecord:
    """One chunk ready to become a Qdrant point."""

    source_path: str
    chunk_index: int
    file_hash: str
    chunk: DocumentChunk
    file_path: Path
    document_fields: dict[str, Any] = field(default_factory=dict)
    index_version: str = DEFAULT_INDEX_VERSION


@runtime_checkable
class PayloadBuilder(Protocol):
    """Map an index record to Qdrant payload keys (not reserved identity)."""

    def build(self, record: IndexRecord) -> dict[str, Any]:
        """Return project payload fields for ``record``."""
        ...

    def payload_indexes(self) -> Sequence[str]:
        """Keyword payload fields to index on the collection."""
        ...


class DefaultPayloadBuilder:
    """Current table-aware-v2 payload: text, headings, table fields."""

    def build(self, record: IndexRecord) -> dict[str, Any]:
        chunk = record.chunk
        payload: dict[str, Any] = {
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
        }
        if chunk.headings:
            payload["headings"] = list(chunk.headings)
        if chunk.table_ref:
            payload["table_ref"] = chunk.table_ref
        if chunk.row_count:
            payload["row_count"] = chunk.row_count
        payload.update(record.document_fields)
        return payload

    def payload_indexes(self) -> Sequence[str]:
        return DEFAULT_PAYLOAD_INDEXES


def merge_payload(
    built: Mapping[str, Any],
    extra: Mapping[str, Any],
    record: IndexRecord,
) -> dict[str, Any]:
    """Compose builder output, profile constants, then reserved identity keys."""
    payload = dict(built)
    payload.update(extra)
    payload["source_path"] = record.source_path
    payload["chunk_index"] = record.chunk_index
    payload["file_hash"] = record.file_hash
    payload["index_version"] = record.index_version
    return payload
