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
    "direction",
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

    def __init__(
        self,
        *,
        default_direction: str = "",
        direction_map: Mapping[str, str] | None = None,
    ) -> None:
        self._default_direction = default_direction.strip()
        self._direction_map = {
            str(key).replace("\\", "/").strip("/"): str(label).strip()
            for key, label in (direction_map or {}).items()
            if str(key).strip() and str(label).strip()
        }

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
        direction = resolve_direction(
            record.source_path,
            default=self._default_direction,
            mapping=self._direction_map,
        )
        if direction:
            payload["direction"] = direction
        payload.update(record.document_fields)
        return payload

    def payload_indexes(self) -> Sequence[str]:
        return DEFAULT_PAYLOAD_INDEXES

    def hash_salt(self) -> str:
        """Include direction config in the file hash so label changes reindex."""
        parts = [self._default_direction]
        for key in sorted(self._direction_map):
            parts.append(f"{key}={self._direction_map[key]}")
        return "\n".join(parts)


def direction_from_source_path(source_path: str) -> str:
    """Folder that contains the file, relative to the indexed root.

    ``Бухгалтерия/акт.docx`` → ``Бухгалтерия``. A file sitting in the root
    (``акт.docx``) has no folder name.
    """
    normalized = source_path.replace("\\", "/").strip("/")
    if not normalized:
        return ""
    parent = Path(normalized).parent
    name = parent.name.strip()
    if not name or name == ".":
        return ""
    return name


def resolve_direction(
    source_path: str,
    *,
    default: str = "",
    mapping: Mapping[str, str] | None = None,
) -> str:
    """Manual map (exact file, then longest folder prefix), else default, else parent folder."""
    normalized = source_path.replace("\\", "/").strip("/")
    mapping = mapping or {}
    if normalized in mapping:
        return mapping[normalized]
    best_key = ""
    best_value = ""
    for key, label in mapping.items():
        prefix = str(key).replace("\\", "/").strip("/")
        if not prefix or not str(label).strip():
            continue
        if normalized == prefix or normalized.startswith(f"{prefix}/"):
            if len(prefix) >= len(best_key):
                best_key = prefix
                best_value = str(label).strip()
    if best_value:
        return best_value
    if default.strip():
        return default.strip()
    return direction_from_source_path(normalized)


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
