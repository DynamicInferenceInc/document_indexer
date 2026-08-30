"""Domain types for the reindex service (no infrastructure imports)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One semantically coherent piece of a source document, ready to embed."""

    text: str
    headings: tuple[str, ...] = ()
    atomic: bool = False
    chunk_type: str = "prose"
    table_ref: str | None = None
    embedding_parts: tuple[str, ...] = ()
    row_count: int = 0
    extra_fields: dict[str, Any] = field(default_factory=dict)
