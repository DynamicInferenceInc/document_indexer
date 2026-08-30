"""Port: split a converted document into embeddable chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from document_indexer.domain.models import DocumentChunk


@runtime_checkable
class DocumentChunker(Protocol):
    """Turn a converted document into domain chunks."""

    def chunk_document(self, document: Any, *, path_name: str) -> Sequence[DocumentChunk]:
        """Return chunks extracted from ``document``."""
        ...
