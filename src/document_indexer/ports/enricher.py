"""Port: extract document-level fields once per file."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from document_indexer.domain.models import DocumentChunk


@runtime_checkable
class DocumentEnricher(Protocol):
    """Return structured fields shared by every chunk of one file."""

    def enrich(self, path: Path, chunks: Sequence[DocumentChunk]) -> dict[str, Any]:
        """Extract canonical fields from ``path`` / ``chunks``. Errors yield ``{}``."""
        ...
