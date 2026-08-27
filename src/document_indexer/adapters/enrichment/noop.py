"""Enricher that adds no document-level fields."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.domain.models import DocumentChunk


class NoopEnricher:
    """Default enricher: empty document_fields, current payload unchanged."""

    def enrich(self, path: Path, chunks: Sequence[DocumentChunk]) -> dict[str, Any]:
        del path, chunks
        return {}
