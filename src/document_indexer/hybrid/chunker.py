"""Vanilla Docling HybridChunker: whatever Docling emits becomes a Qdrant point."""

from __future__ import annotations

import logging
import time
from typing import Any

from document_indexer.domain.models import DocumentChunk

logger = logging.getLogger(__name__)

HYBRID_INDEX_VERSION = "hybrid-v3"


class HybridDocumentChunker:
    """Pass HybridChunker output through with headings; no table post-processing."""

    def __init__(self, *, chunker: Any) -> None:
        self._chunker = chunker

    def chunk_document(self, document: Any, *, path_name: str) -> list[DocumentChunk]:
        started = time.perf_counter()
        chunks: list[DocumentChunk] = []
        raw_count = 0
        for raw in self._chunker.chunk(dl_doc=document):
            raw_count += 1
            text = str(self._chunker.contextualize(raw)).strip()
            if not text:
                continue
            chunks.append(
                DocumentChunk(
                    text=text,
                    headings=_headings_from_chunk(raw),
                    chunk_type="prose",
                )
            )
        logger.info(
            "Hybrid chunker path=%s raw=%s stored=%s tokenizer_max_tokens=%s elapsed=%.2fs",
            path_name,
            raw_count,
            len(chunks),
            _tokenizer_max_tokens(self._chunker),
            time.perf_counter() - started,
        )
        return chunks


def _tokenizer_max_tokens(chunker: Any) -> int | str:
    tokenizer = getattr(chunker, "tokenizer", None)
    getter = getattr(tokenizer, "get_max_tokens", None)
    if callable(getter):
        return getter()
    return getattr(tokenizer, "max_tokens", "?")


def _headings_from_chunk(chunk: Any) -> tuple[str, ...]:
    headings = getattr(getattr(chunk, "meta", None), "headings", None) or ()
    return tuple(str(item) for item in headings if item)
