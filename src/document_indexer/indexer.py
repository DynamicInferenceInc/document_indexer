"""Public composition root for one indexer profile."""

from __future__ import annotations

import logging
import signal
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter

from document_indexer.adapters.document_readers import DoclingDocumentReader, PictureDescriptionConfig
from document_indexer.adapters.docling_chunking import TableAwareDocumentChunker
from document_indexer.adapters.docling_convert import (
    MarkdownChunkSerializerProvider,
    tokenizer_with_max_tokens,
)
from document_indexer.adapters.qdrant.payload import DEFAULT_INDEX_VERSION, PayloadBuilder
from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.config import IndexerSettings, LocalSourceSettings, SmbSourceSettings
from document_indexer.domain.changes import FsChange
from document_indexer.domain.documents import resolve_index_extensions
from document_indexer.infra.embeddings import OllamaEmbedder
from document_indexer.infra.logging_config import configure_logging
from document_indexer.ports import Indexer
from document_indexer.ports.chunker import DocumentChunker
from document_indexer.ports.enricher import DocumentEnricher
from document_indexer.sources.base import DocumentSource
from document_indexer.sources.debounce import DebouncedReindex
from document_indexer.sources.local import LocalFilesystemSource
from document_indexer.sources.smb import SmbStagingSource

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Index documents from one configured source into one Qdrant collection.

    One instance is one profile. Create several instances (or processes) for
    independent sources, collections, or model overrides.
    """

    def __init__(
        self,
        settings: IndexerSettings | None = None,
        *,
        indexer: Indexer | None = None,
        source: DocumentSource | None = None,
        payload_builder: PayloadBuilder | None = None,
        enricher: DocumentEnricher | None = None,
        document_chunker: DocumentChunker | None = None,
        configure_logs: bool = True,
    ) -> None:
        self._settings = settings if settings is not None else IndexerSettings()
        if configure_logs:
            configure_logging(self._settings.log_level)
        allowed = _log_extensions(self._settings)
        self._core = indexer if indexer is not None else build_indexer(
            self._settings,
            payload_builder=payload_builder,
            enricher=enricher,
            document_chunker=document_chunker,
        )
        self._source = source if source is not None else build_source(
            self._settings,
            allowed_extensions=allowed,
        )
        self._stop = threading.Event()
        self._debouncer: DebouncedReindex | None = None
        self._signals_installed = False

    @property
    def settings(self) -> IndexerSettings:
        return self._settings

    def reindex_once(self) -> None:
        """Sync the source if needed and run a full Qdrant reconcile."""
        root = self._source.prepare()
        logger.info(
            "Starting one-shot reindex on %s (qdrant=%s collection=%s)",
            root,
            self._settings.qdrant.url,
            self._settings.qdrant.collection,
        )
        self._core.index(str(root))
        logger.info("One-shot reindex complete path=%s", root)

    def run(self) -> None:
        """Run an initial reconcile, then watch or poll until :meth:`stop`."""
        self._stop.clear()
        self._install_signal_handlers()
        root = self._prepare_with_retry()
        logger.info(
            "Starting indexer on %s (qdrant=%s collection=%s source=%s)",
            root,
            self._settings.qdrant.url,
            self._settings.qdrant.collection,
            self._settings.source.kind,
        )
        try:
            self._core.index(str(root))
        except Exception:
            logger.exception("Initial reindex failed for %s", root)
            if self._stop.is_set():
                return
            raise

        source = self._settings.source
        if isinstance(source, LocalSourceSettings):
            self._debouncer = DebouncedReindex(
                indexer=self._core,
                watch_path=str(root),
                debounce_seconds=source.debounce_seconds,
            )
            self._source.start(self._notify_debounced)
        else:
            self._source.start(self._apply_immediately)

        try:
            while not self._stop.wait(timeout=1.0):
                pass
        except KeyboardInterrupt:
            self.stop()
        finally:
            self._shutdown_source()
            logger.info("Indexer stopped")

    def stop(self) -> None:
        """Request a graceful shutdown of :meth:`run`."""
        logger.info("Shutting down")
        self._stop.set()

    def _notify_debounced(self, changes: Sequence[FsChange]) -> None:
        if self._debouncer is None:
            return
        for change in changes:
            self._debouncer.notify(change)

    def _apply_immediately(self, changes: Sequence[FsChange]) -> None:
        if not changes:
            return
        try:
            self._core.index(str(self._source.local_root()), changes)
        except Exception:
            logger.exception(
                "Indexer.index failed for %s",
                self._source.local_root(),
            )

    def _prepare_with_retry(self) -> Path:
        delay = 1.0
        while not self._stop.is_set():
            try:
                return self._source.prepare()
            except FileNotFoundError:
                raise
            except Exception:
                cap = 60.0
                if isinstance(self._settings.source, SmbSourceSettings):
                    cap = self._settings.source.max_backoff_sec
                delay = min(cap, delay * 2)
                logger.exception(
                    "Source prepare failed; retrying in %.1fs",
                    delay,
                )
                if self._stop.wait(timeout=delay):
                    break
        raise RuntimeError("Indexer stopped before the source became available")

    def _install_signal_handlers(self) -> None:
        if self._signals_installed:
            return

        def _shutdown(signum: int | None = None, _frame: object = None) -> None:
            if signum is not None:
                logger.info("Received signal %s; shutting down", signum)
            self.stop()

        try:
            signal.signal(signal.SIGTERM, _shutdown)
            signal.signal(signal.SIGINT, _shutdown)
            self._signals_installed = True
        except ValueError:
            # Signals can only be set from the main thread.
            logger.debug("Skipping signal handlers outside the main thread")

    def _shutdown_source(self) -> None:
        if self._debouncer is not None:
            self._debouncer.cancel()
            self._debouncer = None
        self._source.stop()


def build_indexer(
    settings: IndexerSettings,
    *,
    payload_builder: PayloadBuilder | None = None,
    enricher: DocumentEnricher | None = None,
    document_chunker: DocumentChunker | None = None,
) -> Indexer:
    models = settings.models
    chunking = settings.chunking
    allowed = resolve_index_extensions(settings.index_extensions)
    if models.picture_description_enabled:
        logger.info(
            "Picture description enabled model=%s ollama=%s concurrency=%s "
            "timeout=%ss threshold=%.3f chunk_size=%s",
            models.vlm_model,
            models.ollama_base_url,
            models.vlm_concurrency,
            models.vlm_timeout_sec,
            models.picture_area_threshold,
            models.chunk_size,
        )
    embedder = OllamaEmbedder(
        base_url=models.ollama_base_url,
        model=models.embedding_model,
        timeout_sec=models.embedding_timeout_sec,
    )
    picture = PictureDescriptionConfig(
        enabled=models.picture_description_enabled,
        ollama_base_url=models.ollama_base_url,
        model=models.vlm_model,
        timeout_sec=models.vlm_timeout_sec,
        concurrency=models.vlm_concurrency,
        area_threshold=models.picture_area_threshold,
    )
    tokenizer = tokenizer_with_max_tokens(models.chunk_size)
    hybrid = HybridChunker(
        merge_peers=chunking.merge_peers,
        # Tables are rendered from TableItem; repeating headers makes fragments.
        repeat_table_header=chunking.repeat_table_header,
        tokenizer=tokenizer,
        serializer_provider=MarkdownChunkSerializerProvider(),
    )
    if document_chunker is None:
        document_chunker = _build_document_chunker(settings, hybrid, tokenizer)
    if settings.chunking.strategy == "resume_project":
        from document_indexer.examples.resume.enricher import bind_resume_enricher

        enricher = bind_resume_enricher(settings, enricher)
    reader = DoclingDocumentReader(
        DocumentConverter(format_options=picture.format_options()),
        hybrid,
        document_chunker=document_chunker,
        max_tokens=models.chunk_size,
        tokenizer=tokenizer,
        picture=picture,
    )
    builder_version = getattr(payload_builder, "index_version", "") if payload_builder is not None else ""
    index_version = settings.qdrant.index_version or builder_version or DEFAULT_INDEX_VERSION
    return QdrantIndexer(
        qdrant_url=settings.qdrant.url,
        collection=settings.qdrant.collection,
        embedder=embedder,
        document_reader=reader,
        allowed_extensions=allowed,
        payload_builder=payload_builder,
        enricher=enricher,
        extra_payload=settings.qdrant.extra_payload,
        payload_indexes=settings.qdrant.payload_indexes,
        distance=settings.qdrant.distance,
        index_version=index_version,
    )


def _build_document_chunker(
    settings: IndexerSettings,
    hybrid: Any,
    tokenizer: Any,
) -> DocumentChunker:
    chunking = settings.chunking
    if chunking.strategy == "resume_project":
        from document_indexer.examples.resume.chunker import ResumeProjectChunker

        return ResumeProjectChunker(
            window_chars=chunking.window_chars,
            window_overlap=chunking.window_overlap,
        )
    return TableAwareDocumentChunker(
        chunker=hybrid,
        max_tokens=settings.models.chunk_size,
        tokenizer=tokenizer,
    )


def build_source(
    settings: IndexerSettings,
    *,
    allowed_extensions: frozenset[str] | None = None,
) -> DocumentSource:
    source = settings.source
    if isinstance(source, SmbSourceSettings):
        return SmbStagingSource(
            source,
            allowed_extensions=allowed_extensions,
        )
    return LocalFilesystemSource(source)


def reindex_once(settings: IndexerSettings | None = None) -> None:
    """Convenience wrapper around :meth:`DocumentIndexer.reindex_once`."""
    DocumentIndexer(settings).reindex_once()


def run(settings: IndexerSettings | None = None) -> None:
    """Convenience wrapper around :meth:`DocumentIndexer.run`."""
    DocumentIndexer(settings).run()


def _log_extensions(settings: IndexerSettings) -> frozenset[str]:
    allowed = resolve_index_extensions(settings.index_extensions)
    logger.info("Index extensions enabled: %s", sorted(allowed))
    return allowed
