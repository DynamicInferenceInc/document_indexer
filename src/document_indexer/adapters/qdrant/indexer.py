"""Index watched documents into Qdrant via embeddings."""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, overload

from qdrant_client.http import models as qmodels

from document_indexer.adapters.docling_chunking import is_useful_chunk_text
from document_indexer.adapters.enrichment.noop import NoopEnricher
from document_indexer.adapters.qdrant.payload import (
    DEFAULT_INDEX_VERSION,
    DefaultPayloadBuilder,
    IndexRecord,
    PayloadBuilder,
    merge_payload,
)
from document_indexer.adapters.qdrant.store import QdrantStore
from document_indexer.domain.changes import FsChange
from document_indexer.domain.documents import iter_document_files
from document_indexer.ports import DocumentReader, Embedder
from document_indexer.ports.enricher import DocumentEnricher

logger = logging.getLogger(__name__)

_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
_HASH_CHUNK_SIZE = 1024 * 1024
_INDEX_VERSION = DEFAULT_INDEX_VERSION


class QdrantIndexer:
    """Index documents into Qdrant with content-hash skip and duplicate dedup."""

    def __init__(
        self,
        *,
        qdrant_url: str,
        collection: str,
        embedder: Embedder,
        document_reader: DocumentReader,
        allowed_extensions: frozenset[str] | set[str] | None = None,
        payload_builder: PayloadBuilder | None = None,
        enricher: DocumentEnricher | None = None,
        extra_payload: Mapping[str, Any] | None = None,
        payload_indexes: Sequence[str] | None = None,
        distance: Literal["cosine", "dot", "euclid"] = "cosine",
        index_version: str = DEFAULT_INDEX_VERSION,
    ) -> None:
        self._store = QdrantStore(
            url=qdrant_url,
            collection=collection,
            distance=distance,
        )
        self._collection = collection
        self._embedder = embedder
        self._document_reader = document_reader
        self._allowed_extensions = frozenset(allowed_extensions or ())
        self._payload_builder = payload_builder or DefaultPayloadBuilder()
        self._enricher = enricher or NoopEnricher()
        self._extra_payload = dict(extra_payload or {})
        self._payload_indexes = (
            tuple(payload_indexes)
            if payload_indexes is not None
            else tuple(self._payload_builder.payload_indexes())
        )
        self._index_version = index_version

    @property
    def _client(self):
        return self._store.client

    @_client.setter
    def _client(self, client) -> None:
        self._store.client = client

    @overload
    def index(self, watch_path: str) -> None: ...

    @overload
    def index(self, watch_path: str, changes: Sequence[FsChange]) -> None: ...

    def index(
        self,
        watch_path: str,
        changes: Sequence[FsChange] | None = None,
    ) -> None:
        if changes is None:
            self._reindex(watch_path)
            return
        self._apply_changes(watch_path, changes)

    def _reindex(self, watch_path: str) -> None:
        root = Path(watch_path)
        files = iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions or None,
        )
        logger.info(
            "Qdrant reindex start path=%s files=%s collection=%s extensions=%s",
            watch_path,
            len(files),
            self._collection,
            sorted(self._allowed_extensions) if self._allowed_extensions else ["*"],
        )

        disk_hashes = self._disk_file_hashes(watch_path)
        if not disk_hashes:
            indexed_paths = self._store.scroll_indexed_paths()
            removed = 0
            for relative in indexed_paths:
                self._store.delete_source(relative)
                removed += 1
            logger.info(
                "Qdrant reindex done collection=%s empty watch_path removed=%s",
                self._collection,
                removed,
            )
            return

        canonical = _canonical_paths(disk_hashes)
        indexed_paths = self._store.scroll_indexed_paths()

        skipped = upserted = dup_removed = stale_removed = 0
        reindex_started = time.perf_counter()
        for relative, content_hash in disk_hashes.items():
            if relative != canonical[content_hash]:
                if relative in indexed_paths:
                    self._store.delete_source(relative)
                    dup_removed += 1
                continue
            if indexed_paths.get(relative) == content_hash:
                skipped += 1
                continue
            self._upsert_file(watch_path, relative, force=True)
            upserted += 1

        for relative in indexed_paths:
            if relative not in disk_hashes:
                self._store.delete_source(relative)
                stale_removed += 1

        logger.info(
            "Qdrant reindex done collection=%s skipped=%s upserted=%s "
            "dup_removed=%s stale_removed=%s elapsed=%.2fs",
            self._collection,
            skipped,
            upserted,
            dup_removed,
            stale_removed,
            time.perf_counter() - reindex_started,
        )

    def _apply_changes(self, watch_path: str, changes: Sequence[FsChange]) -> None:
        if not changes:
            return
        logger.info(
            "Qdrant apply_changes path=%s ops=%s collection=%s",
            watch_path,
            [(item.op, item.path, item.is_prefix) for item in changes],
            self._collection,
        )
        deletes = [item for item in changes if item.op == "delete"]
        upserts = [item for item in changes if item.op == "upsert"]
        for item in deletes:
            if item.is_prefix:
                self._store.delete_prefix(item.path)
            else:
                self._delete_file(watch_path, item.path)
        for item in upserts:
            self._upsert_file(watch_path, item.path)

    def _delete_file(self, watch_path: str, relative: str) -> None:
        indexed_paths = self._store.scroll_indexed_paths()
        deleted_hash = indexed_paths.get(relative)
        disk_hashes = self._disk_file_hashes(watch_path)
        was_canonical = False
        if deleted_hash:
            peers = sorted(
                path for path, file_hash in disk_hashes.items() if file_hash == deleted_hash
            )
            was_canonical = not peers or relative < peers[0]

        self._store.delete_source(relative)

        if was_canonical and deleted_hash:
            next_path = _next_path_for_hash(disk_hashes, deleted_hash, exclude=set())
            if next_path:
                logger.info(
                    "Promoting duplicate source=%s after delete of canonical=%s",
                    next_path,
                    relative,
                )
                self._upsert_file(watch_path, next_path, force=True)

    def _upsert_file(self, watch_path: str, relative: str, *, force: bool = False) -> None:
        if not self._is_indexable_relative(relative):
            return
        path = Path(watch_path) / relative
        if not path.is_file():
            self._store.delete_source(relative)
            return

        content_hash = file_content_hash(path, index_version=self._index_version)
        if not force:
            disk_hashes = self._disk_file_hashes(watch_path)
            canonical = _canonical_paths(disk_hashes).get(content_hash)
            if canonical and canonical != relative:
                logger.info(
                    "Skip duplicate source=%s canonical=%s hash=%s",
                    relative,
                    canonical,
                    content_hash[:12],
                )
                if self._store.get_indexed_hash(relative) is not None:
                    self._store.delete_source(relative)
                return
            indexed_hash = self._store.get_indexed_hash(relative)
            if indexed_hash == content_hash:
                logger.info(
                    "Skip unchanged source=%s hash=%s",
                    relative,
                    content_hash[:12],
                )
                return

        started = time.perf_counter()
        logger.info("Index start source=%s", relative)
        embedded = self._embed_file(path, relative, content_hash)
        if embedded is None:
            logger.error(
                "Keeping existing Qdrant points after failed indexing source=%s elapsed=%.2fs",
                relative,
                time.perf_counter() - started,
            )
            return
        points, vector_size = embedded
        if not points:
            self._store.delete_source(relative)
            logger.info(
                "Index done source=%s status=empty elapsed=%.2fs",
                relative,
                time.perf_counter() - started,
            )
            return
        if vector_size is None:
            logger.error(
                "Missing vector size source=%s; keeping existing points elapsed=%.2fs",
                relative,
                time.perf_counter() - started,
            )
            return
        self._store.ensure_collection(vector_size, self._payload_indexes)
        self._store.delete_source(relative)
        self._store.upsert(points)
        logger.info(
            "Index done source=%s status=upserted points=%s hash=%s elapsed=%.2fs",
            relative,
            len(points),
            content_hash[:12],
            time.perf_counter() - started,
        )

    def _embed_file(
        self,
        path: Path,
        relative: str,
        content_hash: str,
    ) -> tuple[list[qmodels.PointStruct], int | None] | None:
        read_started = time.perf_counter()
        try:
            chunks = list(self._document_reader.read(path))
        except Exception:
            logger.exception("Failed to read document %s", relative)
            return None
        logger.info(
            "Read done path=%s chunks=%s elapsed=%.2fs",
            relative,
            len(chunks),
            time.perf_counter() - read_started,
        )

        chunks = [
            chunk
            for chunk in chunks
            if chunk.text.strip() and is_useful_chunk_text(chunk.text)
        ]
        if not chunks:
            logger.info("Skip empty document %s", relative)
            return [], None

        enrich_started = time.perf_counter()
        try:
            document_fields = dict(self._enricher.enrich(path, chunks) or {})
        except Exception:
            logger.exception("Document enrich failed path=%s; indexing without extra fields", relative)
            document_fields = {}
        logger.info(
            "Enrich done path=%s enricher=%s fields=%s elapsed=%.2fs",
            relative,
            type(self._enricher).__name__,
            len(document_fields),
            time.perf_counter() - enrich_started,
        )

        parts_by_chunk = []
        for chunk in chunks:
            parts = tuple(
                part.strip()
                for part in (chunk.embedding_parts or (chunk.text,))
                if part.strip()
            )
            parts_by_chunk.append(parts or (chunk.text.strip(),))
        embedding_inputs = [
            part
            for parts in parts_by_chunk
            for part in parts
        ]
        logger.info(
            "Embed start path=%s chunks=%s embedding_parts=%s",
            relative,
            len(chunks),
            len(embedding_inputs),
        )
        started = time.perf_counter()
        try:
            part_vectors = self._embedder.embed(embedding_inputs)
        except Exception:
            logger.exception("Failed to embed document %s chunks=%s", relative, len(chunks))
            return None
        logger.info(
            "Embed done path=%s chunks=%s embedding_parts=%s elapsed=%.2fs",
            relative,
            len(chunks),
            len(part_vectors),
            time.perf_counter() - started,
        )
        if len(part_vectors) != len(embedding_inputs):
            logger.error(
                "Embedding count mismatch path=%s got=%s expected_parts=%s",
                relative,
                len(part_vectors),
                len(embedding_inputs),
            )
            return None

        vectors: list[list[float]] = []
        cursor = 0
        for parts in parts_by_chunk:
            count = len(parts)
            grouped = part_vectors[cursor : cursor + count]
            cursor += count
            aggregated = _aggregate_vectors(grouped)
            if aggregated is None:
                logger.error(
                    "Cannot aggregate embeddings path=%s chunk=%s parts=%s",
                    relative,
                    len(vectors),
                    count,
                )
                return None
            vectors.append(aggregated)

        points: list[qmodels.PointStruct] = []
        vector_size: int | None = None
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            if vector_size is None:
                vector_size = len(vector)
            elif len(vector) != vector_size:
                logger.error(
                    "Embedding size mismatch path=%s got=%s expected=%s",
                    relative,
                    len(vector),
                    vector_size,
                )
                continue
            record = IndexRecord(
                source_path=relative,
                chunk_index=index,
                file_hash=content_hash,
                chunk=chunk,
                file_path=path,
                document_fields=document_fields,
                index_version=self._index_version,
            )
            payload = merge_payload(
                self._payload_builder.build(record),
                self._extra_payload,
                record,
            )
            points.append(
                qmodels.PointStruct(
                    id=str(_point_id(relative, index)),
                    vector=vector,
                    payload=payload,
                )
            )
        if not points:
            return [], None
        return points, vector_size

    def _disk_file_hashes(self, watch_path: str) -> dict[str, str]:
        root = Path(watch_path)
        hashes: dict[str, str] = {}
        for path in iter_document_files(
            root,
            allowed_extensions=self._allowed_extensions or None,
        ):
            relative = path.relative_to(root).as_posix()
            hashes[relative] = file_content_hash(path, index_version=self._index_version)
        return hashes

    def _is_indexable_relative(self, relative: str) -> bool:
        path = Path(relative)
        if path.name.startswith("."):
            return False
        suffix = path.suffix.lower()
        allowed = self._allowed_extensions
        if not allowed:
            from document_indexer.domain.formats import SUPPORTED_SUFFIXES

            allowed = frozenset(SUPPORTED_SUFFIXES)
        return suffix in allowed


def file_content_hash(
    path: Path,
    *,
    index_version: str = _INDEX_VERSION,
) -> str:
    """SHA-256 of file bytes and the indexing algorithm version."""
    digest = hashlib.sha256()
    digest.update(index_version.encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_vectors(vectors: list[list[float]]) -> list[float] | None:
    """Average unit vectors and normalize the result for cosine search."""
    if not vectors or not vectors[0]:
        return None
    if len(vectors) == 1:
        return [float(value) for value in vectors[0]]
    size = len(vectors[0])
    if any(len(vector) != size for vector in vectors):
        return None

    summed = [0.0] * size
    used = 0
    for vector in vectors:
        norm = math.sqrt(sum(float(value) ** 2 for value in vector))
        if norm <= 0:
            continue
        used += 1
        for index, value in enumerate(vector):
            summed[index] += float(value) / norm
    if used == 0:
        return None
    averaged = [value / used for value in summed]
    final_norm = math.sqrt(sum(value**2 for value in averaged))
    if final_norm <= 0:
        return None
    return [value / final_norm for value in averaged]


def _canonical_paths(path_hashes: dict[str, str]) -> dict[str, str]:
    """Map content hash to first sorted source_path on disk."""
    by_hash: dict[str, list[str]] = defaultdict(list)
    for relative, content_hash in path_hashes.items():
        by_hash[content_hash].append(relative)
    return {content_hash: sorted(paths)[0] for content_hash, paths in by_hash.items()}


def _next_path_for_hash(
    path_hashes: dict[str, str],
    content_hash: str,
    *,
    exclude: set[str],
) -> str | None:
    candidates = sorted(
        relative
        for relative, file_hash in path_hashes.items()
        if file_hash == content_hash and relative not in exclude
    )
    return candidates[0] if candidates else None


def _point_id(source_path: str, chunk_index: int) -> uuid.UUID:
    return uuid.uuid5(_POINT_NAMESPACE, f"{source_path}::{chunk_index}")
