"""Low-level Qdrant collection operations used by QdrantIndexer."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

_SCROLL_LIMIT = 100
# Qdrant rejects HTTP JSON larger than 32MiB. Leave headroom for the envelope.
_MAX_UPSERT_JSON_BYTES = 24 * 1024 * 1024
_DISTANCE = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
}


class QdrantStore:
    """Create collections, index payload fields, scroll hashes, delete, upsert."""

    def __init__(
        self,
        *,
        url: str,
        collection: str,
        distance: Literal["cosine", "dot", "euclid"] = "cosine",
        client: QdrantClient | None = None,
        max_upsert_json_bytes: int = _MAX_UPSERT_JSON_BYTES,
    ) -> None:
        self.collection = collection
        self.distance = distance
        self.client = client or QdrantClient(url=url, check_compatibility=False)
        self._max_upsert_json_bytes = max_upsert_json_bytes

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, vector_size: int, payload_indexes: Sequence[str]) -> None:
        if not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=_DISTANCE[self.distance],
                ),
            )
        for field in payload_indexes:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=qmodels.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug(
                    "Could not create payload index on %s",
                    field,
                    exc_info=True,
                )

    def scroll_indexed_paths(self) -> dict[str, str]:
        if not self.collection_exists():
            return {}
        indexed: dict[str, str] = {}
        offset: object | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=["source_path", "file_hash"],
                with_vectors=False,
                limit=_SCROLL_LIMIT,
                offset=offset,
            )
            for record in records:
                payload = record.payload or {}
                source = str(payload.get("source_path") or "")
                if not source or source in indexed:
                    continue
                indexed[source] = str(payload.get("file_hash") or "")
            if offset is None:
                break
        return indexed

    def get_indexed_hash(self, relative: str) -> str | None:
        if not self.collection_exists():
            return None
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="source_path",
                        match=qmodels.MatchValue(value=relative),
                    )
                ]
            ),
            with_payload=["file_hash"],
            with_vectors=False,
            limit=1,
        )
        if not records:
            return None
        payload = records[0].payload or {}
        value = payload.get("file_hash")
        return str(value) if value else None

    def delete_source(self, relative: str) -> None:
        if not self.collection_exists():
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="source_path",
                            match=qmodels.MatchValue(value=relative),
                        )
                    ]
                )
            ),
        )
        logger.debug("Qdrant deleted source=%s", relative)

    def scroll_payloads(self, fields: Sequence[str]) -> list[dict[str, object]]:
        if not self.collection_exists():
            return []
        payloads: list[dict[str, object]] = []
        offset: object | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=list(fields),
                with_vectors=False,
                limit=_SCROLL_LIMIT,
                offset=offset,
            )
            for record in records:
                payloads.append(record.payload or {})
            if offset is None:
                break
        return payloads

    def delete_prefix(self, prefix: str) -> None:
        if not self.collection_exists():
            return
        normalized = prefix.strip("/")
        if not normalized:
            return
        dir_prefix = f"{normalized}/"
        ids: list[object] = []
        offset: object | None = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=["source_path"],
                with_vectors=False,
                limit=_SCROLL_LIMIT,
                offset=offset,
            )
            for record in records:
                payload = record.payload or {}
                source = str(payload.get("source_path") or "")
                if source == normalized or source.startswith(dir_prefix):
                    ids.append(record.id)
            if offset is None:
                break
        if ids:
            self.client.delete(
                collection_name=self.collection,
                points_selector=ids,
            )
            logger.info(
                "Qdrant deleted prefix=%s points=%s",
                normalized,
                len(ids),
            )

    def upsert(self, points: Sequence[qmodels.PointStruct]) -> None:
        items = list(points)
        if not items:
            return
        batches = list(_iter_upsert_batches(items, self._max_upsert_json_bytes))
        if len(batches) > 1:
            logger.info(
                "Qdrant upsert split collection=%s points=%s batches=%s limit_bytes=%s",
                self.collection,
                len(items),
                len(batches),
                self._max_upsert_json_bytes,
            )
        for batch in batches:
            self.client.upsert(collection_name=self.collection, points=batch)


def _iter_upsert_batches(
    points: Sequence[qmodels.PointStruct],
    max_bytes: int,
) -> Iterator[list[qmodels.PointStruct]]:
    batch: list[qmodels.PointStruct] = []
    used = 0
    for point in points:
        size = _encoded_point_size(point)
        if batch and used + size > max_bytes:
            yield batch
            batch = []
            used = 0
        if size > max_bytes and not batch:
            logger.warning(
                "Qdrant point JSON is %s bytes, limit is %s; sending it alone",
                size,
                max_bytes,
            )
            yield [point]
            continue
        batch.append(point)
        used += size
    if batch:
        yield batch


def _encoded_point_size(point: qmodels.PointStruct) -> int:
    dump = getattr(point, "model_dump_json", None)
    if callable(dump):
        encoded = dump()
        if isinstance(encoded, str):
            return len(encoded.encode("utf-8")) + 1
        return len(encoded) + 1
    return len(str(point).encode("utf-8")) + 1
