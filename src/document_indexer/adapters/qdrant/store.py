"""Low-level Qdrant collection operations used by QdrantIndexer."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger(__name__)

_SCROLL_LIMIT = 100
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
    ) -> None:
        self.collection = collection
        self.distance = distance
        self.client = client or QdrantClient(url=url, check_compatibility=False)

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
        self.client.upsert(collection_name=self.collection, points=list(points))
