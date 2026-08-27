"""Importable document indexer: Docling processing, Qdrant vectors, local or SMB sources."""

from document_indexer.adapters.enrichment import JsonSchemaEnricher, NoopEnricher
from document_indexer.adapters.qdrant.payload import DefaultPayloadBuilder, IndexRecord, PayloadBuilder
from document_indexer.config import (
    IndexerSettings,
    LocalSourceSettings,
    ModelSettings,
    ProfileLocal,
    ProfileSmb,
    QdrantSettings,
    SmbSourceSettings,
)
from document_indexer.indexer import DocumentIndexer, reindex_once, run

__version__ = "0.1.0"

__all__ = [
    "DefaultPayloadBuilder",
    "DocumentIndexer",
    "IndexRecord",
    "IndexerSettings",
    "JsonSchemaEnricher",
    "LocalSourceSettings",
    "ModelSettings",
    "NoopEnricher",
    "PayloadBuilder",
    "ProfileLocal",
    "ProfileSmb",
    "QdrantSettings",
    "SmbSourceSettings",
    "reindex_once",
    "run",
    "__version__",
]
