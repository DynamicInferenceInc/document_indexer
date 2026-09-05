"""Importable document indexer: Docling processing, Qdrant vectors, local or SMB sources."""

from document_indexer.config import (
    ChunkingSettings,
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
    "ChunkingSettings",
    "DocumentIndexer",
    "IndexerSettings",
    "LocalSourceSettings",
    "ModelSettings",
    "ProfileLocal",
    "ProfileSmb",
    "QdrantSettings",
    "SmbSourceSettings",
    "reindex_once",
    "run",
    "__version__",
]
