from types import SimpleNamespace
from unittest.mock import MagicMock

from document_indexer.config import ChunkingSettings, IndexerSettings, QdrantSettings
from document_indexer.domain.models import DocumentChunk
from document_indexer.hybrid.chunker import HYBRID_INDEX_VERSION, HybridDocumentChunker
from document_indexer.indexer import build_indexer


def test_hybrid_chunker_contextualizes_raw_chunks() -> None:
    raw = MagicMock()
    raw.meta.headings = ["Раздел", "Подраздел"]
    hybrid = MagicMock()
    hybrid.chunk.return_value = [raw]
    hybrid.contextualize.return_value = "Раздел\n\nтекст"

    chunks = HybridDocumentChunker(chunker=hybrid).chunk_document(
        SimpleNamespace(),
        path_name="note.pdf",
    )

    assert chunks == [
        DocumentChunk(
            text="Раздел\n\nтекст",
            headings=("Раздел", "Подраздел"),
            chunk_type="prose",
        ),
    ]
    hybrid.chunk.assert_called_once()
    hybrid.contextualize.assert_called_once_with(raw)


def test_hybrid_chunker_skips_empty_text() -> None:
    empty = MagicMock()
    empty.meta.headings = []
    filled = MagicMock()
    filled.meta.headings = ["H"]
    hybrid = MagicMock()
    hybrid.chunk.return_value = [empty, filled]
    hybrid.contextualize.side_effect = ["  ", "kept"]

    chunks = HybridDocumentChunker(chunker=hybrid).chunk_document(
        SimpleNamespace(),
        path_name="note.pdf",
    )
    assert chunks == [DocumentChunk(text="kept", headings=("H",), chunk_type="prose")]


def test_build_indexer_hybrid_uses_chunk_size_tokenizer(monkeypatch) -> None:
    captured: dict = {}
    tokenizer = MagicMock(name="hybrid-tokenizer")

    class FakeIndexer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("document_indexer.indexer.QdrantIndexer", FakeIndexer)
    monkeypatch.setattr("document_indexer.indexer.DocumentConverter", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.OllamaEmbedder", MagicMock)
    monkeypatch.setattr(
        "document_indexer.indexer.tokenizer_with_max_tokens",
        lambda max_tokens, **_kwargs: tokenizer if max_tokens == 1024 else (_ for _ in ()).throw(
            AssertionError(f"unexpected max_tokens={max_tokens}")
        ),
    )
    hybrid_cls = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", hybrid_cls)

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="hybrid"),
        models={
            "chunk_size": 1024,
            "picture_description_enabled": True,
            "vlm_model": "qwen3-vl:8b",
        },
    )
    build_indexer(settings)
    assert captured["index_version"] == HYBRID_INDEX_VERSION
    assert isinstance(captured["document_reader"]._document_chunker, HybridDocumentChunker)
    hybrid_cls.assert_called_once()
    kwargs = hybrid_cls.call_args.kwargs
    assert kwargs["tokenizer"] is tokenizer
    assert "serializer_provider" in kwargs
    assert captured["document_reader"]._picture.enabled is True
    assert captured["document_reader"]._picture.model == "qwen3-vl:8b"


def test_build_indexer_hybrid_keeps_explicit_index_version(monkeypatch) -> None:
    captured: dict = {}

    class FakeIndexer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("document_indexer.indexer.QdrantIndexer", FakeIndexer)
    monkeypatch.setattr("document_indexer.indexer.DocumentConverter", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.OllamaEmbedder", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.tokenizer_with_max_tokens", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", MagicMock)

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="hybrid"),
        qdrant=QdrantSettings(index_version="docs-hybrid-v3"),
    )
    build_indexer(settings)
    assert captured["index_version"] == "docs-hybrid-v3"
