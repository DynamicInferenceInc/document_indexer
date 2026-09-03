from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from document_indexer.adapters.docling_chunking import HybridDocumentChunker
from document_indexer.adapters.document_readers import DoclingDocumentReader
from document_indexer.config import ChunkingSettings, IndexerSettings, QdrantSettings
from document_indexer.domain.models import DocumentChunk
from document_indexer.indexer import _build_document_chunker, build_indexer


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
        DocumentChunk(text="Раздел\n\nтекст", headings=("Раздел", "Подраздел")),
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
    assert chunks == [DocumentChunk(text="kept", headings=("H",))]


def test_docling_reader_uses_hybrid_document_chunker(tmp_path: Path) -> None:
    path = tmp_path / "note.pdf"
    path.write_bytes(b"%PDF")
    result = MagicMock()
    result.document = SimpleNamespace()
    converter = MagicMock()
    converter.convert.return_value = result
    raw = MagicMock()
    raw.meta.headings = ["Title"]
    hybrid = MagicMock()
    hybrid.chunk.return_value = [raw]
    hybrid.contextualize.return_value = "Title\n\nbody"

    chunks = DoclingDocumentReader(
        converter,
        hybrid,
        document_chunker=HybridDocumentChunker(chunker=hybrid),
    ).read(path)

    converter.convert.assert_called_once_with(str(path))
    hybrid.chunk.assert_called_once_with(dl_doc=result.document)
    assert chunks[0].text == "Title\n\nbody"


def test_build_document_chunker_hybrid(monkeypatch) -> None:
    fake_hybrid = MagicMock()
    hybrid_cls = MagicMock(return_value=fake_hybrid)
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", hybrid_cls)
    monkeypatch.setattr(
        "document_indexer.indexer.tokenizer_with_max_tokens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hybrid must use HybridChunker() defaults")
        ),
    )
    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="hybrid"),
    )
    chunker, hybrid, tokenizer = _build_document_chunker(settings)
    assert isinstance(chunker, HybridDocumentChunker)
    assert hybrid is fake_hybrid
    assert tokenizer is None
    assert chunker._chunker is fake_hybrid
    hybrid_cls.assert_called_once_with()


def test_build_indexer_hybrid_defaults_index_version(monkeypatch) -> None:
    captured: dict = {}

    class FakeIndexer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("document_indexer.indexer.QdrantIndexer", FakeIndexer)
    monkeypatch.setattr("document_indexer.indexer.DocumentConverter", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.OllamaEmbedder", MagicMock)
    monkeypatch.setattr(
        "document_indexer.indexer.tokenizer_with_max_tokens",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("hybrid must use HybridChunker() defaults")
        ),
    )
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", lambda: MagicMock())

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="hybrid"),
    )
    indexer = build_indexer(settings)
    assert isinstance(indexer, FakeIndexer)
    assert captured["index_version"] == "hybrid-v1"
    assert isinstance(
        captured["document_reader"]._document_chunker,
        HybridDocumentChunker,
    )


def test_build_indexer_hybrid_keeps_explicit_index_version(monkeypatch) -> None:
    captured: dict = {}

    class FakeIndexer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("document_indexer.indexer.QdrantIndexer", FakeIndexer)
    monkeypatch.setattr("document_indexer.indexer.DocumentConverter", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.OllamaEmbedder", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", lambda: MagicMock())

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="hybrid"),
        qdrant=QdrantSettings(index_version="docs-hybrid-v3"),
    )
    build_indexer(settings)
    assert captured["index_version"] == "docs-hybrid-v3"
