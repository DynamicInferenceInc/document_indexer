"""Composition root wires table_aware vs resume internally."""

from __future__ import annotations

from unittest.mock import MagicMock

from document_indexer.adapters.qdrant.payload import DefaultPayloadBuilder
from document_indexer.config import ChunkingSettings, IndexerSettings, ModelSettings, QdrantSettings
from document_indexer.indexer import build_indexer
from document_indexer.resume.chunker import ResumeProjectChunker
from document_indexer.resume.payload import INDEX_VERSION, ResumePayloadBuilder
from document_indexer.table_aware.chunker import TableAwareDocumentChunker


def _capture_indexer(monkeypatch) -> dict:
    captured: dict = {}

    class FakeIndexer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("document_indexer.indexer.QdrantIndexer", FakeIndexer)
    monkeypatch.setattr("document_indexer.indexer.DocumentConverter", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.OllamaEmbedder", MagicMock)
    monkeypatch.setattr("document_indexer.indexer.tokenizer_with_max_tokens", lambda *_a, **_k: MagicMock())
    monkeypatch.setattr("document_indexer.indexer.HybridChunker", MagicMock)
    return captured


def test_build_indexer_table_aware_default(monkeypatch) -> None:
    captured = _capture_indexer(monkeypatch)
    indexer = build_indexer(IndexerSettings(_env_file=None))
    assert isinstance(indexer, object)
    assert isinstance(captured["payload_builder"], DefaultPayloadBuilder)
    assert captured["enricher"] is None
    assert captured["index_version"] == "table-aware-v2"
    assert isinstance(captured["document_reader"]._document_chunker, TableAwareDocumentChunker)


def test_build_indexer_resume_wires_llm_chunker_and_payload(monkeypatch) -> None:
    captured = _capture_indexer(monkeypatch)
    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="resume_project"),
        models=ModelSettings(
            extraction_model="qwen3.8:27b-q8_0",
            extraction_num_ctx=32768,
            extraction_num_predict=4096,
        ),
    )
    build_indexer(settings)
    assert isinstance(captured["payload_builder"], ResumePayloadBuilder)
    assert captured["enricher"] is None
    assert captured["index_version"] == INDEX_VERSION
    chunker = captured["document_reader"]._document_chunker
    assert isinstance(chunker, ResumeProjectChunker)
    assert chunker.llm_enabled is True
    assert chunker._llm._chat._model == "qwen3.8:27b-q8_0"
    assert chunker._llm._chat._num_ctx == 32768
    assert chunker._llm._chat._num_predict == 4096


def test_build_indexer_resume_without_extraction_model_has_no_llm(monkeypatch) -> None:
    captured = _capture_indexer(monkeypatch)
    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="resume_project"),
    )
    build_indexer(settings)
    assert isinstance(captured["payload_builder"], ResumePayloadBuilder)
    assert captured["enricher"] is None
    chunker = captured["document_reader"]._document_chunker
    assert isinstance(chunker, ResumeProjectChunker)
    assert chunker.llm_enabled is False


def test_build_indexer_keeps_explicit_index_version(monkeypatch) -> None:
    captured = _capture_indexer(monkeypatch)
    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="resume_project"),
        qdrant=QdrantSettings(index_version="resume-custom"),
        models=ModelSettings(extraction_model="qwen3:8b"),
    )
    build_indexer(settings)
    assert captured["index_version"] == "resume-custom"
