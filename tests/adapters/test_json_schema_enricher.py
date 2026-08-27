"""JsonSchemaEnricher: concatenate chunks and project schema keys."""

from __future__ import annotations

from pathlib import Path

from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher
from document_indexer.domain.models import DocumentChunk

SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": ["string", "null"]},
        "projects": {"type": "array"},
    },
}


class FakeChat:
    def __init__(self) -> None:
        self.messages = None
        self.format = None

    def complete(self, *, messages, format):
        self.messages = messages
        self.format = format
        assert "Звание" in messages[-1]["content"]
        return {
            "grade": "Senior",
            "projects": ["Альфа"],
            "ignored": True,
        }


def test_json_schema_enricher_uses_chunk_text_and_drops_extra_keys(tmp_path: Path) -> None:
    chat = FakeChat()
    enricher = JsonSchemaEnricher(SCHEMA, "Extract fields.", chat=chat)
    chunks = [
        DocumentChunk(text="Иванов. Звание: старший разработчик."),
        DocumentChunk(text="Проект Альфа, Python."),
    ]
    fields = enricher.enrich(tmp_path / "cv.md", chunks)
    assert fields == {"grade": "Senior", "projects": ["Альфа"]}
    assert "ignored" not in fields
    assert "Иванов" in chat.messages[-1]["content"]
    assert "Альфа" in chat.messages[-1]["content"]
    assert chat.format == SCHEMA


def test_json_schema_enricher_returns_empty_on_chat_error(tmp_path: Path) -> None:
    class Boom:
        def complete(self, *, messages, format):
            raise RuntimeError("ollama down")

    enricher = JsonSchemaEnricher(SCHEMA, "Extract fields.", chat=Boom())
    fields = enricher.enrich(
        tmp_path / "cv.md",
        [DocumentChunk(text="Звание: Middle")],
    )
    assert fields == {}
