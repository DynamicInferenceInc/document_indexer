"""JsonSchemaEnricher: concatenate chunks and project schema keys."""

from __future__ import annotations

import logging
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


def test_ollama_chat_sends_num_ctx(monkeypatch, caplog) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"message": {"content": '{"ok": true}'}}

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def post(self, url: str, json: dict):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "document_indexer.adapters.enrichment.json_schema.httpx.Client",
        FakeClient,
    )
    from document_indexer.adapters.enrichment.json_schema import OllamaChatCompleter

    caplog.set_level(logging.INFO)
    chat = OllamaChatCompleter(base_url="http://ollama:11434", model="qwen3:4b")
    result = chat.complete(messages=[{"role": "user", "content": "hi"}], format={"type": "object"})
    assert result == {"ok": True}
    assert captured["url"] == "http://ollama:11434/api/chat"
    assert captured["json"]["think"] is False
    assert captured["json"]["options"] == {
        "num_ctx": 16384,
        "num_predict": 4096,
        "temperature": 0.0,
    }
    assert captured["json"]["keep_alive"] == -1
    assert captured["json"]["messages"][0]["content"].startswith("/no_think")
    assert "Ollama chat request sent" in caplog.text
    assert "model=qwen3:4b" in caplog.text
    assert "Ollama chat response received" in caplog.text


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


def test_json_schema_enricher_windows_long_source_and_merges_arrays(tmp_path: Path) -> None:
    class RecordingChat:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *, messages, format):
            del format
            self.calls += 1
            text = messages[-1]["content"]
            projects = []
            if "ALPHA_PROJECT" in text:
                projects.append("alpha")
            if "BETA_PROJECT" in text:
                projects.append("beta")
            return {"grade": "Senior", "projects": projects}

    head = "ALPHA_PROJECT " + ("x" * 80)
    tail = "BETA_PROJECT " + ("y" * 80)
    chat = RecordingChat()
    enricher = JsonSchemaEnricher(
        SCHEMA,
        "Extract fields.",
        chat=chat,
        max_source_chars=len(head),
        overlap_chars=10,
    )
    fields = enricher.enrich(tmp_path / "cv.md", [DocumentChunk(text=head + tail)])
    assert chat.calls >= 2
    assert fields == {"grade": "Senior", "projects": ["alpha", "beta"]}
