"""Ollama chat client used by the resume enricher."""

from __future__ import annotations

import logging

from document_indexer.adapters.enrichment.ollama import OllamaChatCompleter


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
        "document_indexer.adapters.enrichment.ollama.httpx.Client",
        FakeClient,
    )

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
