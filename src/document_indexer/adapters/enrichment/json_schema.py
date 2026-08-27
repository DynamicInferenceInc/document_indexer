"""Extract canonical JSON fields from concatenated chunk text via Ollama chat."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from document_indexer.domain.models import DocumentChunk

logger = logging.getLogger(__name__)

_KEEP_ALIVE = -1
_MAX_SOURCE_CHARS = 24_000


class ChatCompleter(Protocol):
    """Minimal chat client used by JsonSchemaEnricher (Ollama or FakeChat)."""

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        format: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return parsed JSON matching ``format``."""
        ...


class OllamaChatCompleter:
    """Ollama ``/api/chat`` with structured ``format`` (JSON Schema)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_sec: float = 180.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        format: Mapping[str, Any],
    ) -> dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(
                url,
                json={
                    "model": self._model,
                    "messages": messages,
                    "stream": False,
                    "format": dict(format),
                    "keep_alive": _KEEP_ALIVE,
                },
            )
            response.raise_for_status()
            data = response.json()
        content = _message_content(data)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama JSON was not an object")
        return parsed


class JsonSchemaEnricher:
    """One LLM call per file: concatenate chunk text, fill schema keys."""

    def __init__(
        self,
        schema: Mapping[str, Any],
        prompt: str,
        *,
        chat: ChatCompleter | None = None,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        timeout_sec: float = 180.0,
        max_source_chars: int = _MAX_SOURCE_CHARS,
    ) -> None:
        self._schema = dict(schema)
        self._prompt = prompt.strip()
        self._max_source_chars = max_source_chars
        if chat is not None:
            self._chat = chat
        elif model:
            self._chat = OllamaChatCompleter(
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
        else:
            raise ValueError("JsonSchemaEnricher requires chat= or a non-empty model")

    def enrich(self, path: Path, chunks: Sequence[DocumentChunk]) -> dict[str, Any]:
        source = _join_chunks(chunks, self._max_source_chars)
        if not source.strip():
            logger.info("Skip enrich empty text path=%s", path)
            return {}
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": source},
        ]
        try:
            raw = self._chat.complete(messages=messages, format=self._schema)
        except Exception:
            logger.exception("LLM enrich failed path=%s; indexing without extra fields", path)
            return {}
        return _project_schema_keys(raw, self._schema)


def _join_chunks(chunks: Sequence[DocumentChunk], limit: int) -> str:
    text = "\n\n".join(chunk.text.strip() for chunk in chunks if chunk.text.strip())
    if len(text) <= limit:
        return text
    return text[:limit]


def _project_schema_keys(raw: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return dict(raw)
    return {key: raw[key] for key in properties if key in raw}


def _message_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Ollama chat response was not an object")
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    raise ValueError("Ollama chat response missing message.content")
