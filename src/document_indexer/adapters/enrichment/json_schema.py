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
_NUM_CTX = 16_384
_TEMPERATURE = 0.0


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
        num_ctx: int = _NUM_CTX,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._num_ctx = num_ctx

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
                    "messages": _with_no_think(messages),
                    "stream": False,
                    "format": dict(format),
                    "keep_alive": _KEEP_ALIVE,
                    "think": False,
                    "options": {
                        "num_ctx": self._num_ctx,
                        "temperature": _TEMPERATURE,
                    },
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
        num_ctx: int = _NUM_CTX,
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
                num_ctx=num_ctx,
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


def _with_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Qwen3 native switch; pairs with top-level ``think: false``."""
    if not messages:
        return messages
    patched = [dict(message) for message in messages]
    content = patched[0].get("content", "")
    if "/no_think" not in content:
        patched[0]["content"] = f"/no_think\n{content}"
    return patched


def _strip_think(content: str) -> str:
    marker = "</think>"
    if marker in content:
        return content.split(marker, 1)[1].strip()
    return content.strip()


def _message_content(data: object) -> str:
    if not isinstance(data, dict):
        raise ValueError("Ollama chat response was not an object")
    message = data.get("message")
    if not isinstance(message, dict):
        raise ValueError("Ollama chat response missing message.content")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return _strip_think(content)
    thinking = message.get("thinking")
    if isinstance(thinking, str) and thinking.strip():
        return _strip_think(thinking)
    raise ValueError("Ollama chat response missing message.content")
