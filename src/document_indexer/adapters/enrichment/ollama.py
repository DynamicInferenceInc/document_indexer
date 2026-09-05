"""Ollama ``/api/chat`` client with structured JSON Schema ``format``."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

_KEEP_ALIVE = -1
_NUM_CTX = 16_384
_NUM_PREDICT = 4_096
_TEMPERATURE = 0.0


class ChatCompleter(Protocol):
    """Minimal chat client used by the resume enricher (Ollama or FakeChat)."""

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
        num_predict: int = _NUM_PREDICT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_sec
        self._num_ctx = num_ctx
        self._num_predict = num_predict

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        format: Mapping[str, Any],
    ) -> dict[str, Any]:
        url = f"{self._base_url}/api/chat"
        user_chars = sum(
            len(message.get("content", ""))
            for message in messages
            if message.get("role") == "user"
        )
        logger.info(
            "Ollama chat request sent url=%s model=%s num_ctx=%s num_predict=%s "
            "user_chars=%s timeout=%.0fs",
            url,
            self._model,
            self._num_ctx,
            self._num_predict,
            user_chars,
            self._timeout,
        )
        _flush_logs()
        started = time.perf_counter()
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
                        "num_predict": self._num_predict,
                        "temperature": _TEMPERATURE,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
        logger.info(
            "Ollama chat response received url=%s model=%s status=%s elapsed=%.2fs "
            "prompt_eval_count=%s eval_count=%s done_reason=%s",
            url,
            self._model,
            getattr(response, "status_code", "?"),
            time.perf_counter() - started,
            data.get("prompt_eval_count") if isinstance(data, dict) else None,
            data.get("eval_count") if isinstance(data, dict) else None,
            data.get("done_reason") if isinstance(data, dict) else None,
        )
        content = _message_content(data)
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Ollama JSON was not an object")
        return parsed


def _flush_logs() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


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
