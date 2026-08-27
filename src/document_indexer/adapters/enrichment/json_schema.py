"""Extract canonical JSON fields from concatenated chunk text via Ollama chat."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from document_indexer.domain.models import DocumentChunk

logger = logging.getLogger(__name__)

_KEEP_ALIVE = -1
# Small windows: qwen3:4b with num_ctx=16384 drops the tail of a 70k-char CV
# if everything is sent in one prompt. Overlap + merge keeps all projects.
_MAX_SOURCE_CHARS = 16_000
_WINDOW_OVERLAP_CHARS = 3_000
_NUM_CTX = 16_384
_NUM_PREDICT = 4_096
_TEMPERATURE = 0.0
_FRAGMENT_HINT = (
    "Это фрагмент {index}/{total} длинного документа. "
    "Извлеки только то, что явно есть в этом фрагменте. "
    "Если в фрагменте нет проектов, верни пустой массив. "
    "Не выдумывай один элемент «за кандидата» по фрагменту без проектов."
)


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


class JsonSchemaEnricher:
    """LLM calls per file: concatenate chunk text, fill schema keys.

    Long documents are split into overlapping windows so the tail is not dropped.
    """

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
        overlap_chars: int = _WINDOW_OVERLAP_CHARS,
    ) -> None:
        self._schema = dict(schema)
        self._prompt = prompt.strip()
        self._max_source_chars = max_source_chars
        self._overlap_chars = overlap_chars
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
        source = _join_chunks(chunks)
        if not source.strip():
            logger.info("Skip enrich empty text path=%s", path)
            return {}
        windows = _text_windows(source, self._max_source_chars, self._overlap_chars)
        started = time.perf_counter()
        logger.info(
            "LLM enrich start path=%s chars=%s windows=%s window_chars=%s overlap=%s",
            path,
            len(source),
            len(windows),
            self._max_source_chars,
            self._overlap_chars,
        )
        if len(windows) > 1:
            logger.info(
                "LLM enrich split path=%s: sending %s overlapping windows instead of "
                "truncating the document tail",
                path,
                len(windows),
            )
        results: list[dict[str, Any]] = []
        for index, window in enumerate(windows, start=1):
            projected = self._enrich_window(path, window, index, len(windows))
            if projected:
                results.append(projected)
        logger.info(
            "LLM enrich done path=%s windows=%s elapsed=%.2fs",
            path,
            len(windows),
            time.perf_counter() - started,
        )
        if not results:
            return {}
        return _merge_projected(results, self._schema)

    def _enrich_window(
        self,
        path: Path,
        window: str,
        index: int,
        total: int,
    ) -> dict[str, Any]:
        header = f"Имя файла: {path.name}"
        if total > 1:
            header = f"{header}\n{_FRAGMENT_HINT.format(index=index, total=total)}"
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": f"{header}\n\n{window}"},
        ]
        try:
            raw = self._chat.complete(messages=messages, format=self._schema)
        except Exception:
            logger.exception(
                "LLM enrich failed path=%s window=%s/%s; continuing with other windows",
                path,
                index,
                total,
            )
            return {}
        return _project_schema_keys(raw, self._schema)


def _join_chunks(chunks: Sequence[DocumentChunk]) -> str:
    return "\n\n".join(chunk.text.strip() for chunk in chunks if chunk.text.strip())


def _text_windows(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or len(text) <= size:
        return [text]
    step = max(size - max(overlap, 0), 1)
    windows: list[str] = []
    start = 0
    while start < len(text):
        windows.append(text[start : start + size])
        if start + size >= len(text):
            break
        start += step
    return windows


def _project_schema_keys(raw: Mapping[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return dict(raw)
    return {key: raw[key] for key in properties if key in raw}


def _merge_projected(results: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return dict(results[-1])
    merged: dict[str, Any] = {}
    for key, spec in properties.items():
        if _is_array_spec(spec):
            items: list[Any] = []
            for result in results:
                value = result.get(key)
                if isinstance(value, list):
                    items.extend(value)
            merged[key] = _dedupe_items(items)
            continue
        for result in results:
            if key in result and result[key] not in (None, "", []):
                merged[key] = result[key]
                break
        else:
            if any(key in result for result in results):
                merged[key] = next(result[key] for result in results if key in result)
    return merged


def _is_array_spec(spec: object) -> bool:
    if not isinstance(spec, dict):
        return False
    type_value = spec.get("type")
    if type_value == "array":
        return True
    return isinstance(type_value, list) and "array" in type_value


def _dedupe_items(items: Sequence[Any]) -> list[Any]:
    """Drop near-duplicates from overlapping windows (trailing punctuation, slash vs backslash)."""
    order: list[str] = []
    chosen: dict[str, Any] = {}
    for item in items:
        key = _item_identity(item)
        previous = chosen.get(key)
        if previous is None:
            order.append(key)
            chosen[key] = item
            continue
        if _filled_count(item) > _filled_count(previous):
            chosen[key] = item
    return [chosen[key] for key in order]


def _item_identity(item: Any) -> str:
    if isinstance(item, dict) and (
        "project_description" in item or "project_position" in item
    ):
        return json.dumps(
            {
                "project_description": _norm_text(item.get("project_description")),
                "project_position": _norm_text(item.get("project_position")),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    return json.dumps(_normalize_value(item), sort_keys=True, ensure_ascii=False, default=str)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _norm_text(value)
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    return value


def _norm_text(value: object) -> str:
    if value is None:
        return ""
    collapsed = " ".join(str(value).split())
    collapsed = collapsed.replace("\\\\", "/").replace("\\", "/")
    return collapsed.strip(" \t.,;:!?…").casefold()


def _filled_count(item: Any) -> int:
    if not isinstance(item, dict):
        return 0
    return sum(1 for value in item.values() if value not in (None, "", []))


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
