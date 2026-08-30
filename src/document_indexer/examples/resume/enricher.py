"""LLM: functional direction from project role, else from work performed."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.enrichment.json_schema import ChatCompleter, OllamaChatCompleter
from document_indexer.domain.models import DocumentChunk

logger = logging.getLogger(__name__)


class FunctionalDirectionEnricher:
    """One short chat call per project chunk. Window chunks are left untouched."""

    def __init__(
        self,
        schema: Mapping[str, Any],
        prompt: str,
        *,
        chat: ChatCompleter | None = None,
        base_url: str = "http://127.0.0.1:11434",
        model: str = "",
        timeout_sec: float = 180.0,
    ) -> None:
        self._schema = dict(schema)
        self._prompt = prompt.strip()
        if chat is not None:
            self._chat = chat
        elif model:
            self._chat = OllamaChatCompleter(
                base_url=base_url,
                model=model,
                timeout_sec=timeout_sec,
            )
        else:
            raise ValueError("FunctionalDirectionEnricher requires chat= or a non-empty model")

    def enrich(self, path: Path, chunks: Sequence[DocumentChunk]) -> dict[str, Any]:
        directions: list[str | None] = []
        for chunk in chunks:
            if chunk.chunk_type != "project":
                directions.append(None)
                continue
            extra = chunk.extra_fields or {}
            role = extra.get("project_position")
            work = extra.get("work_performed")
            header_position = extra.get("candidate_position")
            if not role and not work and not header_position:
                directions.append(None)
                continue
            directions.append(self._extract(path, role, work, header_position))
        return {"functional_directions": directions}

    def _extract(
        self,
        path: Path,
        role: object,
        work: object,
        header_position: object,
    ) -> str | None:
        lines = []
        if role:
            lines.append(f"Роль на проекте: {role}")
        if work:
            lines.append(f"Выполненные работы: {work}")
        if header_position:
            lines.append(f"Должность из шапки: {header_position}")
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": f"Имя файла: {path.name}\n\n" + "\n".join(lines)},
        ]
        try:
            raw = self._chat.complete(messages=messages, format=self._schema)
        except Exception:
            logger.exception("Functional direction failed path=%s", path)
            return None
        value = raw.get("functional_direction") if isinstance(raw, dict) else None
        if value in (None, "", []):
            return None
        return str(value).strip() or None
