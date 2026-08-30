"""LLM: functional direction from project role, else from work performed."""

from __future__ import annotations

import logging
import time
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
        if not any(chunk.chunk_type == "project" for chunk in chunks):
            logger.info("Skip functional direction path=%s: no project chunks", path)
            return {"functional_directions": [None] * len(chunks)}
        project_count = sum(1 for chunk in chunks if chunk.chunk_type == "project")
        started = time.perf_counter()
        logger.info(
            "Functional direction enrich start path=%s projects=%s",
            path,
            project_count,
        )
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
        filled = sum(1 for value in directions if value)
        logger.info(
            "Functional direction enrich done path=%s filled=%s/%s elapsed=%.2fs",
            path,
            filled,
            len(directions),
            time.perf_counter() - started,
        )
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
        if value in (None, "", [], "null", "None"):
            return None
        return str(value).strip() or None


def bind_resume_enricher(settings: Any, enricher: Any) -> Any:
    """Use per-project direction extraction when chunking resumes.

    A whole-document JsonSchemaEnricher writes ``functional_direction`` once;
    the payload reads ``functional_directions`` per chunk, so the field stays empty.
    """
    if getattr(getattr(settings, "chunking", None), "strategy", None) != "resume_project":
        return enricher
    if isinstance(enricher, FunctionalDirectionEnricher):
        return enricher
    models = getattr(settings, "models", None)
    model = str(getattr(models, "extraction_model", "") or "").strip()
    from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher

    if not model:
        return None if isinstance(enricher, JsonSchemaEnricher) else enricher
    if enricher is not None and not isinstance(enricher, JsonSchemaEnricher):
        return enricher
    from document_indexer.examples.resume.payload import load_resume_prompt, load_resume_schema

    logger.info(
        "Using FunctionalDirectionEnricher for resume_project (was %s)",
        type(enricher).__name__ if enricher is not None else "None",
    )
    return FunctionalDirectionEnricher(
        load_resume_schema(),
        load_resume_prompt(),
        base_url=str(getattr(models, "ollama_base_url", "http://127.0.0.1:11434")),
        model=model,
        timeout_sec=float(getattr(models, "extraction_timeout_sec", 180.0)),
    )
