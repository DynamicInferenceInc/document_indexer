"""Resume chunker: one project per chunk, or sliding windows if none."""

from __future__ import annotations

import logging
from typing import Any

from document_indexer.domain.models import DocumentChunk
from document_indexer.examples.resume.parser import (
    document_tables,
    document_text,
    format_project_text,
    parse_header,
    parse_projects,
)
from document_indexer.infra.chunking import chunk_text

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_CHARS = 1200
_DEFAULT_WINDOW_OVERLAP = 150


class ResumeProjectChunker:
    """Parse labeled projects; fall back to overlapping windows of the resume."""

    def __init__(
        self,
        *,
        window_chars: int = _DEFAULT_WINDOW_CHARS,
        window_overlap: int = _DEFAULT_WINDOW_OVERLAP,
    ) -> None:
        self._window_chars = window_chars
        self._window_overlap = window_overlap

    def chunk_document(self, document: Any, *, path_name: str) -> list[DocumentChunk]:
        text = document_text(document)
        header = parse_header(text)
        projects = parse_projects(text, tables=document_tables(document))
        logger.info(
            "Resume parse path=%s name=%s position=%s projects=%s",
            path_name,
            header.get("candidate_name"),
            header.get("candidate_position"),
            len(projects),
        )
        if projects:
            return [_project_chunk(header, project) for project in projects]
        logger.warning(
            "Resume has no parsed projects path=%s name=%s — falling back to prose windows",
            path_name,
            header.get("candidate_name") or "?",
        )
        return [
            _window_chunk(header, piece)
            for piece in chunk_text(
                text,
                chunk_size=self._window_chars,
                overlap=self._window_overlap,
            )
        ]


def _project_chunk(
    header: dict[str, str | None],
    project: dict[str, str | None],
) -> DocumentChunk:
    extra = {
        "candidate_name": header.get("candidate_name"),
        "candidate_position": header.get("candidate_position"),
        "project_industry": project.get("project_industry"),
        "project_description": project.get("project_description"),
        "project_position": project.get("project_position"),
        "work_performed": project.get("work_performed"),
    }
    return DocumentChunk(
        text=format_project_text(project),
        chunk_type="project",
        extra_fields=extra,
    )


def _window_chunk(header: dict[str, str | None], text: str) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        chunk_type="prose",
        extra_fields={
            "candidate_name": header.get("candidate_name"),
            "candidate_position": header.get("candidate_position"),
        },
    )
