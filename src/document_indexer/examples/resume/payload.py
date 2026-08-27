"""Resume profile: schema-backed payload plus document-level LLM fields."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.qdrant.payload import IndexRecord

INDEX_VERSION = "resume-v10"
NO_PROJECTS_LABEL = "Проекты не указаны"
_HERE = Path(__file__).resolve().parent
SCHEMA_PATH = _HERE / "schema.json"
PROMPT_PATH = _HERE / "prompt.txt"
SAMPLE_PATH = _HERE / "sample.md"


def load_resume_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_resume_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_resume_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


class ResumePayloadBuilder:
    """Chunk text plus candidate FIO and project fields on every chunk."""

    index_version = INDEX_VERSION

    def build(self, record: IndexRecord) -> dict[str, Any]:
        fields = record.document_fields
        payload: dict[str, Any] = {
            "text": record.chunk.text,
            "chunk_type": record.chunk.chunk_type,
            "candidate_name": fields.get("candidate_name"),
            "project_experiences": _project_experiences(fields),
        }
        if record.chunk.headings:
            payload["headings"] = list(record.chunk.headings)
        return payload

    def payload_indexes(self) -> Sequence[str]:
        return (
            "source_path",
            "file_hash",
            "chunk_type",
            "candidate_name",
            "project_experiences[].project_description",
            "project_experiences[].project_position",
            "project_experiences[].project_industry",
        )


def _project_experiences(fields: dict[str, Any]) -> list[dict[str, Any]]:
    items = [
        item
        for item in (fields.get("project_experiences") or [])
        if isinstance(item, dict)
    ]
    if not items:
        return [_no_projects()]
    if len(items) == 1:
        only = items[0]
        if not only.get("project_description") and not only.get("project_industry"):
            return [_no_projects(only.get("project_position"))]
    return items


def _no_projects(position: Any = None) -> dict[str, Any]:
    return {
        "project_description": NO_PROJECTS_LABEL,
        "project_position": position,
        "project_industry": None,
    }
