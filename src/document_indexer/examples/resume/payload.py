"""Resume profile: schema-backed payload plus document-level LLM fields."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.qdrant.payload import IndexRecord

INDEX_VERSION = "resume-v7"
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
    """Chunk text plus three exact fields for every resume project."""

    index_version = INDEX_VERSION

    def build(self, record: IndexRecord) -> dict[str, Any]:
        fields = record.document_fields
        payload: dict[str, Any] = {
            "text": record.chunk.text,
            "chunk_type": record.chunk.chunk_type,
            "project_experiences": list(fields.get("project_experiences") or []),
        }
        if record.chunk.headings:
            payload["headings"] = list(record.chunk.headings)
        return payload

    def payload_indexes(self) -> Sequence[str]:
        return (
            "source_path",
            "file_hash",
            "chunk_type",
            "project_experiences[].project_description",
            "project_experiences[].project_position",
            "project_experiences[].project_industry",
        )
