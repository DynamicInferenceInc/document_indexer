"""Resume profile: chunk text plus flat search fields."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.qdrant.payload import IndexRecord

INDEX_VERSION = "resume-v18"
_HERE = Path(__file__).resolve().parent
SCHEMA_PATH = _HERE / "schema.json"
PROMPT_PATH = _HERE / "prompt.txt"
SAMPLE_PATH = _HERE / "sample.md"

_PROJECT_PAYLOAD_KEYS = (
    "project_industry",
    "project_description",
    "project_position",
    "solution_platform",
)


def load_resume_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_resume_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_resume_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


class ResumePayloadBuilder:
    """One point: project or window text, FIO/position, optional project filters."""

    index_version = INDEX_VERSION

    def build(self, record: IndexRecord) -> dict[str, Any]:
        extra = dict(record.chunk.extra_fields or {})
        payload: dict[str, Any] = {
            "text": record.chunk.text,
            "chunk_type": record.chunk.chunk_type,
            "candidate_name": extra.get("candidate_name"),
            "candidate_position": extra.get("candidate_position"),
            "functional_direction": _from_list_or_extra(
                record, extra, "functional_directions", "functional_direction"
            ),
        }
        if record.chunk.chunk_type == "project":
            for key in _PROJECT_PAYLOAD_KEYS:
                if key == "solution_platform":
                    payload[key] = _from_list_or_extra(
                        record, extra, "solution_platforms", "solution_platform"
                    )
                    continue
                payload[key] = extra.get(key)
        if record.chunk.headings:
            payload["headings"] = list(record.chunk.headings)
        return payload

    def payload_indexes(self) -> Sequence[str]:
        return (
            "source_path",
            "file_hash",
            "chunk_type",
            "candidate_name",
            "candidate_position",
            "project_description",
            "project_position",
            "project_industry",
            "functional_direction",
            "solution_platform",
        )


def _from_list_or_extra(
    record: IndexRecord,
    extra: dict[str, Any],
    list_key: str,
    extra_key: str,
) -> str | None:
    values = record.document_fields.get(list_key)
    if isinstance(values, list) and record.chunk_index < len(values):
        value = values[record.chunk_index]
        if value not in (None, "", [], "null", "None"):
            return str(value).strip() or None
    extra_value = extra.get(extra_key)
    if extra_value not in (None, "", [], "null", "None"):
        return str(extra_value).strip() or None
    return None
