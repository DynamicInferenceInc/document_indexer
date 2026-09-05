"""Resume profile: chunk text plus flat search fields."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.qdrant.payload import IndexRecord

INDEX_VERSION = "resume-v20"
_HERE = Path(__file__).resolve().parent
PROMPT_PATH = _HERE / "prompt.txt"
SAMPLE_PATH = _HERE / "sample.md"

PROJECT_CHUNK_TYPES = frozenset({"project", "experience"})
_PROJECT_PAYLOAD_KEYS = (
    "customer",
    "duration",
    "project_industry",
    "project_description",
    "project_position",
    "work_performed",
    "solution_platform",
)
_PROFILE_PAYLOAD_KEYS = ("total_experience", "skills", "platforms", "directions")


def load_resume_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_resume_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


class ResumePayloadBuilder:
    """One point: project / experience / profile / prose text plus flat filters."""

    index_version = INDEX_VERSION

    def build(self, record: IndexRecord) -> dict[str, Any]:
        extra = dict(record.chunk.extra_fields or {})
        chunk_type = record.chunk.chunk_type
        payload: dict[str, Any] = {
            "text": record.chunk.text,
            "chunk_type": chunk_type,
            "candidate_name": extra.get("candidate_name"),
            "candidate_position": extra.get("candidate_position"),
            "functional_direction": _from_list_or_extra(
                record, extra, "functional_directions", "functional_direction"
            ),
        }
        if extra.get("extraction_source"):
            payload["extraction_source"] = extra["extraction_source"]
        if extra.get("needs_review"):
            payload["needs_review"] = True
            if extra.get("review_reason"):
                payload["review_reason"] = extra["review_reason"]
        if chunk_type in PROJECT_CHUNK_TYPES:
            for key in _PROJECT_PAYLOAD_KEYS:
                if key == "solution_platform":
                    payload[key] = _from_list_or_extra(
                        record, extra, "solution_platforms", "solution_platform"
                    )
                    continue
                payload[key] = extra.get(key)
        elif chunk_type == "profile":
            for key in _PROFILE_PAYLOAD_KEYS:
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
            "customer",
            "project_description",
            "project_position",
            "project_industry",
            "functional_direction",
            "solution_platform",
            "extraction_source",
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
