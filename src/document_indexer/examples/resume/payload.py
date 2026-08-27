"""Resume profile: schema-backed payload plus document-level LLM fields."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.qdrant.payload import IndexRecord

INDEX_VERSION = "resume-v12"
NO_PROJECTS_LABEL = "Проекты не указаны"
_HERE = Path(__file__).resolve().parent
SCHEMA_PATH = _HERE / "schema.json"
PROMPT_PATH = _HERE / "prompt.txt"
SAMPLE_PATH = _HERE / "sample.md"
_DATE_POSITION_RE = re.compile(
    r"\d{4}\s*[-–—]\s*(\d{4}|текущ|н\.?\s*в|настоящ)",
    re.IGNORECASE,
)
_DATE_ONLY_DESC_RE = re.compile(
    r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}\s*[-–—]\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}$"
)
_MESSY_COLON_RE = re.compile(r"[А-Яа-яA-Za-z]{3,}:[А-Яа-яA-Za-z]")
_COMPANY_ONLY_RE = re.compile(
    r"\bг\.\s|.+,\s*(россия|russia)\s*$",
    re.IGNORECASE,
)
_PROJECT_HINT_RE = re.compile(
    r"(проект|внедрен|переход|миграц|автоматиз|интеграц)",
    re.IGNORECASE,
)


def load_resume_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_resume_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_resume_sample() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


class ResumePayloadBuilder:
    """Chunk text plus candidate FIO/title and cleaned project fields."""

    index_version = INDEX_VERSION

    def build(self, record: IndexRecord) -> dict[str, Any]:
        fields = record.document_fields
        payload: dict[str, Any] = {
            "text": record.chunk.text,
            "chunk_type": record.chunk.chunk_type,
            "candidate_name": fields.get("candidate_name"),
            "candidate_position": fields.get("candidate_position"),
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
            "candidate_position",
            "project_experiences[].project_description",
            "project_experiences[].project_position",
            "project_experiences[].project_industry",
        )


def _project_experiences(fields: dict[str, Any]) -> list[dict[str, Any]]:
    header_position = fields.get("candidate_position")
    items = [
        item
        for item in (fields.get("project_experiences") or [])
        if isinstance(item, dict)
    ]
    cleaned = _dedupe_by_description(_drop_header_noise(items))
    if not cleaned:
        return [_no_projects(header_position)]
    if len(cleaned) == 1:
        only = cleaned[0]
        if (
            not only.get("project_description")
            or only.get("project_description") == NO_PROJECTS_LABEL
        ) and not only.get("project_industry"):
            return [_no_projects(only.get("project_position") or header_position)]
    return cleaned


def _drop_header_noise(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop date/company header rows and mashed table cells; keep distinct real projects."""
    kept: list[dict[str, Any]] = []
    for item in items:
        if _is_date_position(item.get("project_position")):
            continue
        if _is_date_description(item.get("project_description")):
            continue
        if _is_company_header(item.get("project_description")):
            continue
        kept.append(item)
    mashed, proper = [], []
    for item in kept:
        if _is_mashed_table_row(item):
            mashed.append(item)
        else:
            proper.append(item)
    return proper or mashed


def _dedupe_by_description(items: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge window copies of the same project; keep distinct descriptions."""
    order: list[str] = []
    chosen: dict[str, dict[str, Any]] = {}
    for item in items:
        key = _norm_desc(item.get("project_description"))
        if not key:
            continue
        previous = chosen.get(key)
        if previous is None:
            order.append(key)
            chosen[key] = item
            continue
        if _item_quality(item) > _item_quality(previous):
            chosen[key] = item
    return [chosen[key] for key in order]


def _norm_desc(value: object) -> str:
    collapsed = " ".join(str(value or "").split())
    collapsed = collapsed.replace("\\\\", "/").replace("\\", "/")
    return collapsed.strip(" \t.,;:!?…").casefold()


def _item_quality(item: dict[str, Any]) -> tuple[int, int, int]:
    position = str(item.get("project_position") or "")
    clean_colon = 0 if _MESSY_COLON_RE.search(position) else 1
    filled = sum(1 for value in item.values() if value not in (None, "", []))
    return (clean_colon, filled, len(position))


def _is_date_description(value: object) -> bool:
    text = str(value or "").strip()
    if not text or _PROJECT_HINT_RE.search(text):
        return False
    return bool(_DATE_ONLY_DESC_RE.fullmatch(text) or _DATE_POSITION_RE.fullmatch(text))


def _is_date_position(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return bool(_DATE_POSITION_RE.search(text))


def _is_company_header(value: object) -> bool:
    text = str(value or "").strip()
    if not text or _PROJECT_HINT_RE.search(text):
        return False
    return bool(_COMPANY_ONLY_RE.search(text))


def _is_mashed_table_row(item: dict[str, Any]) -> bool:
    description = str(item.get("project_description") or "")
    position = str(item.get("project_position") or "").strip()
    if len(position) < 8:
        return False
    return position.casefold() in description.casefold()


def _no_projects(position: Any = None) -> dict[str, Any]:
    return {
        "project_description": NO_PROJECTS_LABEL,
        "project_position": position or None,
        "project_industry": None,
    }
