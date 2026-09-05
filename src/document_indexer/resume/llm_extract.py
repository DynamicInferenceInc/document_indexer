"""LLM steps of the resume pipeline: find projects, fill gaps, summarize jobs.

Every value that comes back is checked against the resume text by
:class:`~document_indexer.resume.grounding.Grounder`; anything the model made
up becomes ``None``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from document_indexer.adapters.enrichment.ollama import ChatCompleter
from document_indexer.config import ResumeSettings
from document_indexer.resume.grounding import (
    Grounder,
    clean_direction,
    infer_solution_platform,
    normalize_platform,
)
from document_indexer.resume.parser import (
    empty_project,
    format_project_text,
    merge_projects,
    split_sections,
)

logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
PROJECTS_PROMPT_PATH = _HERE / "prompt_projects.txt"
PROJECTS_SCHEMA_PATH = _HERE / "schema_projects.json"
REFINE_PROMPT_PATH = _HERE / "prompt_refine.txt"
EXPERIENCE_PROMPT_PATH = _HERE / "prompt_experience.txt"
EXPERIENCE_SCHEMA_PATH = _HERE / "schema_experience.json"
CLASSIFY_PROMPT_PATH = _HERE / "prompt.txt"

PROJECT_FIELDS = (
    "customer",
    "duration",
    "project_industry",
    "project_description",
    "project_position",
    "work_performed",
)
PROFILE_FIELDS = ("total_experience", "skills", "platforms", "directions")
_LIST_FIELDS = frozenset({"work_performed", "project_description", *PROFILE_FIELDS})
_CHARS_PER_TOKEN = 2.5
_INPUT_SHARE = 0.75

JobRecord = dict[str, str | None]


class LlmStepFailed(RuntimeError):
    """The chat call itself failed; the caller decides on a fallback."""


class ResumeLlmExtractor:
    """Three grounded LLM steps sharing one chat client and one text budget."""

    def __init__(
        self,
        chat: ChatCompleter,
        *,
        settings: ResumeSettings | None = None,
        num_ctx: int = 65_536,
        num_predict: int = 8_192,
    ) -> None:
        self._chat = chat
        self._settings = settings or ResumeSettings()
        budget = int(max(num_ctx - num_predict, 1024) * _INPUT_SHARE * _CHARS_PER_TOKEN)
        self._max_chars = max(500, min(self._settings.section_max_chars, budget))
        self._projects_prompt = _read(PROJECTS_PROMPT_PATH)
        self._projects_schema = json.loads(_read(PROJECTS_SCHEMA_PATH))
        self._refine_prompt = _read(REFINE_PROMPT_PATH) + "\n" + _read(CLASSIFY_PROMPT_PATH)
        self._experience_prompt = _read(EXPERIENCE_PROMPT_PATH)
        self._experience_schema = json.loads(_read(EXPERIENCE_SCHEMA_PATH))

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def extract_projects(
        self,
        residual: str,
        grounder: Grounder,
        *,
        path_name: str,
    ) -> list[dict[str, str | None]]:
        """Projects the parser missed. Only the unparsed text is shown to the model."""
        found: list[dict[str, str | None]] = []
        sections = self._sections(residual)
        for number, section in enumerate(sections, start=1):
            user = f"Имя файла: {path_name}\n\nТекст резюме (часть {number} из {len(sections)}):\n{section}"
            raw = self._call("projects", path_name, self._projects_prompt, user, self._projects_schema)
            items = raw.get("projects") if isinstance(raw, dict) else None
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                project = empty_project()
                for field in PROJECT_FIELDS:
                    project[field] = _ground_field(grounder, field, item.get(field))
                found.append(project)
        merged = merge_projects([], found)
        logger.info(
            "LLM projects path=%s sections=%s raw=%s kept=%s dropped_values=%s",
            path_name,
            len(sections),
            len(found),
            len(merged),
            grounder.dropped,
        )
        return merged

    def refine(
        self,
        text: str,
        projects: list[dict[str, str | None]],
        header: dict[str, str | None],
        grounder: Grounder,
        *,
        path_name: str,
    ) -> list[dict[str, str | None]]:
        """One call per resume: fill empty fields, classify direction and platform.

        Returns one dict per project with the filled fields plus
        ``functional_direction`` / ``solution_platform``. Parser values are never replaced.
        """
        results: list[dict[str, str | None]] = [
            {"functional_direction": None, "solution_platform": None} for _ in projects
        ]
        if not projects:
            return results
        missing = [[f for f in PROJECT_FIELDS if not project.get(f)] for project in projects]
        schema = _refine_schema(sorted({f for fields in missing for f in fields}))
        listing = _describe_projects(projects, missing)
        sections = self._sections(text)
        for number, section in enumerate(sections, start=1):
            user = (
                f"Имя файла: {path_name}\n"
                f"Должность из шапки: {header.get('candidate_position') or '—'}\n\n"
                f"Текст резюме (часть {number} из {len(sections)}):\n{section}\n\n"
                f"Проекты:\n{listing}"
            )
            raw = self._call("refine", path_name, self._refine_prompt, user, schema)
            items = raw.get("projects") if isinstance(raw, dict) else None
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                if not isinstance(index, int) or not 0 <= index < len(projects):
                    continue
                target = results[index]
                for field in missing[index]:
                    if target.get(field):
                        continue
                    value = _ground_field(grounder, field, item.get(field))
                    if value:
                        target[field] = value
                if not target.get("functional_direction"):
                    target["functional_direction"] = clean_direction(item.get("functional_direction"))
                if not target.get("solution_platform"):
                    target["solution_platform"] = normalize_platform(item.get("solution_platform"))
        for project, result in zip(projects, results, strict=True):
            merged = {**project, **{k: v for k, v in result.items() if k in PROJECT_FIELDS and v}}
            explicit = infer_solution_platform(
                merged.get("project_position"),
                merged.get("project_description"),
                merged.get("work_performed"),
                format_project_text(merged),
            )
            if explicit:
                result["solution_platform"] = explicit
        filled = sum(1 for r in results if any(r.get(f) for f in PROJECT_FIELDS))
        logger.info(
            "LLM refine path=%s projects=%s filled_projects=%s directions=%s platforms=%s dropped_values=%s",
            path_name,
            len(projects),
            filled,
            sum(1 for r in results if r.get("functional_direction")),
            sum(1 for r in results if r.get("solution_platform")),
            grounder.dropped,
        )
        return results

    def extract_experience(
        self,
        text: str,
        grounder: Grounder,
        *,
        path_name: str,
    ) -> tuple[list[dict[str, str | None]], dict[str, str | None]]:
        """Jobs in project shape plus a profile summary when no projects exist."""
        jobs: list[dict[str, str | None]] = []
        profile: dict[str, str | None] = {field: None for field in PROFILE_FIELDS}
        sections = self._sections(text)
        for number, section in enumerate(sections, start=1):
            user = f"Имя файла: {path_name}\n\nТекст резюме (часть {number} из {len(sections)}):\n{section}"
            raw = self._call(
                "experience", path_name, self._experience_prompt, user, self._experience_schema
            )
            if not isinstance(raw, dict):
                continue
            for item in raw.get("jobs") or []:
                if not isinstance(item, dict):
                    continue
                job = empty_project()
                job["customer"] = _ground_field(grounder, "employer", item.get("employer"))
                job["duration"] = _ground_field(grounder, "period", item.get("period"))
                job["project_position"] = _ground_field(grounder, "position", item.get("position"))
                job["work_performed"] = _ground_field(
                    grounder, "work_performed", item.get("work_performed")
                )
                job["project_description"] = _ground_field(grounder, "systems", item.get("systems"))
                jobs.append(job)
            raw_profile = raw.get("profile")
            if isinstance(raw_profile, dict):
                for field in PROFILE_FIELDS:
                    if profile.get(field):
                        continue
                    profile[field] = _ground_field(grounder, field, raw_profile.get(field))
        merged = merge_projects([], jobs)
        logger.info(
            "LLM experience path=%s sections=%s jobs=%s profile_fields=%s dropped_values=%s",
            path_name,
            len(sections),
            len(merged),
            sum(1 for value in profile.values() if value),
            grounder.dropped,
        )
        return merged, profile

    def _sections(self, text: str) -> list[str]:
        sections = split_sections(
            text,
            max_chars=self._max_chars,
            overlap_chars=self._settings.section_overlap_chars,
        )
        return sections or [text.strip() or "—"]

    def _call(
        self,
        step: str,
        path_name: str,
        system: str,
        user: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        started = time.perf_counter()
        try:
            raw = self._chat.complete(messages=messages, format=schema)
        except Exception as exc:
            logger.exception("LLM step failed step=%s path=%s", step, path_name)
            raise LlmStepFailed(f"{step}: {type(exc).__name__}: {exc}") from exc
        logger.info(
            "LLM step done step=%s path=%s user_chars=%s elapsed=%.2fs",
            step,
            path_name,
            len(user),
            time.perf_counter() - started,
        )
        return raw if isinstance(raw, dict) else {}


def _ground_field(grounder: Grounder, field: str, value: object) -> str | None:
    if field in _LIST_FIELDS:
        return grounder.ground_fragments(value, field=field)
    return grounder.ground(value, field=field)


def _refine_schema(fields: list[str]) -> dict[str, Any]:
    properties: dict[str, Any] = {"index": {"type": "integer"}}
    for field in fields:
        properties[field] = {"type": ["string", "null"]}
    properties["functional_direction"] = {"type": ["string", "null"]}
    properties["solution_platform"] = {"type": ["string", "null"]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "projects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                },
            }
        },
        "required": ["projects"],
    }


def _describe_projects(
    projects: list[dict[str, str | None]],
    missing: list[list[str]],
) -> str:
    blocks = []
    for index, (project, fields) in enumerate(zip(projects, missing, strict=True)):
        body = format_project_text(project) or "(поля не заполнены)"
        empty = ", ".join(fields) if fields else "нет"
        blocks.append(f"[index={index}]\n{body}\nПустые поля для заполнения: {empty}")
    return "\n\n".join(blocks)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()
