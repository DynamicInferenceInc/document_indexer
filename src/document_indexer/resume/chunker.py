"""Resume chunker: parser projects, LLM projects, LLM jobs, prose as a last resort."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from document_indexer.config import ResumeSettings
from document_indexer.domain.models import DocumentChunk
from document_indexer.infra.chunking import chunk_text
from document_indexer.resume.grounding import Grounder, infer_solution_platform
from document_indexer.resume.llm_extract import (
    PROFILE_FIELDS,
    PROJECT_FIELDS,
    LlmStepFailed,
    ResumeLlmExtractor,
)
from document_indexer.resume.parser import (
    document_tables,
    document_text,
    format_project_text,
    has_experience_hints,
    merge_projects,
    parse_header,
    parse_projects,
    residual_text,
    same_project,
)

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW_CHARS = 1200
_DEFAULT_WINDOW_OVERLAP = 150
_PROFILE_TITLES = {
    "total_experience": "Стаж",
    "skills": "Навыки",
    "platforms": "Платформы",
    "directions": "Направления",
}

SOURCE_PARSER = "parser"
SOURCE_LLM = "llm"
SOURCE_EXPERIENCE = "experience"


@dataclass
class ResumeParseStats:
    """What happened to one resume; surfaced in logs, audit and the report."""

    parser_projects: int = 0
    llm_projects: int = 0
    experience_jobs: int = 0
    llm_calls: list[str] = field(default_factory=list)
    ungrounded_dropped: int = 0
    needs_review: bool = False
    review_reason: str | None = None


class ResumeProjectChunker:
    """Parse labeled projects; ask the LLM for what the parser missed.

    Without ``chat`` (parse-only audit, no extraction model) the LLM steps are
    skipped and a resume without projects falls back to prose windows.
    """

    def __init__(
        self,
        *,
        window_chars: int = _DEFAULT_WINDOW_CHARS,
        window_overlap: int = _DEFAULT_WINDOW_OVERLAP,
        chat: Any | None = None,
        settings: ResumeSettings | None = None,
        num_ctx: int = 65_536,
        num_predict: int = 8_192,
    ) -> None:
        self._window_chars = window_chars
        self._window_overlap = window_overlap
        self._settings = settings or ResumeSettings()
        self._llm: ResumeLlmExtractor | None = None
        if chat is not None:
            self._llm = ResumeLlmExtractor(
                chat,
                settings=self._settings,
                num_ctx=num_ctx,
                num_predict=num_predict,
            )
        self.last_stats: ResumeParseStats = ResumeParseStats()

    @property
    def llm_enabled(self) -> bool:
        return self._llm is not None

    def chunk_document(self, document: Any, *, path_name: str) -> list[DocumentChunk]:
        stats = ResumeParseStats()
        self.last_stats = stats
        text = document_text(document)
        header = parse_header(text)
        projects = parse_projects(text, tables=document_tables(document))
        stats.parser_projects = len(projects)
        sources = [SOURCE_PARSER] * len(projects)
        grounder = Grounder(text, min_ratio=self._settings.evidence_min_ratio)
        logger.info(
            "Resume parse path=%s name=%s position=%s parser_projects=%s llm=%s",
            path_name,
            header.get("candidate_name"),
            header.get("candidate_position"),
            len(projects),
            self._llm is not None,
        )

        if self._llm is not None and self._settings.llm_projects:
            projects, sources = self._llm_projects(
                text, header, projects, sources, grounder, stats, path_name
            )

        if projects:
            refinements = self._refine(text, header, projects, grounder, stats, path_name)
            chunks = [
                _project_chunk(header, project, refinement, source)
                for project, refinement, source in zip(projects, refinements, sources, strict=True)
            ]
            stats.ungrounded_dropped = grounder.dropped
            self._log_done(path_name, header, stats, chunks)
            return chunks

        if self._llm is not None and self._settings.llm_experience:
            chunks = self._experience(text, header, grounder, stats, path_name)
            if chunks:
                stats.ungrounded_dropped = grounder.dropped
                self._log_done(path_name, header, stats, chunks)
                return chunks

        stats.needs_review = True
        stats.review_reason = stats.review_reason or (
            "no_llm" if self._llm is None else "no_projects_no_jobs"
        )
        stats.ungrounded_dropped = grounder.dropped
        logger.warning(
            "Resume has no structured chunks path=%s name=%s reason=%s — prose windows",
            path_name,
            header.get("candidate_name") or "?",
            stats.review_reason,
        )
        chunks = [
            _window_chunk(header, piece, stats.review_reason)
            for piece in chunk_text(
                text,
                chunk_size=self._window_chars,
                overlap=self._window_overlap,
            )
        ]
        self._log_done(path_name, header, stats, chunks)
        return chunks

    def _llm_projects(
        self,
        text: str,
        header: dict[str, str | None],
        projects: list[dict[str, str | None]],
        sources: list[str],
        grounder: Grounder,
        stats: ResumeParseStats,
        path_name: str,
    ) -> tuple[list[dict[str, str | None]], list[str]]:
        assert self._llm is not None
        residual = residual_text(text, header, projects)
        trigger = None
        if not projects:
            trigger = "no_parser_projects"
        elif len(residual) >= self._settings.residual_min_chars and has_experience_hints(residual):
            trigger = "large_residual"
        if trigger is None or not residual.strip():
            logger.info(
                "LLM projects skipped path=%s parser_projects=%s residual_chars=%s",
                path_name,
                len(projects),
                len(residual),
            )
            return projects, sources
        logger.info(
            "LLM projects start path=%s trigger=%s residual_chars=%s",
            path_name,
            trigger,
            len(residual),
        )
        stats.llm_calls.append("projects")
        try:
            found = self._llm.extract_projects(residual, grounder, path_name=path_name)
        except LlmStepFailed as exc:
            stats.needs_review = True
            stats.review_reason = f"llm_projects_failed: {exc}"
            return projects, sources
        merged = merge_projects(projects, found)
        merged_sources = [
            SOURCE_PARSER if any(same_project(item, known) for known in projects) else SOURCE_LLM
            for item in merged
        ]
        stats.llm_projects = sum(1 for source in merged_sources if source == SOURCE_LLM)
        return merged, merged_sources

    def _refine(
        self,
        text: str,
        header: dict[str, str | None],
        projects: list[dict[str, str | None]],
        grounder: Grounder,
        stats: ResumeParseStats,
        path_name: str,
    ) -> list[dict[str, str | None]]:
        if self._llm is None or not self._settings.llm_refine:
            return [
                {"functional_direction": None, "solution_platform": _explicit_platform(project)}
                for project in projects
            ]
        stats.llm_calls.append("refine")
        try:
            return self._llm.refine(text, projects, header, grounder, path_name=path_name)
        except LlmStepFailed as exc:
            stats.needs_review = True
            stats.review_reason = f"llm_refine_failed: {exc}"
            return [
                {"functional_direction": None, "solution_platform": _explicit_platform(project)}
                for project in projects
            ]

    def _experience(
        self,
        text: str,
        header: dict[str, str | None],
        grounder: Grounder,
        stats: ResumeParseStats,
        path_name: str,
    ) -> list[DocumentChunk]:
        assert self._llm is not None
        stats.llm_calls.append("experience")
        try:
            jobs, profile = self._llm.extract_experience(text, grounder, path_name=path_name)
        except LlmStepFailed as exc:
            stats.needs_review = True
            stats.review_reason = f"llm_experience_failed: {exc}"
            return []
        stats.experience_jobs = len(jobs)
        chunks = [
            _project_chunk(
                header,
                job,
                {"functional_direction": None, "solution_platform": _explicit_platform(job)},
                SOURCE_EXPERIENCE,
                chunk_type="experience",
            )
            for job in jobs
        ]
        profile_chunk = _profile_chunk(header, profile)
        if profile_chunk is not None:
            chunks.append(profile_chunk)
        return chunks

    def _log_done(
        self,
        path_name: str,
        header: dict[str, str | None],
        stats: ResumeParseStats,
        chunks: list[DocumentChunk],
    ) -> None:
        logger.info(
            "Resume chunks path=%s name=%s parser_projects=%s llm_projects=%s "
            "experience=%s chunks=%s llm_calls=%s dropped=%s needs_review=%s",
            path_name,
            header.get("candidate_name"),
            stats.parser_projects,
            stats.llm_projects,
            stats.experience_jobs,
            len(chunks),
            ",".join(stats.llm_calls) or "-",
            stats.ungrounded_dropped,
            stats.needs_review,
        )


def _explicit_platform(project: dict[str, str | None]) -> str | None:
    return infer_solution_platform(
        project.get("project_position"),
        project.get("project_description"),
        project.get("work_performed"),
    )


def _project_chunk(
    header: dict[str, str | None],
    project: dict[str, str | None],
    refinement: dict[str, str | None],
    source: str,
    *,
    chunk_type: str = "project",
) -> DocumentChunk:
    merged = dict(project)
    for key in PROJECT_FIELDS:
        if not merged.get(key) and refinement.get(key):
            merged[key] = refinement[key]
    extra: dict[str, Any] = {
        "candidate_name": header.get("candidate_name"),
        "candidate_position": header.get("candidate_position"),
        "customer": merged.get("customer"),
        "duration": merged.get("duration"),
        "project_industry": merged.get("project_industry"),
        "project_description": merged.get("project_description"),
        "project_position": merged.get("project_position"),
        "work_performed": merged.get("work_performed"),
        "extraction_source": source,
    }
    direction = refinement.get("functional_direction")
    platform = refinement.get("solution_platform") or _explicit_platform(merged)
    if direction:
        extra["functional_direction"] = direction
    if platform:
        extra["solution_platform"] = platform
    return DocumentChunk(
        text=format_project_text(merged),
        chunk_type=chunk_type,
        extra_fields=extra,
    )


def _profile_chunk(
    header: dict[str, str | None],
    profile: dict[str, str | None],
) -> DocumentChunk | None:
    lines: list[str] = []
    if header.get("candidate_name"):
        lines.append(f"ФИО: {header['candidate_name']}")
    if header.get("candidate_position"):
        lines.append(f"Должность: {header['candidate_position']}")
    filled = 0
    for key in PROFILE_FIELDS:
        value = profile.get(key)
        if value:
            filled += 1
            lines.append(f"{_PROFILE_TITLES[key]}: {value}")
    if filled == 0:
        return None
    extra: dict[str, Any] = {
        "candidate_name": header.get("candidate_name"),
        "candidate_position": header.get("candidate_position"),
        "extraction_source": SOURCE_EXPERIENCE,
    }
    for key in PROFILE_FIELDS:
        extra[key] = profile.get(key)
    return DocumentChunk(text="\n".join(lines), chunk_type="profile", extra_fields=extra)


def _window_chunk(
    header: dict[str, str | None],
    text: str,
    reason: str | None,
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        chunk_type="prose",
        extra_fields={
            "candidate_name": header.get("candidate_name"),
            "candidate_position": header.get("candidate_position"),
            "needs_review": True,
            "review_reason": reason,
        },
    )
