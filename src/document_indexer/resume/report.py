"""Per-resume report: ФИО, должность, project counts. Shared by indexer and audits."""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPORT_FIELDS = (
    "candidate_name",
    "candidate_position",
    "project_count",
    "llm_project_count",
    "experience_count",
    "profile_count",
    "prose_count",
    "needs_review",
    "error",
    "source_path",
)
REPORT_PAYLOAD_KEYS = (
    "source_path",
    "chunk_type",
    "candidate_name",
    "candidate_position",
    "extraction_source",
    "needs_review",
)
_REPORT_BASENAME = "resume_report"


def empty_row(source_path: str) -> dict[str, Any]:
    return {
        "source_path": source_path,
        "candidate_name": None,
        "candidate_position": None,
        "project_count": 0,
        "llm_project_count": 0,
        "experience_count": 0,
        "profile_count": 0,
        "prose_count": 0,
        "needs_review": False,
        "error": None,
    }


def row_from_chunks(source_path: str, chunks: Sequence[Any]) -> dict[str, Any]:
    """Report row from the chunks of one file (audit mode, no Qdrant)."""
    row = empty_row(source_path)
    for chunk in chunks:
        extra = chunk.extra_fields or {}
        _count_chunk(row, chunk.chunk_type, extra)
    return row


def collect_resume_report(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group Qdrant payloads by resume file."""
    per_file: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        source = str(payload.get("source_path") or "")
        if not source:
            continue
        row = per_file.setdefault(source, empty_row(source))
        _count_chunk(row, str(payload.get("chunk_type") or ""), payload)
    return sorted(per_file.values(), key=_sort_key)


def format_resume_report(rows: Sequence[Mapping[str, Any]], *, title: str = "Отчёт по резюме") -> str:
    """Aligned table ФИО | Должность | Проектов | LLM | Мест работы | Проверить | Файл."""
    ordered = sorted(rows, key=_sort_key)
    header = ("ФИО", "Должность", "Проектов", "из них LLM", "Мест работы", "Проверить", "Файл")
    table: list[tuple[str, ...]] = []
    for row in ordered:
        if row.get("error"):
            table.append(
                (
                    _cell(row.get("candidate_name")),
                    _cell(row.get("candidate_position")),
                    "err",
                    "",
                    "",
                    "ошибка",
                    str(row.get("source_path") or ""),
                )
            )
            continue
        table.append(
            (
                _cell(row.get("candidate_name")),
                _cell(row.get("candidate_position"), missing="роль не распознана"),
                str(int(row.get("project_count") or 0)),
                str(int(row.get("llm_project_count") or 0)),
                str(int(row.get("experience_count") or 0)),
                "да" if row.get("needs_review") else "",
                str(row.get("source_path") or ""),
            )
        )
    widths = [len(col) for col in header]
    for line in table:
        for index, value in enumerate(line):
            widths[index] = max(widths[index], len(value))
    lines = [title, _format_line(header, widths), _format_line(tuple("-" * w for w in widths), widths)]
    lines.extend(_format_line(line, widths) for line in table)

    ok = [row for row in ordered if not row.get("error")]
    errors = [row for row in ordered if row.get("error")]
    with_projects = [row for row in ok if int(row.get("project_count") or 0) > 0]
    experience_only = [
        row
        for row in ok
        if int(row.get("project_count") or 0) == 0 and int(row.get("experience_count") or 0) > 0
    ]
    review = [row for row in ok if row.get("needs_review")]
    total_projects = sum(int(row.get("project_count") or 0) for row in ok)
    llm_projects = sum(int(row.get("llm_project_count") or 0) for row in ok)
    lines.append("")
    lines.append(
        f"Итого: резюме={len(rows)} с проектами={len(with_projects)} "
        f"только места работы={len(experience_only)} требуют проверки={len(review)} "
        f"ошибок={len(errors)} проектов всего={total_projects} (из них LLM={llm_projects})"
    )
    no_name = [row for row in ok if not row.get("candidate_name")]
    no_position = [row for row in ok if not row.get("candidate_position")]
    if no_name:
        lines.append(f"Без распознанного ФИО ({len(no_name)}):")
        lines.extend(f"  {row.get('source_path')}" for row in no_name)
    if no_position:
        lines.append(f"Без распознанной должности ({len(no_position)}):")
        lines.extend(
            f"  {row.get('candidate_name') or '?'}  {row.get('source_path')}" for row in no_position
        )
    if review:
        lines.append(f"Требуют ручной проверки ({len(review)}):")
        lines.extend(
            f"  {row.get('candidate_name') or '?'}  {row.get('source_path')}  "
            f"prose={row.get('prose_count')}"
            for row in review
        )
    if errors:
        lines.append(f"Ошибки ({len(errors)}):")
        lines.extend(f"  {row.get('source_path')}  {row.get('error')}" for row in errors)
    return "\n".join(lines)


def write_resume_report(
    watch_path: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    text: str | None = None,
) -> tuple[Path, Path]:
    """Write ``resume_report.csv`` and ``.txt`` next to the watch dir (or /tmp)."""
    report_text = text if text is not None else format_resume_report(rows)
    candidates = [
        Path(watch_path) / f".{_REPORT_BASENAME}",
        Path(watch_path).resolve().parent / _REPORT_BASENAME,
        Path("/tmp") / _REPORT_BASENAME,
    ]
    last_error: OSError | None = None
    for base in candidates:
        csv_path = base.with_suffix(".csv")
        txt_path = base.with_suffix(".txt")
        try:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(sorted(rows, key=_sort_key))
            txt_path.write_text(report_text + "\n", encoding="utf-8")
            return csv_path, txt_path
        except OSError as exc:
            last_error = exc
    raise OSError("Could not write resume report") from last_error


def publish_resume_report(watch_path: str, rows: Sequence[Mapping[str, Any]], *, title: str) -> str:
    """Print, log and save the report. Returns the formatted text."""
    text = format_resume_report(rows, title=title)
    print(text, flush=True)
    logger.info("%s", text)
    try:
        csv_path, txt_path = write_resume_report(watch_path, rows, text=text)
    except OSError:
        logger.exception("Could not save resume report")
        return text
    logger.info("Resume report saved csv=%s txt=%s", csv_path, txt_path)
    print(f"Отчёт: {txt_path}  CSV: {csv_path}", flush=True)
    return text


def _count_chunk(row: dict[str, Any], chunk_type: str, fields: Mapping[str, Any]) -> None:
    name = fields.get("candidate_name")
    if name and not row["candidate_name"]:
        row["candidate_name"] = str(name)
    position = fields.get("candidate_position")
    if position and not row["candidate_position"]:
        row["candidate_position"] = str(position)
    if chunk_type == "project":
        row["project_count"] += 1
        if fields.get("extraction_source") == "llm":
            row["llm_project_count"] += 1
    elif chunk_type == "experience":
        row["experience_count"] += 1
    elif chunk_type == "profile":
        row["profile_count"] += 1
    elif chunk_type == "prose":
        row["prose_count"] += 1
    if fields.get("needs_review"):
        row["needs_review"] = True


def _sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    name = str(row.get("candidate_name") or "")
    return (0 if name else 1, name.casefold(), str(row.get("source_path") or ""))


def _cell(value: object, *, missing: str = "?") -> str:
    text = " ".join(str(value or "").split())
    return text or missing


def _format_line(values: tuple[str, ...], widths: list[int]) -> str:
    return "  ".join(value.ljust(width) for value, width in zip(values, widths, strict=True)).rstrip()
