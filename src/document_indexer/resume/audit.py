"""One-shot resume parse audit: Docling + parser, no LLM / embed / Qdrant."""

from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.document_readers import PictureDescriptionConfig
from document_indexer.config import IndexerSettings
from document_indexer.domain.documents import iter_document_files, resolve_index_extensions
from document_indexer.resume.chunker import ResumeProjectChunker
from document_indexer.indexer import build_source
from document_indexer.infra.logging_config import configure_logging

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_CSV_FIELDS = (
    "source_path",
    "candidate_name",
    "candidate_position",
    "project_count",
    "prose_count",
    "error",
)

ConvertFn = Callable[[Path], Any]


def parse_only_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("RESUME_PARSE_ONLY", "")).strip().lower() in _TRUTHY


def run_resume_parse_audit(
    settings: IndexerSettings,
    *,
    convert: ConvertFn | None = None,
    chunker: ResumeProjectChunker | None = None,
) -> list[dict[str, Any]]:
    """Walk the watch path, parse projects, print and save counts. Does not index."""
    configure_logging(settings.log_level)
    allowed = resolve_index_extensions(settings.index_extensions)
    source = build_source(settings, allowed_extensions=allowed)
    root = source.prepare()
    files = iter_document_files(root, allowed_extensions=allowed)
    parser = chunker or ResumeProjectChunker(
        window_chars=settings.chunking.window_chars,
        window_overlap=settings.chunking.window_overlap,
    )
    convert_fn = convert or _docling_convert()
    logger.info(
        "Resume parse-only start path=%s files=%s (no LLM, no embed, no Qdrant)",
        root,
        len(files),
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        rows.append(_audit_one(path, relative=relative, chunker=parser, convert=convert_fn))
    elapsed = time.perf_counter() - started
    report = format_audit_report(rows)
    logger.info(
        "Resume parse-only done files=%s with_projects=%s without_projects=%s "
        "errors=%s elapsed=%.2fs",
        len(rows),
        sum(1 for row in rows if not row["error"] and row["project_count"] > 0),
        sum(1 for row in rows if not row["error"] and row["project_count"] == 0),
        sum(1 for row in rows if row["error"]),
        elapsed,
    )
    print(report, flush=True)
    csv_path = _write_audit_csv(str(root), rows)
    logger.info("Resume parse-only counts saved path=%s", csv_path)
    print(f"Таблица: {csv_path}", flush=True)
    return rows


def row_from_chunks(relative: str, chunks: Sequence[Any]) -> dict[str, Any]:
    extra = (chunks[0].extra_fields or {}) if chunks else {}
    return {
        "source_path": relative,
        "candidate_name": extra.get("candidate_name"),
        "candidate_position": extra.get("candidate_position"),
        "project_count": sum(1 for chunk in chunks if chunk.chunk_type == "project"),
        "prose_count": sum(1 for chunk in chunks if chunk.chunk_type == "prose"),
        "error": None,
    }


def format_audit_report(rows: Sequence[Mapping[str, Any]]) -> str:
    ok = [row for row in rows if not row.get("error")]
    missing = [row for row in ok if int(row.get("project_count") or 0) == 0]
    errors = [row for row in rows if row.get("error")]
    lines = [
        f"Парсер без LLM: файлов={len(rows)} с проектами="
        f"{len(ok) - len(missing)} без проектов={len(missing)} ошибок={len(errors)}"
    ]
    if missing:
        lines.append(f"Резюме без выделенных проектов ({len(missing)} из {len(ok)}):")
        for row in missing:
            lines.append(
                f"  {row.get('candidate_name') or '?'}  "
                f"{row.get('candidate_position') or 'роль не распознана'}  "
                f"{row.get('source_path')}  prose={row.get('prose_count')}"
            )
    if errors:
        lines.append(f"Ошибки конвертации ({len(errors)}):")
        for row in errors:
            lines.append(f"  {row.get('source_path')}  {row.get('error')}")
    lines.append("Резюме — число проектов:")
    for row in rows:
        if row.get("error"):
            count = "err"
        else:
            count = str(row.get("project_count") or 0)
        lines.append(
            f"  {count:>4}  {row.get('candidate_name') or '?'}  "
            f"{row.get('candidate_position') or 'роль не распознана'}  "
            f"{row.get('source_path')}"
        )
    return "\n".join(lines)


def _audit_one(
    path: Path,
    *,
    relative: str,
    chunker: ResumeProjectChunker,
    convert: ConvertFn,
) -> dict[str, Any]:
    try:
        document = convert(path)
        chunks = chunker.chunk_document(document, path_name=relative)
    except Exception as exc:
        logger.exception("Resume parse-only failed source=%s", relative)
        return {
            "source_path": relative,
            "candidate_name": None,
            "candidate_position": None,
            "project_count": 0,
            "prose_count": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }
    row = row_from_chunks(relative, chunks)
    logger.info(
        "Resume parse-only source=%s name=%s position=%s projects=%s prose=%s",
        relative,
        row["candidate_name"],
        row["candidate_position"],
        row["project_count"],
        row["prose_count"],
    )
    return row


def _docling_convert() -> ConvertFn:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(
        format_options=PictureDescriptionConfig(enabled=False).format_options()
    )

    def convert(path: Path) -> Any:
        return converter.convert(str(path)).document

    return convert


def _write_audit_csv(watch_path: str, rows: Sequence[Mapping[str, Any]]) -> Path:
    candidates = [
        Path(watch_path) / ".resume_project_stats.csv",
        Path(watch_path).resolve().parent / "resume_project_stats.csv",
        Path("/tmp/resume_project_stats.csv"),
    ]
    last_error: OSError | None = None
    for path in candidates:
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            return path
        except OSError as exc:
            last_error = exc
    raise OSError("Could not write resume parse-only CSV") from last_error
