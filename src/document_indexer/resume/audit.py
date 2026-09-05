"""One-shot resume audits: Docling + parser (+ LLM), no embed / Qdrant."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from document_indexer.adapters.document_readers import PictureDescriptionConfig
from document_indexer.config import IndexerSettings
from document_indexer.domain.documents import iter_document_files, resolve_index_extensions
from document_indexer.indexer import build_resume_chunker, build_source
from document_indexer.infra.logging_config import configure_logging
from document_indexer.resume.chunker import ResumeProjectChunker
from document_indexer.resume.report import (
    empty_row,
    format_resume_report,
    publish_resume_report,
    row_from_chunks,
)

logger = logging.getLogger(__name__)

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_CHUNKS_BASENAME = "resume_chunks.jsonl"

ConvertFn = Callable[[Path], Any]


def parse_only_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("RESUME_PARSE_ONLY", "")).strip().lower() in _TRUTHY


def llm_audit_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("RESUME_LLM_AUDIT", "")).strip().lower() in _TRUTHY


def run_resume_parse_audit(
    settings: IndexerSettings,
    *,
    convert: ConvertFn | None = None,
    chunker: ResumeProjectChunker | None = None,
    with_llm: bool = False,
) -> list[dict[str, Any]]:
    """Walk the watch path, chunk every resume, print and save the report. Does not index.

    ``with_llm=False`` is the parser-only audit (``RESUME_PARSE_ONLY``);
    ``with_llm=True`` also runs the LLM steps (``RESUME_LLM_AUDIT``) and dumps
    every chunk into ``resume_chunks.jsonl`` for manual review.
    """
    configure_logging(settings.log_level)
    allowed = resolve_index_extensions(settings.index_extensions)
    source = build_source(settings, allowed_extensions=allowed)
    root = source.prepare()
    files = iter_document_files(root, allowed_extensions=allowed)
    parser = chunker or build_resume_chunker(settings, with_llm=with_llm)
    convert_fn = convert or _docling_convert()
    mode = "llm" if with_llm else "parser"
    logger.info(
        "Resume audit start mode=%s path=%s files=%s (no embed, no Qdrant)",
        mode,
        root,
        len(files),
    )
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    dump: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        row, chunks = _audit_one(path, relative=relative, chunker=parser, convert=convert_fn)
        rows.append(row)
        dump.extend(_chunk_records(relative, chunks))
    elapsed = time.perf_counter() - started
    logger.info(
        "Resume audit done mode=%s files=%s with_projects=%s without_projects=%s "
        "errors=%s elapsed=%.2fs",
        mode,
        len(rows),
        sum(1 for row in rows if not row["error"] and row["project_count"] > 0),
        sum(1 for row in rows if not row["error"] and row["project_count"] == 0),
        sum(1 for row in rows if row["error"]),
        elapsed,
    )
    title = "Аудит резюме (парсер без LLM)" if not with_llm else "Аудит резюме (парсер + LLM)"
    publish_resume_report(str(root), rows, title=title)
    if with_llm:
        try:
            dump_path = _write_chunks_jsonl(str(root), dump)
        except OSError:
            logger.exception("Could not write resume chunks JSONL")
        else:
            logger.info("Resume chunks saved path=%s records=%s", dump_path, len(dump))
            print(f"Чанки: {dump_path}", flush=True)
    return rows


def format_audit_report(rows: Sequence[Mapping[str, Any]]) -> str:
    return format_resume_report(rows, title="Аудит резюме")


def _audit_one(
    path: Path,
    *,
    relative: str,
    chunker: ResumeProjectChunker,
    convert: ConvertFn,
) -> tuple[dict[str, Any], list[Any]]:
    try:
        document = convert(path)
        chunks = list(chunker.chunk_document(document, path_name=relative))
    except Exception as exc:
        logger.exception("Resume audit failed source=%s", relative)
        row = empty_row(relative)
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, []
    row = row_from_chunks(relative, chunks)
    stats = getattr(chunker, "last_stats", None)
    if stats is not None:
        row["parser_projects"] = stats.parser_projects
        row["llm_calls"] = ",".join(stats.llm_calls)
        row["ungrounded_dropped"] = stats.ungrounded_dropped
        row["review_reason"] = stats.review_reason
    logger.info(
        "Resume audit source=%s name=%s position=%s projects=%s llm=%s experience=%s prose=%s",
        relative,
        row["candidate_name"],
        row["candidate_position"],
        row["project_count"],
        row["llm_project_count"],
        row["experience_count"],
        row["prose_count"],
    )
    return row, chunks


def _chunk_records(relative: str, chunks: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for index, chunk in enumerate(chunks):
        if dataclasses.is_dataclass(chunk):
            data = dataclasses.asdict(chunk)
        else:
            data = {
                "text": getattr(chunk, "text", ""),
                "chunk_type": getattr(chunk, "chunk_type", ""),
                "extra_fields": dict(getattr(chunk, "extra_fields", {}) or {}),
            }
        records.append(
            {
                "source_path": relative,
                "chunk_index": index,
                "chunk_type": data.get("chunk_type"),
                "text": data.get("text"),
                "fields": data.get("extra_fields") or {},
            }
        )
    return records


def _docling_convert() -> ConvertFn:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter(
        format_options=PictureDescriptionConfig(enabled=False).format_options()
    )

    def convert(path: Path) -> Any:
        return converter.convert(str(path)).document

    return convert


def _write_chunks_jsonl(watch_path: str, records: Sequence[Mapping[str, Any]]) -> Path:
    candidates = [
        Path(watch_path) / f".{_CHUNKS_BASENAME}",
        Path(watch_path).resolve().parent / _CHUNKS_BASENAME,
        Path("/tmp") / _CHUNKS_BASENAME,
    ]
    last_error: OSError | None = None
    for path in candidates:
        try:
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return path
        except OSError as exc:
            last_error = exc
    raise OSError("Could not write resume chunks JSONL") from last_error
