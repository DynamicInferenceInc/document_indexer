"""HybridChunker post-processing: one TableItem becomes one Qdrant chunk."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Sequence
from typing import Any

from docling_core.types.doc.items.table.table import TableItem

from document_indexer.adapters.docling_convert import item_caption
from document_indexer.domain.models import DocumentChunk
from document_indexer.infra.chunking import chunk_text

logger = logging.getLogger(__name__)

_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)


def is_useful_chunk_text(text: str) -> bool:
    """Return false for whitespace, markup-only and table-separator-only chunks."""
    stripped = re.sub(r"<[^>]+>", " ", text or "")
    return bool(re.search(r"[^\W_]", stripped, flags=re.UNICODE))


class TableAwareChunker:
    """Walk HybridChunker output and emit prose plus one chunk per TableItem."""

    def __init__(
        self,
        document: Any,
        *,
        chunker: Any,
        max_tokens: int,
        tokenizer: Any | None,
        path_name: str,
    ) -> None:
        self._document = document
        self._chunker = chunker
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._path_name = path_name
        self._serializer = chunker.serializer_provider.get_serializer(document)
        self._table_registry = {table.self_ref: table for table in document.tables}
        self._emitted_tables: set[str] = set()
        self._emitted_mixed_items: set[str] = set()
        self._rendered_tables: list[str] = []
        self._chunks: list[DocumentChunk] = []
        self._raw_count = 0
        self._table_fragments_dropped = 0
        self._junk_chunks_dropped = 0

    def run(self) -> list[DocumentChunk]:
        started = time.perf_counter()
        for raw in self._chunker.chunk(dl_doc=self._document):
            self._raw_count += 1
            self._ingest_raw(raw)
        self._emit_omitted_tables()
        self._assert_table_invariant()
        table_chunks = [chunk for chunk in self._chunks if chunk.chunk_type == "table"]
        logger.info(
            "Chunked document path=%s raw=%s stored=%s tables_detected=%s "
            "tables_emitted=%s table_fragments_dropped=%s junk_chunks_dropped=%s "
            "elapsed=%.2fs",
            self._path_name,
            self._raw_count,
            len(self._chunks),
            len(self._table_registry),
            len(table_chunks),
            self._table_fragments_dropped,
            self._junk_chunks_dropped,
            time.perf_counter() - started,
        )
        return self._chunks

    def _ingest_raw(self, raw: Any) -> None:
        headings = _headings_from_chunk(raw)
        doc_items = list(raw.meta.doc_items or [])
        if any(isinstance(item, TableItem) for item in doc_items):
            # Never contextualize a chunk that mentions a TableItem: HybridChunker
            # may emit many raw segments with the same metadata. Serialize only
            # non-table items once and inject the full structural table once.
            self._ingest_mixed(doc_items, headings)
            return
        text = str(self._chunker.contextualize(raw)).strip()
        if text:
            self._append_text(text, headings=headings)

    def _ingest_mixed(self, doc_items: list[Any], headings: tuple[str, ...]) -> None:
        pending_prose: list[Any] = []
        for item in doc_items:
            if isinstance(item, TableItem):
                self._flush_pending_prose(pending_prose, headings)
                if item.self_ref in self._emitted_tables:
                    self._table_fragments_dropped += 1
                else:
                    self._emit_table(
                        self._table_registry.get(item.self_ref, item),
                        headings=headings,
                    )
                continue
            if item.self_ref in self._emitted_mixed_items:
                continue
            self._emitted_mixed_items.add(item.self_ref)
            pending_prose.append(item)
        self._flush_pending_prose(pending_prose, headings)

    def _flush_pending_prose(
        self,
        pending_prose: list[Any],
        headings: tuple[str, ...],
    ) -> None:
        if not pending_prose:
            return
        prose = _serialize_items(pending_prose, serializer=self._serializer)
        pending_prose.clear()
        if prose:
            self._append_text(prose, headings=headings)

    def _append_text(self, text: str, *, headings: tuple[str, ...]) -> None:
        for piece in split_oversized_text(
            text,
            max_tokens=self._max_tokens,
            tokenizer=self._tokenizer,
        ):
            if not is_useful_chunk_text(piece):
                self._junk_chunks_dropped += 1
                continue
            if _is_residual_table_fragment(piece, self._rendered_tables):
                self._table_fragments_dropped += 1
                continue
            if _is_pure_table(piece):
                fallback_ref = _fallback_table_ref(piece, len(self._chunks))
                self._chunks.append(
                    _make_table_chunk(
                        markdown=piece,
                        headings=headings,
                        table_ref=fallback_ref,
                        caption="",
                        max_tokens=self._max_tokens,
                        tokenizer=self._tokenizer,
                    )
                )
            else:
                self._chunks.append(DocumentChunk(text=piece, headings=headings))

    def _emit_table(self, table: TableItem, *, headings: tuple[str, ...]) -> None:
        ref = table.self_ref
        self._table_registry.setdefault(ref, table)
        markdown = _text_from_serialize(self._serializer, table)
        if not is_useful_chunk_text(markdown):
            raise RuntimeError(
                f"Docling table rendered without useful content: "
                f"path={self._path_name} ref={ref}"
            )
        caption = item_caption(table, self._document)
        table_chunk = _make_table_chunk(
            markdown=markdown,
            headings=headings,
            table_ref=ref,
            caption=caption,
            max_tokens=self._max_tokens,
            tokenizer=self._tokenizer,
        )
        self._chunks.append(table_chunk)
        self._rendered_tables.append(markdown)
        self._emitted_tables.add(ref)
        logger.info(
            "Table chunk path=%s ref=%s rows=%s chars=%s embedding_parts=%s",
            self._path_name,
            ref,
            table_chunk.row_count,
            len(table_chunk.text),
            len(table_chunk.embedding_parts),
        )

    def _emit_omitted_tables(self) -> None:
        # A valid TableItem should normally appear in HybridChunker metadata.
        # Render any structural table it omitted so the document never silently
        # loses it.
        for ref, table in self._table_registry.items():
            if ref not in self._emitted_tables:
                logger.warning(
                    "Table missing from HybridChunker metadata path=%s ref=%s; "
                    "appending at document end",
                    self._path_name,
                    ref,
                )
                self._emit_table(table, headings=())

    def _assert_table_invariant(self) -> None:
        table_chunks = [chunk for chunk in self._chunks if chunk.chunk_type == "table"]
        counts: dict[str, int] = {}
        for chunk in table_chunks:
            if chunk.table_ref:
                counts[chunk.table_ref] = counts.get(chunk.table_ref, 0) + 1
        duplicate_refs = sorted(ref for ref, count in counts.items() if count != 1)
        missing_refs = sorted(set(self._table_registry) - set(counts))
        if duplicate_refs or missing_refs:
            raise RuntimeError(
                f"Table chunk invariant failed path={self._path_name} "
                f"duplicates={duplicate_refs} missing={missing_refs}"
            )


def _headings_from_chunk(chunk: Any) -> tuple[str, ...]:
    headings = chunk.meta.headings or ()
    return tuple(str(item) for item in headings if item)


def _make_table_chunk(
    *,
    markdown: str,
    headings: tuple[str, ...],
    table_ref: str,
    caption: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> DocumentChunk:
    rendered = markdown.strip()
    if caption and caption not in rendered:
        rendered = f"Подпись таблицы: {caption}\n\n{rendered}"
    embedding_parts = _table_embedding_parts(
        markdown,
        headings=headings,
        caption=caption,
        max_tokens=max_tokens,
        tokenizer=tokenizer,
    )
    return DocumentChunk(
        text=rendered,
        headings=headings,
        atomic=True,
        chunk_type="table",
        table_ref=table_ref,
        embedding_parts=embedding_parts,
        row_count=_table_row_count(markdown),
    )


def _table_embedding_parts(
    markdown: str,
    *,
    headings: tuple[str, ...],
    caption: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> tuple[str, ...]:
    context_lines = ["[Таблица]"]
    if headings:
        context_lines.append(f"Раздел: {' / '.join(headings)}")
    if caption:
        context_lines.append(f"Подпись: {caption}")
    prefix = "\n".join(context_lines)
    full = f"{prefix}\n\n{markdown.strip()}"
    if _token_count(full, tokenizer) <= max_tokens:
        return (full,)

    parsed = _parse_pipe_table(markdown)
    if parsed is None:
        pieces = _hard_split_text(markdown.strip(), max_tokens=max_tokens)
        return tuple(f"{prefix}\n\n{piece}" for piece in pieces if piece.strip())

    header, separator, rows = parsed
    header_block = "\n".join(item for item in (header, separator) if item)
    fixed = f"{prefix}\n\n{header_block}".strip()
    parts: list[str] = []
    buffered_rows: list[str] = []

    def flush() -> None:
        if buffered_rows:
            parts.append(f"{fixed}\n" + "\n".join(buffered_rows))
            buffered_rows.clear()

    for row in rows:
        trial_rows = [*buffered_rows, row]
        trial = f"{fixed}\n" + "\n".join(trial_rows)
        if buffered_rows and _token_count(trial, tokenizer) > max_tokens:
            flush()
            trial = f"{fixed}\n{row}"
        if _token_count(trial, tokenizer) <= max_tokens:
            buffered_rows.append(row)
            continue
        # A single very wide row may exceed the embedding budget. It remains
        # intact in payload text; only its searchable representation is split.
        flush()
        row_parts = _hard_split_text(row, max_tokens=max_tokens)
        parts.extend(f"{fixed}\n{part}" for part in row_parts if part.strip())
    flush()
    return tuple(parts or [full])


def _table_row_count(markdown: str) -> int:
    parsed = _parse_pipe_table(markdown)
    if parsed is not None:
        return len(parsed[2])
    return 0


def _fallback_table_ref(text: str, ordinal: int) -> str:
    digest = hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]
    return f"markdown:{ordinal}:{digest}"


def _is_residual_table_fragment(text: str, full_tables: Sequence[str]) -> bool:
    if not full_tables or "|" not in text:
        return False
    normalized = _normalized_search_text(text)
    if not normalized:
        return True
    return any(
        normalized != _normalized_search_text(table)
        and normalized in _normalized_search_text(table)
        for table in full_tables
    )


def _normalized_search_text(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE))


def _text_from_serialize(serializer: Any, item: Any) -> str:
    text = serializer.serialize(item=item).text
    return text.strip() if text else ""


def _serialize_items(items: list[Any], *, serializer: Any) -> str:
    parts = [_text_from_serialize(serializer, item) for item in items]
    return "\n\n".join(part for part in parts if part)


def split_oversized_text(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    """Split prose that exceeds ``max_tokens``. Tables stay one piece each."""
    stripped = text.strip()
    if not stripped:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    pieces: list[str] = []
    for kind, block in _iter_content_blocks(stripped):
        if kind == "table":
            tokens = _token_count(block, tokenizer)
            if tokens > max_tokens:
                logger.warning(
                    "Keeping oversized table as one chunk tokens=%s max_tokens=%s chars=%s",
                    tokens,
                    max_tokens,
                    len(block),
                )
            pieces.append(block)
            continue
        if _token_count(block, tokenizer) <= max_tokens:
            pieces.append(block)
            continue
        pieces.extend(_split_prose(block, max_tokens=max_tokens, tokenizer=tokenizer))
    return pieces


def _iter_content_blocks(text: str) -> list[tuple[str, str]]:
    spans = _table_spans(text)
    if not spans:
        return [("prose", text.strip())] if text.strip() else []

    blocks: list[tuple[str, str]] = []
    cursor = 0
    for start, end in spans:
        if start > cursor:
            prose = text[cursor:start].strip()
            if prose:
                blocks.append(("prose", prose))
        table = text[start:end].strip()
        if table:
            blocks.append(("table", table))
        cursor = end
    if cursor < len(text):
        prose = text[cursor:].strip()
        if prose:
            blocks.append(("prose", prose))
    return blocks


def _table_spans(text: str) -> list[tuple[int, int]]:
    html_spans = [(match.start(), match.end()) for match in _HTML_TABLE_RE.finditer(text)]
    spans = list(html_spans)
    for start, end in _markdown_table_spans(text):
        if any(start < html_end and end > html_start for html_start, html_end in html_spans):
            continue
        spans.append((start, end))
    spans.sort()
    return spans


def _markdown_table_spans(text: str) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    stripped = [line.strip() for line in lines]
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)

    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        if not _is_markdown_table_start(stripped, index):
            index += 1
            continue
        end = index + 1
        while end < len(lines):
            current = stripped[end]
            if not current:
                break
            if _is_table_row(current) or _is_table_sep(current):
                end += 1
                continue
            break
        last = end - 1
        spans.append((starts[index], starts[last] + len(lines[last])))
        index = end
    return spans


def _is_markdown_table_start(stripped_lines: list[str], index: int) -> bool:
    if (
        index >= len(stripped_lines)
        or not _is_table_row(stripped_lines[index])
        or _is_table_sep(stripped_lines[index])
    ):
        return False
    if index + 1 >= len(stripped_lines):
        return False
    return _is_table_sep(stripped_lines[index + 1])


def _is_pure_table(text: str) -> bool:
    blocks = _iter_content_blocks(text)
    return len(blocks) == 1 and blocks[0][0] == "table"


def _parse_pipe_table(text: str) -> tuple[str, str | None, list[str]] | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    for index, line in enumerate(lines[:-1]):
        if _is_table_row(line) and _is_table_sep(lines[index + 1]):
            header = line
            sep = lines[index + 1]
            body = [item for item in lines[index + 2 :] if _is_table_row(item)]
            return header, sep, body
    if all(_is_table_row(line) or _is_table_sep(line) for line in lines):
        header = lines[0]
        rest = [line for line in lines[1:] if _is_table_row(line)]
        if not rest:
            return None
        return header, None, rest
    html_match = _HTML_TABLE_RE.search(text)
    if html_match and html_match.group(0).strip() == text.strip():
        return text.strip(), None, []
    return None


def _split_prose(
    text: str,
    *,
    max_tokens: int,
    tokenizer: Any | None = None,
) -> list[str]:
    separator = "\n\n" if "\n\n" in text else "\n"
    parts = [part.strip() for part in text.split(separator) if part.strip()]
    if len(parts) <= 1:
        return _hard_split_text(text, max_tokens=max_tokens)

    packed: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            packed.append(separator.join(buffer))
            buffer.clear()

    for part in parts:
        if _token_count(part, tokenizer) > max_tokens:
            flush()
            packed.extend(_hard_split_text(part, max_tokens=max_tokens))
            continue
        trial = separator.join(buffer + [part]) if buffer else part
        if buffer and _token_count(trial, tokenizer) > max_tokens:
            flush()
        buffer.append(part)
    flush()
    return packed or [text]


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return max(1, (len(text) + 3) // 4)
    return int(tokenizer.count_tokens(text))


def _hard_split_text(text: str, *, max_tokens: int) -> list[str]:
    window = max(64, max_tokens * 4)
    overlap = min(150, max(0, window // 6))
    return chunk_text(text, chunk_size=window, overlap=overlap)


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return bool(_TABLE_ROW_RE.match(stripped)) and stripped.count("|") >= 2


def _is_table_sep(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line.strip()))
