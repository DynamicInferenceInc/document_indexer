from pathlib import Path

import pytest

from document_indexer.config import IndexerSettings
from document_indexer.domain.documents import (
    iter_document_files,
    parse_index_extensions,
    resolve_index_extensions,
)
from document_indexer.domain.formats import SUPPORTED_SUFFIXES


def test_parse_index_extensions() -> None:
    assert parse_index_extensions("md, .PDF; txt") == frozenset({".md", ".pdf", ".txt"})
    assert parse_index_extensions("") == frozenset()


def test_resolve_empty_uses_all_docling_suffixes() -> None:
    assert resolve_index_extensions("") == frozenset(SUPPORTED_SUFFIXES)
    assert resolve_index_extensions(None) == frozenset(SUPPORTED_SUFFIXES)


def test_resolve_rejects_unknown_extensions() -> None:
    with pytest.raises(ValueError, match="Docling cannot read"):
        resolve_index_extensions(".md,.xlsx,.pdf,.bin")


def test_settings_reject_unknown_index_extensions() -> None:
    with pytest.raises(ValueError, match="Docling cannot read"):
        IndexerSettings(_env_file=None, index_extensions=".md,.exe")


def test_iter_document_files_filters_by_extension(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("md", encoding="utf-8")
    (tmp_path / "b.txt").write_text("txt", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    (tmp_path / "d.docx").write_bytes(b"PK")

    files = iter_document_files(tmp_path, allowed_extensions={".md", ".pdf"})
    names = {path.name for path in files}
    assert names == {"a.md", "c.pdf"}
