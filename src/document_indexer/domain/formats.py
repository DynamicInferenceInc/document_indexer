"""What Docling can convert. Not the list of files to index.

Which suffixes are actually indexed comes from ``INDEX_EXTENSIONS`` in env /
``IndexerSettings``. This set is only the capability whitelist used to reject
unknown types like ``.bin``.
"""

from __future__ import annotations

# Plain text: Docling still converts + HybridChunker (headings for Markdown).
TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".log"}

# Structured / office formats converted via Docling.
DOCLING_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".csv",
}

SUPPORTED_SUFFIXES = TEXT_SUFFIXES | DOCLING_SUFFIXES
