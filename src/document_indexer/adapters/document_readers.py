"""Document reader adapters (Docling HybridChunker)."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from document_indexer.table_aware.chunker import TableAwareDocumentChunker
from document_indexer.adapters.docling_convert import (
    PictureDescriptionConfig,
    picture_description,
)
from document_indexer.domain.formats import SUPPORTED_SUFFIXES
from document_indexer.domain.models import DocumentChunk
from document_indexer.ports.chunker import DocumentChunker

logger = logging.getLogger(__name__)

_DEFAULT_MAX_TOKENS = 1024
_VLM_URL_MARKERS = ("/v1/chat/completions", "/api/chat")

__all__ = [
    "DoclingDocumentReader",
    "PictureDescriptionConfig",
]


class DoclingDocumentReader:
    """Convert a file with Docling and split it via HybridChunker."""

    def __init__(
        self,
        converter: Any,
        chunker: Any,
        *,
        document_chunker: DocumentChunker | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        tokenizer: Any | None = None,
        picture: PictureDescriptionConfig | None = None,
    ) -> None:
        self._converter = converter
        self._chunker = chunker
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer
        self._picture = picture or PictureDescriptionConfig()
        self._document_chunker = document_chunker or TableAwareDocumentChunker(
            chunker=chunker,
            max_tokens=max_tokens,
            tokenizer=tokenizer,
        )

    def read(self, path: Path) -> list[DocumentChunk]:
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported document type: {path}")
        picture = self._picture
        logger.info(
            "Convert start path=%s vlm=%s model=%s",
            path.name,
            picture.enabled,
            picture.model if picture.enabled else "-",
        )
        started = time.perf_counter()
        with log_vlm_http():
            result = self._converter.convert(str(path))
        convert_sec = time.perf_counter() - started
        described, skipped = log_picture_outcomes(result.document, vlm_enabled=picture.enabled)

        logger.info(
            "Convert done path=%s elapsed=%.1fs pictures_described=%s pictures_skipped=%s",
            path.name,
            convert_sec,
            described,
            skipped,
        )
        return list(
            self._document_chunker.chunk_document(result.document, path_name=path.name)
        )


@contextmanager
def log_vlm_http() -> Iterator[None]:
    """Log Docling VLM POSTs (they use ``requests``, not httpx)."""
    original = requests.Session.post

    def _logged(self: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        url_text = str(url)
        if not any(marker in url_text for marker in _VLM_URL_MARKERS):
            return original(self, url, *args, **kwargs)
        payload = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}
        model = payload.get("model", "?") if isinstance(payload, dict) else "?"
        started = time.perf_counter()
        logger.info("VLM request sent url=%s model=%s", url_text, model)
        try:
            response = original(self, url, *args, **kwargs)
        except Exception:
            logger.exception(
                "VLM request failed url=%s model=%s elapsed=%.1fs",
                url_text,
                model,
                time.perf_counter() - started,
            )
            raise
        description_chars = _vlm_response_chars(response)
        logger.info(
            "VLM response url=%s model=%s status=%s elapsed=%.1fs description_chars=%s",
            url_text,
            model,
            getattr(response, "status_code", "?"),
            time.perf_counter() - started,
            description_chars,
        )
        return response

    requests.Session.post = _logged  # type: ignore[method-assign]
    try:
        yield
    finally:
        requests.Session.post = original  # type: ignore[method-assign]


def _vlm_response_chars(response: Any) -> int:
    try:
        if not getattr(response, "ok", False):
            return 0
        data = response.json()
        if not isinstance(data, dict):
            return 0
        choices = data.get("choices") or []
        if not choices:
            return 0
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content") or ""
        return len(str(content))
    except Exception:
        return 0


def log_picture_outcomes(document: Any, *, vlm_enabled: bool) -> tuple[int, int]:
    pictures = getattr(document, "pictures", None)
    try:
        pictures = list(pictures) if pictures is not None else []
    except TypeError:
        pictures = []
    described = skipped = 0
    for picture in pictures:
        ref = getattr(picture, "self_ref", "?")
        text = picture_description(picture)
        if text:
            described += 1
            logger.debug(
                "VLM picture described ref=%s chars=%s",
                ref,
                len(text),
            )
            continue
        skipped += 1
        reason = (
            "below area threshold or API returned empty"
            if vlm_enabled
            else "VLM disabled"
        )
        logger.debug("VLM picture skipped ref=%s reason=%s", ref, reason)
    if not pictures:
        logger.debug("VLM pictures found=0")
    return described, skipped
