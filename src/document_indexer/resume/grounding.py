"""Keep only LLM values that actually occur in the resume text."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_EMPTY = (None, "", [], "null", "None")
_NON_WORD = re.compile(r"[^0-9a-zа-я]+")
_TOKEN = re.compile(r"[0-9a-zа-я]+")
_FRAGMENT_SEP = re.compile(r"\s*;\s*|\n+")
_SAP = re.compile(r"\bSAP\b|S\s*/?\s*4\s*/?\s*HANA", re.I)
_ONE_C = re.compile(r"1[СCсc]")
_TRANSITION = re.compile(
    r"(?:переход\w*|миграц\w*)\s+с\s+(?P<src>.+?)\s+на\s+(?P<dst>.+)",
    re.I | re.S,
)
_MAX_DIRECTION_CHARS = 60


def normalize_text(value: object) -> str:
    """Lowercase, ё→е, Latin ``1C``→``1С``, punctuation → single spaces."""
    text = str(value or "").casefold().replace("ё", "е")
    text = re.sub(r"1c", "1с", text)
    return " ".join(_NON_WORD.sub(" ", text).split())


def text_tokens(normalized: str) -> list[str]:
    """Tokens worth checking: words of 3+ chars or numbers of 2+ digits."""
    out: list[str] = []
    for token in _TOKEN.findall(normalized):
        if len(token) >= 3 or (token.isdigit() and len(token) >= 2):
            out.append(token)
    return out


def clean_value(value: object) -> str | None:
    if value in _EMPTY:
        return None
    text = " ".join(str(value).split()).strip(" \t.,;*")
    return text or None


class Grounder:
    """Check LLM output against one resume. Counts what it had to drop."""

    def __init__(self, source_text: str, *, min_ratio: float = 0.85) -> None:
        self._normalized = normalize_text(source_text)
        self._tokens = set(text_tokens(self._normalized))
        self._min_ratio = min_ratio
        self.dropped = 0
        self.dropped_values: list[str] = []

    def is_grounded(self, value: object) -> bool:
        text = clean_value(value)
        if not text:
            return False
        normalized = normalize_text(text)
        if not normalized:
            return False
        if normalized in self._normalized:
            return True
        tokens = text_tokens(normalized)
        if not tokens:
            return False
        hits = sum(1 for token in tokens if token in self._tokens)
        return hits / len(tokens) >= self._min_ratio

    def ground(self, value: object, *, field: str = "") -> str | None:
        """Return the cleaned value or ``None`` (and count) when it is not in the text."""
        text = clean_value(value)
        if not text:
            return None
        if self.is_grounded(text):
            return text
        self._drop(field, text)
        return None

    def ground_fragments(self, value: object, *, field: str = "") -> str | None:
        """Per-fragment check for lists (``;``-separated). Keeps the grounded ones."""
        text = clean_value(value)
        if not text:
            return None
        kept: list[str] = []
        for fragment in _FRAGMENT_SEP.split(text):
            piece = clean_value(fragment)
            if not piece:
                continue
            if self.is_grounded(piece):
                kept.append(piece)
            else:
                self._drop(field, piece)
        return "; ".join(kept) or None

    def _drop(self, field: str, text: str) -> None:
        self.dropped += 1
        self.dropped_values.append(f"{field}={text}" if field else text)
        logger.warning(
            "Dropped ungrounded LLM value field=%s value=%r",
            field or "?",
            text[:120],
        )


def clean_direction(value: object) -> str | None:
    """Short classification label; not checked against the text."""
    text = clean_value(value)
    if not text or len(text) > _MAX_DIRECTION_CHARS:
        return None
    return text


def normalize_platform(value: object) -> str | None:
    text = clean_value(value)
    if not text:
        return None
    folded = text.casefold().replace(" ", "")
    if folded in {"1с", "1c"} or folded.startswith("1с") or folded.startswith("1c"):
        return "1С"
    if "sap" in folded or "hana" in folded:
        return "SAP"
    return None


def infer_solution_platform(*parts: object) -> str | None:
    """Return 1С or SAP when the project text names exactly one; else None for the LLM."""
    text = "\n".join(str(part) for part in parts if part)
    if not text.strip():
        return None
    transition = _TRANSITION.search(text)
    if transition:
        target = _platforms_in(transition.group("dst"))
        if len(target) == 1:
            return next(iter(target))
    found = _platforms_in(text)
    if len(found) == 1:
        return next(iter(found))
    return None


def _platforms_in(text: str) -> set[str]:
    found: set[str] = set()
    if _SAP.search(text):
        found.add("SAP")
    if _ONE_C.search(text):
        found.add("1С")
    return found
