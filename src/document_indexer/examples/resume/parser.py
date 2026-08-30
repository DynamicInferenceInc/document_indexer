"""Parse resume header and labeled project tables without an LLM."""

from __future__ import annotations

import re
from typing import Any

_HEADER_LINES = 20
_MIN_PROJECT_LABELS = 2

_LABEL_TO_FIELD = {
    "заказчик": "customer",
    "продолжительность проекта": "duration",
    "продолжительность": "duration",
    "отрасль проекта": "project_industry",
    "отрасль": "project_industry",
    "описание проекта": "project_description",
    "продукт": "project_description",
    "роль на проекте": "project_position",
    "должность на проекте": "project_position",
    "выполненные работы": "work_performed",
    "обязанности": "work_performed",
}

_FIO_LABEL = re.compile(r"^(?:фио|ф\.?\s*и\.?\s*о\.?|фамилия\s+имя\s+отчество)\s*[:\-–—]?\s*(.+)$", re.I)
_POSITION_LABEL = re.compile(
    r"^(?:желаемая\s+)?должность\s*[:\-–—]\s*(.+)$",
    re.I,
)
_FIO_NAME = re.compile(
    r"^[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?(?:\s+[А-ЯЁ][а-яё]+){1,3}$"
)
_FIO_CAPS = re.compile(r"^[А-ЯЁ]{2,}(?:\s+[А-ЯЁ]{2,}){1,3}$")
_SKIP_HEADER_LINE = re.compile(
    r"^(должность|телефон|тел\.|моб|email|e-mail|почта|дата|возраст|город|"
    r"адрес|образование|опыт|гражданство|семейное)\b",
    re.I,
)
_DATE_ONLY = re.compile(
    r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}\s*[-–—]\s*\d{1,2}[./]\d{1,2}[./]\d{2,4}$"
    r"|^\d{4}\s*[-–—]\s*(\d{4}|текущ|н\.?\s*в|настоящ)",
    re.I,
)
_COMPANY_ONLY = re.compile(r"\bг\.\s|.+,\s*(россия|russia)\s*$", re.I)
_PROJECT_HINT = re.compile(r"(проект|внедрен|переход|миграц|автоматиз|интеграц)", re.I)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_KEY_VALUE = re.compile(r"^([^:\n]{2,80})\s*[:\-–—]\s+(.+)$")

_FIELD_TITLES = {
    "customer": "Заказчик",
    "duration": "Продолжительность проекта",
    "project_industry": "Отрасль проекта",
    "project_description": "Описание проекта",
    "project_position": "Роль на проекте",
    "work_performed": "Выполненные работы",
}


def document_text(document: Any) -> str:
    """Plain text of a Docling document, or ``document`` if it is already text."""
    if isinstance(document, str):
        return document.strip()
    for name in ("export_to_markdown", "export_to_text"):
        export = getattr(document, name, None)
        if not callable(export):
            continue
        try:
            text = export()
        except TypeError:
            text = export(strict=False)
        if text:
            return str(text).strip()
    return ""


def parse_header(text: str) -> dict[str, str | None]:
    """ФИО and должность from the start of the resume. ФИО may be unlabeled."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip().strip("*").strip()
        if not line:
            continue
        lines.append(line)
        if len(lines) >= _HEADER_LINES:
            break
    name = None
    position = None
    for line in lines:
        labeled_name = _FIO_LABEL.match(line)
        if labeled_name:
            name = _clean(labeled_name.group(1)) or name
            continue
        labeled_position = _POSITION_LABEL.match(line)
        if labeled_position:
            position = _clean(labeled_position.group(1)) or position
            continue
    if not name:
        name = _unlabeled_fio("\n".join(lines))
    return {"candidate_name": name, "candidate_position": position}


def parse_projects(text: str) -> list[dict[str, str | None]]:
    """One dict per project from markdown tables or labeled blocks."""
    projects: list[dict[str, str | None]] = []
    seen: set[str] = set()

    def add(item: dict[str, str | None]) -> None:
        key = _project_key(item)
        if key in seen or _is_junk(item):
            return
        seen.add(key)
        projects.append(item)

    for table in _markdown_tables(text):
        for item in _projects_from_table(table):
            add(item)
    if not projects:
        for item in _projects_from_blocks(text):
            add(item)
    return projects


def format_project_text(project: dict[str, str | None]) -> str:
    lines = []
    for key, title in _FIELD_TITLES.items():
        value = _clean(project.get(key))
        if value:
            lines.append(f"{title}: {value}")
    return "\n".join(lines)


def _unlabeled_fio(head: str) -> str | None:
    seen = 0
    for raw in head.splitlines():
        line = raw.strip().strip("*#").strip()
        if not line:
            continue
        seen += 1
        if seen > _HEADER_LINES:
            break
        if _SKIP_HEADER_LINE.match(line) or _FIO_LABEL.match(line) or ":" in line:
            continue
        if "@" in line or any(ch.isdigit() for ch in line):
            continue
        if "ооо" in line.casefold() or "http" in line.casefold():
            continue
        if _FIO_NAME.match(line) or _FIO_CAPS.match(line):
            return line
    return None


def _markdown_tables(text: str) -> list[list[list[str]]]:
    lines = text.splitlines()
    tables: list[list[list[str]]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not _TABLE_ROW.match(stripped):
            index += 1
            continue
        rows: list[list[str]] = []
        while index < len(lines):
            current = lines[index].strip()
            if not current:
                break
            if _TABLE_SEP.match(current):
                index += 1
                continue
            if not _TABLE_ROW.match(current):
                break
            cells = [cell.strip() for cell in current.strip("|").split("|")]
            if any(cells):
                rows.append(cells)
            index += 1
        if rows:
            tables.append(_pad_rows(rows))
        else:
            index += 1
    return tables


def _pad_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _projects_from_table(rows: list[list[str]]) -> list[dict[str, str | None]]:
    if not rows or not rows[0]:
        return []
    first_col_labels = [_field_key(row[0]) for row in rows]
    first_row_labels = [_field_key(cell) for cell in rows[0]]
    col_hits = sum(1 for key in first_col_labels if key)
    row_hits = sum(1 for key in first_row_labels if key)
    if col_hits >= _MIN_PROJECT_LABELS and col_hits >= row_hits:
        return _projects_from_label_column(rows)
    if row_hits >= _MIN_PROJECT_LABELS:
        fields = first_row_labels
        projects = []
        for row in rows[1:]:
            item = {field: None for field in _FIELD_TITLES}
            for column, field in enumerate(fields):
                if not field or column >= len(row):
                    continue
                item[field] = _clean(row[column])
            projects.append(item)
        return projects
    return []


def _projects_from_label_column(rows: list[list[str]]) -> list[dict[str, str | None]]:
    """Each extra column is a project. Repeated labels start the next stacked project."""
    width = len(rows[0])
    projects: list[dict[str, str | None]] = []
    for column in range(1, width):
        current: dict[str, str | None] | None = None
        for row in rows:
            field = _field_key(row[0])
            if not field:
                continue
            value = _clean(row[column] if column < len(row) else "")
            if current is None or current.get(field):
                if current:
                    projects.append(current)
                current = {key: None for key in _FIELD_TITLES}
            if value:
                current[field] = value
        if current:
            projects.append(current)
    return projects


def _projects_from_blocks(text: str) -> list[dict[str, str | None]]:
    current: dict[str, str | None] | None = None
    projects: list[dict[str, str | None]] = []
    for raw in text.splitlines():
        match = _KEY_VALUE.match(raw.strip())
        if not match:
            if current and not raw.strip():
                projects.append(current)
                current = None
            continue
        field = _field_key(match.group(1))
        if not field:
            continue
        if current is None or current.get(field):
            if current:
                projects.append(current)
            current = {key: None for key in _FIELD_TITLES}
        current[field] = _clean(match.group(2))
    if current:
        projects.append(current)
    return projects


def _field_key(label: object) -> str | None:
    normalized = " ".join(str(label or "").casefold().split()).strip(" .:;-–—")
    return _LABEL_TO_FIELD.get(normalized)


def _clean(value: object) -> str | None:
    text = " ".join(str(value or "").split()).strip(" \t.,;")
    return text or None


def _project_key(item: dict[str, str | None]) -> str:
    return "|".join(
        str(item.get(key) or "").casefold()
        for key in ("customer", "project_description", "project_position")
    )


def _is_junk(item: dict[str, str | None]) -> bool:
    description = item.get("project_description") or ""
    position = item.get("project_position") or ""
    if _DATE_ONLY.match(description.strip()) or _DATE_ONLY.search(position):
        return True
    if _COMPANY_ONLY.search(description) and not _PROJECT_HINT.search(description):
        return True
    return not any(
        item.get(key)
        for key in (
            "customer",
            "project_industry",
            "project_description",
            "project_position",
            "work_performed",
        )
    )
