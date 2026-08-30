"""Parse resume header and labeled project tables without an LLM."""

from __future__ import annotations

import re
from typing import Any

_HEADER_LINES = 40
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

_FIO_LABEL = re.compile(
    r"^(?:фио|ф\.?\s*и\.?\s*о\.?|фамилия\s+имя\s+отчество)\s*[:\-–—]?\s*(.+)$",
    re.I,
)
_POSITION_HEADER = re.compile(r"^(?:желаемая\s+)?должность$", re.I)
_FIO_NAME = re.compile(
    r"^[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?(?:\s+[А-ЯЁ][а-яё]+){1,3}$"
)
_FIO_CAPS = re.compile(r"^[А-ЯЁ]{2,}(?:\s+[А-ЯЁ]{2,}){1,3}$")
_SKIP_HEADER_LINE = re.compile(
    r"^(должность|телефон|тел\.|моб|email|e-mail|почта|дата|возраст|город|"
    r"адрес|образование|опыт|гражданство|семейное|резюме)\b",
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
_DIRECTION_FROM_ROLE = (
    re.compile(r"по\s+направлению\s+([^,.;(\n]+)", re.I),
    re.compile(r"по\s+блоку\s+([^,.;(\n]+)", re.I),
)

_FIELD_TITLES = {
    "customer": "Заказчик",
    "duration": "Продолжительность проекта",
    "project_industry": "Отрасль проекта",
    "project_description": "Описание проекта",
    "project_position": "Роль на проекте",
    "work_performed": "Выполненные работы",
}
_LABELS_BY_LENGTH = tuple(
    sorted(_LABEL_TO_FIELD.items(), key=lambda item: len(item[0]), reverse=True)
)


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


def document_tables(document: Any) -> list[list[list[str]]]:
    """Structural Docling tables as cell grids (label/value layout survives better)."""
    grids: list[list[list[str]]] = []
    for table in getattr(document, "tables", None) or []:
        grid = _grid_from_table(table)
        if grid:
            grids.append(grid)
    return grids


def parse_header(text: str) -> dict[str, str | None]:
    """ФИО and должность from the start of the resume. ФИО may be unlabeled."""
    tokens = _header_tokens(text)
    name = None
    position = None
    pending_position = False
    for token in tokens:
        labeled_name = _FIO_LABEL.match(token)
        if labeled_name:
            name = _clean(labeled_name.group(1)) or name
            continue
        if _POSITION_HEADER.match(token):
            pending_position = True
            continue
        field, value = _line_field_value(token)
        if field == "project_position":
            continue
        if pending_position:
            position = _clean(token) or position
            pending_position = False
            continue
        if _is_header_position_line(token):
            position = _header_position_value(token) or position
            continue
        if not name and _looks_like_fio(token):
            name = token
    return {"candidate_name": name, "candidate_position": position}


def parse_projects(
    text: str,
    tables: list[list[list[str]]] | None = None,
) -> list[dict[str, str | None]]:
    """One dict per project from Docling grids, markdown tables or labeled lines."""
    projects: list[dict[str, str | None]] = []

    def add(item: dict[str, str | None]) -> None:
        if _is_junk(item):
            return
        replace_at: int | None = None
        for index, existing in enumerate(projects):
            existing_covers = _covers(existing, item)
            item_covers = _covers(item, existing)
            if existing_covers and item_covers:
                return
            if existing_covers:
                return
            if item_covers:
                replace_at = index
                break
        if replace_at is not None:
            projects[replace_at] = item
            return
        projects.append(item)

    for table in list(tables or []) + _markdown_tables(text):
        for item in _projects_from_table(table):
            add(item)
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


def infer_functional_direction(role: object, work: object = None) -> str | None:
    """Pull a direction from «по направлению» / «по блоку» without an LLM."""
    for source in (role, work):
        text = str(source or "")
        if not text:
            continue
        for pattern in _DIRECTION_FROM_ROLE:
            match = pattern.search(text)
            if match:
                return _clean(match.group(1))
    return None


def _header_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen_lines = 0
    for raw in text.splitlines():
        line = raw.strip().strip("*")
        if not line:
            continue
        seen_lines += 1
        if seen_lines > _HEADER_LINES:
            break
        if _TABLE_SEP.match(line):
            continue
        if _TABLE_ROW.match(line):
            for cell in _row_cells(line):
                if cell:
                    tokens.append(cell)
            continue
        tokens.append(line)
    return tokens


def _looks_like_fio(text: str) -> bool:
    candidate = text.strip().strip("*#")
    if not candidate or _SKIP_HEADER_LINE.match(candidate) or ":" in candidate:
        return False
    if "@" in candidate or any(ch.isdigit() for ch in candidate):
        return False
    folded = candidate.casefold()
    if "ооо" in folded or "http" in folded or "резюме" in folded:
        return False
    return bool(_FIO_NAME.match(candidate) or _FIO_CAPS.match(candidate))


def _is_header_position_line(line: str) -> bool:
    folded = line.casefold()
    if folded.startswith("должность на проекте"):
        return False
    return folded.startswith("должность") or folded.startswith("желаемая должность")


def _header_position_value(line: str) -> str | None:
    match = re.match(
        r"^(?:желаемая\s+)?должность(?!\s+на\s+проекте)\s*[:\-–—|]?\s*(.+)$",
        line,
        re.I,
    )
    if not match:
        return None
    return _clean(match.group(1))


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
                peek = index + 1
                while peek < len(lines) and not lines[peek].strip():
                    peek += 1
                nxt = lines[peek].strip() if peek < len(lines) else ""
                if (
                    rows
                    and nxt
                    and (_TABLE_ROW.match(nxt) or _TABLE_SEP.match(nxt))
                    and not _TABLE_SEP.match(nxt)
                    and len(_row_cells(nxt)) == len(rows[0])
                ):
                    index += 1
                    continue
                if rows and nxt and _TABLE_SEP.match(nxt):
                    index += 1
                    continue
                break
            if _TABLE_SEP.match(current):
                index += 1
                continue
            if not _TABLE_ROW.match(current):
                break
            cells = _row_cells(current)
            if any(cells):
                rows.append(cells)
            index += 1
        if rows:
            tables.append(_pad_rows(rows))
        else:
            index += 1
    return tables


def _row_cells(line: str) -> list[str]:
    return [cell.strip().strip("*") for cell in line.strip().strip("|").split("|")]


def _pad_rows(rows: list[list[str]]) -> list[list[str]]:
    width = max(len(row) for row in rows)
    return [row + [""] * (width - len(row)) for row in rows]


def _grid_from_table(table: Any) -> list[list[str]]:
    data = getattr(table, "data", None)
    grid = getattr(data, "grid", None) if data is not None else None
    if grid:
        rows: list[list[str]] = []
        for row in grid:
            cells = [
                " ".join(str(getattr(cell, "text", cell) or "").split())
                for cell in row
            ]
            if any(cells):
                rows.append(cells)
        return _pad_rows(rows) if rows else []
    export = getattr(table, "export_to_markdown", None)
    if callable(export):
        tables = _markdown_tables(str(export() or ""))
        return tables[0] if tables else []
    return []


def _projects_from_table(rows: list[list[str]]) -> list[dict[str, str | None]]:
    if not rows or not rows[0]:
        return []
    first_col_labels = [_field_key(row[0]) for row in rows]
    first_row_labels = [_field_key(cell) for cell in rows[0]]
    col_hits = sum(1 for key in first_col_labels if key)
    row_hits = sum(1 for key in first_row_labels if key)
    live_columns = [
        column
        for column in range(len(rows[0]))
        if any(_clean(row[column]) for row in rows)
    ]
    if len(live_columns) <= 1 and col_hits >= _MIN_PROJECT_LABELS:
        column = live_columns[0] if live_columns else 0
        return _projects_from_alternating_cells([row[column] for row in rows])
    if col_hits >= _MIN_PROJECT_LABELS and col_hits >= row_hits:
        return _projects_from_label_column(rows)
    if row_hits >= _MIN_PROJECT_LABELS:
        fields = first_row_labels
        projects = []
        for row in rows[1:]:
            if sum(1 for cell in row if _field_key(cell)) >= _MIN_PROJECT_LABELS:
                continue
            item = {field: None for field in _FIELD_TITLES}
            for column, field in enumerate(fields):
                if not field or column >= len(row):
                    continue
                item[field] = _cell_value(row[column])
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
            value = _cell_value(row[column] if column < len(row) else "")
            if current is None or current.get(field):
                if current:
                    projects.append(current)
                current = {key: None for key in _FIELD_TITLES}
            if value:
                current[field] = value
        if current:
            projects.append(current)
    return projects


def _projects_from_alternating_cells(cells: list[str]) -> list[dict[str, str | None]]:
    """Word often emits label and value on consecutive rows of one column."""
    current: dict[str, str | None] | None = None
    projects: list[dict[str, str | None]] = []
    index = 0
    while index < len(cells):
        field = _field_key(cells[index])
        if not field:
            index += 1
            continue
        value = None
        if index + 1 < len(cells) and not _field_key(cells[index + 1]):
            value = _cell_value(cells[index + 1])
            index += 2
        else:
            index += 1
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
    pending_field: str | None = None
    for raw in text.splitlines():
        line = raw.strip().strip("*")
        if _TABLE_ROW.match(line) or _TABLE_SEP.match(line):
            continue
        if not line:
            if current and pending_field is None:
                projects.append(current)
                current = None
            continue
        field, value = _line_field_value(line)
        if pending_field and not field:
            if current is None:
                current = {key: None for key in _FIELD_TITLES}
            current[pending_field] = _cell_value(line)
            pending_field = None
            continue
        if not field:
            pending_field = None
            continue
        if current is None or current.get(field):
            if current:
                projects.append(current)
            current = {key: None for key in _FIELD_TITLES}
        real = _cell_value(value)
        if real:
            current[field] = real
            pending_field = None
        else:
            pending_field = field
    if current:
        projects.append(current)
    return projects


def _line_field_value(line: str) -> tuple[str | None, str | None]:
    stripped = " ".join(line.split())
    whole = _field_key(stripped)
    if whole:
        return whole, None
    folded = stripped.casefold()
    for label, key in _LABELS_BY_LENGTH:
        if not folded.startswith(label):
            continue
        rest = stripped[len(label) :].lstrip(" \t:;–—-|")
        if rest:
            return key, rest
    return None, None


def _field_key(label: object) -> str | None:
    normalized = " ".join(str(label or "").casefold().split()).strip(" .:;-–—")
    return _LABEL_TO_FIELD.get(normalized)


def _clean(value: object) -> str | None:
    text = " ".join(str(value or "").split()).strip(" \t.,;")
    return text or None


def _cell_value(value: object) -> str | None:
    """Drop empty cells and copies of the field title (table header echoed as data)."""
    cleaned = _clean(value)
    if not cleaned or _field_key(cleaned) is not None:
        return None
    return cleaned


def _covers(full: dict[str, str | None], part: dict[str, str | None]) -> bool:
    """True if ``full`` has every filled field of ``part`` with the same value."""
    has_any = False
    for key, value in part.items():
        if not value:
            continue
        has_any = True
        if full.get(key) != value:
            return False
    return has_any


def _is_junk(item: dict[str, str | None]) -> bool:
    description = item.get("project_description") or ""
    position = item.get("project_position") or ""
    if _DATE_ONLY.match(description.strip()) or _DATE_ONLY.search(position):
        return True
    if _COMPANY_ONLY.search(description) and not _PROJECT_HINT.search(description):
        return True
    real = [
        item.get(key)
        for key in (
            "customer",
            "duration",
            "project_industry",
            "project_description",
            "project_position",
            "work_performed",
        )
        if _cell_value(item.get(key))
    ]
    return len(real) < _MIN_PROJECT_LABELS
