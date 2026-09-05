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
_TABLE_SEP_CHARS = re.compile(r"^[\s|:\-]+$")
_LINE_BULLET = re.compile(r"^(?:[-*•–—]+\s+|\d+[.)]\s+)")
_MARKDOWN_BOLD = re.compile(r"\*\*([^*]+)\*\*")

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
            if (
                _POSITION_HEADER.match(token)
                or _field_key(token)
                or _SKIP_HEADER_LINE.match(token)
            ):
                continue
            cleaned = _clean(token)
            if not cleaned or _looks_like_fio(cleaned):
                continue
            position = cleaned
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
        ident = _project_identity(item)
        for index, existing in enumerate(projects):
            same = bool(ident) and ident == _project_identity(existing)
            if not same and not _covers(existing, item) and not _covers(item, existing):
                continue
            projects[index] = _prefer(item, existing)
            return
        projects.append(item)

    for table in list(tables or []) + _markdown_tables(text):
        for item in _projects_from_table(table):
            add(item)
    for item in _projects_from_scattered_label_rows(_all_pipe_rows(text)):
        add(item)
    for item in _projects_from_blocks(text):
        add(item)
    return _drop_subset_projects(projects)


def format_project_text(project: dict[str, str | None]) -> str:
    lines = []
    for key, title in _FIELD_TITLES.items():
        value = _clean(project.get(key))
        if value:
            lines.append(f"{title}: {value}")
    return "\n".join(lines)


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
        if _is_table_sep(line):
            continue
        if _is_table_row(line):
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


def _is_table_sep(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and "-" in stripped and bool(_TABLE_SEP_CHARS.match(stripped))


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _is_table_sep(stripped):
        return False
    if stripped.startswith("|"):
        return True
    return stripped.count("|") >= 2


def _markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    current: list[list[str]] | None = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if _is_table_sep(stripped):
            continue
        if _is_table_row(stripped):
            cells = _row_cells(stripped)
            if not any(cells):
                continue
            if current is None:
                current = []
            current.append(cells)
            continue
        if current is not None:
            tables.append(_pad_rows(current))
            current = None
    if current is not None:
        tables.append(_pad_rows(current))
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
    label_col = _label_column(rows)
    col_hits = sum(
        1 for row in rows if label_col < len(row) and _field_key(row[label_col])
    )
    first_row_labels = [_field_key(cell) for cell in rows[0]]
    row_hits = sum(1 for key in first_row_labels if key)
    live_columns = [
        column
        for column in range(len(rows[0]))
        if any(column < len(row) and _clean(row[column]) for row in rows)
    ]
    if len(live_columns) <= 1 and col_hits >= _MIN_PROJECT_LABELS:
        column = live_columns[0] if live_columns else 0
        return _projects_from_alternating_cells(
            [row[column] if column < len(row) else "" for row in rows]
        )
    if _has_distinct_side_by_side(rows) and col_hits >= _MIN_PROJECT_LABELS:
        return _projects_from_label_column(rows, label_col)
    scattered = _projects_from_scattered_label_rows(rows)
    if scattered:
        return scattered
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
    cells = [cell for row in rows for cell in row if _clean(cell)]
    return _projects_from_alternating_cells(cells)


def _label_column(rows: list[list[str]]) -> int:
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return 0
    scores = [0] * width
    for row in rows:
        for index, cell in enumerate(row):
            if _field_key(cell):
                scores[index] += 1
    if not any(scores):
        return 0
    return max(range(width), key=lambda index: scores[index])


def _has_distinct_side_by_side(rows: list[list[str]]) -> bool:
    """True when one labeled row holds two different project values (wide table)."""
    for row in rows:
        field = None
        values: list[str] = []
        for cell in row:
            key = _field_key(cell)
            if key:
                field = key
                continue
            value = _cell_value(cell)
            if value:
                values.append(value)
        if field and len(_merge_related_values(values)) >= 2:
            return True
    return False


def _projects_from_label_column(
    rows: list[list[str]],
    label_col: int = 0,
) -> list[dict[str, str | None]]:
    """Each extra column is a project. Repeated labels start the next stacked project."""
    width = max((len(row) for row in rows), default=0)
    projects: list[dict[str, str | None]] = []
    for column in range(width):
        if column == label_col:
            continue
        current: dict[str, str | None] | None = None
        for row in rows:
            label = row[label_col] if label_col < len(row) else ""
            field = _field_key(label)
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


def _all_pipe_rows(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw in text.splitlines():
        if not _is_table_row(raw) or _is_table_sep(raw):
            continue
        cells = _row_cells(raw)
        if any(cells):
            rows.append(cells)
    return rows


def _projects_from_scattered_label_rows(
    rows: list[list[str]],
) -> list[dict[str, str | None]]:
    """Find a field label anywhere in the row; duplicated value columns collapse."""
    current: dict[str, str | None] | None = None
    projects: list[dict[str, str | None]] = []
    last_field: str | None = None
    for row in rows:
        field: str | None = None
        values: list[str] = []
        mixed_labels = False
        for cell in row:
            key = _field_key(cell)
            if key:
                if field is not None and key != field:
                    mixed_labels = True
                    break
                field = key
                continue
            value = _cell_value(cell)
            if value:
                values.append(value)
        if mixed_labels or not field:
            continue
        unique = _merge_related_values(values)
        if len(unique) >= 2:
            continue
        value = unique[0] if unique else None
        if not value:
            continue
        if current is not None and current.get(field):
            if last_field == field and _same_value(str(current[field]), value):
                if len(value) > len(current[field] or ""):
                    current[field] = value
                continue
            projects.append(current)
            current = {key: None for key in _FIELD_TITLES}
        if current is None:
            current = {key: None for key in _FIELD_TITLES}
        current[field] = value
        last_field = field
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
        if _is_table_row(line) or _is_table_sep(line):
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
    raw = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = []
    for line in raw.split("\n"):
        stripped = _LINE_BULLET.sub("", line.strip())
        stripped = _MARKDOWN_BOLD.sub(r"\1", stripped)
        if stripped:
            parts.append(stripped)
    text = " ".join(" ".join(parts).split()).strip(" \t.,;*")
    return text or None


def _same_value(left: str, right: str) -> bool:
    first = _norm_field(left)
    second = _norm_field(right)
    if not first or not second:
        return False
    return first == second or first in second or second in first


def _merge_related_values(values: list[str]) -> list[str]:
    merged: list[str] = []
    for value in values:
        found = False
        for index, existing in enumerate(merged):
            if not _same_value(value, existing):
                continue
            if len(value) > len(existing):
                merged[index] = value
            found = True
            break
        if not found:
            merged.append(value)
    return merged


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
        if _norm_field(full.get(key)) != _norm_field(value):
            return False
    return has_any


def _norm_field(value: object) -> str:
    text = " ".join(str(value or "").split())
    text = text.replace("\\\\", "/").replace("\\", "/")
    return text.casefold().strip(" \t.,;:!?…")


def _project_identity(item: dict[str, str | None]) -> str:
    parts = [
        _norm_field(item.get("customer")),
        _norm_field(item.get("project_description")),
        _norm_field(item.get("project_position")),
    ]
    return "|".join(parts) if any(parts) else ""


def _filled_count(item: dict[str, str | None]) -> int:
    return sum(1 for value in item.values() if value)


def _prefer(
    left: dict[str, str | None],
    right: dict[str, str | None],
) -> dict[str, str | None]:
    left_n = _filled_count(left)
    right_n = _filled_count(right)
    if left_n != right_n:
        return left if left_n > right_n else right
    left_len = sum(len(value or "") for value in left.values())
    right_len = sum(len(value or "") for value in right.values())
    return left if left_len >= right_len else right


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


def _drop_subset_projects(
    projects: list[dict[str, str | None]],
) -> list[dict[str, str | None]]:
    """Drop fragments that are a proper subset of a fuller parse of the same project."""
    kept: list[dict[str, str | None]] = []
    for index, item in enumerate(projects):
        item_n = _filled_count(item)
        drop = False
        for other_index, other in enumerate(projects):
            if other_index == index:
                continue
            other_n = _filled_count(other)
            if other_n > item_n and _covers(other, item):
                drop = True
                break
            if (
                other_n == item_n
                and other_index < index
                and _covers(other, item)
                and _covers(item, other)
            ):
                drop = True
                break
        if not drop:
            kept.append(item)
    return kept
