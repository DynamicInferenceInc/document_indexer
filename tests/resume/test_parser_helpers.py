"""Residual text, experience hints and sectioning for the LLM steps."""

from __future__ import annotations

from document_indexer.resume.parser import (
    has_experience_hints,
    merge_projects,
    parse_header,
    parse_projects,
    residual_text,
    same_project,
    split_sections,
)

TEMPLATE = """
Иванов Иван Иванович
Должность: ведущий консультант

| Заказчик | Банк |
| Продолжительность проекта | 12 месяцев |
| Отрасль проекта | банковский сектор |
| Описание проекта | внедрение 1С:ЗУП |
| Роль на проекте | Консультант по направлению Казначейство |
| Выполненные работы | настройка казначейства |

Дополнительно
2015 – 2018  ООО «Вектор», аналитик
Внедрение CRM для отдела продаж
"""


def test_residual_drops_header_and_parsed_project_lines() -> None:
    header = parse_header(TEMPLATE)
    projects = parse_projects(TEMPLATE)
    residual = residual_text(TEMPLATE, header, projects)
    assert "Иванов Иван Иванович" not in residual
    assert "ведущий консультант" not in residual
    assert "Банк" not in residual
    assert "настройка казначейства" not in residual
    assert "Заказчик" not in residual
    assert "ООО «Вектор», аналитик" in residual
    assert "Внедрение CRM для отдела продаж" in residual


def test_residual_of_fully_parsed_template_is_tiny() -> None:
    text = TEMPLATE.split("Дополнительно")[0]
    residual = residual_text(text, parse_header(text), parse_projects(text))
    assert residual == ""


def test_experience_hints() -> None:
    assert has_experience_hints("2015 – 2018  ООО «Вектор», аналитик")
    assert has_experience_hints("Работал в компании как руководитель проекта, заказчик доволен")
    assert not has_experience_hints("Навыки: Python, SQL")
    assert not has_experience_hints("")


def test_split_sections_respects_paragraphs_and_overlap() -> None:
    paragraphs = [f"Абзац {index} " + "текст " * 20 for index in range(10)]
    text = "\n\n".join(paragraphs)
    sections = split_sections(text, max_chars=400, overlap_chars=150)
    assert len(sections) > 1
    assert all(len(section) <= 400 for section in sections)
    assert sections[0].startswith("Абзац 0")
    assert sections[-1].rstrip().endswith("текст")
    joined = "\n\n".join(sections)
    for paragraph in paragraphs:
        assert paragraph.strip() in joined
    # overlap: the last paragraph of a section reappears in the next one
    first_tail = sections[0].split("\n\n")[-1]
    assert first_tail in sections[1]


def test_split_sections_short_text_and_huge_paragraph() -> None:
    assert split_sections("короткий", max_chars=100) == ["короткий"]
    assert split_sections("   ", max_chars=100) == []
    sections = split_sections("x" * 1000, max_chars=300)
    assert len(sections) == 4
    assert "".join(sections) == "x" * 1000


def test_merge_projects_and_same_project() -> None:
    parser = [
        {
            "customer": "Банк",
            "duration": None,
            "project_industry": None,
            "project_description": "внедрение 1С:ЗУП",
            "project_position": "Консультант",
            "work_performed": None,
        }
    ]
    llm = [
        {
            "customer": "ООО «Вектор»",
            "duration": "2015 – 2018",
            "project_industry": None,
            "project_description": "Внедрение CRM",
            "project_position": "аналитик",
            "work_performed": None,
        },
        {
            "customer": "Банк",
            "duration": None,
            "project_industry": None,
            "project_description": "внедрение 1С:ЗУП",
            "project_position": "Консультант",
            "work_performed": None,
        },
        {"customer": None, "duration": None, "project_industry": None,
         "project_description": "только описание", "project_position": None,
         "work_performed": None},
    ]
    merged = merge_projects(parser, llm)
    assert [item["customer"] for item in merged] == ["Банк", "ООО «Вектор»"]
    assert same_project(merged[0], parser[0])
    assert not same_project(merged[1], parser[0])
