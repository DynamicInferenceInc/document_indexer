"""Resume parser and project chunker."""

from __future__ import annotations

from types import SimpleNamespace

from document_indexer.examples.resume import ResumeProjectChunker, load_resume_sample
from document_indexer.examples.resume.parser import parse_header, parse_projects


def test_header_reads_unlabeled_fio_and_position() -> None:
    text = "Иванов Иван Иванович\nДолжность: ведущий консультант\nНавыки: 1С"
    header = parse_header(text)
    assert header["candidate_name"] == "Иванов Иван Иванович"
    assert header["candidate_position"] == "ведущий консультант"


def test_header_reads_labeled_fio() -> None:
    text = "ФИО: Петров Пётр\nЖелаемая должность: разработчик"
    header = parse_header(text)
    assert header["candidate_name"] == "Петров Пётр"
    assert header["candidate_position"] == "разработчик"


def test_projects_from_column_table() -> None:
    projects = parse_projects(load_resume_sample())
    assert len(projects) == 2
    assert projects[0]["customer"] == "Банк"
    assert projects[0]["project_industry"] == "банковский сектор"
    assert projects[0]["project_position"] == "Консультант по направлению Казначейство"
    assert projects[1]["project_description"] == (
        "модернизация производственного учёта на 1С:ERP"
    )


def test_projects_from_stacked_label_value_table() -> None:
    """Resumes often stack projects as one 2-column table; a repeated label starts the next."""
    text = """
| Заказчик | Северсталь |
| Продолжительность проекта | 01.2010 – 12.2011 |
| Отрасль проекта | Металлургия |
| Описание проекта | Внедрение SAP ERP |
| Роль на проекте | Консультант MM |
| Выполненные работы | Настройка закупок |
| --- | --- |
| Заказчик | Газпром добыча Надым |
| Продолжительность проекта | 01.2008 – 12.2008 |
| Отрасль проекта | Нефть и газ |
| Описание проекта | Сопровождение ИУС ПХД |
| Роль на проекте | Руководитель проекта |
| Выполненные работы | Услуги по сопровождению оказаны |
| --- | --- |
| Заказчик | Газпром добыча Надым |
| Продолжительность проекта | 09.2006 – 12.2007 |
| Отрасль проекта | Нефть и газ |
| Описание проекта | Разработка и внедрение ИУС ПХД на базе SAP |
| Роль на проекте | Руководитель проекта |
| Выполненные работы | Разработана и внедрена ИУС ПХД |
"""
    projects = parse_projects(text)
    assert [item["project_description"] for item in projects] == [
        "Внедрение SAP ERP",
        "Сопровождение ИУС ПХД",
        "Разработка и внедрение ИУС ПХД на базе SAP",
    ]
    assert projects[0]["customer"] == "Северсталь"
    assert projects[1]["customer"] == "Газпром добыча Надым"
    assert projects[2]["duration"] == "09.2006 – 12.2007"
    assert projects[1]["project_position"] == "Руководитель проекта"


def test_projects_from_labeled_blocks() -> None:
    text = """
Иванов Иван

Заказчик: Альфа
Отрасль проекта: банки
Описание проекта: внедрение ERP
Роль на проекте: консультант
Выполненные работы: настройка НСИ

Заказчик: Бета
Роль на проекте: разработчик
Описание проекта: личный кабинет
"""
    projects = parse_projects(text)
    assert len(projects) == 2
    assert projects[0]["customer"] == "Альфа"
    assert projects[1]["project_position"] == "разработчик"


def test_chunker_one_project_one_chunk() -> None:
    document = SimpleNamespace(export_to_markdown=load_resume_sample)
    chunks = ResumeProjectChunker().chunk_document(document, path_name="cv.md")
    assert len(chunks) == 2
    assert all(chunk.chunk_type == "project" for chunk in chunks)
    assert "Заказчик: Банк" in chunks[0].text
    assert "Роль на проекте: Консультант по направлению Казначейство" in chunks[0].text
    assert chunks[0].extra_fields["candidate_name"] == "Иванов Иван Иванович"
    assert chunks[0].extra_fields["candidate_position"] == "ведущий консультант"
    assert chunks[0].extra_fields["project_industry"] == "банковский сектор"
    assert chunks[1].extra_fields["candidate_name"] == chunks[0].extra_fields["candidate_name"]


def test_chunker_without_projects_uses_sliding_window() -> None:
    text = (
        "Сидоров Сидор\n"
        "Должность: Разработчик-стажер\n\n"
        "Навыки: Python, SQL. Образование: бакалавр.\n"
        + ("опыт разработки внутренних сервисов. " * 80)
    )
    document = SimpleNamespace(export_to_markdown=lambda: text)
    chunks = ResumeProjectChunker(window_chars=200, window_overlap=40).chunk_document(
        document,
        path_name="cv.md",
    )
    assert len(chunks) >= 2
    assert all(chunk.chunk_type == "prose" for chunk in chunks)
    for chunk in chunks:
        assert chunk.extra_fields["candidate_name"] == "Сидоров Сидор"
        assert chunk.extra_fields["candidate_position"] == "Разработчик-стажер"
        assert "project_industry" not in chunk.extra_fields
