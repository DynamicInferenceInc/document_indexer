"""Resume parser and project chunker."""

from __future__ import annotations

from types import SimpleNamespace

from document_indexer.resume import ResumeProjectChunker, load_resume_sample
from document_indexer.resume.parser import parse_header, parse_projects


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


def test_header_reads_unlabeled_fio_in_two_column_hat() -> None:
    text = """
# Резюме консультанта

| | Цырлин Игорь |
| Должность | Ведущий консультант |
"""
    header = parse_header(text)
    assert header["candidate_name"] == "Цырлин Игорь"
    assert header["candidate_position"] == "Ведущий консультант"


def test_projects_from_one_column_then_two_column_tables() -> None:
    """Docling often emits the first projects as label/value on consecutive rows."""
    text = """
| Заказчик |
| Группа агропредприятий РЕСУРС |
| Продолжительность проекта |
| 05.2025 – наст.время |
| Отрасль проекта |
| Производство продуктов питания |
| Описание проекта |
| Внедрение подсистемы Казначейство 1С: ERP Управление холдингом |
| Роль на проекте |
| Консультант по направлению Казначейство |
| Выполненные работы |
| Подготовка функциональных спецификаций |

| Заказчик |
| JTI |
| Продолжительность проекта |
| 06.2024 – наст.время |
| Отрасль проекта |
| Производство табачных изделий |
| Описание проекта |
| Внедрение 1С: ERP Управление холдингом |
| Роль на проекте |
| Консультант по направлению Управленческий учет |
| Выполненные работы |
| Подготовка спецификаций |

| Заказчик | АО БТК Групп |
| Продолжительность проекта | 03.2024 – 06.2024 |
| Отрасль проекта | Легкая промышленность |
| Описание проекта | Внедрение 1С: ERP 2.5 |
| Роль на проекте | Консультант по направлению Управленческий учет |
| Выполненные работы | Разработка проектных решений |

| Заказчик | Группа агропредприятий РЕСУРС |
| Продолжительность проекта | 09.2022 – 09.2023 |
| Отрасль проекта | Производство продуктов питания |
| Описание проекта | Внедрение SAP S/4 HANA В ТК РЕСУРС-ЮГ |
| Роль на проекте | Архитектор по блоку Финансы (Бухгалтерский и налоговый учет) |
| Выполненные работы | Разработка архитектуры решения |
"""
    projects = parse_projects(text)
    assert [item["customer"] for item in projects] == [
        "Группа агропредприятий РЕСУРС",
        "JTI",
        "АО БТК Групп",
        "Группа агропредприятий РЕСУРС",
    ]
    assert projects[0]["project_description"].startswith("Внедрение подсистемы Казначейство")
    assert projects[3]["project_description"].startswith("Внедрение SAP S/4 HANA")


def test_chunker_leaves_functional_direction_for_llm() -> None:
    document = SimpleNamespace(export_to_markdown=load_resume_sample)
    chunks = ResumeProjectChunker().chunk_document(document, path_name="cv.md")
    assert "functional_direction" not in chunks[0].extra_fields
    assert chunks[0].extra_fields["project_position"] == (
        "Консультант по направлению Казначейство"
    )


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


def test_projects_drop_header_echo_and_incomplete_duplicates() -> None:
    """Docling often re-emits the same table without Заказчик plus a label-only row."""
    text = """
| Заказчик | Группа агропредприятий РЕСУРС | Норникель | СИБУР |
| Продолжительность проекта | 09.2022 – 09.2023 | 09.2017 – 12.2020 | 02.2014 – 06.2015 |
| Отрасль проекта | Производство продуктов питания | Металлургия | Химическая промышленность |
| Описание проекта | Внедрение SAP S/4 HANA В ТК РЕСУРС-ЮГ | Тиражирование SAP ERP в НПР | Внедрение ERP на базе SAP |
| Роль на проекте | Архитектор по блоку Финансы | Руководитель группы Планирование | Консультант |
| Выполненные работы | Разработка архитектуры | Новый функционал АСУ Бюджет | Миграция НСИ |

| Заказчик |
| Заказчик |
| Отрасль проекта |
| Отрасль проекта |
| Описание проекта |
| Описание проекта |
| Роль на проекте |
| Роль на проекте |
| Выполненные работы |
| Выполненные работы |

| Заказчик |
| Группа агропредприятий РЕСУРС |
| Продолжительность проекта |
| 05.2025 – наст.время |
| Отрасль проекта |
| Производство продуктов питания |
| Описание проекта |
| Внедрение подсистемы Казначейство 1С: ERP Управление холдингом |
| Роль на проекте |
| Консультант по направлению Казначейство |
| Выполненные работы |
| Подготовка функциональных спецификаций |

Продолжительность проекта: 09.2017 – 12.2020
Отрасль проекта: Металлургия
Описание проекта: Тиражирование SAP ERP в НПР
Роль на проекте: Руководитель группы Планирование

Продолжительность проекта: 02.2014 – 06.2015
Отрасль проекта: Химическая промышленность
Описание проекта: Внедрение ERP на базе SAP
Роль на проекте: Консультант
"""
    projects = parse_projects(text)
    customers = [item["customer"] for item in projects]
    descriptions = [item["project_description"] for item in projects]
    assert customers == [
        "Группа агропредприятий РЕСУРС",
        "Норникель",
        "СИБУР",
        "Группа агропредприятий РЕСУРС",
    ]
    assert descriptions[0].startswith("Внедрение SAP S/4 HANA")
    assert descriptions[3].startswith("Внедрение подсистемы Казначейство")
    assert all(item["customer"] not in {None, "Заказчик"} for item in projects)
    assert all(item["project_description"] != "Описание проекта" for item in projects)
    assert len(projects) == 4


def test_projects_merge_when_work_performed_only_differs_by_markdown_dashes() -> None:
    """Docling emits the same project as prose and as a markdown list."""
    text = """
Заказчик: ПАО Северсталь
Отрасль проекта: Металлургия
Описание проекта: Внедрение системы SAP S4/HANA
Роль на проекте: Функциональный консультант SAP MM
Выполненные работы: Область решений: Планирование потребностей Разработка процесса контроля

Заказчик: ПАО Северсталь
Отрасль проекта: Металлургия
Описание проекта: Внедрение системы SAP S4/HANA
Роль на проекте: Функциональный консультант SAP MM
Выполненные работы:
- Область решений: Планирование потребностей
- Разработка процесса контроля
"""
    projects = parse_projects(text)
    assert len(projects) == 1
    assert projects[0]["customer"] == "ПАО Северсталь"
    assert projects[0]["project_description"] == "Внедрение системы SAP S4/HANA"
    assert "Область решений" in (projects[0]["work_performed"] or "")
    assert "- " not in (projects[0]["work_performed"] or "")


def test_projects_from_broken_docling_grid_like_vlasova() -> None:
    """Docling often emits labels off column 0, duplicated values, unclosed pipes."""
    text = """
|                          | Елена Власова                          |
|--------------------------|----------------------------------------|
| Должность                | Должность                              | Руководитель проектов
| Офис                     | ООО «1С-Перспектива», Россия, Москва   |

| Краткие профессиональные сведения
| Руководитель проектов, Архитектор решений (финансовое направление)

| Опыт работы на проектах SAP (подробно)
|                          | Заказчик                               | Телекоммуникационная компания, Финляндия | Телекоммуникационная компания, Финляндия
| Продолжительность проекта | 4  месяца                              | 4  месяца

| Отрасль проекта          | Телекоммуникации                       | Телекоммуникации |
| Описание проекта         | Тиражирование решения SAP S/4HANA, запуск релиза | Тиражирование решения SAP
| Роль на проекте          | Руководитель проектного офиса          | Руководитель проектного офиса
| Выполненные работы       | Планирование и контроль исполнения работ по проекту | Планирование и контроль исполнения работ по проекту

| Заказчик                 | Телекоммуникационная компания, Финляндия | Телекоммуникационная компания, Финляндия
| Продолжительность проекта | 20  месяцев                            | 20  месяцев
| Отрасль проекта          | Телекоммуникации
| Описание проекта         | Разработка ИТ решения для поддержки трансформации бизнеса | Разработка ИТ решения для поддержки трансформации бизнеса
| Роль на проекте          | Руководитель интеграционного направления «Управление  данными»
| Выполненные работы       | Организация перехода в продуктивную эксплуатацию (cutover)

| Заказчик                 | **Крупный российский банк**
| Продолжительность проекта | 9  месяцев                             | 9  месяцев |
| Отрасль проекта          | Банковская                             | Банковская
| Описание проекта         | Разработка и внедрения решения по управлению командировками на базе SAP S/4 HANA
| Роль на проекте          | Руководитель проекта                   | Руководитель проекта
| Выполненные работы       | Организация и отслеживание исполнения проектных работ
"""
    header = parse_header(text)
    assert header["candidate_name"] == "Елена Власова"
    assert header["candidate_position"] == "Руководитель проектов"

    projects = parse_projects(text)
    assert [item["customer"] for item in projects] == [
        "Телекоммуникационная компания, Финляндия",
        "Телекоммуникационная компания, Финляндия",
        "Крупный российский банк",
    ]
    assert projects[0]["duration"] == "4 месяца"
    assert "S/4HANA" in (projects[0]["project_description"] or "")
    assert projects[0]["project_position"] == "Руководитель проектного офиса"
    assert projects[1]["duration"] == "20 месяцев"
    assert "трансформации бизнеса" in (projects[1]["project_description"] or "")
    assert projects[2]["project_industry"] == "Банковская"
    assert "командировками" in (projects[2]["project_description"] or "")

    chunks = ResumeProjectChunker().chunk_document(
        SimpleNamespace(export_to_markdown=lambda: text),
        path_name="Власова_CV_RU.docx",
    )
    assert [chunk.chunk_type for chunk in chunks] == ["project", "project", "project"]

