"""Parser → LLM projects → refine → experience → prose, with a scripted chat."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.config import ResumeSettings
from document_indexer.resume import (
    INDEX_VERSION,
    ResumePayloadBuilder,
    ResumeProjectChunker,
    load_resume_sample,
)
from document_indexer.resume.llm_extract import LlmStepFailed, ResumeLlmExtractor

FREEFORM = """
Петров Пётр Петрович
Должность: Консультант SAP

Опыт работы

03.2019 – 08.2021  АО «Северсталь»
Функциональный консультант SAP MM
Внедрение SAP S/4HANA в дирекции по снабжению
- настройка процессов закупок
- обучение ключевых пользователей

09.2021 – наст. время  ПАО «Газпром нефть»
Руководитель направления закупок
Тиражирование SAP Ariba на дочерние общества
- управление командой из 6 консультантов
"""

PROFILE_ONLY = """
Сидоров Сидор
Должность: Разработчик 1С

2016 – 2020  ООО «Лента», программист 1С
Доработка 1С:Управление торговлей, обмен с сайтом
Навыки: 1С:Предприятие 8.3, СКД, HTTP-сервисы
"""


class ScriptedChat:
    """Returns canned JSON per step; records what it was asked."""

    def __init__(
        self,
        *,
        projects: dict[str, Any] | None = None,
        refine: dict[str, Any] | None = None,
        experience: dict[str, Any] | None = None,
        fail: set[str] = frozenset(),
    ) -> None:
        self._projects = projects if projects is not None else {"projects": []}
        self._refine = refine if refine is not None else {"projects": []}
        self._experience = experience if experience is not None else {"jobs": [], "profile": {}}
        self._fail = set(fail)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.systems: list[str] = []

    def complete(self, *, messages, format):
        step = _step_of(format)
        user = messages[-1]["content"]
        self.calls.append((step, user, format))
        self.systems.append(messages[0]["content"])
        if step in self._fail:
            raise RuntimeError(f"boom {step}")
        return {"projects": self._projects, "refine": self._refine, "experience": self._experience}[step]

    def steps(self) -> list[str]:
        return [step for step, _, _ in self.calls]


def _step_of(schema: dict[str, Any]) -> str:
    properties = schema["properties"]
    if "jobs" in properties:
        return "experience"
    item_props = properties["projects"]["items"]["properties"]
    return "refine" if "index" in item_props else "projects"


def _doc(text: str) -> SimpleNamespace:
    return SimpleNamespace(export_to_markdown=lambda: text)


def _chunker(chat, **settings) -> ResumeProjectChunker:
    return ResumeProjectChunker(chat=chat, settings=ResumeSettings(**settings))


def test_template_resume_runs_only_refine_and_keeps_parser_values() -> None:
    chat = ScriptedChat(
        refine={
            "projects": [
                {
                    "index": 0,
                    "functional_direction": "Казначейство",
                    "solution_platform": "SAP",
                },
                {
                    "index": 1,
                    "functional_direction": "Производственный учёт",
                    "solution_platform": None,
                },
            ]
        }
    )
    chunks = _chunker(chat).chunk_document(_doc(load_resume_sample()), path_name="cv.md")
    assert chat.steps() == ["refine"]
    assert [chunk.chunk_type for chunk in chunks] == ["project", "project"]
    first, second = chunks
    assert first.extra_fields["extraction_source"] == "parser"
    assert first.extra_fields["project_position"] == "Консультант по направлению Казначейство"
    assert first.extra_fields["functional_direction"] == "Казначейство"
    # both platforms are named in the text, so the LLM answer is used
    assert first.extra_fields["solution_platform"] == "SAP"
    # only 1С is named → regex wins over the (empty) LLM answer
    assert second.extra_fields["solution_platform"] == "1С"
    assert second.extra_fields["functional_direction"] == "Производственный учёт"
    refine_user = chat.calls[0][1]
    assert "[index=0]" in refine_user and "[index=1]" in refine_user
    assert "Пустые поля для заполнения: нет" in refine_user
    # prompt.txt rules travel with the refine system prompt
    assert "solution_platform — на каком ИТ решении выполнен проект" in chat.systems[0]
    assert "Дозаполни ТОЛЬКО те поля" in chat.systems[0]
    assert chat.calls[0][2]["properties"]["projects"]["items"]["required"] == [
        "index",
        "functional_direction",
        "solution_platform",
    ]


def test_refine_fills_only_empty_fields_and_drops_ungrounded() -> None:
    text = """
Иванов Иван
Должность: консультант

Заказчик: Банк
Описание проекта: внедрение 1С:ЗУП
Роль на проекте: консультант

Проект длился 8 месяцев, отрасль — банковский сектор.
"""
    chat = ScriptedChat(
        refine={
            "projects": [
                {
                    "index": 0,
                    "duration": "8 месяцев",
                    "project_industry": "банковский сектор",
                    "work_performed": "миграция на SAP HANA",
                    "customer": "Другой банк",
                    "functional_direction": "Зарплата и кадры",
                    "solution_platform": "SAP",
                }
            ]
        }
    )
    chunker = _chunker(chat, residual_min_chars=10_000)
    chunks = chunker.chunk_document(_doc(text), path_name="cv.md")
    assert chat.steps() == ["refine"]
    (chunk,) = chunks
    extra = chunk.extra_fields
    assert extra["customer"] == "Банк"  # parser value kept, LLM "Другой банк" ignored
    assert extra["duration"] == "8 месяцев"
    assert extra["project_industry"] == "банковский сектор"
    assert extra["work_performed"] is None  # not in the text → dropped
    assert extra["solution_platform"] == "1С"  # explicit in text beats LLM
    assert extra["functional_direction"] == "Зарплата и кадры"
    assert "Продолжительность проекта: 8 месяцев" in chunk.text
    assert chunker.last_stats.ungrounded_dropped == 1
    fields = chat.calls[0][2]["properties"]["projects"]["items"]["properties"]
    assert "customer" not in fields
    assert set(fields) == {
        "index",
        "duration",
        "project_industry",
        "work_performed",
        "functional_direction",
        "solution_platform",
    }


def test_llm_finds_projects_when_parser_has_none() -> None:
    chat = ScriptedChat(
        projects={
            "projects": [
                {
                    "customer": "АО «Северсталь»",
                    "duration": "03.2019 – 08.2021",
                    "project_industry": None,
                    "project_description": "Внедрение SAP S/4HANA в дирекции по снабжению",
                    "project_position": "Функциональный консультант SAP MM",
                    "work_performed": "настройка процессов закупок; обучение ключевых пользователей",
                },
                {
                    "customer": "ПАО «Газпром нефть»",
                    "duration": "09.2021 – наст. время",
                    "project_industry": "Нефтегазовая отрасль",
                    "project_description": "Тиражирование SAP Ariba на дочерние общества",
                    "project_position": "Руководитель направления закупок",
                    "work_performed": "управление командой из 6 консультантов; внедрение SAP Fieldglass",
                },
            ]
        },
        refine={
            "projects": [
                {"index": 0, "project_industry": "Металлургия", "functional_direction": "Закупки", "solution_platform": "SAP"},
                {"index": 1, "project_industry": None, "functional_direction": "Закупки", "solution_platform": "SAP"},
            ]
        },
    )
    chunker = _chunker(chat)
    chunks = chunker.chunk_document(_doc(FREEFORM), path_name="petrov.docx")
    assert chat.steps() == ["projects", "refine"]
    projects_user = chat.calls[0][1]
    assert "Петров Пётр Петрович" not in projects_user  # header removed from residual
    assert "АО «Северсталь»" in projects_user
    assert [chunk.chunk_type for chunk in chunks] == ["project", "project"]
    first, second = chunks
    assert first.extra_fields["extraction_source"] == "llm"
    assert first.extra_fields["customer"] == "АО «Северсталь»"
    assert first.extra_fields["work_performed"] == (
        "настройка процессов закупок; обучение ключевых пользователей"
    )
    assert first.extra_fields["project_industry"] is None  # "Металлургия" is not in the text
    assert first.extra_fields["functional_direction"] == "Закупки"
    assert first.extra_fields["solution_platform"] == "SAP"
    assert second.extra_fields["project_industry"] is None  # "Нефтегазовая отрасль" made up
    assert second.extra_fields["work_performed"] == "управление командой из 6 консультантов"
    assert "Заказчик: ПАО «Газпром нефть»" in second.text
    stats = chunker.last_stats
    assert stats.parser_projects == 0
    assert stats.llm_projects == 2
    assert stats.ungrounded_dropped == 3
    assert stats.needs_review is False


def test_large_residual_triggers_llm_projects_and_merges_with_parser() -> None:
    template_part = load_resume_sample()
    text = template_part + "\n\nПрошлый опыт\n" + FREEFORM.split("Опыт работы", 1)[1] * 3
    chat = ScriptedChat(
        projects={
            "projects": [
                {
                    "customer": "АО «Северсталь»",
                    "duration": "03.2019 – 08.2021",
                    "project_industry": None,
                    "project_description": "Внедрение SAP S/4HANA в дирекции по снабжению",
                    "project_position": "Функциональный консультант SAP MM",
                    "work_performed": None,
                },
                {
                    "customer": "Банк",
                    "duration": "12 месяцев",
                    "project_industry": "банковский сектор",
                    "project_description": "внедрение 1С:ЗУП и интеграция с SAP S/4HANA",
                    "project_position": "Консультант по направлению Казначейство",
                    "work_performed": "настройка казначейства",
                },
            ]
        }
    )
    chunker = _chunker(chat, residual_min_chars=200)
    chunks = chunker.chunk_document(_doc(text), path_name="mixed.docx")
    assert chat.steps() == ["projects", "refine"]
    sources = [chunk.extra_fields["extraction_source"] for chunk in chunks]
    assert sources == ["parser", "parser", "llm"]
    assert chunks[2].extra_fields["customer"] == "АО «Северсталь»"
    assert chunker.last_stats.parser_projects == 2
    assert chunker.last_stats.llm_projects == 1


def test_small_residual_without_hints_skips_llm_projects() -> None:
    text = load_resume_sample() + "\n\nНавыки: Python, SQL\nОбразование: бакалавр\n"
    chat = ScriptedChat()
    chunker = _chunker(chat, residual_min_chars=200)
    chunker.chunk_document(_doc(text), path_name="cv.md")
    assert chat.steps() == ["refine"]
    # below the threshold the leftover ("Проект 1 | Проект 2", skills) never reaches the LLM
    assert "Навыки" not in chat.calls[0][1].split("Проекты:")[0].split("Текст резюме")[0]


def test_experience_and_profile_when_no_projects_anywhere() -> None:
    chat = ScriptedChat(
        projects={"projects": []},
        experience={
            "jobs": [
                {
                    "employer": "ООО «Лента»",
                    "period": "2016 – 2020",
                    "position": "программист 1С",
                    "work_performed": "Доработка 1С:Управление торговлей; обмен с сайтом; внедрение SAP",
                    "systems": "1С:Управление торговлей",
                }
            ],
            "profile": {
                "total_experience": None,
                "skills": "1С:Предприятие 8.3; СКД; HTTP-сервисы; Kotlin",
                "platforms": "1С",
                "directions": "торговля",
            },
        },
    )
    chunker = _chunker(chat)
    chunks = chunker.chunk_document(_doc(PROFILE_ONLY), path_name="sidorov.docx")
    assert chat.steps() == ["projects", "experience"]
    assert [chunk.chunk_type for chunk in chunks] == ["experience", "profile"]
    job, profile = chunks
    assert job.extra_fields["extraction_source"] == "experience"
    assert job.extra_fields["customer"] == "ООО «Лента»"
    assert job.extra_fields["duration"] == "2016 – 2020"
    assert job.extra_fields["project_position"] == "программист 1С"
    assert job.extra_fields["work_performed"] == "Доработка 1С:Управление торговлей; обмен с сайтом"
    assert job.extra_fields["project_description"] == "1С:Управление торговлей"
    assert job.extra_fields["solution_platform"] == "1С"
    assert job.text.startswith("Заказчик: ООО «Лента»")
    assert "Роль на проекте: программист 1С" in job.text
    assert profile.extra_fields["skills"] == "1С:Предприятие 8.3; СКД; HTTP-сервисы"
    assert profile.extra_fields["directions"] is None  # "торговля" alone is not in the text
    assert "ФИО: Сидоров Сидор" in profile.text
    assert "Навыки: 1С:Предприятие 8.3; СКД; HTTP-сервисы" in profile.text
    assert chunker.last_stats.experience_jobs == 1
    assert chunker.last_stats.needs_review is False


def test_llm_failure_falls_back_to_prose_with_needs_review() -> None:
    chat = ScriptedChat(fail={"projects", "experience"})
    chunker = ResumeProjectChunker(chat=chat, window_chars=200, window_overlap=20)
    chunks = chunker.chunk_document(_doc(FREEFORM), path_name="petrov.docx")
    assert chat.steps() == ["projects", "experience"]
    assert chunks and all(chunk.chunk_type == "prose" for chunk in chunks)
    assert all(chunk.extra_fields["needs_review"] is True for chunk in chunks)
    assert chunks[0].extra_fields["review_reason"].startswith("llm_experience_failed")
    assert chunks[0].extra_fields["candidate_name"] == "Петров Пётр Петрович"
    assert chunker.last_stats.needs_review is True


def test_refine_failure_keeps_parser_projects() -> None:
    chat = ScriptedChat(fail={"refine"})
    chunker = _chunker(chat)
    chunks = chunker.chunk_document(_doc(load_resume_sample()), path_name="cv.md")
    assert [chunk.chunk_type for chunk in chunks] == ["project", "project"]
    assert "functional_direction" not in chunks[0].extra_fields
    assert chunks[1].extra_fields["solution_platform"] == "1С"
    assert chunker.last_stats.needs_review is True
    assert chunker.last_stats.review_reason.startswith("llm_refine_failed")


def test_no_chat_means_parser_only_behaviour() -> None:
    chunker = ResumeProjectChunker(window_chars=300, window_overlap=30)
    chunks = chunker.chunk_document(_doc(FREEFORM), path_name="petrov.docx")
    assert all(chunk.chunk_type == "prose" for chunk in chunks)
    assert chunks[0].extra_fields["review_reason"] == "no_llm"
    assert chunker.last_stats.llm_calls == []


def test_long_text_is_sectioned_and_results_merged() -> None:
    chat = ScriptedChat(
        projects={
            "projects": [
                {
                    "customer": "АО «Северсталь»",
                    "duration": None,
                    "project_industry": None,
                    "project_description": "Внедрение SAP S/4HANA в дирекции по снабжению",
                    "project_position": None,
                    "work_performed": None,
                }
            ]
        }
    )
    extractor = ResumeLlmExtractor(
        chat,
        settings=ResumeSettings(section_max_chars=600, section_overlap_chars=50),
        num_ctx=65_536,
    )
    assert extractor.max_chars == 600
    long_text = "\n\n".join([FREEFORM.strip()] * 4)
    from document_indexer.resume.grounding import Grounder

    found = extractor.extract_projects(long_text, Grounder(long_text), path_name="long.docx")
    assert len(chat.calls) > 1
    assert all("часть" in user for _, user, _ in chat.calls)
    assert len(found) == 1  # identical answers from every section collapse


def test_llm_step_failed_wraps_chat_errors() -> None:
    chat = ScriptedChat(fail={"projects"})
    extractor = ResumeLlmExtractor(chat)
    from document_indexer.resume.grounding import Grounder

    with pytest.raises(LlmStepFailed):
        extractor.extract_projects("текст", Grounder("текст"), path_name="x.md")


class FakeEmbedder:
    def embed(self, text):
        if isinstance(text, str):
            return [float(len(text)), 1.0]
        return [[float(len(item)), 1.0] for item in text]


class ChunkReader:
    def __init__(self, chunker: ResumeProjectChunker, text: str) -> None:
        self._chunker = chunker
        self._text = text

    def read(self, path: Path):
        return self._chunker.chunk_document(_doc(self._text), path_name=path.name)


def test_qdrant_payload_for_llm_projects_and_experience(tmp_path: Path) -> None:
    docs = tmp_path / "cv"
    docs.mkdir()
    (docs / "petrov.md").write_text("x", encoding="utf-8")
    chat = ScriptedChat(
        projects={"projects": []},
        experience={
            "jobs": [
                {
                    "employer": "АО «Северсталь»",
                    "period": "03.2019 – 08.2021",
                    "position": "Функциональный консультант SAP MM",
                    "work_performed": "настройка процессов закупок",
                    "systems": "SAP S/4HANA",
                }
            ],
            "profile": {"total_experience": None, "skills": None, "platforms": "SAP", "directions": None},
        },
    )
    chunker = _chunker(chat)
    client = MagicMock()
    client.collection_exists.return_value = False
    client.scroll.return_value = ([], None)
    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs-cv",
        embedder=FakeEmbedder(),
        document_reader=ChunkReader(chunker, FREEFORM),
        payload_builder=ResumePayloadBuilder(),
        index_version=INDEX_VERSION,
    )
    indexer._client = client
    indexer.index(str(docs))

    points = client.upsert.call_args.kwargs["points"]
    assert [point.payload["chunk_type"] for point in points] == ["experience", "profile"]
    job, profile = (point.payload for point in points)
    assert job["candidate_name"] == "Петров Пётр Петрович"
    assert job["candidate_position"] == "Консультант SAP"
    assert job["customer"] == "АО «Северсталь»"
    assert job["duration"] == "03.2019 – 08.2021"
    assert job["project_position"] == "Функциональный консультант SAP MM"
    assert job["project_description"] == "SAP S/4HANA"
    assert job["solution_platform"] == "SAP"
    assert job["extraction_source"] == "experience"
    assert "needs_review" not in job
    assert job["index_version"] == INDEX_VERSION == "resume-v20"
    assert profile["platforms"] == "SAP"
    assert "customer" not in profile
    index_names = {call.kwargs["field_name"] for call in client.create_payload_index.call_args_list}
    assert {"customer", "extraction_source", "solution_platform", "functional_direction"} <= index_names
