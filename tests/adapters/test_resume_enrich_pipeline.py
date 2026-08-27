"""End-to-end resume schema → FakeChat → Qdrant payload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher
from document_indexer.adapters.qdrant.payload import IndexRecord
from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.domain.models import DocumentChunk
from document_indexer.examples.resume import (
    INDEX_VERSION,
    NO_PROJECTS_LABEL,
    ResumePayloadBuilder,
    load_resume_prompt,
    load_resume_sample,
    load_resume_schema,
)

_PROJECT_KEYS = (
    "project_description",
    "project_position",
    "project_industry",
)


class FakeChat:
    def complete(self, *, messages, format):
        content = messages[-1]["content"]
        assert "Имя файла:" in content
        assert "Описание проекта" in content
        assert "ведущий консультант" in content
        assert "candidate_name" in format["properties"]
        assert "candidate_position" in format["properties"]
        project_schema = format["properties"]["project_experiences"]["items"]
        props = project_schema["properties"]
        assert project_schema["additionalProperties"] is False
        assert set(props) == set(_PROJECT_KEYS)
        for key in _PROJECT_KEYS:
            assert "enum" not in props[key]
        return {
            "candidate_name": "Иванов Иван",
            "candidate_position": "ведущий консультант",
            "project_experiences": [
                {
                    "project_description": (
                        "внедрение 1С:ЗУП и интеграция с SAP S/4HANA"
                    ),
                    "project_position": "ведущий консультант",
                    "project_industry": "банковский сектор",
                },
                {
                    "project_description": (
                        "модернизация производственного учёта на 1С:ERP"
                    ),
                    "project_position": "архитектор 1С",
                    "project_industry": "нефтегаз",
                },
                {
                    "project_description": (
                        "разработка внутреннего портала для обработки заявок"
                    ),
                    "project_position": "backend-разработчик",
                    "project_industry": None,
                },
            ]
        }


class TwoChunkReader:
    def read(self, path: Path) -> list[DocumentChunk]:
        del path
        return [
            DocumentChunk(
                text=(
                    "ФИО: Иванов Иван. "
                    "Описание проекта: внедрение 1С:ЗУП и интеграция с "
                    "SAP S/4HANA. Роль на проекте: ведущий консультант. "
                    "Отрасль проекта: банковский сектор."
                )
            ),
            DocumentChunk(
                text=(
                    "Описание проекта: модернизация производственного учёта "
                    "на 1С:ERP. Должность на проекте: архитектор 1С. "
                    "Отрасль проекта: нефтегаз."
                )
            ),
            DocumentChunk(
                text=(
                    "Описание проекта: разработка внутреннего портала. "
                    "Роль на проекте: backend-разработчик."
                )
            ),
        ]


class FakeEmbedder:
    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return [float(len(text)), 1.0]
        return [[float(len(item)), 1.0] for item in text]


def test_resume_schema_defines_filter_fields() -> None:
    schema = load_resume_schema()
    props = schema["properties"]
    assert tuple(props) == ("candidate_name", "candidate_position", "project_experiences")
    project_props = props["project_experiences"]["items"]["properties"]
    assert set(project_props) == set(_PROJECT_KEYS)
    assert all("enum" not in value for value in project_props.values())
    sample = load_resume_sample()
    assert "1С:ERP" in sample
    assert "ведущий консультант" in sample
    assert "Отрасль проекта в резюме не указана" in sample


def test_resume_without_projects_uses_explicit_candidate_position(
    tmp_path: Path,
) -> None:
    class CandidateOnlyChat:
        def complete(self, *, messages, format):
            del format
            text = messages[-1]["content"]
            assert "Имя файла:" in text
            assert "Желаемая должность: ведущий консультант" in text
            return {
                "candidate_name": "Петров Пётр",
                "candidate_position": "ведущий консультант",
                "project_experiences": [
                    {
                        "project_description": NO_PROJECTS_LABEL,
                        "project_position": "ведущий консультант",
                        "project_industry": None,
                    }
                ]
            }

    prompt = load_resume_prompt()
    assert "Проекты не указаны" in prompt
    assert "candidate_name" in prompt or "ФИО" in prompt
    enricher = JsonSchemaEnricher(
        load_resume_schema(),
        prompt,
        chat=CandidateOnlyChat(),
    )
    fields = enricher.enrich(
        tmp_path / "candidate.md",
        [
            DocumentChunk(
                text=(
                    "ФИО: Петров Пётр. "
                    "Желаемая должность: ведущий консультант. "
                    "Навыки: 1С, SQL. Отдельных проектов нет."
                )
            )
        ],
    )
    assert fields == {
        "candidate_name": "Петров Пётр",
        "candidate_position": "ведущий консультант",
        "project_experiences": [
            {
                "project_description": NO_PROJECTS_LABEL,
                "project_position": "ведущий консультант",
                "project_industry": None,
            }
        ]
    }


def test_resume_schema_file_fake_chat_points(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cv" / "ivanov.md").parent.mkdir()
    (docs / "cv" / "ivanov.md").write_text("source", encoding="utf-8")

    client = MagicMock()
    client.collection_exists.return_value = False
    client.scroll.return_value = ([], None)

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs-cv",
        embedder=FakeEmbedder(),
        document_reader=TwoChunkReader(),
        payload_builder=ResumePayloadBuilder(),
        enricher=JsonSchemaEnricher(
            load_resume_schema(),
            load_resume_prompt(),
            chat=FakeChat(),
        ),
        index_version=INDEX_VERSION,
    )
    indexer._client = client
    indexer.index(str(docs))

    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 3
    assert "1С:ЗУП" in points[0].payload["text"]
    assert "1С:ERP" in points[1].payload["text"]
    for point in points:
        experiences = point.payload["project_experiences"]
        assert len(experiences) == 3
        first, second, third = experiences
        assert set(first) == set(_PROJECT_KEYS)
        assert first["project_position"] == "ведущий консультант"
        assert first["project_industry"] == "банковский сектор"
        assert second["project_position"] == "архитектор 1С"
        assert second["project_industry"] == "нефтегаз"
        assert third["project_position"] == "backend-разработчик"
        assert third["project_industry"] is None
        assert point.payload["source_path"] == "cv/ivanov.md"
        assert point.payload["candidate_name"] == "Иванов Иван"
        assert point.payload["candidate_position"] == "ведущий консультант"
        assert point.payload["index_version"] == "resume-v12"
        assert len(point.payload["file_hash"]) == 64
    assert points[0].payload["chunk_index"] == 0
    assert points[1].payload["chunk_index"] == 1
    assert points[2].payload["chunk_index"] == 2
    assert points[0].payload["text"] != points[1].payload["text"]

    index_names = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert index_names >= {
        "candidate_name",
        "candidate_position",
        "project_experiences[].project_description",
        "project_experiences[].project_position",
        "project_experiences[].project_industry",
    }


def test_resume_payload_writes_no_projects_label() -> None:
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="навыки"),
        file_path=Path("cv.md"),
        document_fields={
            "candidate_name": "Сидоров",
            "candidate_position": "Разработчик-стажер",
            "project_experiences": [],
        },
        index_version=INDEX_VERSION,
    )
    payload = ResumePayloadBuilder().build(record)
    assert payload["candidate_name"] == "Сидоров"
    assert payload["candidate_position"] == "Разработчик-стажер"
    assert payload["project_experiences"] == [
        {
            "project_description": NO_PROJECTS_LABEL,
            "project_position": "Разработчик-стажер",
            "project_industry": None,
        }
    ]


def test_resume_payload_drops_mashed_duplicate_keeps_labeled_description() -> None:
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="table"),
        file_path=Path("cv.md"),
        document_fields={
            "candidate_name": "Шевчик",
            "candidate_position": None,
            "project_experiences": [
                {
                    "project_description": (
                        "Джи-Ти-Ай Россия. Консультант по миграции данных "
                        "Оуществлял перенос данных SAP и 1С:УПП на 1C:ERP УХ"
                    ),
                    "project_position": "Консультант по миграции данных",
                    "project_industry": None,
                },
                {
                    "project_description": (
                        "Перехода автоматизированной информационной системы "
                        "управления компанией с SAP ERP и 1С:УПП на 1С:ERP УХ"
                    ),
                    "project_position": "Консультант по миграции данных",
                    "project_industry": None,
                },
            ],
        },
        index_version=INDEX_VERSION,
    )
    payload = ResumePayloadBuilder().build(record)
    assert payload["project_experiences"] == [
        {
            "project_description": (
                "Перехода автоматизированной информационной системы "
                "управления компанией с SAP ERP и 1С:УПП на 1С:ERP УХ"
            ),
            "project_position": "Консультант по миграции данных",
            "project_industry": None,
        }
    ]


def test_resume_payload_drops_date_and_company_header_keeps_real_projects() -> None:
    sap_mm = "Функциональный консультант SAP MM"
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="cv"),
        file_path=Path("cv.md"),
        document_fields={
            "project_experiences": [
                {
                    "project_description": "JTI, г. Санкт – Петербург, Россия",
                    "project_position": "2025 – текущее время",
                    "project_industry": None,
                },
                {
                    "project_description": "Проект перехода на SAP S4/HANA",
                    "project_position": sap_mm,
                    "project_industry": "Металлургия",
                },
                {
                    "project_description": "Внедрение системы SAP R/3",
                    "project_position": sap_mm,
                    "project_industry": "Горнодобывающий комбинат",
                },
            ]
        },
        index_version=INDEX_VERSION,
    )
    experiences = ResumePayloadBuilder().build(record)["project_experiences"]
    assert [item["project_description"] for item in experiences] == [
        "Проект перехода на SAP S4/HANA",
        "Внедрение системы SAP R/3",
    ]
    assert all(item["project_position"] == sap_mm for item in experiences)


def test_resume_payload_merges_typo_copies_and_drops_date_description() -> None:
    sap_mm = "Функциональный консультант SAP MM"
    wms = "Консультант по складской логистике (WMS), консультант по направлению запасы"
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="cv"),
        file_path=Path("cv.md"),
        document_fields={
            "project_experiences": [
                {
                    "project_description": "Проект перехода с SAP R\\3 на 1C ERP",
                    "project_position": wms,
                    "project_industry": "шины",
                },
                {
                    "project_description": "Проект цифровой трансформации",
                    "project_position": sap_mm,
                    "project_industry": "Машиностроение",
                },
                {
                    "project_description": "Внедрение системы SAP R/3",
                    "project_position": sap_mm,
                    "project_industry": "Горнодобывающий комбинат",
                },
                {
                    "project_description": "22.06.2016-11.06.2017",
                    "project_position": sap_mm,
                    "project_industry": "Металлургия",
                },
                {
                    "project_description": "Проект перехода с SAP R\\3 на 1C ERP",
                    "project_position": (
                        "Консультант по скласдкой логистике (WMS), "
                        "консультант по направлению запасы"
                    ),
                    "project_industry": "шины.",
                },
                {
                    "project_description": "Проект цифровой трансформации",
                    "project_position": "Функциона:льный консультант SAP MM",
                    "project_industry": "Машиностроение",
                },
            ]
        },
        index_version=INDEX_VERSION,
    )
    experiences = ResumePayloadBuilder().build(record)["project_experiences"]
    assert [item["project_description"] for item in experiences] == [
        "Проект перехода с SAP R\\3 на 1C ERP",
        "Проект цифровой трансформации",
        "Внедрение системы SAP R/3",
    ]
    assert experiences[0]["project_position"] == wms
    assert experiences[1]["project_position"] == sap_mm
