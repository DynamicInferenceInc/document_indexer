"""End-to-end resume schema → FakeChat → Qdrant payload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher
from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.domain.models import DocumentChunk
from document_indexer.examples.resume import (
    INDEX_VERSION,
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
        assert "Описание проекта" in content
        assert "ведущий консультант" in content
        project_schema = format["properties"]["project_experiences"]["items"]
        props = project_schema["properties"]
        assert project_schema["additionalProperties"] is False
        assert set(props) == set(_PROJECT_KEYS)
        for key in _PROJECT_KEYS:
            assert "enum" not in props[key]
        return {
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
    assert tuple(props) == ("project_experiences",)
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
            assert "Желаемая должность: ведущий консультант" in messages[-1]["content"]
            return {
                "project_experiences": [
                    {
                        "project_description": None,
                        "project_position": "ведущий консультант",
                        "project_industry": None,
                    }
                ]
            }

    prompt = load_resume_prompt()
    assert "Если проекты в резюме не выделены" in prompt
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
                    "Желаемая должность: ведущий консультант. "
                    "Навыки: 1С, SQL. Отдельных проектов нет."
                )
            )
        ],
    )
    assert fields == {
        "project_experiences": [
            {
                "project_description": None,
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
        assert point.payload["index_version"] == "resume-v6"
        assert len(point.payload["file_hash"]) == 64
    assert points[0].payload["chunk_index"] == 0
    assert points[1].payload["chunk_index"] == 1
    assert points[2].payload["chunk_index"] == 2
    assert points[0].payload["text"] != points[1].payload["text"]

    index_names = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert index_names >= {
        "project_experiences[].project_description",
        "project_experiences[].project_position",
        "project_experiences[].project_industry",
    }
