"""Resume schema → FakeChat → Qdrant payload."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from document_indexer.adapters.qdrant.payload import IndexRecord
from document_indexer.adapters.qdrant_indexer import QdrantIndexer
from document_indexer.domain.models import DocumentChunk
from document_indexer.examples.resume import (
    INDEX_VERSION,
    FunctionalDirectionEnricher,
    ResumePayloadBuilder,
    load_resume_prompt,
    load_resume_sample,
    load_resume_schema,
)


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def complete(self, *, messages, format):
        content = messages[-1]["content"]
        self.calls.append(content)
        assert "Имя файла:" in content
        assert "candidate_name" not in format["properties"]
        assert "functional_direction" in format["properties"]
        assert "solution_platform" in format["properties"]
        platform = None
        if "SAP" in content:
            platform = "SAP"
        elif "1С" in content or "1C" in content:
            platform = "1С"
        if "Казначейство" in content:
            return {"functional_direction": "Казначейство", "solution_platform": platform}
        if "архитектор" in content:
            return {"functional_direction": "архитектура 1С", "solution_platform": platform}
        return {"functional_direction": None, "solution_platform": platform}


class ProjectReader:
    def read(self, path: Path) -> list[DocumentChunk]:
        del path
        return [
            DocumentChunk(
                text="Заказчик: Банк\nРоль на проекте: Консультант по направлению Казначейство",
                chunk_type="project",
                extra_fields={
                    "candidate_name": "Иванов Иван",
                    "candidate_position": "ведущий консультант",
                    "project_industry": "банковский сектор",
                    "project_description": "внедрение 1С:ЗУП",
                    "project_position": "Консультант по направлению Казначейство",
                    "work_performed": "настройка казначейства",
                },
            ),
            DocumentChunk(
                text="Описание проекта: модернизация учёта на 1С:ERP",
                chunk_type="project",
                extra_fields={
                    "candidate_name": "Иванов Иван",
                    "candidate_position": "ведущий консультант",
                    "project_industry": "нефтегаз",
                    "project_description": "модернизация производственного учёта на 1С:ERP",
                    "project_position": "архитектор 1С",
                    "work_performed": "проектирование",
                },
            ),
        ]


class FakeEmbedder:
    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return [float(len(text)), 1.0]
        return [[float(len(item)), 1.0] for item in text]


def test_resume_schema_has_direction_and_platform() -> None:
    schema = load_resume_schema()
    assert tuple(schema["properties"]) == ("functional_direction", "solution_platform")
    sample = load_resume_sample()
    assert "1С:ERP" in sample
    assert "ведущий консультант" in sample
    assert "Иванов Иван Иванович" in sample


def test_resume_payload_skips_llm_for_window_chunks() -> None:
    chat = FakeChat()
    enricher = FunctionalDirectionEnricher(
        load_resume_schema(),
        load_resume_prompt(),
        chat=chat,
    )
    fields = enricher.enrich(
        Path("cv.md"),
        [
            DocumentChunk(
                text="навыки python",
                extra_fields={
                    "candidate_name": "Петров Пётр",
                    "candidate_position": "разработчик",
                },
            )
        ],
    )
    assert fields == {"functional_directions": [None], "solution_platforms": [None]}
    assert chat.calls == []


def test_resume_enricher_always_calls_llm_for_project_role() -> None:
    chat = FakeChat()
    enricher = FunctionalDirectionEnricher(
        load_resume_schema(),
        load_resume_prompt(),
        chat=chat,
    )
    fields = enricher.enrich(
        Path("cv.md"),
        [
            DocumentChunk(
                text="роль",
                chunk_type="project",
                extra_fields={
                    "candidate_position": "Ведущий консультант",
                    "project_position": "Консультант по направлению Казначейство",
                    "work_performed": "тестирование",
                },
            )
        ],
    )
    assert fields == {"functional_directions": ["Казначейство"], "solution_platforms": [None]}
    assert len(chat.calls) == 1
    assert "Роль на проекте: Консультант по направлению Казначейство" in chat.calls[0]
    assert "Должность из шапки: Ведущий консультант" in chat.calls[0]


def test_resume_schema_file_fake_chat_points(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "cv" / "ivanov.md").parent.mkdir()
    (docs / "cv" / "ivanov.md").write_text("source", encoding="utf-8")

    client = MagicMock()
    client.collection_exists.return_value = False
    client.scroll.return_value = ([], None)
    chat = FakeChat()

    indexer = QdrantIndexer(
        qdrant_url="http://127.0.0.1:6333",
        collection="docs-cv",
        embedder=FakeEmbedder(),
        document_reader=ProjectReader(),
        payload_builder=ResumePayloadBuilder(),
        enricher=FunctionalDirectionEnricher(
            load_resume_schema(),
            load_resume_prompt(),
            chat=chat,
        ),
        index_version=INDEX_VERSION,
    )
    indexer._client = client
    indexer.index(str(docs))

    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    assert len(chat.calls) == 2
    first, second = points
    assert first.payload["chunk_type"] == "project"
    assert first.payload["candidate_name"] == "Иванов Иван"
    assert first.payload["candidate_position"] == "ведущий консультант"
    assert first.payload["project_position"] == "Консультант по направлению Казначейство"
    assert first.payload["functional_direction"] == "Казначейство"
    assert first.payload["solution_platform"] == "1С"
    assert first.payload["project_industry"] == "банковский сектор"
    assert "project_experiences" not in first.payload
    assert second.payload["functional_direction"] == "архитектура 1С"
    assert second.payload["solution_platform"] == "1С"
    assert second.payload["project_description"] != first.payload["project_description"]
    assert first.payload["index_version"] == INDEX_VERSION

    index_names = {
        call.kwargs["field_name"] for call in client.create_payload_index.call_args_list
    }
    assert index_names >= {
        "candidate_name",
        "candidate_position",
        "project_description",
        "project_position",
        "project_industry",
        "functional_direction",
        "solution_platform",
    }


def test_resume_payload_window_has_name_and_position_without_project_fields() -> None:
    record = IndexRecord(
        source_path="cv.md",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(
            text="навыки python",
            extra_fields={
                "candidate_name": "Сидоров",
                "candidate_position": "Разработчик-стажер",
            },
        ),
        file_path=Path("cv.md"),
        document_fields={"functional_directions": [None]},
        index_version=INDEX_VERSION,
    )
    payload = ResumePayloadBuilder().build(record)
    assert payload["candidate_name"] == "Сидоров"
    assert payload["candidate_position"] == "Разработчик-стажер"
    assert payload["chunk_type"] == "prose"
    assert "project_experiences" not in payload
    assert "project_industry" not in payload
    assert payload["functional_direction"] is None
    assert "solution_platform" not in payload


def test_bind_replaces_json_schema_enricher_for_resume_strategy() -> None:
    from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher
    from document_indexer.config import ChunkingSettings, IndexerSettings, ModelSettings
    from document_indexer.examples.resume.enricher import bind_resume_enricher

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="resume_project"),
        models=ModelSettings(extraction_model="qwen3:4b"),
    )
    incoming = JsonSchemaEnricher(load_resume_schema(), load_resume_prompt(), chat=FakeChat())
    bound = bind_resume_enricher(settings, incoming)
    assert isinstance(bound, FunctionalDirectionEnricher)


def test_bind_keeps_json_schema_enricher_for_table_aware() -> None:
    from document_indexer.adapters.enrichment.json_schema import JsonSchemaEnricher
    from document_indexer.config import ChunkingSettings, IndexerSettings, ModelSettings
    from document_indexer.examples.resume.enricher import bind_resume_enricher

    settings = IndexerSettings(
        _env_file=None,
        chunking=ChunkingSettings(strategy="table_aware"),
        models=ModelSettings(extraction_model="qwen3:4b"),
    )
    incoming = JsonSchemaEnricher(load_resume_schema(), load_resume_prompt(), chat=FakeChat())
    assert bind_resume_enricher(settings, incoming) is incoming


def test_infer_solution_platform_explicit_and_transition() -> None:
    from document_indexer.examples.resume.enricher import infer_solution_platform

    assert infer_solution_platform("Функциональный консультант SAP MM") == "SAP"
    assert infer_solution_platform("Внедрение 1С:ERP") == "1С"
    assert infer_solution_platform("архитектор 1С") == "1С"
    assert infer_solution_platform(
        "Проект перехода с SAP R/3 на 1C ERP"
    ) == "1С"
    assert infer_solution_platform("внедрение 1С:ЗУП и интеграция с SAP S/4HANA") is None
    assert infer_solution_platform("настройка казначейства") is None


def test_explicit_sap_still_calls_llm_for_direction() -> None:
    chat = FakeChat()
    enricher = FunctionalDirectionEnricher(
        load_resume_schema(),
        load_resume_prompt(),
        chat=chat,
    )
    fields = enricher.enrich(
        Path("cv.md"),
        [
            DocumentChunk(
                text="роль",
                chunk_type="project",
                extra_fields={
                    "project_position": "Функциональный консультант SAP MM",
                    "project_description": "Внедрение системы SAP S4/HANA",
                    "work_performed": "настройка MM",
                },
            )
        ],
    )
    assert fields["solution_platforms"] == ["SAP"]
    assert len(chat.calls) == 1
    assert "SAP MM" in chat.calls[0]
