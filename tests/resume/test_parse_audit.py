"""Parse-only resume audit (no LLM)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from document_indexer.config import LocalSourceSettings, ProfileLocal
from document_indexer.domain.models import DocumentChunk
from document_indexer.resume.audit import (
    format_audit_report,
    parse_only_enabled,
    row_from_chunks,
    run_resume_parse_audit,
)
from document_indexer.resume.chunker import ResumeProjectChunker


def test_parse_only_enabled_reads_truthy_env() -> None:
    assert parse_only_enabled({"RESUME_PARSE_ONLY": "1"}) is True
    assert parse_only_enabled({"RESUME_PARSE_ONLY": "true"}) is True
    assert parse_only_enabled({"RESUME_PARSE_ONLY": "0"}) is False
    assert parse_only_enabled({}) is False


def test_resume_parse_only_empty_env_is_false(monkeypatch) -> None:
    from document_indexer.config import IndexerSettings

    monkeypatch.setenv("RESUME_PARSE_ONLY", "")
    settings = IndexerSettings(_env_file=None)
    assert settings.resume_parse_only is False


def test_row_from_chunks_counts_projects() -> None:
    chunks = [
        DocumentChunk(
            text="p1",
            chunk_type="project",
            extra_fields={
                "candidate_name": "Елена Власова",
                "candidate_position": "Руководитель проектов",
            },
        ),
        DocumentChunk(
            text="p2",
            chunk_type="project",
            extra_fields={
                "candidate_name": "Елена Власова",
                "candidate_position": "Руководитель проектов",
            },
        ),
    ]
    row = row_from_chunks("Власова/CV.docx", chunks)
    assert row["project_count"] == 2
    assert row["prose_count"] == 0
    assert row["candidate_name"] == "Елена Власова"
    assert row["candidate_position"] == "Руководитель проектов"
    assert row["error"] is None


def test_format_audit_report_lists_people_without_projects() -> None:
    text = format_audit_report(
        [
            {
                "source_path": "a.docx",
                "candidate_name": "А",
                "candidate_position": "архитектор",
                "project_count": 3,
                "prose_count": 0,
                "error": None,
            },
            {
                "source_path": "b.docx",
                "candidate_name": "Б",
                "candidate_position": "стажер",
                "project_count": 0,
                "prose_count": 12,
                "error": None,
            },
        ]
    )
    assert "с проектами=1 без проектов=1" in text
    assert "Б  стажер  b.docx" in text
    assert "   3  А  архитектор  a.docx" in text


def test_run_resume_parse_audit_skips_llm_and_writes_csv(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SOURCE__WATCH_PATH", raising=False)
    watch = tmp_path / "cv"
    watch.mkdir()
    (watch / "good.md").write_text(
        "Иванов Иван\nДолжность: консультант\n\n"
        "Заказчик: Банк\nОтрасль проекта: банки\n"
        "Описание проекта: внедрение ERP\nРоль на проекте: консультант\n"
        "Выполненные работы: настройка\n",
        encoding="utf-8",
    )
    (watch / "empty.md").write_text(
        "Сидоров Сидор\nДолжность: стажер\n\nНавыки: Python\n",
        encoding="utf-8",
    )

    def convert(path: Path) -> SimpleNamespace:
        text = path.read_text(encoding="utf-8")
        return SimpleNamespace(export_to_markdown=lambda: text)

    rows = run_resume_parse_audit(
        ProfileLocal(
            _env_file=None,
            source=LocalSourceSettings(watch_path=str(watch)),
        ),
        convert=convert,
        chunker=ResumeProjectChunker(),
    )
    by_path = {row["source_path"]: row for row in rows}
    assert by_path["good.md"]["project_count"] == 1
    assert by_path["empty.md"]["project_count"] == 0
    assert by_path["empty.md"]["candidate_position"] == "стажер"
    assert (watch / ".resume_project_stats.csv").is_file()
    csv_text = (watch / ".resume_project_stats.csv").read_text(encoding="utf-8")
    assert "good.md" in csv_text
    assert "empty.md" in csv_text


def test_document_indexer_parse_only_skips_build_indexer(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from document_indexer import DocumentIndexer, IndexerSettings, LocalSourceSettings

    monkeypatch.delenv("RESUME_PARSE_ONLY", raising=False)
    watch = tmp_path / "cv"
    watch.mkdir()
    built: list[str] = []
    monkeypatch.setattr(
        "document_indexer.indexer.build_indexer",
        lambda *args, **kwargs: built.append("built") or MagicMock(),
    )
    monkeypatch.setattr(
        "document_indexer.resume.audit.run_resume_parse_audit",
        lambda settings: built.append("audit") or [],
    )
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path=str(watch)),
        chunking={"strategy": "resume_project"},
        resume_parse_only=True,
    )
    DocumentIndexer(settings, configure_logs=False).run()
    assert built == ["audit"]
