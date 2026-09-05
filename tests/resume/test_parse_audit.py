"""Resume audits: parser-only and parser + LLM, no embed / Qdrant."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from document_indexer.config import LocalSourceSettings, ProfileLocal
from document_indexer.domain.models import DocumentChunk
from document_indexer.resume.audit import (
    format_audit_report,
    llm_audit_enabled,
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
    assert llm_audit_enabled({"RESUME_LLM_AUDIT": "yes"}) is True
    assert llm_audit_enabled({}) is False


def test_resume_audit_flags_empty_env_is_false(monkeypatch) -> None:
    from document_indexer.config import IndexerSettings

    monkeypatch.setenv("RESUME_PARSE_ONLY", "")
    monkeypatch.setenv("RESUME_LLM_AUDIT", "")
    settings = IndexerSettings(_env_file=None)
    assert settings.resume_parse_only is False
    assert settings.resume_llm_audit is False


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
                "needs_review": False,
                "error": None,
            },
            {
                "source_path": "b.docx",
                "candidate_name": "Б",
                "candidate_position": "стажер",
                "project_count": 0,
                "prose_count": 12,
                "needs_review": True,
                "error": None,
            },
        ]
    )
    assert "резюме=2 с проектами=1" in text
    assert "требуют проверки=1" in text
    borisov = next(line for line in text.splitlines() if "b.docx" in line and "стажер" in line)
    assert "да" in borisov
    antonov = next(line for line in text.splitlines() if "a.docx" in line and "архитектор" in line)
    assert "  3  " in antonov


def _convert(path: Path) -> SimpleNamespace:
    text = path.read_text(encoding="utf-8")
    return SimpleNamespace(export_to_markdown=lambda: text)


def _write_resumes(watch: Path) -> None:
    (watch / "good.md").write_text(
        "Иванов Иван\nДолжность: консультант\n\n"
        "Заказчик: Банк\nОтрасль проекта: банки\n"
        "Описание проекта: внедрение ERP\nРоль на проекте: консультант\n"
        "Выполненные работы: настройка\n",
        encoding="utf-8",
    )
    (watch / "empty.md").write_text(
        "Сидоров Сидор\nДолжность: стажер\n\nНавыки: Python\n"
        "2020 – 2021  ООО «Ромашка», стажер\n",
        encoding="utf-8",
    )


def test_run_resume_parse_audit_skips_llm_and_writes_report(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("SOURCE__WATCH_PATH", raising=False)
    watch = tmp_path / "cv"
    watch.mkdir()
    _write_resumes(watch)

    rows = run_resume_parse_audit(
        ProfileLocal(
            _env_file=None,
            source=LocalSourceSettings(watch_path=str(watch)),
        ),
        convert=_convert,
        chunker=ResumeProjectChunker(),
    )
    by_path = {row["source_path"]: row for row in rows}
    assert by_path["good.md"]["project_count"] == 1
    assert by_path["empty.md"]["project_count"] == 0
    assert by_path["empty.md"]["needs_review"] is True
    assert by_path["empty.md"]["candidate_position"] == "стажер"
    assert (watch / ".resume_report.csv").is_file()
    assert (watch / ".resume_report.txt").is_file()
    csv_text = (watch / ".resume_report.csv").read_text(encoding="utf-8")
    assert "good.md" in csv_text and "empty.md" in csv_text
    assert "Иванов Иван" in (watch / ".resume_report.txt").read_text(encoding="utf-8")
    assert not (watch / ".resume_chunks.jsonl").exists()


def test_run_resume_llm_audit_writes_chunks_jsonl(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("SOURCE__WATCH_PATH", raising=False)
    watch = tmp_path / "cv"
    watch.mkdir()
    _write_resumes(watch)

    class Chat:
        def complete(self, *, messages, format):
            props = format["properties"]
            if "jobs" in props:
                return {
                    "jobs": [
                        {
                            "employer": "ООО «Ромашка»",
                            "period": "2020 – 2021",
                            "position": "стажер",
                            "work_performed": None,
                            "systems": "Python",
                        }
                    ],
                    "profile": {"total_experience": None, "skills": "Python", "platforms": None, "directions": None},
                }
            if "index" in props["projects"]["items"]["properties"]:
                return {"projects": [{"index": 0, "functional_direction": "ERP", "solution_platform": None}]}
            return {"projects": []}

    rows = run_resume_parse_audit(
        ProfileLocal(
            _env_file=None,
            source=LocalSourceSettings(watch_path=str(watch)),
        ),
        convert=_convert,
        chunker=ResumeProjectChunker(chat=Chat()),
        with_llm=True,
    )
    by_path = {row["source_path"]: row for row in rows}
    assert by_path["good.md"]["project_count"] == 1
    assert by_path["good.md"]["llm_calls"] == "refine"
    assert by_path["empty.md"]["experience_count"] == 1
    assert by_path["empty.md"]["profile_count"] == 1
    assert by_path["empty.md"]["needs_review"] is False
    dump = watch / ".resume_chunks.jsonl"
    assert dump.is_file()
    records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
    assert {record["source_path"] for record in records} == {"good.md", "empty.md"}
    good = [record for record in records if record["source_path"] == "good.md"][0]
    assert good["chunk_type"] == "project"
    assert good["fields"]["functional_direction"] == "ERP"
    assert good["text"].startswith("Заказчик: Банк")


def test_document_indexer_parse_only_skips_build_indexer(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock

    from document_indexer import DocumentIndexer, IndexerSettings, LocalSourceSettings

    monkeypatch.delenv("RESUME_PARSE_ONLY", raising=False)
    monkeypatch.delenv("RESUME_LLM_AUDIT", raising=False)
    watch = tmp_path / "cv"
    watch.mkdir()
    built: list[str] = []
    monkeypatch.setattr(
        "document_indexer.indexer.build_indexer",
        lambda *args, **kwargs: built.append("built") or MagicMock(),
    )
    monkeypatch.setattr(
        "document_indexer.resume.audit.run_resume_parse_audit",
        lambda settings, with_llm=False: built.append(f"audit:{with_llm}") or [],
    )
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path=str(watch)),
        chunking={"strategy": "resume_project"},
        resume_parse_only=True,
    )
    DocumentIndexer(settings, configure_logs=False).run()
    assert built == ["audit:False"]

    built.clear()
    settings = IndexerSettings(
        _env_file=None,
        source=LocalSourceSettings(watch_path=str(watch)),
        chunking={"strategy": "resume_project"},
        resume_llm_audit=True,
    )
    DocumentIndexer(settings, configure_logs=False).run()
    assert built == ["audit:True"]
