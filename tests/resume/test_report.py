"""ФИО / должность / project counts report."""

from __future__ import annotations

import csv
from pathlib import Path

from document_indexer.domain.models import DocumentChunk
from document_indexer.resume.report import (
    collect_resume_report,
    format_resume_report,
    row_from_chunks,
    write_resume_report,
)


def _rows() -> list[dict]:
    return [
        {
            "source_path": "b.docx",
            "candidate_name": "Борисов Борис",
            "candidate_position": "стажер",
            "project_count": 0,
            "llm_project_count": 0,
            "experience_count": 0,
            "profile_count": 0,
            "prose_count": 12,
            "needs_review": True,
            "error": None,
        },
        {
            "source_path": "a.docx",
            "candidate_name": "Антонов Антон",
            "candidate_position": "архитектор",
            "project_count": 3,
            "llm_project_count": 1,
            "experience_count": 0,
            "profile_count": 0,
            "prose_count": 0,
            "needs_review": False,
            "error": None,
        },
        {
            "source_path": "c.docx",
            "candidate_name": None,
            "candidate_position": None,
            "project_count": 0,
            "llm_project_count": 0,
            "experience_count": 2,
            "profile_count": 1,
            "prose_count": 0,
            "needs_review": False,
            "error": None,
        },
        {
            "source_path": "d.docx",
            "candidate_name": None,
            "candidate_position": None,
            "project_count": 0,
            "llm_project_count": 0,
            "experience_count": 0,
            "profile_count": 0,
            "prose_count": 0,
            "needs_review": False,
            "error": "ValueError: broken",
        },
    ]


def test_format_resume_report_table_and_totals() -> None:
    text = format_resume_report(_rows(), title="Отчёт")
    lines = text.splitlines()
    assert lines[0] == "Отчёт"
    assert lines[1].startswith("ФИО")
    assert "Должность" in lines[1] and "Проектов" in lines[1]
    # sorted by ФИО, unnamed last
    names = [line.split("  ")[0].strip() for line in lines[3:7]]
    assert names == ["Антонов Антон", "Борисов Борис", "?", "?"]
    antonov = lines[3]
    assert "архитектор" in antonov and "  3  " in antonov and "a.docx" in antonov
    borisov = lines[4]
    assert "стажер" in borisov and "да" in borisov
    assert "err" in lines[6]
    assert (
        "Итого: резюме=4 с проектами=1 только места работы=1 требуют проверки=1 "
        "ошибок=1 проектов всего=3 (из них LLM=1)"
    ) in text
    assert "Без распознанного ФИО (1):" in text
    assert "Без распознанной должности (1):" in text
    assert "Требуют ручной проверки (1):" in text
    assert "d.docx  ValueError: broken" in text


def test_write_resume_report_csv_and_txt(tmp_path: Path) -> None:
    watch = tmp_path / "cv"
    watch.mkdir()
    csv_path, txt_path = write_resume_report(str(watch), _rows())
    assert csv_path == watch / ".resume_report.csv"
    assert txt_path == watch / ".resume_report.txt"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["candidate_name"] for row in rows] == ["Антонов Антон", "Борисов Борис", "", ""]
    assert rows[0]["project_count"] == "3"
    assert rows[0]["candidate_position"] == "архитектор"
    assert "Итого:" in txt_path.read_text(encoding="utf-8")


def test_row_from_chunks_counts_every_chunk_type() -> None:
    chunks = [
        DocumentChunk(
            text="p",
            chunk_type="project",
            extra_fields={"candidate_name": "И", "candidate_position": "к", "extraction_source": "parser"},
        ),
        DocumentChunk(text="p", chunk_type="project", extra_fields={"extraction_source": "llm"}),
        DocumentChunk(text="e", chunk_type="experience", extra_fields={}),
        DocumentChunk(text="pr", chunk_type="profile", extra_fields={}),
        DocumentChunk(text="w", chunk_type="prose", extra_fields={"needs_review": True}),
    ]
    row = row_from_chunks("x.docx", chunks)
    assert row["candidate_name"] == "И"
    assert row["candidate_position"] == "к"
    assert row["project_count"] == 2
    assert row["llm_project_count"] == 1
    assert row["experience_count"] == 1
    assert row["profile_count"] == 1
    assert row["prose_count"] == 1
    assert row["needs_review"] is True


def test_collect_resume_report_reads_position_from_any_point() -> None:
    rows = collect_resume_report(
        [
            {"source_path": "a.docx", "chunk_type": "project", "candidate_name": "А"},
            {"source_path": "a.docx", "chunk_type": "project", "candidate_position": "лид"},
        ]
    )
    assert rows[0]["candidate_name"] == "А"
    assert rows[0]["candidate_position"] == "лид"
    assert rows[0]["project_count"] == 2
