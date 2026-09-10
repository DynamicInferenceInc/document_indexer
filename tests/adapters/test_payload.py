from pathlib import Path

from document_indexer.adapters.qdrant.payload import (
    DefaultPayloadBuilder,
    IndexRecord,
    direction_from_source_path,
    resolve_direction,
)
from document_indexer.domain.models import DocumentChunk


def test_direction_from_source_path() -> None:
    assert direction_from_source_path("Бухгалтерия/акт.docx") == "Бухгалтерия"
    assert direction_from_source_path("Зарплата/подпапка/file.pdf") == "подпапка"
    assert direction_from_source_path(r"Зарплата\file.pdf") == "Зарплата"
    assert direction_from_source_path("акт.docx") == ""
    assert direction_from_source_path("") == ""


def test_resolve_direction_uses_default_for_whole_tree() -> None:
    assert (
        resolve_direction("Казначейство/акт.docx", default="JTI") == "JTI"
    )
    assert resolve_direction("акт.docx", default="JTI") == "JTI"


def test_resolve_direction_map_exact_file_wins() -> None:
    mapping = {
        "Казначейство": "Финансы",
        "Казначейство/акт.docx": "Оплата",
    }
    assert resolve_direction("Казначейство/акт.docx", default="JTI", mapping=mapping) == "Оплата"
    assert resolve_direction("Казначейство/другое.docx", default="JTI", mapping=mapping) == "Финансы"


def test_resolve_direction_map_longest_folder_prefix() -> None:
    mapping = {"Казначейство": "Финансы", "Казначейство/2025": "Казначейство 2025"}
    assert resolve_direction("Казначейство/2025/акт.docx", mapping=mapping) == "Казначейство 2025"


def test_default_payload_includes_direction() -> None:
    record = IndexRecord(
        source_path="Бухгалтерия/акт.docx",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="текст", chunk_type="prose"),
        file_path=Path("/data/staging-hybrid/Бухгалтерия/акт.docx"),
    )
    payload = DefaultPayloadBuilder().build(record)
    assert payload["text"] == "текст"
    assert payload["direction"] == "Бухгалтерия"


def test_default_payload_uses_configured_direction() -> None:
    record = IndexRecord(
        source_path="Казначейство/акт.docx",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="текст"),
        file_path=Path("/data/staging-hybrid/Казначейство/акт.docx"),
    )
    payload = DefaultPayloadBuilder(default_direction="JTI").build(record)
    assert payload["direction"] == "JTI"


def test_default_payload_map_overrides_folder_name() -> None:
    record = IndexRecord(
        source_path="Казначейство/акт.docx",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="текст"),
        file_path=Path("/data/staging-hybrid/Казначейство/акт.docx"),
    )
    payload = DefaultPayloadBuilder(
        default_direction="JTI",
        direction_map={"Казначейство/акт.docx": "Оплата"},
    ).build(record)
    assert payload["direction"] == "Оплата"


def test_default_payload_omits_direction_for_root_file() -> None:
    record = IndexRecord(
        source_path="акт.docx",
        chunk_index=0,
        file_hash="abc",
        chunk=DocumentChunk(text="текст"),
        file_path=Path("/data/staging-hybrid/акт.docx"),
    )
    payload = DefaultPayloadBuilder().build(record)
    assert "direction" not in payload
