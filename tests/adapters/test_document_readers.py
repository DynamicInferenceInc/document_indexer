from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.table.table_data import TableData

from document_indexer.table_aware.chunker import split_oversized_text
from document_indexer.adapters.docling_convert import (
    chat_completions_url,
    format_picture_block,
)
from document_indexer.adapters.document_readers import DoclingDocumentReader
from document_indexer.domain.models import DocumentChunk


def _table(self_ref: str) -> TableItem:
    return TableItem(self_ref=self_ref, data=TableData())


def _document(*tables: TableItem) -> SimpleNamespace:
    return SimpleNamespace(tables=list(tables))


class _Serializer:
    def __init__(self, rendered: dict[str, str]) -> None:
        self._rendered = rendered

    def serialize(self, *, item):
        return SimpleNamespace(text=self._rendered.get(item.self_ref, ""))


def _chunker(*, raws, rendered: dict[str, str] | None = None):
    provider = MagicMock()
    if rendered is not None:
        provider.get_serializer.return_value = _Serializer(rendered)
    chunker = MagicMock()
    chunker.chunk.return_value = raws
    chunker.serializer_provider = provider
    return chunker


def test_docling_reader_unsupported_suffix(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"\x00\x01")
    try:
        DoclingDocumentReader(MagicMock(), MagicMock()).read(path)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unsupported" in str(exc)


def test_docling_reader_hybrid_chunks(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = ["Раздел 1", "Подраздел"]
    result = MagicMock()
    result.document = _document()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "Раздел 1\n\nтекст чанка"

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)

    assert chunks == [
        DocumentChunk(text="Раздел 1\n\nтекст чанка", headings=("Раздел 1", "Подраздел")),
    ]
    converter.convert.assert_called_once_with(str(path))
    chunker.chunk.assert_called_once_with(dl_doc=result.document)
    chunker.contextualize.assert_called_once_with(raw_chunk)


def test_docling_reader_reads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text("# Title", encoding="utf-8")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = ["Title"]
    result = MagicMock()
    result.document = _document()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "# Title\n\nbody"

    reader = DoclingDocumentReader(converter=converter, chunker=chunker)
    chunks = list(reader.read(path))
    assert chunks[0].text == "# Title\n\nbody"
    assert chunks[0].headings == ("Title",)
    converter.convert.assert_called_once_with(str(path))


def test_docling_reader_reads_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")

    raw_chunk = MagicMock()
    raw_chunk.meta.headings = []
    result = MagicMock()
    result.document = _document()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [raw_chunk]
    chunker.contextualize.return_value = "| a | b |\n|---|---|\n| 1 | 2 |"

    reader = DoclingDocumentReader(converter=converter, chunker=chunker)
    chunks = list(reader.read(path))
    assert "1 | 2" in chunks[0].text
    converter.convert.assert_called_once_with(str(path))


def test_docling_reader_skips_empty_contextualized_chunks(tmp_path: Path) -> None:
    path = tmp_path / "a.docx"
    path.write_bytes(b"PK")

    empty = MagicMock()
    empty.meta.headings = []
    filled = MagicMock()
    filled.meta.headings = ["H"]
    result = MagicMock()
    result.document = _document()
    converter = MagicMock()
    converter.convert.return_value = result
    chunker = MagicMock()
    chunker.chunk.return_value = [empty, filled]
    chunker.contextualize.side_effect = ["  ", "kept"]

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)
    assert chunks == [DocumentChunk(text="kept", headings=("H",))]


def test_format_picture_block_includes_description_and_caption() -> None:
    text = format_picture_block(
        description="Форма настройки обмена, кнопка Записать",
        caption="Рис. 1. Обмен",
    )
    assert text.startswith("[Изображение]")
    assert "Описание: Форма настройки обмена, кнопка Записать" in text
    assert "Подпись: Рис. 1. Обмен" in text
    assert "<!-- image -->" not in text


def test_format_picture_block_empty_description() -> None:
    assert format_picture_block(description="  ", caption="Рис. 1") == ""
    assert "<!-- image -->" not in format_picture_block(description="")


def test_chat_completions_url_uses_ollama_base() -> None:
    assert (
        chat_completions_url("http://127.0.0.1:11434")
        == "http://127.0.0.1:11434/v1/chat/completions"
    )
    assert (
        chat_completions_url("http://ollama:11434/")
        == "http://ollama:11434/v1/chat/completions"
    )


def test_picture_serializer_uses_annotation_and_caption() -> None:
    from document_indexer.adapters.docling_convert import PictureDescriptionSerializer

    item = MagicMock()
    annotation = MagicMock()
    annotation.text = "кнопка Провести"
    item.annotations = [annotation]
    item.caption_text = lambda doc=None: "Скрин формы"
    item.self_ref = "#/pictures/0"
    result = PictureDescriptionSerializer().serialize(item=item, doc=None)
    text = result if isinstance(result, str) else getattr(result, "text", "")
    assert "[Изображение]" in text
    assert "Описание: кнопка Провести" in text
    assert "Подпись: Скрин формы" in text


def test_picture_serializer_is_accepted_by_chunking_doc_serializer() -> None:
    pytest.importorskip("docling_core")
    from docling_core.transforms.chunker.hierarchical_chunker import ChunkingDocSerializer
    from docling_core.types.doc.document import DoclingDocument

    from document_indexer.adapters.docling_convert import PictureDescriptionSerializer

    serializer = ChunkingDocSerializer(
        doc=DoclingDocument(name="probe"),
        picture_serializer=PictureDescriptionSerializer(),
    )
    assert serializer.picture_serializer is not None


class _CountingTokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)


def test_split_oversized_text_keeps_short_chunk() -> None:
    assert split_oversized_text("hello", max_tokens=32) == ["hello"]


def test_split_oversized_text_splits_paragraphs() -> None:
    part_a = "a" * 40
    part_b = "b" * 40
    pieces = split_oversized_text(
        f"{part_a}\n\n{part_b}",
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert len(pieces) == 2
    assert part_a in pieces[0]
    assert part_b in pieces[1]


def test_split_oversized_text_keeps_whole_table_as_one_chunk() -> None:
    table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| Intern 1 | адаптация |\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    pieces = split_oversized_text(
        table,
        max_tokens=20,
        tokenizer=_CountingTokenizer(),
    )
    assert pieces == [table]
    assert "K0-2 | PM готов покупать по K0" in pieces[0]
    assert "K1 | после K0-2" in pieces[0]


def test_split_oversized_text_extracts_table_from_surrounding_prose() -> None:
    table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    text = f"{'a' * 40}\n\n{table}\n\n{'b' * 40}"
    pieces = split_oversized_text(
        text,
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert len(pieces) == 3
    assert pieces[0] == "a" * 40
    assert pieces[1] == table
    assert pieces[2] == "b" * 40


def test_split_oversized_text_keeps_html_table_as_one_chunk() -> None:
    table = (
        "<table><thead><tr><th>Этап</th><th>Условие</th></tr></thead>"
        "<tbody><tr><td>K0-2</td><td>PM готов покупать по K0</td></tr>"
        "<tr><td>K1</td><td>после K0-2</td></tr></tbody></table>"
    )
    pieces = split_oversized_text(
        f"{'a' * 40}\n\n{table}\n\n{'b' * 40}",
        max_tokens=12,
        tokenizer=_CountingTokenizer(),
    )
    assert pieces[1] == table
    assert "K0-2" in pieces[1] and "K1" in pieces[1]


def test_docling_reader_renders_full_table_item_once(tmp_path: Path) -> None:
    path = tmp_path / "grades.pptx"
    path.write_bytes(b"PK")

    full_table = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| Intern 1 | адаптация |\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    table = _table("#/tables/0")
    first = MagicMock()
    first.meta.headings = ["Критерии"]
    first.meta.doc_items = [table]
    second = MagicMock()
    second.meta.headings = ["Критерии"]
    second.meta.doc_items = [table]
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(table))
    chunker = _chunker(
        raws=[first, second],
        rendered={table.self_ref: full_table},
    )

    chunks = DoclingDocumentReader(
        converter=converter,
        chunker=chunker,
        max_tokens=8,
    ).read(path)

    assert len(chunks) == 1
    assert chunks[0].atomic is True
    assert chunks[0].text == full_table
    assert "Intern 1 | адаптация" in chunks[0].text
    assert "K1 | после K0-2" in chunks[0].text
    assert chunks[0].headings == ("Критерии",)
    assert chunks[0].chunk_type == "table"
    assert chunks[0].table_ref == "#/tables/0"
    chunker.contextualize.assert_not_called()


def test_docling_reader_splits_wide_table_embedding_parts(tmp_path: Path) -> None:
    path = tmp_path / "sheet.xlsx"
    path.write_bytes(b"PK")
    markdown = (
        "| Этап | Условие |\n"
        "|------|---------|\n"
        "| K0-2 | PM готов покупать по K0 |\n"
        "| K1 | после K0-2 |"
    )
    table = _table("#/tables/3")
    raw = MagicMock()
    raw.meta.headings = ["Лист"]
    raw.meta.doc_items = [table]
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(table))
    chunker = _chunker(raws=[raw], rendered={table.self_ref: markdown})

    chunks = DoclingDocumentReader(
        converter=converter,
        chunker=chunker,
        max_tokens=4,
    ).read(path)

    assert len(chunks) == 1
    assert chunks[0].atomic is True
    assert chunks[0].text.startswith("| Этап | Условие |")
    assert "K0-2" in chunks[0].text
    assert "K1" in chunks[0].text
    assert len(chunks[0].embedding_parts) >= 2
    assert all("[Таблица]" in part for part in chunks[0].embedding_parts)


def test_docling_reader_preserves_mixed_prose_table_order(tmp_path: Path) -> None:
    path = tmp_path / "mixed.docx"
    path.write_bytes(b"PK")

    before = type("TextItem", (), {"self_ref": "#/texts/0"})()
    after = type("TextItem", (), {"self_ref": "#/texts/1"})()
    table = _table("#/tables/0")
    raw = MagicMock()
    raw.meta.headings = ["Раздел"]
    raw.meta.doc_items = [before, table, after]
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(table))
    chunker = _chunker(
        raws=[raw],
        rendered={
            before.self_ref: "Текст до таблицы",
            table.self_ref: "| A | B |\n|---|---|\n| 1 | 2 |",
            after.self_ref: "Текст после таблицы",
        },
    )

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)

    assert [chunk.chunk_type for chunk in chunks] == ["prose", "table", "prose"]
    assert [chunk.text for chunk in chunks] == [
        "Текст до таблицы",
        "| A | B |\n|---|---|\n| 1 | 2 |",
        "Текст после таблицы",
    ]


def test_docling_reader_keeps_same_header_tables_separate(tmp_path: Path) -> None:
    path = tmp_path / "two-tables.pptx"
    path.write_bytes(b"PK")
    markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    tables = [_table(f"#/tables/{index}") for index in range(2)]
    raws = []
    for table in tables:
        raw = MagicMock()
        raw.meta.headings = ["Один раздел"]
        raw.meta.doc_items = [table]
        raws.append(raw)
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(*tables))
    chunker = _chunker(
        raws=raws,
        rendered={table.self_ref: markdown for table in tables},
    )

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)

    assert len(chunks) == 2
    assert [chunk.table_ref for chunk in chunks] == ["#/tables/0", "#/tables/1"]


def test_docling_reader_skips_structural_table_without_content(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty-table.pptx"
    path.write_bytes(b"PK")
    table = _table("#/tables/0")
    raw = MagicMock()
    raw.meta.headings = []
    raw.meta.doc_items = [table]
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(table))
    chunker = _chunker(raws=[raw], rendered={table.self_ref: "|\n|------|------|"})

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)
    assert chunks == []


def test_docling_reader_indexes_when_one_omitted_table_is_empty(tmp_path: Path) -> None:
    path = tmp_path / "mixed.docx"
    path.write_bytes(b"PK")
    useful = _table("#/tables/0")
    empty = _table("#/tables/1")
    raw = MagicMock()
    raw.meta.headings = []
    raw.meta.doc_items = [useful]
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document(useful, empty))
    markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
    chunker = _chunker(
        raws=[raw],
        rendered={
            useful.self_ref: markdown,
            empty.self_ref: "|\n|------|------|",
        },
    )

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)
    assert len(chunks) == 1
    assert chunks[0].table_ref == "#/tables/0"


def test_docling_reader_drops_separator_only_non_table_chunk(tmp_path: Path) -> None:
    path = tmp_path / "junk.pptx"
    path.write_bytes(b"PK")
    raw = MagicMock()
    raw.meta.headings = ["Раздел"]
    raw.meta.doc_items = []
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(document=_document())
    chunker = _chunker(raws=[raw])
    chunker.contextualize.return_value = (
        "|\n|-----------------------------------------------------------------------|"
    )

    chunks = DoclingDocumentReader(converter=converter, chunker=chunker).read(path)

    assert chunks == []


def test_vlm_http_logging_records_chat_completions(monkeypatch, caplog) -> None:
    import logging as logging_mod

    requests = pytest.importorskip("requests")

    class DummyResponse:
        ok = True
        status_code = 200

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "скрин формы обмена"}}]}

    def fake_post(self, url, *args, **kwargs):
        return DummyResponse()

    monkeypatch.setattr(requests.Session, "post", fake_post)
    from document_indexer.adapters.document_readers import log_vlm_http

    caplog.set_level(logging_mod.INFO)
    with log_vlm_http():
        session = requests.Session()
        session.post("https://huggingface.co/api/models/x")
        session.post(
            "http://spark.pers.local:11434/v1/chat/completions",
            json={"model": "qwen3-vl:8b"},
        )

    assert "VLM request sent" in caplog.text
    assert "qwen3-vl:8b" in caplog.text
    assert "huggingface.co" not in caplog.text
