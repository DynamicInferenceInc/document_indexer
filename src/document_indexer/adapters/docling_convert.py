"""Docling converter, VLM picture enrichment, and HybridChunker construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ConvertPipelineOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import (
    ExcelFormatOption,
    HTMLFormatOption,
    PdfFormatOption,
    PowerpointFormatOption,
    WordFormatOption,
)
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
    get_default_tokenizer,
)
from docling_core.transforms.serializer.base import BasePictureSerializer
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownTableSerializer,
)
from docling_core.types.doc.document import DocItem

logger = logging.getLogger(__name__)

_TOKENIZER_MODEL_MAX_LENGTH = 8192
_VLM_PROMPT = (
    "Опиши изображение из документации 1С для поиска. "
    "Кратко и точно: что изображено (скрин формы, схема, таблица), "
    "какой видимый текст на кнопках, полях и заголовках. "
    "Не выдумывай то, чего нет на картинке."
)


@dataclass(frozen=True, slots=True)
class PictureDescriptionConfig:
    """Adapter config for Docling picture-description enrichment via Ollama."""

    enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:3b"
    timeout_sec: float = 90.0
    concurrency: int = 2
    area_threshold: float = 0.02

    def format_options(self) -> dict[Any, Any]:
        pdf_options = PdfPipelineOptions(do_ocr=False)
        if not self.enabled:
            return {InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)}
        description = PictureDescriptionApiOptions(
            url=chat_completions_url(self.ollama_base_url),
            params={"model": self.model, "max_completion_tokens": 400},
            prompt=_VLM_PROMPT,
            timeout=self.timeout_sec,
            concurrency=max(1, int(self.concurrency)),
            picture_area_threshold=self.area_threshold,
        )
        pdf_options.do_picture_description = True
        pdf_options.enable_remote_services = True
        pdf_options.generate_picture_images = True
        pdf_options.images_scale = 2
        pdf_options.picture_description_options = description
        convert_options = ConvertPipelineOptions(
            do_picture_description=True,
            enable_remote_services=True,
            picture_description_options=description,
        )
        return {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=convert_options),
            InputFormat.PPTX: PowerpointFormatOption(pipeline_options=convert_options),
            InputFormat.HTML: HTMLFormatOption(pipeline_options=convert_options),
            InputFormat.XLSX: ExcelFormatOption(pipeline_options=convert_options),
        }


def chat_completions_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/v1/chat/completions"


def format_picture_block(*, description: str, caption: str = "") -> str:
    """Render a picture as searchable text, or empty if VLM returned nothing."""
    desc = description.strip()
    cap = caption.strip()
    if not desc:
        return ""
    lines = ["[Изображение]", f"Описание: {desc}"]
    if cap:
        lines.append(f"Подпись: {cap}")
    return "\n".join(lines)


def tokenizer_with_max_tokens(max_tokens: int) -> HuggingFaceTokenizer:
    hf_tokenizer = get_default_tokenizer().tokenizer
    # Default pretrained cap is 512; HybridChunker then warns on tables (1742 > 512)
    # and can truncate token counts used for splitting.
    hf_tokenizer.model_max_length = max(_TOKENIZER_MODEL_MAX_LENGTH, max_tokens * 8)
    logger.debug(
        "HybridChunker tokenizer max_tokens=%s model_max_length=%s",
        max_tokens,
        getattr(hf_tokenizer, "model_max_length", "?"),
    )
    return HuggingFaceTokenizer(tokenizer=hf_tokenizer, max_tokens=max_tokens)


class MarkdownChunkSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc: Any) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
            params=MarkdownParams(compact_tables=False),
            picture_serializer=PictureDescriptionSerializer(),
        )


class PictureDescriptionSerializer(BasePictureSerializer):
    """Serialize PictureItem as captioned VLM text instead of an image tag."""

    def serialize(self, *, item: Any, doc_serializer: Any = None, doc: Any = None, **kwargs: Any) -> Any:
        del doc_serializer, kwargs
        text = format_picture_block(
            description=picture_description(item),
            caption=item_caption(item, doc),
        )
        if not text:
            logger.debug(
                "Skipping picture without VLM description: %s",
                getattr(item, "self_ref", "?"),
            )
        if isinstance(item, DocItem):
            return create_ser_result(text=text, span_source=item)
        return create_ser_result(text=text)


def picture_description(item: Any) -> str:
    annotations = getattr(item, "annotations", None) or ()
    for annotation in annotations:
        text = getattr(annotation, "text", None)
        if text and str(text).strip():
            return str(text).strip()
        nested = getattr(annotation, "description", None)
        if nested and str(nested).strip() and not callable(nested):
            return str(nested).strip()
    meta = getattr(item, "meta", None)
    description = getattr(meta, "description", None) if meta is not None else None
    if description is None:
        return ""
    text = getattr(description, "text", None)
    if text and str(text).strip():
        return str(text).strip()
    if isinstance(description, str) and description.strip():
        return description.strip()
    return ""


def item_caption(item: Any, doc: Any) -> str:
    return str(item.caption_text(doc) or "").strip()
