"""LLM values must exist in the resume text."""

from __future__ import annotations

from document_indexer.resume.grounding import (
    Grounder,
    clean_direction,
    infer_solution_platform,
    normalize_platform,
    normalize_text,
)

TEXT = """
Иванов Иван Иванович
Опыт работы
2019 – 2022  ООО «Ромашка», консультант 1C:ERP
- Внедрение подсистемы Казначейство
- Настройка бюджетирования и платёжного календаря
"""


def test_normalize_text_folds_case_yo_and_latin_1c() -> None:
    assert normalize_text("1C:ERP — Казначейство, платёжный") == "1с erp казначейство платежный"


def test_exact_and_reformatted_substrings_are_grounded() -> None:
    grounder = Grounder(TEXT)
    assert grounder.ground("ООО «Ромашка»", field="customer") == "ООО «Ромашка»"
    assert grounder.ground("консультант 1С:ERP", field="position") == "консультант 1С:ERP"
    assert grounder.ground("2019 – 2022", field="duration") == "2019 – 2022"
    assert grounder.dropped == 0


def test_made_up_values_are_dropped_and_counted() -> None:
    grounder = Grounder(TEXT)
    assert grounder.ground("Газпром нефть", field="customer") is None
    assert grounder.ground("Металлургия", field="project_industry") is None
    assert grounder.dropped == 2
    assert grounder.dropped_values == ["customer=Газпром нефть", "project_industry=Металлургия"]


def test_token_ratio_allows_small_reordering_but_not_new_facts() -> None:
    grounder = Grounder(TEXT, min_ratio=0.85)
    assert grounder.is_grounded("Настройка платёжного календаря и бюджетирования")
    assert not grounder.is_grounded("Настройка бюджетирования в SAP TRM для холдинга")


def test_fragments_keep_only_grounded_items() -> None:
    grounder = Grounder(TEXT)
    value = grounder.ground_fragments(
        "Внедрение подсистемы Казначейство; Миграция на SAP S/4HANA; Настройка бюджетирования",
        field="work_performed",
    )
    assert value == "Внедрение подсистемы Казначейство; Настройка бюджетирования"
    assert grounder.dropped == 1


def test_empty_and_null_like_values() -> None:
    grounder = Grounder(TEXT)
    assert grounder.ground(None) is None
    assert grounder.ground("null") is None
    assert grounder.ground("   ") is None
    assert grounder.ground_fragments("") is None
    assert grounder.dropped == 0


def test_direction_and_platform_normalization() -> None:
    assert clean_direction("Казначейство") == "Казначейство"
    assert clean_direction("x" * 61) is None
    assert normalize_platform("1C") == "1С"
    assert normalize_platform("SAP S/4HANA") == "SAP"
    assert normalize_platform("Oracle") is None
    assert infer_solution_platform("Внедрение 1С:ERP") == "1С"
    assert infer_solution_platform("Проект перехода с SAP R/3 на 1C ERP") == "1С"
    assert infer_solution_platform("внедрение 1С:ЗУП и интеграция с SAP S/4HANA") is None
