# document-indexer

Python-модуль, который индексирует документы в Qdrant в одном из двух режимов:
`table_aware` (Docling HybridChunker + таблицы) или `resume_project` (проекты CV).
Пакет не содержит CLI. Один экземпляр `DocumentIndexer` — один источник, одна коллекция и одна стратегия.

## Установка

Python 3.12+. Рекомендуется виртуальное окружение.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# CPU-сборка PyTorch нужна Docling; ставьте до extras.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -e ".[docling,runtime,smb,dev]"
```

Extras:

- `docling` — конвертация и HybridChunker
- `runtime` — watchdog для локальной папки
- `smb` — нативный SMB-клиент (`smbprotocol`)
- `dev` — pytest

## Быстрый старт

Локальная папка:

```python
from document_indexer import DocumentIndexer, LocalSourceSettings, ProfileLocal, QdrantSettings

settings = ProfileLocal(
    source=LocalSourceSettings(watch_path="/data/docs"),
    qdrant=QdrantSettings(url="http://127.0.0.1:6333", collection="legal"),
)
indexer = DocumentIndexer(settings)
indexer.reindex_once()   # полная сверка диска и Qdrant
indexer.run()            # сверка, затем наблюдение до stop()/SIGINT
```

SMB-шара (VPN должен быть уже поднят снаружи модуля):

```python
from document_indexer import DocumentIndexer, ProfileSmb, QdrantSettings, SmbSourceSettings

settings = ProfileSmb(
    source=SmbSourceSettings(
        server="fileserver",
        share="docs",
        username="svc_indexer",
        password="...",
        domain="CORP",
        subpath="1c-docs",
        staging_path="/var/lib/document-indexer/legal-staging",
        poll_interval_sec=15,
    ),
    qdrant=QdrantSettings(url="http://127.0.0.1:6333", collection="legal"),
)
indexer = DocumentIndexer(settings)
indexer.run()
```

Два независимых профиля. `IndexerSettings` — общий конструктор: по `source.kind` выбирает local или smb.

```python
from document_indexer import (
    DocumentIndexer,
    LocalSourceSettings,
    ProfileLocal,
    ProfileSmb,
    QdrantSettings,
    SmbSourceSettings,
)

legal = DocumentIndexer(ProfileLocal(
    source=LocalSourceSettings(watch_path="/data/legal"),
    qdrant=QdrantSettings(collection="legal"),
))
hr = DocumentIndexer(ProfileSmb(
    source=SmbSourceSettings(...),
    qdrant=QdrantSettings(url="http://qdrant-b:6333", collection="hr"),
))
```

Языковые модели по умолчанию те же, что в исходном reindex (`nomic-embed-text`, `qwen3-vl:8b`). Их можно переопределить через `ModelSettings` / env, не меняя алгоритмы чанкирования.

## Конфигурация

Настройки — вложенные объекты `source`, `qdrant`, `models` на `IndexerSettings`. `ProfileLocal` и `ProfileSmb` фиксируют тип источника. Новый профиль = модель источника с полем `kind` + подкласс `Profile*`.

`IndexerSettings` читает конструктор, переменные окружения и `.env`. Вложенные ключи через `__`:

| Переменная | Смысл | По умолчанию |
|---|---|---|
| `SOURCE__KIND` | `local` или `smb` | `local` |
| `SOURCE__WATCH_PATH` | локальный корень | `/var/lib/document-indexer/docs` |
| `SOURCE__DEBOUNCE_SECONDS` | пауза перед apply для local | `1.0` |
| `SOURCE__SERVER` / `SOURCE__SHARE` | хост и имя шары | — |
| `SOURCE__USERNAME` / `SOURCE__PASSWORD` / `SOURCE__DOMAIN` | учётная запись | — |
| `SOURCE__SUBPATH` | каталог внутри шары | пусто (корень шары) |
| `SOURCE__STAGING_PATH` | локальное зеркало | `/var/lib/document-indexer/staging` |
| `SOURCE__PORT` | порт SMB | `445` |
| `SOURCE__TIMEOUT_SEC` | таймаут сессии | `30` |
| `SOURCE__POLL_INTERVAL_SEC` | период опроса | `15` |
| `SOURCE__MAX_BACKOFF_SEC` | потолок backoff при сбоях | `60` |
| `QDRANT__URL` / `QDRANT__COLLECTION` | куда писать векторы | `http://127.0.0.1:6333` / `docs` |
| `QDRANT__EXTRA_PAYLOAD` | JSON-константы на каждую точку | `{}` |
| `QDRANT__PAYLOAD_INDEXES` | keyword-индексы (через запятую); пусто = индексы builder’а | builder |
| `QDRANT__DISTANCE` | cosine / dot / euclid | `cosine` |
| `QDRANT__INDEX_VERSION` | версия алгоритма в hash/payload; пусто = `table-aware-v2` или `resume-v20` | пусто |
| `MODELS__OLLAMA_BASE_URL` | embeddings, VLM, extraction LLM | `http://127.0.0.1:11434` |
| `MODELS__EMBEDDING_MODEL` | модель эмбеддингов | `nomic-embed-text` |
| `MODELS__EXTRACTION_MODEL` | text LLM для резюме (`/api/chat`, structured output) | пусто = только парсер, без LLM |
| `MODELS__EXTRACTION_TIMEOUT_SEC` | таймаут одного вызова LLM | `1800` |
| `MODELS__EXTRACTION_NUM_CTX` / `MODELS__EXTRACTION_NUM_PREDICT` | контекст и максимум токенов ответа | `65536` / `8192` |
| `MODELS__EXTRACTION_THINK` | режим размышлений Qwen3 (`think`) | `false` |
| `MODELS__CHUNK_SIZE` | max tokens HybridChunker только для `table_aware` | `1024` |
| `CHUNKING__STRATEGY` | `table_aware` или `resume_project` | `table_aware` |
| `CHUNKING__WINDOW_CHARS` / `CHUNKING__WINDOW_OVERLAP` | prose-окна, если LLM недоступна и проектов нет | `1200` / `150` |
| `RESUME__LLM_PROJECTS` | LLM ищет проекты в неразобранном тексте | `true` |
| `RESUME__LLM_REFINE` | LLM дозаполняет пустые поля и классифицирует направление/платформу | `true` |
| `RESUME__LLM_EXPERIENCE` | LLM строит места работы + профиль, если проектов нет | `true` |
| `RESUME__RESIDUAL_MIN_CHARS` | сколько неразобранного текста нужно, чтобы звать LLM при найденных проектах | `1500` |
| `RESUME__EVIDENCE_MIN_RATIO` | доля токенов значения, которые обязаны быть в тексте резюме | `0.85` |
| `RESUME__SECTION_MAX_CHARS` / `RESUME__SECTION_OVERLAP_CHARS` | секционирование очень длинных резюме | `120000` / `2000` |
| `RESUME_PARSE_ONLY` / `RESUME_LLM_AUDIT` | one-shot аудиты без embed/Qdrant (см. ниже) | `0` / `0` |
| `MODELS__PICTURE_DESCRIPTION_ENABLED` | VLM-описания картинок | `true` |
| `MODELS__VLM_MODEL` | модель описаний | `qwen3-vl:8b` |
| `INDEX_EXTENSIONS` | суффиксы для индексации (проверка на Docling) | пусто = все поддерживаемые |
| `LOG_LEVEL` | уровень логов | `INFO` |

Пароль SMB хранится как `SecretStr` и маскируется в `model_dump_safe()`. Не кладите `.env` в git.

Тот же набор доступен вложенными объектами: `LocalSourceSettings`, `SmbSourceSettings`, `QdrantSettings`, `ModelSettings`. В Python читайте `settings.source`, `settings.qdrant`, `settings.models`.

## Источники

Ядро индексатора всегда читает обычные локальные `pathlib.Path`. Источник только готовит это дерево и шлёт `FsChange`.

**local.** watchdog/inotify, debounce и семантика create/modify/delete/move как в исходном reindex. Каталог `SOURCE__WATCH_PATH` должен существовать.

**smb.** Нативный клиент `smbprotocol`, без CIFS-mount и без inotify на шаре. Модуль:

1. Рекурсивно обходит share/subpath.
2. Скачивает файл во временный `.*.tmp` и публикует через `os.replace`.
3. Перед публикацией повторно читает size/mtime; изменившийся файл не индексируется в этом цикле.
4. Пишет манифест `.document_indexer_smb_manifest.json` в staging (скрытый, в индекс не попадает).
5. После первой синхронизации вызывает полный `index(staging)`; дальше — `index(staging, changes)`.

`source_path` в Qdrant — POSIX-путь относительно `SOURCE__SUBPATH`, не UNC и не путь staging. Регистр имён сохраняется.

### Нюансы SMB и VPN

- VPN и маршруты до файлового сервера модуль не поднимает: он ожидает, что TCP/445 уже доступен.
- inotify по CIFS ненадёжен, поэтому используется polling.
- Неполный или упавший listing **не считается** массовым удалением: staging и Qdrant не чистятся.
- При обрыве VPN действует ограниченный exponential backoff. После восстановления — обычная сверка.
- Частичная запись на шаре (файл ещё копируется) отсекается проверкой size/mtime до `os.replace`.
- Учётные данные лучше держать в env-файле процесса, не в коде.

## Совместимость с Qdrant

Payload точек по умолчанию не менялся: `source_path`, `chunk_index`, `text`, `file_hash`, `index_version=table-aware-v2`, поля таблиц. Существующий RAG-клиент может читать коллекции как раньше. Коллекция при reconcile не дропается.

Кастомные поля **добавляются** к этим ключам. Ядро всегда перезаписывает `source_path`, `chunk_index`, `file_hash`, `index_version`.

## Два режима индексации

`CHUNKING__STRATEGY` выбирает один из двух встроенных режимов. Плагины чанкера, payload и enricher снаружи не подключаются.

**table_aware** (по умолчанию). Docling HybridChunker + постобработка таблиц: одна таблица — один чанк. Payload: `text`, `headings`, `chunk_type`, поля таблиц. Версия `table-aware-v2`.

**resume_project**. Один проект — один чанк (`chunk_type=project`). На каждой точке лежат `candidate_name` и `candidate_position` из шапки (ФИО может быть без подписи). Строки-заголовки таблицы и неполные копии того же проекта отбрасываются. Пример CV: `resume/sample.md`.

Порядок обработки одного резюме (`resume/chunker.py`):

1. **Парсер** (`resume/parser.py`) — шапка и проекты из Docling-таблиц / размеченных блоков. Поля проекта: `customer`, `duration`, `project_industry`, `project_description`, `project_position`, `work_performed`.
2. **LLM-1, проекты** (`RESUME__LLM_PROJECTS`) — вызывается, если парсер не нашёл проектов, либо после парсера осталось ≥ `RESUME__RESIDUAL_MIN_CHARS` неразобранного текста с признаками опыта (диапазоны дат, «заказчик», «внедрение», «ООО» …). Модель видит только `residual_text` — то, что парсер не понял. Результат сливается с проектами парсера (dedup), у чанков `extraction_source=llm`.
3. **LLM-2, дозаполнение и классификация** (`RESUME__LLM_REFINE`) — один вызов на резюме: полный текст + список проектов. JSON Schema строится под каждый вызов — в неё попадают только **пустые** поля каждого проекта плюс `functional_direction` и `solution_platform` (правила из `resume/prompt.txt`). Значения парсера никогда не перезаписываются. `solution_platform` (`1С` / `SAP`): явное упоминание в тексте проекта имеет приоритет над ответом модели.
4. **LLM-3, места работы + профиль** (`RESUME__LLM_EXPERIENCE`) — только если проектов нет ни у парсера, ни у LLM-1. На каждое место работы — чанк `chunk_type=experience` в тех же шести полях (`customer`=работодатель, `duration`=период, `project_position`=должность, `work_performed`=что делал, `project_description`=системы). Плюс один чанк `chunk_type=profile` на резюме: стаж, навыки, платформы, направления.
5. **prose** — только если LLM выключена (`MODELS__EXTRACTION_MODEL` пуст) или упала: скользящие окна `CHUNKING__WINDOW_*` с `needs_review=true` и `review_reason` в payload.

**Защита от выдумывания** (`resume/grounding.py`). Каждое значение от LLM (кроме `functional_direction` — короткая метка ≤ 60 символов) проверяется по нормализованному тексту резюме: либо подстрока, либо ≥ `RESUME__EVIDENCE_MIN_RATIO` его токенов (≥ 3 символов) встречаются в тексте; списки (`work_performed`, `skills` …) проверяются по фрагментам через `;`. Не прошедшее → `null` + `WARNING` в логе + счётчик `ungrounded_dropped`. Промпты требуют дословных формулировок, `temperature=0`, схемы с `additionalProperties:false`.

Длинные резюме: текст делится на секции по абзацам (`RESUME__SECTION_MAX_CHARS`, по умолчанию с запасом под `num_ctx=65536`), LLM-шаги идут по секциям, результаты сливаются dedup-ом.

Payload проектных точек (`project` и `experience`): `candidate_name`, `candidate_position`, `customer`, `duration`, `project_industry`, `project_description`, `project_position`, `work_performed`, `functional_direction`, `solution_platform`, `extraction_source` (`parser` | `llm` | `experience`). У `profile`: `total_experience`, `skills`, `platforms`, `directions`. У `prose`: `needs_review`, `review_reason`. Версия `resume-v20`.

```python
from document_indexer import DocumentIndexer, ProfileLocal, QdrantSettings

settings = ProfileLocal(
    qdrant=QdrantSettings(collection="docs-cv"),
    models={"extraction_model": "qwen3.8:27b-q8_0"},
    chunking={"strategy": "resume_project"},
)
DocumentIndexer(settings).run()
```

### Модель и параметры под NVIDIA DGX Spark

Spark (GB10, 128 ГБ unified, ARM64) упирается в пропускную способность памяти: ~10 t/s decode на 27B, поэтому ограничение — длина ответа, а не контекст. Рекомендованные значения (они же по умолчанию в коде и в `.env.*.example`):

- `MODELS__EXTRACTION_MODEL=qwen3.8:27b-q8_0` (если тега нет — `qwen3.8:27b`), `MODELS__EXTRACTION_NUM_CTX=65536`, `MODELS__EXTRACTION_NUM_PREDICT=8192`, `MODELS__EXTRACTION_TIMEOUT_SEC=1800`, `MODELS__EXTRACTION_THINK=false`.
- Клиент шлёт `temperature=0`, `keep_alive=-1` (модель не выгружается между файлами), запросы последовательные.
- На стороне Ollama: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_NUM_PARALLEL=1`.
- Образ собирается на самом Spark (`docker compose build`): база `python:3.12-slim-bookworm` мультиархитектурная, `torch`/`torchvision` CPU-wheels для `linux_aarch64` берутся с `download.pytorch.org/whl/cpu` с fallback на PyPI. Индексатор GPU не использует, GPU целиком у Ollama.
- Оценка: шаблонное резюме — 1 вызов (LLM-2) ≈ 1–2 мин, нешаблонное — 2–3 вызова ≈ 5–10 мин.

### Аудиты и отчёт

Два one-shot режима без embed и Qdrant (только для `resume_project`):

- `RESUME_PARSE_ONLY=1` — Docling + парсер.
- `RESUME_LLM_AUDIT=1` — Docling + парсер + все LLM-шаги. Дополнительно пишет `.resume_chunks.jsonl` (все чанки: `source_path`, `chunk_type`, `text`, `fields`) для ручной сверки с исходниками.

В обоих режимах и после каждого полного `reindex` печатается и сохраняется отчёт `resume_report` (`<watch>/.resume_report.csv|.txt`, иначе рядом с watch-каталогом или в `/tmp`):

```text
ФИО                   Должность            Проектов  из них LLM  Мест работы  Проверить  Файл
Иванов Иван Иванович  ведущий консультант  5         2           0                       a/ivanov.docx
?                     роль не распознана   0         0           3                       b/x.docx

Итого: резюме=2 с проектами=1 только места работы=1 требуют проверки=0 ошибок=0 проектов всего=5 (из них LLM=2)
Без распознанного ФИО (1): …
```

После reindex источник отчёта — payload из Qdrant (фактическое состояние коллекции, включая файлы, пропущенные по хешу). В CSV дополнительно `profile_count`, `prose_count`, `needs_review`, `error`.

Фильтр по полям одного проекта — обычный `FieldCondition` (массива `project_experiences` больше нет):

```python
from qdrant_client.http import models as qmodels

project_filter = qmodels.Filter(
    must=[
        qmodels.FieldCondition(
            key="candidate_name",
            match=qmodels.MatchValue(value="Иванов Иван Иванович"),
        ),
        qmodels.FieldCondition(
            key="functional_direction",
            match=qmodels.MatchValue(value="Казначейство"),
        ),
    ]
)
```

Новая схема — новая коллекция или bump `index_version`, иначе skip по hash не пересчитает поля. `resume-v20` форсирует полную переиндексацию всех резюме.

## Тесты

```bash
source .venv/bin/activate
pytest
```

Живой SMB-тест включается только явно:

```bash
DOCUMENT_INDEXER_SMB_TEST=1 SMB_SERVER=... SMB_SHARE=... \
  SMB_USERNAME=... SMB_PASSWORD=... pytest -m smb_integration
```
