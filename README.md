# document-indexer

Python-модуль, который читает документы, режет их через Docling HybridChunker и кладёт векторы в Qdrant. Логика обработки документов, чанкирования и записи в Qdrant перенесена из `it-consultant-1c/reindex` без изменения алгоритмов.

Пакет не содержит CLI. Один экземпляр `DocumentIndexer` — один профиль: свой источник, свой Qdrant URL и имя коллекции. Несколько сущностей запускаются как несколько экземпляров (или процессов) с разными настройками.

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
| `QDRANT__INDEX_VERSION` | версия алгоритма в hash/payload; пусто = builder, `hybrid-v1` или `table-aware-v2` | пусто |
| `MODELS__OLLAMA_BASE_URL` | embeddings, VLM, extraction LLM | `http://127.0.0.1:11434` |
| `MODELS__EMBEDDING_MODEL` | модель эмбеддингов | `nomic-embed-text` |
| `MODELS__EXTRACTION_MODEL` | text LLM для полей документа (`/api/chat`) | пусто = без enricher |
| `MODELS__CHUNK_SIZE` | max tokens HybridChunker только для `table_aware` | `1024` |
| `CHUNKING__STRATEGY` | `table_aware`, `resume_project` или `hybrid` | `table_aware` |
| `CHUNKING__WINDOW_CHARS` / `CHUNKING__WINDOW_OVERLAP` | sliding window, если в резюме нет проектов | `1200` / `150` |
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

## Кастомный payload и LLM-поля

`PayloadBuilder` раскладывает уже известные данные по ключам Qdrant. `DocumentEnricher` достаёт поля из текста (не VLM).

Resume-профиль режет документ так: один проект — один чанк (`chunk_type=project`). Если проектов нет — overlapping windows (`chunk_type=prose`). На каждой точке лежат `candidate_name` и `candidate_position` из шапки (ФИО может быть без подписи). `functional_direction` всегда через LLM. `solution_platform` (`1С` или `SAP`) сначала из явного упоминания в роли/описании/работах, иначе из того же LLM-вызова. Строки-заголовки таблицы и неполные копии того же проекта отбрасываются. Пример CV: `examples/resume/sample.md`.

```python
from document_indexer import DocumentIndexer, ProfileLocal, QdrantSettings
from document_indexer.examples.resume import (
    INDEX_VERSION,
    FunctionalDirectionEnricher,
    ResumePayloadBuilder,
    ResumeProjectChunker,
    load_resume_prompt,
    load_resume_schema,
)

settings = ProfileLocal(
    qdrant=QdrantSettings(collection="docs-cv", index_version=INDEX_VERSION),
    models={"extraction_model": "qwen3:8b"},
    chunking={"strategy": "resume_project"},
)
DocumentIndexer(
    settings,
    payload_builder=ResumePayloadBuilder(),
    document_chunker=ResumeProjectChunker(),
    enricher=FunctionalDirectionEnricher(
        load_resume_schema(),
        load_resume_prompt(),
        base_url=settings.models.ollama_base_url,
        model=settings.models.extraction_model,
    ),
).run()
```

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

Новая схема — новая коллекция или bump `index_version`, иначе skip по hash не пересчитает поля.

Без `payload_builder` / `enricher` поведение как у исходного reindex.

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
