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
from document_indexer import DocumentIndexer, IndexerSettings, LocalSourceSettings, QdrantSettings

settings = IndexerSettings(
    source=LocalSourceSettings(watch_path="/data/docs"),
    qdrant=QdrantSettings(url="http://127.0.0.1:6333", collection="legal"),
)
indexer = DocumentIndexer(settings)
indexer.reindex_once()   # полная сверка диска и Qdrant
indexer.run()            # сверка, затем наблюдение до stop()/SIGINT
```

SMB-шара (VPN должен быть уже поднят снаружи модуля):

```python
from document_indexer import DocumentIndexer, IndexerSettings, SmbSourceSettings, QdrantSettings

settings = IndexerSettings(
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

Два независимых профиля:

```python
legal = DocumentIndexer(IndexerSettings(
    watch_path="/data/legal",
    qdrant_collection="legal",
))
hr = DocumentIndexer(IndexerSettings(
    source=SmbSourceSettings(...),
    qdrant_url="http://qdrant-b:6333",
    qdrant_collection="hr",
))
```

Языковые модели по умолчанию те же, что в исходном reindex (`nomic-embed-text`, `qwen3-vl:8b`). Их можно переопределить через `ModelSettings` / env, не меняя алгоритмы чанкирования.

## Конфигурация

`IndexerSettings` читает конструктор, переменные окружения и `.env`. Основные ключи:

| Переменная | Смысл | По умолчанию |
|---|---|---|
| `SOURCE_TYPE` | `local` или `smb` | `local` |
| `WATCH_PATH` | локальный корень | `/var/lib/document-indexer/docs` |
| `DEBOUNCE_SECONDS` | пауза перед apply для local | `1.0` |
| `SMB_SERVER` / `SMB_SHARE` | хост и имя шары | — |
| `SMB_USERNAME` / `SMB_PASSWORD` / `SMB_DOMAIN` | учётная запись | — |
| `SMB_SUBPATH` | каталог внутри шары | пусто (корень шары) |
| `SMB_STAGING_PATH` | локальное зеркало | `/var/lib/document-indexer/staging` |
| `SMB_PORT` | порт SMB | `445` |
| `SMB_TIMEOUT_SEC` | таймаут сессии | `30` |
| `SMB_POLL_INTERVAL_SEC` | период опроса | `15` |
| `SMB_MAX_BACKOFF_SEC` | потолок backoff при сбоях | `60` |
| `QDRANT_URL` / `QDRANT_COLLECTION` | куда писать векторы | `http://127.0.0.1:6333` / `docs` |
| `OLLAMA_BASE_URL` | embeddings и VLM | `http://127.0.0.1:11434` |
| `EMBEDDING_MODEL` | модель эмбеддингов | `nomic-embed-text` |
| `CHUNK_SIZE` | max tokens HybridChunker | `1024` |
| `PICTURE_DESCRIPTION_ENABLED` | VLM-описания картинок | `true` |
| `VLM_MODEL` | модель описаний | `qwen3-vl:8b` |
| `INDEX_EXTENSIONS` | список суффиксов | `.txt,.md,...,.htm` |
| `LOG_LEVEL` | уровень логов | `INFO` |

Пароль SMB хранится как `SecretStr` и маскируется в `model_dump_safe()`. Не кладите `.env` в git.

Тот же набор доступен вложенными объектами: `LocalSourceSettings`, `SmbSourceSettings`, `QdrantSettings`, `ModelSettings`.

## Источники

Ядро индексатора всегда читает обычные локальные `pathlib.Path`. Источник только готовит это дерево и шлёт `FsChange`.

**local.** watchdog/inotify, debounce и семантика create/modify/delete/move как в исходном reindex. Каталог `WATCH_PATH` должен существовать.

**smb.** Нативный клиент `smbprotocol`, без CIFS-mount и без inotify на шаре. Модуль:

1. Рекурсивно обходит share/subpath.
2. Скачивает файл во временный `.*.tmp` и публикует через `os.replace`.
3. Перед публикацией повторно читает size/mtime; изменившийся файл не индексируется в этом цикле.
4. Пишет манифест `.document_indexer_smb_manifest.json` в staging (скрытый, в индекс не попадает).
5. После первой синхронизации вызывает полный `reindex(staging)`; дальше — `apply_changes`.

`source_path` в Qdrant — POSIX-путь относительно `SMB_SUBPATH`, не UNC и не путь staging. Регистр имён сохраняется.

### Нюансы SMB и VPN

- VPN и маршруты до файлового сервера модуль не поднимает: он ожидает, что TCP/445 уже доступен.
- inotify по CIFS ненадёжен, поэтому используется polling.
- Неполный или упавший listing **не считается** массовым удалением: staging и Qdrant не чистятся.
- При обрыве VPN действует ограниченный exponential backoff. После восстановления — обычная сверка.
- Частичная запись на шаре (файл ещё копируется) отсекается проверкой size/mtime до `os.replace`.
- Учётные данные лучше держать в env-файле процесса, не в коде.

## Совместимость с Qdrant

Payload точек не менялся: `source_path`, `chunk_index`, `text`, `file_hash`, `index_version=table-aware-v2`, поля таблиц. Существующий RAG-клиент может читать коллекции как раньше. Коллекция при reconcile не дропается.

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
