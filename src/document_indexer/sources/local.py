"""Local filesystem source backed by watchdog/inotify."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from document_indexer.config import LocalSourceSettings
from document_indexer.domain.changes import FsChange
from document_indexer.sources.base import ChangeCallback

logger = logging.getLogger(__name__)

# inotify also emits opened/closed when reindex *reads* files. Those must not
# retrigger indexing, or the watcher loops forever.
_TRIGGER_EVENTS = frozenset({"created", "deleted", "modified", "moved"})


def _watchdog_handler_base() -> type:
    try:
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        return object
    return FileSystemEventHandler


class ChangeHandler(_watchdog_handler_base()):
    """Forward relevant FS events as incremental ``FsChange`` records."""

    def __init__(
        self,
        watch_path: str,
        on_change: Callable[[FsChange], None],
    ) -> None:
        super().__init__()
        self._watch_path = watch_path
        self._on_change = on_change

    def on_any_event(self, event: Any) -> None:
        if event.event_type not in _TRIGGER_EVENTS:
            return
        if event.is_directory and event.event_type in {"modified", "created"}:
            return
        logger.debug(
            "FS event: type=%s path=%s dest=%s is_directory=%s",
            event.event_type,
            event.src_path,
            getattr(event, "dest_path", ""),
            event.is_directory,
        )
        for change in self._changes_from_event(event):
            self._on_change(change)

    def _changes_from_event(self, event: Any) -> list[FsChange]:
        if event.event_type == "moved":
            return self._changes_from_move(event)
        relative = self._relative(str(event.src_path))
        if relative is None:
            return []
        if event.is_directory:
            if event.event_type == "deleted":
                return [FsChange("delete", relative, is_prefix=True)]
            return []
        if event.event_type == "deleted":
            return [FsChange("delete", relative)]
        return [FsChange("upsert", relative)]

    def _changes_from_move(self, event: Any) -> list[FsChange]:
        src = self._relative(str(event.src_path))
        dest = self._relative(str(getattr(event, "dest_path", "") or ""))
        changes: list[FsChange] = []
        if event.is_directory:
            if src:
                changes.append(FsChange("delete", src, is_prefix=True))
            if dest:
                dest_dir = Path(self._watch_path) / dest
                if dest_dir.is_dir():
                    for path in dest_dir.rglob("*"):
                        if not path.is_file():
                            continue
                        relative = self._relative(str(path))
                        if relative is not None:
                            changes.append(FsChange("upsert", relative))
            return changes
        if src:
            changes.append(FsChange("delete", src))
        if dest:
            changes.append(FsChange("upsert", dest))
        return changes

    def _relative(self, src: str) -> str | None:
        if not src:
            return None
        try:
            relative = Path(src).relative_to(self._watch_path)
        except ValueError:
            return None
        posix = relative.as_posix()
        if posix == ".":
            return None
        return posix


def create_observer(
    watch_path: str,
    handler: Any,
) -> Any:
    """Create and schedule a recursive observer for ``watch_path``."""
    try:
        from watchdog.observers import Observer
    except ImportError as exc:
        raise RuntimeError(
            "Local source requires watchdog; install with: "
            "pip install 'document-indexer[runtime]'"
        ) from exc
    observer = Observer()
    observer.schedule(handler, path=watch_path, recursive=True)
    return observer


class LocalFilesystemSource:
    """Watch a local directory and emit the original watchdog ``FsChange`` stream."""

    def __init__(self, settings: LocalSourceSettings) -> None:
        self._settings = settings
        self._observer: Any | None = None

    def local_root(self) -> Path:
        return Path(self._settings.watch_path)

    def prepare(self) -> Path:
        watch = self.local_root()
        if not watch.exists():
            raise FileNotFoundError(f"watch_path does not exist: {watch}")
        return watch

    def start(self, on_changes: ChangeCallback) -> None:
        watch = str(self.prepare())
        handler = ChangeHandler(watch, lambda change: on_changes((change,)))
        self._observer = create_observer(watch, handler)
        self._observer.start()
        logger.info("Local watcher started path=%s", watch)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("Local watcher stopped")
