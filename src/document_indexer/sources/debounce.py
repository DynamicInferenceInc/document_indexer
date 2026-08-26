"""Coalesce filesystem change records and apply them after a quiet period."""

from __future__ import annotations

import logging
import threading

from document_indexer.domain.changes import FsChange
from document_indexer.ports import Indexer

logger = logging.getLogger(__name__)


class DebouncedReindex:
    """Coalesce filesystem events and apply them after a quiet period."""

    def __init__(
        self,
        indexer: Indexer,
        watch_path: str,
        debounce_seconds: float,
    ) -> None:
        self._indexer = indexer
        self._watch_path = watch_path
        self._debounce_seconds = debounce_seconds
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._running = False
        self._pending: dict[tuple[str, bool], FsChange] = {}

    def notify(self, change: FsChange) -> None:
        """Record a change and (re)schedule apply after the debounce window."""
        with self._lock:
            self._pending[(change.path, change.is_prefix)] = change
            if self._running:
                return
            self._arm_timer_locked()

    def _arm_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._debounce_seconds, self._run)
        self._timer.daemon = True
        self._timer.start()

    def _run(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._timer = None
            changes = list(self._pending.values())
            self._pending.clear()
        if changes:
            logger.info(
                "Debounce elapsed; applying %s change(s) under %s",
                len(changes),
                self._watch_path,
            )
            try:
                self._indexer.apply_changes(self._watch_path, changes)
            except Exception:
                logger.exception(
                    "Indexer.apply_changes failed for %s",
                    self._watch_path,
                )
        with self._lock:
            self._running = False
            has_more = bool(self._pending)
            if has_more:
                self._arm_timer_locked()

    def cancel(self) -> None:
        """Cancel any pending apply timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._pending.clear()
