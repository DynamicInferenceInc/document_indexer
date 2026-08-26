"""Port: build or refresh an index over the watched directory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, overload, runtime_checkable

from document_indexer.domain.changes import FsChange


@runtime_checkable
class Indexer(Protocol):
    """Builds or refreshes an index over the watched database directory."""

    @overload
    def index(self, watch_path: str) -> None: ...

    @overload
    def index(self, watch_path: str, changes: Sequence[FsChange]) -> None: ...

    def index(
        self,
        watch_path: str,
        changes: Sequence[FsChange] | None = None,
    ) -> None:
        """Rebuild ``watch_path``, or apply incremental ``changes`` when given."""
        ...
