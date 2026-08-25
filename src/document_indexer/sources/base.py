"""Document source port: a local tree plus incremental ``FsChange`` events."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from document_indexer.domain.changes import FsChange

ChangeCallback = Callable[[Sequence[FsChange]], None]


@runtime_checkable
class DocumentSource(Protocol):
    """Provides a local document tree and optional live change notifications."""

    def local_root(self) -> Path:
        """Directory the indexer should read files from."""
        ...

    def prepare(self) -> Path:
        """Create/sync the local tree and return :meth:`local_root`."""
        ...

    def start(self, on_changes: ChangeCallback) -> None:
        """Begin watching or polling. ``on_changes`` receives coalesced batches."""
        ...

    def stop(self) -> None:
        """Stop watching or polling and release resources."""
        ...
