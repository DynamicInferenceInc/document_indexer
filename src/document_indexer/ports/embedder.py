"""Port: dense embeddings for document chunks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, overload, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Turn text into a dense vector, or a batch of texts into vectors."""

    @overload
    def embed(self, text: str) -> list[float]: ...

    @overload
    def embed(self, text: Sequence[str]) -> list[list[float]]: ...

    def embed(self, text: str | Sequence[str]) -> list[float] | list[list[float]]:
        """Embed one string or a sequence of strings."""
        ...
