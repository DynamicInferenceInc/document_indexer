"""Tests that DocumentIndexer consumes SMB change batches."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from document_indexer.config import IndexerSettings, SmbSourceSettings
from document_indexer.domain.changes import FsChange
from document_indexer.indexer import DocumentIndexer
from document_indexer.ports import Indexer
from document_indexer.sources.smb import RemoteFileMeta, SmbStagingSource


class RecordingIndexer(Indexer):
    def __init__(self) -> None:
        self.reindex_calls: list[str] = []
        self.applied: list[tuple[str, list[FsChange]]] = []

    def index(
        self,
        watch_path: str,
        changes: Sequence[FsChange] | None = None,
    ) -> None:
        if changes is None:
            self.reindex_calls.append(watch_path)
            return
        self.applied.append((watch_path, list(changes)))


class FakeRemote:
    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, RemoteFileMeta]] = {
            "guide.md": (b"hello", RemoteFileMeta(size=5, mtime=1.0)),
        }

    def list_files(self) -> dict[str, RemoteFileMeta]:
        return {relative: meta for relative, (_data, meta) in self.files.items()}

    def stat(self, relative: str) -> RemoteFileMeta:
        return self.files[relative][1]

    def download(self, relative: str, dest: Path) -> None:
        dest.write_bytes(self.files[relative][0])


def test_smb_reindex_once_uses_staging_root(tmp_path: Path) -> None:
    remote = FakeRemote()
    settings = IndexerSettings(
        _env_file=None,
        source=SmbSourceSettings(
            server="fileserver",
            share="docs",
            username="svc",
            password="secret",
            staging_path=str(tmp_path / "staging"),
        ),
    )
    source = SmbStagingSource(settings.source, remote=remote, allowed_extensions={".md"})
    core = RecordingIndexer()
    DocumentIndexer(
        settings,
        indexer=core,
        source=source,
        configure_logs=False,
    ).reindex_once()
    assert core.reindex_calls == [str(tmp_path / "staging")]
    assert (tmp_path / "staging" / "guide.md").read_bytes() == b"hello"
