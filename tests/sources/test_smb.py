"""Unit tests for SMB staging without a live share."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from document_indexer.config import SmbSourceSettings
from document_indexer.domain.changes import FsChange
from document_indexer.sources.smb import (
    RemoteFileMeta,
    SmbListingError,
    SmbStagingSource,
)


class FakeRemote:
    def __init__(self) -> None:
        self.files: dict[str, tuple[bytes, RemoteFileMeta]] = {}
        self.list_error: Exception | None = None
        self.stat_overrides: dict[str, RemoteFileMeta] = {}
        self.downloads: list[str] = []

    def put(self, relative: str, data: bytes, *, size: int | None = None, mtime: float = 1.0) -> None:
        meta = RemoteFileMeta(size=size if size is not None else len(data), mtime=mtime)
        self.files[relative] = (data, meta)

    def list_files(self) -> dict[str, RemoteFileMeta]:
        if self.list_error is not None:
            raise self.list_error
        return {relative: meta for relative, (_data, meta) in self.files.items()}

    def stat(self, relative: str) -> RemoteFileMeta:
        if relative in self.stat_overrides:
            return self.stat_overrides[relative]
        return self.files[relative][1]

    def download(self, relative: str, dest: Path) -> None:
        self.downloads.append(relative)
        dest.write_bytes(self.files[relative][0])


def _source(tmp_path: Path, remote: FakeRemote) -> SmbStagingSource:
    settings = SmbSourceSettings(
        server="fileserver",
        share="docs",
        username="svc",
        password="secret",
        staging_path=str(tmp_path / "staging"),
        poll_interval_sec=0.05,
        max_backoff_sec=0.4,
    )
    return SmbStagingSource(
        settings,
        remote=remote,
        allowed_extensions={".md", ".txt", ".pdf"},
    )


def test_smb_prepare_downloads_and_preserves_relative_paths(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("guide.md", b"hello")
    remote.put("sub/nested.txt", b"nested")
    source = _source(tmp_path, remote)

    root = source.prepare()
    assert (root / "guide.md").read_bytes() == b"hello"
    assert (root / "sub" / "nested.txt").read_bytes() == b"nested"
    changes = source.sync()
    assert changes == []


def test_smb_sync_upserts_new_and_changed_files(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("guide.md", b"v1", mtime=1.0)
    source = _source(tmp_path, remote)
    source.prepare()

    remote.put("guide.md", b"v2-changed", mtime=2.0)
    remote.put("new.md", b"fresh", mtime=2.0)
    changes = source.sync()
    ops = {(item.op, item.path) for item in changes}
    assert ops == {("upsert", "guide.md"), ("upsert", "new.md")}
    staging = source.local_root()
    assert (staging / "guide.md").read_bytes() == b"v2-changed"
    assert (staging / "new.md").read_bytes() == b"fresh"


def test_smb_sync_deletes_removed_files(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("keep.md", b"keep")
    remote.put("gone.md", b"gone")
    source = _source(tmp_path, remote)
    source.prepare()
    del remote.files["gone.md"]

    changes = source.sync()
    assert changes == [FsChange("delete", "gone.md")]
    assert not (source.local_root() / "gone.md").exists()
    assert (source.local_root() / "keep.md").exists()


def test_smb_rename_is_delete_plus_upsert(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("old.md", b"same-bytes")
    source = _source(tmp_path, remote)
    source.prepare()
    data, meta = remote.files.pop("old.md")
    remote.files["new.md"] = (data, meta)

    changes = source.sync()
    ops = {(item.op, item.path) for item in changes}
    assert ops == {("delete", "old.md"), ("upsert", "new.md")}
    assert not (source.local_root() / "old.md").exists()
    assert (source.local_root() / "new.md").read_bytes() == b"same-bytes"


def test_smb_skips_unstable_file_before_publish(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("growing.md", b"partial", size=7, mtime=1.0)
    remote.stat_overrides["growing.md"] = RemoteFileMeta(size=99, mtime=2.0)
    source = _source(tmp_path, remote)

    source.prepare()
    assert not (source.local_root() / "growing.md").exists()
    assert list(source.local_root().glob("*.md")) == []
    # Hidden temp files must not remain after an unstable skip.
    hidden = [path for path in source.local_root().rglob("*") if path.name.startswith(".")]
    assert all(path.name.endswith("manifest.json") or path.suffix == ".tmp" for path in hidden)
    assert not any(path.suffix == ".tmp" and path.exists() for path in hidden if "manifest" not in path.name)


def test_smb_listing_failure_does_not_delete(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("keep.md", b"keep")
    source = _source(tmp_path, remote)
    source.prepare()
    remote.list_error = SmbListingError("vpn down")

    with pytest.raises(SmbListingError):
        source.sync()
    assert (source.local_root() / "keep.md").read_bytes() == b"keep"
    manifest = json.loads(
        (source.local_root() / ".document_indexer_smb_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "keep.md" in manifest["files"]


def test_smb_backoff_increases_and_caps() -> None:
    settings = SmbSourceSettings(
        server="fileserver",
        share="docs",
        username="svc",
        password="secret",
        staging_path="/tmp/staging",
        poll_interval_sec=2.0,
        max_backoff_sec=10.0,
    )
    source = SmbStagingSource(settings, remote=FakeRemote())
    source._failures = 1
    assert source._backoff_delay() == 2.0
    source._failures = 2
    assert source._backoff_delay() == 4.0
    source._failures = 8
    assert source._backoff_delay() == 10.0


def test_smb_skips_unsupported_and_hidden_files(tmp_path: Path) -> None:
    remote = FakeRemote()
    remote.put("ok.md", b"ok")
    remote.put(".hidden.md", b"no")
    remote.put("skip.bin", b"bin")
    source = _source(tmp_path, remote)
    source.prepare()
    names = {path.name for path in source.local_root().iterdir() if path.is_file()}
    assert "ok.md" in names
    assert ".hidden.md" not in names
    assert "skip.bin" not in names


@pytest.mark.smb_integration
@pytest.mark.skipif(
    os.environ.get("DOCUMENT_INDEXER_SMB_TEST") != "1",
    reason="set DOCUMENT_INDEXER_SMB_TEST=1 against a real share",
)
def test_live_smb_share_lists_and_downloads(tmp_path: Path) -> None:
    settings = SmbSourceSettings(
        server=os.environ["SMB_SERVER"],
        share=os.environ["SMB_SHARE"],
        username=os.environ["SMB_USERNAME"],
        password=os.environ["SMB_PASSWORD"],
        staging_path=str(tmp_path / "staging"),
        domain=os.environ.get("SMB_DOMAIN"),
        subpath=os.environ.get("SMB_SUBPATH", ""),
    )
    source = SmbStagingSource(settings)
    root = source.prepare()
    assert root.is_dir()
