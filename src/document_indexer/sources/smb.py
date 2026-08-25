"""Native SMB source: poll a share and mirror it into a local staging tree."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from document_indexer.config import SmbSourceSettings
from document_indexer.domain.changes import FsChange
from document_indexer.domain.formats import SUPPORTED_SUFFIXES
from document_indexer.sources.base import ChangeCallback

logger = logging.getLogger(__name__)

_MANIFEST_NAME = ".document_indexer_smb_manifest.json"
_HASH_CHUNK = 1024 * 1024


class SmbListingError(RuntimeError):
    """Raised when a remote listing is incomplete or the share is unreachable."""


@dataclass(frozen=True, slots=True)
class RemoteFileMeta:
    """Identity of one remote file used to detect changes without hashing."""

    size: int
    mtime: float


@runtime_checkable
class SmbRemote(Protocol):
    """Minimal SMB operations the staging source needs."""

    def list_files(self) -> dict[str, RemoteFileMeta]:
        """Return a complete relative-path listing, or raise ``SmbListingError``."""
        ...

    def stat(self, relative: str) -> RemoteFileMeta:
        """Return current size/mtime for ``relative``."""
        ...

    def download(self, relative: str, dest: Path) -> None:
        """Write the remote file bytes into ``dest``."""
        ...


class SmbprotocolRemote:
    """``smbclient`` backend talking to a real SMB share."""

    def __init__(self, settings: SmbSourceSettings) -> None:
        self._settings = settings
        self._root = _unc(
            settings.server,
            settings.share,
            settings.subpath,
        )
        self._registered = False

    def _ensure_session(self) -> None:
        try:
            import smbclient
        except ImportError as exc:
            raise RuntimeError(
                "SMB source requires smbprotocol; install with: "
                "pip install 'document-indexer[smb]'"
            ) from exc
        if self._registered:
            return
        username = self._settings.username
        if self._settings.domain:
            username = f"{self._settings.domain}\\{username}"
        smbclient.register_session(
            self._settings.server,
            username=username,
            password=self._settings.password.get_secret_value(),
            port=self._settings.port,
            connection_timeout=self._settings.timeout_sec,
        )
        self._registered = True

    def list_files(self) -> dict[str, RemoteFileMeta]:
        self._ensure_session()
        import smbclient

        listed: dict[str, RemoteFileMeta] = {}
        try:
            for dirpath, _dirnames, filenames in smbclient.walk(self._root):
                for name in filenames:
                    if name.startswith("."):
                        continue
                    full = _join_unc(dirpath, name)
                    relative = _relative_posix(full, self._root)
                    if relative is None:
                        continue
                    stat_result = smbclient.stat(full)
                    listed[relative] = RemoteFileMeta(
                        size=int(stat_result.st_size),
                        mtime=float(stat_result.st_mtime),
                    )
        except SmbListingError:
            raise
        except Exception as exc:
            raise SmbListingError(f"SMB listing failed for {self._root}: {exc}") from exc
        return listed

    def stat(self, relative: str) -> RemoteFileMeta:
        self._ensure_session()
        import smbclient

        full = _join_unc(self._root, relative.replace("/", "\\"))
        try:
            stat_result = smbclient.stat(full)
        except Exception as exc:
            raise SmbListingError(f"SMB stat failed for {relative}: {exc}") from exc
        return RemoteFileMeta(size=int(stat_result.st_size), mtime=float(stat_result.st_mtime))

    def download(self, relative: str, dest: Path) -> None:
        self._ensure_session()
        import smbclient

        full = _join_unc(self._root, relative.replace("/", "\\"))
        with smbclient.open_file(full, mode="rb") as remote, dest.open("wb") as local:
            shutil.copyfileobj(remote, local, length=_HASH_CHUNK)


class SmbStagingSource:
    """Mirror an SMB share into ``staging_path`` and emit ``FsChange`` batches."""

    def __init__(
        self,
        settings: SmbSourceSettings,
        *,
        remote: SmbRemote | None = None,
        allowed_extensions: Iterable[str] | None = None,
    ) -> None:
        self._settings = settings
        self._remote = remote or SmbprotocolRemote(settings)
        self._allowed = frozenset(
            ext if ext.startswith(".") else f".{ext}"
            for ext in (
                item.strip().lower()
                for item in (allowed_extensions or SUPPORTED_SUFFIXES)
            )
            if ext
        )
        self._mirrored: dict[str, RemoteFileMeta] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failures = 0

    def local_root(self) -> Path:
        return Path(self._settings.staging_path)

    def prepare(self) -> Path:
        root = self.local_root()
        root.mkdir(parents=True, exist_ok=True)
        self._load_manifest()
        self.sync()
        return root

    def start(self, on_changes: ChangeCallback) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            args=(on_changes,),
            name="document-indexer-smb",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "SMB poller started server=%s share=%s subpath=%s staging=%s interval=%ss",
            self._settings.server,
            self._settings.share,
            self._settings.subpath or "/",
            self._settings.staging_path,
            self._settings.poll_interval_sec,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._settings.timeout_sec + 5)
            self._thread = None
        logger.info("SMB poller stopped")

    def sync(self) -> list[FsChange]:
        """Download a stable snapshot. Deletions run only after a complete listing."""
        remote_files = self._remote.list_files()
        changes: list[FsChange] = []
        staging = self.local_root()

        for relative, meta in remote_files.items():
            if not self._is_indexable(relative):
                continue
            current = self._mirrored.get(relative)
            dest = staging / relative
            if current == meta and dest.is_file():
                continue
            if not self._publish_stable(relative, meta, dest):
                continue
            self._mirrored[relative] = meta
            changes.append(FsChange("upsert", relative))

        for relative in list(self._mirrored):
            if relative in remote_files:
                continue
            dest = staging / relative
            if dest.is_file():
                dest.unlink()
            self._mirrored.pop(relative, None)
            changes.append(FsChange("delete", relative))
            self._cleanup_empty_parents(dest.parent, staging)

        self._save_manifest()
        if changes:
            logger.info(
                "SMB sync changes=%s staging=%s",
                [(item.op, item.path) for item in changes],
                staging,
            )
        return changes

    def _loop(self, on_changes: ChangeCallback) -> None:
        while not self._stop.is_set():
            try:
                changes = self.sync()
                self._failures = 0
                if changes:
                    on_changes(changes)
            except Exception:
                self._failures += 1
                delay = self._backoff_delay()
                logger.exception(
                    "SMB poll failed; keeping staging and index unchanged; backoff=%.1fs",
                    delay,
                )
                if self._stop.wait(timeout=delay):
                    return
                continue
            if self._stop.wait(timeout=self._settings.poll_interval_sec):
                return

    def _backoff_delay(self) -> float:
        exponent = max(0, self._failures - 1)
        delay = self._settings.poll_interval_sec * (2 ** exponent)
        return min(self._settings.max_backoff_sec, delay)

    def _publish_stable(self, relative: str, meta: RemoteFileMeta, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.parent / f".{dest.name}.{os.getpid()}.tmp"
        try:
            self._remote.download(relative, tmp)
            fresh = self._remote.stat(relative)
            if fresh.size != meta.size or fresh.mtime != meta.mtime:
                logger.info(
                    "Skip unstable SMB file path=%s size %s->%s mtime %s->%s",
                    relative,
                    meta.size,
                    fresh.size,
                    meta.mtime,
                    fresh.mtime,
                )
                tmp.unlink(missing_ok=True)
                return False
            os.replace(tmp, dest)
            return True
        except Exception:
            tmp.unlink(missing_ok=True)
            logger.exception("SMB download failed path=%s; file not published", relative)
            return False

    def _is_indexable(self, relative: str) -> bool:
        path = Path(relative)
        if path.name.startswith("."):
            return False
        return path.suffix.lower() in self._allowed

    def _manifest_path(self) -> Path:
        return self.local_root() / _MANIFEST_NAME

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not path.is_file():
            self._mirrored = {}
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Cannot read SMB manifest %s; starting empty", path)
            self._mirrored = {}
            return
        files = raw.get("files") if isinstance(raw, dict) else None
        mirrored: dict[str, RemoteFileMeta] = {}
        if isinstance(files, dict):
            for relative, meta in files.items():
                if not isinstance(meta, dict):
                    continue
                try:
                    mirrored[str(relative)] = RemoteFileMeta(
                        size=int(meta["size"]),
                        mtime=float(meta["mtime"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        self._mirrored = mirrored

    def _save_manifest(self) -> None:
        payload = {
            "files": {
                relative: {"size": meta.size, "mtime": meta.mtime}
                for relative, meta in self._mirrored.items()
            }
        }
        path = self._manifest_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)

    def _cleanup_empty_parents(self, directory: Path, root: Path) -> None:
        current = directory
        while current != root and root in current.parents:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent


def _unc(server: str, share: str, *parts: str) -> str:
    chunks = [server.strip("\\"), share.strip("\\/")]
    for part in parts:
        cleaned = part.replace("/", "\\").strip("\\")
        if cleaned:
            chunks.append(cleaned)
    return "\\\\" + "\\".join(chunks)


def _join_unc(left: str, right: str) -> str:
    return left.rstrip("\\") + "\\" + right.replace("/", "\\").lstrip("\\")


def _relative_posix(full: str, root: str) -> str | None:
    full_n = full.replace("/", "\\").rstrip("\\")
    root_n = root.replace("/", "\\").rstrip("\\")
    if len(full_n) < len(root_n):
        return None
    # Compare case-insensitively (Windows share) but keep listed casing.
    if full_n[: len(root_n)].casefold() != root_n.casefold():
        return None
    rest = full_n[len(root_n) :].lstrip("\\")
    if not rest:
        return None
    return rest.replace("\\", "/")
