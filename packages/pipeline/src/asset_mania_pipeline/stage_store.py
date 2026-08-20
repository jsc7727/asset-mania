"""Immutable stage storage.

Each stage stages its outputs in a private temporary tree and then publishes them with a
single atomic no-replace rename. A stage never mutates a parent run, never overwrites an
existing run directory, and reaches exactly one terminal state.
"""

import ctypes
import errno
import os
import re
import shutil
import sys
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

from asset_mania_contracts import canonical_json

from .artifacts import PathEscape, contained_path

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_RENAME_EXCL = 0x00000004


class StageStoreError(Exception):
    """A sanitized stage-storage failure."""


class StorageUnavailable(StageStoreError):
    """Storage could not be prepared, so no manifest is promised."""


class OutputCollision(StageStoreError):
    """The destination run directory already exists and is never replaced."""


class StageState(Enum):
    OPEN = "open"
    PUBLISHED = "published"
    FAILED = "failed"


class StageRun:
    """One open stage with a private staging tree below the output parent."""

    def __init__(self, *, stage: str, run_id: str, directory_name: str, staging: Path) -> None:
        self.stage = stage
        self.run_id = run_id
        self.directory_name = directory_name
        self.staging = staging
        self.state = StageState.OPEN

    def stage_path(self, relative: str) -> Path:
        """A writable path inside this run, with its parent directories created."""
        target = contained_path(self.staging, relative)
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as error:
            raise PathEscape("the staged directory could not be created") from error
        return target


class StageStore:
    """Publishes stage runs below one output parent."""

    def __init__(self, output_parent: Path) -> None:
        self.output_parent = output_parent

    def begin(self, *, stage: str, run_id: str, created_at: str) -> StageRun:
        """Open a new run: validate identity, then prepare a private staging tree."""
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise ValueError("run id must contain only portable identifier characters")
        if not _UTC_TIMESTAMP.fullmatch(created_at):
            raise ValueError("created_at must be an RFC 3339 UTC timestamp ending in Z")

        # The pattern above already pins an RFC 3339 UTC instant, so the compact
        # directory stamp is that same string with its punctuation removed.
        stamp = created_at.replace("-", "").replace(":", "")
        directory_name = f"{stamp}-{run_id}"

        try:
            self.output_parent.mkdir(parents=True, exist_ok=True)
            if not self.output_parent.is_dir():
                raise OSError(errno.ENOTDIR, "output parent is not a directory")
            staging = Path(
                tempfile.mkdtemp(prefix=f".{directory_name}.tmp-", dir=self.output_parent)
            )
            staging.chmod(0o700)
            (staging / "logs").mkdir(mode=0o700)
        except OSError as error:
            raise StorageUnavailable(
                "OUTPUT_STORAGE_UNAVAILABLE: the run directory could not be prepared"
            ) from error

        return StageRun(stage=stage, run_id=run_id, directory_name=directory_name, staging=staging)

    def publish(self, run: StageRun, *, manifest: dict[str, Any], report: dict[str, Any]) -> Path:
        """Publish a succeeded run atomically."""
        return self._finalize(run, manifest=manifest, report=report, state=StageState.PUBLISHED)

    def fail(self, run: StageRun, *, manifest: dict[str, Any], report: dict[str, Any]) -> Path:
        """Publish a terminal failed run rather than discarding the evidence."""
        return self._finalize(run, manifest=manifest, report=report, state=StageState.FAILED)

    def _finalize(
        self,
        run: StageRun,
        *,
        manifest: dict[str, Any],
        report: dict[str, Any],
        state: StageState,
    ) -> Path:
        if run.state is not StageState.OPEN:
            raise ValueError(f"run {run.run_id!r} already reached the terminal state {run.state}")

        final = self.output_parent / run.directory_name
        try:
            _write_canonical_json(run.staging / "manifest.json", manifest)
            _write_canonical_json(run.staging / "report.json", report)
            _rename_no_replace(run.staging, final)
        except FileExistsError as error:
            shutil.rmtree(run.staging, ignore_errors=True)
            raise OutputCollision(
                "OUTPUT_COLLISION: the destination run directory already exists"
            ) from error
        except OSError as error:
            shutil.rmtree(run.staging, ignore_errors=True)
            raise StorageUnavailable(
                "OUTPUT_STORAGE_UNAVAILABLE: the run could not be published"
            ) from error

        run.state = state
        return final


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a directory only when the destination does not exist."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)

    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, _RENAME_EXCL)
    elif sys.platform == "linux":
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(errno.ENOSYS, "atomic no-replace rename is unavailable") from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(_AT_FDCWD, source_bytes, _AT_FDCWD, destination_bytes, _RENAME_NOREPLACE)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")

    if result != 0:
        number = ctypes.get_errno()
        if number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(number, os.strerror(number), os.fspath(destination))
        raise OSError(number, os.strerror(number), os.fspath(destination))


def _write_canonical_json(path: Path, value: object) -> None:
    payload = canonical_json(value).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
