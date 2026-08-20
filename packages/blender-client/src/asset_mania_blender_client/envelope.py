"""The private, ephemeral request envelope.

The envelope is the only place absolute local paths and real datablock names appear. Its
directory is mode 0700, its files are mode 0600, it lives below the private staging root,
and it is deleted in `finally` whether the worker succeeded or not.
"""

import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from asset_mania_pipeline import PathEscape

REQUEST_NAME = "request.json"
RESPONSE_NAME = "response.json"


class PrivateEnvelope:
    """A mode-0700 directory below staging holding one request and one response path."""

    def __init__(self, staging_root: Path) -> None:
        self.staging_root = staging_root
        self.directory: Path | None = None

    def __enter__(self) -> Self:
        anchor = self.staging_root.resolve(strict=True)
        directory = Path(tempfile.mkdtemp(prefix=".envelope-", dir=anchor))
        directory.chmod(0o700)
        if not directory.resolve().is_relative_to(anchor):
            shutil.rmtree(directory, ignore_errors=True)
            raise PathEscape("the envelope resolved outside the staging root")
        self.directory = directory
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.directory is not None:
            shutil.rmtree(self.directory, ignore_errors=True)
            self.directory = None

    def _require_open(self) -> Path:
        if self.directory is None:
            raise RuntimeError("the private envelope is closed")
        return self.directory

    @property
    def request_path(self) -> Path:
        return self._require_open() / REQUEST_NAME

    @property
    def response_path(self) -> Path:
        return self._require_open() / RESPONSE_NAME

    def write_request(self, request: Mapping[str, Any]) -> Path:
        """Write the private request at mode 0600, refusing to overwrite it."""
        path = self.request_path
        payload = json.dumps(request, ensure_ascii=False, sort_keys=True).encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path
