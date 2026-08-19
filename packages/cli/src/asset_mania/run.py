"""Atomic persistence for Asset Mania run directories."""

import os
import re
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from asset_mania_contracts import canonical_json

Clock = Callable[[], datetime]
IdFactory = Callable[[], str]

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class RunStorageError(Exception):
    """A sanitized storage-boundary failure."""


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: str
    created_at: str
    directory_name: str


def create_run_identity(*, clock: Clock, id_factory: IdFactory) -> RunIdentity:
    """Create portable run identity fields from injected sources."""
    created = clock()
    if created.tzinfo is None:
        raise ValueError("clock must return a timezone-aware datetime")
    created = created.astimezone(UTC).replace(microsecond=0)
    run_id = id_factory()
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("run id must contain only portable identifier characters")
    timestamp = created.strftime("%Y%m%dT%H%M%SZ")
    created_at = created.isoformat().replace("+00:00", "Z")
    return RunIdentity(
        run_id=run_id,
        created_at=created_at,
        directory_name=f"{timestamp}-{run_id}",
    )


def persist_run(
    *,
    output_parent: Path,
    directory_name: str,
    manifest: dict[str, object],
    report: dict[str, object],
) -> Path:
    """Publish a complete run directory atomically without overwriting another run."""
    temporary: Path | None = None
    try:
        output_parent.mkdir(parents=True, exist_ok=True)
        if not output_parent.is_dir():
            raise OSError("output parent is not a directory")

        final = output_parent / directory_name
        if final.exists():
            raise FileExistsError("run already exists")

        temporary = Path(tempfile.mkdtemp(prefix=f".{directory_name}.tmp-", dir=output_parent))
        temporary.chmod(0o700)
        (temporary / "logs").mkdir(mode=0o700)
        _write_canonical_json(temporary / "manifest.json", manifest)
        _write_canonical_json(temporary / "report.json", report)

        if final.exists():
            raise FileExistsError("run already exists")
        os.rename(temporary, final)
        temporary = None
        return final
    except (OSError, ValueError) as error:
        raise RunStorageError from error
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def _write_canonical_json(path: Path, value: object) -> None:
    payload = canonical_json(value).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
