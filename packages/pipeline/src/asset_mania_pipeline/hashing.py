"""Source digests and the source-integrity guard.

A source `.blend` or image is opened read-only and must stay byte-identical across a
run. The fingerprint therefore records content identity *and* local file identity, so a
same-size replacement between two reads is caught as well as an edit in place.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

_CHUNK_BYTES = 1024 * 1024


class SourceChanged(Exception):
    """The source file changed while the run was reading it."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SourceFingerprint:
    """Content and local identity of one source read. It records no path or basename."""

    sha256: str
    byte_size: int
    device: int
    inode: int
    mtime_ns: int


def fingerprint_source(path: Path) -> SourceFingerprint:
    status = path.stat()
    return SourceFingerprint(
        sha256=sha256_file(path),
        byte_size=status.st_size,
        device=status.st_dev,
        inode=status.st_ino,
        mtime_ns=status.st_mtime_ns,
    )


def verify_source_unchanged(path: Path, before: SourceFingerprint) -> None:
    """Fail with `SOURCE_CHANGED_DURING_RUN` unless the source is the same bytes and file."""
    try:
        after = fingerprint_source(path)
    except OSError as error:
        raise SourceChanged(
            "SOURCE_CHANGED_DURING_RUN: the source became unreadable during the run"
        ) from error

    if after != before:
        raise SourceChanged("SOURCE_CHANGED_DURING_RUN: the source changed during the run")
