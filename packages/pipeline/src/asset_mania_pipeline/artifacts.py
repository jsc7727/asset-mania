"""Artifact records, staging-path containment, and content-origin inheritance."""

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from asset_mania_contracts import CONTENT_ORIGINS

from .hashing import sha256_file


class PathEscape(Exception):
    """A relative path would read or write outside its run directory."""


# `unknown` outranks `derived` so an undeclared upstream origin is never quietly
# downgraded to a confident claim, while `generated` still dominates everything.
_ORIGIN_RANK: dict[str, int] = {
    "observed": 0,
    "derived": 1,
    "unknown": 2,
    "generated": 3,
}


def contained_path(root: Path, relative: str | Path) -> Path:
    """Resolve `relative` under `root`, rejecting every way out of the run.

    Absolute paths, drive prefixes, backslashes, empty segments, `.`/`..` segments, a
    trailing separator, and any symlink component that leaves `root` all fail.
    """
    text = str(relative)
    if not text or text.startswith("/") or "\\" in text or text.endswith("/"):
        raise PathEscape(f"{text!r} is not a contained relative path")

    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise PathEscape(f"{text!r} is not a contained relative path")

    try:
        anchor = root.resolve(strict=True)
    except OSError as error:
        raise PathEscape("the containment root does not exist") from error

    target = anchor
    for part in parts:
        target = target / part
        if target.is_symlink():
            target = target.resolve()
            if not target.is_relative_to(anchor):
                raise PathEscape(f"{text!r} leaves the run through a symlink")

    if not target.is_relative_to(anchor):
        raise PathEscape(f"{text!r} is not a contained relative path")
    return target


def relative_staging_path(staging_root: Path, path: Path) -> str:
    """The portable, run-relative POSIX path of one staged file."""
    anchor = staging_root.resolve(strict=True)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PathEscape("the staged artifact does not exist") from error
    if not resolved.is_relative_to(anchor):
        raise PathEscape("the staged artifact lies outside the run")
    return resolved.relative_to(anchor).as_posix()


def inherit_content_origin(operation_origin: str, parent_origins: Iterable[str]) -> str:
    """Combine the operation's own origin with every consumed parent's origin.

    Derived processing never erases upstream provenance: reprojecting a generated image
    produces a generated texture.
    """
    candidates = [operation_origin, *parent_origins]
    unknown = [origin for origin in candidates if origin not in CONTENT_ORIGINS]
    if unknown:
        raise ValueError(f"undeclared content origins: {sorted(set(unknown))}")
    return max(candidates, key=lambda origin: _ORIGIN_RANK[origin])


def describe_artifact(
    *,
    role: str,
    staging_root: Path,
    path: Path,
    media_type: str,
    parents: Sequence[Mapping[str, Any]],
    operation: str,
    content_origin: str,
    sensitivity: str,
    upload_eligible: bool,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe one staged artifact by hashing it, never by trusting a caller's digest."""
    relative = relative_staging_path(staging_root, path)
    return {
        "role": role,
        "path": relative,
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
        "media_type": media_type,
        "parents": sorted(
            (dict(parent) for parent in parents), key=lambda parent: parent["sha256"]
        ),
        "operation": operation,
        "content_origin": content_origin,
        "sensitivity": sensitivity,
        "upload_eligible": upload_eligible,
        "validation": dict(validation),
    }
