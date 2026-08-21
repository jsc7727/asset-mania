"""Blender discovery and the exact profile check.

Only Blender 5.2.0 is accepted. Every run records the full version, build hash, platform,
and executable fingerprint, so a different build fails preflight instead of silently
producing artifacts under an unverified profile.
"""

import re
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asset_mania_pipeline import sha256_file

PROFILE_ID = "blender-5.2.0-cpu-v1"
FIXTURE_PROFILE_ID = "blender-5.2.0-cpu-v1-fixture"
REQUIRED_VERSION = "5.2.0"
_VERSION_TIMEOUT_SECONDS = 30
_VERSION = re.compile(r"^Blender\s+(?P<version>\d+\.\d+\.\d+)")
_BUILD_HASH = re.compile(r"^\s*build hash:\s*(?P<hash>[0-9A-Za-z]+)\s*$", re.MULTILINE)

Runner = Callable[..., subprocess.CompletedProcess[str]]


class BlenderNotFound(Exception):
    """No candidate executable could be probed."""


class BlenderVersionMismatch(Exception):
    """A Blender was found, but not the exact pinned version."""


@dataclass(frozen=True, slots=True)
class BlenderFingerprint:
    """The recorded identity of one Blender executable."""

    executable: Path
    version: str
    build_hash: str
    executable_sha256: str
    profile: str = PROFILE_ID

    def to_manifest_record(self) -> dict[str, Any]:
        """The portable `environment.blender` record; it carries no local path."""
        return {
            "profile": self.profile,
            "version": self.version,
            "build_hash": self.build_hash,
            "executable_sha256": self.executable_sha256,
        }


def _probe(executable: Path, runner: Runner) -> str:
    completed = runner(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=_VERSION_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        raise BlenderNotFound(
            "BLENDER_NOT_FOUND: the candidate executable did not report a version"
        )
    return completed.stdout


def fingerprint_executable(
    executable: Path, *, runner: Runner = subprocess.run
) -> BlenderFingerprint:
    """Probe one candidate and require the exact pinned version."""
    try:
        output = _probe(executable, runner)
    except (OSError, subprocess.SubprocessError) as error:
        raise BlenderNotFound(
            "BLENDER_NOT_FOUND: the candidate executable is missing or inaccessible"
        ) from error

    version_match = _VERSION.search(output)
    if version_match is None:
        raise BlenderVersionMismatch(
            "BLENDER_VERSION_MISMATCH: the candidate did not identify itself as Blender"
        )

    version = version_match.group("version")
    if version != REQUIRED_VERSION:
        raise BlenderVersionMismatch(
            f"BLENDER_VERSION_MISMATCH: this profile requires Blender {REQUIRED_VERSION}, "
            f"found {version}"
        )

    build_hash_match = _BUILD_HASH.search(output)
    if build_hash_match is None:
        raise BlenderVersionMismatch(
            "BLENDER_VERSION_MISMATCH: the candidate reported no build hash to record"
        )

    try:
        digest = sha256_file(executable)
    except OSError as error:
        raise BlenderNotFound(
            "BLENDER_NOT_FOUND: the candidate executable could not be fingerprinted"
        ) from error

    return BlenderFingerprint(
        executable=executable,
        version=version,
        build_hash=build_hash_match.group("hash"),
        executable_sha256=digest,
    )


def discover_blender(
    candidates: Iterable[Path] | Sequence[Path], *, runner: Runner = subprocess.run
) -> BlenderFingerprint:
    """Return the first candidate that is exactly the pinned Blender.

    A candidate that exists but is the wrong version is reported as a version mismatch
    rather than skipped, so a stale install never hides behind a later candidate.
    """
    inaccessible = 0
    for candidate in candidates:
        try:
            return fingerprint_executable(candidate, runner=runner)
        except BlenderNotFound:
            inaccessible += 1
            continue

    raise BlenderNotFound(
        f"BLENDER_NOT_FOUND: no accessible Blender candidate ({inaccessible} probed)"
    )
