"""Only the exact pinned Blender is accepted."""

import subprocess
from pathlib import Path

import pytest
from asset_mania_blender_client import (
    PROFILE_ID,
    REQUIRED_VERSION,
    BlenderNotFound,
    BlenderVersionMismatch,
    discover_blender,
    fingerprint_executable,
)
from asset_mania_pipeline import sha256_file


def test_the_pinned_version_is_accepted_and_fingerprinted(version_reporting_blender: Path) -> None:
    fingerprint = fingerprint_executable(version_reporting_blender)
    assert fingerprint.version == REQUIRED_VERSION
    assert fingerprint.build_hash == "fbe6228777e7"
    assert fingerprint.executable_sha256 == sha256_file(version_reporting_blender)
    assert fingerprint.profile == PROFILE_ID


def test_the_manifest_record_carries_no_local_path(fake_blender) -> None:
    """The basename is distinctive so a leak cannot hide inside the profile identifier."""
    executable = fake_blender(
        "import sys\nsys.stdout.write('Blender 5.2.0 LTS\\n\\tbuild hash: fbe6228777e7\\n')\n",
        name="private-install-name",
    )
    record = fingerprint_executable(executable).to_manifest_record()
    assert set(record) == {"profile", "version", "build_hash", "executable_sha256"}
    assert str(executable) not in str(record)
    assert executable.name not in str(record)


@pytest.mark.parametrize("version", ["5.1.0", "5.2.1", "4.2.0", "5.20.0", "6.0.0"])
def test_another_version_fails_rather_than_running(fake_blender, version: str) -> None:
    executable = fake_blender(
        f"import sys\nsys.stdout.write('Blender {version} LTS\\n\\tbuild hash: aaaaaaaaaaaa\\n')\n"
    )
    with pytest.raises(BlenderVersionMismatch, match="BLENDER_VERSION_MISMATCH"):
        fingerprint_executable(executable)


def test_a_candidate_that_is_not_blender_fails(fake_blender) -> None:
    executable = fake_blender("import sys\nsys.stdout.write('GNU bash, version 5.2\\n')\n")
    with pytest.raises(BlenderVersionMismatch, match="BLENDER_VERSION_MISMATCH"):
        fingerprint_executable(executable)


def test_a_build_without_a_recordable_hash_fails(fake_blender) -> None:
    executable = fake_blender("import sys\nsys.stdout.write('Blender 5.2.0 LTS\\n')\n")
    with pytest.raises(BlenderVersionMismatch, match="build hash"):
        fingerprint_executable(executable)


def test_an_absent_executable_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
        fingerprint_executable(tmp_path / "absent" / "blender")


def test_an_inaccessible_executable_is_reported_as_not_found(tmp_path: Path) -> None:
    executable = tmp_path / "blender"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o000)
    try:
        with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
            fingerprint_executable(executable)
    finally:
        executable.chmod(0o600)


def test_a_candidate_that_exits_nonzero_is_reported_as_not_found(fake_blender) -> None:
    executable = fake_blender("raise SystemExit(1)\n")
    with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
        fingerprint_executable(executable)


def test_a_hanging_candidate_does_not_block_forever(fake_blender) -> None:
    def timing_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 30))

    executable = fake_blender("import time\ntime.sleep(600)\n")
    with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
        fingerprint_executable(executable, runner=timing_out)


def test_discovery_walks_candidates_until_one_is_accessible(
    tmp_path: Path, version_reporting_blender: Path
) -> None:
    fingerprint = discover_blender(
        [tmp_path / "absent-one", tmp_path / "absent-two", version_reporting_blender]
    )
    assert fingerprint.version == REQUIRED_VERSION


def test_a_present_but_wrong_version_is_not_skipped_for_a_later_candidate(
    fake_blender, version_reporting_blender: Path
) -> None:
    """A stale install must be reported, never hidden behind a correct later candidate."""
    stale = fake_blender(
        "import sys\nsys.stdout.write('Blender 5.1.0 LTS\\n\\tbuild hash: bbbbbbbbbbbb\\n')\n",
        name="stale-blender",
    )
    with pytest.raises(BlenderVersionMismatch):
        discover_blender([stale, version_reporting_blender])


def test_no_candidate_at_all_is_reported_as_not_found(tmp_path: Path) -> None:
    with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
        discover_blender([tmp_path / "absent-one", tmp_path / "absent-two"])


def test_an_empty_candidate_list_is_reported_as_not_found() -> None:
    with pytest.raises(BlenderNotFound, match="BLENDER_NOT_FOUND"):
        discover_blender([])
