import json
from pathlib import Path

import pytest
from asset_mania.environment import inspect_environment
from asset_mania.inspectors.blend import inspect_blend
from asset_mania_contracts import DiagnosticCode


@pytest.mark.parametrize(
    ("header", "version", "pointer_size", "endianness"),
    [
        (b"BLENDER-v280", 280, 64, "little"),
        (b"BLENDER_V400", 400, 32, "big"),
    ],
)
def test_inspect_blend_reads_only_a_valid_header(
    tmp_path: Path, header: bytes, version: int, pointer_size: int, endianness: str
) -> None:
    source = tmp_path / "private-scene.blend"
    source.write_bytes(header + b"uninspected scene bytes")

    report, diagnostics = inspect_blend(source)

    assert report == {
        "version": version,
        "pointer_size": pointer_size,
        "endianness": endianness,
        "header_valid": True,
    }
    assert diagnostics == []
    assert source.name not in json.dumps(report)


@pytest.mark.parametrize(
    "header",
    [
        b"INVALID-v280",
        b"BLENDER?v280",
        b"BLENDER-x280",
        b"BLENDER-v2x0",
        b"BLENDER-v099",
        b"BLENDER-v280"[:11],
    ],
)
def test_inspect_blend_sanitizes_invalid_headers(tmp_path: Path, header: bytes) -> None:
    source = tmp_path / "private-invalid-scene.blend"
    source.write_bytes(header)

    report, diagnostics = inspect_blend(source)

    assert report == {"header_valid": False}
    assert diagnostics == [DiagnosticCode.BLEND_HEADER_INVALID]
    assert source.name not in json.dumps(report)


def test_inspect_environment_finds_configured_executable_without_invoking_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "private-configured-blender"
    invoked_marker = tmp_path / "invoked"
    executable.write_text(f"#!/bin/sh\ntouch {invoked_marker}\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", "")

    report, diagnostics = inspect_environment(configured_blender=executable)

    assert report["blender"] == {"status": "found"}
    assert diagnostics == []
    assert not invoked_marker.exists()
    assert executable.name not in json.dumps(report)


def test_inspect_environment_searches_controlled_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "blender"
    executable.write_text("#!/bin/sh\nexit 99\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))

    report, diagnostics = inspect_environment()

    assert report["blender"] == {"status": "found"}
    assert diagnostics == []


def test_inspect_environment_sanitizes_missing_blender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asset_mania import environment

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(environment, "_MACOS_BLENDER_EXECUTABLE", tmp_path / "missing")

    report, diagnostics = inspect_environment()

    assert report["blender"] == {"status": "not_found"}
    assert diagnostics == [DiagnosticCode.BLENDER_NOT_FOUND]
    assert "missing" not in json.dumps(report)


def test_inspect_environment_treats_inaccessible_blender_candidate_as_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asset_mania import environment

    inaccessible = tmp_path / "private-inaccessible-blender"
    original_is_file = Path.is_file

    def deny_candidate(path: Path) -> bool:
        if path == inaccessible:
            raise PermissionError(str(path))
        return original_is_file(path)

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(environment, "_MACOS_BLENDER_EXECUTABLE", inaccessible)
    monkeypatch.setattr(Path, "is_file", deny_candidate)

    report, diagnostics = inspect_environment()

    assert report["blender"] == {"status": "not_found"}
    assert diagnostics == [DiagnosticCode.BLENDER_NOT_FOUND]
    assert inaccessible.name not in json.dumps(report)
