import hashlib
import json
import stat
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest
from asset_mania.service import InspectRequest, execute_inspect
from asset_mania_contracts import load_manifest_schema
from jsonschema import validate
from PIL import Image

FIXED_TIME = datetime(2026, 8, 19, tzinfo=UTC)


def _execute(source: Path, output_parent: Path):
    return execute_inspect(
        InspectRequest(input_path=source, output_parent=output_parent),
        clock=lambda: FIXED_TIME,
        id_factory=lambda: "abc123",
    )


def _save_png(path: Path) -> None:
    Image.new("RGBA", (8, 6), color=(20, 40, 60, 80)).save(path)


def _manifest(result) -> dict[str, object]:
    assert result.run_dir is not None
    return json.loads((result.run_dir / "manifest.json").read_text())


def _assert_valid_run(result) -> None:
    assert result.run_dir is not None
    manifest_path = result.run_dir / "manifest.json"
    report_path = result.run_dir / "report.json"
    assert manifest_path.is_file()
    assert report_path.is_file()
    assert (result.run_dir / "logs").is_dir()
    assert list((result.run_dir / "logs").iterdir()) == []
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(report_path.stat().st_mode) == 0o600
    assert (
        manifest_path.read_text()
        == json.dumps(
            json.loads(manifest_path.read_text()),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    validate(instance=json.loads(manifest_path.read_text()), schema=load_manifest_schema())


def test_image_defaults_create_a_portable_atomic_success_run(tmp_path: Path) -> None:
    source = tmp_path / "private-source.png"
    _save_png(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    result = _execute(source, tmp_path / "runs")

    assert result.exit_code == 0
    assert result.primary_diagnostic is None
    assert result.run_dir is not None
    assert result.run_dir.name == "20260819T000000Z-abc123"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert result.report["parameters"] == {"workflow": "image-to-3d", "kind": "object"}
    assert result.report["inputs"] == [
        {
            "label": "input-1",
            "sha256": before,
            "byte_size": source.stat().st_size,
            "media_type": "image/png",
        }
    ]
    assert result.report["environment"]["operating_system"] in {"macos", "linux"}
    assert result.report["environment"]["architecture"] in {"arm64", "x86_64"}
    assert result.report["environment"]["python_version"]
    assert result.report["environment"]["blender"]["status"] in {"found", "not_found"}
    assert result.report["capabilities"] == {
        "image-to-3d": "not_implemented",
        "scene-to-image": "not_implemented",
    }
    assert result.report["result"] == {
        "status": "succeeded",
        "diagnostics": ["WORKFLOW_NOT_IMPLEMENTED"],
    }
    _assert_valid_run(result)
    assert _manifest(result)["inputs"] == result.report["inputs"]
    assert source.name not in json.dumps(result.report)
    assert str(source) not in json.dumps(result.report)


def test_blend_defaults_succeed_when_blender_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asset_mania import environment

    source = tmp_path / "private-scene.blend"
    source.write_bytes(b"BLENDER-v400" + b"scene payload")
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(environment, "_MACOS_BLENDER_EXECUTABLE", tmp_path / "absent")

    result = _execute(source, tmp_path / "runs")

    assert result.exit_code == 0
    assert result.report["parameters"] == {"workflow": "scene-to-image"}
    assert result.report["environment"]["blender"] == {"status": "not_found"}
    assert result.report["warnings"] == ["BLENDER_NOT_FOUND"]
    assert result.report["result"] == {
        "status": "succeeded",
        "diagnostics": ["WORKFLOW_NOT_IMPLEMENTED"],
    }
    _assert_valid_run(result)


def test_blend_rejects_kind_before_creating_a_run(tmp_path: Path) -> None:
    source = tmp_path / "scene.blend"
    source.write_bytes(b"BLENDER-v400")
    output_parent = tmp_path / "runs"

    with pytest.raises(ValueError, match="kind"):
        execute_inspect(
            InspectRequest(input_path=source, output_parent=output_parent, kind="object"),
            clock=lambda: FIXED_TIME,
            id_factory=lambda: "abc123",
        )

    assert not output_parent.exists()


def test_omitted_output_parent_uses_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    _save_png(source)
    monkeypatch.chdir(tmp_path)

    result = execute_inspect(
        InspectRequest(input_path=source),
        clock=lambda: FIXED_TIME,
        id_factory=lambda: "abc123",
    )

    assert result.run_dir == tmp_path / ".asset-mania/runs/20260819T000000Z-abc123"
    _assert_valid_run(result)


def test_declared_face_head_is_an_advisory_not_an_approval_status(tmp_path: Path) -> None:
    source = tmp_path / "synthetic-head.png"
    _save_png(source)

    result = execute_inspect(
        InspectRequest(
            input_path=source,
            output_parent=tmp_path / "runs",
            kind="face-head",
        ),
        clock=lambda: FIXED_TIME,
        id_factory=lambda: "abc123",
    )

    assert result.exit_code == 0
    assert result.report["result"]["status"] == "succeeded"
    assert result.report["advisories"] == [
        {
            "code": "FUTURE_FACE_RIGHTS_ADVISORY",
            "message": (
                "Future external or generative face processing requires rights and consent "
                "confirmation."
            ),
        }
    ]
    assert "FACE_RIGHTS_CONFIRMATION_REQUIRED" not in json.dumps(result.report)
    _assert_valid_run(result)


def test_reports_are_deterministic_after_masking_run_identity(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _save_png(source)
    first = execute_inspect(
        InspectRequest(input_path=source, output_parent=tmp_path / "first"),
        clock=lambda: FIXED_TIME,
        id_factory=lambda: "first-id",
    )
    second = execute_inspect(
        InspectRequest(input_path=source, output_parent=tmp_path / "second"),
        clock=lambda: FIXED_TIME.replace(day=20),
        id_factory=lambda: "second-id",
    )

    first_report = deepcopy(first.report)
    second_report = deepcopy(second.report)
    for report in (first_report, second_report):
        report.pop("run_id")
        report.pop("created_at")

    assert json.dumps(first_report, sort_keys=True) == json.dumps(second_report, sort_keys=True)


def test_sensitive_values_never_reach_portable_outputs_or_logs(tmp_path: Path) -> None:
    secret_token = "token-private-123"
    secret_home = "/Users/private-person"
    source = tmp_path / "temporary-private-basename.jpeg"
    exif = Image.Exif()
    exif[315] = f"{secret_token} {secret_home}"
    Image.new("RGB", (8, 6)).save(source, exif=exif)

    result = _execute(source, tmp_path / "runs")

    assert result.run_dir is not None
    persisted = "".join(
        path.read_text(errors="replace") for path in result.run_dir.rglob("*") if path.is_file()
    )
    all_output = persisted + json.dumps(result.report)
    assert secret_token not in all_output
    assert secret_home not in all_output
    assert source.name not in all_output
    assert str(source) not in all_output
    assert result.report["warnings"] == ["EXIF_SENSITIVE_METADATA_PRESENT"]
    _assert_valid_run(result)


@pytest.mark.parametrize(
    ("name", "contents", "expected"),
    [
        ("missing.png", None, "INPUT_NOT_FOUND"),
        ("corrupt.png", b"not an image", "INPUT_UNREADABLE"),
        ("invalid.blend", b"not a blend", "BLEND_HEADER_INVALID"),
        ("unsupported.txt", b"plain text", "UNSUPPORTED_MEDIA_TYPE"),
    ],
)
def test_input_failures_create_schema_valid_failed_runs(
    tmp_path: Path, name: str, contents: bytes | None, expected: str
) -> None:
    source = tmp_path / name
    if contents is not None:
        source.write_bytes(contents)

    result = _execute(source, tmp_path / "runs")

    assert result.exit_code == 3
    assert result.primary_diagnostic == expected
    assert result.report["result"] == {"status": "failed", "diagnostics": [expected]}
    _assert_valid_run(result)


def test_unreadable_input_creates_schema_valid_failed_run(tmp_path: Path) -> None:
    source = tmp_path / "unreadable.png"
    _save_png(source)
    source.chmod(0)
    try:
        result = _execute(source, tmp_path / "runs")
    finally:
        source.chmod(0o600)

    assert result.exit_code == 3
    assert result.primary_diagnostic == "INPUT_UNREADABLE"
    _assert_valid_run(result)


def test_inaccessible_input_parent_is_an_unreadable_input_failure(tmp_path: Path) -> None:
    locked_parent = tmp_path / "locked-parent"
    locked_parent.mkdir()
    source = locked_parent / "private-source.png"
    _save_png(source)
    locked_parent.chmod(0)
    try:
        result = _execute(source, tmp_path / "runs")
    finally:
        locked_parent.chmod(stat.S_IRWXU)

    assert result.exit_code == 3
    assert result.primary_diagnostic == "INPUT_UNREADABLE"
    assert result.report["result"] == {
        "status": "failed",
        "diagnostics": ["INPUT_UNREADABLE"],
    }
    _assert_valid_run(result)


def test_internal_inspection_error_is_sanitized_into_a_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asset_mania import service

    source = tmp_path / "private-source.png"
    _save_png(source)

    def fail_without_leaking(_path: Path):
        raise RuntimeError(f"secret-token at {source}")

    monkeypatch.setattr(service, "inspect_image", fail_without_leaking)

    result = _execute(source, tmp_path / "runs")

    assert result.exit_code == 4
    assert result.primary_diagnostic == "INTERNAL_ERROR"
    assert result.report is not None
    assert result.report["result"] == {
        "status": "failed",
        "diagnostics": ["INTERNAL_ERROR"],
    }
    assert "secret-token" not in json.dumps(result.report)
    assert source.name not in json.dumps(result.report)
    _assert_valid_run(result)


def test_existing_run_path_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _save_png(source)
    existing = tmp_path / "runs/20260819T000000Z-abc123"
    existing.mkdir(parents=True)
    sentinel = existing / "owned-by-user"
    sentinel.write_text("preserve")

    result = _execute(source, tmp_path / "runs")

    assert result.exit_code == 73
    assert result.report is None
    assert result.run_dir is None
    assert result.primary_diagnostic == "OUTPUT_STORAGE_UNAVAILABLE"
    assert sentinel.read_text() == "preserve"
    assert sorted(path.name for path in existing.iterdir()) == ["owned-by-user"]


def test_atomic_publish_rejects_a_destination_created_at_the_rename_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from asset_mania import run

    output_parent = tmp_path / "runs"
    destination = output_parent / "20260819T000000Z-abc123"
    collision_inode: list[int] = []
    rename_no_replace = run._rename_no_replace

    def race_with_an_empty_destination(source: Path, final: Path) -> None:
        final.mkdir()
        collision_inode.append(final.stat().st_ino)
        rename_no_replace(source, final)

    monkeypatch.setattr(run, "_rename_no_replace", race_with_an_empty_destination)

    with pytest.raises(run.RunStorageError):
        run.persist_run(
            output_parent=output_parent,
            directory_name=destination.name,
            manifest={"kind": "manifest"},
            report={"kind": "report"},
        )

    assert destination.is_dir()
    assert destination.stat().st_ino == collision_inode[0]
    assert list(destination.iterdir()) == []
    assert sorted(path.name for path in output_parent.iterdir()) == [destination.name]


def test_unwritable_output_parent_returns_73_without_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _save_png(source)
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory")

    result = _execute(source, blocked_parent)

    assert result.exit_code == 73
    assert result.report is None
    assert result.run_dir is None
    assert result.primary_diagnostic == "OUTPUT_STORAGE_UNAVAILABLE"
    assert not list(tmp_path.rglob("manifest.json"))
