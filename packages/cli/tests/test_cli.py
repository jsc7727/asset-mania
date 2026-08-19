import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from asset_mania.cli import main
from asset_mania_contracts import load_manifest_schema
from jsonschema import validate
from PIL import Image


def _command() -> str:
    return str(Path(sys.executable).with_name("asset-mania"))


def _run(
    cwd: Path,
    *arguments: str,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PATH"] = ""
    environment.update(extra_environment or {})
    return subprocess.run(
        [_command(), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _save_png(path: Path) -> None:
    Image.new("RGBA", (8, 6), color=(20, 40, 60, 80)).save(path)


def _only_run(output_parent: Path) -> Path:
    runs = [path for path in output_parent.iterdir() if path.is_dir()]
    assert len(runs) == 1
    return runs[0]


def _assert_canonical_json(path: Path) -> dict[str, object]:
    payload = path.read_text()
    value = json.loads(payload)
    assert (
        payload
        == json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    return value


def test_main_help_returns_zero_without_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--help"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("usage: asset-mania")
    assert captured.err == ""


def test_default_format_emits_canonical_report_json_on_stdout(tmp_path: Path) -> None:
    source = tmp_path / "temporary-source-name.png"
    _save_png(source)

    completed = _run(tmp_path, "inspect", str(source), "--out", "runs")

    assert completed.returncode == 0
    assert completed.stderr == ""
    stdout_report = json.loads(completed.stdout)
    assert (
        completed.stdout
        == json.dumps(
            stdout_report,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    run_dir = _only_run(tmp_path / "runs")
    assert completed.stdout == (run_dir / "report.json").read_text()
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    validate(instance=manifest, schema=load_manifest_schema())
    assert source.name not in completed.stdout
    assert str(source) not in completed.stdout


def test_text_format_changes_only_stdout_and_keeps_json_run_files(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    _save_png(source)

    completed = _run(
        tmp_path,
        "inspect",
        str(source),
        "--out",
        "runs",
        "--format",
        "text",
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.startswith("Asset Mania inspection: succeeded\n")
    with pytest.raises(json.JSONDecodeError):
        json.loads(completed.stdout)
    run_dir = _only_run(tmp_path / "runs")
    report = _assert_canonical_json(run_dir / "report.json")
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    assert report["result"]["status"] == "succeeded"
    validate(instance=manifest, schema=load_manifest_schema())


@pytest.mark.parametrize(
    ("source_kind", "extra_arguments"),
    [
        ("png", ("--format", "token-private-123")),
        ("png", ("--workflow", "scene-to-image")),
        ("blend", ("--workflow", "image-to-3d")),
        ("blend", ("--kind", "object")),
        ("png", ("--workflow", "scene-to-image", "--kind", "object")),
    ],
)
def test_invalid_usage_exits_2_without_creating_a_run(
    tmp_path: Path, source_kind: str, extra_arguments: tuple[str, ...]
) -> None:
    if source_kind == "png":
        source = tmp_path / "private-input.png"
        _save_png(source)
    else:
        source = tmp_path / "private-input.blend"
        source.write_bytes(b"BLENDER-v400")

    completed = _run(
        tmp_path,
        "inspect",
        str(source),
        "--out",
        "runs",
        *extra_arguments,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "error:" in completed.stderr
    assert source.name not in completed.stderr
    assert str(source) not in completed.stderr
    assert "token-private-123" not in completed.stderr
    assert not (tmp_path / "runs").exists()


def test_missing_required_arguments_use_argparse_stderr_only(tmp_path: Path) -> None:
    completed = _run(tmp_path, "inspect")

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "usage:" in completed.stderr
    assert "error:" in completed.stderr


@pytest.mark.parametrize(
    ("name", "contents", "expected"),
    [
        ("missing.png", None, "INPUT_NOT_FOUND"),
        ("corrupt.png", b"not an image", "INPUT_UNREADABLE"),
        ("invalid.blend", b"not a blend", "BLEND_HEADER_INVALID"),
        ("unsupported.txt", b"plain text", "UNSUPPORTED_MEDIA_TYPE"),
    ],
)
def test_completed_input_failures_emit_report_stdout_and_diagnostic_stderr(
    tmp_path: Path, name: str, contents: bytes | None, expected: str
) -> None:
    source = tmp_path / name
    if contents is not None:
        source.write_bytes(contents)

    completed = _run(tmp_path, "inspect", str(source), "--out", "runs")

    assert completed.returncode == 3
    assert completed.stderr == f"{expected}\n"
    report = json.loads(completed.stdout)
    assert report["result"] == {"status": "failed", "diagnostics": [expected]}
    run_dir = _only_run(tmp_path / "runs")
    assert completed.stdout == (run_dir / "report.json").read_text()
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    validate(instance=manifest, schema=load_manifest_schema())
    all_output = completed.stdout + completed.stderr
    assert source.name not in all_output
    assert str(source) not in all_output


def test_unreadable_input_uses_completed_failure_stream_contract(tmp_path: Path) -> None:
    source = tmp_path / "unreadable.png"
    _save_png(source)
    source.chmod(0)
    try:
        completed = _run(tmp_path, "inspect", str(source), "--out", "runs")
    finally:
        source.chmod(stat.S_IRUSR | stat.S_IWUSR)

    assert completed.returncode == 3
    assert completed.stderr == "INPUT_UNREADABLE\n"
    assert json.loads(completed.stdout)["result"] == {
        "status": "failed",
        "diagnostics": ["INPUT_UNREADABLE"],
    }
    manifest = _assert_canonical_json(_only_run(tmp_path / "runs") / "manifest.json")
    validate(instance=manifest, schema=load_manifest_schema())


def test_internal_failure_uses_completed_failure_stream_contract(tmp_path: Path) -> None:
    source = tmp_path / "private-source.png"
    _save_png(source)
    injection = tmp_path / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(
        "from asset_mania import service\n"
        "def fail():\n"
        "    raise RuntimeError('private internal details')\n"
        "service.inspect_environment = fail\n"
    )

    completed = _run(
        tmp_path,
        "inspect",
        str(source),
        "--out",
        "runs",
        extra_environment={"PYTHONPATH": str(injection)},
    )

    assert completed.returncode == 4
    assert completed.stderr == "INTERNAL_ERROR\n"
    report = json.loads(completed.stdout)
    assert report["result"] == {
        "status": "failed",
        "diagnostics": ["INTERNAL_ERROR"],
    }
    run_dir = _only_run(tmp_path / "runs")
    assert completed.stdout == (run_dir / "report.json").read_text()
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    validate(instance=manifest, schema=load_manifest_schema())
    assert "private internal details" not in completed.stdout + completed.stderr
    assert source.name not in completed.stdout + completed.stderr


def test_execute_exception_creates_a_sanitized_completed_failure_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from asset_mania import cli

    source = tmp_path / "private-source.png"
    _save_png(source)

    def fail_before_completion(_request):
        raise RuntimeError(f"private failure at {source}")

    monkeypatch.setattr(cli, "execute_inspect", fail_before_completion)

    exit_code = cli.main(["inspect", str(source), "--out", str(tmp_path / "runs")])

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.err == "INTERNAL_ERROR\n"
    report = json.loads(captured.out)
    assert report["result"] == {
        "status": "failed",
        "diagnostics": ["INTERNAL_ERROR"],
    }
    run_dir = _only_run(tmp_path / "runs")
    assert captured.out == (run_dir / "report.json").read_text()
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    validate(instance=manifest, schema=load_manifest_schema())
    assert source.name not in captured.out + captured.err
    assert str(source) not in captured.out + captured.err


def test_report_render_exception_preserves_the_completed_run_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from asset_mania import cli

    source = tmp_path / "private-source.png"
    _save_png(source)

    def fail_rendering(_report):
        raise RuntimeError(f"private rendering failure at {source}")

    monkeypatch.setattr(cli, "_text_report", fail_rendering)

    exit_code = cli.main(
        [
            "inspect",
            str(source),
            "--out",
            str(tmp_path / "runs"),
            "--format",
            "text",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert captured.err == "INTERNAL_ERROR\n"
    run_dir = _only_run(tmp_path / "runs")
    report = _assert_canonical_json(run_dir / "report.json")
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    assert report["result"]["status"] == "succeeded"
    validate(instance=manifest, schema=load_manifest_schema())
    assert source.name not in captured.err
    assert str(source) not in captured.err


def test_stdout_write_exception_preserves_the_completed_run_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from asset_mania import cli

    source = tmp_path / "private-source.png"
    _save_png(source)

    class FailingStdout:
        def write(self, _payload: str) -> int:
            raise OSError(f"private output failure at {source}")

    captured_stdout = cli.sys.stdout
    monkeypatch.setattr(cli.sys, "stdout", FailingStdout())
    try:
        exit_code = cli.main(["inspect", str(source), "--out", str(tmp_path / "runs")])
    finally:
        monkeypatch.setattr(cli.sys, "stdout", captured_stdout)

    captured = capsys.readouterr()
    assert exit_code == 4
    assert captured.out == ""
    assert captured.err == "INTERNAL_ERROR\n"
    run_dir = _only_run(tmp_path / "runs")
    report = _assert_canonical_json(run_dir / "report.json")
    manifest = _assert_canonical_json(run_dir / "manifest.json")
    assert report["result"]["status"] == "succeeded"
    validate(instance=manifest, schema=load_manifest_schema())
    assert source.name not in captured.err
    assert str(source) not in captured.err


def test_storage_failure_emits_only_sanitized_stderr_and_no_manifest(tmp_path: Path) -> None:
    source = tmp_path / "private-source.png"
    _save_png(source)
    blocked_parent = tmp_path / "private-blocked-output"
    blocked_parent.write_text("not a directory")

    completed = _run(tmp_path, "inspect", str(source), "--out", str(blocked_parent))

    assert completed.returncode == 73
    assert completed.stdout == ""
    assert completed.stderr == "OUTPUT_STORAGE_UNAVAILABLE\n"
    assert source.name not in completed.stderr
    assert blocked_parent.name not in completed.stderr
    assert not list(tmp_path.rglob("manifest.json"))
