import json
import sys
from pathlib import Path

import numpy as np
import pytest
from asset_mania_pipeline import FacePluginResult
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_face_plugin_e2e import (
    CHECKPOINT_BYTES,
    CHECKPOINT_URL,
    DAD_REVISION,
    SOURCE_URL,
    _plugin_environment,
    main,
)


def _planned_run(tmp_path: Path) -> Path:
    output = tmp_path / "runs"
    assert (
        main(
            ["plan", "--out", str(output), "--plugin", "dad3dheads-local"],
            now="2026-08-23T00:00:00+00:00",
            id_factory=lambda: "fixedrun",
        )
        == 0
    )
    return output / "20260823T000000Z-fixedrun"


def _acquired_run(tmp_path: Path) -> Path:
    run = _planned_run(tmp_path)

    def fake_git(_url: str, _revision: str, destination: Path) -> None:
        destination.mkdir()
        (destination / "LICENSE").write_text("CC BY-NC-SA 4.0", encoding="utf-8")

    def fake_download(_url: str, destination: Path, _expected_bytes: int) -> None:
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"checkpoint")

    main(
        [
            "acquire",
            "--run",
            str(run),
            "--approval-reference",
            "face-plugin-approval-20260823",
        ],
        git_acquirer=fake_git,
        checkpoint_downloader=fake_download,
        revision_reader=lambda _source: DAD_REVISION,
        expected_checkpoint_bytes=len(b"checkpoint"),
    )
    return run


def _cuda_probe(_python: Path) -> dict:
    return {
        "python": "3.11.9",
        "torch": "2.13.0+cu130",
        "cuda_runtime": "13.0",
        "cuda_available": True,
        "device_type": "cuda",
    }


def _fake_plugin(*, request, **_kwargs) -> FacePluginResult:
    request.output_directory.mkdir()
    mesh = request.output_directory / "head.obj"
    projection = request.output_directory / "projection.npz"
    mesh.write_text(
        """
v -1 -1 0
v 1 -1 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
""".lstrip(),
        encoding="utf-8",
    )
    with Image.open(request.source_image) as opened:
        width, height = opened.size
    np.savez_compressed(
        projection,
        projected_vertices=np.array(
            [[0, 0], [width - 1, 0], [width // 2, height - 1], [width // 2, height // 2]],
            dtype=float,
        ),
        camera_vertices=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        ),
        image_shape=np.array([height, width], dtype=np.int64),
    )
    return FacePluginResult(
        schema="asset-mania.face-plugin-result.v0",
        plugin=request.plugin,
        status="succeeded",
        raw_mesh=mesh,
        projection_data=projection,
        vertex_count=4,
        triangle_count=4,
        elapsed_seconds=0.01,
        device="cuda",
        checkpoint_sha256=request.checkpoint_sha256,
    )


def test_plan_fixes_model_license_runtime_and_no_egress(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    plan = json.loads((run / "plan/plan.json").read_text(encoding="utf-8"))

    assert plan["plugin_revision"] == DAD_REVISION
    assert plan["source_url"] == SOURCE_URL
    assert plan["checkpoint_url"] == CHECKPOINT_URL
    assert plan["checkpoint_expected_bytes"] == CHECKPOINT_BYTES
    assert plan["license"] == "CC-BY-NC-SA-4.0"
    assert plan["commercial_use"] == "forbidden-for-this-profile"
    assert plan["device"] == "cuda"
    assert plan["torch"] == "2.13.0+cu130"
    assert plan["retry_count"] == 0
    assert plan["face_egress"] == "none"
    assert len(plan["plan_sha256"]) == 64


def test_plugin_environment_keeps_path_but_drops_provider_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("PATH", "synthetic-path")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    environment = _plugin_environment(tmp_path / "source", tmp_path / "home")

    assert environment["PATH"] == "synthetic-path"
    assert "OPENAI_API_KEY" not in environment


def test_acquire_requires_exact_approval_reference(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)

    with pytest.raises(ValueError, match="fresh acquisition approval is required"):
        main(["acquire", "--run", str(run), "--approval-reference", "yes"])


def test_acquire_records_exact_revision_and_checkpoint(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    checkpoint_bytes = b"checkpoint"

    def fake_git(url: str, revision: str, destination: Path) -> None:
        assert url == SOURCE_URL
        assert revision == DAD_REVISION
        destination.mkdir()
        (destination / "LICENSE").write_text("CC BY-NC-SA 4.0", encoding="utf-8")

    def fake_download(url: str, destination: Path, expected_bytes: int) -> None:
        assert url == CHECKPOINT_URL
        assert expected_bytes == CHECKPOINT_BYTES
        destination.parent.mkdir(parents=True)
        destination.write_bytes(checkpoint_bytes)

    assert (
        main(
            [
                "acquire",
                "--run",
                str(run),
                "--approval-reference",
                "face-plugin-approval-20260823",
            ],
            git_acquirer=fake_git,
            checkpoint_downloader=fake_download,
            revision_reader=lambda _source: DAD_REVISION,
            expected_checkpoint_bytes=len(checkpoint_bytes),
        )
        == 0
    )
    receipt = json.loads((run / "acquisition/receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_revision"] == DAD_REVISION
    assert receipt["checkpoint_bytes"] == len(checkpoint_bytes)
    assert len(receipt["checkpoint_sha256"]) == 64
    assert receipt["redistribution"] == "uncleared"
    assert "clearance" not in receipt


def test_acquire_is_create_only(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    (run / "acquisition/source").mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "acquire",
                "--run",
                str(run),
                "--approval-reference",
                "face-plugin-approval-20260823",
            ]
        )


def test_smoke_requires_exact_cuda_runtime_and_seals_result(tmp_path: Path) -> None:
    run = _acquired_run(tmp_path)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "plugin.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")

    assert (
        main(
            [
                "smoke",
                "--run",
                str(run),
                "--python",
                str(python),
                "--plugin-command",
                str(plugin),
            ],
            runtime_probe=_cuda_probe,
            plugin_runner=_fake_plugin,
        )
        == 0
    )
    record = json.loads((run / "smoke/record.json").read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["device"] == "cuda"
    assert record["torch"] == "2.13.0+cu130"
    assert record["vertex_count"] == 4
    assert len(record["record_sha256"]) == 64


def test_smoke_refuses_cpu_or_wrong_torch(tmp_path: Path) -> None:
    run = _acquired_run(tmp_path)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "plugin.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")

    def bad_probe(_python: Path) -> dict:
        return {**_cuda_probe(_python), "torch": "2.12.0+cu130", "cuda_available": False}

    with pytest.raises(ValueError, match="approved CUDA runtime"):
        main(
            [
                "smoke",
                "--run",
                str(run),
                "--python",
                str(python),
                "--plugin-command",
                str(plugin),
            ],
            runtime_probe=bad_probe,
            plugin_runner=_fake_plugin,
        )


def test_smoke_resumes_same_created_input_after_plugin_failure(tmp_path: Path) -> None:
    run = _acquired_run(tmp_path)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "plugin.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")

    def fail_once(**_kwargs):
        raise ValueError("synthetic plugin failure")

    with pytest.raises(ValueError, match="synthetic plugin failure"):
        main(
            [
                "smoke",
                "--run",
                str(run),
                "--python",
                str(python),
                "--plugin-command",
                str(plugin),
            ],
            runtime_probe=_cuda_probe,
            plugin_runner=fail_once,
        )

    assert (
        main(
            [
                "smoke",
                "--run",
                str(run),
                "--python",
                str(python),
                "--plugin-command",
                str(plugin),
            ],
            runtime_probe=_cuda_probe,
            plugin_runner=_fake_plugin,
        )
        == 0
    )


def test_private_run_preserves_source_and_redacts_path(tmp_path: Path) -> None:
    run = _acquired_run(tmp_path)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "plugin.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")
    main(
        [
            "smoke",
            "--run",
            str(run),
            "--python",
            str(python),
            "--plugin-command",
            str(plugin),
        ],
        runtime_probe=_cuda_probe,
        plugin_runner=_fake_plugin,
    )
    source = tmp_path / "private-person.png"
    Image.new("RGB", (32, 32), (210, 180, 160)).save(source)
    before = source.read_bytes()

    assert (
        main(
            [
                "run",
                "--run",
                str(run),
                "--source",
                str(source),
                "--python",
                str(python),
                "--plugin-command",
                str(plugin),
            ],
            runtime_probe=_cuda_probe,
            plugin_runner=_fake_plugin,
        )
        == 0
    )
    assert source.read_bytes() == before
    record_text = (run / "inference/record.json").read_text(encoding="utf-8")
    assert str(source) not in record_text
    assert source.name not in record_text
    record = json.loads(record_text)
    assert record["source_unchanged"] is True
    assert record["identity_consistency"] == "unmeasured"


def _private_inference_run(tmp_path: Path) -> tuple[Path, Path]:
    run = _acquired_run(tmp_path)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "plugin.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")
    main(
        [
            "smoke",
            "--run",
            str(run),
            "--python",
            str(python),
            "--plugin-command",
            str(plugin),
        ],
        runtime_probe=_cuda_probe,
        plugin_runner=_fake_plugin,
    )
    source = tmp_path / "private-person.png"
    Image.new("RGB", (32, 32), (210, 180, 160)).save(source)
    main(
        [
            "run",
            "--run",
            str(run),
            "--source",
            str(source),
            "--python",
            str(python),
            "--plugin-command",
            str(plugin),
        ],
        runtime_probe=_cuda_probe,
        plugin_runner=_fake_plugin,
    )
    return run, source


def test_convert_seals_plain_and_colored_glbs(tmp_path: Path) -> None:
    run, _source = _private_inference_run(tmp_path)

    assert main(["convert", "--run", str(run)]) == 0

    record = json.loads((run / "conversion/record.json").read_text(encoding="utf-8"))
    assert record["component_count"] == 1
    assert record["non_manifold_edge_count"] == 0
    assert record["identity_consistency"] == "unmeasured"
    assert len(record["plain_glb_sha256"]) == 64
    assert len(record["colored_glb_sha256"]) == 64


def test_convert_attempt_two_uses_new_create_only_paths(tmp_path: Path) -> None:
    run, _source = _private_inference_run(tmp_path)
    main(["convert", "--run", str(run)])

    assert main(["convert", "--run", str(run), "--attempt", "2"]) == 0

    assert (run / "conversion/head-v2.glb").is_file()
    assert (run / "conversion/head-colored-v2.glb").is_file()
    assert (run / "conversion/record-v2.json").is_file()


def test_verify_writes_three_row_comparison_and_unreviewed_report(tmp_path: Path) -> None:
    run, source = _private_inference_run(tmp_path)
    main(["convert", "--run", str(run)])
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"exe")
    anchor = tmp_path / "anchor.glb"
    hybrid = tmp_path / "hybrid.glb"
    anchor.write_bytes(b"prior-anchor")
    hybrid.write_bytes(b"prior-hybrid")
    calls = []

    def fake_preview(mesh: Path, output: Path, executable: Path) -> None:
        calls.append((mesh, executable))
        Image.new("RGB", (80, 40), (20 * len(calls), 30, 40)).save(output)

    assert (
        main(
            [
                "verify",
                "--run",
                str(run),
                "--source",
                str(source),
                "--blender",
                str(blender),
                "--triposr-anchor",
                str(anchor),
                "--triposr-hybrid",
                str(hybrid),
            ],
            preview_runner=fake_preview,
        )
        == 0
    )
    assert len(calls) == 3
    report = json.loads((run / "verification/report.json").read_text(encoding="utf-8"))
    assert report["visual_quality"] == "unreviewed"
    assert report["identity_consistency"] == "unmeasured"
    assert report["source_unchanged"] is True
    assert (run / "verification/comparison.png").is_file()
