"""Maintainer-only staged E2E for the private face-anchor visual-hull profile."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_triposr import FaceHybridResult
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_face_hybrid_e2e import main

pytestmark = [
    pytest.mark.filterwarnings("ignore:'pkgutil.find_loader' is deprecated:DeprecationWarning"),
    pytest.mark.filterwarnings("ignore:__array_wrap__ must accept context:DeprecationWarning"),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _export_glb(mesh, output_path: Path) -> None:
    """Exercise real trimesh export under the root NumPy-2 test environment."""
    import trimesh

    original = trimesh.util.allclose
    trimesh.util.allclose = lambda a, b, atol=1e-8: float(np.ptp(a - b)) < atol
    try:
        mesh.export(output_path, file_type="glb")
    finally:
        trimesh.util.allclose = original


def _private_views(directory: Path) -> Path:
    directory.mkdir(parents=True)
    for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315)):
        image = Image.new("RGB", (1024, 1024), (210, 212, 216))
        mask = Image.new("L", (1024, 1024), 0)
        box = (250 + index, 150, 774 + index, 874)
        ImageDraw.Draw(image).ellipse(box, fill=(60 + index * 12, 90, 120))
        ImageDraw.Draw(mask).ellipse(box, fill=255)
        image.save(directory / f"yaw-{yaw:03d}.png")
        mask.save(directory / f"yaw-{yaw:03d}-mask.png")
    return directory


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    views = _private_views(tmp_path / "private-views")
    code = main(
        ["prepare", "--views", str(views), "--out", str(tmp_path / "runs")],
        now="2026-08-23T01:00:00Z",
        id_factory=lambda: "run-test",
    )
    assert code == 0
    return next((tmp_path / "runs").iterdir()), views


def test_prepare_is_create_only_private_and_source_read_only(tmp_path: Path) -> None:
    views = _private_views(tmp_path / "private-views")
    before = {path: _sha256(path) for path in views.glob("*.png")}

    code = main(
        ["prepare", "--views", str(views), "--out", str(tmp_path / "runs")],
        now="2026-08-23T01:00:00Z",
        id_factory=lambda: "run-test",
    )

    assert code == 0
    run = next((tmp_path / "runs").iterdir())
    manifest = json.loads((run / "prepare/manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"] == "face-anchor-visual-hull-v1"
    assert len(manifest["source_views"]) == 8
    assert len(list((run / "prepare/canonical").glob("yaw-*.png"))) == 16
    assert before == {path: _sha256(path) for path in before}
    serialized = (run / "prepare/manifest.json").read_text(encoding="utf-8")
    assert str(views) not in serialized
    assert "private-views" not in serialized


def test_anchor_requires_cuda_and_writes_measured_create_only_glb(tmp_path: Path) -> None:
    run, _views = _prepare(tmp_path)
    clearance = ROOT / "tests/fixtures/v2/engine-clearance-v1.json"

    with pytest.raises(ValueError, match="CUDA"):
        main(
            [
                "anchor",
                "--run",
                str(run),
                "--clearance",
                str(clearance),
                "--engine-root",
                str(tmp_path / "engine"),
                "--weights",
                str(tmp_path / "weights"),
                "--hub-cache",
                str(tmp_path / "hub"),
                "--device",
                "cpu",
            ],
            now="2026-08-23T01:01:00Z",
        )

    def fake_anchor_runner(*, output_path: Path, **kwargs):
        import trimesh

        _export_glb(trimesh.creation.icosphere(subdivisions=2, radius=0.4), output_path)
        return {
            "device": "cuda",
            "elapsed_seconds": 1.25,
            "peak_allocated_bytes": 1024,
            "triangle_count": 320,
            "vertex_count": 162,
            "manifold": "closed",
        }

    code = main(
        [
            "anchor",
            "--run",
            str(run),
            "--clearance",
            str(clearance),
            "--engine-root",
            str(tmp_path / "engine"),
            "--weights",
            str(tmp_path / "weights"),
            "--hub-cache",
            str(tmp_path / "hub"),
            "--device",
            "cuda",
        ],
        now="2026-08-23T01:01:00Z",
        anchor_runner=fake_anchor_runner,
    )

    assert code == 0
    record = json.loads((run / "anchor/record.json").read_text(encoding="utf-8"))
    assert record["device"] == "cuda"
    assert record["peak_allocated_bytes"] == 1024
    assert (run / "anchor/anchor.glb").is_file()


def test_fuse_and_verify_recompute_artifact_and_source_integrity(tmp_path: Path) -> None:
    run, views = _prepare(tmp_path)
    clearance = ROOT / "tests/fixtures/v2/engine-clearance-v1.json"

    def fake_anchor_runner(*, output_path: Path, **kwargs):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        _export_glb(mesh, output_path)
        return {
            "device": "cuda",
            "elapsed_seconds": 1.0,
            "peak_allocated_bytes": 2048,
            "triangle_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
            "manifold": "closed",
        }

    assert (
        main(
            [
                "anchor",
                "--run",
                str(run),
                "--clearance",
                str(clearance),
                "--engine-root",
                str(tmp_path / "engine"),
                "--weights",
                str(tmp_path / "weights"),
                "--hub-cache",
                str(tmp_path / "hub"),
            ],
            now="2026-08-23T01:01:00Z",
            anchor_runner=fake_anchor_runner,
        )
        == 0
    )

    def fake_fusion_runner(*, output_path: Path, **kwargs):
        import trimesh

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.45)
        mesh.visual.vertex_colors = [120, 140, 160, 255]
        _export_glb(mesh, output_path)
        return FaceHybridResult(320, 162, "closed", float(mesh.volume), 1, 0.8, 0.9, 0.95, 1.0)

    assert main(["fuse", "--run", str(run)], fusion_runner=fake_fusion_runner) == 0

    def fake_preview_runner(mesh: Path, preview: Path, blender: Path | None):
        assert mesh == run / "fusion/face-hybrid.glb"
        Image.new("RGB", (64, 64), (20, 30, 40)).save(preview)

    code = main(
        ["verify", "--run", str(run), "--views", str(views)],
        preview_runner=fake_preview_runner,
    )

    assert code == 0
    report = json.loads((run / "verification/report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["source_unchanged"] is True
    assert report["artifact"]["watertight"] is True
    assert report["visual_quality"] == "unreviewed"
    assert (run / "verification/preview.png").is_file()
