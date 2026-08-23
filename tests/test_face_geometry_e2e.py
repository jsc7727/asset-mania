import json
import sys
from pathlib import Path

import numpy as np
from asset_mania_contracts import canonical_json
from asset_mania_pipeline import (
    export_clay_glb,
    issue_receipt,
    load_face_geometry_plugin_result,
    sha256_file,
)
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_face_geometry_e2e import main


def topology_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(9976, dtype=np.int64)
    faces = np.stack([indices % 5023, (indices + 1) % 5023, (indices + 2) % 5023], axis=1)
    face_indices = np.arange(5023, dtype=np.int64)
    inner = np.arange(256, dtype=np.int64)
    return faces, face_indices, inner


def geometry_vertices() -> np.ndarray:
    index = np.arange(5023, dtype=np.float64)
    angle = index / 5023 * 2 * np.pi
    y = np.linspace(-0.12, 0.12, 5023)
    return np.stack([0.08 * np.cos(angle), y, -0.06 * np.sin(angle)], axis=1)


def planned_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "private-person.png"
    source.write_bytes(b"authorized portrait")
    topology = tmp_path / "flame-topology.npz"
    faces, face_indices, inner = topology_arrays()
    np.savez_compressed(topology, faces=faces, face_indices=face_indices, inner_face_indices=inner)
    output = tmp_path / "runs"
    assert (
        main(
            [
                "geometry-plan",
                "--out",
                str(output),
                "--source-sha256",
                sha256_file(source),
                "--topology",
                str(topology),
                "--topology-sha256",
                sha256_file(topology),
                "--mica-revision",
                "a" * 40,
                "--mica-checkpoint-sha256",
                "b" * 64,
                "--deca-revision",
                "c" * 40,
                "--deca-checkpoint-sha256",
                "d" * 64,
                "--flame-sha256",
                "e" * 64,
            ],
            now="2026-08-23T00:00:00+00:00",
            id_factory=lambda: "fixedrun",
        )
        == 0
    )
    return output / "20260823T000000Z-fixedrun", source, topology


def test_geometry_plan_seals_plugins_gates_and_no_source_path(tmp_path: Path) -> None:
    run, source, _topology = planned_run(tmp_path)

    plan_text = (run / "plan.json").read_text(encoding="utf-8")
    plan = json.loads(plan_text)

    assert plan["plugins"] == [
        {"plugin": "mica-local", "profile": "identity-neutral-v1"},
        {"plugin": "deca-local", "profile": "detail-displacement-v1"},
    ]
    assert plan["required_gate"] == "face_rights"
    assert plan["source_image_sha256"] == sha256_file(source)
    assert str(source) not in plan_text
    assert source.name not in plan_text
    assert "yaw" not in plan_text.lower()


def fake_plugin_runner(command, request, request_path, result_path, **_kwargs):
    assert command
    request.output_directory.mkdir()
    displacement = (
        np.zeros(5023, dtype=np.float32)
        if request.plugin == "mica-local"
        else np.full(5023, 0.0005, dtype=np.float32)
    )
    faces, _face_indices, _inner = topology_arrays()
    np.savez_compressed(
        request.output_directory / "geometry.npz",
        vertices=geometry_vertices().astype(np.float32),
        faces=faces,
        source_projection=np.zeros((5023, 2), dtype=np.float32),
        detail_displacement=displacement,
    )
    result_path.write_text(
        canonical_json(
            {
                "schema": "asset-mania.face-geometry-plugin-result.v1",
                "plugin": request.plugin,
                "profile": request.profile,
                "status": "succeeded",
                "geometry": str((request.output_directory / "geometry.npz").resolve()),
                "vertex_count": 5023,
                "triangle_count": 9976,
                "elapsed_seconds": 0.1,
                "device": "cuda",
                "checkpoint_sha256": request.checkpoint_sha256,
                "topology": "flame-2020-5023",
                "ephemeral_identity_feature_used": request.plugin == "mica-local",
                "persisted_identity_feature_count": 0,
            }
        ),
        encoding="utf-8",
    )
    return load_face_geometry_plugin_result(result_path, request)


def issue_rights(run: Path, store: Path) -> None:
    plan = json.loads((run / "plan.json").read_text(encoding="utf-8"))
    receipt = issue_receipt(
        receipt_id="face-geometry-test",
        plan_sha256=plan["plan_sha256"],
        gate="face_rights",
        acknowledgement=f"face_rights:{plan['plan_sha256']}",
        disclosure="synthetic test rights",
        issued_at="2026-08-22T23:00:00+00:00",
        expires_at="2026-08-24T00:00:00+00:00",
    )
    store.mkdir()
    (store / "receipt.json").write_text(canonical_json(receipt), encoding="utf-8")


def test_plugin_stages_consume_rights_once_and_write_closed_records(tmp_path: Path) -> None:
    run, source, _topology = planned_run(tmp_path)
    rights = tmp_path / "rights"
    issue_rights(run, rights)
    python = tmp_path / "python.exe"
    mica = tmp_path / "mica.exe"
    deca = tmp_path / "deca.exe"
    for executable in (python, mica, deca):
        executable.write_bytes(b"exe")

    assert (
        main(
            [
                "mica-run",
                "--run",
                str(run),
                "--source",
                str(source),
                "--rights-store",
                str(rights),
                "--python",
                str(python),
                "--plugin",
                str(mica),
            ],
            plugin_runner=fake_plugin_runner,
            now="2026-08-23T00:00:00+00:00",
        )
        == 0
    )
    assert (
        main(
            [
                "deca-run",
                "--run",
                str(run),
                "--source",
                str(source),
                "--python",
                str(python),
                "--plugin",
                str(deca),
            ],
            plugin_runner=fake_plugin_runner,
        )
        == 0
    )
    assert len(list((rights / "consumed").glob("*.json"))) == 1
    assert (run / "mica/record.json").is_file()
    assert (run / "deca/record.json").is_file()


def completed_plugin_run(tmp_path: Path) -> tuple[Path, Path]:
    run, source, _topology = planned_run(tmp_path)
    rights = tmp_path / "rights"
    issue_rights(run, rights)
    executable = tmp_path / "tool.exe"
    executable.write_bytes(b"exe")
    main(
        [
            "mica-run",
            "--run",
            str(run),
            "--source",
            str(source),
            "--rights-store",
            str(rights),
            "--python",
            str(executable),
            "--plugin",
            str(executable),
        ],
        plugin_runner=fake_plugin_runner,
        now="2026-08-23T00:00:00+00:00",
    )
    main(
        [
            "deca-run",
            "--run",
            str(run),
            "--source",
            str(source),
            "--python",
            str(executable),
            "--plugin",
            str(executable),
        ],
        plugin_runner=fake_plugin_runner,
    )
    return run, source


def test_fuse_verify_and_manual_review_are_create_only(tmp_path: Path) -> None:
    run, _source = completed_plugin_run(tmp_path)

    assert main(["geometry-fuse", "--run", str(run)]) == 0
    assert (run / "fusion/mica-clay.glb").is_file()
    assert (run / "fusion/deca-clay.glb").is_file()
    assert (run / "fusion/mica-deca-clay.glb").is_file()
    assert len(list((run / "fusion").glob("likeness-*.json"))) == 3

    dad = tmp_path / "dad.glb"
    faces, _face_indices, _inner = topology_arrays()
    from asset_mania_pipeline.face_geometry import FaceGeometryData

    export_clay_glb(
        FaceGeometryData(
            geometry_vertices(),
            faces,
            np.zeros((5023, 2)),
            np.zeros(5023),
        ),
        dad,
    )
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"exe")

    def fake_preview(_mesh: Path, output: Path, _blender: Path) -> None:
        Image.new("RGB", (80, 80), (120, 120, 120)).save(output)

    assert (
        main(
            [
                "geometry-verify",
                "--run",
                str(run),
                "--blender",
                str(blender),
                "--dad-baseline",
                str(dad),
            ],
            preview_runner=fake_preview,
        )
        == 0
    )
    assert (
        main(
            [
                "geometry-review",
                "--run",
                str(run),
                "--verdict",
                "passed",
                "--reason",
                "synthetic clay comparison passed",
            ]
        )
        == 0
    )
    review = json.loads((run / "verification/manual-review.json").read_text(encoding="utf-8"))
    assert review["visual_quality"] == "passed"
    assert len(review["review_sha256"]) == 64
