import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from asset_mania_contracts import canonical_digest, canonical_json
from asset_mania_pipeline import (
    build_local_face_standing_consent,
    export_clay_glb,
    issue_receipt,
    load_face_geometry_plugin_result,
    sha256_file,
)
from PIL import Image
from scipy.spatial import Delaunay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_face_geometry_e2e as face_geometry_script
from scripts.run_face_geometry_e2e import main


def topology_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces = Delaunay(geometry_points()).simplices.astype(np.int64)
    assert faces.shape == (9976, 3)
    face_indices = np.arange(5023, dtype=np.int64)
    inner = np.arange(256, dtype=np.int64)
    return faces, face_indices, inner


def geometry_vertices() -> np.ndarray:
    points = geometry_points()
    radius_squared = np.sum(points**2, axis=1)
    return np.stack(
        [0.08 * points[:, 0], 0.12 * points[:, 1], 0.03 * (1 - radius_squared)],
        axis=1,
    )


def geometry_points() -> np.ndarray:
    boundary_count = 68
    boundary_angle = np.arange(boundary_count) / boundary_count * 2 * np.pi
    boundary = np.stack([np.cos(boundary_angle), np.sin(boundary_angle)], axis=1)
    interior_count = 5023 - boundary_count
    index = np.arange(interior_count, dtype=np.float64)
    angle = index * np.pi * (3 - np.sqrt(5))
    radius = 0.95 * np.sqrt((index + 0.5) / interior_count)
    interior = np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)
    return np.concatenate([boundary, interior], axis=0)


def standing_consent(path: Path, source: Path) -> dict:
    consent = build_local_face_standing_consent(
        source_sha256=sha256_file(source),
        issued_at="2026-08-22T23:00:00+00:00",
        authorization_evidence_sha256="f" * 64,
    )
    path.write_text(canonical_json(consent), encoding="utf-8")
    return consent


def planned_run(
    tmp_path: Path, *, consent: Path | None = None, run_id: str = "fixedrun"
) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "private-person.png"
    source.write_bytes(b"authorized portrait")
    topology = tmp_path / "flame-topology.npz"
    faces, face_indices, inner = topology_arrays()
    np.savez_compressed(topology, faces=faces, face_indices=face_indices, inner_face_indices=inner)
    output = tmp_path / "runs"
    mica_python = tmp_path / "mica-python.exe"
    mica_plugin = tmp_path / "mica-plugin.py"
    deca_python = tmp_path / "deca-python.exe"
    deca_plugin = tmp_path / "deca-plugin.py"
    for path in (mica_python, mica_plugin, deca_python, deca_plugin):
        path.write_bytes(b"exe")
    checkpoint = tmp_path / "checkpoint.bin"
    flame = tmp_path / "flame.bin"
    detector = tmp_path / "scrfd"
    checkpoint.write_bytes(b"checkpoint")
    flame.write_bytes(b"flame")
    detector.mkdir()
    (detector / "detector.onnx").write_bytes(b"scrfd")
    detector_sha256 = face_geometry_script._directory_sha256(detector)
    for prefix in ("MICA", "DECA"):
        os.environ.update(
            {
                f"ASSET_MANIA_{prefix}_SOURCE_ROOT": str(tmp_path / prefix.lower()),
                f"ASSET_MANIA_{prefix}_ISOLATED_HOME": str(tmp_path / f"{prefix.lower()}-home"),
                f"ASSET_MANIA_{prefix}_CHECKPOINT_PATH": str(checkpoint),
                f"ASSET_MANIA_{prefix}_FLAME_PATH": str(flame),
                f"ASSET_MANIA_{prefix}_FLAME_SHA256": sha256_file(flame),
                f"ASSET_MANIA_{prefix}_DETECTOR_PATH": str(detector),
                f"ASSET_MANIA_{prefix}_DETECTOR_SHA256": detector_sha256,
            }
        )
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
                sha256_file(checkpoint),
                "--mica-detector-sha256",
                detector_sha256,
                "--mica-python",
                str(mica_python),
                "--mica-plugin",
                str(mica_plugin),
                "--deca-revision",
                "c" * 40,
                "--deca-checkpoint-sha256",
                sha256_file(checkpoint),
                "--deca-detector-sha256",
                detector_sha256,
                "--deca-python",
                str(deca_python),
                "--deca-plugin",
                str(deca_plugin),
                "--flame-sha256",
                sha256_file(flame),
            ]
            + (["--standing-consent", str(consent)] if consent is not None else []),
            now="2026-08-23T00:00:00+00:00",
            id_factory=lambda: run_id,
        )
        == 0
    )
    return output / f"20260823T000000Z-{run_id}", source, topology


def test_geometry_plan_seals_plugins_gates_and_no_source_path(tmp_path: Path) -> None:
    run, source, _topology = planned_run(tmp_path)

    plan_text = (run / "plan.json").read_text(encoding="utf-8")
    plan = json.loads(plan_text)

    assert plan["plugins"] == [
        {"plugin": "mica-local", "profile": "identity-neutral-v1"},
        {"plugin": "deca-local", "profile": "detail-displacement-v1"},
    ]
    assert plan["required_gate"] == "face_rights"
    assert plan["gates"]["minimum_head_extent_metres"] == 0.15
    assert plan["gates"]["maximum_head_extent_metres"] == 0.32
    assert (
        plan["gates"]["deca_extent_validation"]
        == "positive-finite-prealignment-then-similarity-fit"
    )
    plan_preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert plan["plan_sha256"] == canonical_digest(plan_preimage)
    assert plan["source_image_sha256"] == sha256_file(source)
    assert str(source) not in plan_text
    assert source.name not in plan_text
    assert "yaw" not in plan_text.lower()


def test_legacy_plan_is_rejected_before_source_or_plugin(tmp_path: Path, monkeypatch) -> None:
    run, source, _topology = planned_run(tmp_path)
    plan_path = run / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    del plan["mica_python_sha256"]
    plan["plan_sha256"] = canonical_digest(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    plan_path.write_text(canonical_json(plan), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        face_geometry_script, "fingerprint_source", lambda *_args: calls.append("source")
    )

    with pytest.raises(ValueError, match="create a new geometry plan"):
        main(
            [
                "mica-run",
                "--run",
                str(run),
                "--source",
                str(source),
                "--rights-store",
                str(tmp_path / "absent-rights"),
                "--python",
                str(tmp_path / "mica-python.exe"),
                "--plugin",
                str(tmp_path / "mica-plugin.py"),
            ],
            plugin_runner=lambda *_args, **_kwargs: calls.append("plugin"),
        )

    assert calls == []


def test_mica_rejects_changed_detector_before_source_or_plugin(tmp_path: Path, monkeypatch) -> None:
    run, source, _topology = planned_run(tmp_path)
    rights = tmp_path / "rights"
    issue_rights(run, rights)
    (tmp_path / "scrfd" / "detector.onnx").write_bytes(b"changed")
    calls = []
    monkeypatch.setattr(
        face_geometry_script, "fingerprint_source", lambda *_args: calls.append("source")
    )

    with pytest.raises(ValueError, match="MICA detector asset digest mismatch"):
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
                str(tmp_path / "mica-python.exe"),
                "--plugin",
                str(tmp_path / "mica-plugin.py"),
            ],
            plugin_runner=lambda *_args, **_kwargs: calls.append("plugin"),
            now="2026-08-23T00:00:00+00:00",
        )

    assert calls == []


def test_standing_consent_changes_plan_digest_and_binds_only_mode_and_digest(
    tmp_path: Path,
) -> None:
    ordinary_run, source, _topology = planned_run(tmp_path / "ordinary")
    consent_path = tmp_path / "private" / "standing-consent.json"
    consent_path.parent.mkdir()
    consent = standing_consent(consent_path, source)
    consent_run, _source, _topology = planned_run(tmp_path / "consented", consent=consent_path)

    ordinary = json.loads((ordinary_run / "plan.json").read_text(encoding="utf-8"))
    plan_text = (consent_run / "plan.json").read_text(encoding="utf-8")
    plan = json.loads(plan_text)

    assert plan["authorization_mode"] == "standing_local_source_consent_v1"
    assert plan["standing_consent_sha256"] == consent["consent_sha256"]
    assert plan["plan_sha256"] != ordinary["plan_sha256"]
    assert str(consent_path) not in plan_text
    assert consent_path.name not in plan_text


def test_standing_consent_mismatch_and_edit_fail_before_source_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"synthetic source")
    consent_path = tmp_path / "standing-consent.json"
    consent = standing_consent(consent_path, source)
    monkeypatch.setattr(
        "scripts.run_face_geometry_e2e.fingerprint_source",
        lambda _path: pytest.fail("source fingerprinted before consent validation"),
    )

    with pytest.raises(ValueError, match="different source digest"):
        planned_run(tmp_path / "mismatch", consent=consent_path)

    consent["issued_at"] = "2026-08-23T00:00:00+00:00"
    consent_path.write_text(canonical_json(consent), encoding="utf-8")
    with pytest.raises(ValueError, match="consent_sha256"):
        planned_run(tmp_path / "edited", consent=consent_path)


@pytest.mark.parametrize("failure", ["mismatch", "tamper"])
def test_mica_revalidates_standing_consent_before_source_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    seed_run, source, _ = planned_run(tmp_path / "seed")
    assert seed_run.is_dir()
    consent_path = tmp_path / "standing-consent.json"
    consent = standing_consent(consent_path, source)
    run, _planned_source, _ = planned_run(tmp_path / "planned", consent=consent_path)
    if failure == "mismatch":
        invalid = build_local_face_standing_consent(
            source_sha256="0" * 64,
            issued_at="2026-08-22T23:00:00+00:00",
            authorization_evidence_sha256="f" * 64,
        )
        expected = "different source digest"
    else:
        invalid = {**consent, "issued_at": "2026-08-23T00:00:00+00:00"}
        expected = "consent_sha256"
    consent_path.write_text(canonical_json(invalid), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        "scripts.run_face_geometry_e2e.fingerprint_source",
        lambda _path: calls.append("fingerprint"),
    )

    with pytest.raises(ValueError, match=expected):
        main(
            [
                "mica-run",
                "--run",
                str(run),
                "--source",
                str(tmp_path / "must-not-resolve.png"),
                "--standing-consent",
                str(consent_path),
                "--python",
                str(tmp_path / "must-not-resolve-python.exe"),
                "--plugin",
                str(tmp_path / "must-not-resolve-plugin.exe"),
            ],
            plugin_runner=lambda *_args, **_kwargs: calls.append("plugin"),
        )

    assert calls == []
    assert not (run / "mica/authorization.json").exists()
    assert not (run / "mica/plugin-output").exists()


def test_two_create_only_plans_reuse_standing_consent_without_consumption(
    tmp_path: Path,
) -> None:
    seed_run, source, _topology = planned_run(tmp_path / "seed")
    assert seed_run.is_dir()
    consent_path = tmp_path / "private-consent.json"
    consent = standing_consent(consent_path, source)
    executable = tmp_path / "tool.exe"
    executable.write_bytes(b"exe")

    for index in (1, 2):
        run, planned_source, _ = planned_run(
            tmp_path / f"run-{index}", consent=consent_path, run_id=f"run{index}"
        )
        assert (
            main(
                [
                    "mica-run",
                    "--run",
                    str(run),
                    "--source",
                    str(planned_source),
                    "--standing-consent",
                    str(consent_path),
                    "--python",
                    str(executable),
                    "--plugin",
                    str(executable),
                ],
                plugin_runner=fake_plugin_runner,
                now="2026-08-23T00:00:00+00:00",
            )
            == 0
        )
        authorization_text = (run / "mica/authorization.json").read_text(encoding="utf-8")
        authorization = json.loads(authorization_text)
        assert authorization["authorization_mode"] == "standing_local_source_consent_v1"
        assert authorization["receipt_sha256"] == consent["consent_sha256"]
        assert str(consent_path) not in authorization_text
        assert consent_path.name not in authorization_text


def fake_plugin_runner(command, request, request_path, result_path, **_kwargs):
    assert command
    request.output_directory.mkdir()
    displacement = (
        np.zeros(5023, dtype=np.float32)
        if request.plugin == "mica-local"
        else np.full(5023, 0.0005, dtype=np.float32)
    )
    faces, _face_indices, _inner = topology_arrays()
    vertices = geometry_vertices().astype(np.float32)
    if request.plugin == "deca-local":
        vertices *= 0.324885711 / float(np.ptp(vertices, axis=0).max())
    np.savez_compressed(
        request.output_directory / "geometry.npz",
        vertices=vertices,
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
    mica_record = json.loads((run / "mica/record.json").read_text(encoding="utf-8"))
    assert mica_record["python_sha256"] == sha256_file(python)
    assert mica_record["plugin_executable_sha256"] == sha256_file(mica)


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


def verified_run(tmp_path: Path) -> Path:
    run, _source = completed_plugin_run(tmp_path)
    assert main(["geometry-fuse", "--run", str(run)]) == 0
    dad = tmp_path / "dad.glb"
    dad.write_bytes(b"baseline")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"blender")

    def fake_preview(_mesh: Path, output: Path, _blender: Path) -> None:
        Image.new("RGB", (8, 8), (120, 120, 120)).save(output)

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
    return run


def test_fuse_verify_and_manual_review_are_create_only(tmp_path: Path) -> None:
    run, _source = completed_plugin_run(tmp_path)

    with np.load(run / "deca/plugin-output/geometry.npz", allow_pickle=False) as archive:
        raw_deca = archive["vertices"].copy()
        assert np.isclose(np.ptp(archive["vertices"], axis=0).max(), 0.324885711)

    assert main(["geometry-fuse", "--run", str(run)]) == 0
    assert (run / "fusion/mica-clay.glb").is_file()
    assert (run / "fusion/deca-clay.glb").is_file()
    assert (run / "fusion/mica-deca-clay.glb").is_file()
    assert len(list((run / "fusion").glob("likeness-*.json"))) == 3
    deca_scene = trimesh.load(run / "fusion/deca-clay.glb", process=False)
    deca_export = next(iter(deca_scene.geometry.values()))
    assert np.allclose(deca_export.vertices, geometry_vertices(), atol=1e-6)
    with np.load(run / "deca/plugin-output/geometry.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["vertices"], raw_deca)

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


def test_fusion_rejects_nonzero_mica_detail(tmp_path: Path) -> None:
    run, _source = completed_plugin_run(tmp_path)
    geometry_path = run / "mica/plugin-output/geometry.npz"
    with np.load(geometry_path, allow_pickle=False) as archive:
        payload = {name: archive[name].copy() for name in archive.files}
    payload["detail_displacement"][0] = 0.0001
    np.savez_compressed(geometry_path, **payload)
    record_path = run / "mica/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["geometry_sha256"] = sha256_file(geometry_path)
    preimage = {key: value for key, value in record.items() if key != "record_sha256"}
    record["record_sha256"] = canonical_digest(preimage)
    record_path.write_text(canonical_json(record), encoding="utf-8")

    with pytest.raises(ValueError, match="MICA detail displacement must be zero"):
        main(["geometry-fuse", "--run", str(run)])


def test_default_preview_uses_current_python_and_canonical_front_angle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(
        face_geometry_script.subprocess,
        "run",
        lambda command, **_kwargs: captured.append(command) or Completed(),
    )

    face_geometry_script._default_preview(
        tmp_path / "mesh.glb", tmp_path / "preview.png", tmp_path / "blender.exe"
    )

    command = captured[0]
    assert command[0] == sys.executable
    angle_index = command.index("--start-angle-degrees")
    assert command[angle_index + 1] == "90"


def test_fusion_rejects_topology_changed_after_plan(tmp_path: Path) -> None:
    run, _source = completed_plugin_run(tmp_path)
    (run / "topology.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="topology digest mismatch"):
        main(["geometry-fuse", "--run", str(run)])


@pytest.mark.parametrize("stage", ["mica", "deca"])
def test_fusion_rejects_plugin_geometry_changed_after_record(tmp_path: Path, stage: str) -> None:
    run, _source = completed_plugin_run(tmp_path)
    (run / stage / "plugin-output/geometry.npz").write_bytes(b"tampered")

    with pytest.raises(ValueError, match=f"{stage.upper()} geometry digest mismatch"):
        main(["geometry-fuse", "--run", str(run)])


@pytest.mark.parametrize(
    "artifact",
    ["mica-clay.glb", "deca-clay.glb", "mica-deca-clay.glb"],
)
def test_verification_rejects_fusion_artifact_changed_after_record(
    tmp_path: Path, artifact: str
) -> None:
    run, _source = completed_plugin_run(tmp_path)
    assert main(["geometry-fuse", "--run", str(run)]) == 0
    (run / "fusion" / artifact).write_bytes(b"tampered")
    blender = tmp_path / "blender.exe"
    blender.write_bytes(b"blender")
    dad = tmp_path / "dad.glb"
    dad.write_bytes(b"baseline")

    with pytest.raises(ValueError, match="fusion artifact digest mismatch"):
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
            preview_runner=lambda *_args: pytest.fail("preview must not run"),
        )


def test_verification_report_is_canonically_self_sealed(tmp_path: Path) -> None:
    run = verified_run(tmp_path)
    report = json.loads((run / "verification/report.json").read_text(encoding="utf-8"))

    seal = report.pop("report_sha256")
    assert seal == canonical_digest(report)


def test_review_rejects_mutated_verification_report(tmp_path: Path) -> None:
    run = verified_run(tmp_path)
    report_path = run / "verification/report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report_path.write_text(canonical_json(report), encoding="utf-8")

    with pytest.raises(ValueError, match="verification report digest"):
        main(
            [
                "geometry-review",
                "--run",
                str(run),
                "--verdict",
                "failed",
                "--reason",
                "tampered report",
            ]
        )


def test_review_rejects_comparison_changed_after_verification(tmp_path: Path) -> None:
    run = verified_run(tmp_path)
    (run / "verification/comparison.png").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="comparison digest mismatch"):
        main(
            [
                "geometry-review",
                "--run",
                str(run),
                "--verdict",
                "failed",
                "--reason",
                "tampered comparison",
            ]
        )


@pytest.mark.parametrize(
    ("target", "message"),
    [("python", "Python runtime digest mismatch"), ("plugin", "plugin executable digest mismatch")],
)
def test_plugin_stage_rejects_executable_changed_during_execution(
    tmp_path: Path, target: str, message: str
) -> None:
    run, source, _topology = planned_run(tmp_path)
    rights = tmp_path / "rights"
    issue_rights(run, rights)
    python = tmp_path / "python.exe"
    plugin = tmp_path / "mica.exe"
    python.write_bytes(b"exe")
    plugin.write_bytes(b"exe")

    def mutating_runner(*args, **kwargs):
        result = fake_plugin_runner(*args, **kwargs)
        (python if target == "python" else plugin).write_bytes(b"mutated executable")
        return result

    with pytest.raises(ValueError, match=message):
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
                str(plugin),
            ],
            plugin_runner=mutating_runner,
            now="2026-08-23T00:00:00+00:00",
        )
