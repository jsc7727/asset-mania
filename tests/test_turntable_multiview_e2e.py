"""Maintainer E2E orchestration for turntable generation and local fusion."""

import base64
import hashlib
import io
import json
import socket
import sys
from pathlib import Path

from asset_mania_contracts import canonical_digest
from asset_mania_engine_triposr import FusionResult
from asset_mania_pipeline import acknowledgement_text, issue_receipt
from asset_mania_provider_openai.transport import ProviderResponse, RecordingTransport
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_turntable_multiview_e2e import main


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    image_path = tmp_path / "synthetic-subject.png"
    mask_path = tmp_path / "synthetic-mask.png"
    image = Image.new("RGB", (1024, 1024), (245, 245, 245))
    mask = Image.new("L", (1024, 1024), 0)
    ImageDraw.Draw(image).ellipse((290, 190, 734, 834), fill=(40, 80, 120))
    ImageDraw.Draw(mask).ellipse((290, 190, 734, 834), fill=255)
    image.save(image_path)
    mask.save(mask_path)

    clearance_path = tmp_path / "clearance.json"
    clearance_path.write_text(
        (ROOT / "tests/fixtures/v2/engine-clearance-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    evidence = json.loads(
        (ROOT / "tests/fixtures/v2/provider-evidence-v1.json").read_text(encoding="utf-8")
    )
    evidence["retrieved_at"] = "2026-08-22T08:00:00Z"
    evidence["expires_at"] = "2026-08-23T08:00:00Z"
    for source in evidence["sources"]:
        source["retrieved_at"] = evidence["retrieved_at"]
    evidence["pricing"]["retrieved_at"] = evidence["retrieved_at"]
    preimage = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    evidence = {**preimage, "evidence_sha256": canonical_digest(preimage)}
    evidence_path = tmp_path / "provider-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    prompt_path = tmp_path / "private-prompt.txt"
    prompt_path.write_text(
        "Preserve the synthetic subject in a neutral studio turntable.\n",
        encoding="utf-8",
    )
    return image_path, mask_path, clearance_path, evidence_path, prompt_path


def test_plan_is_offline_create_only_and_source_read_only(tmp_path: Path, monkeypatch) -> None:
    def deny(*args, **kwargs):
        raise AssertionError("offline planning must not open a socket")

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    image, mask, clearance, evidence, prompt = _inputs(tmp_path)
    before = hashlib.sha256(image.read_bytes()).hexdigest()
    runs = tmp_path / "runs"

    code = main(
        [
            "plan",
            "--image",
            str(image),
            "--mask",
            str(mask),
            "--clearance",
            str(clearance),
            "--evidence",
            str(evidence),
            "--prompt-file",
            str(prompt),
            "--subject",
            "synthetic-person",
            "--out",
            str(runs),
        ],
        now="2026-08-22T09:00:00Z",
        id_factory=lambda: "run-test-1",
    )

    assert code == 0
    children = list(runs.iterdir())
    assert len(children) == 1
    run = children[0]
    plan = json.loads((run / "plan/turntable-plan.json").read_text(encoding="utf-8"))
    assert plan["yaws"] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert plan["model"] == "gpt-image-2-2026-04-21"
    assert (run / "plan/source-cutout.png").is_file()
    assert (run / "provider-quarantine").is_dir()
    assert (run / "viewset").is_dir()
    assert (run / "per-view-meshes").is_dir()
    assert (run / "fusion").is_dir()
    assert (run / "verification").is_dir()
    assert hashlib.sha256(image.read_bytes()).hexdigest() == before
    persisted = "".join(path.read_text(encoding="utf-8") for path in run.rglob("*.json"))
    assert prompt.read_text(encoding="utf-8").strip() not in persisted


def test_plan_refuses_to_replace_an_existing_run(tmp_path: Path) -> None:
    image, mask, clearance, evidence, prompt = _inputs(tmp_path)
    runs = tmp_path / "runs"
    existing = runs / "20260822T090000Z-run-test-1"
    existing.mkdir(parents=True)

    code = main(
        [
            "plan",
            "--image",
            str(image),
            "--mask",
            str(mask),
            "--clearance",
            str(clearance),
            "--evidence",
            str(evidence),
            "--prompt-file",
            str(prompt),
            "--subject",
            "synthetic-person",
            "--out",
            str(runs),
        ],
        now="2026-08-22T09:00:00Z",
        id_factory=lambda: "run-test-1",
    )

    assert code == 73
    assert list(runs.iterdir()) == [existing]


def _planned_run(tmp_path: Path):
    image, mask, clearance, evidence, prompt = _inputs(tmp_path)
    runs = tmp_path / "runs"
    code = main(
        [
            "plan",
            "--image",
            str(image),
            "--mask",
            str(mask),
            "--clearance",
            str(clearance),
            "--evidence",
            str(evidence),
            "--prompt-file",
            str(prompt),
            "--subject",
            "synthetic-person",
            "--out",
            str(runs),
        ],
        now="2026-08-22T09:00:00Z",
        id_factory=lambda: "run-generate-1",
    )
    assert code == 0
    run = next(runs.iterdir())
    plan = json.loads((run / "plan/turntable-plan.json").read_text(encoding="utf-8"))
    return run, plan, evidence, prompt


def _receipts(tmp_path: Path, plan: dict) -> list[Path]:
    paths = []
    for gate in plan["required_gates"]:
        receipt = issue_receipt(
            receipt_id=f"receipt-{gate.replace('_', '-')}-1",
            plan_sha256=plan["plan_sha256"],
            gate=gate,
            acknowledgement=acknowledgement_text(gate, plan["plan_sha256"]),
            disclosure="Synthetic turntable provider test.",
            issued_at="2026-08-22T08:55:00Z",
            expires_at="2026-08-22T10:00:00Z",
        )
        path = tmp_path / f"{gate}.json"
        path.write_text(json.dumps(receipt), encoding="utf-8")
        paths.append(path)
    return paths


def _generated_response(index: int) -> ProviderResponse:
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    ImageDraw.Draw(image).ellipse(
        (290, 190, 734, 834),
        fill=(50 + index * 3, 80 + index * 4, 110 + index * 5),
    )
    output = io.BytesIO()
    image.save(output, format="PNG")
    return ProviderResponse(
        status=200,
        body={
            "data": [{"b64_json": base64.b64encode(output.getvalue()).decode("ascii")}],
            "usage": {"input_tokens": 100 + index, "output_tokens": 200 + index},
            "actual_cost": "0.053000",
        },
        request_id=f"request-{index}",
    )


def test_generate_publishes_an_audited_eight_viewset(tmp_path: Path) -> None:
    run, plan, evidence, prompt = _planned_run(tmp_path)
    receipt_paths = _receipts(tmp_path, plan)
    transport = RecordingTransport([_generated_response(index) for index in range(1, 8)])
    argv = [
        "generate",
        "--run",
        str(run),
        "--evidence",
        str(evidence),
        "--prompt-file",
        str(prompt),
    ]
    for receipt in receipt_paths:
        argv.extend(["--receipt", str(receipt)])

    code = main(
        argv,
        now="2026-08-22T09:05:00Z",
        transport=transport,
        secret_resolver=lambda: "PROVIDER-CREDENTIAL-FOR-TESTS",
    )

    assert code == 0
    viewset = json.loads((run / "viewset/turntable-viewset.json").read_text(encoding="utf-8"))
    assert [item["target_yaw"] for item in viewset["views"]] == [
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
    ]
    assert viewset["audit"]["status"] == "passed"
    assert viewset["actual_cost"] == "0.371000"
    assert (run / "viewset/contact-sheet.png").is_file()
    assert len(list((run / "provider-quarantine").glob("yaw-[0-9][0-9][0-9].png"))) == 7
    assert len(transport.sent) == 7


def _generated_run(tmp_path: Path):
    run, plan, evidence, prompt = _planned_run(tmp_path)
    receipt_paths = _receipts(tmp_path, plan)
    transport = RecordingTransport([_generated_response(index) for index in range(1, 8)])
    argv = [
        "generate",
        "--run",
        str(run),
        "--evidence",
        str(evidence),
        "--prompt-file",
        str(prompt),
    ]
    for receipt in receipt_paths:
        argv.extend(["--receipt", str(receipt)])
    assert (
        main(
            argv,
            now="2026-08-22T09:05:00Z",
            transport=transport,
            secret_resolver=lambda: "PROVIDER-CREDENTIAL-FOR-TESTS",
        )
        == 0
    )
    return run, plan


def test_reconstruct_records_eight_meshes_and_fused_glb(tmp_path: Path) -> None:
    run, plan = _generated_run(tmp_path)
    clearance = tmp_path / "clearance.json"
    clearance.write_text(
        (ROOT / "tests/fixtures/v2/engine-clearance-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_mesh_runner(*, yaw, image_path, mask_path, output_path, **kwargs):
        assert image_path.is_file() and mask_path.is_file()
        output_path.write_bytes(b"mesh" + str(yaw).encode())
        return {
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "triangle_count": 1000 + yaw,
            "vertex_count": 500 + yaw,
            "manifold": "closed",
        }

    def fake_fusion_runner(inputs, output_path, settings):
        assert [item.yaw for item in inputs] == [0, 45, 90, 135, 180, 225, 270, 315]
        output_path.write_bytes(b"glTF" + bytes(128))
        return FusionResult(
            triangle_count=24000,
            vertex_count=12002,
            manifold="closed",
            signed_volume=0.31,
            eligible_mesh_count=8,
            minimum_votes=4,
        )

    code = main(
        [
            "reconstruct",
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
            "--resolution",
            "32",
            "--fusion-grid",
            "48",
        ],
        now="2026-08-22T09:10:00Z",
        mesh_runner=fake_mesh_runner,
        fusion_runner=fake_fusion_runner,
    )

    assert code == 0
    record = json.loads((run / "fusion/multiview-reconstruction.json").read_text(encoding="utf-8"))
    assert len(record["meshes"]) == 8
    assert record["fused_mesh"]["path"] == "fused.glb"
    assert record["disclosure"]["likeness_basis"] == {"views": 8, "inferred": True}
    assert record["disclosure"]["source_image_sha256"] == plan["source_image_sha256"]
    assert (run / "fusion/fused.glb").is_file()


def test_verify_recomputes_artifacts_source_and_preview(tmp_path: Path) -> None:
    run, _plan = _generated_run(tmp_path)
    clearance = tmp_path / "clearance.json"
    clearance.write_text(
        (ROOT / "tests/fixtures/v2/engine-clearance-v1.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    def fake_mesh_runner(*, yaw, output_path, **kwargs):
        output_path.write_bytes(b"mesh" + str(yaw).encode())
        return {
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            "triangle_count": 1000 + yaw,
            "vertex_count": 500 + yaw,
            "manifold": "closed",
        }

    def fake_fusion_runner(inputs, output_path, settings):
        output_path.write_bytes(b"glTF" + bytes(128))
        return FusionResult(24000, 12002, "closed", 0.31, 8, 4)

    assert (
        main(
            [
                "reconstruct",
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
                "--resolution",
                "32",
                "--fusion-grid",
                "48",
            ],
            now="2026-08-22T09:10:00Z",
            mesh_runner=fake_mesh_runner,
            fusion_runner=fake_fusion_runner,
        )
        == 0
    )

    def fake_artifact_validator(path: Path):
        assert path == run / "fusion/fused.glb"
        return {"glb_version": "2.0", "watertight": True, "signed_volume": 0.31}

    def fake_preview_runner(mesh: Path, preview: Path, blender: Path | None):
        assert mesh == run / "fusion/fused.glb"
        Image.new("RGB", (64, 64), (20, 30, 40)).save(preview)

    code = main(
        [
            "verify",
            "--run",
            str(run),
            "--source",
            str(tmp_path / "synthetic-subject.png"),
        ],
        artifact_validator=fake_artifact_validator,
        preview_runner=fake_preview_runner,
    )

    assert code == 0
    report = json.loads((run / "verification/report.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["source_unchanged"] is True
    assert report["mesh_count"] == 8
    assert report["artifact"]["watertight"] is True
    assert (run / "verification/preview.png").is_file()
