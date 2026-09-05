#!/usr/bin/env python3
"""Plan, generate, reconstruct, and verify one private turntable multi-view run."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from asset_mania_contracts import (
    TURNTABLE_CONTROLS,
    build_multiview_reconstruction_record,
    build_turntable_plan,
    canonical_digest,
    canonical_json,
)
from asset_mania_engine_triposr import (
    FusionSettings,
    YawMesh,
    fuse_turntable_meshes,
)
from asset_mania_engine_triposr.adapter import EngineRequest
from asset_mania_engine_triposr.ports.triposr import TripoSRPort, TripoSRSettings
from asset_mania_pipeline import (
    ConsumptionJournal,
    TurntableCandidate,
    acknowledgement_text,
    audit_turntable,
    derive_white_background_mask,
    parse_subject,
    prepare_turntable_source,
    publish_turntable_viewset,
    run_if_cleared,
    sha256_bytes,
    sha256_file,
    validate_glb,
    verify_engine_clearance,
    write_contact_sheet,
)
from asset_mania_provider_openai import HTTPSMultipartTransport, generate_turntable
from asset_mania_provider_openai.client import verify_evidence_freshness
from asset_mania_provider_openai.transport import (
    SecretResolver,
    Transport,
)

RUN_DIRECTORIES = (
    "plan",
    "provider-quarantine",
    "viewset",
    "per-view-meshes",
    "fusion",
    "verification",
)


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(UTC).replace(microsecond=0)


def _directory_name(now: datetime, run_id: str) -> str:
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{run_id}"


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _aggregate_output_cost(evidence: dict) -> str:
    row = next(
        item
        for item in evidence["pricing"]["output_cost_rows"]
        if item["quality"] == "medium" and item["size"] == "1024x1024"
    )
    return str((Decimal(row["usd"]) * 7).quantize(Decimal("0.000001")))


def _run_plan(arguments: argparse.Namespace, *, now: datetime, run_id: str) -> int:
    clearance = _load_object(arguments.clearance, "engine clearance")
    evidence = _load_object(arguments.evidence, "provider evidence")
    verify_engine_clearance(clearance, engine="triposr-local", now=now)
    verify_evidence_freshness(evidence, now=now)
    if evidence.get("model_snapshot") != "gpt-image-2-2026-04-21":
        raise ValueError("provider evidence names another model snapshot")
    try:
        prompt = arguments.prompt_file.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("prompt file is unreadable") from error
    if not prompt.strip():
        raise ValueError("prompt file is empty")
    subject = parse_subject(arguments.subject)

    arguments.output_parent.mkdir(parents=True, exist_ok=True)
    run_directory = arguments.output_parent / _directory_name(now, run_id)
    try:
        run_directory.mkdir()
    except FileExistsError:
        return 73
    for relative in RUN_DIRECTORIES:
        (run_directory / relative).mkdir()
    prepared = prepare_turntable_source(
        image_path=arguments.image,
        mask_path=arguments.mask,
        staging_root=run_directory / "plan",
    )
    estimated_cost = _aggregate_output_cost(evidence)
    plan = build_turntable_plan(
        source_image_sha256=prepared["source_image_sha256"],
        source_width=prepared["width"],
        source_height=prepared["height"],
        source_mask_sha256=prepared["source_mask_sha256"],
        source_cutout_sha256=prepared["cutout_sha256"],
        prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
        provider_evidence_sha256=evidence["evidence_sha256"],
        controls=TURNTABLE_CONTROLS,
        subject=subject,
        estimated_cost=estimated_cost,
        maximum_cost="0.700000",
    )
    (run_directory / "plan/turntable-plan.json").write_text(canonical_json(plan), encoding="utf-8")
    approval_request = {
        "plan_sha256": plan["plan_sha256"],
        "required_gates": plan["required_gates"],
        "acknowledgements": [
            acknowledgement_text(gate, plan["plan_sha256"]) for gate in plan["required_gates"]
        ],
    }
    (run_directory / "plan/approval-request.json").write_text(
        canonical_json(approval_request), encoding="utf-8"
    )
    print(canonical_json({"run_directory": str(run_directory), "plan": plan}))
    return 0


def _environment_secret() -> str:
    return os.environ.get("OPENAI_API_KEY", "")


def _run_generate(
    arguments: argparse.Namespace,
    *,
    now: datetime,
    transport: Transport | None,
    secret_resolver: SecretResolver,
) -> int:
    run_directory = arguments.run.resolve(strict=True)
    plan = _load_object(run_directory / "plan/turntable-plan.json", "turntable plan")
    evidence = _load_object(arguments.evidence, "provider evidence")
    receipts = [_load_object(path, "approval receipt") for path in arguments.receipt]
    try:
        prompt = arguments.prompt_file.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("prompt file is unreadable") from error
    selected_transport = transport or HTTPSMultipartTransport()
    results = generate_turntable(
        plan=plan,
        evidence=evidence,
        receipts=receipts,
        base_prompt=prompt,
        cutout_path=run_directory / "plan/source-cutout.png",
        journal=ConsumptionJournal(run_directory / "plan/approval-journal"),
        now=now,
        consumed_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        transport=selected_transport,
        secret_resolver=secret_resolver,
    )

    candidates = [
        TurntableCandidate(
            yaw=0,
            origin="observed",
            image_path=run_directory / "plan/reconstruction-input.png",
            mask_path=run_directory / "plan/reconstruction-mask.png",
            provider_request_id=None,
            reported_usage={},
            actual_cost=None,
        )
    ]
    quarantine = run_directory / "provider-quarantine"
    for result in results:
        image_path = quarantine / f"yaw-{result.yaw:03d}.png"
        mask_path = quarantine / f"yaw-{result.yaw:03d}-mask.png"
        if image_path.exists() or mask_path.exists():
            raise ValueError(f"refusing to overwrite quarantine yaw {result.yaw}")
        image_path.write_bytes(result.image_bytes)
        derive_white_background_mask(image_path, mask_path)
        candidates.append(
            TurntableCandidate(
                yaw=result.yaw,
                origin="generated",
                image_path=image_path,
                mask_path=mask_path,
                provider_request_id=result.request_id,
                reported_usage=result.reported_usage,
                actual_cost=result.actual_cost,
            )
        )

    audit = audit_turntable(candidates)
    (run_directory / "viewset/audit.json").write_text(canonical_json(audit), encoding="utf-8")
    if audit["status"] != "passed":
        return 3
    write_contact_sheet(candidates, run_directory / "viewset/contact-sheet.png")
    costs = [Decimal(result.actual_cost) for result in results if result.actual_cost is not None]
    actual_cost = (
        str(sum(costs, start=Decimal(0)).quantize(Decimal("0.000001")))
        if len(costs) == len(results)
        else None
    )
    viewset = publish_turntable_viewset(
        plan_sha256=plan["plan_sha256"],
        candidates=candidates,
        audit=audit,
        actual_cost=actual_cost,
    )
    (run_directory / "viewset/turntable-viewset.json").write_text(
        canonical_json(viewset), encoding="utf-8"
    )
    (run_directory / "plan/approval-receipts.json").write_text(
        canonical_json({"receipts": receipts}), encoding="utf-8"
    )
    print(canonical_json({"run_directory": str(run_directory), "viewset": viewset}))
    return 0


def _verify_seal(document: dict, field: str, label: str) -> None:
    preimage = {key: value for key, value in document.items() if key != field}
    if canonical_digest(preimage) != document.get(field):
        raise ValueError(f"{label} digest does not match its content")


def _private_view_paths(run_directory: Path, yaw: int) -> tuple[Path, Path]:
    if yaw == 0:
        return (
            run_directory / "plan/reconstruction-input.png",
            run_directory / "plan/reconstruction-mask.png",
        )
    return (
        run_directory / f"provider-quarantine/yaw-{yaw:03d}.png",
        run_directory / f"provider-quarantine/yaw-{yaw:03d}-mask.png",
    )


def _run_reconstruct(
    arguments: argparse.Namespace,
    *,
    now: datetime,
    mesh_runner,
    fusion_runner,
) -> int:
    run_directory = arguments.run.resolve(strict=True)
    plan = _load_object(run_directory / "plan/turntable-plan.json", "turntable plan")
    viewset = _load_object(run_directory / "viewset/turntable-viewset.json", "turntable viewset")
    clearance = _load_object(arguments.clearance, "engine clearance")
    _verify_seal(plan, "plan_sha256", "turntable plan")
    _verify_seal(viewset, "viewset_sha256", "turntable viewset")
    if viewset.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("viewset belongs to another turntable plan")
    clearance_sha256 = verify_engine_clearance(clearance, engine="triposr-local", now=now)

    selected_mesh_runner = mesh_runner
    if selected_mesh_runner is None:
        port = TripoSRPort(
            settings=TripoSRSettings(
                engine_root=arguments.engine_root,
                weights_dir=arguments.weights,
                device="cpu",
                mc_resolution=arguments.resolution,
                mc_threshold=25.0,
                hub_cache=arguments.hub_cache,
            )
        )

        def selected_mesh_runner(*, yaw, image_path, mask_path, output_path, **kwargs):
            def execute(verified_digest: str):
                return port.run(
                    EngineRequest(
                        engine="triposr-local",
                        profile="triposr-local-cpu-v1",
                        plan_sha256=plan["plan_sha256"],
                        clearance_sha256=verified_digest,
                        image_path=image_path,
                        mask_path=mask_path,
                        output_path=output_path,
                        mesh_format="glb",
                    )
                )

            _, result = run_if_cleared(
                clearance=clearance,
                engine="triposr-local",
                now=now,
                run=execute,
            )
            return {
                "sha256": sha256_file(output_path),
                "triangle_count": result.triangle_count,
                "vertex_count": result.vertex_count,
                "manifold": result.manifold,
            }

    mesh_records = []
    yaw_meshes = []
    for index, view in enumerate(viewset["views"], start=1):
        yaw = int(view["target_yaw"])
        image_path, mask_path = _private_view_paths(run_directory, yaw)
        if sha256_file(image_path) != view["image_sha256"]:
            raise ValueError(f"image for yaw {yaw} differs from the audited viewset")
        if sha256_file(mask_path) != view["mask_sha256"]:
            raise ValueError(f"mask for yaw {yaw} differs from the audited viewset")
        output_path = run_directory / f"per-view-meshes/yaw-{yaw:03d}.glb"
        if output_path.exists():
            raise ValueError(f"refusing to overwrite mesh yaw {yaw}")
        result = selected_mesh_runner(
            yaw=yaw,
            image_path=image_path,
            mask_path=mask_path,
            output_path=output_path,
            plan=plan,
            clearance=clearance,
            clearance_sha256=clearance_sha256,
            resolution=arguments.resolution,
        )
        mesh_records.append(
            {
                "label": f"mesh-{index}",
                "target_yaw": yaw,
                "sha256": result["sha256"],
                "triangle_count": int(result["triangle_count"]),
                "vertex_count": int(result["vertex_count"]),
                "manifold": result["manifold"],
            }
        )
        yaw_meshes.append(YawMesh(yaw=yaw, path=output_path, sha256=result["sha256"]))

    fused_path = run_directory / "fusion/fused.glb"
    settings = FusionSettings(grid_resolution=arguments.fusion_grid)
    selected_fusion_runner = fusion_runner or fuse_turntable_meshes
    fusion_result = selected_fusion_runner(yaw_meshes, fused_path, settings)
    closed_count = sum(record["manifold"] == "closed" for record in mesh_records)
    rights_receipt_sha256 = None
    if plan["subject"] == "real_person":
        approvals = _load_object(
            run_directory / "plan/approval-receipts.json", "private approval receipts"
        )
        face = next(
            receipt for receipt in approvals["receipts"] if receipt["gate"] == "face_rights"
        )
        rights_receipt_sha256 = face["receipt_sha256"]
    record = build_multiview_reconstruction_record(
        turntable_plan_sha256=plan["plan_sha256"],
        viewset_sha256=viewset["viewset_sha256"],
        observed_source_image_sha256=plan["source_image_sha256"],
        meshes=mesh_records,
        fusion={
            "normalization": "bounds_center_unit_longest_extent",
            "yaw_axis": "+Z",
            "grid_resolution": arguments.fusion_grid,
            "minimum_votes": fusion_result.minimum_votes,
            "eligible_mesh_count": closed_count,
            "input_mesh_count": 8,
        },
        fused_mesh={
            "role": "fused_mesh",
            "path": "fused.glb",
            "sha256": sha256_file(fused_path),
            "byte_size": fused_path.stat().st_size,
            "media_type": "model/gltf-binary",
            "triangle_count": fusion_result.triangle_count,
            "vertex_count": fusion_result.vertex_count,
            "manifold": fusion_result.manifold,
            "signed_volume": fusion_result.signed_volume,
            "content_origin": "generated",
            "sensitivity": "user-content",
            "upload_eligible": False,
        },
        subject=plan["subject"],
        rights_receipt_sha256=rights_receipt_sha256,
    )
    (run_directory / "fusion/multiview-reconstruction.json").write_text(
        canonical_json(record), encoding="utf-8"
    )
    print(canonical_json({"run_directory": str(run_directory), "reconstruction": record}))
    return 0


def _default_artifact_validator(path: Path) -> dict:
    container = validate_glb(path)
    import trimesh

    mesh = trimesh.load(str(path), process=False, force="mesh")
    return {
        "glb_version": container.json_chunk["asset"]["version"],
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "signed_volume": float(mesh.volume),
        "triangle_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
    }


def _find_blender(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(os.environ["BLENDER"]) if os.environ.get("BLENDER") else None,
        Path(found) if (found := shutil.which("blender")) else None,
        Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise ValueError("Blender 5.2 was not found for verification")


def _default_preview_runner(mesh: Path, preview: Path, blender: Path | None) -> None:
    executable = _find_blender(blender)
    command = [
        sys.executable,
        str(ROOT / "scripts/blender_preview.py"),
        "render",
        "--blender",
        str(executable),
        "--mesh",
        str(mesh),
        "--out",
        str(preview),
        "--samples",
        "16",
        "--resolution",
        "400",
        "--views",
        "4",
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValueError(f"Blender preview failed with exit {completed.returncode}")


def _run_verify(arguments: argparse.Namespace, *, artifact_validator, preview_runner) -> int:
    run_directory = arguments.run.resolve(strict=True)
    plan = _load_object(run_directory / "plan/turntable-plan.json", "turntable plan")
    viewset = _load_object(run_directory / "viewset/turntable-viewset.json", "turntable viewset")
    record = _load_object(
        run_directory / "fusion/multiview-reconstruction.json",
        "multiview reconstruction",
    )
    _verify_seal(plan, "plan_sha256", "turntable plan")
    _verify_seal(viewset, "viewset_sha256", "turntable viewset")
    _verify_seal(record, "record_sha256", "multiview reconstruction")
    if viewset["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("viewset plan binding is invalid")
    if record["turntable_plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("reconstruction plan binding is invalid")
    if record["viewset_sha256"] != viewset["viewset_sha256"]:
        raise ValueError("reconstruction viewset binding is invalid")
    source_unchanged = sha256_file(arguments.source) == plan["source_image_sha256"]
    if not source_unchanged:
        raise ValueError("the observed source changed after planning")
    for mesh in record["meshes"]:
        path = run_directory / f"per-view-meshes/yaw-{int(mesh['target_yaw']):03d}.glb"
        if sha256_file(path) != mesh["sha256"]:
            raise ValueError(f"mesh yaw {mesh['target_yaw']} differs from its record")
    fused_path = run_directory / "fusion/fused.glb"
    if sha256_file(fused_path) != record["fused_mesh"]["sha256"]:
        raise ValueError("fused GLB differs from its record")
    disclosure = record["disclosure"]
    if (
        disclosure["likeness_basis"] != {"views": 8, "inferred": True}
        or disclosure["source_image_sha256"] != plan["source_image_sha256"]
        or disclosure["mesh_sha256"] != record["fused_mesh"]["sha256"]
    ):
        raise ValueError("likeness disclosure does not describe this source and fused mesh")
    artifact = artifact_validator(fused_path)
    if not artifact.get("watertight") or float(artifact.get("signed_volume", 0)) <= 0:
        raise ValueError("fused GLB is not watertight and positive-volume")
    preview_path = run_directory / "verification/preview.png"
    if preview_path.exists():
        raise ValueError("refusing to overwrite verification preview")
    preview_runner(fused_path, preview_path, arguments.blender)
    if not preview_path.is_file():
        raise ValueError("preview runner produced no image")
    report = {
        "status": "passed",
        "source_unchanged": True,
        "mesh_count": len(record["meshes"]),
        "artifact": artifact,
        "plan_sha256": plan["plan_sha256"],
        "viewset_sha256": viewset["viewset_sha256"],
        "record_sha256": record["record_sha256"],
        "preview_sha256": sha256_file(preview_path),
    }
    report_path = run_directory / "verification/report.json"
    if report_path.exists():
        raise ValueError("refusing to overwrite verification report")
    report_path.write_text(canonical_json(report), encoding="utf-8")
    print(canonical_json({"run_directory": str(run_directory), "verification": report}))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--image", type=Path, required=True)
    plan.add_argument("--mask", type=Path, required=True)
    plan.add_argument("--clearance", type=Path, required=True)
    plan.add_argument("--evidence", type=Path, required=True)
    plan.add_argument("--prompt-file", type=Path, required=True)
    plan.add_argument(
        "--subject",
        choices=("real-person", "synthetic-person"),
        required=True,
    )
    plan.add_argument("--out", dest="output_parent", type=Path, required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--run", type=Path, required=True)
    generate.add_argument("--evidence", type=Path, required=True)
    generate.add_argument("--prompt-file", type=Path, required=True)
    generate.add_argument("--receipt", action="append", type=Path, required=True)
    reconstruct = commands.add_parser("reconstruct")
    reconstruct.add_argument("--run", type=Path, required=True)
    reconstruct.add_argument("--clearance", type=Path, required=True)
    reconstruct.add_argument("--engine-root", type=Path, required=True)
    reconstruct.add_argument("--weights", type=Path, required=True)
    reconstruct.add_argument("--hub-cache", type=Path, required=True)
    reconstruct.add_argument("--resolution", type=int, choices=(32, 128, 256), default=128)
    reconstruct.add_argument("--fusion-grid", type=int, choices=(48, 96, 192), default=96)
    verify = commands.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--source", type=Path, required=True)
    verify.add_argument("--blender", type=Path, default=None)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    now: str | None = None,
    id_factory: Callable[[], str] = lambda: secrets.token_hex(4),
    transport: Transport | None = None,
    secret_resolver: SecretResolver = _environment_secret,
    mesh_runner=None,
    fusion_runner=None,
    artifact_validator=_default_artifact_validator,
    preview_runner=_default_preview_runner,
) -> int:
    parser = _build_parser()
    try:
        arguments = parser.parse_args(list(argv) if argv is not None else None)
        current = _parse_time(now)
        if arguments.command == "plan":
            return _run_plan(arguments, now=current, run_id=id_factory())
        if arguments.command == "generate":
            return _run_generate(
                arguments,
                now=current,
                transport=transport,
                secret_resolver=secret_resolver,
            )
        if arguments.command == "reconstruct":
            return _run_reconstruct(
                arguments,
                now=current,
                mesh_runner=mesh_runner,
                fusion_runner=fusion_runner,
            )
        if arguments.command == "verify":
            return _run_verify(
                arguments,
                artifact_validator=artifact_validator,
                preview_runner=preview_runner,
            )
    except (ValueError, KeyError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
