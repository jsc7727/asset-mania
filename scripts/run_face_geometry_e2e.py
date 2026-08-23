#!/usr/bin/env python3
"""Run the private, rights-gated MICA plus DECA clay geometry evaluation."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from asset_mania_contracts import (
    build_likeness_disclosure,
    canonical_digest,
    canonical_json,
)
from asset_mania_pipeline import (
    ConsumptionJournal,
    authorize_conditioning,
    build_face_geometry_plugin_request,
    export_clay_glb,
    fingerprint_source,
    fuse_mica_deca_geometry,
    load_face_geometry,
    run_face_geometry_plugin,
    sha256_file,
    verify_source_unchanged,
    write_face_geometry_plugin_request,
)
from PIL import Image, ImageDraw

PROFILE = "mica-deca-clay-face-v1"


def _parse_time(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC).replace(microsecond=0)
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromisoformat(value).astimezone(UTC).replace(microsecond=0)


def _load(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _verify_seal(document: dict, field: str, label: str) -> None:
    preimage = {key: value for key, value in document.items() if key != field}
    if canonical_digest(preimage) != document.get(field):
        raise ValueError(f"{label} digest does not match its content")


def _create_run(output: Path, timestamp: datetime, run_id: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    run = output / f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{run_id}"
    run.mkdir()
    for name in ("mica", "deca", "fusion", "verification"):
        (run / name).mkdir()
    return run


def _run_plan(arguments: argparse.Namespace, *, now: datetime, run_id: str) -> int:
    source_digest = arguments.source_sha256.lower()
    if len(source_digest) != 64 or any(char not in "0123456789abcdef" for char in source_digest):
        raise ValueError("source SHA-256 is invalid")
    topology = arguments.topology.resolve(strict=True)
    if sha256_file(topology) != arguments.topology_sha256:
        raise ValueError("topology digest mismatch")
    run = _create_run(arguments.output.resolve(), now, run_id)
    copied_topology = run / "topology.npz"
    shutil.copyfile(topology, copied_topology)
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-plan",
        "schema_version": "0.1",
        "profile": PROFILE,
        "asset_kind": "face_head",
        "subject": "real_person",
        "required_gate": "face_rights",
        "source_image_sha256": source_digest,
        "topology_sha256": arguments.topology_sha256,
        "flame_sha256": arguments.flame_sha256,
        "plugins": [
            {"plugin": "mica-local", "profile": "identity-neutral-v1"},
            {"plugin": "deca-local", "profile": "detail-displacement-v1"},
        ],
        "mica_revision": arguments.mica_revision,
        "mica_checkpoint_sha256": arguments.mica_checkpoint_sha256,
        "deca_revision": arguments.deca_revision,
        "deca_checkpoint_sha256": arguments.deca_checkpoint_sha256,
        "gates": {
            "maximum_displacement_metres": 0.003,
            "maximum_rms_displacement_metres": 0.0015,
            "minimum_face_displacement_coverage": 0.90,
            "persisted_identity_feature_count": 0,
        },
        "geometry_sources": ["authorized_observed_front"],
        "overwrite_policy": "create_only",
    }
    plan = {**preimage, "plan_sha256": canonical_digest(preimage)}
    (run / "plan.json").write_text(canonical_json(plan), encoding="utf-8")
    print(canonical_json({"run_directory": str(run), "plan_sha256": plan["plan_sha256"]}))
    return 0


def _matching_receipt(store: Path, plan_sha256: str) -> dict:
    matches = []
    for path in sorted(store.glob("*.json")):
        value = _load(path, "rights receipt")
        if value.get("plan_sha256") == plan_sha256 and value.get("gate") == "face_rights":
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("rights store must contain exactly one matching face_rights receipt")
    return matches[0]


def _plugin_environment(plugin: str) -> dict[str, str]:
    prefix = "ASSET_MANIA_MICA_" if plugin == "mica-local" else "ASSET_MANIA_DECA_"
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed or key.startswith(prefix)
    }


def _plugin_stage(
    arguments: argparse.Namespace,
    *,
    plugin_name: str,
    profile: str,
    stage_name: str,
    plugin_runner: Callable,
    now: datetime,
) -> int:
    run = arguments.run.resolve(strict=True)
    plan = _load(run / "plan.json", "face geometry plan")
    _verify_seal(plan, "plan_sha256", "face geometry plan")
    stage = run / stage_name
    record_path = stage / "record.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite {record_path}")
    source = arguments.source.resolve(strict=True)
    if plugin_name == "mica-local":
        rights_store = arguments.rights_store.resolve(strict=True)
        receipt = _matching_receipt(rights_store, plan["plan_sha256"])
        consumption = authorize_conditioning(
            subject="real_person",
            plan_sha256=plan["plan_sha256"],
            receipt=receipt,
            journal=ConsumptionJournal(rights_store / "consumed"),
            consumption_id=f"{run.name}:mica-run",
            consumed_at=now.isoformat(),
            now=now,
        )
        (stage / "authorization.json").write_text(canonical_json(consumption), encoding="utf-8")
    else:
        authorization = run / "mica/authorization.json"
        if not authorization.is_file():
            raise ValueError("DECA requires the consumed MICA authorization record")
        receipt = None
    before = fingerprint_source(source)
    if before.sha256 != plan["source_image_sha256"]:
        raise ValueError("private source differs from the sealed plan")
    revision = plan["mica_revision"] if plugin_name == "mica-local" else plan["deca_revision"]
    checkpoint = (
        plan["mica_checkpoint_sha256"]
        if plugin_name == "mica-local"
        else plan["deca_checkpoint_sha256"]
    )
    request = build_face_geometry_plugin_request(
        plugin=plugin_name,
        profile=profile,
        plugin_revision=revision,
        source_image=source,
        output_directory=stage / "plugin-output",
        device="cuda",
        checkpoint_sha256=checkpoint,
        topology="flame-2020-5023",
        face_rights_receipt_sha256=(
            receipt["receipt_sha256"]
            if receipt is not None
            else _load(run / "mica/authorization.json", "authorization")["receipt_sha256"]
        ),
    )
    request_path = stage / "request.json"
    result_path = stage / "result.json"
    write_face_geometry_plugin_request(request, request_path)
    result = plugin_runner(
        [str(arguments.python.resolve(strict=True)), str(arguments.plugin.resolve(strict=True))],
        request,
        request_path,
        result_path,
        timeout_seconds=300,
        environment=_plugin_environment(plugin_name),
    )
    verify_source_unchanged(source, before)
    geometry = result.geometry
    if geometry is None:
        raise ValueError("face geometry plugin produced no geometry")
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-plugin-record",
        "schema_version": "0.1",
        "profile": PROFILE,
        "plan_sha256": plan["plan_sha256"],
        "plugin": plugin_name,
        "plugin_profile": profile,
        "checkpoint_sha256": checkpoint,
        "source_unchanged": True,
        "geometry_sha256": sha256_file(geometry),
        "vertex_count": result.vertex_count,
        "triangle_count": result.triangle_count,
        "persisted_identity_feature_count": result.persisted_identity_feature_count,
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    record_path.write_text(canonical_json(record), encoding="utf-8")
    return 0


def _topology(run: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(run / "topology.npz", allow_pickle=False) as archive:
            if set(archive.files) != {"faces", "face_indices", "inner_face_indices"}:
                raise ValueError("topology inventory is invalid")
            return (
                np.asarray(archive["faces"], dtype=np.int64).copy(),
                np.asarray(archive["face_indices"], dtype=np.int64).copy(),
                np.asarray(archive["inner_face_indices"], dtype=np.int64).copy(),
            )
    except (OSError, KeyError, ValueError) as error:
        if isinstance(error, ValueError) and "inventory" in str(error):
            raise
        raise ValueError("topology archive is unreadable") from error


def _run_fuse(arguments: argparse.Namespace) -> int:
    run = arguments.run.resolve(strict=True)
    plan = _load(run / "plan.json", "face geometry plan")
    _verify_seal(plan, "plan_sha256", "face geometry plan")
    output = run / "fusion"
    record_path = output / "record.json"
    if record_path.exists():
        raise FileExistsError("refusing to overwrite face geometry fusion")
    mica_record = _load(run / "mica/record.json", "MICA record")
    deca_record = _load(run / "deca/record.json", "DECA record")
    _verify_seal(mica_record, "record_sha256", "MICA record")
    _verify_seal(deca_record, "record_sha256", "DECA record")
    faces, face_indices, inner = _topology(run)
    mica = load_face_geometry(run / "mica/plugin-output/geometry.npz", expected_topology=faces)
    deca = load_face_geometry(run / "deca/plugin-output/geometry.npz", expected_topology=faces)
    fused, measurements = fuse_mica_deca_geometry(
        mica=mica, deca=deca, face_indices=face_indices, inner_face_indices=inner
    )
    outputs = {
        "mica": (mica, output / "mica-clay.glb", "mica-local", "identity-neutral-v1"),
        "deca": (deca, output / "deca-clay.glb", "deca-local", "detail-displacement-v1"),
        "fusion": (
            fused,
            output / "mica-deca-clay.glb",
            "mica-deca-local",
            "identity-plus-bounded-detail-v1",
        ),
    }
    authorization = _load(run / "mica/authorization.json", "authorization")
    artifacts = {}
    for label, (data, path, engine, profile) in outputs.items():
        measured = export_clay_glb(data, path)
        digest = sha256_file(path)
        disclosure = build_likeness_disclosure(
            plan_sha256=plan["plan_sha256"],
            source_image_sha256=plan["source_image_sha256"],
            mesh_sha256=digest,
            subject="real_person",
            rights_receipt_sha256=authorization["receipt_sha256"],
            engine=engine,
            engine_profile=profile,
            views=1,
        )
        disclosure_path = output / f"likeness-{label}.json"
        disclosure_path.write_text(canonical_json(disclosure), encoding="utf-8")
        artifacts[label] = {
            "mesh_sha256": digest,
            "disclosure_sha256": disclosure["disclosure_sha256"],
            "measurements": asdict(measured),
        }
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-fusion",
        "schema_version": "0.1",
        "profile": PROFILE,
        "plan_sha256": plan["plan_sha256"],
        "mica_record_sha256": mica_record["record_sha256"],
        "deca_record_sha256": deca_record["record_sha256"],
        "fusion_measurements": asdict(measurements),
        "artifacts": artifacts,
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    record_path.write_text(canonical_json(record), encoding="utf-8")
    return 0


def _default_preview(mesh: Path, output: Path, blender: Path) -> None:
    command = [
        os.fspath(Path(__file__).parents[1] / ".venv" / "Scripts" / "python.exe"),
        os.fspath(Path(__file__).with_name("blender_preview.py")),
        "render",
        "--blender",
        os.fspath(blender),
        "--mesh",
        os.fspath(mesh),
        "--out",
        os.fspath(output),
        "--samples",
        "16",
        "--resolution",
        "500",
        "--views",
        "8",
        "--elevation",
        "0.15",
        "--orbit-axis",
        "Z",
        "--start-angle-degrees",
        "-90",
    ]
    completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode != 0:
        raise ValueError(f"Blender clay preview failed with exit {completed.returncode}")


def _comparison(rows: Sequence[tuple[str, Path]], output: Path) -> None:
    opened = []
    try:
        for label, path in rows:
            image = Image.open(path).convert("RGB")
            opened.append((label, image))
        width = max(image.width for _label, image in opened)
        row_height = max(image.height for _label, image in opened)
        sheet = Image.new("RGB", (width, row_height * len(opened)), (30, 30, 32))
        draw = ImageDraw.Draw(sheet)
        for index, (label, image) in enumerate(opened):
            y = index * row_height
            sheet.paste(image, (0, y))
            draw.rectangle((0, y, width, y + 18), fill=(20, 20, 22))
            draw.text((5, y + 3), label, fill=(235, 235, 235))
        sheet.save(output)
    finally:
        for _label, image in opened:
            image.close()


def _run_verify(arguments: argparse.Namespace, *, preview_runner: Callable) -> int:
    run = arguments.run.resolve(strict=True)
    fusion = _load(run / "fusion/record.json", "fusion record")
    _verify_seal(fusion, "record_sha256", "fusion record")
    verification = run / "verification"
    report_path = verification / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite face geometry verification")
    blender = arguments.blender.resolve(strict=True)
    rows = [
        ("MICA identity clay", run / "fusion/mica-clay.glb", verification / "mica.png"),
        ("DECA coarse clay", run / "fusion/deca-clay.glb", verification / "deca.png"),
        ("MICA + DECA clay", run / "fusion/mica-deca-clay.glb", verification / "fusion.png"),
        (
            "Corrected DAD clay",
            arguments.dad_baseline.resolve(strict=True),
            verification / "dad.png",
        ),
    ]
    for _label, mesh, output in rows:
        preview_runner(mesh, output, blender)
    comparison = verification / "comparison.png"
    _comparison([(label, output) for label, _mesh, output in rows], comparison)
    report = {
        "schema_id": "asset-mania/private-face-geometry-verification",
        "schema_version": "0.1",
        "profile": PROFILE,
        "fusion_record_sha256": fusion["record_sha256"],
        "comparison_sha256": sha256_file(comparison),
        "visual_quality": "unreviewed",
        "status": "passed",
    }
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return 0


def _run_review(arguments: argparse.Namespace) -> int:
    run = arguments.run.resolve(strict=True)
    report_path = run / "verification/report.json"
    report = _load(report_path, "verification report")
    if report.get("status") != "passed" or report.get("visual_quality") != "unreviewed":
        raise ValueError("verification is not awaiting manual review")
    reason = arguments.reason.strip()
    if not reason or len(reason) > 500 or "\n" in reason or "\r" in reason:
        raise ValueError("manual review reason must be one line of 1 to 500 characters")
    output = run / "verification/manual-review.json"
    if output.exists():
        raise FileExistsError("refusing to overwrite manual review")
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-manual-review",
        "schema_version": "0.1",
        "profile": PROFILE,
        "verification_sha256": sha256_file(report_path),
        "comparison_sha256": report["comparison_sha256"],
        "visual_quality": arguments.verdict,
        "reason": reason,
        "review_basis": "untextured-front-and-eight-view-clay-comparison",
    }
    review = {**preimage, "review_sha256": canonical_digest(preimage)}
    output.write_text(canonical_json(review), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("geometry-plan")
    plan.add_argument("--out", type=Path, dest="output", required=True)
    plan.add_argument("--source-sha256", required=True)
    plan.add_argument("--topology", type=Path, required=True)
    plan.add_argument("--topology-sha256", required=True)
    plan.add_argument("--mica-revision", required=True)
    plan.add_argument("--mica-checkpoint-sha256", required=True)
    plan.add_argument("--deca-revision", required=True)
    plan.add_argument("--deca-checkpoint-sha256", required=True)
    plan.add_argument("--flame-sha256", required=True)
    for name in ("mica-run", "deca-run"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--python", type=Path, required=True)
        command.add_argument("--plugin", type=Path, required=True)
        if name == "mica-run":
            command.add_argument("--rights-store", type=Path, required=True)
    fuse = commands.add_parser("geometry-fuse")
    fuse.add_argument("--run", type=Path, required=True)
    verify = commands.add_parser("geometry-verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--blender", type=Path, required=True)
    verify.add_argument("--dad-baseline", type=Path, required=True)
    review = commands.add_parser("geometry-review")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--verdict", choices=("passed", "failed"), required=True)
    review.add_argument("--reason", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    now: str | datetime | None = None,
    id_factory: Callable[[], str] | None = None,
    plugin_runner: Callable | None = None,
    preview_runner: Callable | None = None,
) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    timestamp = _parse_time(now)
    if arguments.command == "geometry-plan":
        return _run_plan(
            arguments,
            now=timestamp,
            run_id=(id_factory or (lambda: secrets.token_hex(4)))(),
        )
    if arguments.command == "mica-run":
        return _plugin_stage(
            arguments,
            plugin_name="mica-local",
            profile="identity-neutral-v1",
            stage_name="mica",
            plugin_runner=plugin_runner or run_face_geometry_plugin,
            now=timestamp,
        )
    if arguments.command == "deca-run":
        return _plugin_stage(
            arguments,
            plugin_name="deca-local",
            profile="detail-displacement-v1",
            stage_name="deca",
            plugin_runner=plugin_runner or run_face_geometry_plugin,
            now=timestamp,
        )
    if arguments.command == "geometry-fuse":
        return _run_fuse(arguments)
    if arguments.command == "geometry-verify":
        return _run_verify(arguments, preview_runner=preview_runner or _default_preview)
    return _run_review(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
