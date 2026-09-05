#!/usr/bin/env python3
"""Run the private, rights-gated MICA plus DECA clay geometry evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
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
    FaceGeometryData,
    authorize_conditioning,
    build_face_geometry_plugin_request,
    export_clay_glb,
    fingerprint_source,
    fit_similarity_transform,
    fuse_mica_deca_geometry,
    load_face_geometry,
    run_face_geometry_plugin,
    sha256_file,
    validate_local_face_standing_consent,
    verify_source_unchanged,
    write_face_geometry_plugin_request,
)
from PIL import Image, ImageDraw

PROFILE = "mica-deca-clay-face-v1"
_PLAN_BINDINGS = frozenset(
    {
        "mica_python_sha256",
        "mica_plugin_sha256",
        "mica_detector_sha256",
        "deca_python_sha256",
        "deca_plugin_sha256",
        "deca_detector_sha256",
    }
)
_SETTING_SUFFIXES = (
    "SOURCE_ROOT",
    "ISOLATED_HOME",
    "CHECKPOINT_PATH",
    "FLAME_PATH",
    "FLAME_SHA256",
    "DETECTOR_PATH",
    "DETECTOR_SHA256",
)


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


def _verify_file_digest(path: Path, expected: str, label: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"{label} digest mismatch")


def _directory_sha256(directory: Path) -> str:
    if not directory.is_dir():
        raise ValueError("detector model directory is unavailable")
    digest = hashlib.sha256()
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("detector model directory inventory is invalid")
    for path in files:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _require_current_plan(plan: dict) -> None:
    missing = sorted(_PLAN_BINDINGS.difference(plan))
    if missing:
        raise ValueError(
            "face geometry plan lacks sealed stage bindings; create a new geometry plan"
        )


def _authorization_record(
    *, plan_sha256: str, mode: str, receipt_sha256: str, consumption: dict | None = None
) -> dict:
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-authorization",
        "schema_version": "0.1",
        "plan_sha256": plan_sha256,
        "authorization_mode": mode,
        "receipt_sha256": receipt_sha256,
    }
    if consumption is not None:
        preimage["consumption"] = consumption
    return {**preimage, "authorization_sha256": canonical_digest(preimage)}


def _verify_authorization(document: dict, plan: dict) -> None:
    common = {
        "schema_id",
        "schema_version",
        "plan_sha256",
        "authorization_mode",
        "receipt_sha256",
        "authorization_sha256",
    }
    allowed = common | ({"consumption"} if "consumption" in document else set())
    if set(document) != allowed:
        raise ValueError("MICA authorization record contains fields outside the allowlist")
    _verify_seal(document, "authorization_sha256", "MICA authorization record")
    if (
        document.get("schema_id") != "asset-mania/private-face-geometry-authorization"
        or document.get("schema_version") != "0.1"
    ):
        raise ValueError("MICA authorization record schema mismatch")
    if document.get("plan_sha256") != plan["plan_sha256"]:
        raise ValueError("MICA authorization record plan mismatch")
    expected_mode = (
        "standing_local_source_consent_v1"
        if plan.get("authorization_mode") == "standing_local_source_consent_v1"
        else "single_receipt_v1"
    )
    if document.get("authorization_mode") != expected_mode:
        raise ValueError("MICA authorization mode does not match the sealed plan")
    receipt_sha256 = document.get("receipt_sha256")
    if (
        not isinstance(receipt_sha256, str)
        or len(receipt_sha256) != 64
        or any(char not in "0123456789abcdef" for char in receipt_sha256)
    ):
        raise ValueError("MICA authorization receipt digest is invalid")
    if expected_mode == "standing_local_source_consent_v1":
        if "consumption" in document or receipt_sha256 != plan.get("standing_consent_sha256"):
            raise ValueError("MICA standing authorization does not match the sealed plan")
    else:
        consumption = document.get("consumption")
        if not isinstance(consumption, dict) or set(consumption) != {
            "gate",
            "receipt_sha256",
            "consumption_id",
            "consumed_at",
        }:
            raise ValueError("MICA receipt consumption record is invalid")
        if (
            consumption.get("gate") != "face_rights"
            or consumption.get("receipt_sha256") != receipt_sha256
        ):
            raise ValueError("MICA receipt consumption record does not match authorization")


def _verify_plugin_record(document: dict, label: str) -> None:
    expected = {
        "schema_id",
        "schema_version",
        "profile",
        "plan_sha256",
        "plugin",
        "plugin_profile",
        "checkpoint_sha256",
        "python_sha256",
        "plugin_executable_sha256",
        "authorization_sha256",
        "source_unchanged",
        "geometry_sha256",
        "vertex_count",
        "triangle_count",
        "persisted_identity_feature_count",
        "record_sha256",
    }
    if set(document) != expected:
        raise ValueError(f"{label} contains fields outside the allowlist")
    _verify_seal(document, "record_sha256", label)


def _validated_plugin_environment(plugin: str, plan: dict) -> dict[str, str]:
    label = "MICA" if plugin == "mica-local" else "DECA"
    prefix = f"ASSET_MANIA_{label}_"
    allowed_names = {f"{prefix}{suffix}" for suffix in _SETTING_SUFFIXES}
    unknown = sorted(
        name for name in os.environ if name.startswith(prefix) and name not in allowed_names
    )
    if unknown:
        raise ValueError(f"unknown {label} plugin environment variable")
    missing = sorted(allowed_names.difference(os.environ))
    if missing:
        raise ValueError(f"{label} plugin environment is incomplete: {missing[0]}")
    environment = _plugin_environment(plugin)
    checkpoint = Path(environment[f"{prefix}CHECKPOINT_PATH"]).resolve(strict=True)
    flame = Path(environment[f"{prefix}FLAME_PATH"]).resolve(strict=True)
    detector = Path(environment[f"{prefix}DETECTOR_PATH"]).resolve(strict=True)
    stage = label.lower()
    _verify_file_digest(checkpoint, plan[f"{stage}_checkpoint_sha256"], f"{label} checkpoint")
    if environment[f"{prefix}FLAME_SHA256"] != plan["flame_sha256"]:
        raise ValueError(f"{label} FLAME environment digest mismatch")
    _verify_file_digest(flame, plan["flame_sha256"], f"{label} FLAME asset")
    if environment[f"{prefix}DETECTOR_SHA256"] != plan[f"{stage}_detector_sha256"]:
        raise ValueError(f"{label} detector environment digest mismatch")
    if not detector.is_dir():
        raise ValueError(f"{label} detector path must be a directory")
    if _directory_sha256(detector) != plan[f"{stage}_detector_sha256"]:
        raise ValueError(f"{label} detector asset digest mismatch")
    return environment


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
    for label in (
        "topology_sha256",
        "mica_checkpoint_sha256",
        "mica_detector_sha256",
        "deca_checkpoint_sha256",
        "deca_detector_sha256",
        "flame_sha256",
    ):
        digest = getattr(arguments, label)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{label.replace('_', ' ')} is invalid")
    if arguments.mica_detector_sha256 != arguments.deca_detector_sha256:
        raise ValueError("MICA and DECA must bind the same sealed detector directory")
    consent = None
    if arguments.standing_consent is not None:
        consent = validate_local_face_standing_consent(
            _load(arguments.standing_consent, "standing consent"),
            source_sha256=source_digest,
        )
    topology = arguments.topology.resolve(strict=True)
    if sha256_file(topology) != arguments.topology_sha256:
        raise ValueError("topology digest mismatch")
    executable_digests = {}
    for stage in ("mica", "deca"):
        for kind in ("python", "plugin"):
            path = getattr(arguments, f"{stage}_{kind}").resolve(strict=True)
            executable_digests[f"{stage}_{kind}_sha256"] = sha256_file(path)
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
        "mica_detector_sha256": arguments.mica_detector_sha256,
        "deca_revision": arguments.deca_revision,
        "deca_checkpoint_sha256": arguments.deca_checkpoint_sha256,
        "deca_detector_sha256": arguments.deca_detector_sha256,
        **executable_digests,
        "gates": {
            "minimum_head_extent_metres": 0.15,
            "maximum_head_extent_metres": 0.32,
            "deca_extent_validation": "positive-finite-prealignment-then-similarity-fit",
            "maximum_displacement_metres": 0.003,
            "maximum_rms_displacement_metres": 0.0015,
            "minimum_face_displacement_coverage": 0.90,
            "persisted_identity_feature_count": 0,
        },
        "geometry_sources": ["authorized_observed_front"],
        "overwrite_policy": "create_only",
    }
    if consent is not None:
        preimage.update(
            {
                "authorization_mode": "standing_local_source_consent_v1",
                "standing_consent_sha256": consent["consent_sha256"],
            }
        )
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
    allowed = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
    plugin_settings = {f"{prefix}{suffix}" for suffix in _SETTING_SUFFIXES}
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed or key in plugin_settings
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
    _require_current_plan(plan)
    stage = run / stage_name
    record_path = stage / "record.json"
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite {record_path}")
    if plugin_name == "mica-local":
        if arguments.standing_consent is not None:
            if plan.get("authorization_mode") != "standing_local_source_consent_v1":
                raise ValueError("sealed plan does not select standing consent")
            consent = validate_local_face_standing_consent(
                _load(arguments.standing_consent, "standing consent"),
                source_sha256=plan["source_image_sha256"],
            )
            if consent["consent_sha256"] != plan.get("standing_consent_sha256"):
                raise ValueError("standing consent differs from the sealed plan")
            receipt = None
            authorization = _authorization_record(
                plan_sha256=plan["plan_sha256"],
                mode="standing_local_source_consent_v1",
                receipt_sha256=consent["consent_sha256"],
            )
        else:
            if "standing_consent_sha256" in plan:
                raise ValueError("sealed plan requires standing consent")
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
            authorization = _authorization_record(
                plan_sha256=plan["plan_sha256"],
                mode="single_receipt_v1",
                receipt_sha256=receipt["receipt_sha256"],
                consumption=consumption,
            )
        (stage / "authorization.json").write_text(canonical_json(authorization), encoding="utf-8")
    else:
        authorization_path = run / "mica/authorization.json"
        if not authorization_path.is_file():
            raise ValueError("DECA requires the consumed MICA authorization record")
        authorization = _load(authorization_path, "MICA authorization record")
        _verify_authorization(authorization, plan)
        mica_record = _load(run / "mica/record.json", "MICA record")
        _verify_plugin_record(mica_record, "MICA record")
        if (
            mica_record.get("plan_sha256") != plan["plan_sha256"]
            or mica_record.get("plugin") != "mica-local"
            or mica_record.get("checkpoint_sha256") != plan["mica_checkpoint_sha256"]
            or mica_record.get("python_sha256") != plan["mica_python_sha256"]
            or mica_record.get("plugin_executable_sha256") != plan["mica_plugin_sha256"]
            or mica_record.get("authorization_sha256") != authorization["authorization_sha256"]
        ):
            raise ValueError("MICA record does not match the sealed plan and authorization")
        _verify_file_digest(
            run / "mica/plugin-output/geometry.npz",
            mica_record["geometry_sha256"],
            "MICA geometry",
        )
        if plan.get("authorization_mode") == "standing_local_source_consent_v1":
            if arguments.standing_consent is None:
                raise ValueError("DECA requires the standing consent selected by the sealed plan")
            consent = validate_local_face_standing_consent(
                _load(arguments.standing_consent, "standing consent"),
                source_sha256=plan["source_image_sha256"],
            )
            if consent["consent_sha256"] != plan.get("standing_consent_sha256"):
                raise ValueError("standing consent differs from the sealed plan")
        receipt = None
    stage_key = "mica" if plugin_name == "mica-local" else "deca"
    python = arguments.python.resolve(strict=True)
    plugin = arguments.plugin.resolve(strict=True)
    python_sha256 = sha256_file(python)
    plugin_executable_sha256 = sha256_file(plugin)
    if python_sha256 != plan[f"{stage_key}_python_sha256"]:
        raise ValueError("Python runtime digest does not match the sealed plan")
    if plugin_executable_sha256 != plan[f"{stage_key}_plugin_sha256"]:
        raise ValueError("plugin executable digest does not match the sealed plan")
    environment = _validated_plugin_environment(plugin_name, plan)
    source = arguments.source.resolve(strict=True)
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
            _load(run / "mica/authorization.json", "authorization")["receipt_sha256"]
        ),
    )
    request_path = stage / "request.json"
    result_path = stage / "result.json"
    write_face_geometry_plugin_request(request, request_path)
    result = plugin_runner(
        [str(python), str(plugin)],
        request,
        request_path,
        result_path,
        timeout_seconds=300,
        environment=environment,
    )
    _verify_file_digest(python, python_sha256, "Python runtime")
    _verify_file_digest(plugin, plugin_executable_sha256, "plugin executable")
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
        "python_sha256": python_sha256,
        "plugin_executable_sha256": plugin_executable_sha256,
        "authorization_sha256": _load(run / "mica/authorization.json", "MICA authorization record")[
            "authorization_sha256"
        ],
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
    _require_current_plan(plan)
    output = run / "fusion"
    record_path = output / "record.json"
    if record_path.exists():
        raise FileExistsError("refusing to overwrite face geometry fusion")
    mica_record = _load(run / "mica/record.json", "MICA record")
    deca_record = _load(run / "deca/record.json", "DECA record")
    _verify_plugin_record(mica_record, "MICA record")
    _verify_plugin_record(deca_record, "DECA record")
    authorization = _load(run / "mica/authorization.json", "MICA authorization record")
    _verify_authorization(authorization, plan)
    for stage_name, record, plugin_name in (
        ("mica", mica_record, "mica-local"),
        ("deca", deca_record, "deca-local"),
    ):
        if (
            record.get("plan_sha256") != plan["plan_sha256"]
            or record.get("plugin") != plugin_name
            or record.get("checkpoint_sha256") != plan[f"{stage_name}_checkpoint_sha256"]
            or record.get("python_sha256") != plan[f"{stage_name}_python_sha256"]
            or record.get("plugin_executable_sha256") != plan[f"{stage_name}_plugin_sha256"]
            or record.get("authorization_sha256") != authorization["authorization_sha256"]
        ):
            raise ValueError(f"{stage_name.upper()} record does not match sealed plan bindings")
    _verify_file_digest(
        run / "mica/plugin-output/geometry.npz",
        mica_record["geometry_sha256"],
        "MICA geometry",
    )
    _verify_file_digest(
        run / "deca/plugin-output/geometry.npz",
        deca_record["geometry_sha256"],
        "DECA geometry",
    )
    if sha256_file(run / "topology.npz") != plan["topology_sha256"]:
        raise ValueError("topology digest mismatch")
    faces, face_indices, inner = _topology(run)
    mica = load_face_geometry(run / "mica/plugin-output/geometry.npz", expected_topology=faces)
    deca = load_face_geometry(
        run / "deca/plugin-output/geometry.npz",
        expected_topology=faces,
        validate_extent=False,
    )
    if np.any(mica.detail_displacement != 0.0):
        raise ValueError("MICA detail displacement must be zero")
    fused, measurements = fuse_mica_deca_geometry(
        mica=mica, deca=deca, face_indices=face_indices, inner_face_indices=inner
    )
    deca_transform = fit_similarity_transform(deca.vertices[inner], mica.vertices[inner])
    comparison_deca = FaceGeometryData(
        vertices=deca_transform.apply(deca.vertices),
        faces=deca.faces.copy(),
        source_projection=deca.source_projection.copy(),
        detail_displacement=deca.detail_displacement * deca_transform.scale,
    )
    outputs = {
        "mica": (mica, output / "mica-clay.glb", "mica-local", "identity-neutral-v1"),
        "deca": (
            comparison_deca,
            output / "deca-clay.glb",
            "deca-local",
            "detail-displacement-v1",
        ),
        "fusion": (
            fused,
            output / "mica-deca-clay.glb",
            "mica-deca-local",
            "identity-plus-bounded-detail-v1",
        ),
    }
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
        sys.executable,
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
        "90",
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
    plan = _load(run / "plan.json", "face geometry plan")
    _verify_seal(plan, "plan_sha256", "face geometry plan")
    _require_current_plan(plan)
    authorization = _load(run / "mica/authorization.json", "MICA authorization record")
    _verify_authorization(authorization, plan)
    for stage_name, plugin_name in (("mica", "mica-local"), ("deca", "deca-local")):
        stage_record = _load(run / stage_name / "record.json", f"{stage_name.upper()} record")
        _verify_plugin_record(stage_record, f"{stage_name.upper()} record")
        if (
            stage_record.get("plan_sha256") != plan["plan_sha256"]
            or stage_record.get("plugin") != plugin_name
            or stage_record.get("checkpoint_sha256") != plan[f"{stage_name}_checkpoint_sha256"]
            or stage_record.get("python_sha256") != plan[f"{stage_name}_python_sha256"]
            or stage_record.get("plugin_executable_sha256") != plan[f"{stage_name}_plugin_sha256"]
            or stage_record.get("authorization_sha256") != authorization["authorization_sha256"]
        ):
            raise ValueError(f"{stage_name.upper()} record does not match sealed plan bindings")
    fusion = _load(run / "fusion/record.json", "fusion record")
    _verify_seal(fusion, "record_sha256", "fusion record")
    verification = run / "verification"
    report_path = verification / "report.json"
    if report_path.exists():
        raise FileExistsError("refusing to overwrite face geometry verification")
    blender = arguments.blender.resolve(strict=True)
    blender_sha256 = sha256_file(blender)
    fusion_meshes = {
        "mica": run / "fusion/mica-clay.glb",
        "deca": run / "fusion/deca-clay.glb",
        "fusion": run / "fusion/mica-deca-clay.glb",
    }
    for label, mesh in fusion_meshes.items():
        _verify_file_digest(
            mesh,
            fusion["artifacts"][label]["mesh_sha256"],
            "fusion artifact",
        )
    dad_baseline = arguments.dad_baseline.resolve(strict=True)
    dad_baseline_sha256 = sha256_file(dad_baseline)
    rows = [
        ("MICA identity clay", fusion_meshes["mica"], verification / "mica.png"),
        ("DECA coarse clay", fusion_meshes["deca"], verification / "deca.png"),
        ("MICA + DECA clay", fusion_meshes["fusion"], verification / "fusion.png"),
        (
            "Corrected DAD clay",
            dad_baseline,
            verification / "dad.png",
        ),
    ]
    for _label, mesh, output in rows:
        preview_runner(mesh, output, blender)
    _verify_file_digest(blender, blender_sha256, "Blender executable")
    _verify_file_digest(dad_baseline, dad_baseline_sha256, "DAD baseline")
    for label, mesh in fusion_meshes.items():
        _verify_file_digest(
            mesh,
            fusion["artifacts"][label]["mesh_sha256"],
            "fusion artifact",
        )
    comparison = verification / "comparison.png"
    _comparison([(label, output) for label, _mesh, output in rows], comparison)
    preimage = {
        "schema_id": "asset-mania/private-face-geometry-verification",
        "schema_version": "0.1",
        "profile": PROFILE,
        "fusion_record_sha256": fusion["record_sha256"],
        "blender_sha256": blender_sha256,
        "dad_baseline_sha256": dad_baseline_sha256,
        "comparison_sha256": sha256_file(comparison),
        "visual_quality": "unreviewed",
        "status": "passed",
    }
    report = {**preimage, "report_sha256": canonical_digest(preimage)}
    report_path.write_text(canonical_json(report), encoding="utf-8")
    return 0


def _run_review(arguments: argparse.Namespace) -> int:
    run = arguments.run.resolve(strict=True)
    report_path = run / "verification/report.json"
    report = _load(report_path, "verification report")
    _verify_seal(report, "report_sha256", "verification report")
    _verify_file_digest(
        run / "verification/comparison.png",
        report["comparison_sha256"],
        "comparison",
    )
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
        "verification_sha256": report["report_sha256"],
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
    plan.add_argument("--mica-detector-sha256", required=True)
    plan.add_argument("--mica-python", type=Path, required=True)
    plan.add_argument("--mica-plugin", type=Path, required=True)
    plan.add_argument("--deca-revision", required=True)
    plan.add_argument("--deca-checkpoint-sha256", required=True)
    plan.add_argument("--deca-detector-sha256", required=True)
    plan.add_argument("--deca-python", type=Path, required=True)
    plan.add_argument("--deca-plugin", type=Path, required=True)
    plan.add_argument("--flame-sha256", required=True)
    plan.add_argument("--standing-consent", type=Path)
    for name in ("mica-run", "deca-run"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--python", type=Path, required=True)
        command.add_argument("--plugin", type=Path, required=True)
        if name == "mica-run":
            authorization = command.add_mutually_exclusive_group(required=True)
            authorization.add_argument("--rights-store", type=Path)
            authorization.add_argument("--standing-consent", type=Path)
        else:
            command.add_argument("--standing-consent", type=Path)
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
