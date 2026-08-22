"""Closed contracts for generated turntables and local multi-view fusion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from typing import Any

from .execution import build_likeness_disclosure, canonical_digest, required_gates_for

TURNTABLE_YAWS: tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)
MODEL_SNAPSHOT = "gpt-image-2-2026-04-21"
ENDPOINT = "/v1/images/edits"
PROMPT_TEMPLATE_REVISION = "turntable-face-v1"
TURNTABLE_CONTROLS: dict[str, Any] = {
    "size": "1024x1024",
    "quality": "medium",
    "background": "opaque",
    "output_format": "png",
    "moderation": "auto",
}

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_MONEY = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{6}$")


def _require_digest(name: str, value: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _require_cost(name: str, value: str) -> Decimal:
    if not isinstance(value, str) or _MONEY.fullmatch(value) is None:
        raise ValueError(f"{name} must be a six-decimal USD string")
    try:
        return Decimal(value)
    except InvalidOperation as error:  # pragma: no cover - regex already narrows the input
        raise ValueError(f"{name} is not decimal") from error


def _seal(value: dict[str, Any], field: str) -> dict[str, Any]:
    preimage = {key: item for key, item in value.items() if key != field}
    return {**preimage, field: canonical_digest(preimage)}


def build_turntable_plan(
    *,
    source_image_sha256: str,
    source_width: int,
    source_height: int,
    source_mask_sha256: str,
    prompt_sha256: str,
    provider_evidence_sha256: str,
    controls: Mapping[str, Any],
    subject: str,
    estimated_cost: str,
    maximum_cost: str,
) -> dict[str, Any]:
    """Seal one seven-call GPT Image 2 plan around an observed yaw-zero source."""
    for name, digest in (
        ("source_image_sha256", source_image_sha256),
        ("source_mask_sha256", source_mask_sha256),
        ("prompt_sha256", prompt_sha256),
        ("provider_evidence_sha256", provider_evidence_sha256),
    ):
        _require_digest(name, digest)
    if not isinstance(source_width, int) or not 1 <= source_width <= 8192:
        raise ValueError("source_width must be an integer in 1..8192")
    if not isinstance(source_height, int) or not 1 <= source_height <= 8192:
        raise ValueError("source_height must be an integer in 1..8192")
    if dict(controls) != TURNTABLE_CONTROLS:
        raise ValueError(f"controls must be exactly {TURNTABLE_CONTROLS}")
    if subject not in ("real_person", "synthetic_person"):
        raise ValueError("face_head turntables require real_person or synthetic_person")
    estimated = _require_cost("estimated_cost", estimated_cost)
    maximum = _require_cost("maximum_cost", maximum_cost)
    if maximum < estimated:
        raise ValueError("maximum_cost must not be below estimated_cost")

    return _seal(
        {
            "schema_id": "asset-mania/turntable-plan",
            "schema_version": "1.0",
            "source_image_sha256": source_image_sha256,
            "source_width": source_width,
            "source_height": source_height,
            "source_mask_sha256": source_mask_sha256,
            "asset_kind": "face_head",
            "subject": subject,
            "provider": "openai",
            "endpoint": ENDPOINT,
            "model": MODEL_SNAPSHOT,
            "yaws": list(TURNTABLE_YAWS),
            "pitch": 0,
            "roll": 0,
            "provider_call_count": 7,
            "prompt_template_revision": PROMPT_TEMPLATE_REVISION,
            "prompt_sha256": prompt_sha256,
            "provider_evidence_sha256": provider_evidence_sha256,
            "controls": dict(controls),
            "cost_estimate": {
                "currency": "USD",
                "estimated_cost": estimated_cost,
                "maximum_cost": maximum_cost,
                "provider_call_count": 7,
            },
            "required_gates": required_gates_for(subject),
            "overwrite_policy": "create_only",
            "plan_sha256": "",
        },
        "plan_sha256",
    )


def build_turntable_viewset(
    *,
    plan_sha256: str,
    views: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    reported_usage: Mapping[str, int | float],
    actual_cost: str | None,
) -> dict[str, Any]:
    """Seal the eight ordered images produced by one turntable plan."""
    _require_digest("plan_sha256", plan_sha256)
    records = [dict(item) for item in views]
    yaws = [item.get("target_yaw") for item in records]
    if yaws != list(TURNTABLE_YAWS):
        raise ValueError(f"view yaws must be exactly {list(TURNTABLE_YAWS)} in order")
    for index, (record, yaw) in enumerate(zip(records, TURNTABLE_YAWS, strict=True), start=1):
        if record.get("label") != f"view-{index}":
            raise ValueError("view labels must be view-1 through view-8 in yaw order")
        expected_origin = "observed" if yaw == 0 else "generated"
        if record.get("origin") != expected_origin:
            raise ValueError(f"yaw {yaw} must have origin {expected_origin!r}")
        if record.get("pitch") != 0 or record.get("roll") != 0:
            raise ValueError("turntable views require zero pitch and roll")
        _require_digest("image_sha256", record.get("image_sha256"))
        _require_digest("mask_sha256", record.get("mask_sha256"))
        if record.get("media_type") != "image/png":
            raise ValueError("turntable views must be PNG")
        if (record.get("width"), record.get("height")) != (1024, 1024):
            raise ValueError("turntable views must be 1024x1024")
        if not isinstance(record.get("byte_size"), int) or record["byte_size"] <= 0:
            raise ValueError("turntable view byte_size must be positive")
        usage = record.get("reported_usage")
        if not isinstance(usage, Mapping) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in usage.values()
        ):
            raise ValueError("reported_usage must contain only numeric values")
        request_id = record.get("provider_request_id")
        if yaw == 0 and (request_id is not None or usage):
            raise ValueError("the observed view has no provider request or usage")
        if yaw != 0 and (not isinstance(request_id, str) or not request_id or not usage):
            raise ValueError("generated views require provider request IDs and usage")

    expected_audit_keys = {"status", "diagnostics", "identity_consistency", "metrics"}
    if set(audit) != expected_audit_keys:
        raise ValueError(f"audit fields must be exactly {sorted(expected_audit_keys)}")
    if audit["status"] not in ("passed", "failed"):
        raise ValueError("audit status must be passed or failed")
    if audit["identity_consistency"] != "unmeasured":
        raise ValueError("identity consistency is never inferred")
    if not isinstance(audit["diagnostics"], list) or not all(
        isinstance(item, str) for item in audit["diagnostics"]
    ):
        raise ValueError("audit diagnostics must be a string list")
    if not isinstance(audit["metrics"], Mapping):
        raise TypeError("audit metrics must be an object")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in reported_usage.values()
    ):
        raise ValueError("aggregate reported_usage must contain only numeric values")
    if actual_cost is not None:
        _require_cost("actual_cost", actual_cost)

    return _seal(
        {
            "schema_id": "asset-mania/turntable-viewset",
            "schema_version": "1.0",
            "plan_sha256": plan_sha256,
            "yaws": list(TURNTABLE_YAWS),
            "views": records,
            "audit": dict(audit),
            "reported_usage": dict(sorted(reported_usage.items())),
            "actual_cost": actual_cost,
            "viewset_sha256": "",
        },
        "viewset_sha256",
    )


def build_multiview_reconstruction_record(
    *,
    turntable_plan_sha256: str,
    viewset_sha256: str,
    observed_source_image_sha256: str,
    meshes: Sequence[Mapping[str, Any]],
    fusion: Mapping[str, Any],
    fused_mesh: Mapping[str, Any],
    subject: str,
    rights_receipt_sha256: str | None,
) -> dict[str, Any]:
    """Seal eight per-view meshes and their local voxel-consensus result."""
    for name, digest in (
        ("turntable_plan_sha256", turntable_plan_sha256),
        ("viewset_sha256", viewset_sha256),
        ("observed_source_image_sha256", observed_source_image_sha256),
    ):
        _require_digest(name, digest)
    if subject not in ("real_person", "synthetic_person"):
        raise ValueError("multiview face output requires a face-capable subject")
    if subject == "real_person":
        _require_digest("rights_receipt_sha256", rights_receipt_sha256)
    elif rights_receipt_sha256 is not None:
        raise ValueError("a synthetic_person record cannot carry a rights receipt")

    mesh_records = [dict(item) for item in meshes]
    yaws = [item.get("target_yaw") for item in mesh_records]
    if yaws != list(TURNTABLE_YAWS):
        raise ValueError(f"mesh yaws must be exactly {list(TURNTABLE_YAWS)} in order")
    closed_count = 0
    for index, record in enumerate(mesh_records, start=1):
        if record.get("label") != f"mesh-{index}":
            raise ValueError("mesh labels must be mesh-1 through mesh-8")
        _require_digest("mesh sha256", record.get("sha256"))
        for field in ("triangle_count", "vertex_count"):
            if not isinstance(record.get(field), int) or record[field] <= 0:
                raise ValueError(f"{field} must be positive")
        if record.get("manifold") not in ("closed", "open", "unknown"):
            raise ValueError("mesh manifold must be closed, open, or unknown")
        closed_count += int(record["manifold"] == "closed")
    if closed_count < 6:
        raise ValueError("multiview fusion requires at least six closed meshes")

    fusion_record = dict(fusion)
    expected_fusion = {
        "normalization",
        "yaw_axis",
        "grid_resolution",
        "minimum_votes",
        "eligible_mesh_count",
        "input_mesh_count",
    }
    if set(fusion_record) != expected_fusion:
        raise ValueError(f"fusion fields must be exactly {sorted(expected_fusion)}")
    if fusion_record["normalization"] != "bounds_center_unit_longest_extent":
        raise ValueError("fusion normalization is outside the profile")
    if fusion_record["yaw_axis"] != "+Z":
        raise ValueError("fusion yaw axis must be +Z")
    if fusion_record["input_mesh_count"] != 8:
        raise ValueError("fusion input_mesh_count must be 8")
    if fusion_record["eligible_mesh_count"] != closed_count:
        raise ValueError("fusion eligible_mesh_count must equal the closed mesh count")
    expected_votes = (closed_count + 1) // 2
    if fusion_record["minimum_votes"] != expected_votes:
        raise ValueError(f"fusion minimum_votes must be {expected_votes}")
    if (
        not isinstance(fusion_record["grid_resolution"], int)
        or not 16 <= fusion_record["grid_resolution"] <= 512
    ):
        raise ValueError("fusion grid_resolution must be an integer in 16..512")

    fused = dict(fused_mesh)
    expected_fused = {
        "role",
        "path",
        "sha256",
        "byte_size",
        "media_type",
        "triangle_count",
        "vertex_count",
        "manifold",
        "signed_volume",
        "content_origin",
        "sensitivity",
        "upload_eligible",
    }
    if set(fused) != expected_fused:
        raise ValueError(f"fused mesh fields must be exactly {sorted(expected_fused)}")
    if (fused["role"], fused["path"], fused["media_type"]) != (
        "fused_mesh",
        "fused.glb",
        "model/gltf-binary",
    ):
        raise ValueError("fused mesh must be the declared neutral GLB")
    _require_digest("fused mesh sha256", fused["sha256"])
    if (
        fused["manifold"] != "closed"
        or not isinstance(fused["signed_volume"], (int, float))
        or isinstance(fused["signed_volume"], bool)
        or fused["signed_volume"] <= 0
    ):
        raise ValueError("fused mesh must be closed and positive-volume")
    for field in ("byte_size", "triangle_count", "vertex_count"):
        if not isinstance(fused[field], int) or fused[field] <= 0:
            raise ValueError(f"fused mesh {field} must be positive")
    if (
        fused["content_origin"] != "generated"
        or fused["sensitivity"] != "user-content"
        or fused["upload_eligible"] is not False
    ):
        raise ValueError("fused mesh privacy and provenance fields are fixed")

    disclosure = build_likeness_disclosure(
        plan_sha256=turntable_plan_sha256,
        source_image_sha256=observed_source_image_sha256,
        mesh_sha256=fused["sha256"],
        subject=subject,
        rights_receipt_sha256=rights_receipt_sha256,
        engine="triposr-local",
        engine_profile="triposr-voxel-consensus-v1",
        views=8,
    )
    return _seal(
        {
            "schema_id": "asset-mania/multiview-reconstruction",
            "schema_version": "1.0",
            "turntable_plan_sha256": turntable_plan_sha256,
            "viewset_sha256": viewset_sha256,
            "observed_source_image_sha256": observed_source_image_sha256,
            "meshes": mesh_records,
            "fusion": fusion_record,
            "fused_mesh": fused,
            "disclosure": disclosure,
            "record_sha256": "",
        },
        "record_sha256",
    )


__all__ = [
    "ENDPOINT",
    "MODEL_SNAPSHOT",
    "PROMPT_TEMPLATE_REVISION",
    "TURNTABLE_CONTROLS",
    "TURNTABLE_YAWS",
    "build_multiview_reconstruction_record",
    "build_turntable_plan",
    "build_turntable_viewset",
]
