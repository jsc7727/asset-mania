"""Closed v0.2 execution contracts: the schema registry and the canonical builders.

Every value a portable artifact carries is either supplied by the caller and checked
here, or fixed by this module. Nothing is inferred from pixels, geometry, or a local
path, and no builder invents a public field that the normative examples under
`tests/fixtures/v2` do not already contain.
"""

import hashlib
import hmac
import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import ROUND_CEILING, Decimal
from importlib.resources import files
from typing import Any

from .diagnostics import ApprovalGate, DiagnosticCode, ProviderState, ResultStatus
from .models import canonical_json

STAGE_COMMANDS: dict[str, str] = {
    "scene-preflight": "scene.preflight",
    "scene-plan": "scene.plan",
    "provider-evidence": "provider.evidence.refresh",
    "provider-plan": "view.provider-plan",
    "approval-issue": "approval.issue",
    "condition": "scene.condition",
    "view-ingest": "view.ingest",
    "provider-generate": "view.generate",
    "bake": "texture.bake",
    "export": "export",
}
STAGES: list[str] = list(STAGE_COMMANDS)
NULL_PLAN_STAGES: frozenset[str] = frozenset({"scene-preflight", "provider-evidence"})
OFFICIAL_HOST_STAGE = "provider-evidence"

GATES: list[str] = [gate.value for gate in ApprovalGate]
PARENT_RELATIONSHIPS: list[str] = [
    "planned_from",
    "evidenced_from",
    "approved_by",
    "conditioned_from",
    "view_from",
    "generated_from",
    "baked_from",
    "exported_from",
]
ARTIFACT_PARENT_RELATIONSHIPS: list[str] = ["consumed", "derived_from", "generated_from"]
CONTENT_ORIGINS: list[str] = ["observed", "derived", "generated", "unknown"]
SENSITIVITIES: list[str] = ["portable", "user-content", "local-sensitive"]
SUBJECTS: list[str] = ["non_person", "synthetic_person", "real_person", "unknown"]
DECLARABLE_SUBJECTS: list[str] = [subject for subject in SUBJECTS if subject != "unknown"]
ASSET_KINDS: list[str] = ["object", "character", "face_head"]

PROVIDER_STATE_ORDER: list[str] = [
    ProviderState.PLANNED.value,
    ProviderState.APPROVAL_REQUIRED.value,
    ProviderState.APPROVED.value,
    ProviderState.SUBMITTED.value,
    ProviderState.RUNNING.value,
]
TERMINAL_PROVIDER_STATES: frozenset[str] = frozenset(
    {
        ProviderState.SUCCEEDED.value,
        ProviderState.FAILED.value,
        ProviderState.CANCELED.value,
    }
)

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_CONTRACT = 3
EXIT_INTERNAL = 4
EXIT_NEEDS_APPROVAL = 5
EXIT_CANCELED = 6
EXIT_STORAGE = 73
_EXIT_CODES: dict[str, int] = {
    ResultStatus.SUCCEEDED.value: EXIT_SUCCESS,
    ResultStatus.FAILED.value: EXIT_CONTRACT,
    ResultStatus.UNSUPPORTED.value: EXIT_CONTRACT,
    ResultStatus.NEEDS_APPROVAL.value: EXIT_NEEDS_APPROVAL,
    ResultStatus.CANCELED.value: EXIT_CANCELED,
}

EXPECTED_ARTIFACT_ROLES: list[str] = [
    "conditioning_bundle",
    "beauty_exr",
    "beauty_preview",
    "depth_exr",
    "depth_preview",
    "normal_exr",
    "normal_preview",
    "object_index_exr",
    "mask_png",
    "scene_state_blend",
]
PASS_ROLES: list[str] = [
    "beauty_exr",
    "beauty_preview",
    "depth_exr",
    "depth_preview",
    "normal_exr",
    "normal_preview",
    "object_index_exr",
    "mask_png",
]
PASS_MEDIA_TYPES: dict[str, str] = {
    "beauty_exr": "image/x-exr",
    "beauty_preview": "image/png",
    "depth_exr": "image/x-exr",
    "depth_preview": "image/png",
    "normal_exr": "image/x-exr",
    "normal_preview": "image/png",
    "object_index_exr": "image/x-exr",
    "mask_png": "image/png",
}
PASS_COLOR_SPACES: dict[str, str] = {
    "beauty_exr": "scene_linear",
    "beauty_preview": "srgb",
    "depth_exr": "data",
    "depth_preview": "srgb",
    "normal_exr": "data",
    "normal_preview": "srgb",
    "object_index_exr": "data",
    "mask_png": "data",
}

OFFICIAL_SOURCE_HOSTS: list[str] = ["developers.openai.com", "platform.openai.com"]
COST_TABLE_SIZES: list[str] = ["1024x1024", "1024x1536", "1536x1024"]
OUTPUT_COST_ROWS: tuple[tuple[str, str], ...] = tuple(
    (quality, size) for quality in ("low", "medium", "high") for size in COST_TABLE_SIZES
)
_OUTPUT_MEDIA_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

RENDER_PROFILE: dict[str, Any] = {
    "profile_id": "blender-5.2.0-cpu-v1",
    "blender_version": "5.2.0",
    "engine": "CYCLES",
    "device": "CPU",
    "threads": 1,
    "samples": 16,
    "seed": 0,
    "adaptive_sampling": False,
    "denoise": False,
    "animated_seed": False,
    "motion_blur": False,
    "depth_of_field": False,
    "render_border": False,
    "crop_to_border": False,
    "film_transparent": True,
    "pixel_aspect": [1.0, 1.0],
    "working_color_space": "scene_linear",
    "preview_color_space": "srgb",
    "target_object_index": 1,
    "pass_alpha_threshold": 0.5,
    "atlas_size": [1024, 1024],
    "bake_margin": 16,
    "color_padding": 8,
    "minimum_coverage": 0.15,
    "depth_absolute_tolerance_meters": 0.0001,
    "depth_relative_tolerance": 0.0002,
    "ray_epsilon_scale": 1e-07,
    "ray_epsilon_min_meters": 1e-07,
    "ray_epsilon_max_meters": 0.001,
    "matrix_decimal_places": 9,
    "worker_timeout_seconds": 300,
    "worker_response_max_bytes": 1048576,
    "dependency_policy": "packed_only",
    "unknown_texels": "transparent",
    "animation_profile": "selected_action_range_or_none",
}
FIXTURE_RENDER_PROFILE: dict[str, Any] = {
    **RENDER_PROFILE,
    "profile_id": "blender-5.2.0-cpu-v1-fixture",
    "atlas_size": [64, 64],
    "bake_margin": 2,
    "color_padding": 1,
    "minimum_coverage": 0.25,
}

_SCHEMA_FILES: dict[tuple[str, str], str] = {
    ("run-manifest", "1.0"): "manifest-v1.schema.json",
    ("run-manifest", "2.0"): "manifest-v2.schema.json",
    ("workflow-plan", "1.0"): "workflow-plan-v1.schema.json",
    ("provider-evidence", "1.0"): "provider-evidence-v1.schema.json",
    ("provider-plan", "1.0"): "provider-plan-v1.schema.json",
    ("approval-receipt", "1.0"): "approval-receipt-v1.schema.json",
    ("conditioning-bundle", "1.0"): "conditioning-bundle-v1.schema.json",
    ("view", "1.0"): "view-v1.schema.json",
    ("blender-response", "1.0"): "blender-response-v1.schema.json",
    ("engine-clearance", "1.0"): "engine-clearance-v1.schema.json",
    ("reconstruction-plan", "1.0"): "reconstruction-plan-v1.schema.json",
}

#: Every component role a clearance must cover, in the order the schema pins.
CLEARANCE_COMPONENT_ROLES: list[str] = [
    "engine_code",
    "model_weights",
    "preprocessing_model",
]
#: Only `cleared` permits execution. `unknown` fails with `prohibited`, because an unchecked
#: license is not a smaller problem than a forbidding one.
COMMERCIAL_USE_STATES: list[str] = ["cleared", "prohibited", "unknown"]
MESH_FORMATS: list[str] = ["glb", "obj", "ply"]


def schema_names() -> list[tuple[str, str]]:
    """Every registered schema as a sorted `(name, version)` list."""
    return sorted(_SCHEMA_FILES)


def load_schema(name: str, version: str) -> dict[str, Any]:
    """Load one registered schema by its stable name and version.

    The result is a fresh object, so a caller cannot mutate the packaged contract.
    """
    filename = _SCHEMA_FILES[name, version]
    resource = files("asset_mania_contracts").joinpath(f"schema/{filename}")
    return json.loads(resource.read_text(encoding="utf-8"))


def canonical_digest(value: object) -> str:
    """SHA-256 over the canonical JSON encoding of `value`."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _seal(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    preimage = {key: item for key, item in value.items() if key != digest_field}
    return {**preimage, digest_field: canonical_digest(preimage)}


def selection_digest(*, salt: bytes, identity: Mapping[str, Any]) -> str:
    """HMAC-SHA256 of the private selection identity under a random per-plan salt.

    The portable plan carries only this digest and the stable labels, so a label,
    name, library identity, source hash, or target-type change fails verification
    while the digest itself does not leak a datablock name.
    """
    if len(salt) < 16:
        raise ValueError("selection salt must be at least 16 bytes")
    preimage = canonical_json(dict(identity)).encode("utf-8")
    return hmac.new(salt, preimage, hashlib.sha256).hexdigest()


def exit_code_for(result_status: str) -> int:
    """Map a terminal result status to its fixed v2 CLI exit code."""
    try:
        return _EXIT_CODES[result_status]
    except KeyError:
        raise ValueError(f"{result_status!r} is not a v2 result status") from None


def advance_provider_state(current: str, candidate: str) -> str:
    """Advance the provider state along the one declared path.

    A terminal state never transitions again, `succeeded` is reachable only from
    `running`, and every other step must be the immediate successor.
    """
    if current in TERMINAL_PROVIDER_STATES:
        raise ValueError(f"{current!r} is terminal and never transitions again")
    if current not in PROVIDER_STATE_ORDER:
        raise ValueError(f"provider_state {current!r} is not a declared state")

    if candidate == ProviderState.SUCCEEDED.value:
        if current != ProviderState.RUNNING.value:
            raise ValueError(f"provider_state cannot reach {candidate!r} from {current!r}")
        return candidate
    if candidate in TERMINAL_PROVIDER_STATES:
        return candidate

    position = PROVIDER_STATE_ORDER.index(current)
    successor = PROVIDER_STATE_ORDER[position + 1 :][:1]
    if successor != [candidate]:
        raise ValueError(f"provider_state cannot reach {candidate!r} from {current!r}")
    return candidate


def required_gates_for(subject: str) -> list[str]:
    """The gates a provider request needs for a declared subject, in gate order."""
    _require_declared_subject(subject)
    if subject == "real_person":
        return ["face_rights", "external_egress", "paid_compute"]
    return ["external_egress", "paid_compute"]


def _require_declared_subject(subject: str) -> None:
    if subject not in DECLARABLE_SUBJECTS:
        raise ValueError(
            f"{DiagnosticCode.SUBJECT_DECLARATION_REQUIRED.value}: "
            f"{subject!r} is not an executable subject declaration"
        )


def _sorted_codes(codes: Iterable[str]) -> list[str]:
    values = {code.value if isinstance(code, DiagnosticCode) else str(code) for code in codes}
    unknown = values - {code.value for code in DiagnosticCode}
    if unknown:
        raise ValueError(f"unknown diagnostic codes: {sorted(unknown)}")
    return sorted(values)


def ceil_usd(value: Decimal) -> str:
    """Round a USD amount up to the published six-decimal string form."""
    return str(value.quantize(Decimal("0.000001"), rounding=ROUND_CEILING))


def estimate_provider_cost(
    *,
    pricing: Mapping[str, Any],
    quality: str,
    size: str,
    text_input_tokens: int,
    image_input_tokens: int,
) -> str:
    """Preflight cost from uncached input rates plus the one published output row."""
    if size not in COST_TABLE_SIZES:
        raise ValueError(f"{size!r} is not an official cost-table size")
    rates = pricing["per_million_tokens"]
    row = next(
        (
            candidate
            for candidate in pricing["output_cost_rows"]
            if (candidate["quality"], candidate["size"]) == (quality, size)
        ),
        None,
    )
    if row is None:
        raise ValueError(f"no published output cost row for {quality!r} at {size!r}")

    inputs = (
        Decimal(text_input_tokens) * Decimal(rates["text_input"])
        + Decimal(image_input_tokens) * Decimal(rates["image_input"])
    ) / Decimal(1_000_000)
    return ceil_usd(inputs + Decimal(row["usd"]))


def _stage_parameter_keys(stage: str) -> frozenset[str]:
    """The exact parameter key set the schema requires for one stage."""
    schema = load_schema("run-manifest", "2.0")
    for branch in schema["allOf"]:
        if branch["if"]["properties"]["stage"]["const"] == stage:
            return frozenset(branch["then"]["properties"]["parameters"]["required"])
    raise KeyError(stage)


def _input_order(record: Mapping[str, Any]) -> int:
    return int(str(record["label"]).removeprefix("input-"))


def build_manifest_v2(
    *,
    run_id: str,
    stage: str,
    created_at: str,
    tool_version: str,
    inputs: Sequence[Mapping[str, Any]],
    parents: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    plan_sha256: str | None,
    environment: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    approvals: Sequence[Mapping[str, Any]],
    artifacts: Sequence[Mapping[str, Any]],
    result_status: str,
    diagnostics: Iterable[str],
    provider_state: str | None,
    warnings: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Assemble one closed, canonically ordered `manifest-v2` document."""
    if stage not in STAGE_COMMANDS:
        raise ValueError(f"{stage!r} is not a v2 stage")
    if result_status not in _EXIT_CODES:
        raise ValueError(f"{result_status!r} is not a v2 result status")

    expected_keys = _stage_parameter_keys(stage)
    if set(parameters) != expected_keys:
        raise ValueError(f"parameters for stage {stage!r} must be exactly {sorted(expected_keys)}")

    if plan_sha256 is None and stage not in NULL_PLAN_STAGES:
        raise ValueError(f"plan_sha256 may be null only for {sorted(NULL_PLAN_STAGES)}")

    network = capabilities.get("network")
    if network == "explicit_official_hosts" and stage != OFFICIAL_HOST_STAGE:
        raise ValueError(
            f"capabilities.network {network!r} is reserved for the {OFFICIAL_HOST_STAGE!r} stage"
        )

    if provider_state is not None and provider_state not in {
        state.value for state in ProviderState
    }:
        raise ValueError(f"provider_state {provider_state!r} is not a declared state")

    return {
        "schema_id": "asset-mania/run-manifest",
        "schema_version": "2.0",
        "run_id": run_id,
        "command": STAGE_COMMANDS[stage],
        "stage": stage,
        "tool_version": tool_version,
        "created_at": created_at,
        "inputs": sorted((dict(record) for record in inputs), key=_input_order),
        "parents": sorted(
            (dict(record) for record in parents),
            key=lambda record: (record["relationship"], record["run_id"]),
        ),
        "parameters": dict(parameters),
        "plan_sha256": plan_sha256,
        "environment": dict(environment),
        "capabilities": dict(capabilities),
        "approvals": sorted(
            (dict(record) for record in approvals),
            key=lambda record: GATES.index(record["gate"]),
        ),
        "artifacts": sorted(
            (dict(record) for record in artifacts),
            key=lambda record: record["path"],
        ),
        "result": {
            "status": result_status,
            "diagnostics": _sorted_codes(diagnostics),
            "provider_state": provider_state,
        },
        "warnings": _sorted_codes(warnings or []),
    }


def build_workflow_plan(
    *,
    source_scene_sha256: str,
    preflight_manifest_sha256: str,
    selection: Mapping[str, Any],
    asset_kind: str,
    subject: str,
    frame: int,
    action_range: Sequence[int] | None,
    resolution: Sequence[int],
    render_profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the immutable workflow plan for one conditioning run."""
    _require_declared_subject(subject)
    if asset_kind not in ASSET_KINDS:
        raise ValueError(f"asset_kind {asset_kind!r} is not a declared kind")

    if action_range is not None:
        start, end = action_range
        if start > end:
            raise ValueError(f"action_range {list(action_range)} must be ascending")
        if not start <= frame <= end:
            raise ValueError(f"frame {frame} lies outside action_range {list(action_range)}")

    return _seal(
        {
            "schema_id": "asset-mania/workflow-plan",
            "schema_version": "1.0",
            "source_scene_sha256": source_scene_sha256,
            "preflight_manifest_sha256": preflight_manifest_sha256,
            "selection": dict(selection),
            "asset_kind": asset_kind,
            "subject": subject,
            "frame": frame,
            "action_range": list(action_range) if action_range is not None else None,
            "resolution": list(resolution),
            "pixel_aspect": [1.0, 1.0],
            "blender_profile": render_profile["profile_id"],
            "render_profile": dict(render_profile),
            "expected_artifact_roles": list(EXPECTED_ARTIFACT_ROLES),
            "overwrite_policy": "create_only",
            "plan_sha256": "",
        },
        "plan_sha256",
    )


def build_provider_plan(
    *,
    condition_manifest_sha256: str,
    attachments: Sequence[Mapping[str, Any]],
    prompt_sha256: str,
    controls: Mapping[str, Any],
    subject: str,
    policy_evidence: Mapping[str, Any],
    cost_estimate: Mapping[str, Any],
    expected_view: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the immutable provider plan, deriving the gates from the declared subject."""
    _require_declared_subject(subject)

    size = controls["size"]
    if size not in COST_TABLE_SIZES:
        raise ValueError(f"controls.size {size!r} is not an official cost-table size")

    width, height = (int(part) for part in size.split("x"))
    expected_media_type = _OUTPUT_MEDIA_TYPES[controls["output_format"]]
    if (
        expected_view["width"],
        expected_view["height"],
        expected_view["media_type"],
    ) != (width, height, expected_media_type):
        raise ValueError("expected_view must match the resolved controls")

    if (cost_estimate["size"], cost_estimate["quality"]) != (size, controls["quality"]):
        raise ValueError("cost_estimate must match the resolved controls")
    if Decimal(cost_estimate["maximum_cost"]) < Decimal(cost_estimate["estimated_cost"]):
        raise ValueError("maximum_cost must not fall below estimated_cost")

    return _seal(
        {
            "schema_id": "asset-mania/provider-plan",
            "schema_version": "1.0",
            "condition_manifest_sha256": condition_manifest_sha256,
            "provider": "openai",
            "endpoint": "/v1/images/edits",
            "model": "gpt-image-2-2026-04-21",
            "attachments": [dict(record) for record in attachments],
            "prompt_sha256": prompt_sha256,
            "controls": dict(controls),
            "subject": subject,
            "policy_evidence": dict(policy_evidence),
            "cost_estimate": dict(cost_estimate),
            "expected_view": dict(expected_view),
            "required_gates": required_gates_for(subject),
            "overwrite_policy": "create_only",
            "plan_sha256": "",
        },
        "plan_sha256",
    )


def build_engine_clearance(
    *,
    engine: str,
    components: Sequence[Mapping[str, Any]],
    runtime_dependencies: Sequence[Mapping[str, Any]],
    cleared_at: str,
    expires_at: str,
) -> dict[str, Any]:
    """Seal one engine clearance artifact.

    Only a user clears an engine: a maintainer cannot accept a third party's license terms
    on someone else's behalf, so `cleared_by` is fixed rather than accepted.
    """
    roles = [str(item.get("role")) for item in components]
    if roles != CLEARANCE_COMPONENT_ROLES:
        raise ValueError(
            f"clearance components must be exactly {CLEARANCE_COMPONENT_ROLES} in that order"
        )
    if not runtime_dependencies:
        raise ValueError(
            "a clearance with no runtime dependencies is never true for an inference engine"
        )
    if expires_at <= cleared_at:
        raise ValueError("expires_at must fall after cleared_at")

    return _seal(
        {
            "schema_id": "asset-mania/engine-clearance",
            "schema_version": "1.0",
            "engine": engine,
            "components": [dict(item) for item in components],
            "runtime_dependencies": sorted(
                (dict(item) for item in runtime_dependencies),
                key=lambda item: item["name"],
            ),
            "cleared_by": "user",
            "cleared_at": cleared_at,
            "expires_at": expires_at,
            "clearance_sha256": "",
        },
        "clearance_sha256",
    )


def build_reconstruction_plan(
    *,
    engine: str,
    engine_profile: str,
    clearance_sha256: str,
    source_image_sha256: str,
    source_width: int,
    source_height: int,
    alpha: str,
    mask_sha256: str | None,
    background_removal_clearance_sha256: str | None,
    asset_kind: str,
    subject: str,
    rights_receipt_sha256: str | None,
    expected_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal one immutable reconstruction plan.

    A mask or an audited background-removal clearance is mandatory: a single-image
    reconstructor handed a full scene reconstructs the scene, so "no mask" is not a
    permissive default but a different job.
    """
    _require_declared_subject(subject)
    if asset_kind not in ASSET_KINDS:
        raise ValueError(f"asset_kind {asset_kind!r} is not a declared kind")
    if mask_sha256 is None and background_removal_clearance_sha256 is None:
        raise ValueError(
            f"{DiagnosticCode.MASK_REQUIRED.value}: supply a mask or an audited "
            "background-removal clearance"
        )
    if subject == "real_person" and rights_receipt_sha256 is None:
        raise ValueError(
            f"{DiagnosticCode.FACE_RIGHTS_CONFIRMATION_REQUIRED.value}: real_person needs a "
            "plan-bound rights receipt"
        )
    if subject != "real_person" and rights_receipt_sha256 is not None:
        raise ValueError(f"a rights receipt does not apply to subject {subject!r}")
    if expected_output["mesh_format"] not in MESH_FORMATS:
        raise ValueError(f"mesh_format must be one of {MESH_FORMATS}")

    return _seal(
        {
            "schema_id": "asset-mania/reconstruction-plan",
            "schema_version": "1.0",
            "engine": engine,
            "engine_profile": engine_profile,
            "clearance_sha256": clearance_sha256,
            "source_image_sha256": source_image_sha256,
            "source_width": int(source_width),
            "source_height": int(source_height),
            "color_space": "srgb",
            "alpha": alpha,
            "mask_sha256": mask_sha256,
            "background_removal_clearance_sha256": background_removal_clearance_sha256,
            "asset_kind": asset_kind,
            "subject": subject,
            "rights_receipt_sha256": rights_receipt_sha256,
            "expected_output": dict(expected_output),
            "overwrite_policy": "create_only",
            "plan_sha256": "",
        },
        "plan_sha256",
    )


def build_approval_receipt(
    *,
    receipt_id: str,
    plan_sha256: str,
    gate: str,
    issued_at: str,
    expires_at: str,
    disclosure_digest: str,
    acknowledgement_digest: str,
) -> dict[str, Any]:
    """Seal one single-run approval receipt for one gate on one plan digest."""
    if gate not in GATES:
        raise ValueError(f"gate {gate!r} is not a declared approval gate")
    if expires_at <= issued_at:
        raise ValueError("expires_at must fall after issued_at")

    return _seal(
        {
            "schema_id": "asset-mania/approval-receipt",
            "schema_version": "1.0",
            "receipt_id": receipt_id,
            "plan_sha256": plan_sha256,
            "gate": gate,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "scope": "single_run",
            "disclosure_digest": disclosure_digest,
            "issuer_type": "user",
            "acknowledgement_digest": acknowledgement_digest,
            "receipt_sha256": "",
        },
        "receipt_sha256",
    )
