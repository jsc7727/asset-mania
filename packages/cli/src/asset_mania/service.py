"""Offline inspect command orchestration independent from stdio."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from asset_mania_contracts import DiagnosticCode, ResultStatus

from asset_mania.environment import inspect_environment
from asset_mania.inspectors.blend import inspect_blend
from asset_mania.inspectors.image import inspect_image
from asset_mania.run import Clock, IdFactory, RunStorageError, create_run_identity, persist_run

Workflow = Literal["image-to-3d", "scene-to-image"]
Kind = Literal["object", "character", "face-head"]

_TOOL_VERSION = "0.1.0"
_IMAGE_MEDIA_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
_CAPABILITIES = {
    "image-to-3d": "not_implemented",
    "scene-to-image": "not_implemented",
}
_FACE_ADVISORY = {
    "code": "FUTURE_FACE_RIGHTS_ADVISORY",
    "message": (
        "Future external or generative face processing requires rights and consent confirmation."
    ),
}


@dataclass(frozen=True, slots=True)
class InspectRequest:
    input_path: Path
    output_parent: Path | None = None
    workflow: Workflow | str | None = None
    kind: Kind | str | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    report: dict[str, object] | None
    primary_diagnostic: str | None
    run_dir: Path | None


def execute_inspect(
    request: InspectRequest,
    *,
    clock: Clock = lambda: datetime.now(UTC),
    id_factory: IdFactory = lambda: secrets.token_hex(4),
) -> CommandResult:
    """Inspect one local source without mutation, network access, or external execution."""
    input_path = Path(request.input_path)
    output_parent = (
        Path(request.output_parent)
        if request.output_parent is not None
        else Path.cwd() / ".asset-mania" / "runs"
    )
    parameters = _resolve_parameters(input_path, request.workflow, request.kind)
    identity = create_run_identity(clock=clock, id_factory=id_factory)

    advisories = [_FACE_ADVISORY] if parameters.get("kind") == "face-head" else []

    try:
        environment, environment_diagnostics = inspect_environment()
        exit_code, primary, inputs, inspection, result_diagnostics, input_warnings = _inspect_input(
            input_path
        )
    except Exception:  # noqa: BLE001 - unexpected inspector failures become a sanitized run
        environment = {}
        environment_diagnostics = []
        exit_code = 4
        primary = DiagnosticCode.INTERNAL_ERROR.value
        inputs = []
        inspection = {}
        result_diagnostics = [DiagnosticCode.INTERNAL_ERROR]
        input_warnings = []
    warnings = sorted(
        {
            *(diagnostic.value for diagnostic in environment_diagnostics),
            *(diagnostic.value for diagnostic in input_warnings),
        }
    )
    manifest = _build_manifest(
        run_id=identity.run_id,
        created_at=identity.created_at,
        inputs=inputs,
        environment=environment,
        parameters=parameters,
        status=ResultStatus.SUCCEEDED if exit_code == 0 else ResultStatus.FAILED,
        diagnostics=result_diagnostics,
        warnings=warnings,
    )
    report = {
        **manifest,
        "advisories": advisories,
        "inspection": inspection,
    }

    try:
        run_dir = persist_run(
            output_parent=output_parent,
            directory_name=identity.directory_name,
            manifest=manifest,
            report=report,
        )
    except RunStorageError:
        return CommandResult(
            exit_code=73,
            report=None,
            primary_diagnostic="OUTPUT_STORAGE_UNAVAILABLE",
            run_dir=None,
        )

    return CommandResult(
        exit_code=exit_code,
        report=report,
        primary_diagnostic=primary,
        run_dir=run_dir,
    )


def _resolve_parameters(
    input_path: Path,
    workflow: Workflow | str | None,
    kind: Kind | str | None,
) -> dict[str, object]:
    if workflow not in {None, "image-to-3d", "scene-to-image"}:
        raise ValueError("workflow must be image-to-3d or scene-to-image")
    if kind not in {None, "object", "character", "face-head"}:
        raise ValueError("kind must be object, character, or face-head")

    suffix = input_path.suffix.lower()
    is_blend = suffix == ".blend"
    is_image = suffix in _IMAGE_MEDIA_TYPES
    resolved_workflow = workflow or ("scene-to-image" if is_blend else "image-to-3d")

    if is_blend and resolved_workflow != "scene-to-image":
        raise ValueError("a .blend input requires scene-to-image")
    if is_image and resolved_workflow != "image-to-3d":
        raise ValueError("an image input requires image-to-3d")
    if resolved_workflow == "scene-to-image" and kind is not None:
        raise ValueError("kind is only valid with image-to-3d")

    if resolved_workflow == "image-to-3d":
        return {"workflow": resolved_workflow, "kind": kind or "object"}
    return {"workflow": resolved_workflow}


def _inspect_input(
    input_path: Path,
) -> tuple[
    int,
    str | None,
    list[dict[str, object]],
    dict[str, object],
    list[DiagnosticCode],
    list[DiagnosticCode],
]:
    try:
        exists = input_path.exists()
    except OSError:
        exists = True
    if not exists:
        diagnostic = DiagnosticCode.INPUT_NOT_FOUND
        return 3, diagnostic.value, [], {}, [diagnostic], []
    if not input_path.is_file():
        diagnostic = DiagnosticCode.INPUT_UNREADABLE
        return 3, diagnostic.value, [], {}, [diagnostic], []

    suffix = input_path.suffix.lower()
    media_type = _IMAGE_MEDIA_TYPES.get(suffix)
    if suffix == ".blend":
        media_type = "application/x-blender"
    elif media_type is None:
        media_type = "application/octet-stream"

    try:
        input_record = _build_input_record(input_path, media_type)
    except OSError:
        diagnostic = DiagnosticCode.INPUT_UNREADABLE
        return 3, diagnostic.value, [], {}, [diagnostic], []

    if suffix in _IMAGE_MEDIA_TYPES:
        inspection, diagnostics = inspect_image(input_path)
    elif suffix == ".blend":
        inspection, diagnostics = inspect_blend(input_path)
    else:
        diagnostic = DiagnosticCode.UNSUPPORTED_MEDIA_TYPE
        return 3, diagnostic.value, [input_record], {}, [diagnostic], []

    fatal = next(
        (
            diagnostic
            for diagnostic in diagnostics
            if diagnostic
            in {
                DiagnosticCode.INPUT_UNREADABLE,
                DiagnosticCode.BLEND_HEADER_INVALID,
            }
        ),
        None,
    )
    if fatal is not None:
        return 3, fatal.value, [input_record], inspection, [fatal], []

    return (
        0,
        None,
        [input_record],
        inspection,
        [DiagnosticCode.WORKFLOW_NOT_IMPLEMENTED],
        diagnostics,
    )


def _build_input_record(path: Path, media_type: str) -> dict[str, object]:
    digest = hashlib.sha256()
    byte_size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return {
        "label": "input-1",
        "sha256": digest.hexdigest(),
        "byte_size": byte_size,
        "media_type": media_type,
    }


def _build_manifest(
    *,
    run_id: str,
    created_at: str,
    inputs: list[dict[str, object]],
    environment: dict[str, object],
    parameters: dict[str, object],
    status: ResultStatus,
    diagnostics: list[DiagnosticCode],
    warnings: list[str],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "command": "inspect",
        "tool_version": _TOOL_VERSION,
        "created_at": created_at,
        "inputs": inputs,
        "environment": environment,
        "parameters": parameters,
        "capabilities": dict(_CAPABILITIES),
        "artifacts": [],
        "result": {
            "status": status.value,
            "diagnostics": sorted(diagnostic.value for diagnostic in diagnostics),
        },
        "warnings": warnings,
    }
