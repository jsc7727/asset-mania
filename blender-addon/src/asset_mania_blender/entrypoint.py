# SPDX-License-Identifier: GPL-3.0-or-later
"""The single `--python` entry point Blender executes.

Blender passes worker arguments after a bare `--`. This module reads the private request
envelope, dispatches one operation, and always writes a closed response, so the Apache
client never has to parse Blender's stdout or stderr.

The source file is opened here, from the envelope -- never from the process arguments --
with the UI and script execution disabled. Write surfaces are sanitized immediately after
the open and before any evaluation.
"""

import sys
import traceback
from pathlib import Path

from . import protocol

_WORKER_ARGUMENT_SEPARATOR = "--"
_INTERNAL_ERROR = "INTERNAL_ERROR"
_BLEND_HEADER_INVALID = "BLEND_HEADER_INVALID"
_NOT_IMPLEMENTED = "WORKFLOW_NOT_IMPLEMENTED"


def parse_worker_arguments(argv: list[str]) -> dict[str, str]:
    """Read `--request` and `--response` from the arguments after Blender's own."""
    if _WORKER_ARGUMENT_SEPARATOR in argv:
        argv = argv[argv.index(_WORKER_ARGUMENT_SEPARATOR) + 1 :]

    arguments: dict[str, str] = {}
    remaining = list(argv)
    while remaining:
        flag = remaining.pop(0)
        if flag not in ("--request", "--response"):
            raise ValueError("the worker accepts only --request and --response")
        if not remaining:
            raise ValueError(f"{flag} requires a value")
        arguments[flag.removeprefix("--")] = remaining.pop(0)

    missing = {"request", "response"} - set(arguments)
    if missing:
        raise ValueError(f"the worker requires {sorted(missing)}")
    return arguments


def _open_source(source_path: str) -> None:
    """Open the source read-only with the UI and script execution disabled."""
    import bpy

    bpy.ops.wm.open_mainfile(
        filepath=source_path,
        load_ui=False,
        use_scripts=False,
        display_file_selector=False,
    )


def _run_preflight(request: dict) -> dict:
    from . import scene_inventory

    request_id = str(request["request_id"])
    staging_root = str(request["staging_root"])

    try:
        _open_source(str(request["source_path"]))
    except (RuntimeError, OSError):
        return protocol.failure(
            request_id=request_id, operation="preflight", diagnostics=[_BLEND_HEADER_INVALID]
        )

    sanitized = scene_inventory.sanitize_write_surfaces(staging_root)
    metrics, diagnostics, inventory = scene_inventory.preflight(request)

    if metrics is None:
        response = protocol.failure(
            request_id=request_id, operation="preflight", diagnostics=diagnostics
        )
        response["portable_labels"] = inventory["portable_labels"]
        return response

    return {
        "schema_id": protocol.SCHEMA_ID,
        "schema_version": protocol.SCHEMA_VERSION,
        "request_id": request_id,
        "operation": "preflight",
        "status": "succeeded",
        "diagnostics": [],
        "portable_labels": inventory["portable_labels"],
        "outputs": [],
        "metrics": metrics,
        "response_sha256": "",
        "_sanitized": sanitized,
    }


def _run_fixture(request: dict) -> dict:
    """Generate the runtime fixture into staging. Test-only, never a user-facing stage."""
    from . import fixture_factory, fixture_variants

    request_id = str(request["request_id"])
    destination = Path(str(request["staging_root"])) / str(
        request.get("fixture_name", "fixture.blend")
    )
    variant = request.get("variant")
    if variant:
        description = fixture_variants.write_variant(
            variant=str(variant),
            path=str(destination),
            sentinel_path=request.get("sentinel_path"),
        )
    else:
        description = fixture_factory.write_fixture(str(destination))

    return {
        "schema_id": protocol.SCHEMA_ID,
        "schema_version": protocol.SCHEMA_VERSION,
        "request_id": request_id,
        "operation": "validate",
        "status": "succeeded",
        "diagnostics": [],
        "portable_labels": [],
        "outputs": [],
        "metrics": {
            "kind": "validate",
            "profile": str(description["profile"]),
            "checked_artifact_count": 1,
            "error_count": 0,
            "warning_count": 0,
            "semantic_digest": protocol.canonical_digest(description),
        },
        "response_sha256": "",
        "_fixture": description,
    }


def _run_condition(request: dict) -> dict:
    from . import conditioning, scene_inventory

    request_id = str(request["request_id"])
    staging_root = str(request["staging_root"])

    try:
        _open_source(str(request["source_path"]))
    except (RuntimeError, OSError):
        return protocol.failure(
            request_id=request_id, operation="condition", diagnostics=[_BLEND_HEADER_INVALID]
        )

    scene_inventory.sanitize_write_surfaces(staging_root)

    try:
        result = conditioning.condition(request)
    except conditioning.ConditioningFailed as failure:
        return protocol.failure(
            request_id=request_id, operation="condition", diagnostics=failure.diagnostics
        )

    return {
        "schema_id": protocol.SCHEMA_ID,
        "schema_version": protocol.SCHEMA_VERSION,
        "request_id": request_id,
        "operation": "condition",
        "status": "succeeded",
        "diagnostics": [],
        "portable_labels": sorted(
            value
            for key, value in request["portable_selection"].items()
            if key.endswith("_label") and value
        ),
        "outputs": result["outputs"],
        "metrics": result["metrics"],
        "response_sha256": "",
        "_bundle": result["bundle"],
    }


_OPERATIONS = {
    "preflight": _run_preflight,
    "condition": _run_condition,
    "fixture": _run_fixture,
}


def run(argv: list[str]) -> int:
    arguments = parse_worker_arguments(argv)
    request_path = Path(arguments["request"])
    response_path = Path(arguments["response"])

    request = protocol.read_request(request_path)
    request_id = str(request.get("request_id", "request-unknown-1"))
    operation = str(request.get("operation", "preflight"))

    handler = _OPERATIONS.get(operation)
    if handler is None:
        protocol.write_response(
            response_path,
            protocol.failure(
                request_id=request_id, operation="preflight", diagnostics=[_NOT_IMPLEMENTED]
            ),
        )
        return 0

    try:
        response = handler(request)
    except Exception:  # noqa: BLE001 - the worker must always emit a closed response
        # Anything that escapes a handler still has to become a stable diagnostic code:
        # a traceback reaching Blender's exit path would give the client nothing to
        # publish. The traceback goes to the worker's own stderr, which the client
        # captures and discards.
        traceback.print_exc(file=sys.stderr)
        response = protocol.failure(
            request_id=request_id,
            operation=operation if operation != "fixture" else "validate",
            diagnostics=[_INTERNAL_ERROR],
        )

    protocol.write_response(response_path, response)
    return 0


def main() -> None:  # pragma: no cover - executed only inside Blender
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover - executed only inside Blender
    main()
