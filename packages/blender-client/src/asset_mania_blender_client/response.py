"""Closed worker-response validation.

The worker's response is the only thing the client believes. It is bounded, structurally
closed against the packaged `blender-response-v1` schema, resealed to prove no field was
edited, and every output path is resolved inside the staging root before publication.
"""

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from asset_mania_contracts import canonical_digest, load_schema
from asset_mania_pipeline import PathEscape, contained_path

MAX_RESPONSE_BYTES = 1048576
_INVALID = "BLENDER_RESPONSE_INVALID"


class ResponseInvalid(Exception):
    """The worker response is absent, oversized, malformed, or not the expected response."""


@lru_cache(maxsize=1)
def _schema() -> dict[str, Any]:
    return load_schema("blender-response", "1.0")


def _closed_keys(schema: Mapping[str, Any]) -> tuple[frozenset[str], frozenset[str]]:
    return frozenset(schema["required"]), frozenset(schema["properties"])


def _metrics_shape(operation: str) -> dict[str, Any]:
    """The closed metrics object one operation may report.

    The schema wraps each branch in `oneOf` with null, because a failed run reports no
    metrics; this returns the object branch itself.
    """
    for branch in _schema()["allOf"]:
        condition = branch["if"]["properties"].get("operation")
        if condition is None or condition["const"] != operation:
            continue
        shape = branch["then"]["properties"]["metrics"]
        for candidate in shape.get("oneOf", [shape]):
            if candidate.get("type") != "null":
                return candidate
    raise ResponseInvalid(f"{_INVALID}: {operation!r} is not a worker operation")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResponseInvalid(f"{_INVALID}: {message}")


def load_response(
    path: Path,
    *,
    request_id: str,
    operation: str,
    staging_root: Path,
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> dict[str, Any]:
    """Read and validate one worker response, or refuse it."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ResponseInvalid(f"{_INVALID}: the worker wrote no readable response") from error

    _require(len(raw) <= max_bytes, f"the response exceeds {max_bytes} bytes")

    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponseInvalid(f"{_INVALID}: the response is not UTF-8 JSON") from error

    _require(isinstance(response, dict), "the response is not an object")

    required, allowed = _closed_keys(_schema())
    _require(not (required - set(response)), f"missing fields {sorted(required - set(response))}")
    _require(
        not (set(response) - allowed),
        f"unknown fields {sorted(set(response) - allowed)}",
    )

    _require(response["schema_id"] == "asset-mania/blender-response", "wrong schema identifier")
    _require(response["schema_version"] == "1.0", "wrong schema version")
    _require(response["request_id"] == request_id, "the response answers another request")
    _require(response["operation"] == operation, "the response answers another operation")
    _require(response["status"] in ("succeeded", "failed"), "unknown response status")

    preimage = {key: value for key, value in response.items() if key != "response_sha256"}
    _require(
        canonical_digest(preimage) == response["response_sha256"],
        "response_sha256 does not match the response content",
    )

    diagnostics = response["diagnostics"]
    _require(
        diagnostics == sorted(set(diagnostics)),
        "diagnostics must be sorted and unique",
    )
    labels = response["portable_labels"]
    _require(labels == sorted(set(labels)), "portable labels must be sorted and unique")

    metrics_shape = _metrics_shape(operation)
    metrics = response["metrics"]
    if response["status"] == "failed" and metrics is None:
        # A failed run has no inventory to report. Null is the honest value; fabricating
        # zeroed counts would look like a measurement that never happened.
        pass
    else:
        _require(isinstance(metrics, dict), "metrics must be an object")
        _require(
            set(metrics) == set(metrics_shape["properties"]),
            f"metrics keys must be exactly {sorted(metrics_shape['properties'])}",
        )
        _require(metrics.get("kind") == operation, "metrics describe another operation")

    _validate_outputs(response, staging_root=staging_root)
    return response


def _validate_outputs(response: Mapping[str, Any], *, staging_root: Path) -> None:
    paths: list[str] = []
    for output in response["outputs"]:
        try:
            contained_path(staging_root, output["path"])
        except PathEscape as error:
            raise ResponseInvalid(f"{_INVALID}: an output path leaves the staging root") from error
        paths.append(output["path"])

    _require(paths == sorted(paths), "outputs must be ordered by relative path")
    _require(len(set(paths)) == len(paths), "outputs must not repeat a path")

    if response["status"] == "failed":
        claimed = [
            output["path"]
            for output in response["outputs"]
            if output["validation"]["status"] == "valid"
        ]
        _require(not claimed, "a failed response must not claim a valid output")
