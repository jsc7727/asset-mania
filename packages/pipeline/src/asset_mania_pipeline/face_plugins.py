"""Closed local subprocess protocol for experimental face reconstruction plugins."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from asset_mania_contracts import canonical_json

DAD_PLUGIN = "dad3dheads-local"
REQUEST_SCHEMA = "asset-mania.face-plugin-request.v0"
RESULT_SCHEMA = "asset-mania.face-plugin-result.v0"
_SHA1_LENGTH = 40
_SHA256_LENGTH = 64
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "plugin",
        "status",
        "raw_mesh",
        "projection_data",
        "vertex_count",
        "triangle_count",
        "elapsed_seconds",
        "device",
        "checkpoint_sha256",
    }
)
_STATUSES = frozenset({"succeeded", "incompatible_runtime", "invalid_output", "execution_failed"})


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class FacePluginRequest:
    schema: str
    plugin: str
    plugin_revision: str
    source_image: Path
    output_directory: Path
    device: Literal["cuda"]
    checkpoint_sha256: str
    network: Literal["denied-during-inference"]

    def __post_init__(self) -> None:
        if self.schema != REQUEST_SCHEMA:
            raise ValueError("unsupported face plugin request schema")
        if self.plugin != DAD_PLUGIN:
            raise ValueError("unsupported face plugin")
        if not _is_lower_hex(self.plugin_revision, _SHA1_LENGTH):
            raise ValueError("revision must be a SHA-1")
        if not self.source_image.is_absolute():
            raise ValueError("source image must be absolute")
        if not self.output_directory.is_absolute():
            raise ValueError("output directory must be absolute")
        if (
            self.source_image == self.output_directory
            or self.output_directory in self.source_image.parents
        ):
            raise ValueError("source image must not be contained by the output directory")
        if self.device != "cuda":
            raise ValueError("device must be cuda")
        if not _is_lower_hex(self.checkpoint_sha256, _SHA256_LENGTH):
            raise ValueError("checkpoint digest must be SHA-256")
        if self.network != "denied-during-inference":
            raise ValueError("network must be denied during inference")


@dataclass(frozen=True, slots=True)
class FacePluginResult:
    schema: str
    plugin: str
    status: Literal["succeeded", "incompatible_runtime", "invalid_output", "execution_failed"]
    raw_mesh: Path | None
    projection_data: Path | None
    vertex_count: int
    triangle_count: int
    elapsed_seconds: float
    device: Literal["cuda"]
    checkpoint_sha256: str


def build_face_plugin_request(
    *,
    plugin: str,
    plugin_revision: str,
    source_image: Path,
    output_directory: Path,
    device: str,
    checkpoint_sha256: str,
) -> FacePluginRequest:
    return FacePluginRequest(
        schema=REQUEST_SCHEMA,
        plugin=plugin,
        plugin_revision=plugin_revision,
        source_image=source_image.resolve(),
        output_directory=output_directory.resolve(),
        device=cast(Literal["cuda"], device),
        checkpoint_sha256=checkpoint_sha256,
        network="denied-during-inference",
    )


def _request_document(request: FacePluginRequest) -> dict[str, object]:
    document = asdict(request)
    document["source_image"] = str(request.source_image)
    document["output_directory"] = str(request.output_directory)
    return document


def write_face_plugin_request(request: FacePluginRequest, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    if request.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {request.output_directory}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(_request_document(request)), encoding="utf-8")


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("face plugin result is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError("face plugin result must be an object")
    return value


def _optional_absolute_path(value: object, label: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a path or null")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    return path


def load_face_plugin_result(path: Path, request: FacePluginRequest) -> FacePluginResult:
    document = _load_object(path)
    if set(document) != _RESULT_FIELDS:
        raise ValueError("face plugin result contains fields outside the v0 allowlist")
    if document["schema"] != RESULT_SCHEMA:
        raise ValueError("face plugin result schema mismatch")
    if document["plugin"] != request.plugin:
        raise ValueError("face plugin result plugin mismatch")
    if document["device"] != request.device:
        raise ValueError("face plugin result device mismatch")
    if document["checkpoint_sha256"] != request.checkpoint_sha256:
        raise ValueError("face plugin result checkpoint digest mismatch")
    status = document["status"]
    if status not in _STATUSES:
        raise ValueError("face plugin result status is invalid")
    raw_mesh = _optional_absolute_path(document["raw_mesh"], "raw mesh")
    projection = _optional_absolute_path(document["projection_data"], "projection data")
    vertex_count = document["vertex_count"]
    triangle_count = document["triangle_count"]
    elapsed_seconds = document["elapsed_seconds"]
    if type(vertex_count) is not int or vertex_count < 0:
        raise ValueError("vertex count must be a non-negative integer")
    if type(triangle_count) is not int or triangle_count < 0:
        raise ValueError("triangle count must be a non-negative integer")
    if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
        raise ValueError("elapsed seconds must be non-negative")
    if status == "succeeded":
        expected_mesh = request.output_directory / "head.obj"
        expected_projection = request.output_directory / "projection.npz"
        if raw_mesh != expected_mesh or projection != expected_projection:
            raise ValueError("face plugin output paths differ from the closed inventory")
        if not expected_mesh.is_file() or not expected_projection.is_file():
            raise ValueError("face plugin success output is missing")
        inventory = {item.name for item in request.output_directory.iterdir()}
        if inventory != {"head.obj", "projection.npz"}:
            raise ValueError("face plugin output inventory is unexpected")
    elif raw_mesh is not None or projection is not None:
        raise ValueError("failed face plugin result must not expose output paths")
    return FacePluginResult(
        schema=RESULT_SCHEMA,
        plugin=request.plugin,
        status=cast(FacePluginResult.__annotations__["status"], status),
        raw_mesh=raw_mesh,
        projection_data=projection,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        elapsed_seconds=float(elapsed_seconds),
        device="cuda",
        checkpoint_sha256=request.checkpoint_sha256,
    )
