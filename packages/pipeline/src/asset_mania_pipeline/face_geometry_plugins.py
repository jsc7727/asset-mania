"""Closed local subprocess protocol for authorized face-geometry plugins."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from asset_mania_contracts import canonical_json

MICA_PLUGIN = "mica-local"
DECA_PLUGIN = "deca-local"
REQUEST_SCHEMA = "asset-mania.face-geometry-plugin-request.v1"
RESULT_SCHEMA = "asset-mania.face-geometry-plugin-result.v1"
TOPOLOGY = "flame-2020-5023"
_PLUGIN_PROFILES = {MICA_PLUGIN: "identity-neutral-v1", DECA_PLUGIN: "detail-displacement-v1"}
_STATUSES = frozenset({"succeeded", "incompatible_runtime", "invalid_output", "execution_failed"})
_RESULT_FIELDS = frozenset(
    {
        "schema",
        "plugin",
        "profile",
        "status",
        "geometry",
        "vertex_count",
        "triangle_count",
        "elapsed_seconds",
        "device",
        "checkpoint_sha256",
        "topology",
        "ephemeral_identity_feature_used",
        "persisted_identity_feature_count",
    }
)
_STANDARD_ENVIRONMENT = frozenset({"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"})
_PLUGIN_SETTING_SUFFIXES = frozenset(
    {
        "SOURCE_ROOT",
        "ISOLATED_HOME",
        "CHECKPOINT_PATH",
        "FLAME_PATH",
        "FLAME_SHA256",
        "DETECTOR_PATH",
        "DETECTOR_SHA256",
    }
)


def _is_lower_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class FaceGeometryPluginRequest:
    schema: str
    plugin: Literal["mica-local", "deca-local"]
    profile: Literal["identity-neutral-v1", "detail-displacement-v1"]
    plugin_revision: str
    source_image: Path
    output_directory: Path
    device: Literal["cuda"]
    checkpoint_sha256: str
    topology: Literal["flame-2020-5023"]
    face_rights_receipt_sha256: str
    network: Literal["denied-during-inference"]

    def __post_init__(self) -> None:
        if self.schema != REQUEST_SCHEMA:
            raise ValueError("unsupported face geometry request schema")
        if self.plugin not in _PLUGIN_PROFILES:
            raise ValueError("unsupported face geometry plugin")
        if self.profile != _PLUGIN_PROFILES[self.plugin]:
            raise ValueError("profile does not belong to plugin")
        if not _is_lower_hex(self.plugin_revision, 40):
            raise ValueError("revision must be a SHA-1")
        if not self.source_image.is_absolute():
            raise ValueError("source image must be absolute")
        if not self.output_directory.is_absolute():
            raise ValueError("output directory must be absolute")
        if (
            self.source_image == self.output_directory
            or self.output_directory in self.source_image.parents
        ):
            raise ValueError("source image must not be contained by output directory")
        if self.device != "cuda":
            raise ValueError("device must be cuda")
        if not _is_lower_hex(self.checkpoint_sha256, 64):
            raise ValueError("checkpoint digest must be SHA-256")
        if self.topology != TOPOLOGY:
            raise ValueError("topology must be flame-2020-5023")
        if not _is_lower_hex(self.face_rights_receipt_sha256, 64):
            raise ValueError("rights digest must be SHA-256")
        if self.network != "denied-during-inference":
            raise ValueError("network must be denied during inference")


@dataclass(frozen=True, slots=True)
class FaceGeometryPluginResult:
    schema: str
    plugin: Literal["mica-local", "deca-local"]
    profile: Literal["identity-neutral-v1", "detail-displacement-v1"]
    status: Literal["succeeded", "incompatible_runtime", "invalid_output", "execution_failed"]
    geometry: Path | None
    vertex_count: int
    triangle_count: int
    elapsed_seconds: float
    device: Literal["cuda"]
    checkpoint_sha256: str
    topology: Literal["flame-2020-5023"]
    ephemeral_identity_feature_used: bool
    persisted_identity_feature_count: Literal[0]


def build_face_geometry_plugin_request(
    *,
    plugin: str,
    profile: str,
    plugin_revision: str,
    source_image: Path,
    output_directory: Path,
    device: str,
    checkpoint_sha256: str,
    topology: str,
    face_rights_receipt_sha256: str,
) -> FaceGeometryPluginRequest:
    return FaceGeometryPluginRequest(
        schema=REQUEST_SCHEMA,
        plugin=cast(Literal["mica-local", "deca-local"], plugin),
        profile=cast(Literal["identity-neutral-v1", "detail-displacement-v1"], profile),
        plugin_revision=plugin_revision,
        source_image=source_image.resolve(),
        output_directory=output_directory.resolve(),
        device=cast(Literal["cuda"], device),
        checkpoint_sha256=checkpoint_sha256,
        topology=cast(Literal["flame-2020-5023"], topology),
        face_rights_receipt_sha256=face_rights_receipt_sha256,
        network="denied-during-inference",
    )


def write_face_geometry_plugin_request(request: FaceGeometryPluginRequest, path: Path) -> None:
    if path.exists() or request.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    document = asdict(request)
    document["source_image"] = str(request.source_image)
    document["output_directory"] = str(request.output_directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(document), encoding="utf-8")


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("face geometry plugin result is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError("face geometry plugin result must be an object")
    return value


def _optional_absolute_path(value: object) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("geometry must be a path or null")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("geometry must be absolute")
    return path


def load_face_geometry_plugin_result(
    path: Path, request: FaceGeometryPluginRequest
) -> FaceGeometryPluginResult:
    document = _load_object(path)
    if set(document) != _RESULT_FIELDS:
        raise ValueError("face geometry plugin result contains fields outside the v1 allowlist")
    if document["schema"] != RESULT_SCHEMA:
        raise ValueError("face geometry plugin result schema mismatch")
    if document["plugin"] != request.plugin or document["profile"] != request.profile:
        raise ValueError("face geometry plugin identity mismatch")
    if document["device"] != request.device:
        raise ValueError("face geometry plugin device mismatch")
    if document["checkpoint_sha256"] != request.checkpoint_sha256:
        raise ValueError("face geometry plugin checkpoint digest mismatch")
    if document["topology"] != request.topology:
        raise ValueError("face geometry plugin topology mismatch")
    status = document["status"]
    if status not in _STATUSES:
        raise ValueError("face geometry plugin status is invalid")
    geometry = _optional_absolute_path(document["geometry"])
    vertex_count = document["vertex_count"]
    triangle_count = document["triangle_count"]
    elapsed_seconds = document["elapsed_seconds"]
    ephemeral = document["ephemeral_identity_feature_used"]
    persisted = document["persisted_identity_feature_count"]
    if type(vertex_count) is not int or vertex_count < 0:
        raise ValueError("vertex count must be a non-negative integer")
    if type(triangle_count) is not int or triangle_count < 0:
        raise ValueError("triangle count must be a non-negative integer")
    if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
        raise ValueError("elapsed seconds must be non-negative")
    if type(ephemeral) is not bool:
        raise ValueError("ephemeral identity feature flag must be boolean")
    if type(persisted) is not int or persisted != 0:
        raise ValueError("persisted identity feature count must be zero")
    if status == "succeeded":
        if vertex_count != 5023 or triangle_count != 9976:
            raise ValueError("successful geometry must use exact FLAME topology counts")
        expected_ephemeral = request.plugin == MICA_PLUGIN
        if ephemeral is not expected_ephemeral:
            raise ValueError("identity feature flag does not match the plugin profile")
        expected = request.output_directory / "geometry.npz"
        if geometry != expected or not expected.is_file():
            raise ValueError("face geometry plugin success output is missing")
        if {item.name for item in request.output_directory.iterdir()} != {"geometry.npz"}:
            raise ValueError("face geometry plugin output inventory is unexpected")
    elif geometry is not None:
        raise ValueError("failed result must not expose geometry")
    return FaceGeometryPluginResult(
        schema=RESULT_SCHEMA,
        plugin=request.plugin,
        profile=request.profile,
        status=cast(FaceGeometryPluginResult.__annotations__["status"], status),
        geometry=geometry,
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        elapsed_seconds=float(elapsed_seconds),
        device="cuda",
        checkpoint_sha256=request.checkpoint_sha256,
        topology="flame-2020-5023",
        ephemeral_identity_feature_used=ephemeral,
        persisted_identity_feature_count=0,
    )


def run_face_geometry_plugin(
    command: Sequence[str],
    request: FaceGeometryPluginRequest,
    request_path: Path,
    result_path: Path,
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> FaceGeometryPluginResult:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("face geometry plugin command must be an explicit argument sequence")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    source_environment = dict(environment) if environment is not None else dict(os.environ)
    plugin_prefix = "ASSET_MANIA_MICA_" if request.plugin == MICA_PLUGIN else "ASSET_MANIA_DECA_"
    plugin_settings = {f"{plugin_prefix}{suffix}" for suffix in _PLUGIN_SETTING_SUFFIXES}
    unknown = sorted(
        key
        for key in source_environment
        if key.startswith(plugin_prefix) and key not in plugin_settings
    )
    if unknown:
        label = "MICA" if request.plugin == MICA_PLUGIN else "DECA"
        raise ValueError(f"unknown {label} plugin environment variable")
    sanitized_environment = {
        key: value
        for key, value in source_environment.items()
        if key.upper() in _STANDARD_ENVIRONMENT or key in plugin_settings
    }
    completed = subprocess.run(
        [*command, "--request", str(request_path), "--result", str(result_path)],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=sanitized_environment,
    )
    combined = completed.stdout + completed.stderr
    if str(request.source_image) in combined or request.source_image.name in combined:
        raise ValueError("face geometry plugin exposed the private source")
    if completed.returncode != 0:
        raise ValueError(f"face geometry plugin exited with {completed.returncode}")
    return load_face_geometry_plugin_result(result_path, request)
