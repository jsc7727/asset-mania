"""Guarded launcher and worker for a user-supplied pinned DAD-3DHeads checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from asset_mania_contracts import canonical_json
from asset_mania_pipeline import (
    DAD_PLUGIN,
    FacePluginRequest,
    FacePluginResult,
    load_face_plugin_result,
    sha256_file,
)

DAD_REVISION = "68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7"
CHECKPOINT_NAME = "dad_3dheads.trcd"
_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "plugin",
        "plugin_revision",
        "source_image",
        "output_directory",
        "device",
        "checkpoint_sha256",
        "network",
    }
)


@dataclass(frozen=True, slots=True)
class DADPluginSettings:
    source_root: Path
    isolated_home: Path
    revision: str
    checkpoint_sha256: str


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("pinned source revision is unavailable")
    return completed.stdout.strip()


def validate_dad_runtime(
    settings: DADPluginSettings,
    *,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = sha256_file,
) -> Path:
    if not settings.source_root.is_dir() or not (settings.source_root / "predictor.py").is_file():
        raise ValueError("pinned source revision is unavailable")
    if not (settings.source_root / "model_training/model/static/flame.pkl").is_file():
        raise ValueError("DAD FLAME static assets are unavailable")
    if revision_reader(settings.source_root) != settings.revision:
        raise ValueError("source revision mismatch")
    checkpoint = settings.isolated_home / ".dad_checkpoints" / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise ValueError("preplaced DAD checkpoint is unavailable")
    if digest_reader(checkpoint) != settings.checkpoint_sha256:
        raise ValueError("checkpoint digest mismatch")
    return checkpoint


def _load_request(path: Path) -> FacePluginRequest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("face plugin request is unreadable") from error
    if not isinstance(document, dict) or set(document) != _REQUEST_FIELDS:
        raise ValueError("face plugin request contains fields outside the v0 allowlist")
    return FacePluginRequest(
        schema=document["schema"],
        plugin=document["plugin"],
        plugin_revision=document["plugin_revision"],
        source_image=Path(document["source_image"]),
        output_directory=Path(document["output_directory"]),
        device=document["device"],
        checkpoint_sha256=document["checkpoint_sha256"],
        network=document["network"],
    )


def run_face_plugin(
    command: Sequence[str],
    request: FacePluginRequest,
    request_path: Path,
    result_path: Path,
    *,
    timeout_seconds: int,
    environment: Mapping[str, str] | None = None,
) -> FacePluginResult:
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("face plugin command must be an explicit argument sequence")
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    completed = subprocess.run(
        [*command, "--request", str(request_path), "--result", str(result_path)],
        check=False,
        shell=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(environment) if environment is not None else dict(os.environ),
    )
    if str(request.source_image) in completed.stdout + completed.stderr:
        raise ValueError("face plugin exposed the private source path")
    if completed.returncode != 0:
        raise ValueError(f"face plugin exited with {completed.returncode}")
    return load_face_plugin_result(result_path, request)


def _deny_network() -> None:
    import requests

    def refuse(*_args, **_kwargs):
        raise RuntimeError("network denied during DAD inference")

    requests.sessions.Session.request = refuse


def _tensor_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value.numpy() if hasattr(value, "numpy") else value


def _write_obj(path: Path, vertices, faces) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")


def execute_dad_request(request_path: Path, result_path: Path, settings: DADPluginSettings) -> int:
    request = _load_request(request_path)
    if request.plugin != DAD_PLUGIN or request.plugin_revision != settings.revision:
        raise ValueError("DAD request differs from the pinned plugin")
    if request.checkpoint_sha256 != settings.checkpoint_sha256:
        raise ValueError("DAD request checkpoint digest mismatch")
    validate_dad_runtime(settings)
    if request.output_directory.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite DAD plugin output")

    os.environ.update(
        {
            "HOME": str(settings.isolated_home),
            "USERPROFILE": str(settings.isolated_home),
            "XDG_CACHE_HOME": str(settings.isolated_home / "xdg-cache"),
            "TORCH_HOME": str(settings.isolated_home / "torch-cache"),
            "HF_HOME": str(settings.isolated_home / "hf-cache"),
        }
    )
    _deny_network()
    sys.path.insert(0, str(settings.source_root))

    import cv2
    import numpy as np
    import torch
    from predictor import FaceMeshPredictor

    if not torch.cuda.is_available():
        raise RuntimeError("DAD plugin requires CUDA")
    image_bgr = cv2.imread(str(request.source_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("DAD source image is unreadable")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    predictor = FaceMeshPredictor.dad_3dnet()
    predictions = predictor(image_rgb)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    vertices = np.asarray(_tensor_numpy(predictions["3d_vertices"]), dtype=np.float64)
    projected = np.asarray(_tensor_numpy(predictions["projected_vertices"]), dtype=np.float64)
    vertices = np.squeeze(vertices)
    projected = np.squeeze(projected)
    faces_path = settings.source_root / "model_training/model/static/flame_mesh_faces.pt"
    faces = np.asarray(
        _tensor_numpy(torch.load(faces_path, map_location="cpu", weights_only=True)), dtype=np.int64
    )
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("DAD returned invalid vertices")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("DAD returned invalid faces")
    if projected.shape != (len(vertices), 2) or not np.isfinite(projected).all():
        raise ValueError("DAD returned invalid projected vertices")

    request.output_directory.mkdir()
    mesh_path = request.output_directory / "head.obj"
    projection_path = request.output_directory / "projection.npz"
    _write_obj(mesh_path, vertices, faces + 1)
    np.savez_compressed(
        projection_path,
        projected_vertices=projected,
        image_shape=np.asarray(image_rgb.shape[:2], dtype=np.int64),
    )
    result = {
        "schema": "asset-mania.face-plugin-result.v0",
        "plugin": DAD_PLUGIN,
        "status": "succeeded",
        "raw_mesh": str(mesh_path.resolve()),
        "projection_data": str(projection_path.resolve()),
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "elapsed_seconds": round(elapsed, 6),
        "device": "cuda",
        "checkpoint_sha256": settings.checkpoint_sha256,
    }
    result_path.write_text(canonical_json(result), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    source_root = os.environ.get("ASSET_MANIA_DAD_SOURCE_ROOT")
    isolated_home = os.environ.get("ASSET_MANIA_DAD_ISOLATED_HOME")
    if not source_root or not isolated_home:
        raise ValueError("DAD plugin environment is incomplete")
    request = _load_request(arguments.request)
    settings = DADPluginSettings(
        source_root=Path(source_root).resolve(strict=True),
        isolated_home=Path(isolated_home).resolve(strict=True),
        revision=request.plugin_revision,
        checkpoint_sha256=request.checkpoint_sha256,
    )
    return execute_dad_request(arguments.request, arguments.result, settings)


if __name__ == "__main__":
    raise SystemExit(main())
