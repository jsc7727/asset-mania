"""Guarded launcher and worker for a user-supplied pinned DECA checkout."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from asset_mania_contracts import canonical_json
from asset_mania_pipeline import FaceGeometryPluginRequest, sha256_file

_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "plugin",
        "profile",
        "plugin_revision",
        "source_image",
        "output_directory",
        "device",
        "checkpoint_sha256",
        "topology",
        "face_rights_receipt_sha256",
        "network",
    }
)


@dataclass(frozen=True, slots=True)
class DecaPluginSettings:
    source_root: Path
    isolated_home: Path
    checkpoint_path: Path
    flame_path: Path
    revision: str
    checkpoint_sha256: str
    flame_sha256: str


@dataclass(frozen=True, slots=True)
class DecaPrediction:
    vertices: np.ndarray
    faces: np.ndarray
    source_projection: np.ndarray
    detail_displacement: np.ndarray
    coordinate_unit: str


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("pinned DECA source revision is unavailable")
    return completed.stdout.strip()


def _git_is_clean(source_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


def validate_deca_runtime(
    settings: DecaPluginSettings,
    *,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = sha256_file,
    clean_reader: Callable[[Path], bool] = _git_is_clean,
) -> Path:
    marker = settings.source_root / "decalib" / "deca.py"
    if not settings.source_root.is_dir() or not marker.is_file():
        raise ValueError("pinned DECA source revision is unavailable")
    if revision_reader(settings.source_root) != settings.revision:
        raise ValueError("DECA source revision mismatch")
    if not clean_reader(settings.source_root):
        raise ValueError("DECA source tree is not clean")
    if not settings.checkpoint_path.is_file():
        raise ValueError("preplaced DECA checkpoint is unavailable")
    if digest_reader(settings.checkpoint_path) != settings.checkpoint_sha256:
        raise ValueError("DECA checkpoint digest mismatch")
    if not settings.flame_path.is_file():
        raise ValueError("user-supplied FLAME asset is unavailable")
    if digest_reader(settings.flame_path) != settings.flame_sha256:
        raise ValueError("FLAME asset digest mismatch")
    if not settings.isolated_home.is_dir():
        raise ValueError("DECA isolated home is unavailable")
    return settings.checkpoint_path


def sample_uv_displacement(displacement: np.ndarray, uv_coordinates: np.ndarray) -> np.ndarray:
    image = np.asarray(displacement, dtype=np.float64)
    uv = np.asarray(uv_coordinates, dtype=np.float64)
    if image.ndim != 2 or uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("DECA UV sampling inputs are invalid")
    if not np.isfinite(image).all() or not np.isfinite(uv).all():
        raise ValueError("DECA UV sampling contains non-finite values")
    if np.any(uv < 0) or np.any(uv > 1):
        raise ValueError("DECA UV coordinates are out of range")
    height, width = image.shape
    x = uv[:, 0] * (width - 1)
    y = uv[:, 1] * (height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    return (
        image[y0, x0] * (1 - wx) * (1 - wy)
        + image[y0, x1] * wx * (1 - wy)
        + image[y1, x0] * (1 - wx) * wy
        + image[y1, x1] * wx * wy
    )


def _load_request(path: Path) -> FaceGeometryPluginRequest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DECA request is unreadable") from error
    if not isinstance(document, dict) or set(document) != _REQUEST_FIELDS:
        raise ValueError("DECA request contains fields outside the v1 allowlist")
    return FaceGeometryPluginRequest(
        schema=document["schema"],
        plugin=document["plugin"],
        profile=document["profile"],
        plugin_revision=document["plugin_revision"],
        source_image=Path(document["source_image"]),
        output_directory=Path(document["output_directory"]),
        device=document["device"],
        checkpoint_sha256=document["checkpoint_sha256"],
        topology=document["topology"],
        face_rights_receipt_sha256=document["face_rights_receipt_sha256"],
        network=document["network"],
    )


def _deny_network() -> None:
    def refuse(*_args, **_kwargs):
        raise RuntimeError("network denied during DECA inference")

    socket.socket = refuse
    try:
        import requests
    except ImportError:
        return
    requests.sessions.Session.request = refuse


def _sanitize_credentials() -> None:
    sensitive_fragments = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "ACCESS_KEY")
    for name in tuple(os.environ):
        if any(fragment in name.upper() for fragment in sensitive_fragments):
            os.environ.pop(name, None)


def _require_checkpoint_keys(checkpoint: object) -> dict:
    required_keys = {"E_flame", "E_detail", "D_detail"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError("DECA checkpoint is missing required model keys")
    return checkpoint


def _official_backend(source_image: Path, settings: DecaPluginSettings) -> DecaPrediction:
    import cv2
    import torch
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("DECA source image is unreadable")
    deca_cfg.model.use_tex = False
    deca_cfg.pretrained_modelpath = str(settings.checkpoint_path)
    deca_cfg.model.flame_model_path = str(settings.flame_path)
    _require_checkpoint_keys(
        torch.load(settings.checkpoint_path, map_location="cuda", weights_only=True)
    )
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (224, 224))
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).cuda()[None]
    deca = DECA(config=deca_cfg, device="cuda")
    with torch.no_grad():
        codedict = deca.encode(image, use_detail=True)
        opdict = deca.decode(codedict, rendering=True, return_vis=False, use_detail=True)
        if "displacement_map" not in opdict:
            raise ValueError("DECA did not return the required displacement_map")
        vertices = opdict["verts"][0].detach().cpu().numpy()
        transformed = opdict["trans_verts"][0].detach().cpu().numpy()
        displacement_map = opdict["displacement_map"][0, 0].detach().cpu().numpy()
        faces = deca.render.faces[0].detach().cpu().numpy()
        raw_uv = deca.render.raw_uvcoords[0].detach().cpu().numpy()
    detail = sample_uv_displacement(displacement_map, raw_uv)
    height, width = image_bgr.shape[:2]
    projection = np.stack(
        [
            (transformed[:, 0] + 1.0) * 0.5 * (width - 1),
            (transformed[:, 1] + 1.0) * 0.5 * (height - 1),
        ],
        axis=1,
    )
    canonical = vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])
    canonical_faces = faces[:, [0, 2, 1]]
    del codedict, opdict, image, deca
    torch.cuda.empty_cache()
    return DecaPrediction(canonical, canonical_faces, projection, detail, "metres")


def _validate_prediction(
    prediction: DecaPrediction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(prediction.vertices, dtype=np.float32)
    faces = np.asarray(prediction.faces, dtype=np.int64)
    projection = np.asarray(prediction.source_projection, dtype=np.float32)
    displacement = np.asarray(prediction.detail_displacement, dtype=np.float32)
    if prediction.coordinate_unit != "metres":
        raise ValueError("DECA prediction must explicitly use metres")
    if vertices.shape != (5023, 3):
        raise ValueError("DECA must return exactly 5,023 vertices")
    if faces.shape != (9976, 3):
        raise ValueError("DECA must return exactly 9,976 triangles")
    if projection.shape != (5023, 2):
        raise ValueError("DECA must return one source projection per vertex")
    if displacement.shape != (5023,):
        raise ValueError("DECA must return one displacement per vertex")
    if not all(np.isfinite(array).all() for array in (vertices, projection, displacement)):
        raise ValueError("DECA returned non-finite geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("DECA returned an out-of-range face index")
    extent = float(np.ptp(vertices, axis=0).max())
    if not 0.15 <= extent <= 0.30:
        raise ValueError("DECA geometry extent must be between 0.15 and 0.30 metres")
    return vertices, faces, projection, displacement


def execute_deca_request(
    request_path: Path,
    result_path: Path,
    settings: DecaPluginSettings,
    *,
    backend: Callable[[Path, DecaPluginSettings], DecaPrediction] = _official_backend,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = sha256_file,
    clean_reader: Callable[[Path], bool] = _git_is_clean,
) -> int:
    request = _load_request(request_path)
    if request.plugin != "deca-local" or request.profile != "detail-displacement-v1":
        raise ValueError("request differs from DECA detail profile")
    if (
        request.plugin_revision != settings.revision
        or request.checkpoint_sha256 != settings.checkpoint_sha256
    ):
        raise ValueError("DECA request differs from pinned runtime")
    if request.output_directory.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite DECA plugin output")
    validate_deca_runtime(
        settings,
        revision_reader=revision_reader,
        digest_reader=digest_reader,
        clean_reader=clean_reader,
    )
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
    _sanitize_credentials()
    sys.path.insert(0, str(settings.source_root))
    started = time.perf_counter()
    prediction = backend(request.source_image, settings)
    vertices, faces, projection, displacement = _validate_prediction(prediction)
    elapsed = time.perf_counter() - started
    request.output_directory.mkdir()
    geometry = request.output_directory / "geometry.npz"
    np.savez_compressed(
        geometry,
        vertices=vertices,
        faces=faces,
        source_projection=projection,
        detail_displacement=displacement,
    )
    result = {
        "schema": "asset-mania.face-geometry-plugin-result.v1",
        "plugin": "deca-local",
        "profile": "detail-displacement-v1",
        "status": "succeeded",
        "geometry": str(geometry.resolve()),
        "vertex_count": 5023,
        "triangle_count": 9976,
        "elapsed_seconds": round(elapsed, 6),
        "device": "cuda",
        "checkpoint_sha256": settings.checkpoint_sha256,
        "topology": "flame-2020-5023",
        "ephemeral_identity_feature_used": False,
        "persisted_identity_feature_count": 0,
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
    required = {
        "source_root": os.environ.get("ASSET_MANIA_DECA_SOURCE_ROOT"),
        "isolated_home": os.environ.get("ASSET_MANIA_DECA_ISOLATED_HOME"),
        "checkpoint_path": os.environ.get("ASSET_MANIA_DECA_CHECKPOINT_PATH"),
        "flame_path": os.environ.get("ASSET_MANIA_DECA_FLAME_PATH"),
        "flame_sha256": os.environ.get("ASSET_MANIA_DECA_FLAME_SHA256"),
    }
    if any(not value for value in required.values()):
        raise ValueError("DECA plugin environment is incomplete")
    request = _load_request(arguments.request)
    settings = DecaPluginSettings(
        source_root=Path(required["source_root"]).resolve(strict=True),
        isolated_home=Path(required["isolated_home"]).resolve(strict=True),
        checkpoint_path=Path(required["checkpoint_path"]).resolve(strict=True),
        flame_path=Path(required["flame_path"]).resolve(strict=True),
        revision=request.plugin_revision,
        checkpoint_sha256=request.checkpoint_sha256,
        flame_sha256=required["flame_sha256"],
    )
    return execute_deca_request(arguments.request, arguments.result, settings)


if __name__ == "__main__":
    raise SystemExit(main())
