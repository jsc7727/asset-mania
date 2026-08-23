"""Guarded launcher and worker for a user-supplied pinned MICA checkout."""

from __future__ import annotations

import argparse
import hashlib
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
from asset_mania_pipeline import (
    FaceGeometryPluginRequest,
    sha256_file,
)

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
class MicaPluginSettings:
    source_root: Path
    isolated_home: Path
    checkpoint_path: Path
    flame_path: Path
    detector_path: Path
    revision: str
    checkpoint_sha256: str
    flame_sha256: str
    detector_sha256: str


@dataclass(frozen=True, slots=True)
class MicaPrediction:
    vertices: np.ndarray
    faces: np.ndarray
    source_projection: np.ndarray
    coordinate_unit: str


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("pinned MICA source revision is unavailable")
    return completed.stdout.strip()


def _git_is_clean(source_root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0 and not completed.stdout.strip()


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


def validate_mica_runtime(
    settings: MicaPluginSettings,
    *,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = sha256_file,
    detector_digest_reader: Callable[[Path], str] = _directory_sha256,
    clean_reader: Callable[[Path], bool] = _git_is_clean,
) -> Path:
    if not settings.source_root.is_dir() or not (settings.source_root / "demo.py").is_file():
        raise ValueError("pinned MICA source revision is unavailable")
    if revision_reader(settings.source_root) != settings.revision:
        raise ValueError("MICA source revision mismatch")
    if not clean_reader(settings.source_root):
        raise ValueError("MICA source tree is not clean")
    if not settings.checkpoint_path.is_file():
        raise ValueError("preplaced MICA checkpoint is unavailable")
    if digest_reader(settings.checkpoint_path) != settings.checkpoint_sha256:
        raise ValueError("MICA checkpoint digest mismatch")
    if not settings.flame_path.is_file():
        raise ValueError("user-supplied FLAME asset is unavailable")
    if digest_reader(settings.flame_path) != settings.flame_sha256:
        raise ValueError("FLAME asset digest mismatch")
    expected_detector = settings.isolated_home / ".insightface" / "models" / "antelopev2"
    if settings.detector_path.resolve() != expected_detector.resolve():
        raise ValueError("MICA detector must be inside the isolated InsightFace model directory")
    if not settings.detector_path.is_dir():
        raise ValueError("preplaced MICA detector asset is unavailable")
    if detector_digest_reader(settings.detector_path) != settings.detector_sha256:
        raise ValueError("MICA detector asset digest mismatch")
    if not settings.isolated_home.is_dir():
        raise ValueError("MICA isolated home is unavailable")
    return settings.checkpoint_path


def _load_request(path: Path) -> FaceGeometryPluginRequest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("MICA request is unreadable") from error
    if not isinstance(document, dict) or set(document) != _REQUEST_FIELDS:
        raise ValueError("MICA request contains fields outside the v1 allowlist")
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
        raise RuntimeError("network denied during MICA inference")

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
    required_keys = {"arcface", "flameModel"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError("MICA checkpoint is missing required model keys")
    return checkpoint


def _weak_projection(
    vertices: np.ndarray, bbox: np.ndarray, image_shape: tuple[int, int]
) -> np.ndarray:
    xy = np.asarray(vertices[:, :2], dtype=np.float64)
    lower = xy.min(axis=0)
    extent = np.maximum(np.ptp(xy, axis=0), 1e-8)
    normalized = (xy - lower) / extent
    x0, y0, x1, y1 = map(float, bbox)
    projected = np.empty((len(vertices), 2), dtype=np.float64)
    projected[:, 0] = x0 + normalized[:, 0] * (x1 - x0)
    projected[:, 1] = y1 - normalized[:, 1] * (y1 - y0)
    height, width = image_shape
    projected[:, 0] = np.clip(projected[:, 0], 0, width - 1)
    projected[:, 1] = np.clip(projected[:, 1], 0, height - 1)
    return projected


def _official_backend(source_image: Path, settings: MicaPluginSettings) -> MicaPrediction:
    import cv2
    import torch
    from configs.config import get_cfg_defaults
    from datasets.creation.util import get_arcface_input, get_center
    from insightface.app.common import Face
    from models.arcface import Arcface
    from models.generator import Generator
    from torch.nn import functional
    from utils.landmark_detector import LandmarksDetector, detectors

    image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("MICA source image is unreadable")
    detector = LandmarksDetector(model=detectors.RETINAFACE)
    bboxes, keypoints = detector.detect(image_bgr)
    if bboxes.shape[0] == 0:
        raise ValueError("MICA found no face in the declared face image")
    selected = get_center(bboxes, image_bgr)
    bbox = bboxes[selected, :4]
    face = Face(bbox=bbox, kps=keypoints[selected], det_score=bboxes[selected, 4])
    arcface_blob, _aligned = get_arcface_input(face, image_bgr)
    arcface_tensor = torch.as_tensor(arcface_blob).float().cuda()
    if arcface_tensor.ndim == 3:
        arcface_tensor = arcface_tensor[None]
    cfg = get_cfg_defaults()
    cfg.model.flame_model_path = str(settings.flame_path)
    arcface = Arcface().to("cuda:0")
    generator = Generator(
        512,
        300,
        cfg.model.n_shape,
        cfg.model.mapping_layers,
        cfg.model,
        "cuda:0",
    )
    checkpoint = torch.load(settings.checkpoint_path, map_location="cuda", weights_only=True)
    checkpoint = _require_checkpoint_keys(checkpoint)
    arcface.load_state_dict(checkpoint["arcface"])
    generator.load_state_dict(checkpoint["flameModel"])
    arcface.eval()
    generator.eval()
    with torch.no_grad():
        identity = functional.normalize(arcface(arcface_tensor))
        predicted_vertices, _shape = generator(identity)
        vertices = predicted_vertices[0].detach().cpu().numpy()
        faces = generator.generator.faces_tensor.detach().cpu().numpy()
    projection = _weak_projection(vertices, bbox, image_bgr.shape[:2])
    canonical = vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])
    canonical_faces = faces[:, [0, 2, 1]]
    del identity, predicted_vertices, arcface_tensor, arcface, generator
    torch.cuda.empty_cache()
    return MicaPrediction(canonical, canonical_faces, projection, "metres")


def _validate_prediction(prediction: MicaPrediction) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(prediction.vertices, dtype=np.float32)
    faces = np.asarray(prediction.faces, dtype=np.int64)
    projection = np.asarray(prediction.source_projection, dtype=np.float32)
    if prediction.coordinate_unit != "metres":
        raise ValueError("MICA prediction must explicitly use metres")
    if vertices.shape != (5023, 3):
        raise ValueError("MICA must return exactly 5,023 vertices")
    if faces.shape != (9976, 3):
        raise ValueError("MICA must return exactly 9,976 triangles")
    if projection.shape != (5023, 2):
        raise ValueError("MICA must return one source projection per vertex")
    if not np.isfinite(vertices).all() or not np.isfinite(projection).all():
        raise ValueError("MICA returned non-finite geometry")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("MICA returned an out-of-range face index")
    extent = float(np.ptp(vertices, axis=0).max())
    if not 0.15 <= extent <= 0.30:
        raise ValueError("MICA geometry extent must be between 0.15 and 0.30 metres")
    return vertices, faces, projection


def execute_mica_request(
    request_path: Path,
    result_path: Path,
    settings: MicaPluginSettings,
    *,
    backend: Callable[[Path, MicaPluginSettings], MicaPrediction] = _official_backend,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = sha256_file,
    detector_digest_reader: Callable[[Path], str] = _directory_sha256,
    clean_reader: Callable[[Path], bool] = _git_is_clean,
) -> int:
    request = _load_request(request_path)
    if request.plugin != "mica-local" or request.profile != "identity-neutral-v1":
        raise ValueError("request differs from MICA identity profile")
    if (
        request.plugin_revision != settings.revision
        or request.checkpoint_sha256 != settings.checkpoint_sha256
    ):
        raise ValueError("MICA request differs from pinned runtime")
    if request.output_directory.exists() or result_path.exists():
        raise FileExistsError("refusing to overwrite MICA plugin output")
    validate_mica_runtime(
        settings,
        revision_reader=revision_reader,
        digest_reader=digest_reader,
        detector_digest_reader=detector_digest_reader,
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
    vertices, faces, projection = _validate_prediction(prediction)
    elapsed = time.perf_counter() - started
    request.output_directory.mkdir()
    geometry = request.output_directory / "geometry.npz"
    np.savez_compressed(
        geometry,
        vertices=vertices,
        faces=faces,
        source_projection=projection,
        detail_displacement=np.zeros(5023, dtype=np.float32),
    )
    result = {
        "schema": "asset-mania.face-geometry-plugin-result.v1",
        "plugin": "mica-local",
        "profile": "identity-neutral-v1",
        "status": "succeeded",
        "geometry": str(geometry.resolve()),
        "vertex_count": 5023,
        "triangle_count": 9976,
        "elapsed_seconds": round(elapsed, 6),
        "device": "cuda",
        "checkpoint_sha256": settings.checkpoint_sha256,
        "topology": "flame-2020-5023",
        "ephemeral_identity_feature_used": True,
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
        "source_root": os.environ.get("ASSET_MANIA_MICA_SOURCE_ROOT"),
        "isolated_home": os.environ.get("ASSET_MANIA_MICA_ISOLATED_HOME"),
        "checkpoint_path": os.environ.get("ASSET_MANIA_MICA_CHECKPOINT_PATH"),
        "flame_path": os.environ.get("ASSET_MANIA_MICA_FLAME_PATH"),
        "flame_sha256": os.environ.get("ASSET_MANIA_MICA_FLAME_SHA256"),
        "detector_path": os.environ.get("ASSET_MANIA_MICA_DETECTOR_PATH"),
        "detector_sha256": os.environ.get("ASSET_MANIA_MICA_DETECTOR_SHA256"),
    }
    if any(not value for value in required.values()):
        raise ValueError("MICA plugin environment is incomplete")
    request = _load_request(arguments.request)
    settings = MicaPluginSettings(
        source_root=Path(required["source_root"]).resolve(strict=True),
        isolated_home=Path(required["isolated_home"]).resolve(strict=True),
        checkpoint_path=Path(required["checkpoint_path"]).resolve(strict=True),
        flame_path=Path(required["flame_path"]).resolve(strict=True),
        detector_path=Path(required["detector_path"]).resolve(strict=True),
        revision=request.plugin_revision,
        checkpoint_sha256=request.checkpoint_sha256,
        flame_sha256=required["flame_sha256"],
        detector_sha256=required["detector_sha256"],
    )
    return execute_mica_request(arguments.request, arguments.result, settings)


if __name__ == "__main__":
    raise SystemExit(main())
