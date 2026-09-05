"""Guarded launcher and worker for a user-supplied pinned DECA checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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


@dataclass(frozen=True)
class DecaPluginSettings:
    source_root: Path
    isolated_home: Path
    checkpoint_path: Path
    flame_path: Path
    detector_path: Path
    revision: str
    checkpoint_sha256: str
    flame_sha256: str
    detector_sha256: str


@dataclass(frozen=True)
class DecaPrediction:
    vertices: np.ndarray
    faces: np.ndarray
    source_projection: np.ndarray
    detail_displacement: np.ndarray
    coordinate_unit: str


@dataclass(frozen=True)
class _FaceGeometryPluginRequest:
    schema: str
    plugin: str
    profile: str
    plugin_revision: str
    source_image: Path
    output_directory: Path
    device: str
    checkpoint_sha256: str
    topology: str
    face_rights_receipt_sha256: str
    network: str

    def __post_init__(self) -> None:
        if self.schema != "asset-mania.face-geometry-plugin-request.v1":
            raise ValueError("unsupported face geometry request schema")
        profiles = {"mica-local": "identity-neutral-v1", "deca-local": "detail-displacement-v1"}
        if self.plugin not in profiles:
            raise ValueError("unsupported face geometry plugin")
        if self.profile != profiles[self.plugin]:
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
        if self.topology != "flame-2020-5023":
            raise ValueError("topology must be flame-2020-5023")
        if not _is_lower_hex(self.face_rights_receipt_sha256, 64):
            raise ValueError("rights digest must be SHA-256")
        if self.network != "denied-during-inference":
            raise ValueError("network must be denied during inference")


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def validate_deca_runtime(
    settings: DecaPluginSettings,
    *,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = _sha256_file,
    detector_digest_reader: Callable[[Path], str] = _directory_sha256,
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
    expected_detector = settings.isolated_home / ".insightface" / "models" / "antelopev2"
    if settings.detector_path.resolve() != expected_detector.resolve():
        raise ValueError("DECA detector must be inside the isolated InsightFace model directory")
    if not settings.detector_path.is_dir():
        raise ValueError("preplaced DECA detector asset is unavailable")
    if detector_digest_reader(settings.detector_path) != settings.detector_sha256:
        raise ValueError("DECA detector asset digest mismatch")
    if not settings.isolated_home.is_dir():
        raise ValueError("DECA isolated home is unavailable")
    return settings.checkpoint_path


def sample_uv_displacement(displacement: np.ndarray, uv_coordinates: np.ndarray) -> np.ndarray:
    import numpy as np

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
    y = (1.0 - uv[:, 1]) * (height - 1)
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


def sample_position_uv_displacement(
    displacement: np.ndarray,
    vertex_count: int,
    faces: np.ndarray,
    uv_coordinates: np.ndarray,
    uv_faces: np.ndarray,
) -> np.ndarray:
    import numpy as np

    raw_triangles = np.asarray(faces)
    raw_texture_triangles = np.asarray(uv_faces)
    coordinates = np.asarray(uv_coordinates, dtype=np.float64)
    if vertex_count <= 0:
        raise ValueError("DECA vertex count is invalid")
    if (
        raw_triangles.shape != raw_texture_triangles.shape
        or raw_triangles.ndim != 2
        or raw_triangles.shape[1] != 3
    ):
        raise ValueError("DECA position and UV topology differ")
    if not len(raw_triangles):
        raise ValueError("DECA position and UV topology is empty")
    for values, label in (
        (raw_triangles, "position"),
        (raw_texture_triangles, "UV"),
    ):
        has_real_numeric_dtype = np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.floating
        )
        if (
            not has_real_numeric_dtype
            or not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
        ):
            raise ValueError(f"DECA {label} topology indices must be integers")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("DECA UV coordinates are invalid")
    if not np.isfinite(coordinates).all():
        raise ValueError("DECA UV coordinates contain non-finite values")
    if np.any(coordinates < 0) or np.any(coordinates > 1):
        raise ValueError("DECA UV coordinates are out of range")
    triangles = raw_triangles.astype(np.int64)
    texture_triangles = raw_texture_triangles.astype(np.int64)
    if triangles.min() < 0 or triangles.max() >= vertex_count:
        raise ValueError("DECA position topology is out of range")
    if texture_triangles.min() < 0 or texture_triangles.max() >= len(coordinates):
        raise ValueError("DECA UV topology is out of range")
    corner_displacement = sample_uv_displacement(
        displacement, coordinates[texture_triangles.reshape(-1)]
    )
    accumulated = np.zeros(vertex_count, dtype=np.float64)
    counts = np.zeros(vertex_count, dtype=np.int64)
    np.add.at(accumulated, triangles.reshape(-1), corner_displacement)
    np.add.at(counts, triangles.reshape(-1), 1)
    if np.any(counts == 0):
        raise ValueError("DECA UV topology leaves a position unmapped")
    return accumulated / counts


def _decompose_code(parameters, model_cfg):
    code = {}
    start = 0
    for key in model_cfg.param_list:
        count = int(getattr(model_cfg, f"n_{key}"))
        code[key] = parameters[:, start : start + count]
        start += count
    if start != parameters.shape[1]:
        raise ValueError("DECA parameter dimensions differ from the sealed profile")
    code["light"] = code["light"].reshape(code["light"].shape[0], 9, 3)
    return code


def _load_request(path: Path) -> _FaceGeometryPluginRequest:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("DECA request is unreadable") from error
    if not isinstance(document, dict) or set(document) != _REQUEST_FIELDS:
        raise ValueError("DECA request contains fields outside the v1 allowlist")
    return _FaceGeometryPluginRequest(
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
    """Install an in-process Python guard; this is not an operating-system sandbox."""

    def refuse(*_args, **_kwargs):
        raise RuntimeError("network denied during DECA inference")

    try:
        import requests
    except ImportError:
        requests = None

    if not getattr(socket.socket, "_asset_mania_network_denied", False):

        class DeniedSocket(socket.socket):
            _asset_mania_network_denied = True

            def connect(self, _address) -> None:
                refuse()

            def connect_ex(self, _address) -> int:
                refuse()

            def send(self, *_args, **_kwargs):
                refuse()

            def sendall(self, *_args, **_kwargs) -> None:
                refuse()

            def sendto(self, *_args, **_kwargs):
                refuse()

        if hasattr(socket.socket, "sendmsg"):
            DeniedSocket.sendmsg = refuse

        socket.socket = DeniedSocket
    socket.create_connection = refuse
    if requests is not None:
        requests.sessions.Session.request = refuse


def _sanitize_environment(settings: DecaPluginSettings) -> None:
    inherited = {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP"}
    plugin_settings = {
        "ASSET_MANIA_DECA_SOURCE_ROOT",
        "ASSET_MANIA_DECA_ISOLATED_HOME",
        "ASSET_MANIA_DECA_CHECKPOINT_PATH",
        "ASSET_MANIA_DECA_FLAME_PATH",
        "ASSET_MANIA_DECA_FLAME_SHA256",
        "ASSET_MANIA_DECA_DETECTOR_PATH",
        "ASSET_MANIA_DECA_DETECTOR_SHA256",
    }
    preserved = {
        name: value for name, value in os.environ.items() if name in inherited | plugin_settings
    }
    preserved.update(
        {
            "HOME": str(settings.isolated_home),
            "USERPROFILE": str(settings.isolated_home),
            "XDG_CACHE_HOME": str(settings.isolated_home / "xdg-cache"),
            "TORCH_HOME": str(settings.isolated_home / "torch-cache"),
            "HF_HOME": str(settings.isolated_home / "hf-cache"),
        }
    )
    os.environ.clear()
    os.environ.update(preserved)


def _require_cuda(torch_module) -> None:
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise ValueError("DECA requires an available CUDA device")


def _require_checkpoint_keys(checkpoint: object) -> dict:
    required_keys = {"E_flame", "E_detail", "D_detail"}
    if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint):
        raise ValueError("DECA checkpoint is missing required model keys")
    return checkpoint


def _restore_chumpy_numpy_aliases(np_module=None) -> None:
    if np_module is None:
        import numpy as np_module
    for name, value in {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }.items():
        if name not in np_module.__dict__:
            setattr(np_module, name, value)


def _bind_deca_model_assets(model_cfg: object, settings: DecaPluginSettings) -> None:
    topology = settings.source_root / "data" / "head_template.obj"
    fixed_displacement = settings.source_root / "data" / "fixed_displacement_256.npy"
    landmark_embedding = settings.source_root / "data" / "landmark_embedding.npy"
    if not settings.flame_path.is_file():
        raise ValueError("user-supplied FLAME asset is unavailable")
    if (
        not topology.is_file()
        or not fixed_displacement.is_file()
        or not landmark_embedding.is_file()
    ):
        raise ValueError("tracked DECA model asset is unavailable")
    model_cfg.flame_model_path = str(settings.flame_path.resolve(strict=True))
    model_cfg.topology_path = str(topology.resolve(strict=True))
    model_cfg.fixed_displacement_path = str(fixed_displacement.resolve(strict=True))
    model_cfg.flame_lmk_embedding_path = str(landmark_embedding.resolve(strict=True))


def _select_center_face(bboxes: np.ndarray, image_shape: tuple[int, int]) -> int:
    import numpy as np

    centers = (np.asarray(bboxes)[:, :2] + np.asarray(bboxes)[:, 2:4]) * 0.5
    image_center = np.array([image_shape[1] * 0.5, image_shape[0] * 0.5])
    return int(np.argmin(np.sum((centers - image_center) ** 2, axis=1)))


def _load_scrfd_detector(detector_directory: Path, *, detector_factory=None):
    detector_model = detector_directory / "scrfd_10g_bnkps.onnx"
    if not detector_model.is_file():
        raise ValueError("preplaced sealed SCRFD detector is unavailable")
    if detector_factory is None:
        from insightface import model_zoo

        detector_factory = model_zoo.get_model
    detector = detector_factory(str(detector_model), providers=["CUDAExecutionProvider"])
    session = getattr(detector, "session", None)
    if session is None or "CUDAExecutionProvider" not in session.get_providers():
        raise ValueError("DECA detector requires the CUDA execution provider")
    detector.prepare(ctx_id=0, input_size=(224, 224))
    return detector


def _detect_face_with_scrfd(
    image_bgr: np.ndarray,
    detector_directory: Path,
    *,
    detector_factory=None,
    detector=None,
) -> np.ndarray:
    if detector is None:
        detector = _load_scrfd_detector(detector_directory, detector_factory=detector_factory)
    bboxes, _keypoints = detector.detect(image_bgr, max_num=0, metric="default")
    if bboxes.shape[0] == 0:
        raise ValueError("DECA found no face in the declared face image")
    return bboxes[_select_center_face(bboxes, image_bgr.shape[:2]), :4]


def _deca_face_crop(
    image_bgr: np.ndarray, bbox: np.ndarray, *, output_size: int = 224, cv2_module=None
):
    if cv2_module is None:
        import cv2 as cv2_module
    import numpy as np

    left, top, right, bottom = map(float, bbox)
    if not np.isfinite([left, top, right, bottom]).all() or right <= left or bottom <= top:
        raise ValueError("DECA detector bbox must have finite positive extents")
    old_size = ((right - left) + (bottom - top)) * 0.5
    center = np.array(
        [(left + right) * 0.5, (top + bottom) * 0.5 + old_size * 0.12],
        dtype=np.float64,
    )
    size = int(old_size * 1.25)
    half = size * 0.5
    scale = (output_size - 1) / size
    transform = np.array(
        [[scale, 0.0, -(center[0] - half) * scale], [0.0, scale, -(center[1] - half) * scale]],
        dtype=np.float64,
    )
    inverse = np.array(
        [[1.0 / scale, 0.0, center[0] - half], [0.0, 1.0 / scale, center[1] - half]],
        dtype=np.float64,
    )
    crop = cv2_module.warpAffine(image_bgr, transform, (output_size, output_size))
    return crop, inverse


def _project_crop_to_source(points: np.ndarray, inverse_transform: np.ndarray) -> np.ndarray:
    import numpy as np

    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    return homogeneous @ np.asarray(inverse_transform, dtype=np.float64).T


def _official_backend(
    source_image: Path,
    settings: DecaPluginSettings,
    *,
    cuda_validator: Callable[[object], None] = _require_cuda,
) -> DecaPrediction:
    import cv2
    import numpy as np
    import torch

    cuda_validator(torch)
    detector = _load_scrfd_detector(settings.detector_path)
    _restore_chumpy_numpy_aliases(np)
    from decalib.models.decoders import Generator as DetailGenerator
    from decalib.models.encoders import ResnetEncoder
    from decalib.models.FLAME import FLAME
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    image_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("DECA source image is unreadable")
    bbox = _detect_face_with_scrfd(image_bgr, settings.detector_path, detector=detector)
    cropped_bgr, inverse_transform = _deca_face_crop(image_bgr, bbox)
    model_cfg = deca_cfg.model
    model_cfg.use_tex = False
    model_cfg.extract_tex = False
    _bind_deca_model_assets(model_cfg, settings)
    checkpoint = _require_checkpoint_keys(
        torch.load(settings.checkpoint_path, map_location="cuda", weights_only=True)
    )
    image_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div(255.0).cuda()[None]
    n_parameters = sum(int(getattr(model_cfg, f"n_{key}")) for key in model_cfg.param_list)
    encoder = ResnetEncoder(outsize=n_parameters).cuda().eval()
    detail_encoder = ResnetEncoder(outsize=model_cfg.n_detail).cuda().eval()
    detail_decoder = (
        DetailGenerator(
            latent_dim=model_cfg.n_detail + model_cfg.n_exp + 3,
            out_channels=1,
            out_scale=model_cfg.max_z,
            sample_mode="bilinear",
        )
        .cuda()
        .eval()
    )
    flame = FLAME(model_cfg).cuda().eval()
    util.copy_state_dict(encoder.state_dict(), checkpoint["E_flame"])
    util.copy_state_dict(detail_encoder.state_dict(), checkpoint["E_detail"])
    util.copy_state_dict(detail_decoder.state_dict(), checkpoint["D_detail"])
    if model_cfg.jaw_type != "aa":
        raise ValueError("DECA adapter requires the sealed axis-angle jaw profile")
    with torch.no_grad():
        codedict = _decompose_code(encoder(image), model_cfg)
        detail_code = detail_encoder(image)
        vertices_tensor, _landmarks2d, _landmarks3d = flame(
            shape_params=codedict["shape"],
            expression_params=codedict["exp"],
            pose_params=codedict["pose"],
        )
        transformed_tensor = util.batch_orth_proj(vertices_tensor, codedict["cam"])
        transformed_tensor[:, :, 1:] = -transformed_tensor[:, :, 1:]
        detail_input = torch.cat([codedict["pose"][:, 3:], codedict["exp"], detail_code], dim=1)
        displacement_tensor = detail_decoder(detail_input)
        fixed = torch.from_numpy(np.load(model_cfg.fixed_displacement_path)).float().cuda()
        displacement_tensor = displacement_tensor + fixed[None, None]
        vertices = vertices_tensor[0].detach().cpu().numpy()
        transformed = transformed_tensor[0].detach().cpu().numpy()
        displacement_map = displacement_tensor[0, 0].detach().cpu().numpy()
    _template_vertices, raw_uv, faces, uv_faces = util.load_obj(model_cfg.topology_path)
    faces = faces.cpu().numpy()
    detail = sample_position_uv_displacement(
        displacement_map,
        len(vertices),
        faces,
        raw_uv.cpu().numpy(),
        uv_faces.cpu().numpy(),
    )
    crop_projection = np.stack(
        [
            (transformed[:, 0] + 1.0) * 0.5 * 223,
            (transformed[:, 1] + 1.0) * 0.5 * 223,
        ],
        axis=1,
    )
    projection = _project_crop_to_source(crop_projection, inverse_transform)
    canonical = vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])
    canonical_faces = faces[:, [0, 2, 1]]
    del codedict, image, encoder, detail_encoder, detail_decoder, flame
    torch.cuda.empty_cache()
    return DecaPrediction(canonical, canonical_faces, projection, detail, "metres")


def _validate_prediction(
    prediction: DecaPrediction,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np

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
    axis_extents = np.ptp(vertices, axis=0)
    if not np.isfinite(axis_extents).all() or not (axis_extents > 0).all():
        raise ValueError("DECA geometry must have positive finite extent on every axis")
    return vertices, faces, projection, displacement


def execute_deca_request(
    request_path: Path,
    result_path: Path,
    settings: DecaPluginSettings,
    *,
    backend: Callable[[Path, DecaPluginSettings], DecaPrediction] = _official_backend,
    revision_reader: Callable[[Path], str] = _git_revision,
    digest_reader: Callable[[Path], str] = _sha256_file,
    detector_digest_reader: Callable[[Path], str] = _directory_sha256,
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
        detector_digest_reader=detector_digest_reader,
        clean_reader=clean_reader,
    )
    _sanitize_environment(settings)
    _deny_network()
    sys.path.insert(0, str(settings.source_root))
    started = time.perf_counter()
    prediction = backend(request.source_image, settings)
    import numpy as np

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
    result_path.write_text(_canonical_json(result), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv=None) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    required = {
        "source_root": os.environ.get("ASSET_MANIA_DECA_SOURCE_ROOT"),
        "isolated_home": os.environ.get("ASSET_MANIA_DECA_ISOLATED_HOME"),
        "checkpoint_path": os.environ.get("ASSET_MANIA_DECA_CHECKPOINT_PATH"),
        "flame_path": os.environ.get("ASSET_MANIA_DECA_FLAME_PATH"),
        "flame_sha256": os.environ.get("ASSET_MANIA_DECA_FLAME_SHA256"),
        "detector_path": os.environ.get("ASSET_MANIA_DECA_DETECTOR_PATH"),
        "detector_sha256": os.environ.get("ASSET_MANIA_DECA_DETECTOR_SHA256"),
    }
    if any(not value for value in required.values()):
        raise ValueError("DECA plugin environment is incomplete")
    request = _load_request(arguments.request)
    settings = DecaPluginSettings(
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
    return execute_deca_request(arguments.request, arguments.result, settings)


if __name__ == "__main__":
    raise SystemExit(main())
