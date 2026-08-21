"""Tests for the real execution port.

The adapter's own suite proves `adapter.py` cannot reach an engine. This suite covers the
different risk this layer carries: that the port acquires something, or that it accepts input
it should refuse and produces a mesh of the wrong thing.

Everything here runs without weights. The reconstruction itself is exercised separately by
`scripts/run_reconstruction_e2e.py`, which needs a checkpoint and so cannot be a unit test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from asset_mania_engine_triposr.adapter import EngineRequest, ReconstructionFailed
from asset_mania_engine_triposr.ports import triposr as port_module
from asset_mania_engine_triposr.ports.triposr import (
    BACKGROUND_LEVEL,
    TripoSRPort,
    TripoSRSettings,
    WeightsMissing,
    _load_foreground,
    _normalise,
)

PORT_SOURCE = Path(port_module.__file__)

#: Names that would mean this port fetches its own weights. The clearance gate exists so the
#: user reads the licences before the bytes land; a download here would make that ordering a
#: coincidence rather than a property.
ACQUISITION_NAMES = frozenset(
    {
        "hf_hub_download",
        "snapshot_download",
        "urlretrieve",
        "urlopen",
        "get",
        "post",
    }
)
ACQUISITION_MODULES = frozenset(
    {"requests", "urllib", "urllib.request", "httpx", "huggingface_hub", "wget", "socket"}
)


def _settings(tmp_path: Path) -> TripoSRSettings:
    return TripoSRSettings(engine_root=tmp_path / "engine", weights_dir=tmp_path / "weights")


def _request(tmp_path: Path, image: Path, mask: Path | None = None) -> EngineRequest:
    return EngineRequest(
        engine="triposr-local",
        profile="triposr-local-cpu-v1",
        plan_sha256="0" * 64,
        clearance_sha256="1" * 64,
        image_path=image,
        mask_path=mask,
        output_path=tmp_path / "out.obj",
        mesh_format="obj",
    )


def _rgba(tmp_path: Path, *, alpha: int, name: str = "in.png") -> Path:
    arr = np.zeros((32, 32, 4), dtype=np.uint8)
    arr[..., :3] = 200
    arr[8:24, 8:24, 3] = alpha
    path = tmp_path / name
    Image.fromarray(arr, mode="RGBA").save(path)
    return path


class TestNoAcquisition:
    """The port may read weights from disk. It may not go and get them."""

    def test_imports_no_network_or_hub_module(self) -> None:
        tree = ast.parse(PORT_SOURCE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        offenders = imported & {m.split(".")[0] for m in ACQUISITION_MODULES}
        assert not offenders, f"the port imports acquisition modules: {sorted(offenders)}"

    def test_calls_no_download_helper(self) -> None:
        tree = ast.parse(PORT_SOURCE.read_text(encoding="utf-8"))
        called: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
        offenders = called & ACQUISITION_NAMES
        assert not offenders, f"the port calls download helpers: {sorted(offenders)}"

    def test_source_carries_no_url(self) -> None:
        text = PORT_SOURCE.read_text(encoding="utf-8")
        for scheme in ("http://", "https://", "hf://", "s3://"):
            assert scheme not in text, f"the port source contains a {scheme} URL"

    def test_missing_weights_names_the_absent_file(self, tmp_path: Path) -> None:
        (tmp_path / "weights").mkdir()
        (tmp_path / "weights" / "config.yaml").write_text("{}", encoding="utf-8")
        engine = tmp_path / "engine" / "tsr"
        engine.mkdir(parents=True)
        (engine / "system.py").write_text("", encoding="utf-8")

        port = TripoSRPort(settings=_settings(tmp_path))
        with pytest.raises(WeightsMissing, match="model.ckpt"):
            port._ensure_model(object())

    def test_absent_checkout_is_reported_before_any_load(self, tmp_path: Path) -> None:
        port = TripoSRPort(settings=_settings(tmp_path))
        with pytest.raises(WeightsMissing, match="tsr/system.py"):
            port._import_engine()


class TestForegroundIsRequired:
    """TripoSR treats every opaque pixel as the subject, so the mask is not optional."""

    def test_opaque_rgb_without_mask_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "photo.jpg"
        Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8)).save(path)
        with pytest.raises(ReconstructionFailed, match="no mask was supplied"):
            _load_foreground(path, None)

    def test_fully_transparent_alpha_is_refused(self, tmp_path: Path) -> None:
        path = _rgba(tmp_path, alpha=0)
        with pytest.raises(ReconstructionFailed, match="selects no pixels"):
            _load_foreground(path, None)

    def test_alpha_channel_is_accepted(self, tmp_path: Path) -> None:
        rgba = _load_foreground(_rgba(tmp_path, alpha=255), None)
        assert rgba.shape == (32, 32, 4)
        assert int((rgba[..., 3] > 0).sum()) == 16 * 16

    def test_separate_mask_is_accepted(self, tmp_path: Path) -> None:
        photo = tmp_path / "photo.png"
        Image.fromarray(np.full((16, 16, 3), 90, dtype=np.uint8)).save(photo)
        mask_arr = np.zeros((16, 16), dtype=np.uint8)
        mask_arr[4:12, 4:12] = 255
        mask = tmp_path / "mask.png"
        Image.fromarray(mask_arr, mode="L").save(mask)

        rgba = _load_foreground(photo, mask)
        assert rgba.shape == (16, 16, 4)
        assert int((rgba[..., 3] > 0).sum()) == 64

    def test_mismatched_mask_size_is_refused(self, tmp_path: Path) -> None:
        photo = tmp_path / "photo.png"
        Image.fromarray(np.full((16, 16, 3), 90, dtype=np.uint8)).save(photo)
        mask = tmp_path / "mask.png"
        Image.fromarray(np.full((8, 8), 255, dtype=np.uint8), mode="L").save(mask)
        with pytest.raises(ReconstructionFailed, match="mask is"):
            _load_foreground(photo, mask)


class TestGeometryNormalisation:
    """The two corrections in `_normalise`, checked against a shape with a known answer."""

    #: A unit cube's corners and its twelve outward-wound triangles, written out rather than
    #: generated: `trimesh.creation.box` raises under trimesh 4.0.5 with numpy 2.x, and the
    #: pinned trimesh is the one TripoSR asks for. An explicit fixture also makes the winding
    #: this test depends on visible instead of implied.
    CUBE_VERTICES = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    CUBE_FACES_OUTWARD = np.array(
        [
            [0, 2, 1], [0, 3, 2],   # z = 0
            [4, 5, 6], [4, 6, 7],   # z = 1
            [0, 1, 5], [0, 5, 4],   # y = 0
            [3, 6, 2], [3, 7, 6],   # y = 1
            [0, 7, 3], [0, 4, 7],   # x = 0
            [1, 2, 6], [1, 6, 5],   # x = 1
        ]
    )

    @classmethod
    def _inward_soup(cls) -> object:
        """A unit cube as unmerged triangle soup, wound inward -- what the extractor emits."""
        import trimesh

        inward = cls.CUBE_FACES_OUTWARD[:, ::-1]
        soup = trimesh.Trimesh(
            vertices=cls.CUBE_VERTICES[inward].reshape(-1, 3),
            faces=np.arange(inward.size).reshape(-1, 3),
            process=False,
        )
        assert not soup.is_watertight, "the fixture must start as unmerged soup"
        assert soup.volume < 0, "the fixture must start wound inward"
        return soup

    def test_soup_becomes_watertight_and_outward(self) -> None:
        mesh, manifold = _normalise(self._inward_soup())
        assert manifold == "closed"
        assert mesh.is_watertight
        assert mesh.volume > 0, "normals must point outward after normalisation"
        assert mesh.volume == pytest.approx(1.0, rel=1e-6)

    def test_vertices_are_merged(self) -> None:
        mesh, _ = _normalise(self._inward_soup())
        assert len(mesh.vertices) == 8, "a cube has eight distinct corners"

    def test_open_surface_is_reported_as_open_not_closed(self) -> None:
        import trimesh

        plane = trimesh.Trimesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 3]]),
        )
        _, manifold = _normalise(plane)
        assert manifold == "open"


class TestSettingsAreLogSafe:
    def test_describe_omits_paths(self, tmp_path: Path) -> None:
        described = _settings(tmp_path).describe()
        rendered = repr(described)
        assert str(tmp_path) not in rendered
        assert "engine_root" not in described
        assert "weights_dir" not in described

    def test_background_level_matches_upstream_preprocessing(self) -> None:
        # The network was trained on foregrounds composited onto mid grey. Drifting from this
        # value shifts every silhouette edge, so it is pinned rather than left to a default.
        assert BACKGROUND_LEVEL == 0.5
