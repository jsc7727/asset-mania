"""A real execution port for a locally installed TripoSR.

This is the module that finally touches the engine. Everything the adapter refused to do --
load a checkpoint, run a network, extract a surface -- happens here, behind three rules that
exist because breaking any one of them would quietly undo a guarantee made elsewhere:

1. Nothing is downloaded. `TSR.from_pretrained` accepts either a HuggingFace repo id, in
   which case it fetches, or a local directory, in which case it reads. This port passes a
   directory and raises `WeightsMissing` if the files are absent. A port that fell back to
   fetching would make acquisition a side effect of pressing run, and the point of the
   clearance gate is that the user sees the licences before the bytes arrive, not after.

2. No background remover runs. Upstream's reference script reaches for rembg, whose own
   package licence is not the licence of the u2net weights it downloads on first use. This
   port requires the foreground to be given -- an alpha channel, or a separate mask -- which
   is the `require_mask_or_audited_remover` branch the pipeline already models.

3. Geometry is normalised and then measured, never assumed. See `_normalise` for what is
   corrected and why.

The port reports counts it measured from the mesh it wrote. It does not report the counts the
plan hoped for.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from asset_mania_contracts import DiagnosticCode

from asset_mania_engine_triposr.adapter import (
    EngineRequest,
    EngineResult,
    ReconstructionFailed,
)

#: What `TSR.from_pretrained` reads out of a local weights directory.
CONFIG_NAME = "config.yaml"
WEIGHT_NAME = "model.ckpt"

#: The fourth asset, and the one that is easy to miss. TripoSR's image tokenizer calls
#: `hf_hub_download("facebook/dino-vitb16", "config.json")` inside `configure()`, so loading
#: the model reaches the network even though every weight comes from the local checkpoint --
#: the fetched file is hyperparameters, and the ViT built from it is immediately overwritten
#: by `load_state_dict`. It appears in no requirements file. `acquire_engine_assets.py` fetches
#: it deliberately under its own licence, and this port reads that copy so that a run touches
#: nothing remote.
ARCHITECTURE_CONFIG_NAME = "dino-vitb16-config.json"

#: Grey the reference implementation composites the cut-out foreground onto. Matching it
#: matters: the network was trained on images preprocessed this way, and compositing onto
#: black or white instead shifts every silhouette edge.
BACKGROUND_LEVEL = 0.5

#: Upstream crops to the alpha bounding box and pads to a square of this occupancy. Keeping
#: the reference value keeps the subject at the scale the network expects.
DEFAULT_FOREGROUND_RATIO = 0.85

#: Marching-cubes grid edge. 256 is upstream's default and the resolution its threshold was
#: chosen against.
DEFAULT_MC_RESOLUTION = 256

#: Density level for the isosurface, in the units the triplane decoder emits.
DEFAULT_MC_THRESHOLD = 25.0


class WeightsMissing(Exception):
    """The weights directory does not hold the files the engine needs."""


class _RembgWithheld:
    """Stands in for rembg so the engine imports without it, and fails loudly if reached.

    `tsr/utils.py` imports rembg at module scope even though background removal is one
    optional branch of the reference script. This port never takes that branch, so the
    dependency is withheld rather than installed: rembg's own MIT licence does not cover the
    u2net weights it downloads on first use, and nothing here has read those terms.

    Reaching for a rembg *function* raises. Introspection does not: torch's custom-op
    registration walks `sys.modules` calling `hasattr(module, "__file__")`, so a stub that
    raised on dunder access broke code with no interest in background removal at all. Dunders
    answer with `AttributeError`, which is what "this module does not have that" means, and
    only the public surface fails loudly.

    Returning a silent no-op instead was the other option, and a worse one: it would let a
    caller reconstruct a photo with its background still attached, which is a wrong mesh
    rather than an error.
    """

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ReconstructionFailed(
            f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: rembg.{name} was reached, but "
            f"this port withholds rembg because the licence of the weights it fetches has "
            f"not been verified. Supply a foreground mask instead."
        )


def _withhold_rembg() -> None:
    """Register the stub, unless a real rembg is already installed and therefore chosen."""
    if "rembg" in sys.modules:
        return
    try:  # a rembg the user installed deliberately is theirs to use
        import importlib.util

        if importlib.util.find_spec("rembg") is not None:
            return
    except (ImportError, ValueError):
        pass
    module = ModuleType("rembg")
    module.__getattr__ = _RembgWithheld().__getattr__  # type: ignore[method-assign]
    sys.modules["rembg"] = module


@dataclass(frozen=True, slots=True)
class TripoSRSettings:
    """Where the engine lives and how to run it. All paths are local; none are fetched."""

    engine_root: Path
    weights_dir: Path
    device: str = "cpu"
    mc_resolution: int = DEFAULT_MC_RESOLUTION
    mc_threshold: float = DEFAULT_MC_THRESHOLD
    foreground_ratio: float = DEFAULT_FOREGROUND_RATIO
    chunk_size: int = 8192
    vertex_colors: bool = True
    #: Hub cache holding the pinned architecture config. See `_enforce_offline`.
    hub_cache: Path | None = None

    def describe(self) -> dict[str, Any]:
        """A log-safe description: settings that affect the result, no local paths."""
        return {
            "device": self.device,
            "mc_resolution": self.mc_resolution,
            "mc_threshold": self.mc_threshold,
            "foreground_ratio": self.foreground_ratio,
            "vertex_colors": self.vertex_colors,
        }


def _require_weights(weights_dir: Path) -> tuple[Path, Path]:
    """Resolve the two files the engine loads, or say exactly which one is absent."""
    config = weights_dir / CONFIG_NAME
    weight = weights_dir / WEIGHT_NAME
    missing = [p.name for p in (config, weight) if not p.is_file()]
    if missing:
        raise WeightsMissing(
            f"{DiagnosticCode.ENGINE_UNAVAILABLE.value}: {weights_dir} is missing "
            f"{', '.join(missing)}. Acquire the weights deliberately; this port will not "
            f"download them."
        )
    return config, weight


def _load_foreground(image_path: Path, mask_path: Path | None) -> Any:
    """Build the RGBA the engine's preprocessing expects, from data the caller supplied.

    Refuses an opaque image with no mask rather than reconstructing the background along with
    the subject: TripoSR treats every non-transparent pixel as part of the object, so a photo
    with its background still attached produces a mesh of the room.
    """
    import numpy as np
    from PIL import Image

    image = Image.open(image_path)
    if mask_path is not None:
        rgb = np.asarray(image.convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape != rgb.shape[:2]:
            raise ReconstructionFailed(
                f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: mask is {mask.shape}, "
                f"image is {rgb.shape[:2]}"
            )
        rgba = np.dstack([rgb, mask])
    else:
        if image.mode != "RGBA":
            raise ReconstructionFailed(
                f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: {image_path.name} is "
                f"{image.mode}, and no mask was supplied. Supply an alpha channel or a mask; "
                f"without one the background is reconstructed as part of the subject."
            )
        rgba = np.asarray(image)

    if not (rgba[..., 3] > 0).any():
        raise ReconstructionFailed(
            f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: the mask selects no pixels"
        )
    return rgba


def _preprocess(rgba: Any, foreground_ratio: float, resize_foreground: Any) -> Any:
    """Crop to the subject and composite onto the grey the network was trained against."""
    import numpy as np
    from PIL import Image

    cropped = resize_foreground(Image.fromarray(rgba), foreground_ratio)
    arr = np.asarray(cropped).astype(np.float32) / 255.0
    composited = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * BACKGROUND_LEVEL
    return Image.fromarray((composited * 255.0).astype(np.uint8))


#: Widest boundary loop that counts as extraction noise, as a fraction of the mesh's
#: bounding-box diagonal.
#:
#: Vertex count was the obvious measure and the wrong one: a cube missing an entire face has a
#: four-vertex boundary loop, the same count as a single absent triangle. Signed-volume drift
#: was the second guess and also wrong, because a planar cap across a flat opening adds no
#: volume at all -- it closes the mesh while inventing the surface, which is the exact failure
#: this guard exists to prevent.
#:
#: Spatial span separates them cleanly. Measured on the reference run: 134 loops, widest
#: 3.2% of the diagonal, median 1.6%, against a grid cell of 0.46%. A missing cube face spans
#: 82%. This threshold sits 3x above the worst real hole and 8x below fabrication.
MAX_REPAIRABLE_HOLE_SPAN = 0.10

#: Secondary guard, for a large opening that is not planar and so would add volume when
#: capped. Span catches those too, but the two measures fail differently and neither is
#: expensive.
MAX_REPAIR_VOLUME_DRIFT = 0.01


def _boundary_loop_spans(mesh: Any) -> list[float]:
    """Diagonal extent of each boundary loop, as a fraction of the mesh diagonal, widest first."""
    import numpy as np

    edges, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    boundary = edges[counts == 1]
    if not len(boundary):
        return []

    vertices = np.asarray(mesh.vertices, dtype=float)
    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    if diagonal <= 0:
        return []

    neighbours: dict[int, list[int]] = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))

    seen: set[int] = set()
    spans: list[float] = []
    for start in neighbours:
        if start in seen:
            continue
        loop, stack = [], [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            loop.append(node)
            stack.extend(n for n in neighbours[node] if n not in seen)
        points = vertices[loop]
        spans.append(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) / diagonal)
    return sorted(spans, reverse=True)


def _repair_small_holes(mesh: Any) -> Any:
    """Close extraction-noise holes, and refuse to close anything larger.

    Marching cubes on a density field leaves single missing triangles where the isosurface is
    ambiguous -- on the reference run, 134 of them, enough to make an otherwise sound surface
    report as open while the volume was correct to five decimals. Putting those back is
    repair.

    Capping a large opening is not repair, it is fabrication: the mesh becomes watertight and
    a region the model never reconstructed reads as solid. `MAX_REPAIRABLE_HOLE_SPAN` records
    how that line was drawn and why the two more obvious measures do not draw it.

    All or nothing, because `fill_holes` fills everything: a mesh with one genuinely missing
    region keeps its small holes too and stays `open`, which is the accurate report.
    """
    spans = _boundary_loop_spans(mesh)
    if not spans:
        return mesh
    if spans[0] > MAX_REPAIRABLE_HOLE_SPAN:
        return mesh

    before = float(mesh.volume)
    candidate = mesh.copy()
    candidate.fill_holes()
    candidate.fix_normals()
    if abs(float(candidate.volume) - before) > MAX_REPAIR_VOLUME_DRIFT:
        return mesh
    return candidate


def _normalise(mesh: Any) -> tuple[Any, str]:
    """Merge coincident vertices and orient faces outward, then report what we ended up with.

    Two corrections, both measured on this machine against an analytic sphere rather than
    taken on trust:

    * Marching cubes emits per-cell vertices, so a closed surface arrives as unmerged
      triangle soup and every watertightness check fails on seams that are not really there.
    * The extractor's winding, given TripoSR's sign convention, points normals *inward*: the
      sphere came back with a signed volume of -0.903 against an analytic +0.905. Exporting
      that produces a mesh that renders inside-out, which is the kind of defect that survives
      a triangle count and a file-size check.

    `manifold` is whatever the mesh actually is after those fixes, not what we hoped.
    """
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh = _repair_small_holes(mesh)

    # `closed` / `open` / `unknown` are the states the reconstruction record declares; a value
    # outside them is rejected downstream. `unknown` is the honest answer for a surface whose
    # winding is inconsistent, because then "open" would be claiming more than was measured.
    if mesh.is_watertight:
        manifold = "closed"
    elif mesh.is_winding_consistent:
        manifold = "open"
    else:
        manifold = "unknown"
    return mesh, manifold


@dataclass
class TripoSRPort:
    """Runs a locally installed TripoSR. Constructed by the caller, never self-configuring."""

    settings: TripoSRSettings
    _model: Any = None

    def _import_engine(self) -> tuple[Any, Any]:
        """Import the vendored engine from its checkout, without installing it."""
        root = self.settings.engine_root
        if not (root / "tsr" / "system.py").is_file():
            raise WeightsMissing(
                f"{DiagnosticCode.ENGINE_UNAVAILABLE.value}: {root} does not look like a "
                f"TripoSR checkout (no tsr/system.py)"
            )
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _withhold_rembg()
        self._enforce_offline()
        from tsr.system import TSR
        from tsr.utils import resize_foreground

        return TSR, resize_foreground

    def _enforce_offline(self) -> None:
        """Point the hub client at the local cache and forbid it from leaving.

        `ARCHITECTURE_CONFIG_NAME` explains why this is needed: one `hf_hub_download` sits
        inside the tokenizer's `configure()`, so without this a run silently reaches the
        network. `HF_HUB_OFFLINE` turns that into a `LocalEntryNotFoundError` -- a loud
        failure pointing at the acquisition step -- instead of a quiet fetch.

        The cache is also where the pin lives. The tokenizer asks for `main`, and the
        acquisition step writes `refs/main` to the reviewed commit, so `main` resolves to the
        revision whose licence was read rather than to whatever `main` points at today.
        """
        cache = self.settings.hub_cache
        if cache is not None:
            os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_OFFLINE"] = "1"

    def _ensure_model(self, tsr_cls: Any) -> Any:
        """Load the checkpoint once per port instance."""
        if self._model is None:
            config, _ = _require_weights(self.settings.weights_dir)
            model = tsr_cls.from_pretrained(
                str(config.parent), config_name=CONFIG_NAME, weight_name=WEIGHT_NAME
            )
            model.renderer.set_chunk_size(self.settings.chunk_size)
            model.to(self.settings.device)
            model.eval()
            self._model = model
        return self._model

    def run(self, request: EngineRequest) -> EngineResult:
        """Reconstruct one image into `request.output_path` and report measured counts."""
        import torch

        tsr_cls, resize_foreground = self._import_engine()
        model = self._ensure_model(tsr_cls)

        rgba = _load_foreground(request.image_path, request.mask_path)
        image = _preprocess(rgba, self.settings.foreground_ratio, resize_foreground)

        with torch.no_grad():
            scene_codes = model([image], device=self.settings.device)
            meshes = model.extract_mesh(
                scene_codes,
                self.settings.vertex_colors,
                resolution=self.settings.mc_resolution,
                threshold=self.settings.mc_threshold,
            )

        if not meshes:
            raise ReconstructionFailed(
                f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: the engine returned no mesh"
            )

        mesh, manifold = _normalise(meshes[0])
        if len(mesh.faces) == 0:
            raise ReconstructionFailed(
                f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: the isosurface is empty at "
                f"threshold {self.settings.mc_threshold}"
            )

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(request.output_path), file_type=request.mesh_format)

        return EngineResult(
            triangle_count=len(mesh.faces),
            vertex_count=len(mesh.vertices),
            manifold=manifold,
        )


__all__ = [
    "BACKGROUND_LEVEL",
    "CONFIG_NAME",
    "DEFAULT_FOREGROUND_RATIO",
    "DEFAULT_MC_RESOLUTION",
    "DEFAULT_MC_THRESHOLD",
    "WEIGHT_NAME",
    "TripoSRPort",
    "TripoSRSettings",
    "WeightsMissing",
]
