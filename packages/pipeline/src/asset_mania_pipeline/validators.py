"""Apache-side validation of a conditioning bundle.

The worker measures; this layer re-checks what can be checked without decoding EXR
pixels. Apache code never opens an EXR: it verifies the bundle's declared identities,
semantics, ordering, digests, and the file bytes on disk, and leaves pixel statistics to
the worker response that produced them.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from asset_mania_contracts import (
    PASS_COLOR_SPACES,
    PASS_MEDIA_TYPES,
    PASS_ROLES,
    canonical_digest,
)

from .artifacts import PathEscape, contained_path
from .hashing import sha256_file

BUNDLE_PROFILE = "conditioning-bundle-v1"
SCENE_STATE_ROLE = "scene_state_blend"
_INVALID = "PASS_INVALID"


class BundleInvalid(Exception):
    """A conditioning bundle does not describe a publishable set of passes."""


def _require(condition: bool, message: str, *, code: str = _INVALID) -> None:
    if not condition:
        raise BundleInvalid(f"{code}: {message}")


def verify_bundle_seal(bundle: Mapping[str, Any]) -> None:
    """The bundle digest must cover every other field."""
    preimage = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    _require(
        canonical_digest(preimage) == bundle.get("bundle_sha256"),
        "bundle_sha256 does not match the bundle content",
    )


def verify_pass_inventory(bundle: Mapping[str, Any]) -> None:
    """Every declared pass, in the declared order, with its bound media type and space."""
    passes = bundle["passes"]
    roles = [item["role"] for item in passes]
    _require(roles == list(PASS_ROLES), f"pass roles must be exactly {list(PASS_ROLES)}")

    for item in passes:
        role = item["role"]
        _require(
            item["media_type"] == PASS_MEDIA_TYPES[role],
            f"{role} must be {PASS_MEDIA_TYPES[role]}",
        )
        _require(
            item["color_space"] == PASS_COLOR_SPACES[role],
            f"{role} must record color space {PASS_COLOR_SPACES[role]}",
        )
        _require(item["upload_eligible"] is True, f"{role} must be upload eligible")
        _require(item["byte_size"] > 0, f"{role} must not be empty")

    digests = [item["sha256"] for item in passes]
    _require(len(set(digests)) == len(digests), "two passes share a digest")


def verify_semantics(bundle: Mapping[str, Any]) -> None:
    """The semantics a downstream stage relies on, none of them defaulted."""
    _require(bundle["pixel_origin"] == "top_left", "pixel origin must be top_left")
    _require(bundle["pixel_aspect"] == [1.0, 1.0], "pixel aspect must be 1:1")

    width, height = bundle["resolution"]
    _require(width > 0 and height > 0, "resolution must be positive")

    depth = bundle["depth"]
    _require(depth["space"] == "camera_euclidean_distance", "depth must be camera-euclidean")
    _require(depth["unit"] == "meters", "depth must be in meters")
    _require(
        depth["background"] == "invalid_by_mask",
        "background depth validity must be determined by the mask",
    )
    _require(
        depth["valid_max_meters"] > depth["valid_min_meters"],
        "the depth valid range must be ascending",
    )

    normal = bundle["normal"]
    _require(normal["space"] == "world", "normals must record world space")
    _require(normal["channels"] == ["x", "y", "z"], "normal channels must be x, y, z")
    _require(normal["foreground_unit_expected"] is True, "foreground normals must be unit")

    mask = bundle["mask"]
    _require(mask["target_object_index"] == 1, "the target must own reserved index 1")
    _require((mask["foreground"], mask["background"]) == (255, 0), "the mask must be binary")
    _require(mask["antialiasing"] == "none", "the mask must not be antialiased")
    _require(mask["pass_alpha_threshold"] == 0.5, "the alpha threshold is fixed at 0.5")


def verify_camera(bundle: Mapping[str, Any]) -> None:
    """Matrix shape, layout, and finiteness. The values come from Blender's own API."""
    matrices = bundle["matrices"]
    _require(matrices["layout"] == "row_major", "matrices must be row major")
    for name in ("camera_to_world", "world_to_camera", "projection", "world_to_clip"):
        values = matrices[name]
        _require(len(values) == 16, f"{name} must have sixteen components")
        _require(
            all(isinstance(value, (int, float)) for value in values),
            f"{name} must be numeric",
        )

    camera = bundle["camera"]
    if camera["projection_type"] == "perspective":
        _require(camera["lens_mm"] is not None, "a perspective camera must record its lens")
        _require(camera["ortho_scale"] is None, "a perspective camera has no ortho scale")
    else:
        _require(camera["lens_mm"] is None, "an orthographic camera records no lens")
        _require(
            camera["ortho_scale"] is not None,
            "an orthographic camera must record its scale",
        )
    _require(
        camera["clip_end_meters"] > camera["clip_start_meters"],
        "the clipping range must be ascending",
    )


def verify_measured_passes(bundle: Mapping[str, Any], metrics: Mapping[str, Any]) -> None:
    """Cross-check the worker's pixel measurements against the bundle it published."""
    width, height = bundle["resolution"]
    _require(metrics["width"] == width, "the metrics width contradicts the bundle")
    _require(metrics["height"] == height, "the metrics height contradicts the bundle")

    foreground = metrics["foreground_pixel_count"]
    _require(foreground > 0, "an empty mask has no usable foreground")
    _require(foreground <= width * height, "the foreground exceeds the frame")
    _require(
        metrics["finite_foreground_depth_count"] == foreground,
        "every mask-foreground pixel must have finite foreground depth",
    )
    interior = metrics.get("interior_pixel_count")
    if interior is not None:
        _require(
            metrics["interior_unit_normal_count"] == interior,
            "every eroded-interior pixel must carry a unit normal",
        )
    _require(
        metrics["interior_unit_normal_count"] <= foreground,
        "more interior normals than foreground pixels",
    )
    for name in ("geometry_digest", "uv_digest", "pose_digest"):
        _require(len(metrics[name]) == 64, f"{name} must be a sha256")


def verify_artifacts_on_disk(bundle: Mapping[str, Any], run_directory: Path) -> None:
    """Rehash every declared pass inside its own run."""
    for item in bundle["passes"]:
        try:
            path = contained_path(run_directory, item["path"])
        except PathEscape as error:
            raise BundleInvalid(f"{_INVALID}: a pass path leaves the run") from error
        _require(path.is_file(), f"{item['role']} is missing on disk")
        _require(
            sha256_file(path) == item["sha256"],
            f"{item['role']} does not match its recorded digest",
        )
        _require(
            path.stat().st_size == item["byte_size"],
            f"{item['role']} does not match its recorded size",
        )


def verify_local_scene_is_not_upload_eligible(outputs: Sequence[Mapping[str, Any]]) -> None:
    """`scene-state.blend` is local-sensitive and never leaves the machine."""
    local = [item for item in outputs if item["role"] == SCENE_STATE_ROLE]
    _require(len(local) == 1, "exactly one derived scene state must be published")
    _require(
        not local[0]["path"].startswith("artifacts/conditioning/"),
        "the derived scene must not sit in the upload-eligible directory",
    )


def validate_conditioning_bundle(
    bundle: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any],
    outputs: Sequence[Mapping[str, Any]],
    run_directory: Path | None = None,
) -> None:
    """Run every bundle check. Raises `BundleInvalid` on the first failure."""
    verify_bundle_seal(bundle)
    verify_pass_inventory(bundle)
    verify_semantics(bundle)
    verify_camera(bundle)
    verify_measured_passes(bundle, metrics)
    verify_local_scene_is_not_upload_eligible(outputs)
    if run_directory is not None:
        verify_artifacts_on_disk(bundle, run_directory)
