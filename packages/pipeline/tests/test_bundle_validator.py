"""Conditioning-bundle validation, one failure class per test."""

import copy
import json
from pathlib import Path

import pytest
from asset_mania_contracts import PASS_ROLES, canonical_digest, canonical_json
from asset_mania_pipeline import (
    BundleInvalid,
    sha256_bytes,
    validate_conditioning_bundle,
    verify_artifacts_on_disk,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"


def _reseal(bundle: dict) -> dict:
    preimage = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    return {**preimage, "bundle_sha256": canonical_digest(preimage)}


@pytest.fixture
def bundle() -> dict:
    return json.loads((EXAMPLES / "conditioning-bundle-v1.json").read_text(encoding="utf-8"))


@pytest.fixture
def metrics(bundle) -> dict:
    width, height = bundle["resolution"]
    return {
        "kind": "condition",
        "width": width,
        "height": height,
        "foreground_pixel_count": 476,
        "finite_foreground_depth_count": 476,
        "interior_pixel_count": 319,
        "interior_unit_normal_count": 319,
        "projection_max_error_pixels": 0.2,
        "geometry_digest": "c2" * 32,
        "uv_digest": "c3" * 32,
        "pose_digest": "c4" * 32,
    }


@pytest.fixture
def outputs() -> list[dict]:
    return [
        {"role": "conditioning_bundle", "path": "artifacts/conditioning/bundle.json"},
        {"role": "scene_state_blend", "path": "artifacts/local/scene-state.blend"},
    ]


def _validate(bundle, metrics, outputs, **kwargs) -> None:
    validate_conditioning_bundle(bundle, metrics=metrics, outputs=outputs, **kwargs)


def test_the_normative_bundle_validates(bundle, metrics, outputs) -> None:
    _validate(bundle, metrics, outputs)


# --- Seal and inventory --------------------------------------------------------


def test_an_edited_bundle_fails_the_seal(bundle, metrics, outputs) -> None:
    bundle["frame"] = bundle["frame"] + 1
    with pytest.raises(BundleInvalid, match="bundle_sha256"):
        _validate(bundle, metrics, outputs)


def test_a_missing_pass_is_rejected(bundle, metrics, outputs) -> None:
    del bundle["passes"][3]
    with pytest.raises(BundleInvalid, match="pass roles"):
        _validate(_reseal(bundle), metrics, outputs)


def test_an_unstable_pass_order_is_rejected(bundle, metrics, outputs) -> None:
    bundle["passes"][0], bundle["passes"][1] = bundle["passes"][1], bundle["passes"][0]
    with pytest.raises(BundleInvalid, match="pass roles"):
        _validate(_reseal(bundle), metrics, outputs)


@pytest.mark.parametrize("role", PASS_ROLES)
def test_a_wrong_media_type_is_rejected(bundle, metrics, outputs, role: str) -> None:
    index = PASS_ROLES.index(role)
    bundle["passes"][index]["media_type"] = "image/tiff"
    with pytest.raises(BundleInvalid, match=role):
        _validate(_reseal(bundle), metrics, outputs)


@pytest.mark.parametrize("role", PASS_ROLES)
def test_a_wrong_color_space_is_rejected(bundle, metrics, outputs, role: str) -> None:
    index = PASS_ROLES.index(role)
    bundle["passes"][index]["color_space"] = "filmic"
    with pytest.raises(BundleInvalid, match="color space"):
        _validate(_reseal(bundle), metrics, outputs)


def test_an_empty_pass_file_is_rejected(bundle, metrics, outputs) -> None:
    bundle["passes"][0]["byte_size"] = 0
    with pytest.raises(BundleInvalid, match="must not be empty"):
        _validate(_reseal(bundle), metrics, outputs)


def test_two_passes_sharing_a_digest_are_rejected(bundle, metrics, outputs) -> None:
    bundle["passes"][1]["sha256"] = bundle["passes"][0]["sha256"]
    with pytest.raises(BundleInvalid, match="share a digest"):
        _validate(_reseal(bundle), metrics, outputs)


# --- Semantics ------------------------------------------------------------------


def test_a_non_top_left_pixel_origin_is_rejected(bundle, metrics, outputs) -> None:
    bundle["pixel_origin"] = "bottom_left"
    with pytest.raises(BundleInvalid, match="top_left"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_non_unit_pixel_aspect_is_rejected(bundle, metrics, outputs) -> None:
    bundle["pixel_aspect"] = [1.0, 2.0]
    with pytest.raises(BundleInvalid, match="pixel aspect"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_planar_depth_claim_is_rejected(bundle, metrics, outputs) -> None:
    bundle["depth"]["space"] = "camera_planar_z"
    with pytest.raises(BundleInvalid, match="camera-euclidean"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_background_depth_claim_other_than_the_mask_is_rejected(bundle, metrics, outputs) -> None:
    bundle["depth"]["background"] = "far_clip"
    with pytest.raises(BundleInvalid, match="determined by the mask"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_descending_depth_range_is_rejected(bundle, metrics, outputs) -> None:
    bundle["depth"]["valid_min_meters"] = bundle["depth"]["valid_max_meters"] + 1.0
    with pytest.raises(BundleInvalid, match="ascending"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_tangent_space_normal_claim_is_rejected(bundle, metrics, outputs) -> None:
    bundle["normal"]["space"] = "tangent"
    with pytest.raises(BundleInvalid, match="world space"):
        _validate(_reseal(bundle), metrics, outputs)


def test_an_antialiased_mask_claim_is_rejected(bundle, metrics, outputs) -> None:
    bundle["mask"]["antialiasing"] = "fxaa"
    with pytest.raises(BundleInvalid, match="antialiased"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_non_binary_mask_is_rejected(bundle, metrics, outputs) -> None:
    bundle["mask"]["foreground"] = 200
    with pytest.raises(BundleInvalid, match="binary"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_reassigned_reserved_index_is_rejected(bundle, metrics, outputs) -> None:
    bundle["mask"]["target_object_index"] = 2
    with pytest.raises(BundleInvalid, match="reserved index 1"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_changed_alpha_threshold_is_rejected(bundle, metrics, outputs) -> None:
    bundle["mask"]["pass_alpha_threshold"] = 0.25
    with pytest.raises(BundleInvalid, match="alpha threshold"):
        _validate(_reseal(bundle), metrics, outputs)


# --- Camera ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["camera_to_world", "world_to_camera", "projection", "world_to_clip"]
)
def test_a_wrong_matrix_shape_is_rejected(bundle, metrics, outputs, name: str) -> None:
    bundle["matrices"][name] = bundle["matrices"][name][:15]
    with pytest.raises(BundleInvalid, match="sixteen components"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_non_numeric_matrix_is_rejected(bundle, metrics, outputs) -> None:
    bundle["matrices"]["projection"] = ["0.0"] * 16
    with pytest.raises(BundleInvalid, match="numeric"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_column_major_claim_is_rejected(bundle, metrics, outputs) -> None:
    bundle["matrices"]["layout"] = "column_major"
    with pytest.raises(BundleInvalid, match="row major"):
        _validate(_reseal(bundle), metrics, outputs)


def test_a_perspective_camera_without_a_lens_is_rejected(bundle, metrics, outputs) -> None:
    bundle["camera"]["lens_mm"] = None
    with pytest.raises(BundleInvalid, match="lens"):
        _validate(_reseal(bundle), metrics, outputs)


def test_an_inverted_clip_range_is_rejected(bundle, metrics, outputs) -> None:
    bundle["camera"]["clip_end_meters"] = 0.01
    with pytest.raises(BundleInvalid, match="clipping range"):
        _validate(_reseal(bundle), metrics, outputs)


# --- Measured passes ------------------------------------------------------------


def test_an_empty_mask_is_rejected(bundle, metrics, outputs) -> None:
    metrics["foreground_pixel_count"] = 0
    metrics["finite_foreground_depth_count"] = 0
    with pytest.raises(BundleInvalid, match="empty mask"):
        _validate(bundle, metrics, outputs)


def test_a_foreground_larger_than_the_frame_is_rejected(bundle, metrics, outputs) -> None:
    width, height = bundle["resolution"]
    metrics["foreground_pixel_count"] = width * height + 1
    metrics["finite_foreground_depth_count"] = metrics["foreground_pixel_count"]
    with pytest.raises(BundleInvalid, match="exceeds the frame"):
        _validate(bundle, metrics, outputs)


def test_a_non_finite_foreground_depth_is_rejected(bundle, metrics, outputs) -> None:
    metrics["finite_foreground_depth_count"] = metrics["foreground_pixel_count"] - 1
    with pytest.raises(BundleInvalid, match="finite foreground depth"):
        _validate(bundle, metrics, outputs)


def test_a_non_unit_interior_normal_is_rejected(bundle, metrics, outputs) -> None:
    metrics["interior_unit_normal_count"] = metrics["interior_pixel_count"] - 1
    with pytest.raises(BundleInvalid, match="unit normal"):
        _validate(bundle, metrics, outputs)


def test_metrics_that_contradict_the_bundle_resolution_are_rejected(
    bundle, metrics, outputs
) -> None:
    metrics["width"] = bundle["resolution"][0] + 1
    with pytest.raises(BundleInvalid, match="contradicts the bundle"):
        _validate(bundle, metrics, outputs)


# --- Upload eligibility and disk state -----------------------------------------


def test_the_derived_scene_must_not_be_upload_eligible(bundle, metrics, outputs) -> None:
    outputs[1]["path"] = "artifacts/conditioning/scene-state.blend"
    with pytest.raises(BundleInvalid, match="upload-eligible directory"):
        _validate(bundle, metrics, outputs)


def test_a_missing_derived_scene_is_rejected(bundle, metrics, outputs) -> None:
    with pytest.raises(BundleInvalid, match="exactly one derived scene"):
        _validate(bundle, metrics, outputs[:1])


def test_declared_passes_are_rehashed_on_disk(bundle, tmp_path: Path) -> None:
    for item in bundle["passes"]:
        path = tmp_path / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{item['role']} bytes\n".encode()
        path.write_bytes(payload)
        item["sha256"] = sha256_bytes(payload)
        item["byte_size"] = len(payload)

    sealed = _reseal(bundle)
    verify_artifacts_on_disk(sealed, tmp_path)

    tampered = copy.deepcopy(sealed)
    (tmp_path / tampered["passes"][0]["path"]).write_bytes(b"tampered\n")
    with pytest.raises(BundleInvalid, match="recorded digest"):
        verify_artifacts_on_disk(tampered, tmp_path)


def test_a_pass_path_may_not_leave_the_run(bundle, tmp_path: Path) -> None:
    bundle["passes"][0]["path"] = "../escape.exr"
    with pytest.raises(BundleInvalid, match="leaves the run"):
        verify_artifacts_on_disk(_reseal(bundle), tmp_path)


def test_a_missing_pass_file_is_rejected(bundle, tmp_path: Path) -> None:
    with pytest.raises(BundleInvalid, match="missing on disk"):
        verify_artifacts_on_disk(bundle, tmp_path)


def test_the_bundle_is_canonically_serializable(bundle) -> None:
    assert canonical_json(bundle).endswith("}\n")
