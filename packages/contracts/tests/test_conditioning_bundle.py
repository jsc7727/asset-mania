"""Closed-schema tests for `conditioning-bundle-v1`."""

import copy

import pytest
from asset_mania_contracts import (
    PASS_COLOR_SPACES,
    PASS_MEDIA_TYPES,
    PASS_ROLES,
    canonical_digest,
)
from conftest import load_example


@pytest.fixture
def bundle_validator(validator_for):
    return validator_for("conditioning-bundle", "1.0")


@pytest.fixture
def bundle():
    return load_example("conditioning-bundle-v1")


def test_example_is_valid_and_self_sealed(bundle_validator, bundle) -> None:
    assert list(bundle_validator.iter_errors(bundle)) == []
    preimage = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    assert canonical_digest(preimage) == bundle["bundle_sha256"]


def test_passes_keep_their_fixed_role_order(bundle_validator, bundle) -> None:
    assert [item["role"] for item in bundle["passes"]] == PASS_ROLES

    reordered = copy.deepcopy(bundle)
    reordered["passes"][0], reordered["passes"][1] = reordered["passes"][1], reordered["passes"][0]
    assert list(bundle_validator.iter_errors(reordered))

    dropped = copy.deepcopy(bundle)
    del dropped["passes"][-1]
    assert list(bundle_validator.iter_errors(dropped))


@pytest.mark.parametrize("role", PASS_ROLES)
def test_each_pass_role_pins_one_media_type_and_color_space(
    bundle_validator, bundle, role: str
) -> None:
    index = PASS_ROLES.index(role)
    assert bundle["passes"][index]["media_type"] == PASS_MEDIA_TYPES[role]
    assert bundle["passes"][index]["color_space"] == PASS_COLOR_SPACES[role]

    wrong_media = copy.deepcopy(bundle)
    wrong_media["passes"][index]["media_type"] = "image/tiff"
    assert list(bundle_validator.iter_errors(wrong_media))

    wrong_space = copy.deepcopy(bundle)
    wrong_space["passes"][index]["color_space"] = "filmic"
    assert list(bundle_validator.iter_errors(wrong_space))


def test_pass_paths_stay_relative(bundle_validator, bundle) -> None:
    for path in ("/tmp/passes/beauty.exr", "../beauty.exr", "passes/../../beauty.exr"):
        mutated = copy.deepcopy(bundle)
        mutated["passes"][0]["path"] = path
        assert list(bundle_validator.iter_errors(mutated)), path


def test_axis_and_pixel_conventions_are_pinned(bundle_validator, bundle) -> None:
    assert bundle["pixel_origin"] == "top_left"
    assert bundle["axes"]["world"] == {"handedness": "right", "up": "+Z", "forward": "-Y"}
    assert bundle["axes"]["camera"] == {"right": "+X", "up": "+Y", "view": "-Z"}

    for pointer, value in (
        (("pixel_origin",), "bottom_left"),
        (("axes", "world", "up"), "+Y"),
        (("axes", "camera", "view"), "+Z"),
    ):
        mutated = copy.deepcopy(bundle)
        target = mutated
        for key in pointer[:-1]:
            target = target[key]
        target[pointer[-1]] = value
        assert list(bundle_validator.iter_errors(mutated)), pointer


@pytest.mark.parametrize(
    "name", ["camera_to_world", "world_to_camera", "projection", "world_to_clip"]
)
def test_each_matrix_is_sixteen_row_major_numbers(bundle_validator, bundle, name: str) -> None:
    assert len(bundle["matrices"][name]) == 16
    assert bundle["matrices"]["layout"] == "row_major"

    short = copy.deepcopy(bundle)
    short["matrices"][name] = short["matrices"][name][:15]
    assert list(bundle_validator.iter_errors(short))

    stringly = copy.deepcopy(bundle)
    stringly["matrices"][name] = ["0.0"] * 16
    assert list(bundle_validator.iter_errors(stringly))


def test_perspective_camera_requires_lens_and_forbids_ortho_scale(bundle_validator, bundle) -> None:
    assert bundle["camera"]["projection_type"] == "perspective"

    without_lens = copy.deepcopy(bundle)
    without_lens["camera"]["lens_mm"] = None
    assert list(bundle_validator.iter_errors(without_lens))

    with_ortho = copy.deepcopy(bundle)
    with_ortho["camera"]["ortho_scale"] = 2.0
    assert list(bundle_validator.iter_errors(with_ortho))


def test_orthographic_camera_requires_ortho_scale_and_forbids_lens(
    bundle_validator, bundle
) -> None:
    orthographic = copy.deepcopy(bundle)
    orthographic["camera"]["projection_type"] = "orthographic"
    orthographic["camera"]["lens_mm"] = None
    orthographic["camera"]["ortho_scale"] = 2.0
    del orthographic["bundle_sha256"]
    orthographic["bundle_sha256"] = canonical_digest(orthographic)
    assert list(bundle_validator.iter_errors(orthographic)) == []

    orthographic["camera"]["lens_mm"] = 50.0
    assert list(bundle_validator.iter_errors(orthographic))


def test_depth_and_normal_semantics_are_pinned(bundle_validator, bundle) -> None:
    assert bundle["depth"]["space"] == "camera_euclidean_distance"
    assert bundle["depth"]["unit"] == "meters"
    assert bundle["normal"]["space"] == "world"
    assert bundle["normal"]["channels"] == ["x", "y", "z"]

    mutated = copy.deepcopy(bundle)
    mutated["depth"]["space"] = "z_buffer"
    assert list(bundle_validator.iter_errors(mutated))

    mutated = copy.deepcopy(bundle)
    mutated["normal"]["space"] = "tangent"
    assert list(bundle_validator.iter_errors(mutated))


def test_bundle_is_closed_against_private_scene_identity(bundle_validator, bundle) -> None:
    for key, value in (
        ("source_path", "/Users/example/scenes/private.blend"),
        ("datablock_names", ["Body_LOD0"]),
        ("blend_file_basename", "private.blend"),
    ):
        assert list(bundle_validator.iter_errors({**bundle, key: value})), key

    mutated = copy.deepcopy(bundle)
    mutated["selection"]["blender_object_name"] = "Body_LOD0"
    assert list(bundle_validator.iter_errors(mutated))
