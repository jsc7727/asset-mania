"""Closed-schema tests for `view-v1`."""

import copy

import pytest
from asset_mania_contracts import canonical_digest
from conftest import load_example


@pytest.fixture
def view_validator(validator_for):
    return validator_for("view", "1.0")


@pytest.fixture
def view():
    return load_example("view-v1")


def test_example_is_valid_and_self_sealed(view_validator, view) -> None:
    assert list(view_validator.iter_errors(view)) == []
    preimage = {key: value for key, value in view.items() if key != "view_sha256"}
    assert canonical_digest(preimage) == view["view_sha256"]


def test_view_is_never_upload_eligible_in_v0_2(view_validator, view) -> None:
    assert view["upload_eligible"] is False
    assert list(view_validator.iter_errors({**view, "upload_eligible": True}))


def test_user_supplied_view_is_user_content(view_validator, view) -> None:
    assert view["sensitivity"] == "user-content"
    assert list(view_validator.iter_errors({**view, "sensitivity": "local-sensitive"}))


def test_alignment_stays_declared_unverified_without_a_fiducial_fixture(
    view_validator, view
) -> None:
    assert view["alignment"]["status"] == "declared_unverified"
    assert view["alignment"]["transform"] == "identity"

    mutated = copy.deepcopy(view)
    mutated["alignment"]["status"] = "verified"
    assert list(view_validator.iter_errors(mutated))

    mutated = copy.deepcopy(view)
    mutated["alignment"]["transform"] = "affine"
    assert list(view_validator.iter_errors(mutated))


def test_alignment_is_a_closed_record(view_validator, view) -> None:
    mutated = copy.deepcopy(view)
    mutated["alignment"]["note"] = "eyeballed it"
    assert list(view_validator.iter_errors(mutated))


def test_view_binds_the_condition_manifest_and_bundle_digests(view_validator, view) -> None:
    condition = load_example("manifest-v2-condition")
    bundle = load_example("conditioning-bundle-v1")
    assert view["condition_manifest_sha256"] == canonical_digest(condition)
    assert view["conditioning_bundle_sha256"] == bundle["bundle_sha256"]

    for key in ("condition_manifest_sha256", "conditioning_bundle_sha256"):
        assert list(view_validator.iter_errors({**view, key: "latest"})), key


def test_view_carries_no_image_bytes_path_or_exif(view_validator, view) -> None:
    for key, value in (
        ("image_bytes", "iVBORw0KGgo="),
        ("source_path", "/Users/example/photos/private.png"),
        ("basename", "private.png"),
        ("exif", {"GPSLatitude": 37.5}),
    ):
        assert list(view_validator.iter_errors({**view, key: value})), key


def test_validation_profile_and_semantic_digest_are_required(view_validator, view) -> None:
    assert view["validation"]["profile"] == "view-v1"
    assert view["validation"]["semantic_digest"] is not None

    mutated = copy.deepcopy(view)
    del mutated["validation"]["semantic_digest"]
    assert list(view_validator.iter_errors(mutated))
