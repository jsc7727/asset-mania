"""Closed-schema tests for `blender-response-v1`."""

import copy

import pytest
from asset_mania_contracts import canonical_digest
from conftest import example_names, load_example

OPERATIONS = ["preflight", "condition", "bake", "export", "validate"]


@pytest.fixture
def response_validator(validator_for):
    return validator_for("blender-response", "1.0")


def test_every_operation_has_one_normative_example() -> None:
    assert example_names("blender-response-v1-") == sorted(
        [f"blender-response-v1-{operation}" for operation in OPERATIONS]
        + ["blender-response-v1-failed"]
    )


def test_a_failed_response_reports_no_metrics(response_validator) -> None:
    """A failed run has no inventory; null is honest where zeroed counts would not be."""
    response = load_example("blender-response-v1-failed")
    assert list(response_validator.iter_errors(response)) == []
    assert response["status"] == "failed"
    assert response["metrics"] is None
    assert response["outputs"] == []
    assert response["diagnostics"] == sorted(response["diagnostics"])


def test_a_succeeded_response_may_not_report_null_metrics(response_validator) -> None:
    response = load_example("blender-response-v1-preflight")
    assert list(response_validator.iter_errors({**response, "metrics": None}))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_operation_example_is_valid_and_self_sealed(response_validator, operation: str) -> None:
    response = load_example(f"blender-response-v1-{operation}")
    assert list(response_validator.iter_errors(response)) == []
    assert response["operation"] == operation
    assert response["metrics"]["kind"] == operation
    preimage = {key: value for key, value in response.items() if key != "response_sha256"}
    assert canonical_digest(preimage) == response["response_sha256"]


@pytest.mark.parametrize("operation", OPERATIONS)
def test_metrics_from_another_operation_are_rejected(response_validator, operation: str) -> None:
    response = load_example(f"blender-response-v1-{operation}")
    donor = next(other for other in OPERATIONS if other != operation)
    response["metrics"] = load_example(f"blender-response-v1-{donor}")["metrics"]
    assert list(response_validator.iter_errors(response))


@pytest.mark.parametrize("operation", OPERATIONS)
def test_unknown_metrics_fail_validation(response_validator, operation: str) -> None:
    response = load_example(f"blender-response-v1-{operation}")
    response["metrics"]["gpu_seconds"] = 12
    assert list(response_validator.iter_errors(response))


def test_response_carries_no_timestamp_path_or_traceback(response_validator) -> None:
    response = load_example("blender-response-v1-preflight")
    for key, value in (
        ("created_at", "2026-08-19T09:00:00Z"),
        ("source_path", "/Users/example/scenes/private.blend"),
        ("traceback", "Traceback (most recent call last): ..."),
        ("stdout", "Blender 5.2.0"),
        ("datablock_names", ["Body_LOD0"]),
    ):
        assert list(response_validator.iter_errors({**response, key: value})), key


def test_portable_labels_are_sorted_unique_and_pattern_bound(response_validator) -> None:
    response = load_example("blender-response-v1-preflight")
    assert response["portable_labels"] == sorted(response["portable_labels"])

    duplicated = copy.deepcopy(response)
    duplicated["portable_labels"] = ["camera-1", "camera-1"]
    assert list(response_validator.iter_errors(duplicated))

    for label in ("Body_LOD0", "mesh-0", "mesh_1", "light-1", "mesh-01"):
        mutated = copy.deepcopy(response)
        mutated["portable_labels"] = [label]
        assert list(response_validator.iter_errors(mutated)), label


def test_output_paths_stay_relative(response_validator) -> None:
    response = load_example("blender-response-v1-bake")
    for path in ("/tmp/baked-texture.png", "../baked-texture.png", "a/../../b.png"):
        mutated = copy.deepcopy(response)
        mutated["outputs"][0]["path"] = path
        assert list(response_validator.iter_errors(mutated)), path


def test_coverage_ratio_stays_within_the_unit_interval(response_validator) -> None:
    response = load_example("blender-response-v1-bake")
    for ratio in (-0.1, 1.5):
        mutated = copy.deepcopy(response)
        mutated["metrics"]["coverage_ratio"] = ratio
        assert list(response_validator.iter_errors(mutated)), ratio


def test_projection_error_is_nonnegative_or_null(response_validator) -> None:
    response = load_example("blender-response-v1-condition")
    without_oracle = copy.deepcopy(response)
    without_oracle["metrics"]["projection_max_error_pixels"] = None
    del without_oracle["response_sha256"]
    without_oracle["response_sha256"] = canonical_digest(without_oracle)
    assert list(response_validator.iter_errors(without_oracle)) == []

    negative = copy.deepcopy(response)
    negative["metrics"]["projection_max_error_pixels"] = -1.0
    assert list(response_validator.iter_errors(negative))


def test_request_identifier_uses_the_portable_pattern(response_validator) -> None:
    response = load_example("blender-response-v1-validate")
    for request_id in ("request/validate", "../request", "request validate", ""):
        assert list(response_validator.iter_errors({**response, "request_id": request_id}))
