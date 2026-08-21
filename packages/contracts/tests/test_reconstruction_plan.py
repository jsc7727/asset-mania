"""Closed-schema and builder tests for `reconstruction-plan-v1`."""

import copy

import pytest
from asset_mania_contracts import (
    MESH_FORMATS,
    build_reconstruction_plan,
    canonical_digest,
)
from conftest import load_example


@pytest.fixture
def plan_validator(validator_for):
    return validator_for("reconstruction-plan", "1.0")


@pytest.fixture
def plan():
    return load_example("reconstruction-plan-v1")


def _build(plan, **overrides):
    arguments = {
        "engine": plan["engine"],
        "engine_profile": plan["engine_profile"],
        "clearance_sha256": plan["clearance_sha256"],
        "source_image_sha256": plan["source_image_sha256"],
        "source_width": plan["source_width"],
        "source_height": plan["source_height"],
        "alpha": plan["alpha"],
        "mask_sha256": plan["mask_sha256"],
        "background_removal_clearance_sha256": plan["background_removal_clearance_sha256"],
        "asset_kind": plan["asset_kind"],
        "subject": plan["subject"],
        "rights_receipt_sha256": plan["rights_receipt_sha256"],
        "expected_output": plan["expected_output"],
    }
    arguments.update(overrides)
    return build_reconstruction_plan(**arguments)


def test_the_example_is_valid_and_self_sealed(plan_validator, plan) -> None:
    assert list(plan_validator.iter_errors(plan)) == []
    preimage = {k: v for k, v in plan.items() if k != "plan_sha256"}
    assert canonical_digest(preimage) == plan["plan_sha256"]


def test_the_real_person_example_is_valid(plan_validator) -> None:
    real_person = load_example("reconstruction-plan-v1-real-person")
    assert list(plan_validator.iter_errors(real_person)) == []
    assert real_person["subject"] == "real_person"
    assert real_person["rights_receipt_sha256"] is not None


def test_the_builder_reproduces_both_examples() -> None:
    for name in ("reconstruction-plan-v1", "reconstruction-plan-v1-real-person"):
        expected = load_example(name)
        assert _build(expected) == expected, name


# --- The mask requirement ---------------------------------------------------------


def test_a_plan_with_neither_a_mask_nor_a_background_remover_is_refused(
    plan_validator, plan
) -> None:
    """A single-image reconstructor handed a full scene reconstructs the scene."""
    mutated = {
        **copy.deepcopy(plan),
        "mask_sha256": None,
        "background_removal_clearance_sha256": None,
    }
    assert list(plan_validator.iter_errors(mutated))

    with pytest.raises(ValueError, match="MASK_REQUIRED"):
        _build(plan, mask_sha256=None, background_removal_clearance_sha256=None)


def test_a_mask_alone_is_sufficient(plan_validator, plan) -> None:
    assert plan["mask_sha256"] is not None
    assert plan["background_removal_clearance_sha256"] is None
    assert list(plan_validator.iter_errors(plan)) == []


def test_an_audited_background_remover_alone_is_sufficient(plan_validator, plan) -> None:
    built = _build(
        plan,
        mask_sha256=None,
        background_removal_clearance_sha256=plan["clearance_sha256"],
    )
    assert list(plan_validator.iter_errors(built)) == []


# --- Declarations ------------------------------------------------------------------


def test_an_unknown_subject_is_blocked(plan) -> None:
    with pytest.raises(ValueError, match="SUBJECT_DECLARATION_REQUIRED"):
        _build(plan, subject="unknown")


def test_an_undeclared_subject_is_refused_by_the_schema(plan_validator, plan) -> None:
    for subject in ("unknown", "probably_a_person"):
        assert list(plan_validator.iter_errors({**plan, "subject": subject})), subject


def test_a_real_person_without_a_receipt_is_refused(plan_validator, plan) -> None:
    with pytest.raises(ValueError, match="FACE_RIGHTS_CONFIRMATION_REQUIRED"):
        _build(plan, subject="real_person", rights_receipt_sha256=None)

    mutated = {**plan, "subject": "real_person", "rights_receipt_sha256": None}
    assert list(plan_validator.iter_errors(mutated))


def test_a_receipt_on_a_non_person_plan_is_refused(plan_validator, plan) -> None:
    with pytest.raises(ValueError, match="does not apply"):
        _build(plan, rights_receipt_sha256="f" * 64)

    mutated = {**plan, "rights_receipt_sha256": "f" * 64}
    assert list(plan_validator.iter_errors(mutated))


def test_an_undeclared_asset_kind_is_refused(plan_validator, plan) -> None:
    with pytest.raises(ValueError, match="asset_kind"):
        _build(plan, asset_kind="inferred")
    assert list(plan_validator.iter_errors({**plan, "asset_kind": "inferred"}))


# --- Output and policy ---------------------------------------------------------------


@pytest.mark.parametrize("mesh_format", MESH_FORMATS)
def test_every_declared_mesh_format_is_accepted(plan_validator, plan, mesh_format: str) -> None:
    built = _build(plan, expected_output={**plan["expected_output"], "mesh_format": mesh_format})
    assert list(plan_validator.iter_errors(built)) == []


def test_an_undeclared_mesh_format_is_refused(plan_validator, plan) -> None:
    with pytest.raises(ValueError, match="mesh_format"):
        _build(plan, expected_output={**plan["expected_output"], "mesh_format": "usdz"})
    mutated = copy.deepcopy(plan)
    mutated["expected_output"]["mesh_format"] = "usdz"
    assert list(plan_validator.iter_errors(mutated))


def test_the_expected_output_is_closed(plan_validator, plan) -> None:
    mutated = copy.deepcopy(plan)
    mutated["expected_output"]["quality"] = "high"
    assert list(plan_validator.iter_errors(mutated))


def test_the_overwrite_policy_is_create_only(plan_validator, plan) -> None:
    assert plan["overwrite_policy"] == "create_only"
    assert list(plan_validator.iter_errors({**plan, "overwrite_policy": "replace"}))


def test_the_plan_is_bound_to_one_clearance_digest(plan) -> None:
    preimage = {k: v for k, v in plan.items() if k != "plan_sha256"}
    baseline = plan["plan_sha256"]
    for key in ("clearance_sha256", "source_image_sha256", "mask_sha256", "engine"):
        mutated = {**preimage, key: "f" * 64 if key.endswith("sha256") else "other-engine"}
        assert canonical_digest(mutated) != baseline, key


def test_the_plan_carries_no_path_or_prompt(plan_validator, plan) -> None:
    for field, value in (
        ("source_path", "/Users/example/photos/subject.png"),
        ("prompt", "a person standing"),
        ("weights_path", "/Users/example/weights/model.safetensors"),
    ):
        assert list(plan_validator.iter_errors({**plan, field: value})), field


def test_the_colour_space_is_pinned(plan_validator, plan) -> None:
    assert plan["color_space"] == "srgb"
    assert list(plan_validator.iter_errors({**plan, "color_space": "linear"}))
