"""Closed-schema and builder tests for `provider-plan-v1`."""

import copy
from decimal import Decimal

import pytest
from asset_mania_contracts import (
    GATES,
    build_provider_plan,
    canonical_digest,
    estimate_provider_cost,
    required_gates_for,
)
from conftest import load_example


@pytest.fixture
def plan_validator(validator_for):
    return validator_for("provider-plan", "1.0")


@pytest.fixture
def plan():
    return load_example("provider-plan-v1")


def test_example_is_valid_and_self_sealed(plan_validator, plan) -> None:
    assert list(plan_validator.iter_errors(plan)) == []
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert canonical_digest(preimage) == plan["plan_sha256"]


def test_real_person_example_is_valid(plan_validator) -> None:
    real_person = load_example("provider-plan-v1-real-person")
    assert list(plan_validator.iter_errors(real_person)) == []
    assert real_person["required_gates"] == ["face_rights", "external_egress", "paid_compute"]


def test_builder_reproduces_both_normative_examples() -> None:
    for name in ("provider-plan-v1", "provider-plan-v1-real-person"):
        expected = load_example(name)
        built = build_provider_plan(
            condition_manifest_sha256=expected["condition_manifest_sha256"],
            attachments=expected["attachments"],
            prompt_sha256=expected["prompt_sha256"],
            controls=expected["controls"],
            subject=expected["subject"],
            policy_evidence=expected["policy_evidence"],
            cost_estimate=expected["cost_estimate"],
            expected_view=expected["expected_view"],
        )
        assert built == expected, name


def test_required_gates_are_derived_from_the_declared_subject(plan_validator, plan) -> None:
    assert required_gates_for("non_person") == ["external_egress", "paid_compute"]
    assert required_gates_for("synthetic_person") == ["external_egress", "paid_compute"]
    assert required_gates_for("real_person") == [
        "face_rights",
        "external_egress",
        "paid_compute",
    ]
    with pytest.raises(ValueError, match="SUBJECT_DECLARATION_REQUIRED"):
        required_gates_for("unknown")

    mismatched = {**plan, "required_gates": list(GATES)}
    assert list(plan_validator.iter_errors(mismatched))


def test_unknown_subject_is_never_provider_planable(plan_validator, plan) -> None:
    assert list(plan_validator.iter_errors({**plan, "subject": "unknown"}))
    with pytest.raises(ValueError, match="SUBJECT_DECLARATION_REQUIRED"):
        build_provider_plan(
            condition_manifest_sha256=plan["condition_manifest_sha256"],
            attachments=plan["attachments"],
            prompt_sha256=plan["prompt_sha256"],
            controls=plan["controls"],
            subject="unknown",
            policy_evidence=plan["policy_evidence"],
            cost_estimate=plan["cost_estimate"],
            expected_view=plan["expected_view"],
        )


def test_attachments_keep_their_fixed_role_order_and_multipart_field(plan_validator, plan) -> None:
    assert [item["role"] for item in plan["attachments"]] == [
        "beauty",
        "depth_preview",
        "normal_preview",
        "mask",
    ]
    assert [item["index"] for item in plan["attachments"]] == [0, 1, 2, 3]
    assert {item["multipart_field"] for item in plan["attachments"]} == {"image[]"}

    reordered = copy.deepcopy(plan)
    reordered["attachments"][0], reordered["attachments"][1] = (
        reordered["attachments"][1],
        reordered["attachments"][0],
    )
    assert list(plan_validator.iter_errors(reordered))

    extra = copy.deepcopy(plan)
    extra["attachments"].append(copy.deepcopy(plan["attachments"][0]))
    assert list(plan_validator.iter_errors(extra))


def test_plan_carries_a_prompt_digest_and_never_the_prompt_text(plan_validator, plan) -> None:
    assert "prompt" not in plan
    for key, value in (
        ("prompt", "a studio photo of a person"),
        ("prompt_text", "a studio photo of a person"),
        ("api_key", "sk-live-000"),
        ("signed_url", "https://example.com/upload?signature=abc"),
        ("image_bytes", "iVBORw0KGgo="),
    ):
        assert list(plan_validator.iter_errors({**plan, key: value})), key


@pytest.mark.parametrize("size", ["512x512", "2048x2048", "1024x1025", "1024X1024"])
def test_only_official_cost_table_sizes_are_executable(plan_validator, plan, size: str) -> None:
    mutated = copy.deepcopy(plan)
    mutated["controls"]["size"] = size
    assert list(plan_validator.iter_errors(mutated))


def test_png_forbids_compression_and_jpeg_requires_it(plan_validator, plan) -> None:
    mutated = copy.deepcopy(plan)
    mutated["controls"]["output_compression"] = 80
    assert list(plan_validator.iter_errors(mutated))

    mutated = copy.deepcopy(plan)
    mutated["controls"]["output_format"] = "jpeg"
    assert list(plan_validator.iter_errors(mutated))


def test_cost_estimate_follows_the_published_formula(plan) -> None:
    estimate = plan["cost_estimate"]
    evidence = load_example("provider-evidence-v1")
    computed = estimate_provider_cost(
        pricing=evidence["pricing"],
        quality=estimate["quality"],
        size=estimate["size"],
        text_input_tokens=estimate["text_input_tokens_assumed"],
        image_input_tokens=estimate["image_input_tokens_assumed"],
    )
    assert computed == estimate["estimated_cost"]

    rates = evidence["pricing"]["per_million_tokens"]
    row = next(
        item
        for item in evidence["pricing"]["output_cost_rows"]
        if (item["quality"], item["size"]) == (estimate["quality"], estimate["size"])
    )
    expected = (
        Decimal(estimate["text_input_tokens_assumed"]) * Decimal(rates["text_input"])
        + Decimal(estimate["image_input_tokens_assumed"]) * Decimal(rates["image_input"])
    ) / Decimal(1_000_000) + Decimal(row["usd"])
    assert Decimal(computed) >= expected


def test_cached_assumptions_are_pinned_to_zero(plan_validator, plan) -> None:
    assert plan["cost_estimate"]["cached_text_input_tokens_assumed"] == 0
    assert plan["cost_estimate"]["cached_image_input_tokens_assumed"] == 0
    mutated = copy.deepcopy(plan)
    mutated["cost_estimate"]["cached_text_input_tokens_assumed"] = 10
    assert list(plan_validator.iter_errors(mutated))


def test_maximum_cost_is_never_below_the_estimate(plan) -> None:
    estimate = plan["cost_estimate"]
    assert Decimal(estimate["maximum_cost"]) >= Decimal(estimate["estimated_cost"])
    with pytest.raises(ValueError, match="maximum_cost"):
        build_provider_plan(
            condition_manifest_sha256=plan["condition_manifest_sha256"],
            attachments=plan["attachments"],
            prompt_sha256=plan["prompt_sha256"],
            controls=plan["controls"],
            subject=plan["subject"],
            policy_evidence=plan["policy_evidence"],
            cost_estimate={**estimate, "maximum_cost": "0.000001"},
            expected_view=plan["expected_view"],
        )


def test_expected_view_must_match_the_controls(plan) -> None:
    with pytest.raises(ValueError, match="expected_view"):
        build_provider_plan(
            condition_manifest_sha256=plan["condition_manifest_sha256"],
            attachments=plan["attachments"],
            prompt_sha256=plan["prompt_sha256"],
            controls=plan["controls"],
            subject=plan["subject"],
            policy_evidence=plan["policy_evidence"],
            cost_estimate=plan["cost_estimate"],
            expected_view={**plan["expected_view"], "width": 1536},
        )


def test_cost_estimate_must_match_the_controls(plan) -> None:
    with pytest.raises(ValueError, match="cost_estimate"):
        build_provider_plan(
            condition_manifest_sha256=plan["condition_manifest_sha256"],
            attachments=plan["attachments"],
            prompt_sha256=plan["prompt_sha256"],
            controls=plan["controls"],
            subject=plan["subject"],
            policy_evidence=plan["policy_evidence"],
            cost_estimate={**plan["cost_estimate"], "quality": "high"},
            expected_view=plan["expected_view"],
        )
