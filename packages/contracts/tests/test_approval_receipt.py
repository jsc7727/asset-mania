"""Closed-schema and builder tests for `approval-receipt-v1`."""

import pytest
from asset_mania_contracts import GATES, build_approval_receipt, canonical_digest
from conftest import example_names, load_example


@pytest.fixture
def receipt_validator(validator_for):
    return validator_for("approval-receipt", "1.0")


def test_every_gate_has_one_normative_example() -> None:
    assert example_names("approval-receipt-v1-") == [
        f"approval-receipt-v1-{gate.replace('_', '-')}" for gate in sorted(GATES)
    ]


@pytest.mark.parametrize("gate", sorted(GATES))
def test_gate_example_is_valid_and_self_sealed(receipt_validator, gate: str) -> None:
    receipt = load_example(f"approval-receipt-v1-{gate.replace('_', '-')}")
    assert list(receipt_validator.iter_errors(receipt)) == []
    assert receipt["gate"] == gate
    preimage = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    assert canonical_digest(preimage) == receipt["receipt_sha256"]


@pytest.mark.parametrize("gate", sorted(GATES))
def test_builder_reproduces_every_gate_example(gate: str) -> None:
    receipt = load_example(f"approval-receipt-v1-{gate.replace('_', '-')}")
    built = build_approval_receipt(
        receipt_id=receipt["receipt_id"],
        plan_sha256=receipt["plan_sha256"],
        gate=gate,
        issued_at=receipt["issued_at"],
        expires_at=receipt["expires_at"],
        disclosure_digest=receipt["disclosure_digest"],
        acknowledgement_digest=receipt["acknowledgement_digest"],
    )
    assert built == receipt


def test_scope_is_only_single_run(receipt_validator) -> None:
    receipt = load_example("approval-receipt-v1-external-egress")
    for scope in ("all_runs", "session", "store"):
        assert list(receipt_validator.iter_errors({**receipt, "scope": scope})), scope
    with pytest.raises(ValueError, match="expires_at"):
        build_approval_receipt(
            receipt_id=receipt["receipt_id"],
            plan_sha256=receipt["plan_sha256"],
            gate=receipt["gate"],
            issued_at=receipt["issued_at"],
            expires_at=receipt["issued_at"],
            disclosure_digest=receipt["disclosure_digest"],
            acknowledgement_digest=receipt["acknowledgement_digest"],
        )


def test_issuer_is_always_the_user(receipt_validator) -> None:
    receipt = load_example("approval-receipt-v1-face-rights")
    for issuer in ("maintainer", "provider", "system"):
        assert list(receipt_validator.iter_errors({**receipt, "issuer_type": issuer})), issuer


def test_receipt_carries_no_person_identifier_or_free_form_note(receipt_validator) -> None:
    receipt = load_example("approval-receipt-v1-face-rights")
    for key, value in (
        ("note", "approved by the model on set"),
        ("person_name", "Jane Doe"),
        ("subject_email", "jane@example.com"),
    ):
        assert list(receipt_validator.iter_errors({**receipt, key: value})), key


def test_receipt_binds_one_gate_to_one_plan_digest() -> None:
    receipt = load_example("approval-receipt-v1-paid-compute")
    preimage = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    for key, value in (
        ("gate", "external_egress"),
        ("plan_sha256", "f" * 64),
        ("expires_at", "2026-08-19T23:35:00Z"),
        ("acknowledgement_digest", "e" * 64),
        ("disclosure_digest", "d" * 64),
    ):
        assert canonical_digest({**preimage, key: value}) != receipt["receipt_sha256"], key


def test_builder_rejects_a_gate_outside_the_closed_set() -> None:
    receipt = load_example("approval-receipt-v1-paid-compute")
    with pytest.raises(ValueError, match="gate"):
        build_approval_receipt(
            receipt_id=receipt["receipt_id"],
            plan_sha256=receipt["plan_sha256"],
            gate="local_face_processing",
            issued_at=receipt["issued_at"],
            expires_at=receipt["expires_at"],
            disclosure_digest=receipt["disclosure_digest"],
            acknowledgement_digest=receipt["acknowledgement_digest"],
        )
