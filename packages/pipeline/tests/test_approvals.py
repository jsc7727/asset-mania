"""Approval issuance, validation, and atomic single-use consumption."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from asset_mania_pipeline import (
    AcknowledgementRejected,
    ApprovalRejected,
    ConsumptionJournal,
    ReceiptAlreadyConsumed,
    SubjectDeclarationRequired,
    acknowledgement_text,
    issue_receipt,
    parse_asset_kind,
    parse_gate,
    parse_subject,
    require_rights_receipt,
    require_subject_declaration,
    validate_receipt,
)

PLAN = "b7" * 32
OTHER_PLAN = "c9" * 32
DISCLOSURE = "This run uploads four reference images to OpenAI and spends up to USD 0.100000."
ISSUED_AT = "2026-08-19T09:35:00Z"
EXPIRES_AT = "2026-08-19T10:35:00Z"
NOW = datetime(2026, 8, 19, 9, 40, tzinfo=UTC)


def _receipt(gate: str = "external_egress", plan: str = PLAN) -> dict:
    return issue_receipt(
        receipt_id=f"receipt-{gate.replace('_', '-')}-1",
        plan_sha256=plan,
        gate=gate,
        acknowledgement=acknowledgement_text(gate, plan),
        disclosure=DISCLOSURE,
        issued_at=ISSUED_AT,
        expires_at=EXPIRES_AT,
    )


# --- CLI to JSON normalization ------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("face-rights", "face_rights"),
        ("external-egress", "external_egress"),
        ("paid-compute", "paid_compute"),
    ],
)
def test_gate_spellings_normalize_at_the_parser_boundary(spelling: str, expected: str) -> None:
    assert parse_gate(spelling) == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("non-person", "non_person"),
        ("synthetic-person", "synthetic_person"),
        ("real-person", "real_person"),
        ("unknown", "unknown"),
    ],
)
def test_subject_spellings_normalize_at_the_parser_boundary(spelling: str, expected: str) -> None:
    assert parse_subject(spelling) == expected


def test_asset_kind_spellings_normalize_at_the_parser_boundary() -> None:
    assert parse_asset_kind("face-head") == "face_head"
    assert parse_asset_kind("object") == "object"
    assert parse_asset_kind("character") == "character"


@pytest.mark.parametrize("parser", [parse_gate, parse_subject, parse_asset_kind])
def test_parsers_reject_the_underscore_spelling_on_the_command_line(parser) -> None:
    with pytest.raises(ValueError):
        parser("face_head" if parser is parse_asset_kind else "face_rights")


def test_portable_json_accepts_only_underscore_forms() -> None:
    receipt = _receipt("face_rights")
    assert receipt["gate"] == "face_rights"
    with pytest.raises(ValueError, match="gate"):
        issue_receipt(
            receipt_id="receipt-1",
            plan_sha256=PLAN,
            gate="face-rights",
            acknowledgement=acknowledgement_text("face_rights", PLAN),
            disclosure=DISCLOSURE,
            issued_at=ISSUED_AT,
            expires_at=EXPIRES_AT,
        )


# --- Issuance -----------------------------------------------------------------


def test_issuance_requires_the_full_plan_bound_acknowledgement() -> None:
    assert acknowledgement_text("paid_compute", PLAN) == f"paid_compute:{PLAN}"

    for wrong in (
        "yes",
        "y",
        "true",
        "paid_compute",
        PLAN,
        f"paid_compute:{PLAN[:16]}",
        f"external_egress:{PLAN}",
        f"paid_compute:{OTHER_PLAN}",
        f"PAID_COMPUTE:{PLAN}",
        f" paid_compute:{PLAN}",
    ):
        with pytest.raises(AcknowledgementRejected, match="acknowledgement"):
            issue_receipt(
                receipt_id="receipt-1",
                plan_sha256=PLAN,
                gate="paid_compute",
                acknowledgement=wrong,
                disclosure=DISCLOSURE,
                issued_at=ISSUED_AT,
                expires_at=EXPIRES_AT,
            )


def test_a_boolean_flag_can_never_stand_in_for_an_acknowledgement() -> None:
    for boolean in (True, False, 1, 0, None):
        with pytest.raises((AcknowledgementRejected, TypeError)):
            issue_receipt(
                receipt_id="receipt-1",
                plan_sha256=PLAN,
                gate="paid_compute",
                acknowledgement=boolean,
                disclosure=DISCLOSURE,
                issued_at=ISSUED_AT,
                expires_at=EXPIRES_AT,
            )


def test_the_receipt_records_the_disclosure_digest_not_its_text() -> None:
    from asset_mania_pipeline import sha256_bytes

    receipt = _receipt()
    assert receipt["disclosure_digest"] == sha256_bytes(DISCLOSURE.encode("utf-8"))
    rendered = json.dumps(receipt)
    assert DISCLOSURE not in rendered
    assert "note" not in receipt


def test_the_receipt_carries_no_person_identifier_or_free_text() -> None:
    receipt = _receipt("face_rights")
    assert set(receipt) == {
        "schema_id",
        "schema_version",
        "receipt_id",
        "plan_sha256",
        "gate",
        "issued_at",
        "expires_at",
        "scope",
        "disclosure_digest",
        "issuer_type",
        "acknowledgement_digest",
        "receipt_sha256",
    }
    assert receipt["issuer_type"] == "user"
    assert receipt["scope"] == "single_run"


def test_only_the_user_issues_a_receipt_for_every_gate() -> None:
    for gate in ("face_rights", "external_egress", "paid_compute"):
        receipt = _receipt(gate)
        assert receipt["issuer_type"] == "user"
        for forged in ("maintainer", "provider", "system", "organization"):
            with pytest.raises(ApprovalRejected, match="issuer"):
                validate_receipt(
                    {**receipt, "issuer_type": forged},
                    plan_sha256=PLAN,
                    gate=gate,
                    now=NOW,
                )


# --- Validation ---------------------------------------------------------------


def test_a_fresh_matching_receipt_validates() -> None:
    validate_receipt(_receipt(), plan_sha256=PLAN, gate="external_egress", now=NOW)


def test_a_receipt_for_another_plan_is_rejected() -> None:
    with pytest.raises(ApprovalRejected, match="plan"):
        validate_receipt(
            _receipt(plan=OTHER_PLAN), plan_sha256=PLAN, gate="external_egress", now=NOW
        )


def test_a_receipt_for_another_gate_is_rejected() -> None:
    with pytest.raises(ApprovalRejected, match="gate"):
        validate_receipt(
            _receipt("paid_compute"), plan_sha256=PLAN, gate="external_egress", now=NOW
        )


def test_an_expired_receipt_is_rejected() -> None:
    late = datetime.fromisoformat(EXPIRES_AT) + timedelta(seconds=1)
    with pytest.raises(ApprovalRejected, match="expired"):
        validate_receipt(_receipt(), plan_sha256=PLAN, gate="external_egress", now=late)


def test_a_receipt_used_before_issuance_is_rejected() -> None:
    early = datetime.fromisoformat(ISSUED_AT) - timedelta(seconds=1)
    with pytest.raises(ApprovalRejected, match="issued"):
        validate_receipt(_receipt(), plan_sha256=PLAN, gate="external_egress", now=early)


def test_a_global_or_multi_run_scope_is_rejected() -> None:
    for scope in ("all_runs", "session", "global", "store"):
        with pytest.raises(ApprovalRejected, match="scope"):
            validate_receipt(
                {**_receipt(), "scope": scope},
                plan_sha256=PLAN,
                gate="external_egress",
                now=NOW,
            )


def test_a_receipt_whose_digest_was_edited_is_rejected() -> None:
    receipt = _receipt()
    with pytest.raises(ApprovalRejected, match="receipt_sha256"):
        validate_receipt(
            {**receipt, "expires_at": "2026-08-20T10:35:00Z"},
            plan_sha256=PLAN,
            gate="external_egress",
            now=NOW,
        )


# --- Consumption --------------------------------------------------------------


def test_a_receipt_is_consumable_exactly_once(tmp_path: Path) -> None:
    journal = ConsumptionJournal(tmp_path / "approvals")
    receipt = _receipt()

    record = journal.consume(
        receipt,
        consumption_id="consumption-external-egress-1",
        consumed_at="2026-08-19T09:40:00Z",
    )
    assert record == {
        "gate": "external_egress",
        "receipt_sha256": receipt["receipt_sha256"],
        "consumption_id": "consumption-external-egress-1",
        "consumed_at": "2026-08-19T09:40:00Z",
    }

    with pytest.raises(ReceiptAlreadyConsumed, match="consumed"):
        journal.consume(
            receipt,
            consumption_id="consumption-external-egress-2",
            consumed_at="2026-08-19T09:41:00Z",
        )


def test_a_copy_of_a_consumed_receipt_in_the_same_store_is_rejected(tmp_path: Path) -> None:
    journal = ConsumptionJournal(tmp_path / "approvals")
    receipt = _receipt()
    journal.consume(receipt, consumption_id="consumption-1", consumed_at="2026-08-19T09:40:00Z")

    copied = {**receipt, "receipt_id": "receipt-external-egress-copy"}
    with pytest.raises(ReceiptAlreadyConsumed, match="consumed"):
        journal.consume(copied, consumption_id="consumption-2", consumed_at="2026-08-19T09:42:00Z")


def test_consumption_is_recorded_before_it_returns(tmp_path: Path) -> None:
    journal = ConsumptionJournal(tmp_path / "approvals")
    receipt = _receipt()
    journal.consume(receipt, consumption_id="consumption-1", consumed_at="2026-08-19T09:40:00Z")

    reopened = ConsumptionJournal(tmp_path / "approvals")
    with pytest.raises(ReceiptAlreadyConsumed):
        reopened.consume(
            receipt, consumption_id="consumption-2", consumed_at="2026-08-19T09:41:00Z"
        )


def test_two_gates_consume_independently(tmp_path: Path) -> None:
    journal = ConsumptionJournal(tmp_path / "approvals")
    journal.consume(
        _receipt("external_egress"),
        consumption_id="consumption-1",
        consumed_at="2026-08-19T09:40:00Z",
    )
    journal.consume(
        _receipt("paid_compute"),
        consumption_id="consumption-2",
        consumed_at="2026-08-19T09:40:01Z",
    )


def test_the_journal_stores_only_digests_and_portable_identifiers(tmp_path: Path) -> None:
    journal = ConsumptionJournal(tmp_path / "approvals")
    receipt = _receipt()
    journal.consume(receipt, consumption_id="consumption-1", consumed_at="2026-08-19T09:40:00Z")

    rendered = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((tmp_path / "approvals").rglob("*"))
        if path.is_file()
    )
    assert receipt["receipt_sha256"] in rendered
    assert DISCLOSURE not in rendered
    assert str(tmp_path) not in rendered


# --- Subject gating -----------------------------------------------------------


def test_unknown_is_blocked_before_any_execution_or_receipt() -> None:
    with pytest.raises(SubjectDeclarationRequired, match="SUBJECT_DECLARATION_REQUIRED"):
        require_subject_declaration("unknown")
    with pytest.raises(SubjectDeclarationRequired, match="SUBJECT_DECLARATION_REQUIRED"):
        require_rights_receipt(subject="unknown", receipt=None, plan_sha256=PLAN, now=NOW)


@pytest.mark.parametrize("subject", ["non_person", "synthetic_person", "real_person"])
def test_a_declared_subject_passes_the_declaration_check(subject: str) -> None:
    require_subject_declaration(subject)


@pytest.mark.parametrize("subject", ["non_person", "synthetic_person"])
def test_a_non_person_path_needs_no_receipt_and_runs_no_classifier(subject: str) -> None:
    assert require_rights_receipt(subject=subject, receipt=None, plan_sha256=PLAN, now=NOW) is None


@pytest.mark.parametrize("subject", ["non_person", "synthetic_person"])
def test_a_non_person_path_refuses_a_needless_face_rights_receipt(subject: str) -> None:
    with pytest.raises(ApprovalRejected, match="face_rights"):
        require_rights_receipt(
            subject=subject,
            receipt=_receipt("face_rights"),
            plan_sha256=PLAN,
            now=NOW,
        )


def test_real_person_requires_a_plan_bound_face_rights_receipt() -> None:
    with pytest.raises(ApprovalRejected, match="FACE_RIGHTS_CONFIRMATION_REQUIRED"):
        require_rights_receipt(subject="real_person", receipt=None, plan_sha256=PLAN, now=NOW)

    with pytest.raises(ApprovalRejected, match="gate"):
        require_rights_receipt(
            subject="real_person",
            receipt=_receipt("external_egress"),
            plan_sha256=PLAN,
            now=NOW,
        )

    with pytest.raises(ApprovalRejected, match="plan"):
        require_rights_receipt(
            subject="real_person",
            receipt=_receipt("face_rights", plan=OTHER_PLAN),
            plan_sha256=PLAN,
            now=NOW,
        )

    receipt = _receipt("face_rights")
    assert (
        require_rights_receipt(subject="real_person", receipt=receipt, plan_sha256=PLAN, now=NOW)
        == receipt
    )
