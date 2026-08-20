"""Transport is unreachable until every gate has passed."""

import base64
from pathlib import Path

import pytest
from asset_mania_pipeline import ReceiptAlreadyConsumed, SubjectDeclarationRequired
from asset_mania_provider_openai import client
from asset_mania_provider_openai.errors import (
    ApprovalMissing,
    CredentialUnavailable,
    EvidenceStale,
    PlanMismatch,
    RequestCanceled,
)
from asset_mania_provider_openai.transport import (
    DeniedTransport,
    ProviderResponse,
    RecordingTransport,
)
from conftest import CONSUMED_AT, NOW

PNG = b"\x89PNG\r\n\x1a\n" + b"generated" + bytes(32)
#: Shaped like a credential without matching a real provider key pattern, so the release
#: secret scanner does not flag the test itself.
CREDENTIAL_PLACEHOLDER = "PROVIDER-CREDENTIAL-PLACEHOLDER-0000"


def _ok_response() -> ProviderResponse:
    return ProviderResponse(
        status=200,
        body={
            "data": [{"b64_json": base64.b64encode(PNG).decode("ascii")}],
            "usage": {"input_tokens": 3320, "output_tokens": 1024},
            "actual_cost": "0.081000",
        },
        request_id="req_test_1",
    )


def _generate(plan, evidence, receipts, prompt_file, attachments, journal, credential, **kw):
    arguments = {
        "plan": plan,
        "evidence": evidence,
        "receipts": receipts,
        "prompt_path": prompt_file,
        "run_directory": attachments["run_directory"],
        "journal": journal,
        "now": NOW,
        "consumed_at": CONSUMED_AT,
        "secret_resolver": credential,
    }
    arguments.update(kw)
    return client.generate(**arguments)


# --- The default is no network -----------------------------------------------------


def test_the_default_transport_refuses_every_call() -> None:
    with pytest.raises(AssertionError, match="transport is denied"):
        DeniedTransport().send(object(), credential="x")  # type: ignore[arg-type]


def test_a_full_run_without_a_transport_never_reaches_the_network(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    built = plan()
    with pytest.raises(AssertionError, match="transport is denied"):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
        )


# --- Gate ordering ------------------------------------------------------------------


def test_an_undeclared_subject_fails_before_transport(
    plan, evidence, prompt_file, attachments, journal, credential
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = {**plan(), "subject": "unknown"}
    with pytest.raises(SubjectDeclarationRequired):
        _generate(
            built, evidence, [], prompt_file, attachments, journal, credential, transport=transport
        )
    assert transport.sent == []


def test_stale_evidence_fails_before_credential_access(
    plan, evidence, receipts, prompt_file, attachments, journal
) -> None:
    transport = RecordingTransport([_ok_response()])
    resolved = []

    def resolver() -> str:
        resolved.append("credential")
        return "SECRET"

    stale = {
        **evidence,
        "retrieved_at": "2026-08-17T08:00:00Z",
        "expires_at": "2026-08-18T08:00:00Z",
    }
    built = plan()
    with pytest.raises(EvidenceStale, match="PROVIDER_EVIDENCE_STALE"):
        _generate(
            built,
            stale,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            resolver,
            transport=transport,
        )
    assert resolved == []
    assert transport.sent == []


def test_evidence_without_a_twenty_four_hour_ttl_is_refused(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    wrong = {**evidence, "expires_at": "2026-08-21T08:00:00Z"}
    built = plan()
    with pytest.raises(EvidenceStale, match="24-hour TTL"):
        _generate(
            built,
            wrong,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=RecordingTransport([_ok_response()]),
        )


def test_a_changed_evidence_digest_invalidates_the_approval(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    built = plan()
    other = {**evidence, "evidence_sha256": "f" * 64}
    with pytest.raises(EvidenceStale, match="different evidence"):
        _generate(
            built,
            other,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=RecordingTransport([_ok_response()]),
        )


def test_a_prompt_that_is_not_the_approved_one_fails_before_transport(
    plan, evidence, receipts, attachments, journal, credential, tmp_path: Path
) -> None:
    transport = RecordingTransport([_ok_response()])
    other = tmp_path / "other-prompt.txt"
    other.write_text("a different prompt entirely\n", encoding="utf-8")
    built = plan()
    with pytest.raises(PlanMismatch, match="approved plan digest"):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], built["required_gates"]),
            other,
            attachments,
            journal,
            credential,
            transport=transport,
        )
    assert transport.sent == []


def test_missing_receipts_fail_before_transport(
    plan, evidence, prompt_file, attachments, journal, credential
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    with pytest.raises(ApprovalMissing, match="missing approvals"):
        _generate(
            built, evidence, [], prompt_file, attachments, journal, credential, transport=transport
        )
    assert transport.sent == []


def test_a_receipt_for_another_plan_fails_before_transport(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    from asset_mania_pipeline import ApprovalRejected

    transport = RecordingTransport([_ok_response()])
    built = plan()
    with pytest.raises(ApprovalRejected, match="different plan"):
        _generate(
            built,
            evidence,
            receipts("f" * 64, built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=transport,
        )
    assert transport.sent == []


def test_a_real_person_plan_needs_the_face_rights_gate(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    built = plan(subject="real_person")
    assert built["required_gates"] == ["face_rights", "external_egress", "paid_compute"]

    transport = RecordingTransport([_ok_response()])
    with pytest.raises(ApprovalMissing, match="face_rights"):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], ["external_egress", "paid_compute"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=transport,
        )
    assert transport.sent == []


def test_a_receipt_for_a_gate_this_plan_does_not_need_is_refused(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    built = plan(subject="non_person")
    with pytest.raises(ApprovalMissing, match="does not need"):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], ["face_rights", "external_egress", "paid_compute"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=RecordingTransport([_ok_response()]),
        )


def test_an_edited_plan_cannot_ride_an_old_approval(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    built = plan()
    tampered = {**built, "model": "gpt-image-2-2026-05-01"}
    with pytest.raises(PlanMismatch, match="plan digest"):
        _generate(
            tampered,
            evidence,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=RecordingTransport([_ok_response()]),
        )


def test_a_missing_credential_fails_after_approval_but_before_transport(
    plan, evidence, receipts, prompt_file, attachments, journal
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    with pytest.raises(CredentialUnavailable):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            lambda: "",
            transport=transport,
        )
    assert transport.sent == []


def test_cancellation_before_transport_spends_no_approval(
    plan, evidence, receipts, prompt_file, attachments, journal, credential, tmp_path: Path
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    with pytest.raises(RequestCanceled):
        _generate(
            built,
            evidence,
            receipts(built["plan_sha256"], built["required_gates"]),
            prompt_file,
            attachments,
            journal,
            credential,
            transport=transport,
            canceled=True,
        )
    assert transport.sent == []
    assert not (tmp_path / "approvals").exists()


# --- The permitted call ---------------------------------------------------------------


def test_an_approved_run_reaches_transport_exactly_once(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    result = _generate(
        built,
        evidence,
        receipts(built["plan_sha256"], built["required_gates"]),
        prompt_file,
        attachments,
        journal,
        credential,
        transport=transport,
    )
    assert len(transport.sent) == 1
    assert result["image_sha256"]
    assert result["request_id"] == "req_test_1"
    assert [record["gate"] for record in result["approvals"]] == built["required_gates"]


def test_a_paid_request_is_never_retried_automatically(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    """A retry needs a new approval, so the receipts cannot be spent twice."""
    transport = RecordingTransport([_ok_response(), _ok_response()])
    built = plan()
    supplied = receipts(built["plan_sha256"], built["required_gates"])
    _generate(
        built,
        evidence,
        supplied,
        prompt_file,
        attachments,
        journal,
        credential,
        transport=transport,
    )
    with pytest.raises(ReceiptAlreadyConsumed):
        _generate(
            built,
            evidence,
            supplied,
            prompt_file,
            attachments,
            journal,
            credential,
            transport=transport,
        )
    assert len(transport.sent) == 1


def test_the_credential_never_appears_in_the_request_record(
    plan, evidence, receipts, prompt_file, attachments, journal
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    result = _generate(
        built,
        evidence,
        receipts(built["plan_sha256"], built["required_gates"]),
        prompt_file,
        attachments,
        journal,
        lambda: CREDENTIAL_PLACEHOLDER,
        transport=transport,
    )
    rendered = repr(result["request_record"]) + repr(transport.sent)
    assert CREDENTIAL_PLACEHOLDER not in rendered


def test_the_prompt_text_never_appears_in_the_request_record(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    from conftest import PROMPT_TEXT

    transport = RecordingTransport([_ok_response()])
    built = plan()
    result = _generate(
        built,
        evidence,
        receipts(built["plan_sha256"], built["required_gates"]),
        prompt_file,
        attachments,
        journal,
        credential,
        transport=transport,
    )
    assert PROMPT_TEXT.strip() not in repr(result["request_record"])
    assert PROMPT_TEXT.strip() not in repr(transport.sent)


def test_the_image_bytes_never_appear_in_the_request_record(
    plan, evidence, receipts, prompt_file, attachments, journal, credential
) -> None:
    transport = RecordingTransport([_ok_response()])
    built = plan()
    _generate(
        built,
        evidence,
        receipts(built["plan_sha256"], built["required_gates"]),
        prompt_file,
        attachments,
        journal,
        credential,
        transport=transport,
    )
    record = transport.sent[0]
    for part in record["parts"]:
        assert set(part) == {"field_name", "media_type", "byte_size"}
