"""Approval issuance, validation, and atomic single-use consumption.

Every gate is a scoped user decision bound to one immutable plan digest. A maintainer, a
provider, or a boolean flag can never stand in for it.
"""

import os
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asset_mania_contracts import (
    ASSET_KINDS,
    DECLARABLE_SUBJECTS,
    GATES,
    SUBJECTS,
    build_approval_receipt,
    canonical_json,
)

from .hashing import sha256_bytes

_SUBJECT_REQUIRED = "SUBJECT_DECLARATION_REQUIRED"
_FACE_RIGHTS_REQUIRED = "FACE_RIGHTS_CONFIRMATION_REQUIRED"

# CLI spellings are kebab case and normalize exactly once, at the parser boundary.
# Portable JSON accepts only the underscore forms these produce.
GATE_SPELLINGS: dict[str, str] = {gate.replace("_", "-"): gate for gate in GATES}
SUBJECT_SPELLINGS: dict[str, str] = {
    subject.replace("_", "-") if subject != "unknown" else "unknown": subject
    for subject in SUBJECTS
}
ASSET_KIND_SPELLINGS: dict[str, str] = {kind.replace("_", "-"): kind for kind in ASSET_KINDS}


class ApprovalRejected(Exception):
    """A receipt does not authorize this plan, gate, scope, or moment."""


class AcknowledgementRejected(Exception):
    """The user did not type the exact plan-bound acknowledgement."""


class ReceiptAlreadyConsumed(Exception):
    """A single-use receipt was already consumed in this store."""


class SubjectDeclarationRequired(Exception):
    """The subject category is undeclared, so nothing downstream may run."""


def _parse(spellings: Mapping[str, str], value: str, field: str) -> str:
    try:
        return spellings[value]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{field} must be one of {sorted(spellings)}; got {value!r}") from error


def parse_gate(value: str) -> str:
    return _parse(GATE_SPELLINGS, value, "gate")


def parse_subject(value: str) -> str:
    return _parse(SUBJECT_SPELLINGS, value, "subject")


def parse_asset_kind(value: str) -> str:
    return _parse(ASSET_KIND_SPELLINGS, value, "asset-kind")


def acknowledgement_text(gate: str, plan_sha256: str) -> str:
    """The exact string the user must type to issue a receipt for one plan."""
    return f"{gate}:{plan_sha256}"


def issue_receipt(
    *,
    receipt_id: str,
    plan_sha256: str,
    gate: str,
    acknowledgement: object,
    disclosure: str,
    issued_at: str,
    expires_at: str,
) -> dict[str, Any]:
    """Issue one single-run receipt from an exact plan-bound user acknowledgement."""
    if gate not in GATES:
        raise ValueError(f"gate {gate!r} is not a declared approval gate")

    expected = acknowledgement_text(gate, plan_sha256)
    if not isinstance(acknowledgement, str) or acknowledgement != expected:
        raise AcknowledgementRejected(
            "acknowledgement must be the exact GATE:PLAN_SHA256 string for this plan"
        )

    return build_approval_receipt(
        receipt_id=receipt_id,
        plan_sha256=plan_sha256,
        gate=gate,
        issued_at=issued_at,
        expires_at=expires_at,
        disclosure_digest=sha256_bytes(disclosure.encode("utf-8")),
        acknowledgement_digest=sha256_bytes(acknowledgement.encode("utf-8")),
    )


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ApprovalRejected(f"receipt timestamp {value!r} is not RFC 3339") from error
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def validate_receipt(
    receipt: Mapping[str, Any],
    *,
    plan_sha256: str,
    gate: str,
    now: datetime,
) -> None:
    """Require a fresh, unedited, user-issued receipt for this exact plan and gate."""
    if receipt.get("issuer_type") != "user":
        raise ApprovalRejected("only the user issues an approval; issuer is not the user")
    if receipt.get("scope") != "single_run":
        raise ApprovalRejected("approval scope must be single_run")
    if receipt.get("gate") != gate:
        raise ApprovalRejected(f"receipt gate {receipt.get('gate')!r} does not match {gate!r}")
    if receipt.get("plan_sha256") != plan_sha256:
        raise ApprovalRejected("receipt is bound to a different plan digest")

    resealed = build_approval_receipt(
        receipt_id=receipt["receipt_id"],
        plan_sha256=receipt["plan_sha256"],
        gate=receipt["gate"],
        issued_at=receipt["issued_at"],
        expires_at=receipt["expires_at"],
        disclosure_digest=receipt["disclosure_digest"],
        acknowledgement_digest=receipt["acknowledgement_digest"],
    )
    if resealed != dict(receipt):
        raise ApprovalRejected("receipt_sha256 does not match the receipt content")

    if now < _timestamp(receipt["issued_at"]):
        raise ApprovalRejected("receipt is not yet issued")
    if now > _timestamp(receipt["expires_at"]):
        raise ApprovalRejected("receipt has expired")


class ConsumptionJournal:
    """An append-only, atomic record of which receipts this store already spent.

    This is accidental-scope protection within one store, not a globally enforceable
    signature; copying a whole run store is outside its guarantee.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def consume(
        self,
        receipt: Mapping[str, Any],
        *,
        consumption_id: str,
        consumed_at: str,
    ) -> dict[str, str]:
        """Record the one permitted consumption of a receipt, or refuse a second one."""
        record = {
            "gate": receipt["gate"],
            "receipt_sha256": receipt["receipt_sha256"],
            "consumption_id": consumption_id,
            "consumed_at": consumed_at,
        }
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.directory / f"{receipt['receipt_sha256']}.json"
        payload = canonical_json(record).encode("utf-8")

        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise ReceiptAlreadyConsumed(
                "this receipt was already consumed in this store"
            ) from error

        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return record


def require_subject_declaration(subject: str) -> str:
    """Block `unknown` before receipt issuance, local execution, or provider planning."""
    if subject not in DECLARABLE_SUBJECTS:
        raise SubjectDeclarationRequired(
            f"{_SUBJECT_REQUIRED}: declare non_person, synthetic_person, or real_person"
        )
    return subject


def authorize_conditioning(
    *,
    subject: str,
    plan_sha256: str,
    receipt: Mapping[str, Any] | None,
    journal: "ConsumptionJournal | None" = None,
    consumption_id: str | None = None,
    consumed_at: str | None = None,
    now: datetime,
) -> dict[str, Any] | None:
    """Decide whether a conditioning run may open the source at all.

    Everything happens before any worker is launched: an undeclared subject fails before
    the receipt is even looked at, a `real_person` subject without an exact plan-bound
    `face_rights` receipt fails, and a valid receipt is consumed atomically here -- so a
    blocked run never reaches Blender and never spends its receipt twice.

    Returns the consumption record for a `real_person` run, or None when no gate applies.
    """
    require_subject_declaration(subject)
    validated = require_rights_receipt(
        subject=subject, receipt=receipt, plan_sha256=plan_sha256, now=now
    )
    if validated is None:
        return None

    if journal is None:
        raise ApprovalRejected("a real_person run needs a consumption journal to spend its receipt")
    if consumption_id is None or consumed_at is None:
        raise ApprovalRejected("a consumption record needs an identifier and a timestamp")

    return journal.consume(validated, consumption_id=consumption_id, consumed_at=consumed_at)


def launch_if_authorized(
    *,
    subject: str,
    plan_sha256: str,
    receipt: Mapping[str, Any] | None,
    journal: "ConsumptionJournal | None" = None,
    consumption_id: str | None = None,
    consumed_at: str | None = None,
    now: datetime,
    launch: "Callable[[], Any]",
) -> tuple[dict[str, Any] | None, Any]:
    """Authorize, then launch. The launcher is unreachable until the gate passes.

    Composing the two here is what makes the ordering testable: a blocked subject or a
    missing receipt raises before `launch` is ever called, so no worker starts and no
    source file is opened on a path that was never authorized.
    """
    record = authorize_conditioning(
        subject=subject,
        plan_sha256=plan_sha256,
        receipt=receipt,
        journal=journal,
        consumption_id=consumption_id,
        consumed_at=consumed_at,
        now=now,
    )
    return record, launch()


def require_rights_receipt(
    *,
    subject: str,
    receipt: Mapping[str, Any] | None,
    plan_sha256: str,
    now: datetime,
) -> dict[str, Any] | None:
    """Gate local face/head processing on a declared subject, never on inference.

    No pixel or geometry classifier runs here: the subject is a user declaration, and
    only `real_person` can satisfy the face-rights gate.
    """
    require_subject_declaration(subject)

    if subject != "real_person":
        if receipt is not None:
            raise ApprovalRejected(f"a face_rights receipt does not apply to subject {subject!r}")
        return None

    if receipt is None:
        raise ApprovalRejected(
            f"{_FACE_RIGHTS_REQUIRED}: real_person requires a plan-bound rights receipt"
        )
    validate_receipt(receipt, plan_sha256=plan_sha256, gate="face_rights", now=now)
    return dict(receipt)
