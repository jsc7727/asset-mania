"""The approval-gated adapter.

Ordering is the contract. Nothing touches a credential or a transport until every gate has
passed, in this order:

1. the subject is a declaration, and `unknown` fails first;
2. the policy and pricing evidence is inside its executable TTL;
3. the private prompt matches the approved plan digest;
4. every required receipt is valid for this exact plan digest and is consumed atomically.

Only then is the credential resolved and the request sent. A paid request is never retried
automatically -- a retry is a new approval -- and the returned usage and actual cost are
recorded separately from the preflight estimate so the two can never be confused.
"""

import base64
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from asset_mania_contracts import canonical_digest, required_gates_for
from asset_mania_pipeline import (
    ConsumptionJournal,
    SubjectDeclarationRequired,
    require_subject_declaration,
    sha256_bytes,
    validate_receipt,
)

from .errors import (
    ApprovalMissing,
    CredentialUnavailable,
    EvidenceStale,
    ModerationRejected,
    PlanMismatch,
    ProviderTimeout,
    RequestCanceled,
    ResponseInvalid,
    classify_status,
)
from .normalization import (
    MODEL_SNAPSHOT,
    build_request,
    load_attachments,
    read_prompt,
    validate_controls,
)
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    DeniedTransport,
    ProviderResponse,
    SecretResolver,
    Transport,
    refusing_secret_resolver,
)

PROVIDER = "openai"
#: Evidence older than this is not executable. There is no implicit refresh.
EVIDENCE_TTL = timedelta(hours=24)
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_OUTPUT_MEDIA_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def verify_evidence_freshness(evidence: Mapping[str, Any], *, now: datetime) -> None:
    """Fail closed before credential access when the evidence is stale.

    The TTL is not refreshed implicitly. A maintainer must run the explicit, networked
    evidence-refresh command to write a new hashed artifact.
    """
    retrieved = _timestamp(evidence["retrieved_at"])
    expires = _timestamp(evidence["expires_at"])
    if expires - retrieved != EVIDENCE_TTL:
        raise EvidenceStale("PROVIDER_EVIDENCE_STALE: the evidence does not declare a 24-hour TTL")
    if now > expires:
        raise EvidenceStale("PROVIDER_EVIDENCE_STALE: the evidence has expired")
    if now < retrieved:
        raise EvidenceStale("PROVIDER_EVIDENCE_STALE: the evidence is not yet retrieved")


def verify_evidence_binding(plan: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """A changed evidence digest invalidates the approval rather than being tolerated."""
    if plan["cost_estimate"]["rate_digest"] != evidence["evidence_sha256"]:
        raise EvidenceStale(
            "PROVIDER_EVIDENCE_STALE: the plan was approved against different evidence"
        )
    if evidence["model_snapshot"] != MODEL_SNAPSHOT:
        raise PlanMismatch("the evidence describes another model snapshot")


def verify_plan_seal(plan: Mapping[str, Any]) -> str:
    """Recompute the plan digest so an edited plan cannot ride an old approval."""
    preimage = {
        key: value for key, value in plan.items() if key not in ("plan_sha256", "attachment_paths")
    }
    digest = canonical_digest(preimage)
    if digest != plan["plan_sha256"]:
        raise PlanMismatch("the provider plan digest does not match its content")
    return digest


def consume_receipts(
    *,
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    journal: ConsumptionJournal,
    now: datetime,
    consumed_at: str,
) -> list[dict[str, Any]]:
    """Validate and atomically consume every gate the declared subject requires."""
    subject = plan["subject"]
    require_subject_declaration(subject)
    required = required_gates_for(subject)

    supplied = {}
    for receipt in receipts:
        gate = receipt.get("gate")
        if gate in supplied:
            raise ApprovalMissing(f"two receipts were supplied for the {gate!r} gate")
        supplied[gate] = receipt

    missing = [gate for gate in required if gate not in supplied]
    if missing:
        raise ApprovalMissing(f"missing approvals for gates {missing}")
    extra = sorted(set(supplied) - set(required))
    if extra:
        raise ApprovalMissing(f"receipts supplied for gates this plan does not need: {extra}")

    consumptions = []
    for gate in required:
        receipt = supplied[gate]
        validate_receipt(receipt, plan_sha256=plan["plan_sha256"], gate=gate, now=now)
        consumptions.append(
            journal.consume(
                receipt,
                consumption_id=f"consumption-{gate.replace('_', '-')}-1",
                consumed_at=consumed_at,
            )
        )
    return consumptions


def parse_response(response: ProviderResponse, *, plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the reply against the approved expectation, then quarantine its bytes.

    The image is returned in memory for the caller to normalize and publish. Nothing here
    writes it, so a response that fails any check never becomes an artifact.
    """
    if response.status != 200:
        failure = classify_status(response.status)
        detail = str(response.body.get("error", {}).get("message", "")) or "provider error"
        if failure is not ModerationRejected and "moderation" in detail.lower():
            raise ModerationRejected(detail)
        raise failure(detail)

    data = response.body.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise ResponseInvalid("the provider did not return exactly one image")

    encoded = data[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise ResponseInvalid("the provider returned no base64 image payload")
    try:
        content = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ResponseInvalid("the provider payload is not valid base64") from error
    if not content:
        raise ResponseInvalid("the provider returned an empty image")
    if len(content) > MAX_RESPONSE_BYTES:
        raise ResponseInvalid(f"the provider image exceeds {MAX_RESPONSE_BYTES} bytes")

    expected = plan["expected_view"]
    output_format = plan["controls"]["output_format"]
    expected_media_type = _OUTPUT_MEDIA_TYPES[output_format]
    if expected["media_type"] != expected_media_type:
        raise ResponseInvalid("the approved expectation contradicts the approved controls")

    signature = {
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
        "image/webp": b"RIFF",
    }[expected_media_type]
    if not content.startswith(signature):
        raise ResponseInvalid(f"the provider payload is not {expected_media_type} as approved")

    usage = response.body.get("usage") or {}
    return {
        "image_bytes": content,
        "image_sha256": sha256_bytes(content),
        "media_type": expected_media_type,
        "request_id": response.request_id,
        # Returned usage and actual cost are recorded separately from the preflight
        # estimate; they are never folded into it.
        "reported_usage": {
            key: usage[key] for key in sorted(usage) if isinstance(usage[key], (int, float))
        },
        "actual_cost": response.body.get("actual_cost"),
        "preflight_estimate": {
            "estimated_cost": plan["cost_estimate"]["estimated_cost"],
            "maximum_cost": plan["cost_estimate"]["maximum_cost"],
        },
    }


def generate(
    *,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    prompt_path: Path,
    run_directory: Path,
    journal: ConsumptionJournal,
    now: datetime,
    consumed_at: str,
    transport: Transport | None = None,
    secret_resolver: SecretResolver = refusing_secret_resolver,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    canceled: bool = False,
) -> dict[str, Any]:
    """Run the one permitted provider call, or refuse before reaching transport."""
    transport = transport if transport is not None else DeniedTransport()

    # 1. Declarations and the closed control surface, before anything else.
    require_subject_declaration(plan["subject"])
    validate_controls(plan["controls"])
    verify_plan_seal(plan)

    # 2. Evidence must be executable.
    verify_evidence_freshness(evidence, now=now)
    verify_evidence_binding(plan, evidence)

    # 3. The private prompt must be the approved one, reread now.
    prompt = read_prompt(prompt_path, expected_sha256=plan["prompt_sha256"])
    parts = load_attachments(plan, run_directory=run_directory)

    if canceled:
        raise RequestCanceled("the caller cancelled before any approval was consumed")

    # 4. Only now are receipts consumed, and only then is a credential resolved.
    consumptions = consume_receipts(
        plan=plan, receipts=receipts, journal=journal, now=now, consumed_at=consumed_at
    )

    credential = secret_resolver()
    if not isinstance(credential, str) or not credential:
        raise CredentialUnavailable("the secret interface returned no credential")

    request = build_request(plan=plan, prompt=prompt, parts=parts, timeout_seconds=timeout_seconds)
    try:
        response = transport.send(request, credential=credential)
    except TimeoutError as error:
        raise ProviderTimeout("the provider did not answer within the deadline") from error

    result = parse_response(response, plan=plan)
    return {
        **result,
        "approvals": consumptions,
        # A log-safe record. No credential, no prompt text, no image bytes.
        "request_record": request.redacted(),
    }


__all__ = [
    "EVIDENCE_TTL",
    "PROVIDER",
    "ApprovalMissing",
    "CredentialUnavailable",
    "EvidenceStale",
    "ModerationRejected",
    "PlanMismatch",
    "RequestCanceled",
    "ResponseInvalid",
    "SubjectDeclarationRequired",
    "consume_receipts",
    "generate",
    "parse_response",
    "verify_evidence_binding",
    "verify_evidence_freshness",
    "verify_plan_seal",
]
