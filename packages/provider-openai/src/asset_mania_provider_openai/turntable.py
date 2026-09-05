"""Approval-gated seven-call GPT Image 2 turntable generation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from asset_mania_contracts import (
    TURNTABLE_CONTROLS,
    TURNTABLE_ENDPOINT,
    TURNTABLE_MODEL_SNAPSHOT,
    TURNTABLE_YAWS,
    canonical_digest,
)
from asset_mania_pipeline import ConsumptionJournal, sha256_bytes, sha256_file

from .client import consume_receipts, parse_response, verify_evidence_freshness
from .errors import CredentialUnavailable, EvidenceStale, PlanMismatch, ProviderTimeout
from .transport import (
    DEFAULT_TIMEOUT_SECONDS,
    MultipartPart,
    ProviderRequest,
    SecretResolver,
    Transport,
)

MAX_CUTOUT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class TurntableCallResult:
    yaw: int
    image_bytes: bytes
    image_sha256: str
    media_type: str
    request_id: str | None
    reported_usage: Mapping[str, int | float]
    actual_cost: str | None
    request_record: Mapping[str, Any]


def build_turntable_prompt(base_prompt: str, yaw: int) -> str:
    """Bind one generated view to a fixed camera angle and preservation profile."""
    if yaw not in TURNTABLE_YAWS[1:]:
        raise ValueError(f"generated yaw must be one of {list(TURNTABLE_YAWS[1:])}")
    base = base_prompt.rstrip()
    if not base:
        raise ValueError("base prompt must not be empty")
    return (
        f"{base}\n\n"
        f"Target yaw: {yaw} degrees around the subject; pitch 0 degrees; roll 0 degrees. "
        "Keep the same facial proportions, neutral expression, open eyes, closed mouth, "
        "hairstyle, and visible clothing. Show the centered head and upper neck with a level "
        "long-lens studio camera, even diffuse light, and a flat white background. Do not add "
        "text, watermark, jewelry, accessories, or stylization. Unseen geometry is an inference."
    )


def build_turntable_request(
    *, plan: Mapping[str, Any], yaw: int, prompt: str, cutout: bytes
) -> ProviderRequest:
    """Build the exact multipart request for one yaw without adding an API field for yaw."""
    if plan.get("endpoint") != TURNTABLE_ENDPOINT:
        raise PlanMismatch("the turntable plan names another endpoint")
    if plan.get("model") != TURNTABLE_MODEL_SNAPSHOT:
        raise PlanMismatch("the turntable plan names another model snapshot")
    if plan.get("controls") != TURNTABLE_CONTROLS:
        raise PlanMismatch("the turntable plan controls changed")
    controls = plan["controls"]
    return ProviderRequest(
        method="POST",
        endpoint=TURNTABLE_ENDPOINT,
        fields={
            "model": TURNTABLE_MODEL_SNAPSHOT,
            "prompt": prompt,
            "n": "1",
            "size": controls["size"],
            "quality": controls["quality"],
            "background": controls["background"],
            "output_format": controls["output_format"],
            "moderation": controls["moderation"],
        },
        parts=[
            MultipartPart(
                field_name="image[]",
                filename="source-cutout.png",
                media_type="image/png",
                content=cutout,
            )
        ],
        timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        metadata={"target_yaw": str(yaw)},
    )


def _verify_plan(plan: Mapping[str, Any]) -> None:
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if canonical_digest(preimage) != plan.get("plan_sha256"):
        raise PlanMismatch("the turntable plan digest does not match its content")
    if plan.get("yaws") != list(TURNTABLE_YAWS):
        raise PlanMismatch("the turntable plan yaw schedule changed")


def _parse_turntable_response(response, plan: Mapping[str, Any]) -> dict[str, Any]:
    return parse_response(
        response,
        plan={
            "expected_view": {"media_type": "image/png"},
            "controls": {"output_format": "png"},
            "cost_estimate": {
                "estimated_cost": plan["cost_estimate"]["estimated_cost"],
                "maximum_cost": plan["cost_estimate"]["maximum_cost"],
            },
        },
    )


def generate_turntable(
    *,
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    base_prompt: str,
    cutout_path: Path,
    journal: ConsumptionJournal,
    now: datetime,
    consumed_at: str,
    transport: Transport,
    secret_resolver: SecretResolver,
) -> list[TurntableCallResult]:
    """Consume one plan's approvals and make exactly seven sequential paid calls."""
    _verify_plan(plan)
    verify_evidence_freshness(evidence, now=now)
    if evidence.get("model_snapshot") != TURNTABLE_MODEL_SNAPSHOT:
        raise EvidenceStale("PROVIDER_EVIDENCE_STALE: evidence names another model snapshot")
    if plan.get("provider_evidence_sha256") != evidence.get("evidence_sha256"):
        raise EvidenceStale("PROVIDER_EVIDENCE_STALE: plan binds different evidence")
    if sha256_bytes(base_prompt.encode("utf-8")) != plan.get("prompt_sha256"):
        raise PlanMismatch("the base prompt does not match the approved plan digest")
    if not cutout_path.is_file():
        raise PlanMismatch("the approved cutout is missing")
    if cutout_path.stat().st_size > MAX_CUTOUT_BYTES:
        raise PlanMismatch("the approved cutout is oversized")
    if sha256_file(cutout_path) != plan.get("source_cutout_sha256"):
        raise PlanMismatch("the cutout does not match the approved plan")
    cutout = cutout_path.read_bytes()
    if not cutout.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PlanMismatch("the approved cutout is not PNG")

    consume_receipts(
        plan=plan,
        receipts=receipts,
        journal=journal,
        now=now,
        consumed_at=consumed_at,
    )
    credential = secret_resolver()
    if not isinstance(credential, str) or not credential:
        raise CredentialUnavailable("the secret interface returned no credential")

    results: list[TurntableCallResult] = []
    for yaw in TURNTABLE_YAWS[1:]:
        prompt = build_turntable_prompt(base_prompt, yaw)
        request = build_turntable_request(plan=plan, yaw=yaw, prompt=prompt, cutout=cutout)
        try:
            response = transport.send(request, credential=credential)
        except TimeoutError as error:
            raise ProviderTimeout("the provider did not answer within the deadline") from error
        parsed = _parse_turntable_response(response, plan)
        results.append(
            TurntableCallResult(
                yaw=yaw,
                image_bytes=parsed["image_bytes"],
                image_sha256=parsed["image_sha256"],
                media_type=parsed["media_type"],
                request_id=parsed["request_id"],
                reported_usage=parsed["reported_usage"],
                actual_cost=parsed["actual_cost"],
                request_record=request.redacted(),
            )
        )
    return results


__all__ = [
    "MAX_CUTOUT_BYTES",
    "TurntableCallResult",
    "build_turntable_prompt",
    "build_turntable_request",
    "generate_turntable",
]
