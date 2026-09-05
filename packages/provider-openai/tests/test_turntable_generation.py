"""Seven paid GPT Image calls form one approval-bound turntable run."""

import base64
from pathlib import Path

import pytest
from asset_mania_contracts import build_turntable_plan
from asset_mania_pipeline import sha256_bytes, sha256_file
from asset_mania_provider_openai import generate_turntable
from asset_mania_provider_openai.errors import PlanMismatch, ProviderTimeout
from asset_mania_provider_openai.transport import ProviderResponse, RecordingTransport
from conftest import CONSUMED_AT, NOW
from PIL import Image

BASE_PROMPT = "Preserve the approved subject in a neutral studio turntable.\n"


def _cutout(tmp_path: Path) -> Path:
    path = tmp_path / "source-cutout.png"
    Image.new("RGBA", (1024, 1024), (70, 90, 120, 255)).save(path)
    return path


def _plan(evidence, cutout: Path):
    return build_turntable_plan(
        source_image_sha256="a1" * 32,
        source_width=1024,
        source_height=1024,
        source_mask_sha256="a2" * 32,
        source_cutout_sha256=sha256_file(cutout),
        prompt_sha256=sha256_bytes(BASE_PROMPT.encode("utf-8")),
        provider_evidence_sha256=evidence["evidence_sha256"],
        controls={
            "size": "1024x1024",
            "quality": "medium",
            "background": "opaque",
            "output_format": "png",
            "moderation": "auto",
        },
        subject="real_person",
        estimated_cost="0.371000",
        maximum_cost="0.700000",
    )


def _response(index: int) -> ProviderResponse:
    payload = b"\x89PNG\r\n\x1a\n" + f"generated-{index}".encode() + bytes(24)
    return ProviderResponse(
        status=200,
        body={
            "data": [{"b64_json": base64.b64encode(payload).decode("ascii")}],
            "usage": {"input_tokens": 100 + index, "output_tokens": 200 + index},
            "actual_cost": "0.053000",
        },
        request_id=f"request-{index}",
    )


def test_one_approved_run_sends_seven_ordered_calls(
    tmp_path: Path, evidence, receipts, journal, credential
) -> None:
    cutout = _cutout(tmp_path)
    plan = _plan(evidence, cutout)
    transport = RecordingTransport([_response(index) for index in range(1, 8)])

    results = generate_turntable(
        plan=plan,
        evidence=evidence,
        receipts=receipts(plan["plan_sha256"], plan["required_gates"]),
        base_prompt=BASE_PROMPT,
        cutout_path=cutout,
        journal=journal,
        now=NOW,
        consumed_at=CONSUMED_AT,
        transport=transport,
        secret_resolver=credential,
    )

    assert [result.yaw for result in results] == [45, 90, 135, 180, 225, 270, 315]
    assert [item["metadata"]["target_yaw"] for item in transport.sent] == [
        "45",
        "90",
        "135",
        "180",
        "225",
        "270",
        "315",
    ]
    assert all(len(item["parts"]) == 1 for item in transport.sent)
    assert all(
        set(item["parts"][0]) == {"field_name", "media_type", "byte_size"}
        for item in transport.sent
    )


def test_tampered_cutout_fails_before_transport(
    tmp_path: Path, evidence, receipts, journal, credential
) -> None:
    cutout = _cutout(tmp_path)
    plan = _plan(evidence, cutout)
    cutout.write_bytes(cutout.read_bytes() + b"tampered")
    transport = RecordingTransport([_response(1)])

    with pytest.raises(PlanMismatch, match="cutout"):
        generate_turntable(
            plan=plan,
            evidence=evidence,
            receipts=receipts(plan["plan_sha256"], plan["required_gates"]),
            base_prompt=BASE_PROMPT,
            cutout_path=cutout,
            journal=journal,
            now=NOW,
            consumed_at=CONSUMED_AT,
            transport=transport,
            secret_resolver=credential,
        )
    assert transport.sent == []


def test_first_failed_paid_call_stops_without_retry(
    tmp_path: Path, evidence, receipts, journal, credential
) -> None:
    cutout = _cutout(tmp_path)
    plan = _plan(evidence, cutout)

    class TimeoutTransport:
        def __init__(self) -> None:
            self.calls = 0

        def send(self, request, *, credential):
            self.calls += 1
            raise TimeoutError("provider timed out")

    transport = TimeoutTransport()
    with pytest.raises(ProviderTimeout):
        generate_turntable(
            plan=plan,
            evidence=evidence,
            receipts=receipts(plan["plan_sha256"], plan["required_gates"]),
            base_prompt=BASE_PROMPT,
            cutout_path=cutout,
            journal=journal,
            now=NOW,
            consumed_at=CONSUMED_AT,
            transport=transport,
            secret_resolver=credential,
        )
    assert transport.calls == 1
