"""Provider tests never reach the network. Sockets are denied outright."""

import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest
from asset_mania_contracts import build_provider_plan
from asset_mania_pipeline import (
    ConsumptionJournal,
    acknowledgement_text,
    issue_receipt,
    sha256_bytes,
    sha256_file,
)

NOW = datetime(2026, 8, 19, 9, 40, tzinfo=UTC)
CONSUMED_AT = "2026-08-19T09:40:00Z"
PROMPT_TEXT = "a studio product photo of the target, neutral lighting\n"
DISCLOSURE = "This run uploads four reference images to OpenAI."


@pytest.fixture(autouse=True)
def deny_sockets(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the provider suite must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def evidence():
    from asset_mania_contracts import canonical_digest

    body = {
        "schema_id": "asset-mania/provider-evidence",
        "schema_version": "1.0",
        "provider": "openai",
        "model_snapshot": "gpt-image-2-2026-04-21",
        "sources": [
            {
                "url": "https://platform.openai.com/docs/pricing",
                "retrieved_at": "2026-08-19T08:00:00Z",
                "content_sha256": "e2" * 32,
                "parser_profile": "openai-gpt-image-2-v1",
            }
        ],
        "retrieved_at": "2026-08-19T08:00:00Z",
        "expires_at": "2026-08-20T08:00:00Z",
        "content_digest": "e1" * 32,
        "data_policy": {
            "training_default": "not_used_unless_opted_in",
            "application_state": "none",
            "abuse_monitoring_days": 30,
            "zdr": "eligible_with_approval",
            "csam_review_exception": True,
            "effective_region": "unknown",
        },
        "pricing": {
            "currency": "USD",
            "rate_mode": "standard",
            "per_million_tokens": {
                "text_input": "5.000000",
                "cached_text_input": "1.250000",
                "image_input": "8.000000",
                "cached_image_input": "2.000000",
                "image_output": "30.000000",
            },
            "output_cost_rows": [
                {"quality": "low", "size": "1024x1024", "usd": "0.006000"},
                {"quality": "low", "size": "1024x1536", "usd": "0.005000"},
                {"quality": "low", "size": "1536x1024", "usd": "0.005000"},
                {"quality": "medium", "size": "1024x1024", "usd": "0.053000"},
                {"quality": "medium", "size": "1024x1536", "usd": "0.041000"},
                {"quality": "medium", "size": "1536x1024", "usd": "0.041000"},
                {"quality": "high", "size": "1024x1024", "usd": "0.211000"},
                {"quality": "high", "size": "1024x1536", "usd": "0.165000"},
                {"quality": "high", "size": "1536x1024", "usd": "0.165000"},
            ],
            "retrieved_at": "2026-08-19T08:00:00Z",
            "content_sha256": "e2" * 32,
        },
        "evidence_sha256": "",
    }
    preimage = {key: value for key, value in body.items() if key != "evidence_sha256"}
    return {**preimage, "evidence_sha256": canonical_digest(preimage)}


@pytest.fixture
def attachments(tmp_path: Path):
    """Four PNG attachments on disk, with their approved digests and sizes."""
    directory = tmp_path / "run" / "artifacts" / "conditioning"
    directory.mkdir(parents=True)
    roles = ("beauty", "depth_preview", "normal_preview", "mask")
    files = {
        "beauty": "beauty.png",
        "depth_preview": "depth-preview.png",
        "normal_preview": "normal-preview.png",
        "mask": "mask.png",
    }
    records = []
    paths = {}
    for index, role in enumerate(roles):
        path = directory / files[role]
        payload = b"\x89PNG\r\n\x1a\n" + role.encode("utf-8") + bytes(16)
        path.write_bytes(payload)
        records.append(
            {
                "role": role,
                "multipart_field": "image[]",
                "index": index,
                "sha256": sha256_file(path),
                "byte_size": path.stat().st_size,
                "media_type": "image/png",
                "upload_eligible": True,
            }
        )
        paths[role] = f"artifacts/conditioning/{files[role]}"
    return {"records": records, "paths": paths, "run_directory": tmp_path / "run"}


@pytest.fixture
def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "private-prompt.txt"
    path.write_text(PROMPT_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def plan(evidence, attachments, prompt_file):
    def build(*, subject: str = "synthetic_person", **control_overrides):
        controls = {
            "n": 1,
            "size": "1024x1024",
            "quality": "medium",
            "background": "auto",
            "output_format": "png",
            "output_compression": None,
            "moderation": "auto",
        }
        controls.update(control_overrides)
        cost = {
            "currency": "USD",
            "rate_retrieved_at": evidence["pricing"]["retrieved_at"],
            "rate_digest": evidence["evidence_sha256"],
            "text_input_tokens_assumed": 120,
            "image_input_tokens_assumed": 3200,
            "cached_text_input_tokens_assumed": 0,
            "cached_image_input_tokens_assumed": 0,
            "n": 1,
            "size": controls["size"],
            "quality": controls["quality"],
            "formula": "uncached_inputs_plus_published_output_row_v1",
            "rounding": "ceiling_6_decimal_places",
            "estimate_uncertainty": "input_tokens_assumed",
            "estimated_cost": "0.079200",
            "maximum_cost": "0.100000",
        }
        width, height = (int(part) for part in controls["size"].split("x"))
        media = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
        built = build_provider_plan(
            condition_manifest_sha256="b3" * 32,
            attachments=attachments["records"],
            prompt_sha256=sha256_bytes(PROMPT_TEXT.encode("utf-8")),
            controls=controls,
            subject=subject,
            policy_evidence={
                "artifact_sha256": "e3" * 32,
                "source_urls": ["https://platform.openai.com/docs/pricing"],
                "retrieved_at": evidence["retrieved_at"],
                "expires_at": evidence["expires_at"],
                "content_digest": evidence["content_digest"],
                "training_default": "not_used_unless_opted_in",
                "application_state": "none",
                "abuse_monitoring_days": 30,
                "zdr": "eligible_with_approval",
                "csam_review_exception": True,
                "effective_region": "unknown",
            },
            cost_estimate=cost,
            expected_view={
                "count": 1,
                "width": width,
                "height": height,
                "media_type": media[controls["output_format"]],
                "origin": "generated",
                "alignment_issuer": "provider",
            },
        )
        return {**built, "attachment_paths": attachments["paths"]}

    return build


@pytest.fixture
def receipts():
    def build(plan_sha256: str, gates):
        return [
            issue_receipt(
                receipt_id=f"receipt-{gate.replace('_', '-')}-1",
                plan_sha256=plan_sha256,
                gate=gate,
                acknowledgement=acknowledgement_text(gate, plan_sha256),
                disclosure=DISCLOSURE,
                issued_at="2026-08-19T09:35:00Z",
                expires_at="2026-08-19T10:35:00Z",
            )
            for gate in gates
        ]

    return build


@pytest.fixture
def journal(tmp_path: Path) -> ConsumptionJournal:
    return ConsumptionJournal(tmp_path / "approvals")


@pytest.fixture
def credential():
    return lambda: "PROVIDER-CREDENTIAL-FOR-TESTS"
