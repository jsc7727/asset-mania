"""Response validation, retry classification, and cost separation."""

import base64

import pytest
from asset_mania_provider_openai import client
from asset_mania_provider_openai.errors import (
    CredentialUnavailable,
    ModerationRejected,
    ProviderUnavailable,
    RateLimited,
    RequestRejected,
    ResponseInvalid,
    classify_status,
)
from asset_mania_provider_openai.transport import ProviderResponse

PNG = b"\x89PNG\r\n\x1a\n" + b"payload" + bytes(24)
JPEG = b"\xff\xd8\xff" + b"payload" + bytes(24)
WEBP = b"RIFF" + b"payload" + bytes(24)


def _response(content: bytes, **body) -> ProviderResponse:
    payload = {"data": [{"b64_json": base64.b64encode(content).decode("ascii")}]}
    payload.update(body)
    return ProviderResponse(status=200, body=payload, request_id="req_1")


# --- Accepted payloads ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("output_format", "content", "media_type"),
    [("png", PNG, "image/png"), ("jpeg", JPEG, "image/jpeg"), ("webp", WEBP, "image/webp")],
)
def test_each_approved_output_format_is_accepted(
    plan, output_format: str, content: bytes, media_type: str
) -> None:
    overrides = {"output_format": output_format}
    if output_format != "png":
        overrides["output_compression"] = 80
    built = plan(**overrides)
    result = client.parse_response(_response(content), plan=built)
    assert result["media_type"] == media_type
    assert result["image_bytes"] == content
    assert len(result["image_sha256"]) == 64


def test_the_request_id_is_recorded(plan) -> None:
    assert client.parse_response(_response(PNG), plan=plan())["request_id"] == "req_1"


# --- Rejected payloads ---------------------------------------------------------------


def test_a_payload_whose_type_contradicts_the_approval_is_refused(plan) -> None:
    built = plan()
    with pytest.raises(ResponseInvalid, match="not image/png"):
        client.parse_response(_response(JPEG), plan=built)


def test_more_than_one_image_is_refused(plan) -> None:
    body = {
        "data": [
            {"b64_json": base64.b64encode(PNG).decode("ascii")},
            {"b64_json": base64.b64encode(PNG).decode("ascii")},
        ]
    }
    with pytest.raises(ResponseInvalid, match="exactly one image"):
        client.parse_response(ProviderResponse(status=200, body=body), plan=plan())


def test_a_missing_payload_is_refused(plan) -> None:
    with pytest.raises(ResponseInvalid, match="no base64 image payload"):
        client.parse_response(ProviderResponse(status=200, body={"data": [{}]}), plan=plan())


def test_a_non_base64_payload_is_refused(plan) -> None:
    with pytest.raises(ResponseInvalid, match="not valid base64"):
        client.parse_response(
            ProviderResponse(status=200, body={"data": [{"b64_json": "!!!not base64!!!"}]}),
            plan=plan(),
        )


def test_an_empty_payload_string_is_refused(plan) -> None:
    """An empty string carries no payload at all, which is reported before decoding."""
    with pytest.raises(ResponseInvalid, match="no base64 image payload"):
        client.parse_response(
            ProviderResponse(status=200, body={"data": [{"b64_json": ""}]}), plan=plan()
        )


def test_a_payload_that_is_not_the_approved_image_type_is_refused(plan) -> None:
    with pytest.raises(ResponseInvalid, match="not image/png"):
        client.parse_response(
            ProviderResponse(
                status=200, body={"data": [{"b64_json": base64.b64encode(b"\x00").decode()}]}
            ),
            plan=plan(),
        )


def test_an_oversized_payload_is_refused(plan, monkeypatch) -> None:
    monkeypatch.setattr(client, "MAX_RESPONSE_BYTES", 8)
    with pytest.raises(ResponseInvalid, match="exceeds"):
        client.parse_response(_response(PNG), plan=plan())


# --- Failure classification ------------------------------------------------------------


def test_rate_limiting_is_transient() -> None:
    assert classify_status(429) is RateLimited
    assert RateLimited.transient is True


@pytest.mark.parametrize("status", [500, 502, 503, 599])
def test_server_failures_are_transient(status: int) -> None:
    assert classify_status(status) is ProviderUnavailable
    assert ProviderUnavailable.transient is True


def test_a_usage_error_is_not_transient() -> None:
    assert classify_status(400) is RequestRejected
    assert RequestRejected.transient is False


def test_an_auth_failure_reports_a_credential_problem() -> None:
    assert classify_status(401) is CredentialUnavailable
    assert classify_status(403) is CredentialUnavailable


@pytest.mark.parametrize("status", [429, 500, 400, 401])
def test_a_non_success_status_raises_its_classification(plan, status: int) -> None:
    failure = classify_status(status)
    with pytest.raises(failure):
        client.parse_response(
            ProviderResponse(status=status, body={"error": {"message": "nope"}}), plan=plan()
        )


def test_a_moderation_refusal_is_reported_as_such(plan) -> None:
    with pytest.raises(ModerationRejected):
        client.parse_response(
            ProviderResponse(
                status=400, body={"error": {"message": "Rejected by moderation policy"}}
            ),
            plan=plan(),
        )


# --- Cost separation ---------------------------------------------------------------------


def test_returned_usage_is_recorded_separately_from_the_preflight_estimate(plan) -> None:
    built = plan()
    result = client.parse_response(
        _response(PNG, usage={"input_tokens": 3320, "output_tokens": 1024}, actual_cost="0.081000"),
        plan=built,
    )
    assert result["reported_usage"] == {"input_tokens": 3320, "output_tokens": 1024}
    assert result["actual_cost"] == "0.081000"
    assert result["preflight_estimate"] == {
        "estimated_cost": built["cost_estimate"]["estimated_cost"],
        "maximum_cost": built["cost_estimate"]["maximum_cost"],
    }


def test_the_actual_cost_is_never_folded_into_the_estimate(plan) -> None:
    built = plan()
    result = client.parse_response(_response(PNG, actual_cost="9.999999"), plan=built)
    assert result["preflight_estimate"]["estimated_cost"] == "0.079200"
    assert result["actual_cost"] == "9.999999"


def test_a_response_without_usage_reports_an_empty_record(plan) -> None:
    result = client.parse_response(_response(PNG), plan=plan())
    assert result["reported_usage"] == {}
    assert result["actual_cost"] is None
