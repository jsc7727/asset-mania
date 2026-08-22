"""The live HTTPS boundary is exact-host, bounded, and credential-safe."""

import json

import pytest
from asset_mania_provider_openai import HTTPSMultipartTransport
from asset_mania_provider_openai.errors import RequestRejected, ResponseInvalid
from asset_mania_provider_openai.transport import MultipartPart, ProviderRequest


class FakeResponse:
    def __init__(self, *, status: int = 200, body: bytes = b"{}", headers=None) -> None:
        self.status = status
        self.body = body
        self.headers = headers or {"x-request-id": "request-live-1"}
        self.read_sizes: list[int] = []

    def read(self, size: int) -> bytes:
        self.read_sizes.append(size)
        return self.body[:size]

    def getheaders(self):
        return list(self.headers.items())


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.request_record = None
        self.closed = False

    def request(self, method, endpoint, *, body, headers) -> None:
        self.request_record = {
            "method": method,
            "endpoint": endpoint,
            "body": body,
            "headers": headers,
        }

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def _request(endpoint: str = "/v1/images/edits") -> ProviderRequest:
    return ProviderRequest(
        method="POST",
        endpoint=endpoint,
        fields={
            "model": "gpt-image-2-2026-04-21",
            "prompt": "private prompt",
            "n": "1",
        },
        parts=[
            MultipartPart(
                field_name="image[]",
                filename="source-cutout.png",
                media_type="image/png",
                content=b"\x89PNG\r\n\x1a\nprivate-image",
            )
        ],
        metadata={"target_yaw": "45"},
    )


def test_live_transport_posts_one_bounded_multipart_request() -> None:
    response = FakeResponse(body=json.dumps({"data": []}).encode())
    connection = FakeConnection(response)
    constructed = []

    def factory(host: str, timeout: int):
        constructed.append((host, timeout))
        return connection

    transport = HTTPSMultipartTransport(connection_factory=factory)
    result = transport.send(_request(), credential="PROVIDER-CREDENTIAL-FOR-TESTS")

    assert constructed == [("api.openai.com", 120)]
    assert connection.request_record["method"] == "POST"
    assert connection.request_record["endpoint"] == "/v1/images/edits"
    headers = connection.request_record["headers"]
    assert headers["Authorization"] == "Bearer PROVIDER-CREDENTIAL-FOR-TESTS"
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    body = connection.request_record["body"]
    assert body.index(b'name="model"') < body.index(b'name="prompt"')
    assert body.index(b'name="prompt"') < body.index(b'name="image[]"')
    assert response.read_sizes == [32 * 1024 * 1024 + 1]
    assert result.request_id == "request-live-1"
    assert connection.closed is True


def test_live_transport_refuses_another_endpoint_before_connection() -> None:
    constructed = []
    transport = HTTPSMultipartTransport(
        connection_factory=lambda host, timeout: constructed.append((host, timeout))
    )
    with pytest.raises(RequestRejected, match="endpoint"):
        transport.send(_request("/v1/files"), credential="x")
    assert constructed == []


def test_live_transport_refuses_oversized_or_non_json_responses(monkeypatch) -> None:
    import asset_mania_provider_openai.live_transport as live

    monkeypatch.setattr(live, "MAX_RESPONSE_BYTES", 16)
    oversized = FakeConnection(FakeResponse(body=b"x" * 17))
    with pytest.raises(ResponseInvalid, match="exceeds"):
        HTTPSMultipartTransport(connection_factory=lambda host, timeout: oversized).send(
            _request(), credential="x"
        )

    invalid = FakeConnection(FakeResponse(body=b"not-json"))
    with pytest.raises(ResponseInvalid, match="JSON"):
        HTTPSMultipartTransport(connection_factory=lambda host, timeout: invalid).send(
            _request(), credential="x"
        )


def test_live_transport_refuses_redirects() -> None:
    connection = FakeConnection(FakeResponse(status=302, body=b"{}"))
    with pytest.raises(RequestRejected, match="redirect"):
        HTTPSMultipartTransport(connection_factory=lambda host, timeout: connection).send(
            _request(), credential="x"
        )
