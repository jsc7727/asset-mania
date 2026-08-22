"""Exact-host HTTPS multipart transport for the approved OpenAI image endpoint."""

from __future__ import annotations

import http.client
import json
import secrets
import ssl
from collections.abc import Callable, Mapping
from typing import Any

from .errors import CredentialUnavailable, RequestRejected, ResponseInvalid
from .transport import OFFICIAL_ENDPOINT, OFFICIAL_HOST, ProviderRequest, ProviderResponse

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _default_connection(host: str, timeout: int):
    return http.client.HTTPSConnection(
        host,
        timeout=timeout,
        context=ssl.create_default_context(),
    )


def _quoted(value: str, field: str) -> str:
    if not value or any(character in value for character in ('"', "\r", "\n")):
        raise RequestRejected(f"multipart {field} contains an unsafe value")
    return value


def _multipart(request: ProviderRequest, boundary: str) -> bytes:
    marker = boundary.encode("ascii")
    chunks: list[bytes] = []
    for name, value in request.fields.items():
        safe_name = _quoted(str(name), "field name")
        chunks.extend(
            [
                b"--" + marker + b"\r\n",
                f'Content-Disposition: form-data; name="{safe_name}"\r\n\r\n'.encode(),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )
    for part in request.parts:
        safe_field = _quoted(part.field_name, "part field")
        safe_filename = _quoted(part.filename, "filename")
        chunks.extend(
            [
                b"--" + marker + b"\r\n",
                (
                    f'Content-Disposition: form-data; name="{safe_field}"; '
                    f'filename="{safe_filename}"\r\n'
                ).encode(),
                f"Content-Type: {part.media_type}\r\n\r\n".encode(),
                part.content,
                b"\r\n",
            ]
        )
    chunks.append(b"--" + marker + b"--\r\n")
    return b"".join(chunks)


class HTTPSMultipartTransport:
    """Send one bounded request without proxy inheritance or redirect following."""

    def __init__(
        self,
        connection_factory: Callable[[str, int], Any] | None = None,
    ) -> None:
        self.connection_factory = connection_factory or _default_connection

    def send(self, request: ProviderRequest, *, credential: str) -> ProviderResponse:
        if request.method != "POST":
            raise RequestRejected("the live image transport accepts POST only")
        if request.endpoint != OFFICIAL_ENDPOINT:
            raise RequestRejected(f"the live image transport refuses endpoint {request.endpoint!r}")
        if not isinstance(credential, str) or not credential:
            raise CredentialUnavailable("the live image transport received no credential")

        boundary = f"asset-mania-{secrets.token_hex(16)}"
        body = _multipart(request, boundary)
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {credential}",
            "Content-Length": str(len(body)),
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "asset-mania-openai-turntable/0.2",
        }
        connection = self.connection_factory(OFFICIAL_HOST, request.timeout_seconds)
        try:
            connection.request("POST", OFFICIAL_ENDPOINT, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(MAX_RESPONSE_BYTES + 1)
            response_headers = {
                str(name).lower(): str(value) for name, value in response.getheaders()
            }
        finally:
            connection.close()

        if 300 <= response.status < 400:
            raise RequestRejected("the live image transport refuses redirects")
        if len(payload) > MAX_RESPONSE_BYTES:
            raise ResponseInvalid(f"the provider response exceeds {MAX_RESPONSE_BYTES} bytes")
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ResponseInvalid("the provider response is not JSON") from error
        if not isinstance(decoded, Mapping):
            raise ResponseInvalid("the provider JSON response is not an object")
        return ProviderResponse(
            status=int(response.status),
            body=dict(decoded),
            request_id=response_headers.get("x-request-id"),
            headers=response_headers,
        )


__all__ = ["MAX_RESPONSE_BYTES", "HTTPSMultipartTransport"]
