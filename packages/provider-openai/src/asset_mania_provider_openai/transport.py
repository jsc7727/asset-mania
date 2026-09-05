"""The transport boundary.

Transport is injected. The adapter never constructs a socket, a session, or a URL opener
of its own, which is what makes "no network before approval" testable rather than a
promise: a test can pass a transport that fails loudly, and the suite denies sockets
outright.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

OFFICIAL_ENDPOINT = "/v1/images/edits"
OFFICIAL_HOST = "api.openai.com"
DEFAULT_TIMEOUT_SECONDS = 120
#: Fields whose value never appears in a record; their digest does instead.
REDACTED_FIELDS = frozenset({"prompt"})


@dataclass(frozen=True, slots=True)
class MultipartPart:
    """One multipart field. Binary content is carried by reference, never inlined here."""

    field_name: str
    filename: str
    media_type: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    method: str
    endpoint: str
    fields: Mapping[str, str]
    parts: Sequence[MultipartPart]
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    #: The credential is resolved at call time and never stored on the request.
    credential_reference: str = "secret-interface"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def redacted(self) -> dict[str, Any]:
        """A log-safe view: no credential, no prompt text, no image bytes.

        The prompt is replaced by its digest rather than dropped, so a record still proves
        *which* prompt was sent without ever carrying the text.
        """
        import hashlib

        fields: dict[str, Any] = {}
        for key, value in sorted(self.fields.items()):
            if key in REDACTED_FIELDS:
                fields[f"{key}_sha256"] = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            else:
                fields[key] = value
        return {
            "method": self.method,
            "endpoint": self.endpoint,
            "fields": fields,
            "parts": [
                {
                    "field_name": part.field_name,
                    "media_type": part.media_type,
                    "byte_size": len(part.content),
                }
                for part in self.parts
            ],
            "metadata": dict(sorted(self.metadata.items())),
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    status: int
    body: Mapping[str, Any]
    request_id: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


class Transport(Protocol):
    """What the adapter needs from a transport, and nothing more."""

    def send(self, request: ProviderRequest, *, credential: str) -> ProviderResponse: ...


class DeniedTransport:
    """A transport that refuses every call. The default, so nothing reaches the network."""

    def __init__(self, reason: str = "no transport was provided") -> None:
        self.reason = reason

    def send(self, request: ProviderRequest, *, credential: str) -> ProviderResponse:
        raise AssertionError(f"transport is denied: {self.reason}")


class RecordingTransport:
    """A fake transport for tests: records what it was asked to send and replays a reply."""

    def __init__(self, responses: Sequence[ProviderResponse]) -> None:
        self._responses = list(responses)
        self.sent: list[dict[str, Any]] = []
        self.credentials_seen: list[str] = []

    def send(self, request: ProviderRequest, *, credential: str) -> ProviderResponse:
        self.sent.append(request.redacted())
        self.credentials_seen.append(credential)
        if not self._responses:
            raise AssertionError("the fake transport has no reply left to give")
        return self._responses.pop(0)


SecretResolver = Callable[[], str]


def refusing_secret_resolver() -> str:
    """The default secret interface: there is no credential unless one is supplied."""
    from .errors import CredentialUnavailable

    raise CredentialUnavailable("no credential was supplied through the secret interface")
