"""Provider failures, classified by whether a retry is even permitted.

A paid request is never retried automatically. The classification exists so a caller can
tell a usage error from a transient one and *report* that, not so the adapter can quietly
spend money again.
"""


class ProviderError(Exception):
    """Base class for every provider failure."""

    #: Whether a retry could succeed at all. Even when true, a retry needs a new approval.
    transient = False


class EvidenceStale(ProviderError):
    """The policy and pricing evidence is older than its executable TTL."""


class ApprovalMissing(ProviderError):
    """A required receipt is absent, expired, or bound to another plan."""


class PlanMismatch(ProviderError):
    """The request does not match the approved plan digest."""


class CredentialUnavailable(ProviderError):
    """No credential was supplied through the secret interface."""


class RequestRejected(ProviderError):
    """The provider rejected the request as invalid or disallowed."""


class ModerationRejected(RequestRejected):
    """The provider refused the request on moderation grounds."""


class RateLimited(ProviderError):
    """The provider asked the caller to slow down."""

    transient = True


class ProviderUnavailable(ProviderError):
    """The provider returned a server-side failure."""

    transient = True


class ProviderTimeout(ProviderError):
    """The request exceeded its deadline."""

    transient = True


class RequestCanceled(ProviderError):
    """The caller cancelled before or during transport."""


class ResponseInvalid(ProviderError):
    """The provider's response does not match the approved expectation."""


def classify_status(status: int) -> type[ProviderError]:
    """Map an HTTP status onto the failure this adapter reports."""
    if status == 429:
        return RateLimited
    if 500 <= status <= 599:
        return ProviderUnavailable
    if status == 400:
        return RequestRejected
    if status in (401, 403):
        return CredentialUnavailable
    return RequestRejected
