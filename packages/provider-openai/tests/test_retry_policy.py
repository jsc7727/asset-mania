"""Retries are reported, never performed, and never reuse an approval."""

import pytest
from asset_mania_provider_openai.errors import (
    ApprovalMissing,
    EvidenceStale,
    ModerationRejected,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
    RequestCanceled,
    RequestRejected,
    ResponseInvalid,
)

TRANSIENT = (RateLimited, ProviderUnavailable, ProviderTimeout)
TERMINAL = (
    RequestRejected,
    ModerationRejected,
    ResponseInvalid,
    ApprovalMissing,
    EvidenceStale,
    RequestCanceled,
)


@pytest.mark.parametrize("failure", TRANSIENT)
def test_a_transient_failure_says_a_retry_could_succeed(failure) -> None:
    assert failure.transient is True
    assert issubclass(failure, ProviderError)


@pytest.mark.parametrize("failure", TERMINAL)
def test_a_terminal_failure_says_a_retry_cannot_help(failure) -> None:
    assert failure.transient is False


def test_the_adapter_exposes_no_retry_helper() -> None:
    """A paid retry is a new approval, so there is deliberately nothing to call."""
    from asset_mania_provider_openai import client

    names = [name for name in dir(client) if "retry" in name.lower()]
    assert names == []


def test_transience_is_advisory_not_permission() -> None:
    """Even a transient failure needs a fresh receipt: the journal has already spent one."""
    assert RateLimited.transient is True
    assert issubclass(RateLimited, ProviderError)
