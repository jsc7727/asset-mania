"""Private standing consent for one exact local face-geometry source."""

import os
import re
from collections.abc import Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from asset_mania_contracts import canonical_digest, canonical_json

_SCHEMA_ID = "asset-mania/local-face-standing-consent"
_SCHEMA_VERSION = "0.1"
_SUBJECT = "real_person"
_SCOPE = "local-network-denied-face-geometry-v1"
_ISSUER_TYPE = "user"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$")
_FIELDS = {
    "schema_id",
    "schema_version",
    "subject",
    "source_sha256",
    "scope",
    "issuer_type",
    "issued_at",
    "authorization_evidence_sha256",
    "consent_sha256",
}
_CONSTANTS = {
    "schema_id": _SCHEMA_ID,
    "schema_version": _SCHEMA_VERSION,
    "subject": _SUBJECT,
    "scope": _SCOPE,
    "issuer_type": _ISSUER_TYPE,
}

__all__ = [
    "build_local_face_standing_consent",
    "validate_local_face_standing_consent",
    "write_local_face_standing_consent",
]


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _require_issued_at(value: object) -> str:
    if not isinstance(value, str) or _RFC3339.fullmatch(value) is None:
        raise ValueError("issued_at must be an RFC 3339 timestamp with an explicit UTC offset")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("issued_at must be a valid RFC 3339 UTC instant") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("issued_at must represent an RFC 3339 UTC instant")
    return value


def build_local_face_standing_consent(
    *, source_sha256: str, issued_at: str, authorization_evidence_sha256: str
) -> dict[str, Any]:
    """Seal the closed local-only consent record without retaining authorization text."""
    preimage = {
        "schema_id": _SCHEMA_ID,
        "schema_version": _SCHEMA_VERSION,
        "subject": _SUBJECT,
        "source_sha256": _require_sha256(source_sha256, "source_sha256"),
        "scope": _SCOPE,
        "issuer_type": _ISSUER_TYPE,
        "issued_at": _require_issued_at(issued_at),
        "authorization_evidence_sha256": _require_sha256(
            authorization_evidence_sha256, "authorization_evidence_sha256"
        ),
    }
    return {**preimage, "consent_sha256": canonical_digest(preimage)}


def validate_local_face_standing_consent(
    record: Mapping[str, Any], *, source_sha256: str
) -> dict[str, Any]:
    """Validate an unedited consent for the exact expected source digest."""
    if not isinstance(record, Mapping) or set(record) != _FIELDS:
        raise ValueError("standing consent must contain exactly the closed consent fields")

    for field, expected in _CONSTANTS.items():
        if record[field] != expected:
            raise ValueError(f"{field} does not match the local standing-consent contract")

    expected_source = _require_sha256(source_sha256, "source_sha256")
    observed_source = _require_sha256(record["source_sha256"], "source_sha256")
    _require_sha256(record["authorization_evidence_sha256"], "authorization_evidence_sha256")
    _require_issued_at(record["issued_at"])
    observed_seal = _require_sha256(record["consent_sha256"], "consent_sha256")
    preimage = {key: value for key, value in record.items() if key != "consent_sha256"}
    if observed_seal != canonical_digest(preimage):
        raise ValueError("consent_sha256 does not match the standing consent content")
    if observed_source != expected_source:
        raise ValueError("standing consent is bound to a different source digest")
    return dict(record)


def write_local_face_standing_consent(record: Mapping[str, Any], path: Path) -> None:
    """Create one private canonical consent file without replacing an existing file."""
    validated = validate_local_face_standing_consent(
        record, source_sha256=record.get("source_sha256", "")
    )
    payload = canonical_json(validated).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
