import json

import pytest
from asset_mania_contracts import canonical_digest
from asset_mania_pipeline import (
    build_local_face_standing_consent,
    validate_local_face_standing_consent,
    write_local_face_standing_consent,
)

SOURCE_SHA256 = "a1" * 32
EVIDENCE_SHA256 = "b2" * 32
ISSUED_AT = "2026-08-23T12:34:56Z"
FIELDS = {
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


def _record():
    return build_local_face_standing_consent(
        source_sha256=SOURCE_SHA256,
        issued_at=ISSUED_AT,
        authorization_evidence_sha256=EVIDENCE_SHA256,
    )


def test_builds_exact_closed_local_user_consent_with_canonical_self_seal():
    record = _record()

    assert set(record) == FIELDS
    assert record == {
        "schema_id": "asset-mania/local-face-standing-consent",
        "schema_version": "0.1",
        "subject": "real_person",
        "source_sha256": SOURCE_SHA256,
        "scope": "local-network-denied-face-geometry-v1",
        "issuer_type": "user",
        "issued_at": ISSUED_AT,
        "authorization_evidence_sha256": EVIDENCE_SHA256,
        "consent_sha256": "0de368acd467b95d08408899221fd33741a269a3b409726cd56c01cf4ecba5ae",
    }
    preimage = {key: value for key, value in record.items() if key != "consent_sha256"}
    assert record["consent_sha256"] == canonical_digest(preimage)


@pytest.mark.parametrize("field", ["source_sha256", "authorization_evidence_sha256"])
def test_rejects_non_lowercase_or_malformed_sha256(field):
    values = {
        "source_sha256": SOURCE_SHA256,
        "issued_at": ISSUED_AT,
        "authorization_evidence_sha256": EVIDENCE_SHA256,
    }
    values[field] = "A1" * 32

    with pytest.raises(ValueError, match=field):
        build_local_face_standing_consent(**values)


@pytest.mark.parametrize(
    "issued_at",
    ["2026-08-23T12:34:56", "2026-08-23T12:34:56+09:00", "2026-02-30T12:34:56Z"],
)
def test_rejects_timestamp_that_is_not_a_valid_rfc3339_utc_instant(issued_at):
    with pytest.raises(ValueError, match="issued_at"):
        build_local_face_standing_consent(
            source_sha256=SOURCE_SHA256,
            issued_at=issued_at,
            authorization_evidence_sha256=EVIDENCE_SHA256,
        )


def test_validate_returns_a_copy_for_the_exact_source():
    record = _record()

    validated = validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256)

    assert validated == record
    assert validated is not record


@pytest.mark.parametrize(
    "issued_at",
    ["2026-08-23T12:34:56.123Z", "2026-08-23T12:34:56+00:00"],
)
def test_validate_accepts_rfc3339_fractional_seconds_and_explicit_utc_offset(issued_at):
    record = _record()
    record["issued_at"] = issued_at
    preimage = {key: item for key, item in record.items() if key != "consent_sha256"}
    record["consent_sha256"] = canonical_digest(preimage)

    assert validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256) == record


@pytest.mark.parametrize(
    "issued_at",
    ["2026-08-23T12:34:56-00:00", "2026-08-23T12:34:56.123-00:00"],
)
def test_validate_rejects_rfc3339_unknown_local_offset(issued_at):
    record = _record()
    record["issued_at"] = issued_at
    preimage = {key: item for key, item in record.items() if key != "consent_sha256"}
    record["consent_sha256"] = canonical_digest(preimage)

    with pytest.raises(ValueError, match="issued_at"):
        validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256)


def test_validate_rejects_a_different_source_before_accepting_consent():
    with pytest.raises(ValueError, match="different source"):
        validate_local_face_standing_consent(_record(), source_sha256="c3" * 32)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_id", "another/schema"),
        ("schema_version", "0.2"),
        ("subject", "synthetic_person"),
        ("scope", "external-face-geometry"),
        ("issuer_type", "provider"),
    ],
)
def test_validate_rejects_wrong_closed_constant_even_when_resealed(field, value):
    record = _record()
    record[field] = value
    preimage = {key: item for key, item in record.items() if key != "consent_sha256"}
    record["consent_sha256"] = canonical_digest(preimage)

    with pytest.raises(ValueError, match=field):
        validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256)


def test_validate_rejects_an_edited_self_seal():
    record = _record()
    record["issued_at"] = "2026-08-23T12:34:57Z"

    with pytest.raises(ValueError, match="consent_sha256"):
        validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256)


@pytest.mark.parametrize(
    "field",
    ["path", "source_path", "basename", "name", "prompt", "authorization_text", "free_text"],
)
def test_validate_rejects_extra_path_name_prompt_or_raw_text_fields(field):
    record = _record()
    record[field] = "private value"
    preimage = {key: item for key, item in record.items() if key != "consent_sha256"}
    record["consent_sha256"] = canonical_digest(preimage)

    with pytest.raises(ValueError, match="exactly"):
        validate_local_face_standing_consent(record, source_sha256=SOURCE_SHA256)


def test_write_creates_canonical_json_once_and_never_overwrites(tmp_path):
    record = _record()
    path = tmp_path / "standing-consent.json"

    write_local_face_standing_consent(record, path)

    assert json.loads(path.read_text(encoding="utf-8")) == record
    original = path.read_bytes()
    replacement = build_local_face_standing_consent(
        source_sha256=SOURCE_SHA256,
        issued_at="2026-08-23T12:34:57Z",
        authorization_evidence_sha256=EVIDENCE_SHA256,
    )
    with pytest.raises(FileExistsError):
        write_local_face_standing_consent(replacement, path)
    assert path.read_bytes() == original
