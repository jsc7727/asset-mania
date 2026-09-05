# Local Face Standing Consent Design

Status: approved in chat on 2026-08-23

## Goal

Stop requiring a new `face_rights:<plan_sha256>` acknowledgement for every local retry or new
local face-geometry plan that uses the same already-authorized source bytes.

## Scope

A standing consent is reusable only when all of the following are exact:

- subject is `real_person`;
- source SHA-256 matches the consent;
- scope is `local-network-denied-face-geometry-v1`;
- MICA and DECA remain local and network-denied;
- no source, crop, landmarks, identity feature, or model parameters are persisted.

It does not authorize uploads, remote generation, paid APIs/compute, model or dependency downloads,
publishing, identity comparison, recognition, or a different source digest. Those gates remain
unchanged. Deleting the private consent file revokes it.

## Private record

The create-only canonical JSON record contains exactly:

```text
schema_id = asset-mania/local-face-standing-consent
schema_version = 0.1
subject = real_person
source_sha256 = lowercase SHA-256
scope = local-network-denied-face-geometry-v1
issuer_type = user
issued_at = RFC 3339 UTC
authorization_evidence_sha256 = digest of the explicit user request, never its text
consent_sha256 = canonical digest of every prior field
```

It contains no path, basename, person identifier, prompt, free text, image, feature, or expiry.
The record stays under `.asset-mania/` and release/publication checks reject it if tracked.

## Plan and execution binding

- `geometry-plan --standing-consent PATH` validates the record against `--source-sha256` before
  creating a run.
- The plan binds `authorization_mode=standing_local_source_consent_v1` and the exact
  `standing_consent_sha256`; changing either creates a different plan digest.
- `mica-run --standing-consent PATH` revalidates the same record and writes a private authorization
  audit record before opening the source. No consumption journal is used because the scope is
  intentionally reusable.
- The existing plan-bound single-use receipt path remains supported when no standing consent is
  selected.
- The standing-consent digest fills the existing rights-authorization digest slot in the plugin
  request and likeness disclosure; protocol v1 field names are retained for compatibility.

## Failure behavior

Missing, edited, mismatched-source, wrong-scope, non-user, malformed, or unsealed consent fails
before source open. The controller never auto-expands consent to another source or external action.

