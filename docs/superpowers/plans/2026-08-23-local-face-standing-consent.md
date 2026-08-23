# Local Face Standing Consent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse explicit local face rights for the same source SHA without repeated plan tokens.

**Architecture:** Add a closed source-bound standing-consent record in the pipeline package, bind
its digest into private geometry plans, and let the local E2E runner choose either the existing
single-use receipt or the standing consent. External/paid/download gates are unchanged.

**Tech Stack:** Typed Python, canonical JSON/SHA-256, create-only private files, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-local-face-standing-consent-design.md`

## Global Constraints

- Scope is exactly `local-network-denied-face-geometry-v1` for one exact source SHA-256.
- Record inventory is closed and contains no path, basename, identifier, raw acknowledgement, or
  face feature.
- Existing plan-bound receipts remain supported; external/paid/download approvals are unchanged.
- Consent and authorization are validated before source open.
- Private consent files remain ignored and release/publication checks reject tracked copies.

---

### Task 1: Add the closed standing-consent contract

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/face_consent.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Create: `packages/pipeline/tests/test_face_consent.py`

**Interfaces:**
- Produces: `build_local_face_standing_consent(*, source_sha256, issued_at,
  authorization_evidence_sha256) -> dict`, `validate_local_face_standing_consent(record, *,
  source_sha256) -> dict`, and `write_local_face_standing_consent(record, path) -> None`.

- [ ] **Step 1: Write failing closed-record tests**

Assert exact fields, canonical self-seal, lowercase digests, RFC 3339 timezone, exact scope/user
issuer, create-only write, source mismatch rejection, and rejection of extra path/name/prompt fields.

- [ ] **Step 2: Run RED**

```powershell
uv run pytest packages/pipeline/tests/test_face_consent.py -q
```

Expected: import failure because `face_consent` does not exist.

- [ ] **Step 3: Implement the minimal contract and run GREEN**

Use `canonical_digest`; validate the seal by recomputing without `consent_sha256`; write with
`O_CREAT|O_EXCL`. Export only the three named functions.

- [ ] **Step 4: Verify and commit**

```powershell
uv run pytest packages/pipeline/tests/test_face_consent.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_consent.py packages/pipeline/tests/test_face_consent.py
uv run ruff format --check packages/pipeline/src/asset_mania_pipeline/face_consent.py packages/pipeline/tests/test_face_consent.py
git add packages/pipeline/src/asset_mania_pipeline/face_consent.py packages/pipeline/src/asset_mania_pipeline/__init__.py packages/pipeline/tests/test_face_consent.py
git commit -m "feat(face-geometry): add local standing consent"
```

---

### Task 2: Bind standing consent into the private geometry E2E

**Files:**
- Modify: `scripts/run_face_geometry_e2e.py`
- Modify: `tests/test_face_geometry_e2e.py`
- Modify: `scripts/check_release.py`
- Modify: `tests/test_check_release.py`
- Modify: `rules/agent/behavior-rules.md`

**Interfaces:**
- Consumes: Task 1 consent builders/validators.
- Produces: `geometry-plan --standing-consent`, mutually exclusive `mica-run --rights-store` or
  `--standing-consent`, plan/audit digest binding, and conservative distribution checks.

- [ ] **Step 1: Write failing plan/reuse/privacy tests**

Create a synthetic consent for a synthetic source digest. Assert plan fields and changed plan
digest, two different create-only plans reuse the same consent without a token, source mismatch
fails before `fingerprint_source`, edited consent fails, the existing receipt path still passes,
and portable records contain no consent path or basename.

- [ ] **Step 2: Write failing release tests**

Assert a tracked standing-consent filename/content is rejected with sanitized diagnostics that do
not echo private paths or source digests.

- [ ] **Step 3: Run RED**

```powershell
uv run pytest tests/test_face_geometry_e2e.py packages/pipeline/tests/test_face_consent.py tests/test_check_release.py -q
```

- [ ] **Step 4: Implement plan and stage authorization**

Validate consent during `geometry-plan` and again during `mica-run`; bind mode and digest into the
plan; write an authorization audit whose `receipt_sha256` compatibility field equals the consent
digest and whose mode is explicit. Keep DECA/fusion/disclosure consumers unchanged. Use argparse
mutual exclusion and preserve the existing receipt path.

- [ ] **Step 5: Update distribution and durable rule**

Forbid standing-consent records from tracked/public content. Document source-specific automatic
reuse only for local network-denied geometry; state that remote, paid, download, identity, and
publication gates are not relaxed.

- [ ] **Step 6: Verify and commit**

```powershell
uv run pytest tests/test_face_geometry_e2e.py packages/pipeline/tests/test_face_consent.py tests/test_check_release.py -q
uv run ruff check scripts/run_face_geometry_e2e.py scripts/check_release.py tests/test_face_geometry_e2e.py tests/test_check_release.py
uv run ruff format --check scripts/run_face_geometry_e2e.py scripts/check_release.py tests/test_face_geometry_e2e.py tests/test_check_release.py
uv run python scripts/check_release.py
uv run python scripts/check_publication.py
git diff --check
git add scripts/run_face_geometry_e2e.py tests/test_face_geometry_e2e.py scripts/check_release.py tests/test_check_release.py rules/agent/behavior-rules.md
git commit -m "feat(face-geometry): reuse local source consent"
```
