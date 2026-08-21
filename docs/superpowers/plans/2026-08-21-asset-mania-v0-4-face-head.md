# Asset Mania v0.4 — implementation plan

Design: [2026-08-21-asset-mania-v0-4-face-head-design.md](../specs/2026-08-21-asset-mania-v0-4-face-head-design.md)
Status: complete
Branch: `codex/v0-2-blender-pipeline`

Written after the fact, which is the wrong order and worth recording as such: the repository's
convention is design → plan → TDD, and this version went design → implementation. The design
document was frozen first and the tests were written before each gate was wired, so the
substance held, but the plan below is a record rather than a forecast.

## Task 1 — Prove the bypass exists

Before writing a gate, demonstrate the hole on the frozen v0.3 code.

```
build_reconstruction_plan(asset_kind="face_head", subject="non_person",
                          rights_receipt_sha256=None)
```

**Result: sealed successfully.** A face reconstruction with no rights receipt. Confirmed rather
than assumed, because a gate written against an imagined hole tends to guard the wrong thing.

Status: done. The exact call is now `test_face_head_with_non_person_is_refused`.

## Task 2 — Coherence gate

- `SUBJECT_KIND_INCOHERENT` added to `DiagnosticCode` and to the v2 manifest enum. A code a run
  can emit but a manifest cannot carry is a code nobody sees.
- `FACE_CAPABLE_SUBJECTS = ["real_person", "synthetic_person"]`.
- `_require_coherent_kind_and_subject`, shared. The hole was in **both** builders that take
  `asset_kind` and `subject`; fixing only `build_reconstruction_plan` would have left
  `build_workflow_plan` open while looking like a fix in the diff.
- Distinct from `FACE_RIGHTS_CONFIRMATION_REQUIRED`: a missing receipt and an impossible
  declaration are different problems, and reporting the former sends the caller after a receipt
  they cannot legitimately obtain.

Status: done. 16 tests in `tests/test_face_head_gate.py`.

## Task 3 — Likeness disclosure artifact

- `likeness-disclosure-v1.schema.json`, closed, self-sealed, registered, distributed to the
  Skill and linked from `SKILL.md` (the `skill-check` gate caught the missing link).
- `build_likeness_disclosure`, with `measured_accuracy` fixed by the builder rather than
  accepted from the caller. A disclosure whose accuracy claims came from whoever wanted to
  publish the mesh would disclose nothing.
- `prohibited_claims` as a closed enum with `minItems: 3`, so a caller cannot narrow it by
  omission.

Status: done. 19 tests in `tests/test_likeness_disclosure.py`.

## Task 4 — Wire it to the mesh record

The design says the disclosure is produced by the same call that describes the mesh. It was
not: the builder existed and nothing called it, which is a real gap and not a stylistic one. A
disclosure a caller has to remember to produce eventually travels apart from its mesh, leaving
the artifact this version exists to prevent.

`describe_reconstruction_output` now returns `disclosure` — sealed for `face_head`, `None` for
every other kind. Subject, receipt, plan digest, and engine come from the plan, so the
disclosure cannot disagree with the gate that permitted the run. The mesh digest is taken from
the file being described, not from a caller argument.

Status: done. 6 further tests.

## Task 5 — Publication

- README capability row moved off `Research`.
- `SKILL.md` gained a Faces and heads section: refuse, explain, never suggest `non_person` as a
  route through, never draft the clearance or the receipt.
- Two stale README claims fixed on the way — one bullet still said no engine had ever been
  cleared, downloaded, or run, which had been false for four commits, and the licence section
  described the Blender add-on as hypothetical while `blender-addon/` exists.

Status: done.

## What this version deliberately does not contain

No face detector, landmark model, or identity embedding: each needs its own licence clearance,
and each would convert a declaration into an inference, which is the direction v0.1 closed.

No face benchmark. `face_benchmark` is null and says why. Filling it from self-rendered heads
would produce a number about rendered heads.

## Verification

All seven targets CI runs pass locally under `uv sync --locked`: `check`, `test`,
`skill-check`, `schema-check`, `license-check`, `publication-check`, `release-check`.

Three of them caught defects during this version, all of them mine: lint had never been run,
the engine port's test dependencies existed only in a local venv, and a distributed schema was
unreferenced from `SKILL.md`. Recorded because "the tests pass" meant two of seven gates for
part of this work.
