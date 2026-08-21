# Asset Mania v0.3 Generic Image-to-3D Implementation Plan

**Goal:** make the engine-clearance gate real and fail-closed, then build the reconstruction
plan, input contract, and provider port that a cleared engine plugs into — without
downloading a weight, executing an engine, or claiming that image-to-3D works.

**Spec:** `docs/superpowers/specs/2026-08-21-asset-mania-v0-3-generic-image-to-3d-design.md`

**Architecture:** the clearance contract and its gate live in the existing Apache packages
(`contracts` for schemas, `pipeline` for the gate). The engine adapter is a new optional
wheel `asset-mania-engine-triposr`, discovered through an entry point, with execution
injected exactly as the provider adapter injects transport. The CLI depends on neither.

## Global Constraints

- No stage downloads or bundles an engine, weight, preprocessing model, or dataset.
- No engine is executed in this milestone. Tests use a fake engine port with subprocesses
  and sockets denied.
- This plan asserts no third-party license fact. Every such fact lives in a clearance
  artifact recorded from a verified source at acquisition time.
- v0.1 and v0.2 behaviour, schemas, and fixture bytes stay unchanged.
- A reconstruction output never enters the v0.2 bake path.
- The capability table says `Planned` for generic image-to-3D until a real cleared engine
  run is recorded. The Skill refuses the request until then.
- Conventional Commits, repository-local `jsc7727` identity.

## Task 1: Freeze the v0.3 Design and Plan

**Files:** the design and this plan.

- [x] Add both documents with no implementation change.
- [x] Assert no third-party license fact anywhere in either.
- [ ] Commit `docs: design generic image-to-3d clearance`.

## Task 2: Add the Clearance and Reconstruction Contracts

**Files:** `packages/contracts` schemas, builders, diagnostics, normative examples.

### TDD sequence

- [ ] Add failing schema tests for `engine-clearance-v1`: closed records, one entry per
      required role, name-sorted dependencies, `cleared_by: user` only, sha256 and SPDX
      patterns, and rejection of an unknown field.
- [ ] Add failing tests proving `commercial_use` of `prohibited` **and** `unknown` are both
      refused, and that only `cleared` passes.
- [ ] Add failing schema tests for `reconstruction-plan-v1`: bound clearance digest, mask
      digest **or** background-removal clearance digest but never neither, closed
      `expected_output`, `overwrite_policy: create_only`, and a self-seal.
- [ ] Add the seven new diagnostic codes and assert the v2 enum stays a superset of v1.
- [ ] Commit one normative example per shape, including a deliberately uncleared clearance.
- [ ] Distribute both schemas to the Skill references with byte parity.
- [ ] Commit `feat: define engine clearance contracts`.

## Task 3: Implement the Fail-Closed Clearance Gate

**Files:** `packages/pipeline` clearance module and tests.

### TDD sequence

- [ ] Write failing tests proving the gate refuses: a missing component role, a missing
      runtime dependency, `prohibited`, `unknown`, an absent download receipt, an expired
      clearance, a maintainer-issued clearance, and a clearance whose digest was edited.
- [ ] Write failing tests proving a spy engine port is never called on any refused path.
- [ ] Prove the gate refuses when a dependency list is empty, since "no dependencies" is
      never true for an inference engine and is the most likely way to fake a clearance.
- [ ] Implement `verify_engine_clearance` and `run_if_cleared`, mirroring the approval
      composition so ordering is testable.
- [ ] Commit `feat: gate uncleared inference engines`.

## Task 4: Implement the Reconstruction Input Contract

**Files:** `packages/pipeline` reconstruction input module and tests.

### TDD sequence

- [ ] Reuse the v0.2 view normalization for decode, metadata, orientation, and alpha rules.
- [ ] Write failing tests proving a run without a mask **and** without an audited
      background-removal clearance fails with `MASK_REQUIRED`.
- [ ] Write failing tests proving an unpinned background-removal model fails with
      `BACKGROUND_REMOVAL_UNPINNED`, including a `rembg`-shaped entry with no digest.
- [ ] Write failing tests proving a mask whose dimensions differ from the image is refused
      rather than resized.
- [ ] Prove `unknown` subject and a `real_person` subject without a plan-bound receipt both
      fail before any engine call.
- [ ] Commit `feat: contract reconstruction inputs`.

## Task 5: Add the Engine Port and the Fake Engine

**Files:** `packages/engine-triposr`, tests, notices, lock.

### TDD sequence

- [ ] Scaffold the optional wheel with an entry point. The CLI must not depend on it.
- [ ] Implement execution as an injected port whose default refuses every call.
- [ ] Write failing tests proving the adapter constructs no subprocess, socket, or model
      loader, by scanning its own source.
- [ ] Write failing output tests: a mesh with non-finite vertices, a degenerate triangle,
      zero triangles, an unexpected format, and an oversized payload are all refused.
- [ ] Record the produced mesh as `generated`, `user-content`, `upload_eligible: false`.
- [ ] Prove a reconstruction manifest is refused as a bake input.
- [ ] Commit `feat: add clearance-gated reconstruction port`.

## Task 6: Integrate CLI, Skill, Docs, and Gates

**Files:** CLI commands and tests, Skill, README, docs, publication checks.

### TDD sequence

- [ ] Add `engine clearance verify` and `image reconstruct` with kebab normalization and the
      fixed exit codes.
- [ ] Extend the Skill: refuse generic image-to-3D, and say plainly that the engine is not
      cleared and nothing was downloaded.
- [ ] Keep the README capability row at `Planned`, with the clearance gate listed as the
      blocker rather than effort.
- [ ] Extend `check_publication.py` so a `Planned` row cannot silently become `Available`
      without a recorded engine run.
- [ ] Add forward evals: an uncleared engine, a missing mask, an unpinned background
      remover, a real-person subject, and a request to skip clearance.
- [ ] Commit `docs: publish reconstruction boundary`.

## Final Acceptance Matrix

| Requirement | Evidence |
| --- | --- |
| Clearance is fail-closed | refusal tests for every incomplete closure shape |
| Nothing ran uncleared | spy engine port never called on any refused path |
| No download, no weight | source scan plus tracked-tree and archive checks |
| No license asserted | neither design nor plan states a third-party license fact |
| Mask is mandatory | `MASK_REQUIRED` and `BACKGROUND_REMOVAL_UNPINNED` tests |
| Declarations preserved | `unknown` blocked, `real_person` receipt-bound |
| Generated stays generated | transitive origin tests |
| No bake contamination | a reconstruction manifest is refused as a bake input |
| Claim discipline | capability row stays `Planned`; publication gate enforces it |

## What Comes After

v0.3 closes only the clearance and contract foundation. A working generic image-to-3D
capability additionally needs, in order: a user-acquired cleared engine, a real recorded
run, export and round-trip E2Es for reconstruction output, and public readback. Only then
may the capability table change.
