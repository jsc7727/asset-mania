# Asset Mania v0.4 — Face and head: gating, provenance, and what may be claimed

Status: design, frozen before implementation
Date: 2026-08-21
Scope: the `face_head` asset kind

## What this version is

v0.4 is not a capability. The engine that arrived in v0.3 already reconstructs a head from a
single image — it did so on a rendered Suzanne with a symmetric mean surface error of 6.0% of
the subject's longest axis. v0.4 decides **when that is allowed to happen to a person's face,
what is recorded when it does, and what may be said about the result afterwards.**

The roadmap phrases it as "a separately gated workflow with rights confirmation, provenance,
and conservative likeness claims". Each of the three is a distinct failure this design has to
close.

## The hole this starts from

The v0.3 gate ties the rights receipt to the *subject* declaration:

```python
if subject == "real_person":
    ...require_rights_receipt(...)
```

`asset_kind` is not consulted. Measured on the frozen v0.3 code:

```
build_reconstruction_plan(asset_kind="face_head", subject="non_person",
                         rights_receipt_sha256=None)
  -> sealed successfully
```

A face reconstruction seals with no rights receipt as long as the caller declares the subject
as `non_person`. The gate is not weak, it is bypassed — and the bypass is a single field on the
call that already has to be filled in.

Two facts make this worth fixing rather than documenting:

1. Subject is a *declaration*, deliberately. v0.1 established that a subject is never inferred
   from pixels, because a classifier deciding whose face is real would be a worse system than
   one that asks. That decision is right and it means the declaration is the only thing
   standing between an engine and a stranger's face.
2. `face_head` and `non_person` are not merely an unusual pairing. A head belongs to a person,
   real or synthetic. The combination has no legitimate reading, which is exactly why it is the
   shape a bypass takes.

## Gate 1 — coherence between kind and subject

`face_head` requires a subject that can own a face:

| `asset_kind` | `subject` | Result |
| --- | --- | --- |
| `face_head` | `real_person` | rights receipt required, as v0.3 already does |
| `face_head` | `synthetic_person` | allowed, recorded, no receipt |
| `face_head` | `non_person` | **refused** — incoherent, and the bypass shape |
| `face_head` | `unknown` | refused, as v0.1 already refuses `unknown` everywhere |
| `object` / `character` | any declared | unchanged |

The refusal is a new diagnostic, not a reuse of `FACE_RIGHTS_CONFIRMATION_REQUIRED`: the caller
has not failed to supply a receipt, they have supplied a declaration that cannot be true. Those
are different problems and telling them apart is what makes the message actionable.

`synthetic_person` is allowed without a receipt and still recorded. A synthetic face has no
subject to obtain rights from; pretending otherwise would push callers toward mislabelling, and
a gate that punishes the honest declaration teaches people to lie to it.

## Gate 2 — provenance

A mesh that leaves this pipeline should carry where it came from, because a head mesh with no
provenance is indistinguishable from any other head mesh once it is a file on a disk.

New artifact, `likeness-disclosure-v1`, sealed like every other artifact in this project
(closed schema, canonical digest, self-sealed):

- the source image digest and the plan digest, so the mesh is traceable to one exact input
- `asset_kind` and `subject` as declared
- the rights receipt digest when one was consumed, `null` for `synthetic_person`
- the engine and profile, so the thing that inferred the geometry is named
- `likeness_basis`: the number of views the geometry was inferred from
- `measured_accuracy`: what has actually been measured, which see below

The disclosure is produced by the same call that describes the mesh, so a mesh record and its
disclosure cannot drift apart.

## Gate 3 — what may be claimed

This is the part most likely to be got wrong in a direction that flatters the software, so the
rule is stated as a measurement rather than a caution.

**No face accuracy has been measured.** The 6.0% figure comes from a subdivided Suzanne against
its own source geometry. It says something about the engine on a smooth, symmetric, textureless
subject photographed under controlled light. It says nothing about a human face, and a project
that quoted it next to the word "likeness" would be transferring a number across a gap it does
not span.

So `measured_accuracy` records:

- `ground_truth_available: false` for any real subject — there is no reference mesh for a
  photograph, which is the whole reason single-image reconstruction is being used
- `face_benchmark: null`, with a note naming what would have to exist to fill it: a set of
  faces with reference scans, which this repository does not have and cannot synthesise
- the non-face figure, labelled as such, so the honest number is present and cannot be mistaken
  for a face number

And the claims that may not be made, enforced by the publication gate rather than left to
prose: no output of this workflow may be described as an identification-grade likeness, a
biometric record, or a match to a specific person. Not because those claims are impolite, but
because nothing here has measured whether they would be true, and a single view of a face
underdetermines the geometry behind it.

## Determinism class

Unchanged from v0.3: `repeat-run equivalent` for the reconstruction, `byte-exact` for the
disclosure artifact, which is pure canonical JSON over digests.

## What v0.4 does not do

- No face detector, landmark model, or identity embedding. Each would need its own licence
  clearance and each would move a declaration into an inference, which is the direction v0.1
  deliberately closed.
- No face benchmark. Stating that the number is absent is honest; producing one from
  self-rendered heads would be a number about rendered heads.
- No relaxation of the engine clearance. `face_head` sits behind the v0.3 clearance gate too;
  the two gates are independent and both apply.
