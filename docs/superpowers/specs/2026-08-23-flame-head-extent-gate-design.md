# FLAME Full-Head Extent Gate Design

Status: approved in chat on 2026-08-23

## Decision

The local face-geometry v1 gate accepts a validated authoritative or exported full FLAME head whose
longest metric extent is between `0.15` and `0.32` metres inclusive. The previous `0.30` upper
bound rejected the sealed MICA backend's otherwise valid `0.309499189` metre full-head result.

This is a unit-sanity gate, not a likeness threshold. It applies to the authoritative MICA head and
the exported/fused full 5,023-position, 9,976-triangle FLAME heads. DECA's raw coarse head is a
pre-alignment coordinate source: before similarity fit it must have finite positions, positive
extent on every axis, exact topology, and explicit metre units, but it does not receive the
absolute `0.15..0.32` interval. The existing similarity fit maps it into MICA metric space before
the displacement and exported-geometry gates. This does not change displacement limits, topology,
winding, privacy, or manual clay criteria.

## Fail-closed behavior

- Accept `0.15 <= longest_extent_metres <= 0.32` for authoritative/exported heads.
- Reject values below `0.15` or above `0.32`.
- Reject raw DECA pre-alignment geometry with a non-positive axis extent or a non-finite value;
  defer its absolute interval until similarity fit.
- Never clamp, rescale, normalize, repair, or silently reinterpret an out-of-range result.
- Record both bounds in every new private geometry plan so the plan digest changes from the
  previous `0.30` workflow.
- Record `deca_extent_validation=positive-finite-prealignment-then-similarity-fit` in every new
  private geometry plan.
- Existing plans and failed attempts remain immutable and are never resumed under the new bound.

## Evidence boundary

The only live evidence used to revise the unit-sanity ceiling is the sealed local MICA backend
shape/count/unit preflight: 5,023 positions, 9,976 triangles, explicit metres, longest extent
`0.309499189`. No mesh was persisted from that diagnostic.
