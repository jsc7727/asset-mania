# Local face geometry status

This is an experimental, local research workflow. The public repository contains original
adapters and synthetic tests; external model code, weights, consent records, photographs, meshes,
and renders are not distributed.

## Implemented

- MICA neutral FLAME geometry and DECA detail workers in isolated Python environments.
- Numeric geometry validation, bounded displacement fusion, and neutral GLB export.
- Source-specific standing consent for repeated local processing of identical source bytes.
- Create-only stage records, artifact digests, and Blender comparison orchestration.

## Acceptance correction

The earlier private comparison's visual `passed` verdict is not accepted. Its DAD baseline retained
predicted global head rotation, while its DECA row used pre-alignment geometry. The rows therefore
did not provide a fair comparison of facial proportions. Successful mesh generation and rendering
do not establish likeness or reconstruction accuracy.

The original records remain intact; a separate private correction supersedes the acceptance claim.
Do not begin downstream head assembly, hair fitting, or texture work on the strength of that verdict.

## Work required before acceptance

- Neutralize DAD's predicted global pose while preserving posed source projections.
- Compare DECA in MICA's metric frame and enforce export geometry gates.
- Keep detail strictly inside the sealed face mask with inward boundary feathering.
- Verify runtime assets, authorization, and parent-stage lineage before processing.
- Verify local worker environment and pre-source CUDA gates.
- Crop DECA's input in memory using a sealed detector and map projections back to the source.
- Run a new immutable comparison and judge front and both three-quarter views under matching
  materials, framing, cameras, and lighting. A new manual result may be failed or unverified.

## Verification interpretation

The September 5 pre-fix Windows run reported 1,379 passed, 83 failed, and 115 skipped tests.
Linux CI isolated three real import-boundary test failures among those failures; the remaining
Windows-specific cases include POSIX executable fixtures, permissions, symlinks, and atomic rename.
Do not describe the Windows suite as passing.

The Blender CI workflow can report success while actual Blender suites are unavailable when its
tool inventory has no verified archive. Check the step results, not just the workflow badge.
Private real-model rendering and synthetic CI evidence must be reported separately.
