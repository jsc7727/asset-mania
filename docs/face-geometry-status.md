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

## Implemented corrections

- DAD's predicted global pose is neutralized while posed source projections and camera vertices
  are preserved. Neutral FLAME coordinates preserve X/Y, reverse Z, and reverse triangle winding.
- DECA comparison geometry is aligned into MICA's metric frame before export validation.
- Detail feathers inward within the sealed face mask and never displaces outside vertices.
- New plans bind worker/Python bytes and FLAME/checkpoint/detector digests. Authorization and
  parent records are checked before subsequent stages.
- Workers use exact environment allowlists and validate Torch and detector CUDA availability
  before reading the portrait.
- DECA uses the sealed SCRFD detector and the documented DECA bbox crop transform in memory,
  including its vertical offset, with projections mapped back into source coordinates.
- The camera starts at the declared front; lights rotate with it. All comparison rows share
  materials, camera schedule, resolution, and lighting.

## Latest real-model result

The corrected September 5 local run completed MICA and DECA inference on the authorized source.
Fusion rejected the predicted facial displacement because it exceeded the approved maximum.
The limit was not changed, and no fused GLB was exported from that attempt.

Component meshes and a neutral DAD reference can be rendered as **failed-run diagnostics**. They
are not accepted avatars or evidence of likeness superiority. Head assembly, hair, and texture
work remain outside this failed experiment's acceptance boundary.

## New-plan requirements

`geometry-plan` requires explicit `--mica-python`, `--mica-plugin`, `--deca-python`,
`--deca-plugin`, `--mica-detector-sha256`, and `--deca-detector-sha256`, in addition to source,
topology, FLAME, revision, and checkpoint digests. Older plans without these bindings must be
replaced by a new create-only plan; old evidence stays unchanged.

Both workers require exactly `SOURCE_ROOT`, `ISOLATED_HOME`, `CHECKPOINT_PATH`, `FLAME_PATH`,
`FLAME_SHA256`, `DETECTOR_PATH`, and `DETECTOR_SHA256` settings under their respective
`ASSET_MANIA_MICA_` / `ASSET_MANIA_DECA_` prefixes. In standing-consent mode, supply the same
`--standing-consent` record to both MICA and DECA stages.

Socket and HTTP guards run inside Python; they are not an OS-level sandbox for hostile native
code. External runtimes must be trusted, pinned local installations. No new remote-service,
download, or publication authorization is implied by standing consent.

## Verification interpretation

The September 5 pre-fix Windows run reported 1,379 passed, 83 failed, and 115 skipped tests.
Linux CI isolated three real import-boundary test failures among those failures; the remaining
Windows-specific cases include POSIX executable fixtures, permissions, symlinks, and atomic rename.
Do not describe the Windows suite as passing.

The Blender CI workflow can report success while actual Blender suites are unavailable when its
tool inventory has no verified archive. Check the step results, not just the workflow badge.
Private real-model rendering and synthetic CI evidence must be reported separately.
