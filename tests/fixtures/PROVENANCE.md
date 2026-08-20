# Fixture provenance

Asset Mania tracks no binary fixtures. Image and Blender-header fixtures are generated inside
temporary test directories at runtime and are not copied into the repository or release
artifacts.

- Image tests create tiny synthetic 8 x 6 pixel images with Pillow.
- Blender tests write only synthetic header bytes needed by the header inspector.
- `tests/fixtures/manifest-v1-success.json` is a hand-authored UTF-8 JSON contract fixture, not
  an opaque binary asset or output captured from a user run.
- `tests/fixtures/v2` holds the hand-authored normative v0.2 execution-contract examples. Every
  digest in them is either a deterministic placeholder or a SHA-256 computed over the canonical
  JSON of the example itself, so no example carries a real user digest, path, or datablock name.

## Runtime-generated Blender fixture

The composite Blender fixture and its negative variants are generated at runtime by the
GPL worker and written only into a test staging directory. No `.blend` is tracked in this
repository or uploaded as a CI artifact.

- Generator: `blender-addon/src/asset_mania_blender/fixture_factory.py` and
  `blender-addon/src/asset_mania_blender/fixture_variants.py`.
- Profile and seed: `blender-5.2.0-cpu-v1-fixture`, Cycles CPU, seed `0`, one thread,
  16 samples, 64 x 64 resolution.
- Content: an asymmetric tapered strip with eight vertices, six triangles, one
  non-overlapping UV layer, a two-bone rig, a rest frame plus a 30-degree deformed frame,
  a procedurally generated packed quadrant checker, one camera, and one area light. It
  contains no external file, script, driver, identity content, or real-person material.
- Disposition: the generator and everything it generates are dedicated CC0 for fixture
  use.
- Reproduce with `blender --background --factory-startup --disable-autoexec
  --offline-mode --python blender-addon/tests/run_e2e.py`, or through
  `scripts/run_blender_e2e.py` with a `fixture` request.

If a binary fixture is added later, list its repository path, creation method or source URL,
immutable revision, SHA-256 digest, license, and redistribution evidence here before tracking it.
Begin the inventory bullet with the exact root-relative repository path in an inline code span;
longer paths and prose mentions do not satisfy the release check.
