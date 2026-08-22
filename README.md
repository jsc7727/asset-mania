# Asset Mania

> **Pre-alpha.** Asset Mania performs a real local Blender round trip: inspect, preflight,
> condition, ingest an aligned view, reproject and bake, and export validated BLEND, GLB, and
> FBX. It does not generate images itself; the optional GPT Image adapter is
> contract-verified against a fake transport and has never made a live call.

Asset Mania is an open-source Agent Skill and CLI project for two future, connected workflows:
turning an image into an editable 3D asset, and using a Blender scene to guide image generation.
Its durable foundation is a portable manifest, deterministic report, and explicit approval
boundary—not a claim that generation already works.

## Verified capability

Each row states what has actually been verified, and by what. Nothing here is a plan.

| Capability | State | Evidence |
| --- | --- | --- |
| Offline image and `.blend` inspection | Available | v1 contract tests, byte-identical source checks |
| Versioned manifest and report | Available | closed schemas, canonical digest tests |
| Scene preflight | Available | real Blender 5.2.0 E2E, 15 negative and malicious variants |
| Conditioning bundle | Available | real Cycles render; matrices cross-checked against Blender's own projection to 1.5e-06 px |
| Aligned view ingest | Available | decode, normalize, and reject tests; no implicit resize or rotation |
| Reprojection and bake | Available | coverage 40.5% against a UV island area of 41.4% on the fixture |
| BLEND / GLB / FBX export | Available | container checks plus fresh-process reimport in a separate Blender |
| GPT Image 2 adapter | **experimental, contract-verified** | fake transport with sockets denied; **no live call has ever been made** |
| GPT Image 2 API turntable adapter | **experimental, fake-transport E2E verified** | approvals, no-retry ordering, audit, and provenance verified; the direct API adapter has made no live call |
| Built-in OAuth turntable experiment | **experimental, private live run completed** | eight views passed structural audit through the Codex built-in image tool; the tool disclosed neither model snapshot nor cost, and identity consistency remains unmeasured |
| Multi-view TripoSR voxel fusion | **experimental, visual review failed** | a private generated-face viewset produced a single watertight positive-volume GLB, but consensus removed recognizable facial detail |
| Face-anchor visual hull | **experimental, private visual review failed** | one closed hybrid passed silhouette and topology gates, but voxel resurfacing still removed the recognizable face |
| DAD-3DHeads face plugin | **experimental, fake-plugin E2E verified; live quality unverified** | closed local process protocol, create-only OBJ/GLB conversion, redaction, no-fallback behavior, and Blender comparison orchestration use synthetic fixtures only; the external model is CC BY-NC-SA 4.0 non-commercial research software and is not bundled |
| Khronos glTF Validator | Not run | no release pinned or verified; see `tools/gltf-validator.json` |
| Generic image to 3D | Runs, unbundled | measured below; **clearance is user-issued and unissued here**, and no wheel ships an engine or a weight |
| Face/head reconstruction | Gated, unmeasured | `face_head` + `non_person` now refused (it sealed with no receipt before); every mesh carries a `likeness-disclosure-v1`; **no face accuracy has been measured** |
| Asset Mania Cloud | Later | — |

The reconstruction row above is backed by a run, so here is the run. A synthetic 512×512 RGBA
input, TripoSR at `107cefdc` with `stabilityai/TripoSR` weights at `5b521936` (sha256
`429e2c6b…`, checked against the digest the registry publishes), CPU, marching cubes at 128:

| Measured | mc 128 | mc 256 (default) |
| --- | --- | --- |
| Wall clock | 13.6 s | 23.4 s |
| Triangles / vertices | 83,222 / 41,613 | 338,870 / 169,437 |
| Extent | 0.980 × 0.998 × 1.013 | 0.972 × 0.998 × 1.009 |
| Signed volume | +0.27314 | +0.26401 |
| Surface | closed, watertight | closed, watertight |
| Winding | consistent, outward | consistent, outward |
| Vertex colour | present | present |

Getting to `closed` took a repair pass, and the shape of that pass is worth stating because two
plausible versions of it are wrong. Marching cubes on a density field leaves single absent
triangles where the isosurface is ambiguous — 134 of them on the first run, enough to report
the whole surface as open. Putting those back is repair: the signed volume does not move.

Capping a *large* opening is not repair. It makes the mesh watertight while inventing surface
across a region the model never reconstructed, which then reads as a solid object. So the guard
has to tell the two apart, and the two obvious measures cannot:

- **Vertex count** fails because a cube missing an entire face has a four-vertex boundary loop,
  exactly like a single absent triangle.
- **Volume drift** fails because a planar cap across a flat opening adds no volume at all.

What separates them is spatial span. Noise holes reached 3.2% of the bounding-box diagonal
against a grid cell of 0.46%; a missing cube face spans 82%. The threshold sits at 10% — three
times above the worst real hole, eight times below fabrication — and anything wider leaves the
mesh `open`, reported as such rather than rounded up.

Four licences are involved, each read from its source rather than from memory: the engine code
(MIT), the weights (MIT), and a ViT architecture config that TripoSR fetches at runtime from
`facebook/dino-vitb16` (Apache-2.0) — that last one appears in no requirements file and only
surfaced by running the thing. A background remover is deliberately absent: upstream reaches
for rembg, whose package licence is not the licence of the weights it downloads, so a
foreground mask is required instead. Run `scripts/acquire_engine_assets.py` to fetch and
digest-verify the assets and print every licence declaration.

Two limits are worth stating plainly:

- The macOS Seatbelt profile enforces read-only sources, staging-only writes, and denied
  network, all verified by running `sandbox-exec`. It cannot confine *reads*, and Blender's
  Metal-backed compositor cannot run under it at all, so an isolated conditioning run is the
  Linux bubblewrap backend's job.
- The pinned Blender and glTF Validator inventories in `tools/` record only what was
  verified locally. No official archive URL or digest is asserted, because none has been
  fetched and verified.
- Generic image-to-3D runs, but not for a user of this repository. The engine is not bundled,
  no clearance is issued here, and the adapter refuses until a user issues one. The local
  Blender stages are a different thing entirely: they need a 3D model as *input* and generate no
  geometry at all.
- No face accuracy has been measured. The 6.0% figure above is a rendered geometric subject
  against its own source mesh, and it does not transfer to a human face. `face_head` meshes
  carry a `likeness-disclosure-v1` saying so.

The `inspect` command accepts a local PNG, JPEG, WebP, or `.blend` header input, records only
portable allowlisted metadata and hashes, and writes a new run directory. It does not alter or
embed the input, upload it, invoke Blender, download a model, or perform generation.

## Verified quickstart

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run `make setup`. The commands
below create a synthetic 8 x 6 PNG outside the repository and inspect it with the real CLI:

```console
$ uv run python -c 'from PIL import Image; Image.new("RGBA", (8, 6), (20, 40, 60, 80)).save("/tmp/asset-mania-example.png")'
$ uv run asset-mania inspect /tmp/asset-mania-example.png --out /tmp/asset-mania-runs --format text
Asset Mania inspection: succeeded
Workflow: image-to-3d
Kind: object
Input: input-1
Media type: image/png
SHA-256: 04efed53ff617ce02ad91b51461b5f0130eec0c36c1fef9f10f79486c957f81a
Diagnostics: WORKFLOW_NOT_IMPLEMENTED
Warnings: none
```

`WORKFLOW_NOT_IMPLEMENTED` is expected: inspection succeeded, while image-to-3D generation
remains planned. Each invocation creates a new child directory under the selected output parent
with canonical `manifest.json` and `report.json` files and an empty `logs/` directory. See
[Getting started](docs/getting-started.md) for flags and [Run manifests](docs/concepts/run-manifest.md)
for the stream and exit-code contract.

The maintainer-only turntable runners are intentionally outside the public CLI while they remain
research-grade. The direct API path plans one observed front view plus seven generated yaws and
retains its approval-bound provider contract. A separate built-in OAuth experiment created eight
private views but exposed no model snapshot or cost, so it is not recorded as a GPT Image 2 API
run. Generated side and rear views are model inferences, not observations, and identity
consistency remains unmeasured.

The first live private reconstruction geometrically completed, but eight independently
hallucinated TripoSR meshes lost facial detail when voxel-voted. The replacement research profile,
`face-anchor-visual-hull-v1`, keeps a CUDA TripoSR mesh from the observed front and uses seven-of-
eight silhouette support only for the side and rear envelope. Its deterministic and optional-
runtime tests pass, and a private run produced one closed 151,564-triangle GLB with minimum/mean
silhouette IoU 0.861/0.944 and 96.1% front-volume retention. Blender review still failed: the
recognizable front texture and surface were lost during voxel resurfacing. GPU speed does not fix
that model/profile limitation; a face-specific DECA/FLAME-family experiment is the next research
step and requires a separate model-clearance decision.

The next experiment now has a separate DAD-3DHeads process adapter and deterministic fake-plugin
E2E. It does not vendor the upstream source or checkpoint, and it does not make DAD part of the
Apache distribution. The external dependency is CC BY-NC-SA 4.0 and restricted to non-commercial
research. A live face run has not yet been used to change the capability claim; identity
consistency remains unmeasured.

## Project guide

- [Documentation index](docs/README.md)
- [Architecture](docs/architecture.md)
- [Security and privacy](docs/security-and-privacy.md)
- [Research](docs/research.md) and [roadmap](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md), [security reporting](SECURITY.md), and
  [third-party notices](THIRD_PARTY_NOTICES.md)

## License

The CLI/core, Skill instructions, schemas, and documentation are licensed under
[Apache-2.0](LICENSE), except the adapted Contributor Covenant text identified in
[third-party notices](THIRD_PARTY_NOTICES.md). The Blender add-on under `blender-addon/` is a separate
GPL-3.0-or-later package, because a module that imports `bpy` is a derived work of Blender;
`scripts/check_license_boundary.py` enforces the split in both directions. Inputs and generated outputs are not relicensed by this
repository.
