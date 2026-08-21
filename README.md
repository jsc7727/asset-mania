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
| Khronos glTF Validator | Not run | no release pinned or verified; see `tools/gltf-validator.json` |
| Generic image to 3D | Runs, unbundled | measured below; **clearance is user-issued and unissued here**, and no wheel ships an engine or a weight |
| Face/head reconstruction | Research | not implemented; `real_person` needs a plan-bound rights receipt |
| Asset Mania Cloud | Later | — |

The reconstruction row above is backed by a run, so here is the run. A synthetic 512×512 RGBA
input, TripoSR at `107cefdc` with `stabilityai/TripoSR` weights at `5b521936` (sha256
`429e2c6b…`, checked against the digest the registry publishes), CPU, marching cubes at 128:

| Measured | Value |
| --- | --- |
| Wall clock | 15.8 s |
| Triangles / vertices | 82,948 / 41,613 |
| Extent | 0.980 × 0.998 × 1.013 |
| Winding | consistent, outward (signed volume +0.273) |
| Surface | **open, not watertight** — 548 boundary edges, Euler number −135 |
| Vertex colour | present |

The open surface is the honest headline. Marching cubes on a density field does not owe you a
closed manifold, and `manifold` reports `open` rather than rounding up to `closed`. Anything
downstream that needs watertightness needs a repair pass that does not exist yet.

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
- Generic image-to-3D does not work here. Asset Mania needs a 3D model as input; it does not
  generate geometry. The v0.3 clearance contract and gate are in place, but no inference
  engine has been cleared, downloaded, or run, and the blocker is license clearance across an
  engine's entire dependency closure rather than implementation effort.

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
[third-party notices](THIRD_PARTY_NOTICES.md). A future Blender `bpy` add-on, if published, will
be a separate GPL-3.0-or-later package. Inputs and generated outputs are not relicensed by this
repository.
