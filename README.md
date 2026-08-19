# Asset Mania

> **Pre-alpha.** Asset Mania provides a working offline inspection and planning foundation. It
> does not generate images or 3D assets yet.

Asset Mania is an open-source Agent Skill and CLI project for two future, connected workflows:
turning an image into an editable 3D asset, and using a Blender scene to guide image generation.
Its durable foundation is a portable manifest, deterministic report, and explicit approval
boundary—not a claim that generation already works.

## v0.1 target

| Capability | v0.1 target |
| --- | --- |
| Offline image and .blend inspection | Available |
| Versioned manifest and report | Available |
| Image to 3D generation | Planned |
| Blender scene to GPT Image | Planned |
| Face/head reconstruction | Research |
| Asset Mania Cloud | Later |

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
[Apache-2.0](LICENSE). A future Blender `bpy` add-on, if published, will be a separate
GPL-3.0-or-later package. Inputs and generated outputs are not relicensed by this repository.
