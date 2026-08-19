# Asset Mania

> **Pre-alpha — implementation in progress.** Asset Mania is not ready to install or run yet.
> The initial release is being built as an offline inspection and planning foundation; this
> repository currently has no executable CLI quickstart.

Asset Mania is an open-source Agent Skill and CLI project for two future, connected workflows:
turning an image into an editable 3D asset, and using a Blender scene to guide image generation.
Its durable foundation is a portable manifest, deterministic report, and explicit approval
boundary—not a claim that generation already works.

## v0.1 target

| Capability | v0.1 target |
| --- | --- |
| Offline image and .blend inspection | In development |
| Versioned manifest and report | In development |
| Image to 3D generation | Planned |
| Blender scene to GPT Image | Planned |
| Face/head reconstruction | Research |
| Asset Mania Cloud | Later |

The planned `inspect` contract accepts a local image or `.blend` input, records only portable
metadata and hashes, and writes a new run directory. It will not alter the input, upload it,
invoke Blender, download a model, or perform generation. Its intended flags and stream/exit
contract are documented in [Getting started](docs/getting-started.md) and
[Run manifests](docs/concepts/run-manifest.md); they are design targets, not runnable commands
at this stage.

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
