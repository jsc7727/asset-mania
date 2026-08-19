# Safety and licenses

## v0.1 stopping rule

Only local inspection exists. Generation, uploads, network providers, model or installer
downloads, Blender execution, rendering, GPU jobs, and paid compute are unavailable. Do not ask
for approval for an operation the CLI cannot perform. Report the limitation and stop. Never
silently substitute a provider, model, revision, quality, or workflow.

Inspection may create a new run directory. It must not mutate, move, overwrite, embed, or upload
the source. Portable output excludes paths, basenames, credentials, raw sensitive metadata, image
bytes, and identity embeddings.

## Future external actions

Before a future version requests approval, it must disclose the exact action and provider/model
revision; files or derived data leaving the machine; known retention and region; expected price,
credits, download size, runtime, and outputs; overwrite behavior; and privacy/license implications.
Each run requires fresh explicit approval. Costly retries require renewed approval.

External face/head processing additionally requires confirmation of rights and consent. Face data
must not enter fixtures, examples, telemetry, training, galleries, or bug reports by default.
Hidden geometry must be labeled generated or prior-filled, never observed fact.

## License boundary

- The CLI/core, skill, schemas, and documentation are Apache-2.0.
- Future `bpy` code belongs in a separately packaged GPL-3.0-or-later Blender component across a
  documented CLI/JSON/file boundary.
- The project does not redistribute model weights. Download availability does not establish that
  weights are open source or commercially usable.
- User inputs and generated outputs are not relicensed by the repository license.
- Third-party code and assets require source, immutable revision, license/terms, attribution, and
  redistribution evidence in the repository notices before distribution.
