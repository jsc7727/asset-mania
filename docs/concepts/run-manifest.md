# Run manifests

The planned manifest is a versioned, portable record of one inspection run. It will include an
opaque run ID, timestamp, command, tool version, labeled inputs, environment, parameters,
capabilities, artifacts, result, and warnings. Inputs are identified as `input-1`, `input-2`,
and so on—not by basename or absolute path.

Manifests and reports must omit credentials, signed URLs, raw EXIF values, private prompts,
identity embeddings, image bytes, and absolute source paths. Artifacts will record a relative
path, hash, media type, provenance (`observed`, `derived`, or `generated`), and validation state.

The intended command returns exit 0 for a completed inspection even when a requested future
workflow is unavailable in its report. Usage errors return 2; inspected invalid, unreadable, or
unsupported inputs return 3 when output storage remains available; internal failures return 4;
and storage bootstrap failures return 73 without promising a manifest. This is a design contract
until the CLI is implemented.

Return to [Getting started](../getting-started.md) or [Architecture](../architecture.md).
