# Run manifests

The v0.1 manifest is a versioned, portable record of one inspection run. It includes an opaque
run ID, timestamp, command, tool version, labeled inputs, environment, parameters, capabilities,
artifacts, result, and warnings. Inputs are identified as `input-1`, `input-2`, and so on—not by
basename or absolute path.

Manifests and reports must omit credentials, signed URLs, raw EXIF values, private prompts,
identity embeddings, image bytes, and absolute source paths. Future artifact entries use a
relative path, hash, media type, provenance (`observed`, `derived`, or `generated`), and
validation state.

The command returns exit 0 for a completed inspection even when a requested future workflow is
unavailable in its report. Usage errors return 2; invalid, unreadable, or unsupported inputs
return 3 when output storage remains available; internal failures return 4; and storage
bootstrap failures return 73 without promising a manifest. Exit 73 emits the centrally owned
`OUTPUT_STORAGE_UNAVAILABLE` code only on standard error; that code is never written to a
manifest because usable run storage does not exist.

Return to [Getting started](../getting-started.md) or [Architecture](../architecture.md).
