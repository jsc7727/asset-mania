# Agent behavior and approval gates

- Inspect local source assets read-only. Never mutate, move, overwrite, embed, or upload them.
- Do not silently change provider, model, revision, quality, or workflow when a capability is
  unavailable. Report the stable limitation instead.
- Before future uploads, paid APIs, paid compute, or model downloads, disclose the exact action,
  provider/model/revision, egressed files or derived data, retention/region, cost/runtime,
  output paths, overwrite behavior, and relevant privacy/license implications.
- Require fresh explicit approval for each gated run. A global confirmation flag cannot bypass a
  privacy, upload, paid, or download gate.
- Face/head processing is separately gated: external processing requires rights and consent
  confirmation. Never place face material in examples, fixtures, telemetry, galleries, or bug
  reports by default.
