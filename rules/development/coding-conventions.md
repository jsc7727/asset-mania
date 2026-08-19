# Coding conventions

- Use typed Python and stable string diagnostic codes for machine-readable behavior.
- Produce canonical JSON with deterministic key and diagnostic ordering.
- Use fixed input labels, never paths or basenames, in portable output and user-facing logs.
- Allowlist metadata; report sensitive metadata only as presence/diagnostic information.
- Return sanitized expected errors without tracebacks by default. Never expose credentials,
  tokens, signed URLs, raw metadata values, or face-related private data.
- Preserve the source bytes and create output directories atomically without overwriting runs.
