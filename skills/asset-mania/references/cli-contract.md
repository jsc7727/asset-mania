# CLI contract

The skill launcher delegates to the tested CLI; it contains no inspection or generation logic.

## Invocation

```text
python3 scripts/inspect.py inspect <input>
    [--workflow image-to-3d|scene-to-image]
    [--kind object|character|face-head]
    [--out <runs-parent>]
    [--format json|text]
```

Images default to `image-to-3d` and `object`. Blender files default to `scene-to-image`; `--kind`
is invalid for that workflow. The launcher uses `asset-mania` from `PATH`. Inside an Asset Mania
repository it falls back to `uv run --package asset-mania-cli asset-mania`. Otherwise it exits 127
with installation guidance. It executes an argument list without a shell and passes only a small
allowlist of non-secret locale, temporary-directory, and `PATH` variables.

## Outputs and streams

- The CLI creates one timestamped child beneath `--out`, or beneath `.asset-mania/runs` by default.
- `manifest.json` and `report.json` are always canonical JSON; `logs/` is reserved for run logs.
- Standard output contains the report in the selected format.
- Standard error contains a stable primary diagnostic for completed failures and storage errors.
- Portable output uses `input-1` labels and never includes the source basename or absolute path.

| Exit | Meaning |
| --- | --- |
| `0` | Inspection completed; future generation is still `WORKFLOW_NOT_IMPLEMENTED`. |
| `2` | Invalid command usage; no run directory is created. |
| `3` | Input inspection completed with a sanitized input failure. |
| `4` | Inspection completed with a sanitized internal failure. |
| `73` | Run output could not be persisted. |
| `127` | The skill launcher could not locate the CLI or repository fallback. |

The `WORKFLOW_NOT_IMPLEMENTED` diagnostic is expected in a successful v0.1 preflight. The
`BLENDER_NOT_FOUND` warning limits future scene execution but does not prevent header inspection.
