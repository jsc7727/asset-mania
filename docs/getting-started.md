# Getting started

## Current status

The pre-alpha `inspect` command works for deterministic, offline image and `.blend` header
inspection. It writes a portable run manifest and report, but does not generate an image or 3D
asset. `WORKFLOW_NOT_IMPLEMENTED` is therefore the expected diagnostic after a successful v0.1
inspection.

## Set up the workspace

Asset Mania supports Python 3.11 through 3.13. Install
[uv](https://docs.astral.sh/uv/), clone the repository, and sync the locked workspace:

```console
$ make setup
```

## Run the verified example

This example creates a tiny synthetic image in `/tmp`, outside the repository. It is the exact
input used to capture the privacy-safe CLI output below.

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

The report uses the fixed label `input-1`; it does not reveal the input basename or absolute
path. Do not publish your source asset or run directory when reporting an issue. Reproduce a
problem with a synthetic input and share only redacted diagnostics.

## Command contract

The interface is `asset-mania inspect <input>` with optional workflow, kind, output-parent, and
JSON/text format selection. It accepts PNG, JPEG, WebP, and `.blend` header inputs. Image
inspection defaults to the `image-to-3d` planning workflow; `.blend` inspection defaults to
`scene-to-image` planning.

```text
asset-mania inspect <input>
  [--workflow image-to-3d|scene-to-image]
  [--kind object|character|face-head]
  [--out <output-parent>]
  [--format json|text]
```

The command creates one new child run directory under `.asset-mania/runs/` or the chosen output
parent, never overwrites an existing run, and never changes the source asset. JSON is the default
stdout format; `--format text` selects the report shown above. A completed inspection exits 0
even when the requested future workflow is unavailable and records that limitation with the
stable diagnostic.

Read [Run manifests](concepts/run-manifest.md) for output, diagnostics, and exit semantics, then
return to the [documentation index](README.md).

## Maintainer turntable research path

`scripts/run_turntable_multiview_e2e.py` exposes `plan`, `generate`, `reconstruct`, and `verify`.
`plan` is offline. `generate` uploads the approved cutout seven times and therefore requires fresh
plan-bound face-rights, external-egress, and paid-compute receipts for a real person. Calls execute
once in yaw order and never retry. `reconstruct` and `verify` are local and offline.

All source images, prompts, receipts, generated views, masks, per-view meshes, fused meshes, and
previews belong below `.asset-mania/` and must never be committed. The path is currently
fake-transport and synthetic-fusion verified, not live-provider verified.
