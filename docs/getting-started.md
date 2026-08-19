# Getting started

## Current status

The executable `inspect` command is not available yet. Do not treat this page as an installation
or execution quickstart; Task 6 will replace this status with verified CLI output.

## Intended command contract

The planned interface is `asset-mania inspect <input>` with optional workflow, kind, output
parent, and JSON/text format selection. It accepts PNG, JPEG, WebP, and `.blend` header inputs.
Image inspection defaults to the image-to-3D planning workflow; `.blend` inspection defaults to
scene-to-image planning.

The target command creates one new child run directory under `.asset-mania/runs/` (or the chosen
output parent), never overwrites an existing run, and never changes the source asset. A future
successful local inspection reports planned workflow limitations inside the portable report;
it does not generate a 3D asset or call a provider.

Read [Run manifests](concepts/run-manifest.md) for the planned output, diagnostics, and exit
semantics, then return to the [documentation index](README.md).
