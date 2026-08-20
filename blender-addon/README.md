# Asset Mania Blender worker

This tree is licensed **GPL-3.0-or-later**, not Apache-2.0. It is the only place in the
repository that may import `bpy` or `mathutils`.

## Why it is separate

Every other Asset Mania package is Apache-2.0. Blender's Python API is GPL, so any module
that imports it inherits GPL obligations. Keeping the worker in its own tree, its own
distribution, and its own process means:

- no Apache wheel or sdist contains a GPL file;
- this tree never imports an `asset_mania_*` Apache package;
- the two sides communicate only through closed JSON and relative files inside a private
  staging directory, across a process boundary.

`scripts/check_license_boundary.py` fails the build if any of those three properties break.

## How it is invoked

The Apache client in `packages/blender-client` launches Blender with a fixed argument
vector and an empty environment, then hands this worker one private request file and one
private response path. The worker never receives the source path or basename on the
command line and never writes outside the staging root it is given.

## Packaging

This tree is not a uv workspace member. It is built and published as its own archive with
this LICENSE file included. It is not installed into the Apache development environment.
