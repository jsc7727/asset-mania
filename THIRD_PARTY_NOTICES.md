# Third-Party Notices

## Included adapted text

- `CODE_OF_CONDUCT.md` — adapted and shortened from
  [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/),
  maintained by the Contributor Covenant community. License: CC BY 4.0. Required notice:
  identify the source, link the license, and indicate modification. Status: all three appear in
  the file; no separate upstream NOTICE file is supplied with the v2.1 text.

## Separately distributed GPL component

`blender-addon/` is this repository's own code, not a vendored third party, but it is
licensed GPL-3.0-or-later rather than Apache-2.0 and is therefore inventoried here.

- `blender-addon/LICENSE` — the verbatim FSF GNU General Public License v3, 29 June 2007.
  It covers every file under `blender-addon/`, which is the only tree permitted to import
  `bpy` or `mathutils`. Required notice: keep this license text with any distribution of
  that tree and state the license change relative to the Apache-2.0 packages.
- The tree is built and published as its own archive. It is not a uv workspace member and
  never appears inside an Apache wheel or sdist;
  `scripts/check_license_boundary.py` fails the build if it does.

## Optional separately distributed component

`packages/provider-openai/` is this repository's own Apache-2.0 code, published as its own
optional wheel `asset-mania-provider-openai`. The CLI wheel has no runtime dependency on
it; it is discovered through the `asset_mania.providers` entry point, and the root
development workspace installs it only to run its fake-transport tests. It adds no
third-party runtime dependency of its own.

## Optional clearance-gated engine adapter

`packages/engine-triposr/` is this repository's own Apache-2.0 code, published as its own
optional wheel `asset-mania-engine-triposr` and discovered through the
`asset_mania.engines` entry point. The CLI wheel has no runtime dependency on it.

It bundles and downloads **nothing**: no engine code, no model weight, no preprocessing
model, and no third-party runtime dependency of its own. Execution is an injected port whose
default refuses every call. An engine becomes usable only when a user supplies an
`engine-clearance-v1` artifact recording the revision, digest, license, and download receipt
for the engine code, the weights, the preprocessing model, and every runtime dependency; see
`docs/superpowers/specs/2026-08-21-asset-mania-v0-3-generic-image-to-3d-design.md`. No such
clearance ships with this repository, and no engine has been executed.

## Optional non-commercial face research adapter

`packages/engine-dad3dheads/` is this repository's own Apache-2.0 integration code. It is a
separate optional process adapter and contains no DAD source, checkpoint, FLAME asset, runtime
environment, compatibility patch, or face output.

- External source: [PinataFarms/DAD-3DHeads](https://github.com/PinataFarms/DAD-3DHeads), pinned
  for the private experiment at `68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7`.
- Upstream license: CC BY-NC-SA 4.0. This profile is non-commercial research only and must not be
  described as OSI-open-source or commercially cleared.
- Official checkpoint URL:
  <https://media.pinatafarm.com/public/research/dad-3dheads/dad_3dheads.trcd>. The checkpoint is
  downloaded only after fresh approval into ignored local storage. Its redistribution is
  uncleared, and it never appears in an Asset Mania archive.
- DAD's FLAME-family static assets and every compatibility dependency remain in the external
  private checkout/runtime. Their terms are not replaced by this adapter's Apache license.
- The measured compatibility runtime used upstream-pinned `hydra-core==1.1.0`,
  `chumpy==0.70`, `albumentations==1.0.0`, `smplx==0.1.26`, and
  `pytorch-toolbelt==0.5.0` only from ignored local storage. None is bundled in an Asset Mania
  wheel or source archive.

## Optional local face-geometry research adapters

`packages/engine-mica/` and `packages/engine-deca/` are this repository's own Apache-2.0 process
adapters. They contain no MICA, DECA, FLAME, InsightFace, face-detector, model-weight, runtime, or
face-output bytes.

- External MICA source and weights remain under the upstream Max Planck non-commercial scientific
  research license. They are user-supplied, locally hashed, network-denied during inference, and
  never redistributed by Asset Mania.
- External DECA source and weights remain under the upstream Max Planck non-commercial scientific
  research license. They are user-supplied, locally hashed, network-denied during inference, and
  never redistributed by Asset Mania.
- FLAME2020 and detector assets are separately licensed user-supplied dependencies. Asset Mania
  never accepts account credentials, downloads those assets implicitly, or places them in a wheel.
- Identity features, aligned crops, landmarks, parameter vectors, texture maps, and real-person
  geometry are private transient or ignored-run data and are forbidden in public distributions.

The adapters expose only a closed numeric geometry protocol. Their Apache license does not replace
or broaden any external model, source, dataset, or asset license.

## External tools used but never redistributed

Asset Mania invokes these tools as separate processes and bundles neither their binaries
nor their sources. `tools/` records the pinned acquisition metadata each one needs.

| Component | Pinned target | License | Notice and redistribution status |
| --- | --- | --- | --- |
| Blender | `5.2.0 LTS` — see `tools/blender-5.2.0.json` | GPL-2.0-or-later (see [blender.org/about/license](https://www.blender.org/about/license/)) | Invoked as a subprocess from a user-supplied install; no binary, library, or Python module is redistributed. |
| Gitleaks | `8.30.1` — see `tools/gitleaks.json` | MIT | Invoked as a subprocess for secret scanning; not redistributed. |
| Khronos glTF-Validator | see `tools/gltf-validator.json` | Apache-2.0 (see [KhronosGroup/glTF-Validator](https://github.com/KhronosGroup/glTF-Validator)) | Invoked as a subprocess for export validation; not redistributed. Absent until Task 9 pins a verified release. |

## Runtime dependency

| Component | Locked version and source | License | Notice and redistribution status |
| --- | --- | --- | --- |
| Pillow | [12.3.0](https://github.com/python-pillow/Pillow) | MIT-CMU | Installed separately; not bundled in Asset Mania wheels or sdists. Preserve its upstream license if redistributed. |

## Locked development and build tooling

These packages are resolved by `uv.lock`, except `uv-build`, whose bounded build requirement is
declared in both package metadata files. Asset Mania source archives do not bundle their source
or binary distributions. If a distributor vendors them, it must preserve the applicable
upstream license and any notices shipped by that distribution.

| Component | Locked version and source | License | Notice status |
| --- | --- | --- | --- |
| attrs | [26.1.0](https://github.com/python-attrs/attrs) | MIT | Not bundled; preserve upstream license if redistributed. |
| colorama | [0.4.6](https://github.com/tartley/colorama) | BSD-3-Clause | Conditional lock entry; not bundled. |
| coverage | [7.15.4](https://github.com/coveragepy/coveragepy) | Apache-2.0 | Not bundled; preserve upstream license/notices if redistributed. |
| iniconfig | [2.3.0](https://github.com/pytest-dev/iniconfig) | MIT | Not bundled. |
| jsonschema | [4.26.0](https://github.com/python-jsonschema/jsonschema) | MIT | Not bundled. |
| jsonschema-specifications | [2025.9.1](https://github.com/python-jsonschema/jsonschema-specifications) | MIT | Not bundled. |
| packaging | [26.3](https://github.com/pypa/packaging) | Apache-2.0 OR BSD-2-Clause | Not bundled; preserve selected license/notices if redistributed. |
| pluggy | [1.6.0](https://github.com/pytest-dev/pluggy) | MIT | Not bundled. |
| Pygments | [2.21.0](https://github.com/pygments/pygments) | BSD-2-Clause | Not bundled. |
| pytest | [9.1.1](https://github.com/pytest-dev/pytest) | MIT | Not bundled. |
| pytest-cov | [7.1.0](https://github.com/pytest-dev/pytest-cov) | MIT | Not bundled. |
| referencing | [0.37.0](https://github.com/python-jsonschema/referencing) | MIT | Not bundled. |
| rpds-py | [2026.6.3](https://github.com/crate-py/rpds) | MIT | Not bundled. |
| Ruff | [0.16.3](https://github.com/astral-sh/ruff) | MIT | Not bundled. |
| tomli | [2.4.1](https://github.com/hukkin/tomli) | MIT | Conditional lock entry; not bundled. |
| typing-extensions | [4.16.0](https://github.com/python/typing_extensions) | PSF-2.0 | Not bundled. |
| uv-build | [`>=0.12.1,<0.13`](https://github.com/astral-sh/uv) | MIT OR Apache-2.0 | Isolated build dependency; not bundled. |

## Inventory policy

Before adding another third-party package, asset, icon, texture, HDRI, fixture, or model reference,
record its source, version or immutable revision, license, required notice, and redistribution
evidence here. Model terms are recorded separately from software licenses. Every tracked binary
fixture must also have an entry in `tests/fixtures/PROVENANCE.md`.
Begin each file inventory bullet with its exact root-relative repository path in an inline code
span; longer paths and prose mentions do not satisfy the release check.
