# Third-Party Notices

## Included adapted text

- `CODE_OF_CONDUCT.md` — adapted and shortened from
  [Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/),
  maintained by the Contributor Covenant community. License: CC BY 4.0. Required notice:
  identify the source, link the license, and indicate modification. Status: all three appear in
  the file; no separate upstream NOTICE file is supplied with the v2.1 text.

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
| pytest | [8.4.2](https://github.com/pytest-dev/pytest) | MIT | Not bundled. |
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
