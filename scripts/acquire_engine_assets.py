#!/usr/bin/env python3
"""Acquire the engine's assets deliberately, and report every licence involved.

Acquisition lives here, in a script the user runs, rather than in the port. The port is
offline: it reads what is already on disk and fails if it is absent. That split is the whole
point of the clearance gate -- the licences are read before the bytes arrive, not discovered
afterwards when something has already run.

Four artefacts, from three sources, each with its own terms:

  engine code        the TripoSR checkout (cloned separately; this script only inspects it)
  model weights      stabilityai/TripoSR -- model.ckpt and config.yaml
  architecture       facebook/dino-vitb16 -- config.json, hyperparameters only; the ViT is
                     built from it and then filled from TripoSR's checkpoint, so no DINO
                     weight is used
  preprocessing      withheld. Upstream reaches for rembg, whose package licence is not the
                     licence of the u2net weights it fetches on first use. Supply a mask.

The third item is the reason this script exists. It is not in requirements.txt and not in the
README; it appears at runtime, three call frames inside a tokenizer's `configure`. A gate that
only covered what the documentation mentioned would have passed a fetch it never saw.

Every download is checked against the digest the registry reports for the revision, so a
truncated or substituted file fails here rather than surfacing as a strange mesh later.

The runtime stack is a separate, deliberate install and is never pinned by this repository --
the whole point is that the user chooses it. Measured working set on macOS arm64, Python 3.12:

    uv pip install torch torchvision omegaconf==2.3.0 einops==0.7.0 \
        transformers==4.35.0 huggingface-hub imageio
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    CMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"share","cmake"))');$(python -c 'import pybind11;print(pybind11.get_cmake_dir())')" \
    uv pip install --no-build-isolation "torchmcubes @ git+https://github.com/tatsy/torchmcubes.git"

Notes from getting that to work, none of which are in upstream's requirements.txt:

* `imageio` is imported at module scope by `tsr/utils.py` for a video-writing branch the port
  never takes, so it is required to import the engine at all.
* `xatlas==0.0.9` is in requirements.txt and fails to build under cmake 4, which dropped
  support for `cmake_minimum_required` below 3.5. It is only used for UV unwrapping during
  texture baking, so the port does not need it.
* `torchmcubes` has no wheel and needs Torch's and pybind11's cmake config directories on
  CMAKE_PREFIX_PATH; with build isolation on, the build environment cannot see the installed
  torch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

HF = "https://huggingface.co"

#: Pinned revisions. A moving `main` would make every digest below meaningless.
TRIPOSR_REVISION = "5b521936b01fbe1890f6f9baed0254ab6351c04a"
DINO_REVISION = "f205d5d8e640a89a2b8ef0369670dfc37cc07fc2"


@dataclass(frozen=True)
class Asset:
    repo: str
    revision: str
    filename: str
    local_name: str
    role: str

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def api(self) -> str:
        return f"{HF}/api/models/{self.repo}/tree/{self.revision}"


ASSETS = (
    Asset("stabilityai/TripoSR", TRIPOSR_REVISION, "config.yaml", "config.yaml", "model_weights"),
    Asset("stabilityai/TripoSR", TRIPOSR_REVISION, "model.ckpt", "model.ckpt", "model_weights"),
    Asset(
        "facebook/dino-vitb16",
        DINO_REVISION,
        "config.json",
        "dino-vitb16-config.json",
        "architecture_config",
    ),
)


def _context(ca_bundle: Path | None) -> ssl.SSLContext:
    """A verifying TLS context, optionally trusting a locally exported root store.

    Networks that terminate TLS present a certificate signed by a root that ships in the
    machine's keychain but not in certifi's bundle. Pointing at the system store keeps
    verification on; the alternative that gets reached for -- disabling the check -- would
    make every digest in this file unverifiable in the way that matters.
    """
    if ca_bundle is not None:
        return ssl.create_default_context(cafile=str(ca_bundle))
    return ssl.create_default_context()


def _fetch(url: str, context: ssl.SSLContext) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "asset-mania-acquire"})
    with urllib.request.urlopen(request, context=context, timeout=900) as response:
        return response.read()


def _expected_digests(asset: Asset, context: ssl.SSLContext) -> tuple[str | None, int | None]:
    """The sha256 and size the registry reports, for files it stores in LFS."""
    entries = json.loads(_fetch(asset.api, context))
    for entry in entries:
        if entry.get("path") == asset.filename:
            lfs = entry.get("lfs") or {}
            return lfs.get("oid"), entry.get("size")
    return None, None


def acquire(destination: Path, ca_bundle: Path | None, force: bool) -> int:
    context = _context(ca_bundle)
    destination.mkdir(parents=True, exist_ok=True)
    failures = 0

    for asset in ASSETS:
        target = destination / asset.local_name
        expected, size = _expected_digests(asset, context)

        if target.is_file() and not force:
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            if expected is None or actual == expected:
                print(f"  have  {asset.local_name}  ({asset.role})")
                continue
            print(f"  stale {asset.local_name}: digest differs, refetching")

        hint = f" ({size / 1e6:.0f} MB)" if size else ""
        print(f"  fetch {asset.local_name}  from {asset.repo}@{asset.revision[:8]}{hint}")
        payload = _fetch(asset.url, context)
        actual = hashlib.sha256(payload).hexdigest()

        if expected is not None and actual != expected:
            print(f"    DIGEST MISMATCH: got {actual}, expected {expected}", file=sys.stderr)
            failures += 1
            continue

        target.write_bytes(payload)
        verified = "verified" if expected else "no registry digest published"
        print(f"    sha256 {actual}  ({verified})")

    return failures


def report_licences(engine_root: Path, destination: Path, context: ssl.SSLContext) -> None:
    """Print what each component says about itself, from the component, not from memory."""
    print("\n--- licences, read from source ---\n")

    licence = engine_root / "LICENSE"
    if licence.is_file():
        first = licence.read_text(encoding="utf-8").strip().splitlines()[0]
        head = subprocess.run(
            ["git", "-C", str(engine_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        print(f"engine_code          {first}  ({engine_root.name}@{head[:8]})")
    else:
        print(f"engine_code          NO LICENSE FILE at {engine_root}")

    for repo, role in (
        ("stabilityai/TripoSR", "model_weights"),
        ("facebook/dino-vitb16", "architecture_config"),
    ):
        card = json.loads(_fetch(f"{HF}/api/models/{repo}", context))
        declared = (card.get("cardData") or {}).get("license")
        gated = card.get("gated")
        print(f"{role:20s} {declared}  ({repo}, gated={gated})")

    print("preprocessing_model  withheld -- supply a foreground mask instead of rembg")
    print(
        "\nThese are the declarations the sources publish. Read the full texts before "
        "issuing a clearance;\nthis script reports, it does not clear anything. "
        f"`{destination}` now holds the assets."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "vendor-triposr-weights")
    parser.add_argument("--engine-root", type=Path, default=REPO_ROOT / "vendor-triposr")
    parser.add_argument(
        "--ca-bundle",
        type=Path,
        default=None,
        help="PEM of trusted roots, for networks that terminate TLS",
    )
    parser.add_argument("--force", action="store_true", help="refetch even if digests match")
    args = parser.parse_args(argv)

    bundle = args.ca_bundle
    if bundle is None:
        exported = REPO_ROOT / ".asset-mania" / "system-ca.pem"
        if exported.is_file():
            bundle = exported
    if bundle is None and os.environ.get("REQUESTS_CA_BUNDLE"):
        bundle = Path(os.environ["REQUESTS_CA_BUNDLE"])

    print(f"acquiring into {args.dest}")
    failures = acquire(args.dest, bundle, args.force)
    report_licences(args.engine_root, args.dest, _context(bundle))

    if failures:
        print(f"\n{failures} asset(s) failed verification", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
