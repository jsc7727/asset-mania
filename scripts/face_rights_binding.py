#!/usr/bin/env python3
"""Print the digest a face-rights receipt must be bound to, for one specific photograph.

    .venv/bin/python scripts/face_rights_binding.py photo.png --mask mask.png \\
        --clearance clearance.json

The receipt cannot be bound to the plan digest: the plan seals the receipt digest into its own
preimage, so each would have to be computed from the other. It is bound instead to a digest over
everything that determines what would be reconstructed -- the image, the mask, the engine, the
clearance, the declared kind and subject, the expected output -- with the receipt field excluded.

That is what makes the consent specific. A receipt issued for this photograph does not authorise
a different one, because a different image produces a different digest and the gate recomputes
it rather than accepting one from the caller.

This script computes and prints. It does not issue the receipt, and it never will: a rights
receipt is the user's assertion about a real person's likeness, and software that drafted it on
their behalf would be manufacturing the consent the gate exists to require.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for package in ("contracts", "pipeline"):
    sys.path.insert(0, str(REPO_ROOT / "packages" / package / "src"))

from asset_mania_contracts import reconstruction_binding_digest
from asset_mania_pipeline import acknowledgement_text, prepare_input

DEFAULT_ENGINE = "triposr-local"
DEFAULT_PROFILE = "triposr-local-cpu-v1"
DEFAULT_OUTPUT = {"mesh_format": "glb", "textured": False, "unit_scale_meters": 1.0}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="the photograph to be reconstructed")
    parser.add_argument("--mask", type=Path, default=None, help="single-channel foreground mask")
    parser.add_argument(
        "--clearance",
        type=Path,
        required=True,
        help="the engine-clearance-v1 artifact the run will use",
    )
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--engine-profile", default=DEFAULT_PROFILE)
    parser.add_argument(
        "--subject",
        default="real_person",
        choices=["real_person", "synthetic_person"],
        help="synthetic_person needs no receipt; it is accepted here only to show the digest",
    )
    args = parser.parse_args(argv)

    clearance = json.loads(args.clearance.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as staging:
        prepared = prepare_input(
            image_path=args.image, staging_root=Path(staging), mask_path=args.mask
        )

    binding = reconstruction_binding_digest(
        engine=args.engine,
        engine_profile=args.engine_profile,
        clearance_sha256=clearance["clearance_sha256"],
        source_image_sha256=prepared["image_sha256"],
        source_width=prepared["width"],
        source_height=prepared["height"],
        alpha=prepared["alpha"],
        mask_sha256=prepared["mask_sha256"],
        background_removal_clearance_sha256=None,
        asset_kind="face_head",
        subject=args.subject,
        expected_output=DEFAULT_OUTPUT,
    )

    print(
        json.dumps(
            {
                "binding_sha256": binding,
                "gate": "face_rights",
                "acknowledgement": acknowledgement_text("face_rights", binding),
                "source_image_sha256": prepared["image_sha256"],
                "source_size": [prepared["width"], prepared["height"]],
                "mask_sha256": prepared["mask_sha256"],
                "asset_kind": "face_head",
                "subject": args.subject,
                "engine": args.engine,
                "note": (
                    "Issue the receipt against binding_sha256, not against a plan digest. "
                    "This script does not issue it: a face-rights receipt is your assertion "
                    "about a real person's likeness."
                ),
            },
            indent=2,
        )
    )
    if args.subject == "real_person" and prepared["mask_sha256"] is None:
        print(
            "\nwarning: no mask. TripoSR treats every opaque pixel as the subject, so a "
            "photograph with its background attached reconstructs the background too.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
