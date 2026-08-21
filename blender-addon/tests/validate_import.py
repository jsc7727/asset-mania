# SPDX-License-Identifier: GPL-3.0-or-later
"""Prove the GPL worker imports no Apache package, from inside Blender.

Run with:
    blender --background --factory-startup --python blender-addon/tests/validate_import.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

APACHE_PACKAGES = (
    "asset_mania",
    "asset_mania_contracts",
    "asset_mania_pipeline",
    "asset_mania_blender_client",
    "asset_mania_provider_openai",
)

FAILURES: list[str] = []


def main() -> int:
    from asset_mania_blender import (  # noqa: F401
        entrypoint,
        fixture_factory,
        fixture_variants,
        labels,
        protocol,
        scene_inventory,
        selection,
    )

    for name in APACHE_PACKAGES:
        if name in sys.modules:
            FAILURES.append(f"{name} was imported by the GPL worker")

    for name, module in sorted(sys.modules.items()):
        if not name.startswith("asset_mania"):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        if "blender-addon" not in origin:
            FAILURES.append(f"{name} resolved outside the GPL tree: {origin}")

    for failure in FAILURES:
        print(f"IMPORT_BOUNDARY_VIOLATION {failure}")
    print(f"IMPORT_BOUNDARY {'ok' if not FAILURES else 'failed'}")
    return 1 if FAILURES else 0


raise SystemExit(main())
