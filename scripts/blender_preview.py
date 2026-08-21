#!/usr/bin/env python3
"""Launch the in-Blender preview and fixture scripts, from the Apache side of the boundary.

The scripts this invokes live under `blender-addon/` and are GPL-3.0-or-later, because they
`import bpy` and a module that imports bpy is a derived work of Blender. This launcher only
starts a process, which is why it can stay Apache-2.0 -- the same separation the rest of the
pipeline uses, and the reason `check_license_boundary.py` flags `import bpy` outside that tree.

    .venv/bin/python scripts/blender_preview.py render --mesh out.obj --out preview.png
    .venv/bin/python scripts/blender_preview.py fixture --shape monkey --out fixture.png

Every flag after the subcommand is forwarded to the in-Blender script unchanged, so its own
`--help` is the reference:

    .venv/bin/python scripts/blender_preview.py render --help
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADDON_PREVIEW = REPO_ROOT / "blender-addon" / "src" / "asset_mania_blender" / "preview"

SCRIPTS = {
    "render": ADDON_PREVIEW / "render_mesh_preview.py",
    "fixture": ADDON_PREVIEW / "make_reconstruction_fixture.py",
}

#: The flags every Blender invocation in this project uses. `--factory-startup` keeps a user's
#: preferences from changing a result, `--disable-autoexec` and `--offline-mode` matter because
#: a mesh or a .blend is untrusted input, and one thread makes two runs of the same input agree.
BLENDER_FLAGS = (
    "--background",
    "--factory-startup",
    "--disable-autoexec",
    "--offline-mode",
    "--threads",
    "1",
    "--python-exit-code",
    "86",
)

#: Where a macOS install puts the binary. Overridable, and searched on PATH first.
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def find_blender(explicit: str | None) -> str:
    for candidate in (explicit, os.environ.get("BLENDER"), shutil.which("blender")):
        if candidate and Path(candidate).is_file():
            return candidate
    if Path(DEFAULT_BLENDER).is_file():
        return DEFAULT_BLENDER
    raise SystemExit(
        "Blender was not found. Pass --blender, set BLENDER, or put it on PATH. "
        "This project never downloads it: acquisition is a user step, before the "
        "network-deny boundary."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("command", choices=sorted(SCRIPTS), help="which in-Blender script to run")
    parser.add_argument("--blender", default=None, help="path to the Blender binary")
    parsed, forwarded = parser.parse_known_args(argv)

    script = SCRIPTS[parsed.command]
    if not script.is_file():
        raise SystemExit(f"missing in-Blender script: {script}")

    completed = subprocess.run(
        [find_blender(parsed.blender), *BLENDER_FLAGS, "--python", str(script), "--", *forwarded],
        cwd=REPO_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
