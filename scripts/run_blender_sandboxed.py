"""Run the Blender worker under an OS isolation boundary.

The profile and command construction live in
`asset_mania_blender_client.isolation`; this script is the command-line entry point over
them, so the same boundary is used whether a maintainer invokes it directly or the client
composes it during a stage.
"""

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("contracts", "pipeline", "blender-client"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))

from asset_mania_blender_client.isolation import (
    BACKENDS,
    IsolationUnavailable,
    build_command,
    detect_backend,
)
from asset_mania_blender_client.launcher import build_environment


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--backend", choices=BACKENDS)
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("worker_argv", nargs=argparse.REMAINDER)
    return parser.parse_args(list(argv))


def main(argv: Sequence[str]) -> int:
    arguments = _parse(argv)
    worker_argv = [item for item in arguments.worker_argv if item != "--"]

    try:
        backend = arguments.backend or detect_backend()
        command = build_command(
            backend=backend,
            source=arguments.source,
            staging=arguments.staging,
            executable=arguments.executable,
            argv=worker_argv,
        )
    except IsolationUnavailable as error:
        print(f"ISOLATION_UNAVAILABLE {error}", file=sys.stderr)
        return 4

    if arguments.print_command:
        for item in command:
            print(item)
        return 0

    completed = subprocess.run(
        command,
        env=build_environment(staging_root=arguments.staging),
        cwd=arguments.staging,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
