"""Drive the real Blender worker through the Apache client, end to end.

This is the only script that launches Blender. It builds the private envelope, launches
the pinned executable with the sanitized profile, validates the closed response, and
verifies the source file is byte-identical afterwards. It is used by the Blender E2E tests
and by maintainers reproducing a stage locally; no unit test invokes Blender.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("contracts", "pipeline", "blender-client"):
    sys.path.insert(0, str(ROOT / "packages" / package / "src"))
sys.path.insert(0, str(ROOT / "blender-addon" / "src"))

from asset_mania_blender_client import (
    PrivateEnvelope,
    fingerprint_executable,
    launch_worker,
    load_response,
)
from asset_mania_pipeline import (
    fingerprint_source,
    verify_source_unchanged,
)

DEFAULT_BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
ENTRYPOINT = ROOT / "blender-addon" / "src" / "asset_mania_blender" / "worker_main.py"


class WorkerRunFailed(Exception):
    """The end-to-end run did not produce a valid response."""


def run_worker(
    *,
    request: dict,
    staging_root: Path,
    executable: Path,
    operation: str,
    timeout_seconds: int = 300,
) -> dict:
    """Run one worker operation and return its validated response."""
    fingerprint = fingerprint_executable(executable)
    source = request.get("source_path")
    before = fingerprint_source(Path(source)) if source else None

    with PrivateEnvelope(staging_root) as envelope:
        envelope.write_request({**request, "staging_root": str(staging_root)})
        launch_worker(
            executable=fingerprint.executable,
            entrypoint=ENTRYPOINT,
            envelope=envelope,
            staging_root=staging_root,
            timeout_seconds=timeout_seconds,
        )
        response = load_response(
            envelope.response_path,
            request_id=str(request["request_id"]),
            operation=operation,
            staging_root=staging_root,
        )

    if before is not None:
        verify_source_unchanged(Path(source), before)
    return response


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True, help="JSON request body")
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--executable", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    arguments = _parse(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    try:
        response = run_worker(
            request=request,
            staging_root=arguments.staging,
            executable=arguments.executable,
            operation=arguments.operation,
            timeout_seconds=arguments.timeout,
        )
    except Exception as error:  # noqa: BLE001 - the driver reports, it does not raise
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 3

    print(json.dumps(response, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
