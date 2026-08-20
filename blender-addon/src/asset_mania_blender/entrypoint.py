# SPDX-License-Identifier: GPL-3.0-or-later
"""The single `--python` entry point Blender executes.

Blender passes worker arguments after a bare `--`. This module reads the private request
envelope, dispatches one operation, and always writes a closed response, so the Apache
client never has to parse Blender's stdout or stderr.

Stage operations land in later modules. Until then every operation reports the stable
`WORKFLOW_NOT_IMPLEMENTED` code rather than a partial success.
"""

import sys
from pathlib import Path

from . import protocol

_WORKER_ARGUMENT_SEPARATOR = "--"


def parse_worker_arguments(argv: list[str]) -> dict[str, str]:
    """Read `--request` and `--response` from the arguments after Blender's own."""
    if _WORKER_ARGUMENT_SEPARATOR in argv:
        argv = argv[argv.index(_WORKER_ARGUMENT_SEPARATOR) + 1 :]

    arguments: dict[str, str] = {}
    remaining = list(argv)
    while remaining:
        flag = remaining.pop(0)
        if flag not in ("--request", "--response"):
            raise ValueError("the worker accepts only --request and --response")
        if not remaining:
            raise ValueError(f"{flag} requires a value")
        arguments[flag.removeprefix("--")] = remaining.pop(0)

    missing = {"request", "response"} - set(arguments)
    if missing:
        raise ValueError(f"the worker requires {sorted(missing)}")
    return arguments


def run(argv: list[str]) -> int:
    arguments = parse_worker_arguments(argv)
    request_path = Path(arguments["request"])
    response_path = Path(arguments["response"])

    request = protocol.read_request(request_path)
    request_id = str(request.get("request_id", "request-unknown-1"))
    operation = str(request.get("operation", "preflight"))

    protocol.write_response(
        response_path,
        protocol.failure(
            request_id=request_id,
            operation=operation,
            diagnostics=["WORKFLOW_NOT_IMPLEMENTED"],
        ),
    )
    return 0


def main() -> None:  # pragma: no cover - executed only inside Blender
    raise SystemExit(run(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover - executed only inside Blender
    main()
