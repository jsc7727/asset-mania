#!/usr/bin/env python3
"""Launch the installed Asset Mania CLI without inheriting environment secrets."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_SAFE_ENVIRONMENT_KEYS = (
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "TEMP",
    "TMP",
    "TMPDIR",
)
_INSTALLATION_GUIDANCE = (
    "asset-mania was not found. Install asset-mania-cli on PATH, or run this launcher "
    "from an Asset Mania repository checkout with uv installed."
)


def _safe_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in _SAFE_ENVIRONMENT_KEYS if key in os.environ}


def _find_repository(start: Path) -> Path | None:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "packages" / "cli" / "pyproject.toml"
        ).is_file():
            return candidate
    return None


def _command(arguments: Sequence[str]) -> list[str] | None:
    installed = shutil.which("asset-mania")
    if installed is not None:
        return [installed, *arguments]

    repository = _find_repository(Path.cwd().resolve())
    uv = shutil.which("uv")
    if repository is not None and uv is not None:
        return [uv, "run", "--package", "asset-mania-cli", "asset-mania", *arguments]
    return None


def main(arguments: Sequence[str] | None = None) -> int:
    forwarded = list(sys.argv[1:] if arguments is None else arguments)
    command = _command(forwarded)
    if command is None:
        print(_INSTALLATION_GUIDANCE, file=sys.stderr)
        return 127
    return subprocess.run(
        command,
        env=_safe_environment(),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
