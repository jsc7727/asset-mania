"""The sanitized Blender launch.

Blender starts from an empty environment with a fixed argument vector. The source path and
basename never appear in either; only the private request path does. Raw stdout and stderr
are captured so they cannot reach the user's streams, and they are never returned.
"""

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from .envelope import PrivateEnvelope

PYTHON_EXIT_CODE = 86
DEFAULT_TIMEOUT_SECONDS = 300
TIMEOUT_RANGE_SECONDS = (1, 1800)

FIXED_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
FIXED_LOCALE = "C.UTF-8"
FIXED_TIMEZONE = "UTC"

# Every directory Blender or CPython might write to is created below staging. The
# environment is built as an allowlist, so an inherited PYTHONPATH, BLENDER_USER_SCRIPTS,
# OCIO, proxy variable, or provider credential is absent by construction rather than by
# being filtered out one name at a time.
_STAGING_DIRECTORIES = {
    "HOME": "home",
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "XDG_CONFIG_HOME": "xdg/config",
    "XDG_CACHE_HOME": "xdg/cache",
    "XDG_DATA_HOME": "xdg/data",
    "XDG_STATE_HOME": "xdg/state",
    "XDG_RUNTIME_DIR": "xdg/runtime",
    "BLENDER_USER_RESOURCES": "blender/resources",
}
ENVIRONMENT_KEYS = frozenset(
    {"PATH", "LC_ALL", "LANG", "TZ", "PYTHONNOUSERSITE", *_STAGING_DIRECTORIES}
)

# `ENVIRONMENT_KEYS` is exactly what this launcher passes. A child may still observe a
# few variables the operating system injects after `execve`, which no caller can suppress
# through `env=`. They are named here so a test can tell platform noise apart from an
# inherited value, and none of them can carry a caller secret or home path.
PLATFORM_INJECTED_KEYS = frozenset(
    {
        "__CF_USER_TEXT_ENCODING",
        "__PYVENV_LAUNCHER__",
        "LC_CTYPE",
        "SDKROOT",
        "CPATH",
        "LIBRARY_PATH",
        "MANPATH",
    }
)


class WorkerLaunchFailed(Exception):
    """Blender could not be started, timed out, was signalled, or exited nonzero."""


def build_environment(*, staging_root: Path) -> dict[str, str]:
    """Create the isolated directories and return the complete worker environment."""
    anchor = staging_root.resolve(strict=True)
    environment = {
        "PATH": FIXED_PATH,
        "LC_ALL": FIXED_LOCALE,
        "LANG": FIXED_LOCALE,
        "TZ": FIXED_TIMEZONE,
        "PYTHONNOUSERSITE": "1",
    }
    for key, relative in _STAGING_DIRECTORIES.items():
        directory = anchor / "worker-env" / relative
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        environment[key] = str(directory)

    assert set(environment) == set(ENVIRONMENT_KEYS)
    return environment


def build_argv(
    *,
    executable: Path,
    entrypoint: Path,
    request_path: Path,
    response_path: Path,
    threads: int = 1,
) -> list[str]:
    """The exact launch vector. It names no source file and no datablock."""
    return [
        str(executable),
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--offline-mode",
        "--threads",
        str(threads),
        "--python-exit-code",
        str(PYTHON_EXIT_CODE),
        "--python",
        str(entrypoint),
        "--",
        "--request",
        str(request_path),
        "--response",
        str(response_path),
    ]


def launch_worker(
    *,
    executable: Path,
    entrypoint: Path,
    envelope: PrivateEnvelope,
    staging_root: Path,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
    argv: Sequence[str] | None = None,
) -> None:
    """Run the worker to completion, revealing nothing it printed.

    On any failure the raised message carries a stable diagnostic code and the exit status
    only. Blender's own output is captured and discarded here; a maintainer who needs it
    must enable a local debug log, which passes through `redaction.redact` first.
    """
    low, high = TIMEOUT_RANGE_SECONDS
    if not low <= timeout_seconds <= high:
        raise ValueError(f"worker timeout must fall within {low}..{high} seconds")

    vector = list(
        argv
        if argv is not None
        else build_argv(
            executable=executable,
            entrypoint=entrypoint,
            request_path=envelope.request_path,
            response_path=envelope.response_path,
        )
    )
    worker_environment = dict(
        environment if environment is not None else build_environment(staging_root=staging_root)
    )

    try:
        completed = subprocess.run(
            vector,
            env=worker_environment,
            cwd=staging_root,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise WorkerLaunchFailed(
            f"BLENDER_EXECUTION_FAILED: the worker exceeded {timeout_seconds} seconds"
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise WorkerLaunchFailed(
            "BLENDER_EXECUTION_FAILED: the worker could not be started"
        ) from error

    if completed.returncode < 0:
        raise WorkerLaunchFailed(
            f"BLENDER_EXECUTION_FAILED: the worker was terminated by signal {-completed.returncode}"
        )
    if completed.returncode != 0:
        raise WorkerLaunchFailed(
            f"BLENDER_EXECUTION_FAILED: the worker exited with status {completed.returncode}"
        )
