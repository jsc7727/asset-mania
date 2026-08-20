"""OS isolation for the Blender worker.

Three properties must hold for every isolated run:

1. the source file and its containing directory are read-only;
2. the staging tree is the only writable tree;
3. network access is denied.

`--offline-mode` inside Blender is defence in depth, not an OS guarantee, which is why
this module exists. When the requested backend is unavailable it fails closed: there is no
fallback to an unsandboxed launch.

The launcher composes this with `build_environment`, because isolation alone is not
enough -- Blender needs HOME, the temporary directories, and its user resource directory to
be writable, and they are only writable below staging.
"""

import os
import platform
import shutil
from collections.abc import Sequence
from pathlib import Path

#: A measured limitation, not a guess. Blender 5.2's compositor is Metal-backed: rendering
#: with a compositor graph makes the Metal frontend build a shader module and write it into
#: the darwin user cache, outside staging. A narrow write allowance on exactly that subtree
#: was tried and still fails -- the frontend needs more than a file rule -- and granting
#: what it wants would mean broad system access, voiding the staging-only write property.
#: An isolated conditioning run therefore belongs to the Linux bubblewrap backend, which
#: the design already makes the authoritative byte-exact environment. Every other stage
#: runs under the macOS backend.
MACOS_COMPOSITOR_LIMIT = (
    "Blender 5.2's Metal-backed compositor cannot run under the macOS Seatbelt profile "
    "with a staging-only write boundary; use the Linux bubblewrap backend for an isolated "
    "conditioning run."
)

MACOS_BACKEND = "macos-sandbox"
LINUX_BACKEND = "linux-bubblewrap"
BACKENDS = (MACOS_BACKEND, LINUX_BACKEND)

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"
_BUBBLEWRAP = "bwrap"


class IsolationUnavailable(Exception):
    """The requested isolation backend is not usable, so no run may proceed."""


def detect_backend(system: str | None = None) -> str:
    """Choose the backend for this platform, or refuse to run."""
    system = (system or platform.system()).lower()
    if system == "darwin":
        if not Path(_SANDBOX_EXEC).exists():
            raise IsolationUnavailable(f"{_SANDBOX_EXEC} is absent; refusing to run unsandboxed")
        return MACOS_BACKEND
    if system == "linux":
        if shutil.which(_BUBBLEWRAP) is None:
            raise IsolationUnavailable(f"{_BUBBLEWRAP} is absent; refusing to run unsandboxed")
        return LINUX_BACKEND
    raise IsolationUnavailable(f"no isolation backend is defined for {system!r}")


def _sbpl_literal(value: Path) -> str:
    """Quote a path for a Seatbelt profile."""
    text = os.fspath(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def install_root(executable: Path) -> Path:
    """The tree a launched executable must be able to read to run at all.

    A macOS application bundle owns everything under its `.app`; anything else is treated
    as a conventional `prefix/bin/tool` layout. The path is resolved first, because a
    launcher is often a symlink into a store directory elsewhere on disk.

    The result never widens to the filesystem root or to a bare top-level directory. A
    `/bin/tool` layout would otherwise grant a recursive read of `/`, which would quietly
    undo `(deny default)`.
    """
    resolved = executable.resolve()
    for parent in resolved.parents:
        if parent.suffix == ".app":
            return parent

    prefix = resolved.parent.parent
    if len(prefix.parts) < 3:
        return resolved.parent
    return prefix


def build_macos_profile(
    *,
    source: Path,
    staging: Path,
    executable: Path,
    extra_read_paths: Sequence[Path] = (),
) -> str:
    """The pinned Seatbelt profile.

    It enforces the three required properties: the source and its directory are read-only,
    staging is the only writable tree, and network is denied.

    It deliberately does **not** confine reads. Seatbelt cannot boot a process under
    `(deny default)` with a small read allowlist — the loader needs resources that are not
    enumerable up front — so a read-confining macOS profile would simply fail to run
    Blender. Read confinement is provided by the Linux bubblewrap backend, which binds
    only the trees it names, and that is the platform the design makes authoritative for
    the byte-exact E2E. On macOS, treat read confinement as out of scope and rely on the
    write and network boundaries.

    SBPL evaluates later rules last, so the global write denial comes after the read
    grant, the staging grant after that denial, and the source-directory denial last.
    """
    source = source.resolve()
    staging = staging.resolve()
    resolved = executable.resolve()
    read_roots = [install_root(executable), *(path.resolve() for path in extra_read_paths)]
    return "\n".join(
        [
            "(version 1)",
            "(deny default)",
            "(deny network*)",
            "(allow process-fork)",
            "(allow process-exec*)",
            "(allow sysctl-read)",
            "(allow mach-lookup)",
            "(allow ipc-posix-shm)",
            "(allow signal (target self))",
            # Blender 5.2 probes the Metal backend during `wm_homefile_read_ex` even under
            # `--background`, and dereferences the device without a null check. Denying
            # IOKit therefore makes Blender segfault before it can run anything, so the
            # device-enumeration path has to be allowed. This widens the profile: it grants
            # read access to IOKit device properties. The write and network boundaries --
            # the two this profile exists to enforce -- are unaffected.
            "(allow iokit-open)",
            "(allow file-read*)",
            f"(allow process-exec* (literal {_sbpl_literal(resolved)}))",
            *(f"(allow process-exec* (subpath {_sbpl_literal(root)}))" for root in read_roots),
            "(deny file-write*)",
            f"(allow file-write* (subpath {_sbpl_literal(staging)}))",
            f"(deny file-write* (subpath {_sbpl_literal(source.parent)}))",
            "",
        ]
    )


def build_macos_command(
    *, profile: str, executable: Path, argv: Sequence[str]
) -> tuple[list[str], str]:
    """`sandbox-exec -p PROFILE EXECUTABLE ARGV...`, with the profile passed inline."""
    return [_SANDBOX_EXEC, "-p", profile, os.fspath(executable), *argv], profile


def build_bubblewrap_command(
    *, source: Path, staging: Path, executable: Path, argv: Sequence[str]
) -> list[str]:
    """A bubblewrap vector with a read-only root, one writable bind, and no network."""
    source = source.resolve()
    staging = staging.resolve()
    return [
        _BUBBLEWRAP,
        "--unshare-all",
        "--unshare-net",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind-try",
        "/lib",
        "/lib",
        "--ro-bind-try",
        "/lib64",
        "/lib64",
        "--ro-bind-try",
        "/bin",
        "/bin",
        "--ro-bind-try",
        "/sbin",
        "/sbin",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--ro-bind",
        os.fspath(executable.parent),
        os.fspath(executable.parent),
        "--ro-bind",
        os.fspath(source.parent),
        os.fspath(source.parent),
        "--bind",
        os.fspath(staging),
        os.fspath(staging),
        "--chdir",
        os.fspath(staging),
        os.fspath(executable),
        *argv,
    ]


def build_command(
    *,
    backend: str,
    source: Path,
    staging: Path,
    executable: Path,
    argv: Sequence[str],
    extra_read_paths: Sequence[Path] = (),
) -> list[str]:
    if backend == MACOS_BACKEND:
        profile = build_macos_profile(
            source=source,
            staging=staging,
            executable=executable,
            extra_read_paths=extra_read_paths,
        )
        command, _ = build_macos_command(profile=profile, executable=executable, argv=argv)
        return command
    if backend == LINUX_BACKEND:
        return build_bubblewrap_command(
            source=source, staging=staging, executable=executable, argv=argv
        )
    raise IsolationUnavailable(f"{backend!r} is not a known isolation backend")
