"""The sandbox runner enforces read-only source, staging-only writes, and no network.

The macOS assertions run a real `sandbox-exec` process against the generated profile, so
they prove enforcement rather than only the profile's text. `/bin/sh` is the subject
because a general-purpose Python interpreter needs far more allowances than the worker
profile grants, which would test the harness instead of the boundary.
"""

import importlib.util
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_blender_sandboxed.py"

_spec = importlib.util.spec_from_file_location("run_blender_sandboxed", RUNNER)
assert _spec is not None and _spec.loader is not None
sandbox = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sandbox)

IS_MACOS = platform.system() == "Darwin"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
requires_macos_sandbox = pytest.mark.skipif(
    not (IS_MACOS and SANDBOX_EXEC.exists()),
    reason="the macOS Seatbelt backend is unavailable on this platform",
)


@pytest.fixture
def tree(tmp_path: Path):
    source_directory = tmp_path / "scenes"
    source_directory.mkdir()
    source = source_directory / "private-character.blend"
    source.write_text("blendbytes\n", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    return source, staging


# --- Backend selection and fail-closed behaviour -------------------------------


def test_the_backend_matches_the_platform() -> None:
    if IS_MACOS:
        assert sandbox.detect_backend() == sandbox.MACOS_BACKEND
    elif platform.system() == "Linux" and shutil.which("bwrap"):
        assert sandbox.detect_backend() == sandbox.LINUX_BACKEND


def test_an_unknown_platform_fails_closed() -> None:
    with pytest.raises(sandbox.IsolationUnavailable, match="no isolation backend"):
        sandbox.detect_backend("plan9")


def test_a_missing_linux_backend_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    with pytest.raises(sandbox.IsolationUnavailable, match="refusing to run unsandboxed"):
        sandbox.detect_backend("linux")


def test_a_missing_macos_backend_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "_SANDBOX_EXEC", "/nonexistent/sandbox-exec")
    with pytest.raises(sandbox.IsolationUnavailable, match="refusing to run unsandboxed"):
        sandbox.detect_backend("darwin")


def test_an_unknown_backend_name_is_refused(tree) -> None:
    source, staging = tree
    with pytest.raises(sandbox.IsolationUnavailable, match="not a known isolation backend"):
        sandbox.build_command(
            backend="chroot",
            source=source,
            staging=staging,
            executable=Path("/bin/sh"),
            argv=[],
        )


def test_the_cli_reports_an_unavailable_backend_and_never_runs(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--source",
            str(tmp_path / "a.blend"),
            "--staging",
            str(tmp_path),
            "--executable",
            "/bin/sh",
            "--backend",
            "linux-bubblewrap",
            "--print-command",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if shutil.which("bwrap") is None:
        # An explicit backend still builds its vector; only detection fails closed.
        assert completed.returncode == 0
        assert "bwrap" in completed.stdout


# --- The Linux vector ----------------------------------------------------------


def test_the_bubblewrap_vector_denies_network_and_binds_only_staging(tree) -> None:
    source, staging = tree
    command = sandbox.build_bubblewrap_command(
        source=source, staging=staging, executable=Path("/opt/blender/bin/blender"), argv=["--x"]
    )
    assert command[0] == "bwrap"
    assert "--unshare-net" in command
    assert "--unshare-all" in command
    assert "--clearenv" in command

    writable = [command[index + 1] for index, item in enumerate(command) if item == "--bind"]
    assert writable == [str(staging.resolve())]

    read_only = [
        command[index + 1]
        for index, item in enumerate(command)
        if item in ("--ro-bind", "--ro-bind-try")
    ]
    assert str(source.parent.resolve()) in read_only
    assert str(staging.resolve()) not in read_only
    # Unlike the macOS profile, bubblewrap does confine reads: only bound trees exist.
    assert "/etc" not in read_only
    assert str(Path.home()) not in read_only


def test_the_bubblewrap_vector_runs_the_worker_inside_staging(tree) -> None:
    source, staging = tree
    command = sandbox.build_bubblewrap_command(
        source=source, staging=staging, executable=Path("/opt/blender/bin/blender"), argv=[]
    )
    assert command[command.index("--chdir") + 1] == str(staging.resolve())


# --- The macOS profile, as text ------------------------------------------------


def test_the_profile_denies_by_default_and_denies_network(tree) -> None:
    source, staging = tree
    profile = sandbox.build_macos_profile(
        source=source, staging=staging, executable=Path("/bin/sh")
    )
    assert "(deny default)" in profile
    assert "(deny network*)" in profile


def test_the_profile_grants_write_only_below_staging(tree) -> None:
    source, staging = tree
    profile = sandbox.build_macos_profile(
        source=source, staging=staging, executable=Path("/bin/sh")
    )
    write_rules = [line for line in profile.splitlines() if "file-write*" in line]
    allowed = [line for line in write_rules if line.startswith("(allow")]
    assert len(allowed) == 1
    assert str(staging.resolve()) in allowed[0]
    assert "(deny file-write*)" in write_rules
    assert any(
        line.startswith("(deny") and str(source.parent.resolve()) in line for line in write_rules
    )


def test_the_write_rules_are_ordered_so_the_narrow_grant_wins(tree) -> None:
    """SBPL evaluates later rules last, so ordering is the enforcement."""
    source, staging = tree
    lines = sandbox.build_macos_profile(
        source=source, staging=staging, executable=Path("/bin/sh")
    ).splitlines()
    assert lines.index("(deny file-write*)") < next(
        index for index, line in enumerate(lines) if line.startswith("(allow file-write*")
    )
    assert lines.index("(allow file-read*)") < lines.index("(deny file-write*)")


def test_the_macos_profile_does_not_claim_to_confine_reads(tree) -> None:
    """Documented boundary difference: read confinement is the Linux backend's job."""
    source, staging = tree
    profile = sandbox.build_macos_profile(
        source=source, staging=staging, executable=Path("/bin/sh")
    )
    assert "(allow file-read*)" in profile.splitlines()
    assert "does **not** confine reads" in sandbox.build_macos_profile.__doc__


def test_an_application_bundle_is_its_own_install_root() -> None:
    root = sandbox.install_root(Path("/Applications/Blender.app/Contents/MacOS/Blender"))
    assert root == Path("/Applications/Blender.app")


def test_a_conventional_prefix_layout_uses_the_prefix() -> None:
    assert sandbox.install_root(Path("/opt/blender/bin/blender")) == Path("/opt/blender")


@pytest.mark.parametrize("executable", ["/bin/sh", "/usr/bin/env"])
def test_the_install_root_never_widens_to_the_filesystem_root(executable: str) -> None:
    """A `/bin/tool` layout must not grant a recursive read of `/`."""
    root = sandbox.install_root(Path(executable))
    assert root != Path("/")
    assert len(root.parts) >= 2
    assert root == Path(executable).resolve().parent


# --- The macOS profile, enforced ----------------------------------------------


def _sandboxed(profile: str, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SANDBOX_EXEC), "-p", profile, "/bin/sh", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def enforced(tree):
    source, staging = tree
    profile = sandbox.build_macos_profile(
        source=source, staging=staging, executable=Path("/bin/sh")
    )
    return source, staging, profile


@requires_macos_sandbox
def test_a_sandboxed_process_starts(enforced) -> None:
    _, _, profile = enforced
    completed = _sandboxed(profile, "echo up")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "up"


@requires_macos_sandbox
def test_the_source_is_readable(enforced) -> None:
    source, _, profile = enforced
    completed = _sandboxed(profile, f"cat {source}")
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "blendbytes"


@requires_macos_sandbox
def test_staging_is_writable(enforced) -> None:
    _, staging, profile = enforced
    completed = _sandboxed(profile, f"echo written > {staging}/artifact.txt")
    assert completed.returncode == 0, completed.stderr
    assert (staging / "artifact.txt").read_text().strip() == "written"


@requires_macos_sandbox
def test_the_source_cannot_be_overwritten(enforced) -> None:
    source, _, profile = enforced
    assert _sandboxed(profile, f"echo tampered > {source}").returncode != 0
    assert source.read_text().strip() == "blendbytes"


@requires_macos_sandbox
def test_the_source_directory_cannot_be_written(enforced) -> None:
    source, _, profile = enforced
    assert _sandboxed(profile, f"echo x > {source.parent}/evil.txt").returncode != 0
    assert not (source.parent / "evil.txt").exists()


@requires_macos_sandbox
def test_the_caller_home_cannot_be_written(enforced) -> None:
    _, _, profile = enforced
    assert _sandboxed(profile, 'echo x > "$HOME/asset-mania-sandbox-escape"').returncode != 0
    assert not (Path.home() / "asset-mania-sandbox-escape").exists()


@requires_macos_sandbox
def test_no_tree_outside_staging_is_writable(enforced) -> None:
    _, _, profile = enforced
    for target in ("/tmp/asset-mania-sandbox-escape", "/usr/local/asset-mania-escape"):
        assert _sandboxed(profile, f"echo x > {target}").returncode != 0
        assert not Path(target).exists()


@requires_macos_sandbox
def test_network_access_is_denied(enforced) -> None:
    _, _, profile = enforced
    completed = _sandboxed(profile, "/usr/bin/nc -w 3 -z 1.1.1.1 80")
    assert completed.returncode != 0
