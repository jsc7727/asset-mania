"""The worker starts from an empty environment with a source-free argument vector."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from asset_mania_blender_client import (
    ENVIRONMENT_KEYS,
    FIXED_PATH,
    PLATFORM_INJECTED_KEYS,
    PYTHON_EXIT_CODE,
    TIMEOUT_RANGE_SECONDS,
    PrivateEnvelope,
    WorkerLaunchFailed,
    build_argv,
    build_environment,
    launch_worker,
)

SOURCE_PATH = "/Users/example/scenes/private-character.blend"
SOURCE_BASENAME = "private-character.blend"
PRIVATE_NAMES = ("Body_LOD0", "Camera_Main", "Rig", "Idle")


def _launch(executable: Path, entrypoint: Path, staging: Path, **kwargs) -> None:
    with PrivateEnvelope(staging) as envelope:
        envelope.write_request(
            {
                "request_id": "request-preflight-1",
                "operation": "preflight",
                "source_path": SOURCE_PATH,
                "target_name": "Body_LOD0",
            }
        )
        launch_worker(
            executable=executable,
            entrypoint=entrypoint,
            envelope=envelope,
            staging_root=staging,
            **kwargs,
        )


# --- Argument vector -----------------------------------------------------------


def test_the_argument_vector_is_the_pinned_launch_profile(tmp_path: Path) -> None:
    argv = build_argv(
        executable=Path("/opt/blender"),
        entrypoint=Path("/opt/entrypoint.py"),
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
    )
    assert argv[:10] == [
        "/opt/blender",
        "--background",
        "--factory-startup",
        "--disable-autoexec",
        "--offline-mode",
        "--threads",
        "1",
        "--python-exit-code",
        str(PYTHON_EXIT_CODE),
        "--python",
    ]
    assert argv[10] == "/opt/entrypoint.py"
    assert argv[11] == "--"
    assert argv[12:] == [
        "--request",
        str(tmp_path / "request.json"),
        "--response",
        str(tmp_path / "response.json"),
    ]


def test_the_argument_vector_names_no_source_or_datablock(tmp_path: Path) -> None:
    argv = build_argv(
        executable=Path("/opt/blender"),
        entrypoint=Path("/opt/entrypoint.py"),
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
    )
    rendered = " ".join(argv)
    assert SOURCE_PATH not in rendered
    assert SOURCE_BASENAME not in rendered
    for name in PRIVATE_NAMES:
        assert name not in rendered


# --- Environment ---------------------------------------------------------------


def test_the_environment_is_an_allowlist(staging: Path) -> None:
    environment = build_environment(staging_root=staging)
    assert set(environment) == set(ENVIRONMENT_KEYS)
    assert environment["PATH"] == FIXED_PATH
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["TZ"] == "UTC"


def test_every_writable_directory_sits_below_staging_at_mode_0700(staging: Path) -> None:
    environment = build_environment(staging_root=staging)
    anchor = staging.resolve()
    for key in ("HOME", "TMPDIR", "XDG_CONFIG_HOME", "BLENDER_USER_RESOURCES"):
        directory = Path(environment[key])
        assert directory.is_dir()
        assert directory.resolve().is_relative_to(anchor)
        assert os.stat(directory).st_mode & 0o777 == 0o700


def test_the_blender_user_resources_directory_starts_empty(staging: Path) -> None:
    environment = build_environment(staging_root=staging)
    resources = Path(environment["BLENDER_USER_RESOURCES"])
    assert list(resources.iterdir()) == []


def test_the_worker_inherits_nothing_from_a_polluted_environment(
    capturing_blender, entrypoint: Path, staging: Path, polluted_environment
) -> None:
    executable, capture = capturing_blender
    caller_home, polluted = polluted_environment

    _launch(executable, entrypoint, staging)
    observed = json.loads(capture.read_text())["environment"]

    # The launcher's contract is what it passes; the OS may still inject its own names.
    assert set(observed) - set(PLATFORM_INJECTED_KEYS) == set(ENVIRONMENT_KEYS)
    for denied in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "BLENDER_USER_SCRIPTS",
        "BLENDER_USER_EXTENSIONS",
        "BLENDER_USER_CONFIG",
        "BLENDER_SYSTEM_SCRIPTS",
        "OCIO",
        "http_proxy",
        "https_proxy",
        "ALL_PROXY",
        "NO_PROXY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        assert denied not in observed, denied

    rendered = json.dumps(observed)
    assert str(caller_home) not in rendered
    assert polluted["OPENAI_API_KEY"] not in rendered
    assert observed["LC_ALL"] == "C.UTF-8"
    assert observed["TZ"] == "UTC"


def test_the_worker_never_sees_the_source_path_or_basename(
    capturing_blender, entrypoint: Path, staging: Path
) -> None:
    executable, capture = capturing_blender
    _launch(executable, entrypoint, staging)
    captured = capture.read_text()

    assert SOURCE_PATH not in captured
    assert SOURCE_BASENAME not in captured
    for name in PRIVATE_NAMES:
        assert name not in captured


def test_the_worker_receives_only_the_private_request_and_response_paths(
    capturing_blender, entrypoint: Path, staging: Path
) -> None:
    executable, capture = capturing_blender
    _launch(executable, entrypoint, staging)
    argv = json.loads(capture.read_text())["argv"]

    passed_paths = [item for item in argv if item.startswith("/")]
    anchor = staging.resolve()
    for path in passed_paths:
        assert Path(path) == entrypoint or Path(path).resolve().is_relative_to(anchor)


def test_the_worker_runs_with_staging_as_its_working_directory(
    capturing_blender, entrypoint: Path, staging: Path
) -> None:
    executable, capture = capturing_blender
    _launch(executable, entrypoint, staging)
    assert Path(json.loads(capture.read_text())["cwd"]).resolve() == staging.resolve()


# --- Malicious startup sentinels ----------------------------------------------


def test_a_planted_user_startup_script_never_executes(
    fake_blender, entrypoint: Path, staging: Path, tmp_path: Path, polluted_environment
) -> None:
    """A startup script reachable only through an inherited variable must stay inert."""
    caller_home, _ = polluted_environment
    sentinel = tmp_path / "startup-ran"
    startup = caller_home / "startup.py"
    startup.write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    executable = fake_blender(
        "import os, sys\n"
        "startup = os.environ.get('PYTHONSTARTUP')\n"
        "if startup:\n"
        "    exec(open(startup).read())\n"
        "raise SystemExit(0)\n"
    )
    _launch(executable, entrypoint, staging)
    assert not sentinel.exists()


def test_a_planted_blender_user_script_directory_is_never_reachable(
    fake_blender, entrypoint: Path, staging: Path, tmp_path: Path, polluted_environment
) -> None:
    caller_home, _ = polluted_environment
    sentinel = tmp_path / "addon-ran"
    scripts = caller_home / "blender-scripts" / "addons"
    scripts.mkdir(parents=True)
    (scripts / "malicious.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n",
        encoding="utf-8",
    )

    executable = fake_blender(
        "import os\n"
        "from pathlib import Path\n"
        "scripts = os.environ.get('BLENDER_USER_SCRIPTS')\n"
        "if scripts:\n"
        "    for path in Path(scripts).rglob('*.py'):\n"
        "        exec(path.read_text())\n"
        "raise SystemExit(0)\n"
    )
    _launch(executable, entrypoint, staging)
    assert not sentinel.exists()


def test_a_worker_that_writes_into_its_own_home_does_not_touch_the_caller_home(
    fake_blender, entrypoint: Path, staging: Path, polluted_environment
) -> None:
    caller_home, _ = polluted_environment
    executable = fake_blender(
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['HOME'], 'worker-wrote-here').write_text('ok')\n"
        "raise SystemExit(0)\n"
    )
    _launch(executable, entrypoint, staging)

    assert list(caller_home.rglob("worker-wrote-here")) == []
    assert list(staging.rglob("worker-wrote-here"))


# --- Failure mapping ----------------------------------------------------------


def test_a_nonzero_exit_reports_only_the_status(
    fake_blender, entrypoint: Path, staging: Path
) -> None:
    executable = fake_blender(
        "import sys\n"
        "sys.stdout.write('opening /Users/example/scenes/private-character.blend\\n')\n"
        "sys.stderr.write('Error: Body_LOD0 has no UV map\\n')\n"
        "raise SystemExit(3)\n"
    )
    with pytest.raises(WorkerLaunchFailed) as failure:
        _launch(executable, entrypoint, staging)

    message = str(failure.value)
    assert "BLENDER_EXECUTION_FAILED" in message
    assert "status 3" in message
    assert SOURCE_PATH not in message
    assert SOURCE_BASENAME not in message
    assert "Body_LOD0" not in message


def test_the_python_exit_code_is_reported_as_a_worker_failure(
    fake_blender, entrypoint: Path, staging: Path
) -> None:
    executable = fake_blender(f"raise SystemExit({PYTHON_EXIT_CODE})\n")
    with pytest.raises(WorkerLaunchFailed, match=f"status {PYTHON_EXIT_CODE}"):
        _launch(executable, entrypoint, staging)


def test_a_signalled_worker_is_reported_as_a_worker_failure(
    fake_blender, entrypoint: Path, staging: Path
) -> None:
    executable = fake_blender("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n")
    with pytest.raises(WorkerLaunchFailed, match="signal 9"):
        _launch(executable, entrypoint, staging)


def test_a_timeout_is_reported_as_a_worker_failure(
    fake_blender, entrypoint: Path, staging: Path
) -> None:
    executable = fake_blender("import time\ntime.sleep(30)\n")
    with pytest.raises(WorkerLaunchFailed, match="exceeded 1 seconds"):
        _launch(executable, entrypoint, staging, timeout_seconds=1)


def test_a_missing_executable_is_reported_as_a_worker_failure(
    entrypoint: Path, staging: Path, tmp_path: Path
) -> None:
    with pytest.raises(WorkerLaunchFailed, match="could not be started"):
        _launch(tmp_path / "absent-blender", entrypoint, staging)


@pytest.mark.parametrize("timeout", [0, -1, 1801, 100000])
def test_a_timeout_outside_the_plan_range_is_refused(
    fake_blender, entrypoint: Path, staging: Path, timeout: int
) -> None:
    low, high = TIMEOUT_RANGE_SECONDS
    assert (low, high) == (1, 1800)
    executable = fake_blender("raise SystemExit(0)\n")
    with pytest.raises(ValueError, match="timeout"):
        _launch(executable, entrypoint, staging, timeout_seconds=timeout)


def test_worker_output_is_never_returned(
    fake_blender, entrypoint: Path, staging: Path, capfd
) -> None:
    executable = fake_blender(
        "import sys\n"
        "sys.stdout.write('Blender quit; /Users/example/scenes/private-character.blend\\n')\n"
        "sys.stderr.write('Body_LOD0\\n')\n"
        "raise SystemExit(0)\n"
    )
    _launch(executable, entrypoint, staging)

    captured = capfd.readouterr()
    assert SOURCE_PATH not in captured.out + captured.err
    assert "Body_LOD0" not in captured.out + captured.err


def test_a_relative_executable_is_not_resolved_through_path(
    entrypoint: Path, staging: Path, monkeypatch, tmp_path: Path
) -> None:
    """Only an explicit executable is launched; PATH lookup is not a discovery channel."""
    directory = tmp_path / "hostile-bin"
    directory.mkdir()
    sentinel = tmp_path / "path-lookup-ran"
    hostile = directory / "blender"
    hostile.write_text(
        f"#!/bin/sh\ntouch {sentinel}\nexit 0\n",
        encoding="utf-8",
    )
    hostile.chmod(0o700)
    monkeypatch.setenv("PATH", str(directory))

    with pytest.raises(WorkerLaunchFailed, match="could not be started"):
        _launch(Path("blender"), entrypoint, staging)
    assert not sentinel.exists()


def test_subprocess_is_called_without_shell(monkeypatch, entrypoint: Path, staging: Path) -> None:
    recorded: dict[str, object] = {}
    real_run = subprocess.run

    def recording_run(*args, **kwargs):
        recorded["args"] = args
        recorded["kwargs"] = kwargs
        return real_run([os.fspath(Path(os.sys.executable)), "-c", ""], **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    _launch(Path("/opt/blender"), entrypoint, staging)

    assert recorded["kwargs"].get("shell") is None
    assert isinstance(recorded["args"][0], list)
