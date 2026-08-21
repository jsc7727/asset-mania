"""Client tests use a fake executable. Blender itself is never invoked here."""

import os
import stat
import sys
from pathlib import Path

import pytest

FAKE_VERSION_OUTPUT = """Blender 5.2.0 LTS
\tbuild date: 2026-07-14
\tbuild time: 01:31:22
\tbuild hash: fbe6228777e7
\tbuild branch: blender-v5.2-release
"""


def _write_executable(path: Path, script: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{script}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def fake_blender(tmp_path: Path):
    """Build a fake `blender` whose behaviour each test chooses."""

    def build(script: str, *, name: str = "blender") -> Path:
        directory = tmp_path / "bin"
        directory.mkdir(exist_ok=True)
        return _write_executable(directory / name, script)

    return build


@pytest.fixture
def version_reporting_blender(fake_blender):
    return fake_blender(
        f"import sys\nsys.stdout.write({FAKE_VERSION_OUTPUT!r})\nraise SystemExit(0)\n"
    )


@pytest.fixture
def capturing_blender(fake_blender, tmp_path: Path):
    """A fake worker that records its own argv, environment, and cwd, then exits 0."""
    capture = tmp_path / "capture.json"
    script = "\n".join(
        [
            "import json, os, sys",
            "from pathlib import Path",
            "record = {",
            "    'argv': sys.argv[1:],",
            "    'environment': dict(os.environ),",
            "    'cwd': os.getcwd(),",
            "}",
            f"Path({str(capture)!r}).write_text(json.dumps(record))",
            "raise SystemExit(0)",
            "",
        ]
    )
    return fake_blender(script), capture


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    root = tmp_path / "staging"
    root.mkdir(mode=0o700)
    return root


@pytest.fixture
def entrypoint(tmp_path: Path) -> Path:
    path = tmp_path / "entrypoint.py"
    path.write_text("# SPDX-License-Identifier: GPL-3.0-or-later\n", encoding="utf-8")
    return path


@pytest.fixture
def polluted_environment(monkeypatch, tmp_path: Path):
    """Every variable the launcher must refuse to inherit."""
    caller_home = tmp_path / "caller-home"
    caller_home.mkdir()
    polluted = {
        "PYTHONPATH": str(caller_home / "site-packages"),
        "PYTHONHOME": str(caller_home / "python"),
        "PYTHONSTARTUP": str(caller_home / "startup.py"),
        "PYTHONUSERBASE": str(caller_home / "user-base"),
        "BLENDER_USER_SCRIPTS": str(caller_home / "blender-scripts"),
        "BLENDER_USER_EXTENSIONS": str(caller_home / "blender-extensions"),
        "BLENDER_USER_CONFIG": str(caller_home / "blender-config"),
        "BLENDER_SYSTEM_SCRIPTS": str(caller_home / "blender-system"),
        "OCIO": str(caller_home / "config.ocio"),
        "http_proxy": "http://proxy.invalid:8080",
        "https_proxy": "http://proxy.invalid:8080",
        "ALL_PROXY": "socks5://proxy.invalid:1080",
        "NO_PROXY": "localhost",
        "OPENAI_API_KEY": "PROVIDER-CREDENTIAL-MUST-NOT-BE-FORWARDED",
        "AWS_SECRET_ACCESS_KEY": "must-not-be-forwarded",
        "GOOGLE_APPLICATION_CREDENTIALS": str(caller_home / "gcp.json"),
        "HOME": str(caller_home),
        "TMPDIR": str(caller_home / "tmp"),
        "XDG_CONFIG_HOME": str(caller_home / "config"),
        "LC_ALL": "de_DE.ISO8859-1",
        "TZ": "Europe/Berlin",
    }
    for key, value in polluted.items():
        monkeypatch.setenv(key, value)
    assert os.environ["OPENAI_API_KEY"].startswith("PROVIDER-CREDENTIAL")
    return caller_home, polluted
