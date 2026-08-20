"""The provider adapter is optional, and the CLI must not depend on it."""

import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_MANIFEST = ROOT / "packages" / "cli" / "pyproject.toml"
PROVIDER_MANIFEST = ROOT / "packages" / "provider-openai" / "pyproject.toml"


def _manifest(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_the_cli_has_no_runtime_dependency_on_the_provider() -> None:
    dependencies = _manifest(CLI_MANIFEST)["project"]["dependencies"]
    assert not any("provider" in name for name in dependencies), dependencies


def test_the_provider_is_discovered_through_an_entry_point() -> None:
    manifest = _manifest(PROVIDER_MANIFEST)
    entry_points = manifest["project"]["entry-points"]["asset_mania.providers"]
    assert entry_points["openai"].startswith("asset_mania_provider_openai")


def test_the_provider_is_a_separate_distribution() -> None:
    manifest = _manifest(PROVIDER_MANIFEST)
    assert manifest["project"]["name"] == "asset-mania-provider-openai"
    assert manifest["project"]["license"] == "Apache-2.0"


def test_the_adapter_imports_no_blender_module() -> None:
    source_root = ROOT / "packages" / "provider-openai" / "src"
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for module in ("import bpy", "import mathutils"):
            assert module not in text, f"{path}: {module}"


def test_the_adapter_constructs_no_socket_or_url_opener() -> None:
    """Transport is injected, so the adapter must not reach for one itself."""
    source_root = ROOT / "packages" / "provider-openai" / "src"
    forbidden = (
        "import socket",
        "import http.client",
        "urllib.request",
        "import requests",
        "httpx",
    )
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path}: {name}"


@pytest.mark.parametrize("package", ["asset_mania_cli", "asset_mania_contracts"])
def test_no_apache_wheel_bundles_the_provider(tmp_path: Path, package: str) -> None:
    distribution = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", "--all-packages", "-o", str(distribution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"uv build is unavailable here: {completed.stderr[-200:]}")

    wheels = sorted(distribution.glob(f"{package}-*.whl"))
    assert wheels, f"no wheel built for {package}"
    with zipfile.ZipFile(wheels[0]) as archive:
        leaked = [name for name in archive.namelist() if "provider_openai" in name]
    assert leaked == []


def test_the_provider_wheel_carries_only_its_own_module(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", "--package", "asset-mania-provider-openai", "-o", str(distribution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"uv build is unavailable here: {completed.stderr[-200:]}")

    wheels = sorted(distribution.glob("asset_mania_provider_openai-*.whl"))
    assert wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        modules = {name.split("/")[0] for name in archive.namelist() if name.endswith(".py")}
    assert modules == {"asset_mania_provider_openai"}


def test_the_adapter_is_importable_without_the_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import asset_mania_provider_openai.client as c; "
                "print('asset_mania' in sys.modules, c.PROVIDER)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "False openai" in completed.stdout
