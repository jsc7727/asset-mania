"""The provider adapter is optional, and the CLI must not depend on it."""

import ast
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


ENGINE_MANIFEST = ROOT / "packages" / "engine-triposr" / "pyproject.toml"


def test_the_cli_has_no_runtime_dependency_on_the_engine() -> None:
    dependencies = _manifest(CLI_MANIFEST)["project"]["dependencies"]
    assert not any("engine" in name for name in dependencies), dependencies


def test_the_engine_is_discovered_through_an_entry_point() -> None:
    entry_points = _manifest(ENGINE_MANIFEST)["project"]["entry-points"]["asset_mania.engines"]
    assert entry_points["triposr-local"].startswith("asset_mania_engine_triposr")


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_engine_adapter_bundles_no_weight_or_downloader() -> None:
    """An engine adapter that grows a downloader is how uncleared weights arrive.

    Imports are read from the AST rather than matched as substrings: a field named
    `requests` is not the `requests` library, and a check that cannot tell the difference
    trains people to ignore it.
    """
    source_root = ROOT / "packages" / "engine-triposr" / "src"
    forbidden = {
        "huggingface_hub",
        "torch",
        "rembg",
        "onnxruntime",
        "urllib",
        "requests",
        "httpx",
        "subprocess",
        "socket",
    }
    for path in sorted(source_root.rglob("*.py")):
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        assert not (roots & forbidden), f"{path}: {sorted(roots & forbidden)}"


def test_the_engine_adapter_names_no_download_url() -> None:
    source_root = ROOT / "packages" / "engine-triposr" / "src"
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text, path
        assert "https://" not in text, path


@pytest.mark.parametrize("package", ["asset_mania_cli", "asset_mania_contracts"])
def test_no_apache_wheel_bundles_the_engine(tmp_path: Path, package: str) -> None:
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
        leaked = [name for name in archive.namelist() if "engine_triposr" in name]
    assert leaked == []


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


# --- The planned-capability gate actually fires ------------------------------------


def test_the_planned_capability_gate_catches_an_unbacked_claim(tmp_path: Path) -> None:
    """A `Planned` row must not be able to become `Available` without evidence."""
    clone = tmp_path / "clone"
    (clone / "scripts").mkdir(parents=True)
    (clone / "tools").mkdir()
    (clone / "scripts" / "check_publication.py").write_text(
        (ROOT / "scripts" / "check_publication.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (clone / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n", encoding="utf-8")
    (clone / "skills").mkdir()
    (clone / "skills" / "asset-mania").mkdir()
    (clone / "skills" / "asset-mania" / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    def run() -> subprocess.CompletedProcess[str]:
        subprocess.run(["git", "add", "-A"], cwd=clone, check=True, capture_output=True)
        return subprocess.run(
            [sys.executable, str(clone / "scripts" / "check_publication.py")],
            cwd=clone,
            capture_output=True,
            text=True,
            check=False,
        )

    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)

    # Planned: accepted.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Planned | contracts only |\n", encoding="utf-8"
    )
    assert "UNBACKED_CAPABILITY_CLAIM" not in run().stdout

    # Available with no evidence phrase: refused.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Available | it works now |\n", encoding="utf-8"
    )
    assert "UNBACKED_CAPABILITY_CLAIM" in run().stdout

    # Available with the evidence phrase present: accepted.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Available | see below |\n"
        "no engine is cleared, downloaded, or executed\n",
        encoding="utf-8",
    )
    assert "UNBACKED_CAPABILITY_CLAIM" not in run().stdout


def test_the_readme_still_reports_generic_image_to_3d_as_planned() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    row = next(line for line in readme.splitlines() if line.startswith("| Generic image to 3D |"))
    assert "Planned" in row
    assert "no engine is cleared, downloaded, or executed" in row


def test_the_skill_refuses_generic_image_to_3d() -> None:
    skill = (ROOT / "skills" / "asset-mania" / "SKILL.md").read_text(encoding="utf-8")
    assert "does not generate 3D geometry" in skill
    assert "no cleared engine" in skill.lower()
    assert "make this person 3D" in skill
