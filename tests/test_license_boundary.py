"""The Apache/GPL boundary holds, and the checker actually detects each breach."""

import ast
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_license_boundary.py"
ADDON = ROOT / "blender-addon"
GPL_MODULES = ("bpy", "mathutils", "bmesh", "gpu")


def _run(*arguments: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _tracked() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return completed.stdout.splitlines()


# --- The boundary as it stands -------------------------------------------------


def test_the_repository_satisfies_the_boundary() -> None:
    completed = _run()
    assert completed.returncode == 0, completed.stdout


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_apache_source_imports_a_blender_module() -> None:
    """Real imports only; this very file carries `import bpy` as test data."""
    offenders = []
    for relative in _tracked():
        if not relative.endswith(".py") or relative.startswith("blender-addon/"):
            continue
        roots = _imported_roots((ROOT / relative).read_text(encoding="utf-8"))
        for module in GPL_MODULES:
            if module in roots:
                offenders.append((relative, module))
    assert offenders == []


def test_the_gpl_tree_imports_no_apache_package() -> None:
    """Only real imports count; a docstring may explain why the boundary exists."""
    offenders = []
    for path in sorted(ADDON.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith("asset_mania_") for name in names):
                offenders.append((path.relative_to(ROOT).as_posix(), names))
    assert offenders == []


def test_every_gpl_source_file_carries_the_spdx_header() -> None:
    for path in sorted((ADDON / "src").rglob("*.py")):
        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert header == "# SPDX-License-Identifier: GPL-3.0-or-later", path


def test_the_gpl_tree_tracks_the_full_license_text() -> None:
    license_text = (ADDON / "LICENSE").read_text(encoding="utf-8")
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
    assert "Version 3, 29 June 2007" in license_text
    assert len(license_text) > 30_000
    assert "blender-addon/LICENSE" in _tracked()


def test_the_gpl_tree_is_not_a_uv_workspace_member() -> None:
    workspace = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "blender-addon" not in workspace


def test_the_gpl_archive_is_inventoried_in_the_notices() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "`blender-addon/LICENSE`" in notices
    assert "GPL-3.0-or-later" in notices


# --- The checker detects each breach ------------------------------------------


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """A minimal tracked tree the checker can run against."""
    root = tmp_path / "clone"
    (root / "blender-addon" / "src" / "asset_mania_blender").mkdir(parents=True)
    (root / "packages" / "cli" / "src" / "asset_mania").mkdir(parents=True)
    (root / "scripts").mkdir()

    (root / "scripts" / "check_license_boundary.py").write_text(
        CHECKER.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text("[project]\nname = 'workspace'\n", encoding="utf-8")
    (root / "blender-addon" / "LICENSE").write_text(
        "GNU GENERAL PUBLIC LICENSE\nVersion 3, 29 June 2007\n", encoding="utf-8"
    )
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "# Notices\n\n- `blender-addon/LICENSE` — GPL-3.0-or-later.\n", encoding="utf-8"
    )
    (root / "blender-addon" / "src" / "asset_mania_blender" / "worker.py").write_text(
        "# SPDX-License-Identifier: GPL-3.0-or-later\nimport bpy\n", encoding="utf-8"
    )
    (root / "packages" / "cli" / "src" / "asset_mania" / "cli.py").write_text(
        "import json\n", encoding="utf-8"
    )

    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _check(clone: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(clone / "scripts" / "check_license_boundary.py")],
        cwd=clone,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_fixture_clone_is_clean(clone: Path) -> None:
    completed = _check(clone)
    assert completed.returncode == 0, completed.stdout


@pytest.mark.parametrize("module", GPL_MODULES)
def test_a_blender_import_outside_the_addon_is_detected(clone: Path, module: str) -> None:
    target = clone / "packages" / "cli" / "src" / "asset_mania" / "leak.py"
    target.write_text(f"import {module}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)

    completed = _check(clone)
    assert completed.returncode == 1
    assert "GPL_IMPORT_OUTSIDE_ADDON" in completed.stdout


def test_a_from_import_of_a_blender_module_is_detected(clone: Path) -> None:
    target = clone / "packages" / "cli" / "src" / "asset_mania" / "leak.py"
    target.write_text("from mathutils import Matrix\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)

    completed = _check(clone)
    assert completed.returncode == 1
    assert "GPL_IMPORT_OUTSIDE_ADDON" in completed.stdout


def test_an_apache_import_inside_the_addon_is_detected(clone: Path) -> None:
    target = clone / "blender-addon" / "src" / "asset_mania_blender" / "linked.py"
    target.write_text(
        "# SPDX-License-Identifier: GPL-3.0-or-later\nimport asset_mania_contracts\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)

    completed = _check(clone)
    assert completed.returncode == 1
    assert "APACHE_IMPORT_INSIDE_ADDON" in completed.stdout


def test_a_missing_gpl_header_is_detected(clone: Path) -> None:
    target = clone / "blender-addon" / "src" / "asset_mania_blender" / "unmarked.py"
    target.write_text("import bpy\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)

    completed = _check(clone)
    assert completed.returncode == 1
    assert "GPL_HEADER_MISSING" in completed.stdout


def test_a_missing_gpl_license_is_detected(clone: Path) -> None:
    subprocess.run(["git", "rm", "-qf", "blender-addon/LICENSE"], cwd=clone, check=True)
    completed = _check(clone)
    assert completed.returncode == 1
    assert "GPL_LICENSE_MISSING" in completed.stdout


def test_a_missing_archive_inventory_entry_is_detected(clone: Path) -> None:
    (clone / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)

    completed = _check(clone)
    assert completed.returncode == 1
    assert "GPL_ARCHIVE_NOT_INVENTORIED" in completed.stdout


# --- Built archives -----------------------------------------------------------


def test_a_gpl_file_inside_an_apache_wheel_is_detected(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    wheel = distribution / "asset_mania_contracts-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("asset_mania_contracts/__init__.py", "")
        archive.writestr("asset_mania_blender/worker.py", "import bpy\n")

    completed = _run(str(distribution))
    assert completed.returncode == 1
    assert "GPL_FILE_IN_APACHE_ARCHIVE" in completed.stdout


def test_a_gpl_file_inside_an_apache_sdist_is_detected(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    payload = tmp_path / "worker.py"
    payload.write_text("import bpy\n", encoding="utf-8")
    sdist = distribution / "asset_mania_cli-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="asset_mania_cli-0.1.0/blender-addon/worker.py")

    completed = _run(str(distribution))
    assert completed.returncode == 1
    assert "GPL_FILE_IN_APACHE_ARCHIVE" in completed.stdout


def test_a_clean_distribution_passes(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    wheel = distribution / "asset_mania_contracts-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("asset_mania_contracts/__init__.py", "")

    completed = _run(str(distribution))
    assert completed.returncode == 0, completed.stdout


def test_the_gpl_wheel_itself_is_not_flagged(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    distribution.mkdir()
    wheel = distribution / "asset_mania_blender_addon-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("asset_mania_blender/worker.py", "import bpy\n")

    completed = _run(str(distribution))
    assert completed.returncode == 0, completed.stdout


def test_a_missing_distribution_directory_is_reported() -> None:
    completed = _run("/nonexistent/dist")
    assert completed.returncode == 1
    assert "DISTRIBUTION_MISSING" in completed.stdout
