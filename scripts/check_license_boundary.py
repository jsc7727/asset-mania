"""Enforce the Apache/GPL boundary.

Three properties must hold, and each one fails the build on its own:

1. `bpy` and `mathutils` are imported only inside `blender-addon/`.
2. `blender-addon/` never imports an `asset_mania_*` Apache package.
3. No Apache wheel or sdist contains a file from `blender-addon/`, and the GPL tree
   carries its own license, SPDX headers, and archive inventory entry.
"""

import ast
import re
import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
GPL_TREE = PurePosixPath("blender-addon")
GPL_SOURCE = GPL_TREE / "src"
GPL_MODULES = frozenset({"bpy", "mathutils", "bpy_extras", "bmesh", "gpu", "aud"})
GPL_PACKAGES = frozenset({"asset_mania_blender"})
APACHE_PACKAGES = frozenset(
    {
        "asset_mania",
        "asset_mania_contracts",
        "asset_mania_pipeline",
        "asset_mania_blender_client",
        "asset_mania_provider_openai",
        "asset_mania_engine_dad3dheads",
        "asset_mania_engine_deca",
        "asset_mania_engine_mica",
        "asset_mania_engine_triposr",
    }
)
APACHE_PACKAGE_ROOT = PurePosixPath("packages")
SPDX_HEADER = "# SPDX-License-Identifier: GPL-3.0-or-later"
GPL_LICENSE = GPL_TREE / "LICENSE"
NOTICES = "THIRD_PARTY_NOTICES.md"


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code} {self.path}: {self.message}"


def tracked_files() -> list[PurePosixPath]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [PurePosixPath(entry) for entry in completed.stdout.split("\0") if entry]


def _imported_roots(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _inside(path: PurePosixPath, tree: PurePosixPath) -> bool:
    return path.is_relative_to(tree)


def check_sources(paths: list[PurePosixPath]) -> list[Finding]:
    findings: list[Finding] = []
    for relative in paths:
        if relative.suffix != ".py":
            continue
        try:
            source = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            findings.append(Finding("SOURCE_UNREADABLE", str(relative), "unreadable Python file"))
            continue

        roots = _imported_roots(source)
        in_gpl = _inside(relative, GPL_TREE)

        gpl_imports = sorted(roots & GPL_MODULES)
        if gpl_imports and not in_gpl:
            findings.append(
                Finding(
                    "GPL_IMPORT_OUTSIDE_ADDON",
                    str(relative),
                    f"imports {gpl_imports} outside {GPL_TREE}/",
                )
            )

        apache_imports = sorted(roots & APACHE_PACKAGES)
        if apache_imports and in_gpl:
            findings.append(
                Finding(
                    "APACHE_IMPORT_INSIDE_ADDON",
                    str(relative),
                    f"GPL code imports Apache packages {apache_imports}",
                )
            )

        if in_gpl and _inside(relative, GPL_SOURCE) and SPDX_HEADER not in source:
            findings.append(
                Finding("GPL_HEADER_MISSING", str(relative), f"missing {SPDX_HEADER!r}")
            )

    return findings


def check_inventory(paths: list[PurePosixPath]) -> list[Finding]:
    findings: list[Finding] = []
    if GPL_LICENSE not in set(paths):
        findings.append(
            Finding("GPL_LICENSE_MISSING", str(GPL_LICENSE), "the GPL tree tracks no license")
        )

    notices_path = ROOT / NOTICES
    try:
        notices = notices_path.read_text(encoding="utf-8")
    except OSError:
        findings.append(Finding("NOTICES_UNREADABLE", NOTICES, "notices are unreadable"))
        return findings

    inventoried = re.search(rf"`{re.escape(str(GPL_TREE))}[^`]*`", notices) is not None
    if not inventoried:
        findings.append(
            Finding(
                "GPL_ARCHIVE_NOT_INVENTORIED",
                NOTICES,
                f"the separately distributed {GPL_TREE}/ archive is not inventoried",
            )
        )
    return findings


def _archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def check_archives(distribution: Path) -> list[Finding]:
    """Prove no Apache archive carries a GPL file."""
    findings: list[Finding] = []
    for path in sorted(distribution.glob("*")):
        if path.suffix not in (".whl", ".gz"):
            continue
        if path.name.startswith("asset_mania_blender"):
            continue
        try:
            members = _archive_members(path)
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
            findings.append(Finding("ARCHIVE_UNREADABLE", path.name, str(error)))
            continue

        leaked = sorted(
            member
            for member in members
            if "asset_mania_blender" in member or str(GPL_TREE) in member
        )
        if leaked:
            findings.append(Finding("GPL_FILE_IN_APACHE_ARCHIVE", path.name, f"contains {leaked}"))
    return findings


def main(argv: list[str]) -> int:
    paths = tracked_files()
    findings = check_sources(paths) + check_inventory(paths)

    if argv:
        distribution = Path(argv[0])
        if not distribution.is_dir():
            findings.append(
                Finding("DISTRIBUTION_MISSING", argv[0], "the distribution directory is absent")
            )
        else:
            findings += check_archives(distribution)

    for finding in findings:
        print(finding.render())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
