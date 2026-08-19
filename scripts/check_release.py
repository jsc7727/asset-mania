"""Check tracked release files for privacy, provenance, and contract drift."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

_CONTRACT_SCHEMA = PurePosixPath(
    "packages/contracts/src/asset_mania_contracts/schema/manifest-v1.schema.json"
)
_SKILL_SCHEMA = PurePosixPath("skills/asset-mania/references/manifest-v1.schema.json")
_PROVENANCE = PurePosixPath("tests/fixtures/PROVENANCE.md")
_THIRD_PARTY_NOTICES = PurePosixPath("THIRD_PARTY_NOTICES.md")

_FORBIDDEN_DIRECTORY_NAMES = {
    ".asset-mania",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "datasets",
    "models",
    "weights",
}
_FORBIDDEN_FILE_NAMES = {
    ".coverage",
    ".env",
    "cookie",
    "cookie.jar",
    "cookies",
    "cookies.json",
    "token",
    "token.json",
}
_FORBIDDEN_WEIGHT_SUFFIXES = {".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
_BINARY_FIXTURE_SUFFIXES = {
    ".bin",
    ".blend",
    ".bmp",
    ".gif",
    ".glb",
    ".ico",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
_THIRD_PARTY_DIRECTORIES = {"external", "third-party", "third_party", "vendor"}
_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\((?P<target><[^>]+>|[^\s)]+)(?:\s+[\"'][^)]*[\"'])?\)")
_MACOS_HOME_ROOT = "/" + "Us" + "ers" + "/"
_LINUX_HOME_ROOT = "/" + "ho" + "me" + "/"
_WINDOWS_HOME_ROOT = r"[A-Za-z]:\\" + "Us" + r"ers\\"
_GENERIC_HOME_PATH = re.compile(
    rf"(?:{re.escape(_MACOS_HOME_ROOT)}"
    r"(?!example(?:/|\b)|private-person(?:/|\b))[^/\s]+/|"
    rf"{re.escape(_LINUX_HOME_ROOT)}(?!example(?:/|\b))[^/\s]+/|"
    rf"{_WINDOWS_HOME_ROOT}(?!example(?:\\|\b))[^\\\s]+\\)"
)
_COMMON_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:EC |OPENSSH |PGP |RSA )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)[\"']?(?:api[_-]?token|access[_-]?token|auth[_-]?token|"
        r"session[_-]?cookie|cookie)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_+./=-]{16,}"
    ),
)


@dataclass(frozen=True, order=True)
class Finding:
    """A stable, sortable release-check diagnostic."""

    code: str
    path: str
    message: str


def _tracked_paths(root: Path) -> tuple[list[PurePosixPath], Finding | None]:
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "--cached", "-z"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return [], Finding(
            code="GIT_INDEX_UNAVAILABLE",
            path=".",
            message="unable to enumerate tracked release files",
        )
    paths = [
        PurePosixPath(os.fsdecode(raw_path))
        for raw_path in completed.stdout.split(b"\0")
        if raw_path
    ]
    return sorted(paths, key=lambda path: path.as_posix()), None


def _is_forbidden_path(relative: PurePosixPath) -> bool:
    lowered_parts = tuple(part.lower() for part in relative.parts)
    name = lowered_parts[-1]
    stem = PurePosixPath(name).stem
    return (
        any(part in _FORBIDDEN_DIRECTORY_NAMES for part in lowered_parts[:-1])
        or name in _FORBIDDEN_FILE_NAMES
        or name.endswith(".env")
        or stem == "token"
        or stem.endswith("-token")
        or stem == "cookie"
        or stem == "cookies"
        or PurePosixPath(name).suffix in _FORBIDDEN_WEIGHT_SUFFIXES
    )


def _safe_regular_file(root: Path, relative: PurePosixPath) -> Path | None:
    candidate = root.joinpath(*relative.parts)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
    except OSError:
        return None
    return candidate


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _inventory_text(
    root: Path,
    tracked: set[PurePosixPath],
    relative: PurePosixPath,
) -> str:
    if relative not in tracked:
        return ""
    path = _safe_regular_file(root, relative)
    if path is None:
        return ""
    return _read_text(path) or ""


def _contains_absolute_home(text: str) -> bool:
    home = os.fspath(Path.home())
    separators = (os.sep,) if os.altsep is None else (os.sep, os.altsep)
    if any(f"{home}{separator}" in text for separator in separators):
        return True
    return _GENERIC_HOME_PATH.search(text) is not None


def _contains_common_secret(text: str) -> bool:
    return any(pattern.search(text) is not None for pattern in _COMMON_SECRET_PATTERNS)


def _relative_link_exists(root: Path, source: PurePosixPath, raw_target: str) -> bool:
    target = (
        raw_target[1:-1] if raw_target.startswith("<") and raw_target.endswith(">") else raw_target
    )
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "/")):
        return True
    decoded = unquote(parsed.path)
    if not decoded:
        return True

    normalized = PurePosixPath(os.path.normpath((source.parent / decoded).as_posix()))
    if normalized.is_absolute() or normalized.parts[:1] == ("..",):
        return False

    candidate = root
    for part in normalized.parts:
        candidate = candidate / part
        try:
            if candidate.is_symlink():
                return False
        except OSError:
            return False
    try:
        return candidate.exists()
    except OSError:
        return False


def _markdown_findings(root: Path, relative: PurePosixPath, text: str) -> list[Finding]:
    findings: list[Finding] = []
    for match in _MARKDOWN_LINK.finditer(text):
        if not _relative_link_exists(root, relative, match.group("target")):
            findings.append(
                Finding(
                    code="MARKDOWN_LINK_BROKEN",
                    path=relative.as_posix(),
                    message="relative Markdown link target does not exist",
                )
            )
    return findings


def check_release(root: Path) -> list[Finding]:
    """Return sorted findings for the tracked files under ``root``."""
    root = Path(root).resolve()
    tracked_paths, git_finding = _tracked_paths(root)
    if git_finding is not None:
        return [git_finding]
    tracked = set(tracked_paths)
    findings: list[Finding] = []
    provenance = _inventory_text(root, tracked, _PROVENANCE)
    third_party_notices = _inventory_text(root, tracked, _THIRD_PARTY_NOTICES)

    for relative in tracked_paths:
        if _is_forbidden_path(relative):
            findings.append(
                Finding(
                    code="FORBIDDEN_TRACKED_PATH",
                    path=relative.as_posix(),
                    message="tracked release path is forbidden",
                )
            )
            continue

        path = _safe_regular_file(root, relative)
        if path is None:
            continue

        if (
            relative.parts[:2] == ("tests", "fixtures")
            and relative.suffix.lower() in _BINARY_FIXTURE_SUFFIXES
            and relative.as_posix() not in provenance
        ):
            findings.append(
                Finding(
                    code="FIXTURE_PROVENANCE_MISSING",
                    path=relative.as_posix(),
                    message="tracked binary fixture is absent from tests/fixtures/PROVENANCE.md",
                )
            )

        if (
            any(part.lower() in _THIRD_PARTY_DIRECTORIES for part in relative.parts[:-1])
            and relative.as_posix() not in third_party_notices
        ):
            findings.append(
                Finding(
                    code="THIRD_PARTY_NOTICE_MISSING",
                    path=relative.as_posix(),
                    message="tracked third-party file is absent from THIRD_PARTY_NOTICES.md",
                )
            )

        text = _read_text(path)
        if text is None:
            continue
        if _contains_absolute_home(text):
            findings.append(
                Finding(
                    code="ABSOLUTE_HOME_PATH",
                    path=relative.as_posix(),
                    message="tracked text contains an absolute home path",
                )
            )
        if _contains_common_secret(text):
            findings.append(
                Finding(
                    code="COMMON_SECRET_PATTERN",
                    path=relative.as_posix(),
                    message="tracked text contains a common secret pattern",
                )
            )
        if relative.suffix.lower() == ".md":
            findings.extend(_markdown_findings(root, relative, text))

    contract_path = _safe_regular_file(root, _CONTRACT_SCHEMA)
    skill_path = _safe_regular_file(root, _SKILL_SCHEMA)
    if contract_path is not None and skill_path is not None:
        try:
            schemas_match = contract_path.read_bytes() == skill_path.read_bytes()
        except OSError:
            schemas_match = True
        if not schemas_match:
            findings.append(
                Finding(
                    code="SKILL_SCHEMA_MISMATCH",
                    path=_SKILL_SCHEMA.as_posix(),
                    message="Skill schema differs from the contracts schema",
                )
            )

    return sorted(set(findings))


def main(arguments: list[str] | None = None) -> int:
    """Print sorted findings and return a shell-friendly status."""
    arguments = sys.argv[1:] if arguments is None else arguments
    if len(arguments) > 1:
        print("usage: check_release.py [root]", file=sys.stderr)
        return 2
    root = Path(arguments[0]) if arguments else Path.cwd()
    findings = check_release(root)
    for finding in findings:
        print(f"{finding.code} {finding.path}: {finding.message}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
