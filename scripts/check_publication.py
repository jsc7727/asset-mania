"""Publication gates beyond the release checker.

`check_release.py` guards the tracked tree. This script adds the checks that only matter
when something is about to be published, and each one exists because the failure it catches
would be embarrassing *after* the fact:

* an opaque binary or model weight slipping into the tree or an archive;
* bytecode or a model format arriving where source is expected;
* a capability claim that no evidence supports;
* a private sample name reaching a public file;
* an unlisted runtime dependency or external tool.

Every tracked file is examined, not only test fixtures, and every member of every built
archive is examined too.
"""

import subprocess
import sys
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]

#: Extensions that are never source and never belong in a distribution.
REJECTED_SUFFIXES = frozenset(
    {
        ".pyc",
        ".pyo",
        ".pyd",
        ".so",
        ".dylib",
        ".dll",
        ".a",
        ".o",
        ".tflite",
        ".onnx",
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".trcd",
        ".pkl",
        ".npy",
        ".npz",
        ".gguf",
        ".bin",
        ".weights",
        ".h5",
        ".pb",
        ".blend",
        ".blend1",
        ".fbx",
        ".glb",
        ".gltf",
        ".exr",
        ".hdr",
    }
)
#: Text extensions and exact files that are allowed to be tracked as-is.
ALLOWED_BINARY_PATHS = frozenset({"blender-addon/LICENSE"})

#: Public-facing files whose capability claims must be qualified. Design and plan documents
#: are excluded on purpose: they discuss what a claim would require, and forbid it, which is
#: the opposite of asserting it.
PUBLIC_CLAIM_FILES = ("README.md", "skills/asset-mania/SKILL.md")
#: A guarded claim must travel with at least one of these qualifiers.
#: A capability row may only leave `Planned` when a recorded run backs it. Each entry maps a
#: README row label to the evidence phrase that must accompany a non-`Planned` state.
PLANNED_CAPABILITIES = {
    # This row left `Planned` once a reconstruction actually ran, so the phrase it must carry
    # changed with it. The previous requirement -- "no engine is cleared, downloaded, or
    # executed" -- became false the moment an engine was downloaded and executed, and leaving
    # it in place would have turned this gate into a check that enforced a stale claim.
    #
    # What still needs saying is the part that has not changed: an engine running on a
    # developer's machine is not an engine cleared for a user's, and nothing here ships a
    # weight or accepts a licence on anyone's behalf.
    "Generic image to 3D": "clearance is user-issued and unissued here",
}

GUARDED_CLAIMS = {
    "live-verified": (
        "no live call has ever been made",
        "has never made a live call",
        "never made a live call",
        "do not describe it as",
        "is forbidden",
    ),
    "production-ready": ("pre-alpha",),
}

#: The private deny inventory is a local, gitignored file of literals or content hashes -- one
#: per line, `#` for comments. It is deliberately never tracked: the whole point is that those
#: strings must not exist in this repository. When it is absent there is nothing to scan, and
#: this checker says so rather than pretending it verified something.
DENY_INVENTORY = ROOT / ".asset-mania" / "deny-inventory.txt"
SCANNED_SUFFIXES = frozenset({".md", ".toml", ".yml", ".yaml", ".json", ".py", ".cfg", ".txt"})


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code} {self.path}: {self.message}"


def tracked_files() -> list[PurePosixPath]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [PurePosixPath(entry) for entry in completed.stdout.split("\0") if entry]


def _is_text(path: Path) -> bool:
    try:
        path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return True


def check_tracked_binaries(paths: list[PurePosixPath]) -> list[Finding]:
    """No opaque binary, bytecode, or model weight may be tracked."""
    findings: list[Finding] = []
    for relative in paths:
        text = relative.as_posix()
        if text in ALLOWED_BINARY_PATHS:
            continue
        if relative.suffix.lower() in REJECTED_SUFFIXES:
            findings.append(
                Finding(
                    "REJECTED_BINARY_TRACKED",
                    text,
                    f"{relative.suffix} is never tracked; generate it at runtime instead",
                )
            )
            continue
        absolute = ROOT / relative
        if absolute.is_file() and not _is_text(absolute):
            findings.append(
                Finding(
                    "UNPROVENANCED_BINARY",
                    text,
                    "an opaque binary needs an exact provenance and license entry",
                )
            )
    return findings


def check_capability_claims() -> list[Finding]:
    """A guarded claim in a public-facing file must travel with a qualifier."""
    findings: list[Finding] = []
    for relative in PUBLIC_CLAIM_FILES:
        absolute = ROOT / relative
        if not absolute.is_file():
            findings.append(Finding("PUBLIC_FILE_MISSING", relative, "expected to exist"))
            continue
        text = absolute.read_text(encoding="utf-8").lower()
        for claim, qualifiers in sorted(GUARDED_CLAIMS.items()):
            if claim in text and not any(phrase in text for phrase in qualifiers):
                findings.append(
                    Finding(
                        "STALE_CAPABILITY_CLAIM",
                        relative,
                        f"{claim!r} appears without any of {list(qualifiers)}",
                    )
                )
    return findings


def load_deny_inventory() -> list[str]:
    """Read the local deny inventory, if a maintainer has one."""
    if not DENY_INVENTORY.is_file():
        return []
    entries = []
    for line in DENY_INVENTORY.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.append(stripped)
    return entries


def check_planned_capabilities() -> list[Finding]:
    """A `Planned` row cannot quietly become `Available`.

    The check is deliberately about the *evidence phrase*, not the word `Planned`: a row may
    legitimately change state, but only together with the sentence that says what backs it.
    """
    findings: list[Finding] = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for label, evidence in sorted(PLANNED_CAPABILITIES.items()):
        row = next((line for line in readme.splitlines() if line.startswith(f"| {label} |")), None)
        if row is None:
            findings.append(Finding("CAPABILITY_ROW_MISSING", "README.md", f"no row for {label!r}"))
            continue
        if "Planned" in row:
            continue
        if evidence.lower() not in readme.lower():
            findings.append(
                Finding(
                    "UNBACKED_CAPABILITY_CLAIM",
                    "README.md",
                    f"{label!r} left Planned without the evidence phrase {evidence!r}",
                )
            )
    return findings


def check_private_sample_names(paths: list[PurePosixPath]) -> tuple[list[Finding], str]:
    """Scan every tracked text file against the local deny inventory.

    The result is sanitized: a match reports the file and the number of entries that hit,
    never the entry itself, because printing it would put the private string into a log.
    """
    entries = load_deny_inventory()
    if not entries:
        return [], (
            f"deny inventory absent at {DENY_INVENTORY.relative_to(ROOT).as_posix()}; "
            "0 files scanned, no claim made"
        )

    findings: list[Finding] = []
    scanned = 0
    for relative in paths:
        if relative.suffix.lower() not in SCANNED_SUFFIXES:
            continue
        absolute = ROOT / relative
        if not absolute.is_file():
            continue
        try:
            text = absolute.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        scanned += 1
        hits = sum(1 for entry in entries if entry in text)
        if hits:
            findings.append(
                Finding(
                    "PRIVATE_SAMPLE_NAME",
                    relative.as_posix(),
                    f"{hits} deny-inventory entries matched",
                )
            )
    return (
        findings,
        f"{scanned} files scanned against {len(entries)} deny entries; {len(findings)} matched",
    )


def check_declared_tools(paths: list[PurePosixPath]) -> list[Finding]:
    """Every inventoried external tool must be named in the notices."""
    findings: list[Finding] = []
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for relative in paths:
        if relative.parent.as_posix() != "tools" or relative.suffix != ".json":
            continue
        if relative.as_posix() not in notices:
            findings.append(
                Finding(
                    "UNLISTED_TOOL",
                    relative.as_posix(),
                    "an inventoried tool must be named in THIRD_PARTY_NOTICES.md",
                )
            )
    return findings


def _members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def check_archives(distribution: Path) -> list[Finding]:
    """No archive member may be bytecode, a weight, or an opaque asset."""
    findings: list[Finding] = []
    for path in sorted(distribution.glob("*")):
        if path.suffix not in (".whl", ".gz"):
            continue
        try:
            members = _members(path)
        except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
            findings.append(Finding("ARCHIVE_UNREADABLE", path.name, str(error)))
            continue
        for member in members:
            suffix = PurePosixPath(member).suffix.lower()
            if suffix in REJECTED_SUFFIXES:
                findings.append(
                    Finding(
                        "REJECTED_ARCHIVE_MEMBER",
                        path.name,
                        f"{member} has the rejected extension {suffix}",
                    )
                )
    return findings


def main(argv: list[str]) -> int:
    paths = tracked_files()
    private_findings, private_summary = check_private_sample_names(paths)
    findings = (
        check_tracked_binaries(paths)
        + check_capability_claims()
        + check_planned_capabilities()
        + private_findings
        + check_declared_tools(paths)
    )
    if argv:
        distribution = Path(argv[0])
        if not distribution.is_dir():
            findings.append(
                Finding("DISTRIBUTION_MISSING", argv[0], "the distribution directory is absent")
            )
        else:
            findings += check_archives(distribution)

    print(f"deny-inventory scan: {private_summary}")
    for finding in sorted(findings, key=lambda item: (item.code, item.path)):
        print(finding.render())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
