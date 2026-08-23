import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKER = ROOT / "scripts" / "check_release.py"

import scripts.check_release as release_checker
from scripts.check_release import Finding, check_release, main


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=True,
    )


def _write(root: Path, relative: str, content: str | bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _clean_tree(tmp_path: Path) -> Path:
    _write(tmp_path, ".gitignore", ".asset-mania/\n.cache/\n")
    _write(tmp_path, "README.md", "# Example\n\n[Guide](docs/guide.md)\n")
    _write(tmp_path, "docs/guide.md", "# Guide\n")
    _write(tmp_path, "THIRD_PARTY_NOTICES.md", "# Third-Party Notices\n\nNone.\n")
    _write(
        tmp_path,
        "tests/fixtures/PROVENANCE.md",
        "# Fixture provenance\n\nBinary fixtures are generated at test runtime.\n",
    )
    schema = '{"$schema":"https://json-schema.org/draft/2020-12/schema"}\n'
    _write(
        tmp_path,
        "packages/contracts/src/asset_mania_contracts/schema/manifest-v1.schema.json",
        schema,
    )
    _write(
        tmp_path,
        "skills/asset-mania/references/manifest-v1.schema.json",
        schema,
    )
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "add", ".")
    return tmp_path


def _track(root: Path, relative: str, content: str | bytes) -> None:
    _write(root, relative, content)
    _git(root, "add", "--force", "--", relative)


def _findings_with_code(root: Path, code: str) -> list[Finding]:
    return [finding for finding in check_release(root) if finding.code == code]


@pytest.mark.parametrize(
    "relative",
    [
        ".env",
        ".env.local",
        ".env.production",
        "private/token.json",
        "private/cookies.txt",
        "weights/model.safetensors",
        "artifacts/model.trcd",
        "artifacts/projection.npz",
        "artifacts/face.npy",
        "artifacts/mica.tar",
        "artifacts/deca_model.tar",
        "artifacts/generic_model.pkl",
        "artifacts/identity.npy",
        "artifacts/aligned-face.png",
        "artifacts/mica-clay.glb",
        "models/insightface.onnx",
        ".dad_checkpoints/model.trcd",
        "model_training/model/static/flame.pkl",
        ".cache/download.bin",
        "cache/download.bin",
        "build/cache/download.bin",
    ],
)
def test_forbidden_tracked_paths_are_reported(tmp_path: Path, relative: str) -> None:
    root = _clean_tree(tmp_path)
    _track(root, relative, b"not for publication")

    findings = _findings_with_code(root, "FORBIDDEN_TRACKED_PATH")

    assert [finding.path for finding in findings] == ["private-release-entry"]


def test_absolute_home_string_is_reported_without_echoing_it(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    private_path = str(Path.home() / "private-source.png")
    _track(root, "notes.txt", f"local source: {private_path}\n")

    findings = _findings_with_code(root, "ABSOLUTE_HOME_PATH")

    assert findings == [
        Finding(
            code="ABSOLUTE_HOME_PATH",
            path="notes.txt",
            message="tracked text contains an absolute home path",
        )
    ]
    assert private_path not in findings[0].message


@pytest.mark.parametrize(
    ("relative", "content"),
    [
        ("standing-consent.json", "{}\n"),
        (
            "records/authorization.json",
            '{"schema_id":"asset-mania/local-face-standing-consent",'
            '"source_sha256":"' + "a" * 64 + '","private_path":"PRIVATE"}\n',
        ),
    ],
)
def test_tracked_standing_consent_is_rejected_without_private_details(
    tmp_path: Path, relative: str, content: str
) -> None:
    root = _clean_tree(tmp_path)
    private_path = str(Path.home() / "face" / "standing-consent.json")
    source_digest = "a" * 64
    _track(root, relative, content.replace("PRIVATE", private_path.replace("\\", "\\\\")))

    findings = _findings_with_code(root, "STANDING_CONSENT_TRACKED")

    assert len(findings) == 1
    assert findings[0].message == "tracked local face standing consent is forbidden"
    assert private_path not in findings[0].message
    assert source_digest not in findings[0].message


def test_suspicious_consent_filename_is_rejected_even_when_entry_is_not_regular(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = release_checker.PurePosixPath("private/standing-consent.json")
    monkeypatch.setattr(release_checker, "_tracked_paths", lambda _root: ([relative], None))
    monkeypatch.setattr(release_checker, "_safe_regular_file", lambda _root, _path: None)

    findings = _findings_with_code(tmp_path, "STANDING_CONSENT_TRACKED")

    assert findings == [
        Finding(
            code="STANDING_CONSENT_TRACKED",
            path="private-standing-consent-record",
            message="tracked local face standing consent is forbidden",
        )
    ]


def test_standing_consent_cli_diagnostic_is_fully_sanitized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _clean_tree(tmp_path)
    private_basename = "subject-standing-consent.json"
    private_path = str(Path.home() / "face" / private_basename)
    source_digest = "a" * 64
    raw_content = "explicit-private-consent-content"
    _track(
        root,
        f"records/{private_basename}",
        private_path + source_digest + raw_content,
    )

    assert main([str(root)]) == 1

    rendered = capsys.readouterr().out
    assert rendered == (
        "STANDING_CONSENT_TRACKED private-standing-consent-record: "
        "tracked local face standing consent is forbidden\n"
    )
    for private_value in (private_path, private_basename, source_digest, raw_content):
        assert private_value not in rendered


@pytest.mark.parametrize(
    ("relative", "content", "expected"),
    [
        (
            ".asset-mania/subject-standing-consent.json",
            "private standing consent bytes",
            (
                "STANDING_CONSENT_TRACKED private-standing-consent-record: "
                "tracked local face standing consent is forbidden\n"
            ),
        ),
        (
            ".asset-mania/innocuous.json",
            '{"schema_id":"asset-mania/local-face-standing-consent",'
            '"source_sha256":"' + "a" * 64 + '","raw":"PRIVATE-CONTENT"}\n',
            (
                "STANDING_CONSENT_TRACKED private-standing-consent-record: "
                "tracked local face standing consent is forbidden\n"
            ),
        ),
        (
            ".asset-mania/ordinary-artifact.json",
            "PRIVATE-CONTENT",
            "FORBIDDEN_TRACKED_PATH private-release-entry: tracked release path is forbidden\n",
        ),
    ],
)
def test_private_release_cli_findings_do_not_render_private_entry_details(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    relative: str,
    content: str,
    expected: str,
) -> None:
    root = _clean_tree(tmp_path)
    _track(root, relative, content)

    assert main([str(root)]) == 1

    rendered = capsys.readouterr().out
    assert rendered == expected
    for private_value in (
        relative,
        Path(relative).name,
        "a" * 64,
        "PRIVATE-CONTENT",
    ):
        assert private_value not in rendered


@pytest.mark.parametrize("name", ["api_" + "token", "session_" + "cookie"])
def test_secret_like_assignments_are_reported_without_echoing_values(
    tmp_path: Path, name: str
) -> None:
    root = _clean_tree(tmp_path)
    secret = "Ab9_" * 8
    _track(root, "settings.txt", f"{name}={secret}\n")

    findings = _findings_with_code(root, "COMMON_SECRET_PATTERN")

    assert [finding.path for finding in findings] == ["settings.txt"]
    assert secret not in findings[0].message


@pytest.mark.parametrize("suffix", [".png", ".bin"])
def test_binary_fixture_requires_a_provenance_entry(tmp_path: Path, suffix: str) -> None:
    root = _clean_tree(tmp_path)
    relative = f"tests/fixtures/example{suffix}"
    _track(root, relative, b"\x00synthetic binary")

    findings = _findings_with_code(root, "FIXTURE_PROVENANCE_MISSING")

    assert [finding.path for finding in findings] == [relative]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("archive.zip", b"PK\x03\x04synthetic archive"),
        ("opaque.payload", b"\x00synthetic opaque data"),
        ("mislabelled.txt", b"text prefix\x00binary remainder"),
    ],
)
def test_archive_and_opaque_binary_fixtures_require_provenance(
    tmp_path: Path, name: str, content: bytes
) -> None:
    root = _clean_tree(tmp_path)
    relative = f"tests/fixtures/{name}"
    _track(root, relative, content)

    findings = _findings_with_code(root, "FIXTURE_PROVENANCE_MISSING")

    assert [finding.path for finding in findings] == [relative]


def test_longer_fixture_path_does_not_satisfy_provenance(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    relative = "tests/fixtures/example.bin"
    _track(root, relative, b"\x00synthetic binary")
    _track(
        root,
        "tests/fixtures/PROVENANCE.md",
        "# Fixture provenance\n\n- `tests/fixtures/example.bin.backup` - generated locally\n",
    )

    findings = _findings_with_code(root, "FIXTURE_PROVENANCE_MISSING")

    assert [finding.path for finding in findings] == [relative]


def test_exact_fixture_inventory_entry_satisfies_provenance(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    relative = "tests/fixtures/example.bin"
    _track(root, relative, b"\x00synthetic binary")
    _track(
        root,
        "tests/fixtures/PROVENANCE.md",
        f"# Fixture provenance\n\n- `{relative}` - generated locally\n",
    )

    assert _findings_with_code(root, "FIXTURE_PROVENANCE_MISSING") == []


def test_third_party_file_requires_a_notice_entry(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    _track(root, "third_party/example.py", "VALUE = 1\n")

    findings = _findings_with_code(root, "THIRD_PARTY_NOTICE_MISSING")

    assert [finding.path for finding in findings] == ["third_party/example.py"]


def test_longer_third_party_path_does_not_satisfy_notices(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    relative = "third_party/example.py"
    _track(root, relative, "VALUE = 1\n")
    _track(
        root,
        "THIRD_PARTY_NOTICES.md",
        "# Third-Party Notices\n\n- `third_party/example.py.backup` - Apache-2.0\n",
    )

    findings = _findings_with_code(root, "THIRD_PARTY_NOTICE_MISSING")

    assert [finding.path for finding in findings] == [relative]


def test_exact_third_party_inventory_entry_satisfies_notices(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    relative = "third_party/example.py"
    _track(root, relative, "VALUE = 1\n")
    _track(
        root,
        "THIRD_PARTY_NOTICES.md",
        f"# Third-Party Notices\n\n- `{relative}` - Apache-2.0\n",
    )

    assert _findings_with_code(root, "THIRD_PARTY_NOTICE_MISSING") == []


def test_broken_relative_markdown_link_is_reported(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    _track(root, "docs/broken.md", "[Missing](missing-page.md)\n")

    findings = _findings_with_code(root, "MARKDOWN_LINK_BROKEN")

    assert [finding.path for finding in findings] == ["docs/broken.md"]


def test_skill_schema_must_match_the_contract_schema(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    _track(
        root,
        "skills/asset-mania/references/manifest-v1.schema.json",
        '{"title":"stale copy"}\n',
    )

    findings = _findings_with_code(root, "SKILL_SCHEMA_MISMATCH")

    assert [finding.path for finding in findings] == [
        "skills/asset-mania/references/manifest-v1.schema.json"
    ]


@pytest.mark.parametrize(
    ("relative", "expected_code", "expected_message"),
    [
        (
            "packages/contracts/src/asset_mania_contracts/schema/manifest-v1.schema.json",
            "CONTRACT_SCHEMA_UNAVAILABLE",
            "contracts schema is missing or unreadable",
        ),
        (
            "skills/asset-mania/references/manifest-v1.schema.json",
            "SKILL_SCHEMA_UNAVAILABLE",
            "Skill schema is missing or unreadable",
        ),
    ],
)
@pytest.mark.parametrize("unavailable_state", ["missing", "unreadable"])
def test_schema_parity_fails_closed_when_either_schema_is_unavailable(
    tmp_path: Path,
    relative: str,
    expected_code: str,
    expected_message: str,
    unavailable_state: str,
) -> None:
    root = _clean_tree(tmp_path)
    schema_path = root / relative
    if unavailable_state == "missing":
        schema_path.unlink()
    else:
        schema_path.chmod(0)

    try:
        findings = check_release(root)
    finally:
        if schema_path.exists():
            schema_path.chmod(0o600)

    assert findings == [
        Finding(
            code=expected_code,
            path=relative,
            message=expected_message,
        )
    ]


def test_minimal_clean_tree_has_no_findings(tmp_path: Path) -> None:
    assert check_release(_clean_tree(tmp_path)) == []


def test_checker_does_not_match_its_own_home_path_pattern(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    _track(root, "scripts/check_release.py", CHECKER.read_text(encoding="utf-8"))

    assert _findings_with_code(root, "ABSOLUTE_HOME_PATH") == []


def test_ignored_run_outputs_and_external_symlink_targets_are_not_opened(tmp_path: Path) -> None:
    root = _clean_tree(tmp_path)
    _write(root, ".asset-mania/report.md", "[Missing](private-file.md)\n")
    external = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    external.write_text(str(Path.home() / "private-source.png"), encoding="utf-8")
    linked = root / "linked-private.txt"
    linked.symlink_to(external)
    _git(root, "add", "linked-private.txt")
    try:
        assert check_release(root) == []
    finally:
        external.unlink()


def test_command_prints_sorted_findings_and_returns_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _clean_tree(tmp_path)
    _track(root, "z-token.json", b"private")
    _track(root, "a.env", b"private")

    exit_code = main([os.fspath(root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.err == ""
    assert captured.out.splitlines() == sorted(captured.out.splitlines())
    assert captured.out.splitlines() == [
        "FORBIDDEN_TRACKED_PATH private-release-entry: tracked release path is forbidden",
        "FORBIDDEN_TRACKED_PATH private-release-entry: tracked release path is forbidden",
    ]


def test_command_is_silent_and_returns_zero_for_a_clean_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main([os.fspath(_clean_tree(tmp_path))])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
