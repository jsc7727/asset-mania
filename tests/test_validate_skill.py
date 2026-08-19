import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_skill.py"


def _write_valid_skill(parent: Path, name: str = "example-skill") -> Path:
    skill = parent / name
    (skill / "agents").mkdir(parents=True)
    (skill / "references").mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Use when a repository fixture needs structural validation.\n"
        "---\n\n"
        "# Example Skill\n\n"
        "Read [the contract](references/contract.md) when contract details are needed.\n"
    )
    (skill / "agents" / "openai.yaml").write_text('interface:\n  display_name: "Example Skill"\n')
    (skill / "references" / "contract.md").write_text("# Contract\n")
    return skill


def _run_validator(skill: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), str(skill)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_skill_has_no_findings(tmp_path: Path) -> None:
    completed = _run_validator(_write_valid_skill(tmp_path))

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda skill: (skill / "SKILL.md").write_text("# Missing frontmatter\n"),
            "SKILL.md: missing YAML frontmatter",
        ),
        (
            lambda skill: (skill / "SKILL.md").write_text(
                (skill / "SKILL.md")
                .read_text()
                .replace("name: example-skill", "name: another-skill")
            ),
            "SKILL.md: name 'another-skill' does not match folder 'example-skill'",
        ),
        (
            lambda skill: (skill / "SKILL.md").write_text(
                (skill / "SKILL.md").read_text().replace("name: example-skill\n", "")
            ),
            "SKILL.md: missing name",
        ),
        (
            lambda skill: (skill / "SKILL.md").write_text(
                (skill / "SKILL.md")
                .read_text()
                .replace(
                    "description: Use when a repository fixture needs structural validation.\n",
                    "",
                )
            ),
            "SKILL.md: missing description",
        ),
        (
            lambda skill: (skill / "SKILL.md").write_text(
                (skill / "SKILL.md").read_text() + "\n[TODO: finish this section]\n"
            ),
            "SKILL.md: unfinished scaffold marker '[TODO:'",
        ),
        (
            lambda skill: (skill / "agents" / "openai.yaml").unlink(),
            "agents/openai.yaml: missing",
        ),
        (
            lambda skill: (skill / "SKILL.md").write_text(
                (skill / "SKILL.md").read_text().replace("references/contract.md", "contract")
            ),
            "references/contract.md: not discoverable from SKILL.md",
        ),
    ],
)
def test_validator_reports_stable_structural_findings(
    tmp_path: Path, mutate, expected: str
) -> None:
    skill = _write_valid_skill(tmp_path)
    mutate(skill)

    completed = _run_validator(skill)

    assert completed.returncode == 1
    assert completed.stdout.splitlines() == [expected]
    assert completed.stderr == ""


def test_validator_sorts_multiple_findings(tmp_path: Path) -> None:
    skill = _write_valid_skill(tmp_path)
    (skill / "agents" / "openai.yaml").unlink()
    (skill / "SKILL.md").write_text(
        (skill / "SKILL.md").read_text().replace("references/contract.md", "contract")
    )

    completed = _run_validator(skill)

    assert completed.returncode == 1
    assert completed.stdout.splitlines() == [
        "agents/openai.yaml: missing",
        "references/contract.md: not discoverable from SKILL.md",
    ]
