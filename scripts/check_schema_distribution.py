"""Verify the distributed schema copies and the closed distribution list.

Two properties, both cheap and both easy to break by accident:

1. every schema the Skill is supposed to ship is byte-identical to the contracts copy; and
2. the contracts directory holds exactly the known schemas -- a new one must be added to
   the distribution list on purpose, and the internal worker protocol must stay internal.
"""

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIRECTORY = ROOT / "packages" / "contracts" / "src" / "asset_mania_contracts" / "schema"
SKILL_DIRECTORY = ROOT / "skills" / "asset-mania" / "references"

#: Published contracts a Skill consumer may need to read.
DISTRIBUTED = (
    "approval-receipt-v1",
    "conditioning-bundle-v1",
    "engine-clearance-v1",
    "manifest-v1",
    "manifest-v2",
    "provider-evidence-v1",
    "provider-plan-v1",
    "reconstruction-plan-v1",
    "view-v1",
    "workflow-plan-v1",
)
#: An internal worker protocol, never a published artifact, so it is not distributed.
PRIVATE = ("blender-response-v1",)


@dataclass(frozen=True, slots=True)
class Finding:
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"{self.code} {self.path}: {self.message}"


def check() -> list[Finding]:
    findings: list[Finding] = []

    present = sorted(
        path.name.removesuffix(".schema.json") for path in CONTRACT_DIRECTORY.glob("*.schema.json")
    )
    expected = sorted([*DISTRIBUTED, *PRIVATE])
    if present != expected:
        findings.append(
            Finding(
                "SCHEMA_INVENTORY_CHANGED",
                CONTRACT_DIRECTORY.relative_to(ROOT).as_posix(),
                f"expected {expected}, found {present}",
            )
        )

    for name in DISTRIBUTED:
        contract = CONTRACT_DIRECTORY / f"{name}.schema.json"
        skill = SKILL_DIRECTORY / f"{name}.schema.json"
        if not contract.is_file():
            findings.append(
                Finding("CONTRACT_SCHEMA_MISSING", f"{name}.schema.json", "absent from contracts")
            )
            continue
        if not skill.is_file():
            findings.append(
                Finding("SKILL_SCHEMA_MISSING", f"{name}.schema.json", "absent from the Skill")
            )
            continue
        if skill.read_bytes() != contract.read_bytes():
            findings.append(
                Finding(
                    "SKILL_SCHEMA_MISMATCH",
                    f"{name}.schema.json",
                    "the Skill copy is not byte-identical to the contracts copy",
                )
            )

    for name in PRIVATE:
        if (SKILL_DIRECTORY / f"{name}.schema.json").exists():
            findings.append(
                Finding(
                    "PRIVATE_SCHEMA_DISTRIBUTED",
                    f"{name}.schema.json",
                    "an internal worker protocol must not be distributed",
                )
            )

    for name in DISTRIBUTED:
        contract = CONTRACT_DIRECTORY / f"{name}.schema.json"
        if not contract.is_file():
            continue
        try:
            schema = json.loads(contract.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            findings.append(Finding("SCHEMA_UNREADABLE", f"{name}.schema.json", str(error)))
            continue
        if schema.get("$id") != f"https://asset-mania.dev/schemas/{name}.schema.json":
            findings.append(
                Finding("SCHEMA_IDENTIFIER_UNSTABLE", f"{name}.schema.json", "unexpected $id")
            )
        if schema.get("additionalProperties") is not False:
            findings.append(
                Finding("SCHEMA_NOT_CLOSED", f"{name}.schema.json", "not closed at the top level")
            )

    return findings


def main() -> int:
    findings = check()
    for finding in sorted(findings, key=lambda item: (item.code, item.path)):
        print(finding.render())
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
