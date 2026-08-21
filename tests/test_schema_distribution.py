"""The Skill's distributed schema copies must stay byte-identical to the contracts."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA_DIR = ROOT / "packages" / "contracts" / "src" / "asset_mania_contracts" / "schema"
SKILL_REFERENCE_DIR = ROOT / "skills" / "asset-mania" / "references"

DISTRIBUTED_SCHEMAS = [
    "approval-receipt-v1",
    "conditioning-bundle-v1",
    "engine-clearance-v1",
    "manifest-v1",
    "manifest-v2",
    "provider-evidence-v1",
    "provider-plan-v1",
    "likeness-disclosure-v1",
    "reconstruction-plan-v1",
    "view-v1",
    "workflow-plan-v1",
]
PRIVATE_SCHEMAS = ["blender-response-v1"]


def test_contract_schema_directory_holds_exactly_the_known_schemas() -> None:
    present = sorted(
        path.name.removesuffix(".schema.json") for path in CONTRACT_SCHEMA_DIR.glob("*.schema.json")
    )
    assert present == sorted(DISTRIBUTED_SCHEMAS + PRIVATE_SCHEMAS)


@pytest.mark.parametrize("name", DISTRIBUTED_SCHEMAS)
def test_distributed_schema_is_byte_identical_in_the_skill(name: str) -> None:
    contract = CONTRACT_SCHEMA_DIR / f"{name}.schema.json"
    skill = SKILL_REFERENCE_DIR / f"{name}.schema.json"
    assert skill.exists(), f"{name} is distributed but missing from the Skill references"
    assert skill.read_bytes() == contract.read_bytes()


@pytest.mark.parametrize("name", PRIVATE_SCHEMAS)
def test_private_worker_schema_is_not_distributed(name: str) -> None:
    assert not (SKILL_REFERENCE_DIR / f"{name}.schema.json").exists()


def test_every_distributed_schema_declares_a_stable_identifier() -> None:
    for name in DISTRIBUTED_SCHEMAS:
        schema = json.loads((CONTRACT_SCHEMA_DIR / f"{name}.schema.json").read_text())
        assert schema["$id"] == f"https://asset-mania.dev/schemas/{name}.schema.json"
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["additionalProperties"] is False
