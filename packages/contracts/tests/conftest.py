import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"


def load_example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def example_names(prefix: str) -> list[str]:
    return sorted(path.stem for path in EXAMPLES.glob(f"{prefix}*.json"))


@pytest.fixture
def example():
    return load_example


@pytest.fixture
def validator_for():
    from asset_mania_contracts import load_schema

    def build(name: str, version: str) -> Draft202012Validator:
        schema = load_schema(name, version)
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema)

    return build


def assert_rejects(validator: Draft202012Validator, document: dict, reason: str) -> None:
    errors = list(validator.iter_errors(document))
    assert errors, f"schema accepted an invalid document: {reason}"
