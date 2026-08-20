"""Closed-schema tests for `provider-evidence-v1`."""

import copy
from datetime import datetime, timedelta

import pytest
from asset_mania_contracts import OFFICIAL_SOURCE_HOSTS, OUTPUT_COST_ROWS, canonical_digest
from conftest import load_example


@pytest.fixture
def evidence_validator(validator_for):
    return validator_for("provider-evidence", "1.0")


@pytest.fixture
def evidence():
    return load_example("provider-evidence-v1")


def test_example_is_valid_and_self_sealed(evidence_validator, evidence) -> None:
    assert list(evidence_validator.iter_errors(evidence)) == []
    preimage = {key: value for key, value in evidence.items() if key != "evidence_sha256"}
    assert canonical_digest(preimage) == evidence["evidence_sha256"]


def test_evidence_expires_exactly_twenty_four_hours_after_retrieval(evidence) -> None:
    retrieved = datetime.fromisoformat(evidence["retrieved_at"])
    expires = datetime.fromisoformat(evidence["expires_at"])
    assert expires - retrieved == timedelta(hours=24)


def test_sources_are_url_sorted_and_limited_to_official_hosts(evidence_validator, evidence) -> None:
    urls = [source["url"] for source in evidence["sources"]]
    assert urls == sorted(urls)
    for url in urls:
        assert any(url.startswith(f"https://{host}") for host in OFFICIAL_SOURCE_HOSTS)

    mutated = copy.deepcopy(evidence)
    mutated["sources"][0]["url"] = "https://blog.example.com/openai-pricing"
    assert list(evidence_validator.iter_errors(mutated))

    mutated = copy.deepcopy(evidence)
    mutated["sources"][0]["url"] = "http://platform.openai.com/docs/pricing"
    assert list(evidence_validator.iter_errors(mutated))


def test_output_cost_rows_keep_their_canonical_identity_and_order(
    evidence_validator, evidence
) -> None:
    rows = evidence["pricing"]["output_cost_rows"]
    assert [(row["quality"], row["size"]) for row in rows] == list(OUTPUT_COST_ROWS)

    reordered = copy.deepcopy(evidence)
    reordered["pricing"]["output_cost_rows"][0], reordered["pricing"]["output_cost_rows"][3] = (
        reordered["pricing"]["output_cost_rows"][3],
        reordered["pricing"]["output_cost_rows"][0],
    )
    assert list(evidence_validator.iter_errors(reordered))

    duplicated = copy.deepcopy(evidence)
    duplicated["pricing"]["output_cost_rows"].append(rows[0])
    assert list(evidence_validator.iter_errors(duplicated))

    dropped = copy.deepcopy(evidence)
    del dropped["pricing"]["output_cost_rows"][-1]
    assert list(evidence_validator.iter_errors(dropped))


@pytest.mark.parametrize("value", ["0.06", "0.0600000", ".060000", "1e-2", "0", "00.060000"])
def test_decimal_strings_require_exactly_six_places(
    evidence_validator, evidence, value: str
) -> None:
    mutated = copy.deepcopy(evidence)
    mutated["pricing"]["output_cost_rows"][0]["usd"] = value
    assert list(evidence_validator.iter_errors(mutated))


def test_prices_are_strings_so_binary_floats_never_enter_a_portable_artifact(
    evidence_validator, evidence
) -> None:
    mutated = copy.deepcopy(evidence)
    mutated["pricing"]["output_cost_rows"][0]["usd"] = 0.006
    assert list(evidence_validator.iter_errors(mutated))


def test_evidence_carries_no_credential_or_account_usage(evidence_validator, evidence) -> None:
    for key, value in (
        ("api_key", "sk-live-000"),
        ("organization_id", "org-000"),
        ("account_usage", {"requests": 3}),
    ):
        assert list(evidence_validator.iter_errors({**evidence, key: value})), key


def test_data_policy_is_a_closed_record(evidence_validator, evidence) -> None:
    mutated = copy.deepcopy(evidence)
    mutated["data_policy"]["note"] = "spoke to support"
    assert list(evidence_validator.iter_errors(mutated))

    mutated = copy.deepcopy(evidence)
    mutated["data_policy"]["abuse_monitoring_days"] = 0
    assert list(evidence_validator.iter_errors(mutated))
