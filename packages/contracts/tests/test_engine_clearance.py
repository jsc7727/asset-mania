"""Closed-schema and builder tests for `engine-clearance-v1`.

The gate that consumes this artifact is what makes an engine usable, so the artifact itself
has to be impossible to fill in vaguely. Every test here is a way someone could produce a
document that *looks* like a clearance without having cleared anything.
"""

import copy

import pytest
from asset_mania_contracts import (
    CLEARANCE_COMPONENT_ROLES,
    COMMERCIAL_USE_STATES,
    build_engine_clearance,
    canonical_digest,
)
from conftest import load_example


@pytest.fixture
def clearance_validator(validator_for):
    return validator_for("engine-clearance", "1.0")


@pytest.fixture
def clearance():
    return load_example("engine-clearance-v1")


def test_the_example_is_valid_and_self_sealed(clearance_validator, clearance) -> None:
    assert list(clearance_validator.iter_errors(clearance)) == []
    preimage = {k: v for k, v in clearance.items() if k != "clearance_sha256"}
    assert canonical_digest(preimage) == clearance["clearance_sha256"]


def test_the_uncleared_example_is_schema_valid_but_not_cleared(
    clearance_validator,
) -> None:
    """The realistic failure is a well-formed clearance that cleared nothing."""
    uncleared = load_example("engine-clearance-v1-uncleared")
    assert list(clearance_validator.iter_errors(uncleared)) == []
    states = {item["commercial_use"] for item in uncleared["components"]}
    states |= {item["commercial_use"] for item in uncleared["runtime_dependencies"]}
    assert states - {"cleared"}


# --- Components -------------------------------------------------------------------


def test_every_required_component_role_is_present_in_order(clearance) -> None:
    assert [item["role"] for item in clearance["components"]] == CLEARANCE_COMPONENT_ROLES


@pytest.mark.parametrize("index", range(len(CLEARANCE_COMPONENT_ROLES)))
def test_a_missing_component_is_refused(clearance_validator, clearance, index: int) -> None:
    mutated = copy.deepcopy(clearance)
    del mutated["components"][index]
    assert list(clearance_validator.iter_errors(mutated))


def test_reordered_components_are_refused(clearance_validator, clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"][0], mutated["components"][1] = (
        mutated["components"][1],
        mutated["components"][0],
    )
    assert list(clearance_validator.iter_errors(mutated))


def test_an_extra_component_is_refused(clearance_validator, clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"].append(copy.deepcopy(mutated["components"][0]))
    assert list(clearance_validator.iter_errors(mutated))


@pytest.mark.parametrize(
    "field",
    [
        "name",
        "revision",
        "content_sha256",
        "license_spdx",
        "license_url",
        "commercial_use",
        "download_receipt_sha256",
    ],
)
def test_every_component_field_is_required(clearance_validator, clearance, field: str) -> None:
    mutated = copy.deepcopy(clearance)
    del mutated["components"][1][field]
    assert list(clearance_validator.iter_errors(mutated)), field


def test_a_component_may_not_carry_an_extra_field(clearance_validator, clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"][0]["note"] = "looked fine to me"
    assert list(clearance_validator.iter_errors(mutated))


# --- Runtime dependencies ------------------------------------------------------------


def test_dependencies_are_name_sorted(clearance) -> None:
    names = [item["name"] for item in clearance["runtime_dependencies"]]
    assert names == sorted(names)


def test_an_empty_dependency_list_is_refused(clearance_validator, clearance) -> None:
    """No inference engine has zero dependencies; an empty list is the cheapest fake."""
    mutated = {**copy.deepcopy(clearance), "runtime_dependencies": []}
    assert list(clearance_validator.iter_errors(mutated))


def test_duplicate_dependencies_are_refused(clearance_validator, clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"].append(copy.deepcopy(mutated["runtime_dependencies"][0]))
    assert list(clearance_validator.iter_errors(mutated))


def test_a_dependency_may_not_carry_a_role(clearance_validator, clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"][0]["role"] = "engine_code"
    assert list(clearance_validator.iter_errors(mutated))


# --- Commercial use --------------------------------------------------------------------


def test_the_three_declared_states_are_the_only_ones(clearance_validator, clearance) -> None:
    assert sorted(COMMERCIAL_USE_STATES) == ["cleared", "prohibited", "unknown"]
    for state in COMMERCIAL_USE_STATES:
        mutated = copy.deepcopy(clearance)
        mutated["components"][0]["commercial_use"] = state
        assert list(clearance_validator.iter_errors(mutated)) == [], state

    mutated = copy.deepcopy(clearance)
    mutated["components"][0]["commercial_use"] = "probably_fine"
    assert list(clearance_validator.iter_errors(mutated))


# --- Issuance ---------------------------------------------------------------------------


def test_only_the_user_clears_an_engine(clearance_validator, clearance) -> None:
    for issuer in ("maintainer", "organization", "system", "vendor"):
        mutated = {**copy.deepcopy(clearance), "cleared_by": issuer}
        assert list(clearance_validator.iter_errors(mutated)), issuer


def test_an_unknown_top_level_field_is_refused(clearance_validator, clearance) -> None:
    for field, value in (
        ("weights_path", "/Users/example/weights/model.safetensors"),
        ("download_url", "https://example.invalid/weights"),
        ("api_key", "PROVIDER-CREDENTIAL-PLACEHOLDER"),
    ):
        assert list(clearance_validator.iter_errors({**clearance, field: value})), field


# --- The builder -------------------------------------------------------------------------


def test_the_builder_reproduces_the_example(clearance) -> None:
    built = build_engine_clearance(
        engine=clearance["engine"],
        components=clearance["components"],
        runtime_dependencies=clearance["runtime_dependencies"],
        cleared_at=clearance["cleared_at"],
        expires_at=clearance["expires_at"],
    )
    assert built == clearance


def test_the_builder_sorts_dependencies(clearance) -> None:
    built = build_engine_clearance(
        engine=clearance["engine"],
        components=clearance["components"],
        runtime_dependencies=list(reversed(clearance["runtime_dependencies"])),
        cleared_at=clearance["cleared_at"],
        expires_at=clearance["expires_at"],
    )
    assert built == clearance


def test_the_builder_refuses_a_missing_component_role(clearance) -> None:
    with pytest.raises(ValueError, match="components must be exactly"):
        build_engine_clearance(
            engine=clearance["engine"],
            components=clearance["components"][:2],
            runtime_dependencies=clearance["runtime_dependencies"],
            cleared_at=clearance["cleared_at"],
            expires_at=clearance["expires_at"],
        )


def test_the_builder_refuses_an_empty_dependency_list(clearance) -> None:
    with pytest.raises(ValueError, match="never true for an inference engine"):
        build_engine_clearance(
            engine=clearance["engine"],
            components=clearance["components"],
            runtime_dependencies=[],
            cleared_at=clearance["cleared_at"],
            expires_at=clearance["expires_at"],
        )


def test_the_builder_refuses_an_inverted_validity_window(clearance) -> None:
    with pytest.raises(ValueError, match="expires_at"):
        build_engine_clearance(
            engine=clearance["engine"],
            components=clearance["components"],
            runtime_dependencies=clearance["runtime_dependencies"],
            cleared_at=clearance["expires_at"],
            expires_at=clearance["cleared_at"],
        )


def test_the_builder_never_accepts_a_maintainer_issuer(clearance) -> None:
    """`cleared_by` is fixed, not a parameter: nobody clears a license for someone else."""
    built = build_engine_clearance(
        engine=clearance["engine"],
        components=clearance["components"],
        runtime_dependencies=clearance["runtime_dependencies"],
        cleared_at=clearance["cleared_at"],
        expires_at=clearance["expires_at"],
    )
    assert built["cleared_by"] == "user"


def test_the_digest_covers_every_component_and_dependency(clearance) -> None:
    baseline = clearance["clearance_sha256"]
    preimage = {k: v for k, v in clearance.items() if k != "clearance_sha256"}
    for index in range(len(preimage["components"])):
        mutated = copy.deepcopy(preimage)
        mutated["components"][index]["commercial_use"] = "unknown"
        assert canonical_digest(mutated) != baseline
    for index in range(len(preimage["runtime_dependencies"])):
        mutated = copy.deepcopy(preimage)
        mutated["runtime_dependencies"][index]["content_sha256"] = "f" * 64
        assert canonical_digest(mutated) != baseline
