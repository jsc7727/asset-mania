"""The clearance gate refuses every incomplete closure, and nothing runs uncleared."""

import copy
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from asset_mania_contracts import CLEARANCE_COMPONENT_ROLES, canonical_digest
from asset_mania_pipeline import (
    EngineLicenseUncleared,
    EngineNotCleared,
    clearance_summary,
    require_mask_or_audited_remover,
    run_if_cleared,
    uncleared_entries,
    verify_engine_clearance,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"
ENGINE = "triposr-local"
NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def _example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def _reseal(clearance: dict) -> dict:
    preimage = {k: v for k, v in clearance.items() if k != "clearance_sha256"}
    return {**preimage, "clearance_sha256": canonical_digest(preimage)}


@pytest.fixture
def clearance() -> dict:
    return _example("engine-clearance-v1")


class SpyEngine:
    """Records every call. It must stay empty on every refused path."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, digest: str) -> str:
        self.calls.append(digest)
        return "mesh"


# --- The cleared path ---------------------------------------------------------------


def test_a_complete_clearance_verifies(clearance) -> None:
    assert (
        verify_engine_clearance(clearance, engine=ENGINE, now=NOW) == clearance["clearance_sha256"]
    )


def test_a_cleared_engine_reaches_the_runner_exactly_once(clearance) -> None:
    engine = SpyEngine()
    digest, result = run_if_cleared(clearance=clearance, engine=ENGINE, now=NOW, run=engine)
    assert engine.calls == [digest]
    assert result == "mesh"


# --- Refusals: the engine must never be reached --------------------------------------


def _assert_refused(clearance, *, failure=EngineNotCleared, engine_name=ENGINE, now=NOW):
    engine = SpyEngine()
    with pytest.raises(failure):
        run_if_cleared(clearance=clearance, engine=engine_name, now=now, run=engine)
    assert engine.calls == []


def test_no_clearance_at_all_is_refused() -> None:
    _assert_refused(None)


def test_a_clearance_for_another_engine_is_refused(clearance) -> None:
    _assert_refused(clearance, engine_name="some-other-engine")


def test_a_non_clearance_artifact_is_refused(clearance) -> None:
    _assert_refused({**clearance, "schema_id": "asset-mania/provider-plan"})


def test_a_maintainer_issued_clearance_is_refused(clearance) -> None:
    """Nobody accepts a third party's license terms on someone else's behalf."""
    for issuer in ("maintainer", "organization", "vendor", "system"):
        _assert_refused(_reseal({**clearance, "cleared_by": issuer}))


def test_an_edited_clearance_is_refused(clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"][1]["content_sha256"] = "f" * 64
    _assert_refused(mutated)


def test_an_expired_clearance_is_refused(clearance) -> None:
    late = datetime.fromisoformat(clearance["expires_at"]) + timedelta(seconds=1)
    _assert_refused(clearance, now=late)


def test_a_clearance_not_yet_in_effect_is_refused(clearance) -> None:
    early = datetime.fromisoformat(clearance["cleared_at"]) - timedelta(seconds=1)
    _assert_refused(clearance, now=early)


@pytest.mark.parametrize("role", CLEARANCE_COMPONENT_ROLES)
def test_a_missing_component_role_is_refused(clearance, role: str) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"] = [item for item in mutated["components"] if item["role"] != role]
    _assert_refused(_reseal(mutated))


def test_an_empty_dependency_list_is_refused(clearance) -> None:
    """No inference engine has zero dependencies."""
    _assert_refused(_reseal({**copy.deepcopy(clearance), "runtime_dependencies": []}))


def test_unsorted_dependencies_are_refused(clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"].reverse()
    _assert_refused(_reseal(mutated))


def test_a_duplicated_dependency_is_refused(clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"].append(copy.deepcopy(mutated["runtime_dependencies"][0]))
    mutated["runtime_dependencies"].sort(key=lambda item: item["name"])
    _assert_refused(_reseal(mutated))


@pytest.mark.parametrize(
    "field",
    ["revision", "content_sha256", "license_spdx", "license_url", "download_receipt_sha256"],
)
def test_a_component_missing_any_field_is_refused(clearance, field: str) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"][0][field] = ""
    _assert_refused(_reseal(mutated))


def test_a_dependency_missing_a_download_receipt_is_refused(clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"][1]["download_receipt_sha256"] = ""
    _assert_refused(_reseal(mutated))


# --- Commercial use: `unknown` fails exactly as `prohibited` does ----------------------


@pytest.mark.parametrize("state", ["prohibited", "unknown"])
def test_an_uncleared_component_is_refused(clearance, state: str) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"][1]["commercial_use"] = state
    _assert_refused(_reseal(mutated), failure=EngineLicenseUncleared)


@pytest.mark.parametrize("state", ["prohibited", "unknown"])
def test_an_uncleared_dependency_is_refused(clearance, state: str) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["runtime_dependencies"][2]["commercial_use"] = state
    _assert_refused(_reseal(mutated), failure=EngineLicenseUncleared)


def test_unknown_is_not_treated_as_a_lesser_problem_than_prohibited(clearance) -> None:
    """An unchecked license is the exact failure that produced the v0.1 error."""
    prohibited = copy.deepcopy(clearance)
    prohibited["components"][1]["commercial_use"] = "prohibited"
    unknown = copy.deepcopy(clearance)
    unknown["components"][1]["commercial_use"] = "unknown"

    for artifact in (_reseal(prohibited), _reseal(unknown)):
        with pytest.raises(EngineLicenseUncleared):
            verify_engine_clearance(artifact, engine=ENGINE, now=NOW)


def test_the_shipped_uncleared_example_is_refused() -> None:
    uncleared = _example("engine-clearance-v1-uncleared")
    _assert_refused(uncleared, failure=EngineLicenseUncleared)


def test_the_blocking_entries_are_reported_by_label_only() -> None:
    uncleared = _example("engine-clearance-v1-uncleared")
    blocked = uncleared_entries(uncleared)
    assert blocked
    rendered = " ".join(blocked)
    assert "https://" not in rendered
    for entry in blocked:
        assert entry.startswith(("component:", "dependency:"))


def test_a_cleared_clearance_blocks_nothing(clearance) -> None:
    assert uncleared_entries(clearance) == []


# --- The log-safe summary ---------------------------------------------------------------


def test_the_summary_omits_urls_and_receipts(clearance) -> None:
    summary = clearance_summary(clearance)
    rendered = json.dumps(summary)
    assert "https://" not in rendered
    for record in (*clearance["components"], *clearance["runtime_dependencies"]):
        assert record["download_receipt_sha256"] not in rendered
    assert summary["engine"] == ENGINE
    assert summary["component_count"] == len(CLEARANCE_COMPONENT_ROLES)
    assert summary["dependency_count"] >= 1
    assert summary["cleared_by"] == "user"


# --- Mask or audited remover ---------------------------------------------------------------


def test_a_mask_alone_satisfies_the_requirement() -> None:
    mask, remover = require_mask_or_audited_remover(
        mask_sha256="a" * 64,
        background_removal_clearance=None,
        engine=ENGINE,
        now=NOW,
    )
    assert mask == "a" * 64
    assert remover is None


def test_neither_a_mask_nor_a_remover_is_refused() -> None:
    with pytest.raises(ValueError, match="MASK_REQUIRED"):
        require_mask_or_audited_remover(
            mask_sha256=None,
            background_removal_clearance=None,
            engine=ENGINE,
            now=NOW,
        )


def test_an_audited_remover_satisfies_the_requirement(clearance) -> None:
    mask, remover = require_mask_or_audited_remover(
        mask_sha256=None,
        background_removal_clearance=clearance,
        engine=ENGINE,
        now=NOW,
    )
    assert mask is None
    assert remover == clearance["clearance_sha256"]


def test_an_unpinned_remover_is_refused(clearance) -> None:
    """A background model arriving without a digest is how an uncleared dependency slips in."""
    mutated = copy.deepcopy(clearance)
    for item in mutated["components"]:
        if item["role"] == "preprocessing_model":
            item["content_sha256"] = ""
    with pytest.raises(ValueError, match="BACKGROUND_REMOVAL_UNPINNED"):
        require_mask_or_audited_remover(
            mask_sha256=None,
            background_removal_clearance=_reseal(mutated),
            engine=ENGINE,
            now=NOW,
        )


def test_a_remover_with_no_preprocessing_component_is_refused(clearance) -> None:
    mutated = copy.deepcopy(clearance)
    mutated["components"] = [
        item for item in mutated["components"] if item["role"] != "preprocessing_model"
    ]
    with pytest.raises(ValueError, match="BACKGROUND_REMOVAL_UNPINNED"):
        require_mask_or_audited_remover(
            mask_sha256=None,
            background_removal_clearance=_reseal(mutated),
            engine=ENGINE,
            now=NOW,
        )


def test_an_uncleared_remover_is_refused() -> None:
    with pytest.raises(EngineLicenseUncleared):
        require_mask_or_audited_remover(
            mask_sha256=None,
            background_removal_clearance=_example("engine-clearance-v1-uncleared"),
            engine=ENGINE,
            now=NOW,
        )
