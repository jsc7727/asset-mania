"""Closed-schema and builder tests for `manifest-v2`."""

import copy
import hashlib

import pytest
from asset_mania_contracts import (
    STAGE_COMMANDS,
    canonical_digest,
    canonical_json,
    exit_code_for,
)
from conftest import ROOT, example_names, load_example

V1_FIXTURE_SHA256 = "0ac8cdf16f86e2323e30e028aef72ee5c625522f813fd16a359f9096343d77af"
V1_SCHEMA_SHA256 = "f2fe4e7942cc3e7a2c7bb3294891863c2cc3c52669415a6fbb52c896e0f84dd0"

STAGE_EXAMPLES = {
    "scene-preflight": "manifest-v2-scene-preflight",
    "scene-plan": "manifest-v2-scene-plan",
    "provider-evidence": "manifest-v2-provider-evidence",
    "provider-plan": "manifest-v2-provider-plan",
    "approval-issue": "manifest-v2-approval-issue",
    "condition": "manifest-v2-condition",
    "view-ingest": "manifest-v2-view-ingest",
    "provider-generate": "manifest-v2-provider-generate",
    "bake": "manifest-v2-bake",
    "export": "manifest-v2-export",
}


@pytest.fixture
def manifest_validator(validator_for):
    return validator_for("run-manifest", "2.0")


def _sha256_file(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_v1_fixture_and_schema_bytes_are_unchanged() -> None:
    assert _sha256_file("tests/fixtures/manifest-v1-success.json") == V1_FIXTURE_SHA256
    assert (
        _sha256_file("packages/contracts/src/asset_mania_contracts/schema/manifest-v1.schema.json")
        == V1_SCHEMA_SHA256
    )


def test_every_stage_has_one_normative_example() -> None:
    assert set(STAGE_EXAMPLES) == set(STAGE_COMMANDS)
    assert sorted(STAGE_EXAMPLES.values()) == example_names("manifest-v2-")


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_stage_example_is_valid_and_command_bound(manifest_validator, stage: str) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    assert list(manifest_validator.iter_errors(document)) == []
    assert document["stage"] == stage
    assert document["command"] == STAGE_COMMANDS[stage]


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_stage_rejects_another_stage_command(manifest_validator, stage: str) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    other = next(value for key, value in STAGE_COMMANDS.items() if key != stage)
    document["command"] = other
    assert list(manifest_validator.iter_errors(document))


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_stage_rejects_another_stage_parameters(manifest_validator, stage: str) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    donor = next(name for key, name in STAGE_EXAMPLES.items() if key != stage)
    document["parameters"] = load_example(donor)["parameters"]
    assert list(manifest_validator.iter_errors(document))


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_stage_parameters_are_closed(manifest_validator, stage: str) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    document["parameters"]["extra_parameter"] = "denied"
    assert list(manifest_validator.iter_errors(document))


def test_unknown_top_level_property_is_rejected(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["condition"])
    document["source_basename"] = "private-character.blend"
    assert list(manifest_validator.iter_errors(document))


@pytest.mark.parametrize(
    "path",
    [
        "/Users/example/private-character.blend",
        "C:\\scenes\\private.blend",
        "../outside/asset.glb",
        "exports/../../escape.glb",
        "exports//double.glb",
        "exports\\windows.glb",
    ],
)
def test_artifact_path_rejects_absolute_and_traversing_paths(manifest_validator, path: str) -> None:
    document = load_example(STAGE_EXAMPLES["export"])
    document["artifacts"][0]["path"] = path
    assert list(manifest_validator.iter_errors(document))


@pytest.mark.parametrize(
    "run_id",
    ["run/preflight", "run preflight", "../run", "run\u0000id", "-leading-dash", "r" * 65, ""],
)
def test_identifier_pattern_rejects_separators_traversal_and_control_characters(
    manifest_validator, run_id: str
) -> None:
    document = load_example(STAGE_EXAMPLES["scene-preflight"])
    document["run_id"] = run_id
    assert list(manifest_validator.iter_errors(document))


def test_consumption_identifier_uses_the_same_pattern(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["provider-generate"])
    document["approvals"][0]["consumption_id"] = "consumption/external-egress"
    assert list(manifest_validator.iter_errors(document))


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_only_preflight_and_evidence_permit_a_null_plan_digest(
    manifest_validator, stage: str
) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    document["plan_sha256"] = None
    permitted = stage in {"scene-preflight", "provider-evidence"}
    assert (list(manifest_validator.iter_errors(document)) == []) is permitted


@pytest.mark.parametrize("stage", sorted(STAGE_EXAMPLES))
def test_only_provider_evidence_may_reach_official_hosts(manifest_validator, stage: str) -> None:
    document = load_example(STAGE_EXAMPLES[stage])
    document["capabilities"]["network"] = "explicit_official_hosts"
    permitted = stage == "provider-evidence"
    assert (list(manifest_validator.iter_errors(document)) == []) is permitted


def test_provider_evidence_records_the_exact_official_hosts(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["provider-evidence"])
    assert document["parameters"]["source_hosts"] == [
        "developers.openai.com",
        "platform.openai.com",
    ]
    document["parameters"]["source_hosts"] = ["example.com"]
    assert list(manifest_validator.iter_errors(document))


def test_parent_reference_requires_all_three_identity_fields(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["condition"])
    for dropped in ("run_id", "manifest_sha256", "relationship"):
        mutated = copy.deepcopy(document)
        del mutated["parents"][0][dropped]
        assert list(manifest_validator.iter_errors(mutated)), dropped


def test_parent_reference_rejects_a_mutable_label_instead_of_a_digest(
    manifest_validator,
) -> None:
    document = load_example(STAGE_EXAMPLES["condition"])
    document["parents"][0]["manifest_sha256"] = "latest"
    assert list(manifest_validator.iter_errors(document))
    document = load_example(STAGE_EXAMPLES["condition"])
    document["parents"][0]["path"] = "../run-plan-1/manifest.json"
    assert list(manifest_validator.iter_errors(document))


def test_artifact_identity_requires_every_provenance_field(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["export"])
    required = [
        "role",
        "path",
        "sha256",
        "byte_size",
        "media_type",
        "parents",
        "operation",
        "content_origin",
        "sensitivity",
        "upload_eligible",
        "validation",
    ]
    for dropped in required:
        mutated = copy.deepcopy(document)
        del mutated["artifacts"][0][dropped]
        assert list(manifest_validator.iter_errors(mutated)), dropped


def test_generated_origin_survives_transitive_derivation() -> None:
    """A texture baked from a generated view stays generated, not merely derived."""
    generate = load_example(STAGE_EXAMPLES["provider-generate"])
    bake = load_example(STAGE_EXAMPLES["bake"])
    export = load_example(STAGE_EXAMPLES["export"])

    generated_view = generate["artifacts"][0]
    assert generated_view["content_origin"] == "generated"

    baked = bake["artifacts"][0]
    assert generated_view["sha256"] in {parent["sha256"] for parent in baked["parents"]}
    assert baked["content_origin"] == "generated"

    for exported in export["artifacts"]:
        assert baked["sha256"] in {parent["sha256"] for parent in exported["parents"]}
        assert exported["content_origin"] == "generated"
        assert exported["upload_eligible"] is False


def test_diagnostic_and_warning_arrays_reject_duplicates(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["provider-plan"])
    document["result"]["diagnostics"] = [
        "FACE_RIGHTS_CONFIRMATION_REQUIRED",
        "FACE_RIGHTS_CONFIRMATION_REQUIRED",
    ]
    assert list(manifest_validator.iter_errors(document))


def test_unknown_diagnostic_code_is_rejected(manifest_validator) -> None:
    document = load_example(STAGE_EXAMPLES["condition"])
    document["result"]["diagnostics"] = ["SOMETHING_NEW"]
    assert list(manifest_validator.iter_errors(document))


def test_canonical_json_is_sorted_compact_and_newline_terminated() -> None:
    rendered = canonical_json({"b": 1, "a": [2, 3]})
    assert rendered == '{"a":[2,3],"b":1}\n'


def test_canonical_digest_omits_no_field() -> None:
    document = load_example(STAGE_EXAMPLES["condition"])
    baseline = canonical_digest(document)
    for key in document:
        mutated = copy.deepcopy(document)
        del mutated[key]
        assert canonical_digest(mutated) != baseline, key


def test_canonical_digest_changes_for_every_approval_sensitive_field() -> None:
    plan = load_example("provider-plan-v1")
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert canonical_digest(preimage) == plan["plan_sha256"]

    mutations = {
        "prompt_sha256": "f" * 64,
        "model": "gpt-image-2-2026-04-22",
        "endpoint": "/v1/images/generations",
        "subject": "real_person",
    }
    for key, value in mutations.items():
        mutated = {**preimage, key: value}
        assert canonical_digest(mutated) != plan["plan_sha256"], key

    nested_mutations = {
        "controls": {"moderation": "low"},
        "cost_estimate": {"maximum_cost": "9.000000"},
        "policy_evidence": {"effective_region": "eu-central-1"},
    }
    for key, overrides in nested_mutations.items():
        mutated = {**preimage, key: {**preimage[key], **overrides}}
        assert canonical_digest(mutated) != plan["plan_sha256"], key

    attachments = copy.deepcopy(preimage["attachments"])
    attachments[0]["sha256"] = "e" * 64
    assert canonical_digest({**preimage, "attachments": attachments}) != plan["plan_sha256"]


def test_exit_codes_are_fixed() -> None:
    assert exit_code_for("succeeded") == 0
    assert exit_code_for("failed") == 3
    assert exit_code_for("unsupported") == 3
    assert exit_code_for("needs_approval") == 5
    assert exit_code_for("canceled") == 6
    with pytest.raises(ValueError):
        exit_code_for("partially_succeeded")


def test_schema_registry_resolves_every_name_and_version() -> None:
    from asset_mania_contracts import load_schema, schema_names

    assert schema_names() == [
        ("approval-receipt", "1.0"),
        ("blender-response", "1.0"),
        ("conditioning-bundle", "1.0"),
        ("engine-clearance", "1.0"),
        ("likeness-disclosure", "1.0"),
        ("multiview-reconstruction", "1.0"),
        ("provider-evidence", "1.0"),
        ("provider-plan", "1.0"),
        ("reconstruction-plan", "1.0"),
        ("run-manifest", "1.0"),
        ("run-manifest", "2.0"),
        ("turntable-plan", "1.0"),
        ("turntable-viewset", "1.0"),
        ("view", "1.0"),
        ("workflow-plan", "1.0"),
    ]
    assert load_schema("run-manifest", "1.0")["properties"]["schema_version"]["const"] == "1.0"
    assert load_schema("run-manifest", "2.0")["properties"]["schema_version"]["const"] == "2.0"
    with pytest.raises(KeyError):
        load_schema("run-manifest", "3.0")
    with pytest.raises(KeyError):
        load_schema("selection-map", "1.0")


def test_registry_returns_a_copy_so_callers_cannot_mutate_the_contract() -> None:
    from asset_mania_contracts import load_schema

    first = load_schema("view", "1.0")
    first["additionalProperties"] = True
    assert load_schema("view", "1.0")["additionalProperties"] is False


def test_provider_state_advances_only_along_the_declared_path() -> None:
    from asset_mania_contracts import PROVIDER_STATE_ORDER, advance_provider_state

    assert PROVIDER_STATE_ORDER == [
        "planned",
        "approval_required",
        "approved",
        "submitted",
        "running",
    ]
    state = "planned"
    for expected in PROVIDER_STATE_ORDER[1:]:
        state = advance_provider_state(state, expected)
        assert state == expected
    assert advance_provider_state("running", "succeeded") == "succeeded"

    with pytest.raises(ValueError, match="provider_state"):
        advance_provider_state("planned", "submitted")
    with pytest.raises(ValueError, match="provider_state"):
        advance_provider_state("approved", "planned")


@pytest.mark.parametrize("terminal", ["succeeded", "failed", "canceled"])
def test_no_transition_leaves_a_terminal_provider_state(terminal: str) -> None:
    from asset_mania_contracts import advance_provider_state

    with pytest.raises(ValueError, match="terminal"):
        advance_provider_state(terminal, "running")


def test_builder_reproduces_every_stage_example() -> None:
    from asset_mania_contracts import build_manifest_v2

    for stage, name in sorted(STAGE_EXAMPLES.items()):
        expected = load_example(name)
        built = build_manifest_v2(
            run_id=expected["run_id"],
            stage=stage,
            created_at=expected["created_at"],
            tool_version=expected["tool_version"],
            inputs=expected["inputs"],
            parents=expected["parents"],
            parameters=expected["parameters"],
            plan_sha256=expected["plan_sha256"],
            environment=expected["environment"],
            capabilities=expected["capabilities"],
            approvals=expected["approvals"],
            artifacts=expected["artifacts"],
            result_status=expected["result"]["status"],
            diagnostics=expected["result"]["diagnostics"],
            provider_state=expected["result"]["provider_state"],
        )
        assert built == expected, stage


def test_builder_sorts_diagnostics_parents_artifacts_and_approvals() -> None:
    from asset_mania_contracts import build_manifest_v2

    expected = load_example(STAGE_EXAMPLES["provider-generate"])
    built = build_manifest_v2(
        run_id=expected["run_id"],
        stage="provider-generate",
        created_at=expected["created_at"],
        tool_version=expected["tool_version"],
        inputs=list(reversed(expected["inputs"])),
        parents=list(reversed(expected["parents"])),
        parameters=expected["parameters"],
        plan_sha256=expected["plan_sha256"],
        environment=expected["environment"],
        capabilities=expected["capabilities"],
        approvals=list(reversed(expected["approvals"])),
        artifacts=list(reversed(expected["artifacts"])),
        result_status=expected["result"]["status"],
        diagnostics=expected["result"]["diagnostics"],
        provider_state=expected["result"]["provider_state"],
    )
    assert built == expected


def test_builder_deduplicates_and_sorts_diagnostics() -> None:
    from asset_mania_contracts import build_manifest_v2

    expected = load_example(STAGE_EXAMPLES["provider-plan"])
    built = build_manifest_v2(
        run_id=expected["run_id"],
        stage="provider-plan",
        created_at=expected["created_at"],
        tool_version=expected["tool_version"],
        inputs=expected["inputs"],
        parents=expected["parents"],
        parameters=expected["parameters"],
        plan_sha256=expected["plan_sha256"],
        environment=expected["environment"],
        capabilities=expected["capabilities"],
        approvals=expected["approvals"],
        artifacts=expected["artifacts"],
        result_status=expected["result"]["status"],
        diagnostics=[
            "FACE_RIGHTS_CONFIRMATION_REQUIRED",
            "FACE_RIGHTS_CONFIRMATION_REQUIRED",
        ],
        provider_state=expected["result"]["provider_state"],
        warnings=["PROVIDER_EVIDENCE_STALE", "PROVIDER_EVIDENCE_STALE"],
    )
    assert built["result"]["diagnostics"] == ["FACE_RIGHTS_CONFIRMATION_REQUIRED"]
    assert built["warnings"] == ["PROVIDER_EVIDENCE_STALE"]


def test_builder_rejects_a_stage_command_mismatch() -> None:
    from asset_mania_contracts import build_manifest_v2

    expected = load_example(STAGE_EXAMPLES["bake"])
    with pytest.raises(ValueError, match="parameters"):
        build_manifest_v2(
            run_id=expected["run_id"],
            stage="bake",
            created_at=expected["created_at"],
            tool_version=expected["tool_version"],
            inputs=expected["inputs"],
            parents=expected["parents"],
            parameters=load_example(STAGE_EXAMPLES["export"])["parameters"],
            plan_sha256=expected["plan_sha256"],
            environment=expected["environment"],
            capabilities=expected["capabilities"],
            approvals=expected["approvals"],
            artifacts=expected["artifacts"],
            result_status=expected["result"]["status"],
            diagnostics=expected["result"]["diagnostics"],
            provider_state=expected["result"]["provider_state"],
        )


def test_builder_rejects_a_null_plan_digest_outside_the_two_permitted_stages() -> None:
    from asset_mania_contracts import build_manifest_v2

    expected = load_example(STAGE_EXAMPLES["condition"])
    with pytest.raises(ValueError, match="plan_sha256"):
        build_manifest_v2(
            run_id=expected["run_id"],
            stage="condition",
            created_at=expected["created_at"],
            tool_version=expected["tool_version"],
            inputs=expected["inputs"],
            parents=expected["parents"],
            parameters=expected["parameters"],
            plan_sha256=None,
            environment=expected["environment"],
            capabilities=expected["capabilities"],
            approvals=expected["approvals"],
            artifacts=expected["artifacts"],
            result_status=expected["result"]["status"],
            diagnostics=expected["result"]["diagnostics"],
            provider_state=expected["result"]["provider_state"],
        )


def test_builder_rejects_official_host_reach_outside_provider_evidence() -> None:
    from asset_mania_contracts import build_manifest_v2

    expected = load_example(STAGE_EXAMPLES["condition"])
    with pytest.raises(ValueError, match="network"):
        build_manifest_v2(
            run_id=expected["run_id"],
            stage="condition",
            created_at=expected["created_at"],
            tool_version=expected["tool_version"],
            inputs=expected["inputs"],
            parents=expected["parents"],
            parameters=expected["parameters"],
            plan_sha256=expected["plan_sha256"],
            environment=expected["environment"],
            capabilities={
                **expected["capabilities"],
                "network": "explicit_official_hosts",
            },
            approvals=expected["approvals"],
            artifacts=expected["artifacts"],
            result_status=expected["result"]["status"],
            diagnostics=expected["result"]["diagnostics"],
            provider_state=expected["result"]["provider_state"],
        )
