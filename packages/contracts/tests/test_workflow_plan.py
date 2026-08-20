"""Closed-schema and builder tests for `workflow-plan-v1`."""

import copy

import pytest
from asset_mania_contracts import (
    EXPECTED_ARTIFACT_ROLES,
    RENDER_PROFILE,
    build_workflow_plan,
    canonical_digest,
    selection_digest,
)
from conftest import load_example


@pytest.fixture
def plan_validator(validator_for):
    return validator_for("workflow-plan", "1.0")


@pytest.fixture
def plan():
    return load_example("workflow-plan-v1")


def test_example_is_valid_and_self_sealed(plan_validator, plan) -> None:
    assert list(plan_validator.iter_errors(plan)) == []
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert canonical_digest(preimage) == plan["plan_sha256"]


def test_builder_reproduces_the_normative_example(plan) -> None:
    built = build_workflow_plan(
        source_scene_sha256=plan["source_scene_sha256"],
        preflight_manifest_sha256=plan["preflight_manifest_sha256"],
        selection=plan["selection"],
        asset_kind=plan["asset_kind"],
        subject=plan["subject"],
        frame=plan["frame"],
        action_range=plan["action_range"],
        resolution=plan["resolution"],
        render_profile=plan["render_profile"],
    )
    assert built == plan


def test_render_profile_records_every_binding_value(plan) -> None:
    assert plan["render_profile"] == RENDER_PROFILE
    assert plan["render_profile"].keys() == RENDER_PROFILE.keys()


def test_expected_artifact_roles_keep_their_stable_order(plan_validator, plan) -> None:
    assert plan["expected_artifact_roles"] == EXPECTED_ARTIFACT_ROLES
    reordered = copy.deepcopy(plan)
    reordered["expected_artifact_roles"] = sorted(EXPECTED_ARTIFACT_ROLES)
    assert list(plan_validator.iter_errors(reordered))


def test_plan_is_closed_against_private_and_unknown_fields(plan_validator, plan) -> None:
    for key, value in (
        ("source_path", "/Users/example/scenes/private-character.blend"),
        ("source_basename", "private-character.blend"),
        ("datablock_names", ["Camera_Main", "Body_LOD0"]),
        ("prompt", "a photo of a person"),
        ("api_key", "sk-live-000"),
    ):
        mutated = {**plan, key: value}
        assert list(plan_validator.iter_errors(mutated)), key


def test_selection_carries_labels_and_a_digest_but_no_private_identifier(
    plan_validator, plan
) -> None:
    assert set(plan["selection"]) == {
        "camera_label",
        "target_label",
        "armature_label",
        "action_label",
        "selection_digest",
    }
    mutated = copy.deepcopy(plan)
    mutated["selection"]["blender_object_name"] = "Body_LOD0"
    assert list(plan_validator.iter_errors(mutated))

    mutated = copy.deepcopy(plan)
    mutated["selection"]["target_label"] = "Body_LOD0"
    assert list(plan_validator.iter_errors(mutated))


def test_selection_digest_is_salted_and_binds_every_identity_field() -> None:
    identity = {
        "source_scene_sha256": "1a" * 32,
        "camera": "Camera_Main",
        "target": "Body_LOD0",
        "armature": "Rig",
        "action": "Idle",
        "target_type": "MESH",
    }
    salt = bytes(range(32))
    baseline = selection_digest(salt=salt, identity=identity)

    assert selection_digest(salt=bytes(32), identity=identity) != baseline
    for key, value in identity.items():
        mutated = {**identity, key: f"{value}-changed"}
        assert selection_digest(salt=salt, identity=mutated) != baseline, key


def test_unknown_subject_is_never_a_planable_declaration(plan_validator, plan) -> None:
    mutated = {**plan, "subject": "unknown"}
    assert list(plan_validator.iter_errors(mutated))
    with pytest.raises(ValueError, match="SUBJECT_DECLARATION_REQUIRED"):
        build_workflow_plan(
            source_scene_sha256=plan["source_scene_sha256"],
            preflight_manifest_sha256=plan["preflight_manifest_sha256"],
            selection=plan["selection"],
            asset_kind=plan["asset_kind"],
            subject="unknown",
            frame=plan["frame"],
            action_range=plan["action_range"],
            resolution=plan["resolution"],
            render_profile=plan["render_profile"],
        )


def test_asset_kind_and_subject_are_closed_user_declarations(plan_validator, plan) -> None:
    for key, value in (("asset_kind", "inferred"), ("subject", "probably_a_person")):
        assert list(plan_validator.iter_errors({**plan, key: value})), key


def test_condition_frame_must_lie_inside_a_non_null_action_range(plan) -> None:
    with pytest.raises(ValueError, match="frame"):
        build_workflow_plan(
            source_scene_sha256=plan["source_scene_sha256"],
            preflight_manifest_sha256=plan["preflight_manifest_sha256"],
            selection=plan["selection"],
            asset_kind=plan["asset_kind"],
            subject=plan["subject"],
            frame=48,
            action_range=[1, 24],
            resolution=plan["resolution"],
            render_profile=plan["render_profile"],
        )


def test_builder_rejects_an_inverted_action_range(plan) -> None:
    with pytest.raises(ValueError, match="action_range"):
        build_workflow_plan(
            source_scene_sha256=plan["source_scene_sha256"],
            preflight_manifest_sha256=plan["preflight_manifest_sha256"],
            selection=plan["selection"],
            asset_kind=plan["asset_kind"],
            subject=plan["subject"],
            frame=12,
            action_range=[24, 1],
            resolution=plan["resolution"],
            render_profile=plan["render_profile"],
        )


def test_overwrite_policy_is_create_only(plan_validator, plan) -> None:
    assert plan["overwrite_policy"] == "create_only"
    assert list(plan_validator.iter_errors({**plan, "overwrite_policy": "replace"}))


def test_plan_digest_excludes_no_field(plan) -> None:
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    baseline = canonical_digest(preimage)
    for key in preimage:
        mutated = {name: value for name, value in preimage.items() if name != key}
        assert canonical_digest(mutated) != baseline, key
