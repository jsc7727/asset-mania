"""Canonical plan hashing, tamper detection, and salted selection resolution."""

import json
from pathlib import Path

import pytest
from asset_mania_contracts import canonical_json
from asset_mania_pipeline import (
    PlanTampered,
    build_selection_map,
    load_plan,
    plan_digest,
    verify_plan,
    verify_selection,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"

IDENTITY = {
    "source_scene_sha256": "1a" * 32,
    "camera": "Camera_Main",
    "target": "Body_LOD0",
    "target_type": "MESH",
    "armature": "Rig",
    "action": "Idle",
}
LABELS = {
    "camera_label": "camera-1",
    "target_label": "mesh-1",
    "armature_label": "armature-1",
    "action_label": "action-1",
}
SALT = bytes(range(32))


def _example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def test_a_normative_plan_verifies(tmp_path: Path) -> None:
    plan = _example("workflow-plan-v1")
    assert plan_digest(plan) == plan["plan_sha256"]
    verify_plan(plan)


def test_a_tampered_plan_field_fails(tmp_path: Path) -> None:
    plan = _example("workflow-plan-v1")
    for key, value in (
        ("frame", 13),
        ("subject", "real_person"),
        ("asset_kind", "face_head"),
        ("resolution", [1536, 1024]),
        ("source_scene_sha256", "f" * 64),
    ):
        with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
            verify_plan({**plan, key: value})


def test_a_tampered_render_profile_value_fails() -> None:
    plan = _example("workflow-plan-v1")
    tampered = {
        **plan,
        "render_profile": {**plan["render_profile"], "samples": 32},
    }
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        verify_plan(tampered)


def test_load_plan_verifies_the_self_digest_and_any_expected_digest(tmp_path: Path) -> None:
    plan = _example("workflow-plan-v1")
    path = tmp_path / "workflow-plan.json"
    path.write_text(canonical_json(plan), encoding="utf-8")

    assert load_plan(path) == plan
    assert load_plan(path, expected_sha256=plan["plan_sha256"]) == plan
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        load_plan(path, expected_sha256="f" * 64)


def test_load_plan_rejects_unreadable_or_non_object_content(tmp_path: Path) -> None:
    path = tmp_path / "workflow-plan.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        load_plan(path)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        load_plan(path)


def test_selection_map_is_local_sensitive_and_never_upload_eligible() -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    assert selection_map["sensitivity"] == "local-sensitive"
    assert selection_map["upload_eligible"] is False
    assert selection_map["salt"] == SALT.hex()
    assert selection_map["identity"] == IDENTITY


def test_the_portable_selection_carries_labels_and_the_digest_only() -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    portable = selection_map["portable_selection"]
    assert set(portable) == {
        "camera_label",
        "target_label",
        "armature_label",
        "action_label",
        "selection_digest",
    }
    rendered = canonical_json(portable)
    for private in ("Camera_Main", "Body_LOD0", "Rig", "Idle", SALT.hex()):
        assert private not in rendered


def test_a_matching_selection_verifies() -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    verify_selection(
        selection_map=selection_map,
        portable_selection=selection_map["portable_selection"],
        identity=IDENTITY,
    )


@pytest.mark.parametrize("field", sorted(IDENTITY))
def test_any_identity_change_fails_verification(field: str) -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        verify_selection(
            selection_map=selection_map,
            portable_selection=selection_map["portable_selection"],
            identity={**IDENTITY, field: f"{IDENTITY[field]}-changed"},
        )


@pytest.mark.parametrize("field", sorted(LABELS))
def test_any_label_change_fails_verification(field: str) -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    portable = {**selection_map["portable_selection"], field: "mesh-9"}
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        verify_selection(
            selection_map=selection_map,
            portable_selection=portable,
            identity=IDENTITY,
        )


def test_a_substituted_salt_fails_verification() -> None:
    selection_map = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    forged = {**selection_map, "salt": bytes(32).hex()}
    with pytest.raises(PlanTampered, match="PLAN_TAMPERED"):
        verify_selection(
            selection_map=forged,
            portable_selection=selection_map["portable_selection"],
            identity=IDENTITY,
        )


def test_two_plans_over_the_same_scene_get_different_digests_from_different_salts() -> None:
    first = build_selection_map(salt=SALT, identity=IDENTITY, labels=LABELS)
    second = build_selection_map(salt=bytes(range(32, 64)), identity=IDENTITY, labels=LABELS)
    assert (
        first["portable_selection"]["selection_digest"]
        != second["portable_selection"]["selection_digest"]
    )


def test_a_short_salt_is_refused() -> None:
    with pytest.raises(ValueError, match="salt"):
        build_selection_map(salt=b"short", identity=IDENTITY, labels=LABELS)


def test_a_null_armature_and_action_are_supported() -> None:
    identity = {**IDENTITY, "armature": None, "action": None}
    labels = {**LABELS, "armature_label": None, "action_label": None}
    selection_map = build_selection_map(salt=SALT, identity=identity, labels=labels)
    verify_selection(
        selection_map=selection_map,
        portable_selection=selection_map["portable_selection"],
        identity=identity,
    )
