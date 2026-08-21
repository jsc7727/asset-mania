"""Parent and artifact lineage verification before any worker or provider runs."""

import json
from pathlib import Path

import pytest
from asset_mania_contracts import canonical_digest, canonical_json
from asset_mania_pipeline import (
    ArtifactMismatch,
    ParentMismatch,
    PathEscape,
    describe_artifact,
    inherit_content_origin,
    inherit_rights_basis,
    load_parent,
    parent_reference,
    verify_consumed_artifact,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"


def _example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def _write_run(directory: Path, manifest: dict) -> Path:
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    return directory / "manifest.json"


def test_a_matching_parent_loads_with_its_declared_relationship(tmp_path: Path) -> None:
    manifest = _example("manifest-v2-scene-plan")
    path = _write_run(tmp_path / "run-plan-1", manifest)

    parent = load_parent(
        path,
        expected_sha256=canonical_digest(manifest),
        relationship="conditioned_from",
    )
    assert parent.run_id == manifest["run_id"]
    assert parent.document == manifest
    assert parent_reference(parent) == {
        "run_id": manifest["run_id"],
        "manifest_sha256": canonical_digest(manifest),
        "relationship": "conditioned_from",
    }


def test_a_tampered_parent_manifest_fails_before_execution(tmp_path: Path) -> None:
    manifest = _example("manifest-v2-scene-plan")
    path = _write_run(tmp_path / "run-plan-1", manifest)
    expected = canonical_digest(manifest)

    tampered = {**manifest, "parameters": {**manifest["parameters"], "frame": 13}}
    path.write_text(canonical_json(tampered), encoding="utf-8")

    with pytest.raises(ParentMismatch, match="PARENT_MANIFEST_MISMATCH"):
        load_parent(path, expected_sha256=expected, relationship="conditioned_from")


def test_a_reserialized_parent_still_verifies(tmp_path: Path) -> None:
    """Canonical hashing is over the logical object, not the file's whitespace."""
    manifest = _example("manifest-v2-scene-plan")
    path = _write_run(tmp_path / "run-plan-1", manifest)
    path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")

    parent = load_parent(
        path,
        expected_sha256=canonical_digest(manifest),
        relationship="conditioned_from",
    )
    assert parent.document == manifest


def test_a_missing_or_unreadable_parent_fails(tmp_path: Path) -> None:
    manifest = _example("manifest-v2-scene-plan")
    with pytest.raises(ParentMismatch, match="PARENT_MANIFEST_MISMATCH"):
        load_parent(
            tmp_path / "absent" / "manifest.json",
            expected_sha256=canonical_digest(manifest),
            relationship="conditioned_from",
        )

    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ParentMismatch, match="PARENT_MANIFEST_MISMATCH"):
        load_parent(
            broken / "manifest.json",
            expected_sha256=canonical_digest(manifest),
            relationship="conditioned_from",
        )


def test_an_undeclared_relationship_is_rejected(tmp_path: Path) -> None:
    manifest = _example("manifest-v2-scene-plan")
    path = _write_run(tmp_path / "run-plan-1", manifest)
    with pytest.raises(ValueError, match="relationship"):
        load_parent(
            path,
            expected_sha256=canonical_digest(manifest),
            relationship="inspired_by",
        )


def test_a_consumed_artifact_is_rehashed_from_the_parent_run(tmp_path: Path) -> None:
    run = tmp_path / "run-condition-1"
    (run / "passes").mkdir(parents=True)
    payload = b"synthetic beauty pass\n"
    (run / "passes" / "beauty.exr").write_bytes(payload)

    from asset_mania_pipeline import sha256_bytes

    artifact = {
        "path": "passes/beauty.exr",
        "sha256": sha256_bytes(payload),
        "byte_size": len(payload),
    }
    verify_consumed_artifact(run, artifact)

    (run / "passes" / "beauty.exr").write_bytes(payload + b"tampered")
    with pytest.raises(ArtifactMismatch, match="PARENT_MANIFEST_MISMATCH"):
        verify_consumed_artifact(run, artifact)


def test_a_consumed_artifact_path_may_not_leave_the_parent_run(tmp_path: Path) -> None:
    run = tmp_path / "run-condition-1"
    run.mkdir()
    for path in ("../outside.exr", "/etc/passwd", "passes/../../outside.exr"):
        with pytest.raises(PathEscape):
            verify_consumed_artifact(run, {"path": path, "sha256": "0" * 64, "byte_size": 0})


def test_a_consumed_artifact_symlink_out_of_the_run_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run-condition-1"
    run.mkdir()
    outside = tmp_path / "outside.exr"
    outside.write_bytes(b"outside\n")
    (run / "linked.exr").symlink_to(outside)

    from asset_mania_pipeline import sha256_bytes

    with pytest.raises(PathEscape):
        verify_consumed_artifact(
            run,
            {"path": "linked.exr", "sha256": sha256_bytes(b"outside\n"), "byte_size": 8},
        )


def test_generated_origin_dominates_derived_and_observed() -> None:
    assert inherit_content_origin("derived", ["observed"]) == "derived"
    assert inherit_content_origin("derived", ["generated"]) == "generated"
    assert inherit_content_origin("derived", ["observed", "generated"]) == "generated"
    assert inherit_content_origin("generated", ["observed"]) == "generated"
    assert inherit_content_origin("observed", []) == "observed"


def test_an_unknown_parent_origin_never_silently_becomes_derived() -> None:
    assert inherit_content_origin("derived", ["unknown"]) == "unknown"
    assert inherit_content_origin("derived", ["unknown", "generated"]) == "generated"


def test_describe_artifact_hashes_the_staged_file_and_keeps_the_path_relative(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    (staging / "exports").mkdir(parents=True)
    payload = b"glTF binary stand-in\n"
    (staging / "exports" / "asset.glb").write_bytes(payload)

    from asset_mania_pipeline import sha256_bytes

    record = describe_artifact(
        role="scene_glb",
        staging_root=staging,
        path=staging / "exports" / "asset.glb",
        media_type="model/gltf-binary",
        parents=[{"sha256": "a" * 64, "relationship": "generated_from"}],
        operation="export",
        content_origin="generated",
        sensitivity="user-content",
        upload_eligible=False,
        validation={
            "profile": "gltf-validator-v1",
            "status": "valid",
            "diagnostics": [],
            "semantic_digest": None,
        },
    )
    assert record["path"] == "exports/asset.glb"
    assert record["sha256"] == sha256_bytes(payload)
    assert record["byte_size"] == len(payload)
    assert record["content_origin"] == "generated"
    assert str(tmp_path) not in canonical_json(record)


def test_rights_basis_is_inherited_from_the_condition_run_rather_than_reused() -> None:
    condition = _example("manifest-v2-condition")
    generate = _example("manifest-v2-provider-generate")

    assert inherit_rights_basis([condition]) == []
    assert inherit_rights_basis([generate]) == ["external_egress", "paid_compute"]
    assert inherit_rights_basis([condition, generate]) == ["external_egress", "paid_compute"]
