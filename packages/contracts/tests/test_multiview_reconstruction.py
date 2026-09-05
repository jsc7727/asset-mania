"""Closed contract for one fused eight-view reconstruction."""

import pytest
from asset_mania_contracts import build_multiview_reconstruction_record, canonical_digest


def _meshes():
    return [
        {
            "label": f"mesh-{index}",
            "target_yaw": yaw,
            "sha256": f"{index:02x}" * 32,
            "triangle_count": 1000 + index,
            "vertex_count": 500 + index,
            "manifold": "closed",
        }
        for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315), start=1)
    ]


def _fusion():
    return {
        "normalization": "bounds_center_unit_longest_extent",
        "yaw_axis": "+Z",
        "grid_resolution": 192,
        "minimum_votes": 4,
        "eligible_mesh_count": 8,
        "input_mesh_count": 8,
    }


def _fused_mesh():
    return {
        "role": "fused_mesh",
        "path": "fused.glb",
        "sha256": "f1" * 32,
        "byte_size": 4096,
        "media_type": "model/gltf-binary",
        "triangle_count": 24000,
        "vertex_count": 12002,
        "manifold": "closed",
        "signed_volume": 0.31,
        "content_origin": "generated",
        "sensitivity": "user-content",
        "upload_eligible": False,
    }


def _build(**overrides):
    arguments = {
        "turntable_plan_sha256": "a1" * 32,
        "viewset_sha256": "a2" * 32,
        "observed_source_image_sha256": "a3" * 32,
        "meshes": _meshes(),
        "fusion": _fusion(),
        "fused_mesh": _fused_mesh(),
        "subject": "real_person",
        "rights_receipt_sha256": "a4" * 32,
    }
    arguments.update(overrides)
    return build_multiview_reconstruction_record(**arguments)


def test_record_seals_eight_meshes_fusion_and_likeness_disclosure(validator_for) -> None:
    record = _build()

    assert [item["target_yaw"] for item in record["meshes"]] == [
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
    ]
    assert record["fusion"]["minimum_votes"] == 4
    assert record["fused_mesh"]["manifold"] == "closed"
    assert record["disclosure"]["likeness_basis"] == {"views": 8, "inferred": True}
    assert record["disclosure"]["source_image_sha256"] == "a3" * 32
    assert list(validator_for("multiview-reconstruction", "1.0").iter_errors(record)) == []
    preimage = {key: value for key, value in record.items() if key != "record_sha256"}
    assert canonical_digest(preimage) == record["record_sha256"]


def test_fewer_than_six_closed_meshes_is_refused() -> None:
    meshes = _meshes()
    for mesh in meshes[:3]:
        mesh["manifold"] = "open"
    with pytest.raises(ValueError, match="six closed"):
        _build(meshes=meshes)


def test_nonpositive_fused_volume_is_refused() -> None:
    fused = _fused_mesh()
    fused["signed_volume"] = 0.0
    with pytest.raises(ValueError, match="positive-volume"):
        _build(fused_mesh=fused)
