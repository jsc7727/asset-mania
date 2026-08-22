"""Closed contract for the eight images in one turntable run."""

import copy

import pytest
from asset_mania_contracts import build_turntable_viewset, canonical_digest


def _views():
    records = []
    for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315), start=1):
        generated = yaw != 0
        records.append(
            {
                "label": f"view-{index}",
                "target_yaw": yaw,
                "pitch": 0,
                "roll": 0,
                "origin": "generated" if generated else "observed",
                "image_sha256": f"{index:02x}" * 32,
                "mask_sha256": f"{index + 16:02x}" * 32,
                "byte_size": 1000 + index,
                "media_type": "image/png",
                "width": 1024,
                "height": 1024,
                "provider_request_id": f"request-{index}" if generated else None,
                "reported_usage": {"total_tokens": 100 + index} if generated else {},
            }
        )
    return records


def _audit():
    return {
        "status": "passed",
        "diagnostics": [],
        "identity_consistency": "unmeasured",
        "metrics": {
            "minimum_foreground_coverage": 0.31,
            "maximum_foreground_coverage": 0.42,
            "maximum_centroid_offset": 0.03,
            "maximum_border_contact_ratio": 0.0,
            "minimum_adjacent_area_ratio": 0.82,
            "maximum_adjacent_area_ratio": 1.18,
        },
    }


def _build(**overrides):
    arguments = {
        "plan_sha256": "a1" * 32,
        "views": _views(),
        "audit": _audit(),
        "reported_usage": {"input_tokens": 700, "output_tokens": 1400},
        "actual_cost": "0.381000",
    }
    arguments.update(overrides)
    return build_turntable_viewset(**arguments)


def test_viewset_preserves_yaw_order_origins_and_unmeasured_identity(validator_for) -> None:
    viewset = _build()

    assert [item["target_yaw"] for item in viewset["views"]] == [
        0,
        45,
        90,
        135,
        180,
        225,
        270,
        315,
    ]
    assert viewset["views"][0]["origin"] == "observed"
    assert all(item["origin"] == "generated" for item in viewset["views"][1:])
    assert viewset["audit"]["identity_consistency"] == "unmeasured"
    assert list(validator_for("turntable-viewset", "1.0").iter_errors(viewset)) == []
    preimage = {key: value for key, value in viewset.items() if key != "viewset_sha256"}
    assert canonical_digest(preimage) == viewset["viewset_sha256"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reordered"])
def test_missing_duplicate_or_reordered_yaw_is_refused(mutation: str) -> None:
    views = _views()
    if mutation == "missing":
        views.pop()
    elif mutation == "duplicate":
        views[-1]["target_yaw"] = 270
    else:
        views[1], views[2] = views[2], views[1]

    with pytest.raises(ValueError, match="yaws"):
        _build(views=views)


def test_viewset_cannot_carry_a_private_path(validator_for) -> None:
    viewset = _build()
    mutated = copy.deepcopy(viewset)
    mutated["views"][0]["path"] = "C:/private/face.png"
    assert list(validator_for("turntable-viewset", "1.0").iter_errors(mutated))
