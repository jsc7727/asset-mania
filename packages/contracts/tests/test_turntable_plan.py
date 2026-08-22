"""Closed contract for one approved GPT Image 2 turntable run."""

from asset_mania_contracts import TURNTABLE_YAWS, build_turntable_plan, canonical_digest


def _build(**overrides):
    arguments = {
        "source_image_sha256": "a1" * 32,
        "source_width": 1024,
        "source_height": 1024,
        "source_mask_sha256": "a2" * 32,
        "prompt_sha256": "a3" * 32,
        "provider_evidence_sha256": "a4" * 32,
        "controls": {
            "size": "1024x1024",
            "quality": "medium",
            "background": "opaque",
            "output_format": "png",
            "moderation": "auto",
        },
        "subject": "real_person",
        "estimated_cost": "0.371000",
        "maximum_cost": "0.700000",
    }
    arguments.update(overrides)
    return build_turntable_plan(**arguments)


def test_real_person_plan_is_fixed_to_the_full_profile(validator_for) -> None:
    """A missing angle, gate, or pinned provider field must invalidate the run."""
    plan = _build()

    assert plan["yaws"] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert TURNTABLE_YAWS == (0, 45, 90, 135, 180, 225, 270, 315)
    assert plan["provider"] == "openai"
    assert plan["endpoint"] == "/v1/images/edits"
    assert plan["model"] == "gpt-image-2-2026-04-21"
    assert plan["required_gates"] == ["face_rights", "external_egress", "paid_compute"]
    assert list(validator_for("turntable-plan", "1.0").iter_errors(plan)) == []
    preimage = {key: value for key, value in plan.items() if key != "plan_sha256"}
    assert canonical_digest(preimage) == plan["plan_sha256"]
