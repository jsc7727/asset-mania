"""The request shape is closed and derived from the approved plan."""

from pathlib import Path

import pytest
from asset_mania_pipeline import sha256_bytes
from asset_mania_provider_openai import normalization
from asset_mania_provider_openai.errors import PlanMismatch, RequestRejected
from asset_mania_provider_openai.transport import OFFICIAL_ENDPOINT
from conftest import PROMPT_TEXT

BASE_CONTROLS = {
    "n": 1,
    "size": "1024x1024",
    "quality": "medium",
    "background": "auto",
    "output_format": "png",
    "output_compression": None,
    "moderation": "auto",
}


# --- The bound endpoint and snapshot ------------------------------------------------


def test_the_endpoint_and_snapshot_are_bound(plan, prompt_file, attachments) -> None:
    built = plan()
    parts = normalization.load_attachments(built, run_directory=attachments["run_directory"])
    request = normalization.build_request(
        plan=built, prompt=PROMPT_TEXT, parts=parts, timeout_seconds=120
    )
    assert request.method == "POST"
    assert request.endpoint == OFFICIAL_ENDPOINT
    assert request.fields["model"] == normalization.MODEL_SNAPSHOT


def test_another_endpoint_is_refused(plan, prompt_file, attachments) -> None:
    built = {**plan(), "endpoint": "/v1/images/generations"}
    with pytest.raises(RequestRejected, match="binds /v1/images/edits"):
        normalization.build_request(plan=built, prompt=PROMPT_TEXT, parts=[], timeout_seconds=120)


def test_another_model_snapshot_is_refused(plan) -> None:
    built = {**plan(), "model": "gpt-image-2-2026-05-01"}
    with pytest.raises(RequestRejected, match="snapshot"):
        normalization.build_request(plan=built, prompt=PROMPT_TEXT, parts=[], timeout_seconds=120)


# --- Attachments --------------------------------------------------------------------


def test_the_four_attachments_bind_to_image_indices_zero_to_three(plan, attachments) -> None:
    built = plan()
    assert [item["role"] for item in built["attachments"]] == list(normalization.ATTACHMENT_ROLES)
    assert [item["index"] for item in built["attachments"]] == [0, 1, 2, 3]
    parts = normalization.load_attachments(built, run_directory=attachments["run_directory"])
    assert [part.field_name for part in parts] == [normalization.ATTACHMENT_FIELD] * 4


def test_the_optional_api_mask_part_is_absent(plan, attachments) -> None:
    """The binary mask is a visual reference image here, not an inpainting mask."""
    built = plan()
    parts = normalization.load_attachments(built, run_directory=attachments["run_directory"])
    request = normalization.build_request(
        plan=built, prompt=PROMPT_TEXT, parts=parts, timeout_seconds=120
    )
    assert "mask" not in request.fields
    assert all(part.field_name != "mask" for part in request.parts)


def test_an_attachment_whose_bytes_changed_is_refused(plan, attachments) -> None:
    built = plan()
    target = attachments["run_directory"] / built["attachment_paths"]["beauty"]
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"tampered" + bytes(16))
    with pytest.raises(PlanMismatch, match="approved bytes"):
        normalization.load_attachments(built, run_directory=attachments["run_directory"])


def test_a_missing_attachment_is_refused(plan, attachments) -> None:
    built = plan()
    (attachments["run_directory"] / built["attachment_paths"]["mask"]).unlink()
    with pytest.raises(RequestRejected, match="missing on disk"):
        normalization.load_attachments(built, run_directory=attachments["run_directory"])


def test_a_reordered_attachment_inventory_is_refused(plan, attachments) -> None:
    built = plan()
    reordered = list(built["attachments"])
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(RequestRejected, match="in that order"):
        normalization.load_attachments(
            {**built, "attachments": reordered}, run_directory=attachments["run_directory"]
        )


def test_an_attachment_using_another_multipart_field_is_refused(plan, attachments) -> None:
    built = plan()
    changed = [dict(item) for item in built["attachments"]]
    changed[2]["multipart_field"] = "reference[]"
    with pytest.raises(RequestRejected, match="image\\[\\] field"):
        normalization.load_attachments(
            {**built, "attachments": changed}, run_directory=attachments["run_directory"]
        )


def test_attachment_filenames_are_portable_labels(plan, attachments) -> None:
    built = plan()
    parts = normalization.load_attachments(built, run_directory=attachments["run_directory"])
    assert [part.filename for part in parts] == [
        "beauty.png",
        "depth_preview.png",
        "normal_preview.png",
        "mask.png",
    ]


# --- Closed controls -------------------------------------------------------------------


def test_the_base_controls_are_accepted() -> None:
    normalization.validate_controls(dict(BASE_CONTROLS))


@pytest.mark.parametrize("field", sorted(normalization.CLOSED_CONTROLS))
def test_a_missing_control_is_refused(field: str) -> None:
    controls = dict(BASE_CONTROLS)
    del controls[field]
    with pytest.raises(RequestRejected, match="missing provider controls"):
        normalization.validate_controls(controls)


def test_an_unknown_control_is_refused() -> None:
    with pytest.raises(RequestRejected, match="unknown provider controls"):
        normalization.validate_controls({**BASE_CONTROLS, "seed": 7})


@pytest.mark.parametrize("field", sorted(normalization.FORBIDDEN_CONTROLS))
def test_a_forbidden_control_is_refused(field: str) -> None:
    with pytest.raises(RequestRejected):
        normalization.validate_controls({**BASE_CONTROLS, field: "anything"})


def test_input_fidelity_is_refused_because_the_model_exposes_no_such_control() -> None:
    with pytest.raises(RequestRejected):
        normalization.validate_controls({**BASE_CONTROLS, "input_fidelity": "high"})


@pytest.mark.parametrize("count", [0, 2, 4])
def test_a_count_other_than_one_is_refused(count: int) -> None:
    with pytest.raises(RequestRejected, match="fixes n to 1"):
        normalization.validate_controls({**BASE_CONTROLS, "n": count})


@pytest.mark.parametrize("size", ["auto", "512x512", "2048x2048", "1024x1025"])
def test_a_size_without_a_published_cost_row_is_refused(size: str) -> None:
    with pytest.raises(RequestRejected, match="published cost row"):
        normalization.validate_controls({**BASE_CONTROLS, "size": size})


def test_quality_auto_is_refused() -> None:
    with pytest.raises(RequestRejected, match="published cost row"):
        normalization.validate_controls({**BASE_CONTROLS, "quality": "auto"})


def test_a_transparent_background_is_refused() -> None:
    with pytest.raises(RequestRejected, match="transparent background"):
        normalization.validate_controls({**BASE_CONTROLS, "background": "transparent"})


def test_compression_with_png_is_refused() -> None:
    with pytest.raises(RequestRejected, match="not allowed with PNG"):
        normalization.validate_controls({**BASE_CONTROLS, "output_compression": 80})


@pytest.mark.parametrize("value", [None, -1, 101, "80", True])
def test_compression_outside_the_integer_range_is_refused_for_jpeg(value) -> None:
    controls = {**BASE_CONTROLS, "output_format": "jpeg", "output_compression": value}
    with pytest.raises(RequestRejected, match="output_compression"):
        normalization.validate_controls(controls)


def test_valid_jpeg_compression_is_accepted() -> None:
    normalization.validate_controls(
        {**BASE_CONTROLS, "output_format": "jpeg", "output_compression": 80}
    )


def test_the_size_must_equal_the_conditioning_resolution() -> None:
    normalization.validate_size_matches_conditioning(BASE_CONTROLS, (1024, 1024))
    with pytest.raises(RequestRejected, match="conditioning resolution"):
        normalization.validate_size_matches_conditioning(BASE_CONTROLS, (1536, 1024))


# --- The prompt -------------------------------------------------------------------------


def test_the_approved_prompt_is_accepted(prompt_file: Path) -> None:
    digest = sha256_bytes(PROMPT_TEXT.encode("utf-8"))
    assert normalization.read_prompt(prompt_file, expected_sha256=digest) == PROMPT_TEXT


def test_a_changed_prompt_is_refused(prompt_file: Path) -> None:
    prompt_file.write_text("something else entirely\n", encoding="utf-8")
    digest = sha256_bytes(PROMPT_TEXT.encode("utf-8"))
    with pytest.raises(PlanMismatch, match="approved plan digest"):
        normalization.read_prompt(prompt_file, expected_sha256=digest)


def test_an_unreadable_prompt_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlanMismatch, match="could not be read"):
        normalization.read_prompt(tmp_path / "absent.txt", expected_sha256="0" * 64)


def test_the_prompt_travels_as_a_field_and_is_never_persisted_here(plan, attachments) -> None:
    built = plan()
    parts = normalization.load_attachments(built, run_directory=attachments["run_directory"])
    request = normalization.build_request(
        plan=built, prompt=PROMPT_TEXT, parts=parts, timeout_seconds=120
    )
    assert request.fields["prompt"] == PROMPT_TEXT
    assert "prompt" not in request.redacted()["fields"]
    assert "prompt_sha256" in request.redacted()["fields"]
