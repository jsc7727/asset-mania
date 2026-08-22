"""Local preparation and structural audit for generated turntables."""

import json
import shutil
from pathlib import Path

import pytest
from asset_mania_pipeline import (
    TurntableCandidate,
    audit_turntable,
    derive_white_background_mask,
    prepare_turntable_source,
    publish_turntable_viewset,
    sha256_file,
    write_contact_sheet,
)
from PIL import Image, ImageDraw


def _portrait_and_mask(directory: Path, *, size: tuple[int, int] = (64, 64)) -> tuple[Path, Path]:
    image_path = directory / "private-source.png"
    mask_path = directory / "private-mask.png"
    image = Image.new("RGB", size, (245, 245, 245))
    mask = Image.new("L", size, 0)
    for y in range(8, 58):
        for x in range(18, 46):
            image.putpixel((x, y), (80 + y, 40 + x, 30))
            mask.putpixel((x, y), 255)
    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def test_source_preparation_writes_an_rgba_cutout_without_changing_source(tmp_path: Path) -> None:
    source, mask = _portrait_and_mask(tmp_path)
    before = sha256_file(source)

    prepared = prepare_turntable_source(
        image_path=source,
        mask_path=mask,
        staging_root=tmp_path / "run",
    )

    assert prepared["source_image_sha256"] == before
    assert sha256_file(source) == before
    assert prepared["source_mask_sha256"] == sha256_file(prepared["normalized_mask"])
    assert prepared["cutout"].is_file()
    with Image.open(prepared["cutout"]) as cutout:
        assert cutout.mode == "RGBA"
        assert cutout.size == (64, 64)
        assert cutout.getpixel((0, 0)) == (0, 0, 0, 0)
        assert cutout.getpixel((32, 32))[3] == 255


def _candidate(directory: Path, index: int, yaw: int) -> TurntableCandidate:
    image_path = directory / f"view-{index}.png"
    mask_path = directory / f"mask-{index}.png"
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    mask = Image.new("L", (1024, 1024), 0)
    draw_image = ImageDraw.Draw(image)
    draw_mask = ImageDraw.Draw(mask)
    box = (290, 190, 734, 834)
    colour = (70 + index, 90 + index * 2, 120 + index * 3)
    draw_image.ellipse(box, fill=colour)
    draw_mask.ellipse(box, fill=255)
    image.save(image_path)
    mask.save(mask_path)
    generated = yaw != 0
    return TurntableCandidate(
        yaw=yaw,
        origin="generated" if generated else "observed",
        image_path=image_path,
        mask_path=mask_path,
        provider_request_id=f"request-{index}" if generated else None,
        reported_usage={"total_tokens": 100 + index} if generated else {},
        actual_cost="0.053000" if generated else None,
    )


def _turntable(directory: Path) -> list[TurntableCandidate]:
    return [
        _candidate(directory, index, yaw)
        for index, yaw in enumerate((0, 45, 90, 135, 180, 225, 270, 315), start=1)
    ]


def test_white_background_mask_keeps_only_the_edge_disconnected_subject(tmp_path: Path) -> None:
    source = tmp_path / "generated.png"
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    ImageDraw.Draw(image).ellipse((290, 190, 734, 834), fill=(20, 40, 60))
    image.save(source)

    mask_path = derive_white_background_mask(source, tmp_path / "mask.png")

    with Image.open(mask_path) as mask:
        assert mask.mode == "L"
        assert mask.getpixel((0, 0)) == 0
        assert mask.getpixel((512, 512)) == 255


def test_structural_audit_accepts_a_complete_centered_turntable(tmp_path: Path) -> None:
    result = audit_turntable(_turntable(tmp_path))

    assert result["status"] == "passed"
    assert result["diagnostics"] == []
    assert result["identity_consistency"] == "unmeasured"
    assert 0.20 <= result["metrics"]["minimum_foreground_coverage"]
    assert result["metrics"]["maximum_foreground_coverage"] <= 0.75


@pytest.mark.parametrize(
    "mutation",
    ["missing_yaw", "duplicate_pixels", "off_center", "border_contact", "area_jump"],
)
def test_structural_audit_fails_closed(tmp_path: Path, mutation: str) -> None:
    candidates = _turntable(tmp_path)
    if mutation == "missing_yaw":
        candidates.pop()
    elif mutation == "duplicate_pixels":
        shutil.copyfile(candidates[1].image_path, candidates[2].image_path)
    elif mutation == "off_center":
        mask = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask).ellipse((20, 205, 444, 819), fill=255)
        mask.save(candidates[3].mask_path)
    elif mutation == "border_contact":
        mask = Image.new("L", (1024, 1024), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((290, 190, 734, 834), fill=255)
        draw.rectangle((0, 0, 1023, 1023), outline=255, width=1)
        mask.save(candidates[4].mask_path)
    else:
        mask = Image.new("L", (1024, 1024), 0)
        ImageDraw.Draw(mask).ellipse((440, 430, 584, 594), fill=255)
        mask.save(candidates[5].mask_path)

    result = audit_turntable(candidates)

    assert result["status"] == "failed"
    assert result["diagnostics"] == ["VIEWSET_INCONSISTENT"]
    assert result["identity_consistency"] == "unmeasured"


def test_contact_sheet_and_viewset_are_portable(tmp_path: Path) -> None:
    candidates = _turntable(tmp_path)
    audit = audit_turntable(candidates)

    contact_sheet = write_contact_sheet(candidates, tmp_path / "contact-sheet.png")
    viewset = publish_turntable_viewset(
        plan_sha256="a1" * 32,
        candidates=candidates,
        audit=audit,
        actual_cost="0.371000",
    )

    with Image.open(contact_sheet) as sheet:
        assert sheet.size == (1024, 512)
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
    assert viewset["reported_usage"]["total_tokens"] == sum(range(102, 109))
    rendered = json.dumps(viewset)
    assert str(tmp_path) not in rendered
    assert "private-source" not in rendered


def test_failed_audit_cannot_publish_a_viewset(tmp_path: Path) -> None:
    candidates = _turntable(tmp_path)
    candidates.pop()
    audit = audit_turntable(candidates)

    with pytest.raises(ValueError, match="passed audit"):
        publish_turntable_viewset(
            plan_sha256="a1" * 32,
            candidates=candidates,
            audit=audit,
            actual_cost=None,
        )
