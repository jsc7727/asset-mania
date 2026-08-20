"""View ingest accepts one image shape and rejects everything else."""

import io
from pathlib import Path

import pytest
from asset_mania_contracts import canonical_digest, load_schema
from asset_mania_pipeline import (
    ACCEPTED_FORMATS,
    MAX_INPUT_BYTES,
    SubjectDeclarationRequired,
    ViewRejected,
    alignment_acknowledgement_text,
    build_alignment_attestation,
    ingest_view,
    normalize_pixels,
    sha256_file,
)
from jsonschema import Draft202012Validator
from PIL import Image

RESOLUTION = (64, 64)
CONDITION = "b3" * 32
BUNDLE = "c1" * 32
CAMERA = "c5" * 32
ISSUED_AT = "2026-08-19T09:20:00Z"


def _write(path: Path, image: Image.Image, image_format: str = "PNG", **options) -> Path:
    image.save(path, format=image_format, **options)
    return path


def _rgba(size=RESOLUTION, transparent_corner: bool = False) -> Image.Image:
    image = Image.new("RGBA", size, (10, 120, 200, 255))
    if transparent_corner:
        for x in range(4):
            for y in range(4):
                image.putpixel((x, y), (200, 30, 40, 0))
    return image


def _ingest(path: Path, staging: Path, **overrides):
    digest = sha256_file(path) if path.is_file() and path.stat().st_size else "0" * 64
    arguments = {
        "image_path": path,
        "staging_root": staging,
        "resolution": RESOLUTION,
        "condition_manifest_sha256": CONDITION,
        "conditioning_bundle_sha256": BUNDLE,
        "camera_digest": CAMERA,
        "origin": "observed",
        "subject": "synthetic_person",
        "alignment_acknowledgement": alignment_acknowledgement_text(CONDITION, digest),
        "issued_at": ISSUED_AT,
    }
    arguments.update(overrides)
    return ingest_view(**arguments)


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    path = tmp_path / "staging"
    path.mkdir()
    return path


# --- Accepted shapes ------------------------------------------------------------


@pytest.mark.parametrize("image_format", sorted(ACCEPTED_FORMATS))
def test_every_accepted_format_is_ingested(tmp_path, staging, image_format: str) -> None:
    suffix = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[image_format]
    image = _rgba() if image_format != "JPEG" else Image.new("RGB", RESOLUTION, (10, 120, 200))
    path = _write(tmp_path / f"view.{suffix}", image, image_format)

    result = _ingest(path, staging)
    assert result["view"]["width"] == 64
    assert result["view"]["media_type"] == "image/png"
    assert result["view"]["color_space"] == "srgb"


def test_an_rgb_input_records_no_alpha(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", Image.new("RGB", RESOLUTION, (1, 2, 3)))
    assert _ingest(path, staging)["view"]["alpha"] == "none"


def test_an_rgba_input_records_straight_alpha(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    assert _ingest(path, staging)["view"]["alpha"] == "straight"


def test_the_view_validates_against_the_committed_schema(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    view = _ingest(path, staging)["view"]
    validator = Draft202012Validator(load_schema("view", "1.0"))
    assert list(validator.iter_errors(view)) == []


def test_the_view_is_self_sealed(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    view = _ingest(path, staging)["view"]
    preimage = {key: value for key, value in view.items() if key != "view_sha256"}
    assert canonical_digest(preimage) == view["view_sha256"]


def test_a_view_is_never_upload_eligible(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    assert _ingest(path, staging)["view"]["upload_eligible"] is False


# --- Normalization ---------------------------------------------------------------


def test_the_original_image_is_not_rewritten(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    before = sha256_file(path)
    _ingest(path, staging)
    assert sha256_file(path) == before


def test_the_normalized_copy_lands_in_the_run(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    result = _ingest(path, staging)
    normalized = result["normalized_path"]
    assert normalized == staging / "view.png"
    assert normalized.is_file()
    assert result["normalized_sha256"] == sha256_file(normalized)


def test_hidden_rgb_under_transparent_pixels_is_zeroed(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba(transparent_corner=True))
    result = _ingest(path, staging)
    with Image.open(result["normalized_path"]) as normalized:
        normalized.load()
        assert normalized.getpixel((0, 0)) == (0, 0, 0, 0)
        assert normalized.getpixel((32, 32))[3] == 255


def test_normalize_pixels_zeroes_only_fully_transparent_pixels() -> None:
    image = Image.new("RGBA", (2, 1), (9, 9, 9, 255))
    image.putpixel((0, 0), (7, 7, 7, 0))
    pixels, alpha = normalize_pixels(image)
    assert alpha == "straight"
    assert tuple(pixels[0:4]) == (0, 0, 0, 0)
    assert tuple(pixels[4:8]) == (9, 9, 9, 255)


def test_metadata_is_stripped_from_the_normalized_copy(tmp_path, staging) -> None:
    path = tmp_path / "view.png"
    _rgba().save(path, format="PNG", pnginfo=None)
    result = _ingest(path, staging)
    with Image.open(result["normalized_path"]) as normalized:
        normalized.load()
        for key in ("exif", "icc_profile", "XML:com.adobe.xmp"):
            assert not normalized.info.get(key)


def test_the_semantic_digest_covers_the_decoded_pixels(tmp_path, staging) -> None:
    first = _write(tmp_path / "a.png", _rgba())
    second = _write(tmp_path / "b.webp", _rgba(), "WEBP", lossless=True)
    one = _ingest(first, staging)
    two = _ingest(second, tmp_path / "staging-two")
    assert one["view"]["validation"]["semantic_digest"] == two["decoded_sha256"]


# --- Rejected shapes -------------------------------------------------------------


@pytest.mark.parametrize("mode", ["L", "LA", "P", "CMYK", "I;16", "F"])
def test_an_unsupported_mode_is_rejected(tmp_path, staging, mode: str) -> None:
    container = {"CMYK": "view.jpg", "I;16": "view.tiff", "F": "view.tiff"}.get(mode, "view.png")
    path = tmp_path / container
    Image.new(mode, RESOLUTION).save(path)
    with pytest.raises(ViewRejected) as rejection:
        _ingest(path, staging)
    assert rejection.value.diagnostic in ("UNSUPPORTED_MEDIA_TYPE", "INPUT_UNREADABLE")


def test_an_unsupported_container_is_rejected(tmp_path, staging) -> None:
    path = tmp_path / "view.bmp"
    Image.new("RGB", RESOLUTION).save(path, format="BMP")
    with pytest.raises(ViewRejected, match="UNSUPPORTED_MEDIA_TYPE"):
        _ingest(path, staging)


@pytest.mark.parametrize("size", [(63, 64), (64, 63), (128, 128), (32, 32)])
def test_a_mismatched_resolution_is_rejected_without_resizing(tmp_path, staging, size) -> None:
    path = _write(tmp_path / "view.png", _rgba(size))
    with pytest.raises(ViewRejected, match="never resizes or crops"):
        _ingest(path, staging)
    assert not (staging / "view.png").exists()


@pytest.mark.parametrize("orientation", [2, 3, 6, 8])
def test_a_rotating_exif_orientation_is_rejected(tmp_path, staging, orientation: int) -> None:
    path = tmp_path / "view.jpg"
    image = Image.new("RGB", RESOLUTION, (5, 5, 5))
    exif = image.getexif()
    exif[0x0112] = orientation
    image.save(path, format="JPEG", exif=exif)

    with pytest.raises(ViewRejected, match="VIEW_ALIGNMENT_MISMATCH"):
        _ingest(path, staging)


def test_orientation_one_is_accepted(tmp_path, staging) -> None:
    path = tmp_path / "view.jpg"
    image = Image.new("RGB", RESOLUTION, (5, 5, 5))
    exif = image.getexif()
    exif[0x0112] = 1
    image.save(path, format="JPEG", exif=exif)
    assert _ingest(path, staging)["view"]["width"] == 64


def test_sensitive_metadata_is_reported_rather_than_dropped(tmp_path, staging) -> None:
    path = tmp_path / "view.jpg"
    image = Image.new("RGB", RESOLUTION, (5, 5, 5))
    exif = image.getexif()
    exif[0x010F] = "PrivateCameraMake"
    exif[0x0132] = "2026:08:19 09:20:00"
    image.save(path, format="JPEG", exif=exif)

    with pytest.raises(ViewRejected, match="EXIF_SENSITIVE_METADATA_PRESENT"):
        _ingest(path, staging)


def test_an_unknown_icc_profile_is_rejected(tmp_path, staging) -> None:
    path = tmp_path / "view.png"
    _rgba().save(path, format="PNG", icc_profile=b"not a real profile")
    with pytest.raises(ViewRejected, match="sRGB ICC profile"):
        _ingest(path, staging)


def test_a_truncated_file_is_rejected(tmp_path, staging) -> None:
    buffer = io.BytesIO()
    _rgba().save(buffer, format="PNG")
    path = tmp_path / "view.png"
    path.write_bytes(buffer.getvalue()[: len(buffer.getvalue()) // 2])
    with pytest.raises(ViewRejected):
        _ingest(path, staging)


def test_a_non_image_file_is_rejected(tmp_path, staging) -> None:
    path = tmp_path / "view.png"
    path.write_bytes(b"not an image at all")
    with pytest.raises(ViewRejected, match="INPUT_UNREADABLE"):
        _ingest(path, staging)


def test_an_empty_file_is_rejected(tmp_path, staging) -> None:
    path = tmp_path / "view.png"
    path.write_bytes(b"")
    with pytest.raises(ViewRejected, match="INPUT_UNREADABLE"):
        _ingest(path, staging)


def test_a_missing_file_is_rejected(tmp_path, staging) -> None:
    with pytest.raises(ViewRejected, match="INPUT_UNREADABLE"):
        _ingest(tmp_path / "absent.png", staging)


def test_an_oversized_file_is_rejected(tmp_path, staging, monkeypatch) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    monkeypatch.setattr("asset_mania_pipeline.views.MAX_INPUT_BYTES", 16)
    with pytest.raises(ViewRejected, match="exceeds"):
        _ingest(path, staging)


def test_a_decompression_bomb_is_rejected(tmp_path, staging, monkeypatch) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    monkeypatch.setattr("asset_mania_pipeline.views.MAX_INPUT_PIXELS", 4)
    with pytest.raises(ViewRejected):
        _ingest(path, staging)


def test_the_byte_limit_is_far_above_a_legitimate_view() -> None:
    assert MAX_INPUT_BYTES >= 16 * 1024 * 1024


# --- Declarations and alignment ---------------------------------------------------


def test_an_unknown_subject_is_blocked(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    with pytest.raises(SubjectDeclarationRequired, match="SUBJECT_DECLARATION_REQUIRED"):
        _ingest(path, staging, subject="unknown")


@pytest.mark.parametrize("origin", ["observed", "generated", "unknown"])
def test_every_declared_origin_survives_without_pixel_classification(
    tmp_path, staging, origin: str
) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    assert _ingest(path, staging, origin=origin)["view"]["origin"] == origin


def test_an_undeclared_origin_is_rejected(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    with pytest.raises(ViewRejected, match="origin"):
        _ingest(path, staging, origin="probably_a_render")


@pytest.mark.parametrize("wrong", ["yes", "true", "", CONDITION, "aligned"])
def test_a_wrong_alignment_acknowledgement_is_rejected(tmp_path, staging, wrong: str) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    with pytest.raises(ViewRejected, match="CONDITION_SHA256:VIEW_SHA256"):
        _ingest(path, staging, alignment_acknowledgement=wrong)


@pytest.mark.parametrize("boolean", [True, False, 1, None])
def test_a_boolean_can_never_attest_alignment(tmp_path, staging, boolean) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    with pytest.raises(ViewRejected, match="CONDITION_SHA256:VIEW_SHA256"):
        _ingest(path, staging, alignment_acknowledgement=boolean)


def test_an_acknowledgement_for_another_condition_is_rejected(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    other = alignment_acknowledgement_text("f" * 64, sha256_file(path))
    with pytest.raises(ViewRejected, match="CONDITION_SHA256:VIEW_SHA256"):
        _ingest(path, staging, alignment_acknowledgement=other)


def test_alignment_stays_declared_unverified_for_normal_input(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    alignment = _ingest(path, staging)["view"]["alignment"]
    assert alignment["status"] == "declared_unverified"
    assert alignment["transform"] == "identity"
    assert alignment["issuer"] == "user"


def test_a_provider_issued_alignment_is_supported(tmp_path, staging) -> None:
    path = _write(tmp_path / "view.png", _rgba())
    view = _ingest(path, staging, alignment_issuer="provider", origin="generated")["view"]
    assert view["alignment"]["issuer"] == "provider"
    assert view["origin"] == "generated"


def test_only_the_user_or_provider_may_attest_alignment() -> None:
    for issuer in ("maintainer", "system", "tool"):
        with pytest.raises(ViewRejected, match="cannot issue an alignment claim"):
            build_alignment_attestation(
                condition_manifest_sha256=CONDITION,
                view_image_sha256="a" * 64,
                issuer=issuer,
                issued_at=ISSUED_AT,
            )


def test_the_attestation_binds_the_condition_and_the_view() -> None:
    baseline = build_alignment_attestation(
        condition_manifest_sha256=CONDITION,
        view_image_sha256="a" * 64,
        issuer="user",
        issued_at=ISSUED_AT,
    )
    assert baseline["statement"] == "declared_aligned"
    for key, value in (
        ("condition_manifest_sha256", "f" * 64),
        ("view_image_sha256", "b" * 64),
        ("issuer", "provider"),
        ("issued_at", "2026-08-19T10:20:00Z"),
    ):
        changed = build_alignment_attestation(
            **{
                "condition_manifest_sha256": CONDITION,
                "view_image_sha256": "a" * 64,
                "issuer": "user",
                "issued_at": ISSUED_AT,
                key: value,
            }
        )
        assert changed["attestation_sha256"] != baseline["attestation_sha256"], key


def test_dimensions_alone_never_claim_verified_alignment(tmp_path, staging) -> None:
    """A same-sized image is not evidence of correspondence."""
    path = _write(tmp_path / "view.png", _rgba())
    assert _ingest(path, staging)["view"]["alignment"]["status"] != "verified_fixture"
