import json
from pathlib import Path

import pytest
from asset_mania.inspectors.image import inspect_image
from asset_mania_contracts import DiagnosticCode
from PIL import Image


def _save_image(path: Path, image_format: str) -> None:
    Image.new("RGBA", (8, 6), color=(20, 40, 60, 80)).save(path, format=image_format)


@pytest.mark.parametrize(
    ("image_format", "suffix"),
    [("PNG", ".png"), ("WEBP", ".webp")],
)
def test_inspect_image_emits_allowlisted_properties(
    tmp_path: Path, image_format: str, suffix: str
) -> None:
    source = tmp_path / f"private-source{suffix}"
    _save_image(source, image_format)

    report, diagnostics = inspect_image(source)

    assert report == {
        "format": image_format,
        "width": 8,
        "height": 6,
        "mode": "RGBA",
        "bit_depth": 8,
        "channels": 4,
        "has_alpha": True,
        "orientation": None,
        "metadata_blocks": {
            "exif": False,
            "gps": False,
            "icc": False,
            "iptc": False,
            "xmp": False,
        },
    }
    assert diagnostics == []
    assert source.name not in json.dumps(report)


def test_inspect_image_reports_only_presence_for_sensitive_jpeg_metadata(tmp_path: Path) -> None:
    source = tmp_path / "private-camera-upload.jpeg"
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "camera-owner@example.com"
    exif[272] = "Private Camera"
    exif[306] = "2026:08:19 12:00:00"
    exif[34853] = {1: "N", 2: (1, 1)}
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(source, exif=exif)

    report, diagnostics = inspect_image(source)

    assert report == {
        "format": "JPEG",
        "width": 8,
        "height": 6,
        "mode": "RGB",
        "bit_depth": 8,
        "channels": 3,
        "has_alpha": False,
        "orientation": 6,
        "metadata_blocks": {
            "exif": True,
            "gps": True,
            "icc": False,
            "iptc": False,
            "xmp": False,
        },
    }
    assert "camera-owner@example.com" not in json.dumps(report)
    assert diagnostics == [DiagnosticCode.EXIF_SENSITIVE_METADATA_PRESENT]


def test_inspect_image_detects_palette_transparency_as_alpha(tmp_path: Path) -> None:
    source = tmp_path / "private-palette.png"
    image = Image.new("P", (8, 6))
    image.info["transparency"] = 0
    image.save(source)

    report, diagnostics = inspect_image(source)

    assert report["mode"] == "P"
    assert report["channels"] == 1
    assert report["has_alpha"] is True
    assert diagnostics == []
    assert source.name not in json.dumps(report)


def test_inspect_image_sanitizes_decompression_bomb_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "private-oversized-image.png"
    Image.new("RGB", (8, 6)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)

    report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]
    assert source.name not in json.dumps(report)


def test_inspect_image_sanitizes_corrupt_input_error(tmp_path: Path) -> None:
    source = tmp_path / "private-corrupt-image.png"
    source.write_bytes(b"not a PNG")

    report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]
    assert source.name not in json.dumps(report)


def test_inspect_image_sanitizes_a_structurally_valid_truncated_png(tmp_path: Path) -> None:
    source = tmp_path / "private-truncated-image.png"
    Image.new("RGB", (8, 6)).save(source)
    source.write_bytes(source.read_bytes()[:-12])

    with Image.open(source) as image:
        assert image.format == "PNG"

    report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]
    assert source.name not in json.dumps(report)
