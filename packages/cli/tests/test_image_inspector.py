import json
import warnings
from pathlib import Path

import pytest
from asset_mania.inspectors.image import inspect_image
from asset_mania_contracts import DiagnosticCode
from PIL import Image


def _save_image(path: Path, image_format: str) -> None:
    Image.new("RGBA", (8, 6), color=(20, 40, 60, 80)).save(path, format=image_format)


def _inject_photoshop_resource(path: Path, resource_id: int) -> None:
    resource_data = b"synthetic metadata"
    resource = (
        b"8BIM"
        + resource_id.to_bytes(2, "big")
        + b"\x00\x00"
        + len(resource_data).to_bytes(4, "big")
        + resource_data
        + (b"\x00" if len(resource_data) % 2 else b"")
    )
    payload = b"Photoshop 3.0\x00" + resource
    app13 = b"\xff\xed" + (len(payload) + 2).to_bytes(2, "big") + payload
    jpeg = path.read_bytes()
    path.write_bytes(jpeg[:2] + app13 + jpeg[2:])


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


@pytest.mark.parametrize(
    ("resource_id", "expected_iptc"),
    [(0x0404, True), (0x0405, False)],
)
def test_inspect_image_detects_only_the_iptc_photoshop_resource(
    tmp_path: Path, resource_id: int, expected_iptc: bool
) -> None:
    source = tmp_path / "private-app13.jpeg"
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(source)
    _inject_photoshop_resource(source, resource_id)

    report, diagnostics = inspect_image(source)

    assert report["metadata_blocks"]["iptc"] is expected_iptc
    assert diagnostics == []


@pytest.mark.parametrize("encoded_bit_depth", [1, 2, 4])
def test_inspect_image_reports_the_encoded_palette_png_bit_depth(
    tmp_path: Path, encoded_bit_depth: int
) -> None:
    source = tmp_path / f"private-palette-{encoded_bit_depth}.png"
    image = Image.new("P", (8, 6))
    image.putdata([index % (1 << encoded_bit_depth) for index in range(48)])
    image.save(source, bits=encoded_bit_depth)
    assert source.read_bytes()[24] == encoded_bit_depth

    report, diagnostics = inspect_image(source)

    assert report["bit_depth"] == encoded_bit_depth
    assert diagnostics == []


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


def test_inspect_image_promotes_decompression_bomb_warning_to_sanitized_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "private-warning-image.png"
    Image.new("RGB", (8, 6)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 24)

    with warnings.catch_warnings(record=True) as emitted_warnings:
        warnings.simplefilter("always")
        report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]
    assert emitted_warnings == []


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


def test_inspect_image_fully_decodes_a_structurally_valid_truncated_jpeg(tmp_path: Path) -> None:
    source = tmp_path / "private-truncated-image.jpeg"
    Image.new("RGB", (64, 64), color=(20, 40, 60)).save(source, quality=90)
    source.write_bytes(source.read_bytes()[:-2])

    with Image.open(source) as image:
        image.verify()
    with pytest.raises(OSError), Image.open(source) as image:
        image.load()

    report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]


def test_inspect_image_fully_decodes_a_corrupt_webp_frame(tmp_path: Path) -> None:
    source = tmp_path / "private-corrupt-frame.webp"
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (
                (x * 37 + y * 13) % 256,
                (x * 11 + y * 29) % 256,
                (x * 17 + y * 43) % 256,
            )
    image.save(source, "WEBP", quality=80, method=6)
    corrupted = bytearray(source.read_bytes())
    corrupted[21] ^= 0xFF
    source.write_bytes(corrupted)

    with Image.open(source) as opened:
        opened.verify()
    with pytest.raises(OSError), Image.open(source) as opened:
        opened.load()

    report, diagnostics = inspect_image(source)

    assert report == {}
    assert diagnostics == [DiagnosticCode.INPUT_UNREADABLE]
