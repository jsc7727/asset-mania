"""Privacy-preserving image inspection."""

import warnings
from pathlib import Path

from asset_mania_contracts import DiagnosticCode
from PIL import Image, UnidentifiedImageError

_ORIENTATION_TAG = 274
_GPS_INFO_TAG = 34853
_IPTC_RESOURCE_ID = 0x0404
_ALLOWLISTED_EXIF_TAGS = {_ORIENTATION_TAG}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def inspect_image(path: Path) -> tuple[dict[str, object], list[DiagnosticCode]]:
    """Return only allowlisted image properties and metadata-presence diagnostics."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as source:
                source.verify()

            with Image.open(path) as image:
                image.load()
                exif = image.getexif()
                metadata_blocks = {
                    "exif": bool(exif),
                    "gps": _GPS_INFO_TAG in exif,
                    "icc": "icc_profile" in image.info,
                    "iptc": _has_iptc_resource(image.info),
                    "xmp": "xmp" in image.info,
                }
                report = {
                    "format": image.format,
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "bit_depth": _bit_depth(image, path),
                    "channels": len(image.getbands()),
                    "has_alpha": "A" in image.getbands() or "transparency" in image.info,
                    "orientation": _orientation(exif),
                    "metadata_blocks": metadata_blocks,
                }
                diagnostics = (
                    [DiagnosticCode.EXIF_SENSITIVE_METADATA_PRESENT]
                    if _has_sensitive_exif(exif)
                    else []
                )
                return report, diagnostics
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        SyntaxError,
        UnidentifiedImageError,
        ValueError,
    ):
        return {}, [DiagnosticCode.INPUT_UNREADABLE]


def _orientation(exif: Image.Exif) -> int | None:
    value = exif.get(_ORIENTATION_TAG)
    return value if isinstance(value, int) else None


def _has_sensitive_exif(exif: Image.Exif) -> bool:
    return any(tag not in _ALLOWLISTED_EXIF_TAGS for tag in exif)


def _has_iptc_resource(info: dict[str, object]) -> bool:
    photoshop = info.get("photoshop")
    return isinstance(photoshop, dict) and _IPTC_RESOURCE_ID in photoshop


def _bit_depth(image: Image.Image, path: Path) -> int:
    if image.format == "PNG":
        with path.open("rb") as source:
            header = source.read(25)
        if (
            len(header) != 25
            or header[:8] != _PNG_SIGNATURE
            or header[12:16] != b"IHDR"
            or header[24] not in {1, 2, 4, 8, 16}
        ):
            raise OSError("invalid PNG bit depth")
        return header[24]

    mode = image.mode
    if mode == "1":
        return 1
    if mode.startswith("I;16"):
        return 16
    if mode in {"I", "F"}:
        return 32
    return 8
