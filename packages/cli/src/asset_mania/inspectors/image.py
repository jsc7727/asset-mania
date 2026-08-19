"""Privacy-preserving image inspection."""

from pathlib import Path

from asset_mania_contracts import DiagnosticCode
from PIL import Image, UnidentifiedImageError

_ORIENTATION_TAG = 274
_GPS_INFO_TAG = 34853
_ALLOWLISTED_EXIF_TAGS = {_ORIENTATION_TAG}


def inspect_image(path: Path) -> tuple[dict[str, object], list[DiagnosticCode]]:
    """Return only allowlisted image properties and metadata-presence diagnostics."""
    try:
        with Image.open(path) as source:
            source.verify()

        with Image.open(path) as image:
            exif = image.getexif()
            metadata_blocks = {
                "exif": bool(exif),
                "gps": _GPS_INFO_TAG in exif,
                "icc": "icc_profile" in image.info,
                "iptc": "iptc" in image.info or "photoshop" in image.info,
                "xmp": "xmp" in image.info,
            }
            report = {
                "format": image.format,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "bit_depth": _bit_depth(image.mode),
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
    except (Image.DecompressionBombError, OSError, SyntaxError, UnidentifiedImageError, ValueError):
        return {}, [DiagnosticCode.INPUT_UNREADABLE]


def _orientation(exif: Image.Exif) -> int | None:
    value = exif.get(_ORIENTATION_TAG)
    return value if isinstance(value, int) else None


def _has_sensitive_exif(exif: Image.Exif) -> bool:
    return any(tag not in _ALLOWLISTED_EXIF_TAGS for tag in exif)


def _bit_depth(mode: str) -> int:
    if mode == "1":
        return 1
    if mode.startswith("I;16"):
        return 16
    if mode in {"I", "F"}:
        return 32
    return 8
