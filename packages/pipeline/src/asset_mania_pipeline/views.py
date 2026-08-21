"""View ingest: fully decode a user-supplied aligned image and normalize it.

This profile accepts exactly one image shape -- 8-bit sRGB, RGB or straight-alpha RGBA, at
the conditioning resolution -- and rejects everything else instead of coercing it. There is
no implicit resize, crop, rotation, colour conversion, or orientation fix, because any of
those would silently break the camera alignment the whole pipeline depends on.

Alignment cannot be proven from pixels: an arbitrary same-sized image is indistinguishable
from a correctly framed one. So a normal input is recorded as declared and explicitly
unverified, and the attestation binds the declaration to the exact conditioning digest.
"""

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asset_mania_contracts import DECLARABLE_SUBJECTS, canonical_digest
from PIL import Image, ImageFile
from PIL import features as pil_features

from .approvals import require_subject_declaration
from .hashing import sha256_bytes, sha256_file

#: A view is a single frame at the conditioning resolution; the cap is far above any
#: legitimate input and exists to stop a decompression bomb before decoding.
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_INPUT_PIXELS = 4096 * 4096
ACCEPTED_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
ACCEPTED_MODES = frozenset({"RGB", "RGBA"})
NORMALIZED_NAME = "view.png"
VALIDATION_PROFILE = "view-v1"

_EXIF_ORIENTATION_TAG = 0x0112
_STRIPPED_METADATA_KEYS = (
    "exif",
    "icc_profile",
    "XML:com.adobe.xmp",
    "xmp",
    "iptc",
    "photoshop",
    "comment",
    "Description",
    "dpi",
)
_SENSITIVE_METADATA = "EXIF_SENSITIVE_METADATA_PRESENT"
_UNSUPPORTED = "UNSUPPORTED_MEDIA_TYPE"
_ALIGNMENT_MISMATCH = "VIEW_ALIGNMENT_MISMATCH"
_UNREADABLE = "INPUT_UNREADABLE"


class ViewRejected(Exception):
    """The supplied image is not the one shape this profile can consume."""

    def __init__(self, diagnostic: str, detail: str) -> None:
        super().__init__(f"{diagnostic}: {detail}")
        self.diagnostic = diagnostic


def alignment_acknowledgement_text(condition_sha256: str, view_sha256: str) -> str:
    """The exact string a user must type to attest that a view matches a framing."""
    return f"{condition_sha256}:{view_sha256}"


def build_alignment_attestation(
    *,
    condition_manifest_sha256: str,
    view_image_sha256: str,
    issuer: str,
    issued_at: str,
) -> dict[str, Any]:
    """The canonical declaration object whose digest the view manifest records."""
    if issuer not in ("user", "provider"):
        raise ViewRejected(_ALIGNMENT_MISMATCH, f"{issuer!r} cannot issue an alignment claim")
    attestation = {
        "schema_id": "asset-mania/alignment-attestation",
        "schema_version": "1.0",
        "condition_manifest_sha256": condition_manifest_sha256,
        "view_image_sha256": view_image_sha256,
        "issuer": issuer,
        "statement": "declared_aligned",
        "issued_at": issued_at,
    }
    return {**attestation, "attestation_sha256": canonical_digest(attestation)}


def _require(condition: bool, diagnostic: str, detail: str) -> None:
    if not condition:
        raise ViewRejected(diagnostic, detail)


def _open(path: Path) -> Image.Image:
    _require(path.is_file(), _UNREADABLE, "the view file does not exist")
    size = path.stat().st_size
    _require(size > 0, _UNREADABLE, "the view file is empty")
    _require(size <= MAX_INPUT_BYTES, _UNSUPPORTED, f"the view exceeds {MAX_INPUT_BYTES} bytes")

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_INPUT_PIXELS
    # A truncated file must fail rather than decode to partial pixels.
    previous_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        image = Image.open(path)
        image.load()
    except Image.DecompressionBombError as error:
        raise ViewRejected(_UNSUPPORTED, "the view is a decompression bomb") from error
    except (OSError, ValueError, SyntaxError) as error:
        raise ViewRejected(_UNREADABLE, "the view could not be decoded") from error
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated
    return image


def _reject_unsupported_shape(image: Image.Image) -> None:
    _require(
        image.format in ACCEPTED_FORMATS,
        _UNSUPPORTED,
        f"{image.format!r} is not one of {sorted(ACCEPTED_FORMATS)}",
    )
    _require(
        image.mode in ACCEPTED_MODES,
        _UNSUPPORTED,
        f"mode {image.mode!r} is not one of {sorted(ACCEPTED_MODES)}",
    )
    # Pillow reports 16-bit and float single-channel data as I;16/I/F, and palette and
    # grayscale as P/L/LA. All of them fail the mode check above; naming them here keeps
    # the intent explicit for a reader.
    _require(
        image.mode not in ("L", "LA", "P", "PA", "CMYK", "YCbCr", "I", "I;16", "F", "LAB", "HSV"),
        _UNSUPPORTED,
        f"mode {image.mode!r} is outside the 8-bit sRGB profile",
    )
    bands = image.getbands()
    _require(
        all(len(band) == 1 for band in bands),
        _UNSUPPORTED,
        "the view does not use 8-bit channels",
    )


def _reject_orientation_and_metadata(image: Image.Image) -> None:
    exif = image.getexif()
    orientation = exif.get(_EXIF_ORIENTATION_TAG) if exif else None
    _require(
        orientation in (None, 1),
        _ALIGNMENT_MISMATCH,
        f"EXIF orientation {orientation!r} would rotate an aligned view",
    )

    # Judge the parsed tags, not the presence of a container: a JPEG that carries only
    # `orientation = 1` has no sensitive metadata, and rejecting it would refuse a
    # perfectly ordinary export.
    informative_tags = sorted(set(exif) - {_EXIF_ORIENTATION_TAG}) if exif else []
    sensitive = [key for key in ("XML:com.adobe.xmp", "iptc") if image.info.get(key)]
    if informative_tags:
        sensitive.append(f"exif tags {informative_tags}")
    if sensitive:
        # Metadata is stripped from the normalized copy, but GPS or camera identity in the
        # input is worth reporting rather than discarding silently.
        raise ViewRejected(
            _SENSITIVE_METADATA, f"the view carries metadata this profile removes: {sensitive}"
        )


def _reject_unknown_colour_management(image: Image.Image) -> None:
    profile = image.info.get("icc_profile")
    if not profile:
        return
    _require(
        _is_srgb_profile(profile),
        _UNSUPPORTED,
        "only a recognized sRGB ICC profile may be removed after decoding",
    )


def _is_srgb_profile(profile: bytes) -> bool:
    """Recognize an sRGB profile by its embedded description."""
    if not pil_features.check("littlecms2"):
        # Without a colour engine an ICC profile cannot be recognized, so it is not
        # treated as recognized.
        return False
    from PIL import ImageCms

    try:
        parsed = ImageCms.getOpenProfile(io.BytesIO(profile))
        description = (ImageCms.getProfileDescription(parsed) or "").strip().lower()
    except Exception:  # noqa: BLE001 - an unparseable profile is simply not recognized
        return False
    return "srgb" in description.replace(" ", "").replace("-", "")


def _reject_premultiplied_alpha(image: Image.Image) -> None:
    for key in ("premultiplied", "premultiplied_alpha"):
        _require(
            not image.info.get(key),
            _UNSUPPORTED,
            "premultiplied alpha is outside this straight-alpha profile",
        )


def normalize_pixels(image: Image.Image) -> tuple[bytes, str]:
    """Return straight-alpha RGBA bytes with hidden RGB zeroed, and the alpha convention.

    Zeroing the RGB under fully transparent pixels means a normalized view cannot smuggle
    content in channels a viewer never shows.
    """
    alpha = "straight" if image.mode == "RGBA" else "none"
    rgba = image.convert("RGBA") if image.mode != "RGBA" else image.copy()
    data = bytearray(rgba.tobytes())
    for offset in range(0, len(data), 4):
        if data[offset + 3] == 0:
            data[offset] = data[offset + 1] = data[offset + 2] = 0
    return bytes(data), alpha


def write_normalized_png(*, pixels: bytes, width: int, height: int, destination: Path) -> Path:
    """Write the normalized copy with no metadata; the original is never rewritten."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # `frombytes` produces an image with no `info`, so there is nothing to carry over.
    normalized = Image.frombytes("RGBA", (width, height), pixels)
    normalized.save(destination, format="PNG", optimize=False)

    # Prove nothing came back: re-open and refuse to publish a copy carrying metadata.
    with Image.open(destination) as check:
        check.load()
        leaked = [key for key in _STRIPPED_METADATA_KEYS if check.info.get(key)]
        if leaked:
            destination.unlink(missing_ok=True)
            raise ViewRejected(_SENSITIVE_METADATA, f"the normalized copy still carries {leaked}")
    return destination


def open_bounded(path: Path) -> Image.Image:
    """Open and fully decode one image within the size and bomb limits."""
    return _open(path)


def reject_metadata_and_colour(image: Image.Image) -> None:
    """The checks that apply to any decoded input, whatever its channel layout.

    Orientation, metadata, colour management, and alpha convention are separate from the
    channel-shape rule, because a mask is legitimately single-channel while a view is not.
    """
    _reject_orientation_and_metadata(image)
    _reject_unknown_colour_management(image)
    _reject_premultiplied_alpha(image)


def decode_still(path: Path) -> Image.Image:
    """Decode one colour still: the view and reconstruction-image shape.

    Shared by view ingest and reconstruction input, so the two cannot drift apart.
    """
    image = open_bounded(path)
    try:
        _reject_unsupported_shape(image)
        reject_metadata_and_colour(image)
    except BaseException:
        image.close()
        raise
    return image


def ingest_view(
    *,
    image_path: Path,
    staging_root: Path,
    resolution: tuple[int, int],
    condition_manifest_sha256: str,
    conditioning_bundle_sha256: str,
    camera_digest: str,
    origin: str,
    subject: str,
    alignment_acknowledgement: object,
    alignment_issuer: str = "user",
    issued_at: str,
    rights_basis_manifest_sha256: str | None = None,
    sensitivity: str = "user-content",
    alignment_status: str = "declared_unverified",
) -> dict[str, Any]:
    """Decode, check, normalize, and describe one supplied view.

    The subject is a user declaration carried from the immutable workflow plan; `unknown`
    is blocked here as everywhere else, and nothing about the subject is inferred from
    pixels.
    """
    require_subject_declaration(subject)
    _require(
        origin in ("observed", "generated", "unknown"),
        _UNSUPPORTED,
        f"origin {origin!r} is not a declared origin",
    )
    _require(
        subject in DECLARABLE_SUBJECTS,
        _UNSUPPORTED,
        f"subject {subject!r} is not executable",
    )

    image = decode_still(image_path)
    try:
        width, height = image.size
        expected_width, expected_height = resolution
        _require(
            (width, height) == (expected_width, expected_height),
            _ALIGNMENT_MISMATCH,
            f"the view is {width}x{height}, the conditioning is "
            f"{expected_width}x{expected_height}; this profile never resizes or crops",
        )
        pixels, alpha = normalize_pixels(image)
    finally:
        image.close()

    image_sha256 = sha256_file(image_path)
    expected_acknowledgement = alignment_acknowledgement_text(
        condition_manifest_sha256, image_sha256
    )
    _require(
        isinstance(alignment_acknowledgement, str)
        and alignment_acknowledgement == expected_acknowledgement,
        _ALIGNMENT_MISMATCH,
        "alignment must be attested with the exact CONDITION_SHA256:VIEW_SHA256 string",
    )

    attestation = build_alignment_attestation(
        condition_manifest_sha256=condition_manifest_sha256,
        view_image_sha256=image_sha256,
        issuer=alignment_issuer,
        issued_at=issued_at,
    )
    normalized = write_normalized_png(
        pixels=pixels, width=width, height=height, destination=staging_root / NORMALIZED_NAME
    )

    view = {
        "schema_id": "asset-mania/view",
        "schema_version": "1.0",
        "image_sha256": image_sha256,
        "width": width,
        "height": height,
        "media_type": "image/png",
        "color_space": "srgb",
        "alpha": alpha,
        "condition_manifest_sha256": condition_manifest_sha256,
        "conditioning_bundle_sha256": conditioning_bundle_sha256,
        "camera_digest": camera_digest,
        "origin": origin,
        "subject": subject,
        "alignment": {
            "transform": "identity",
            "attestation_sha256": attestation["attestation_sha256"],
            "issuer": alignment_issuer,
            "status": alignment_status,
        },
        "rights_basis_manifest_sha256": rights_basis_manifest_sha256,
        "sensitivity": sensitivity,
        "upload_eligible": False,
        "validation": {
            "profile": VALIDATION_PROFILE,
            "status": "valid",
            "diagnostics": [],
            "semantic_digest": sha256_bytes(pixels),
        },
        "view_sha256": "",
    }
    preimage = {key: value for key, value in view.items() if key != "view_sha256"}
    return {
        "view": {**preimage, "view_sha256": canonical_digest(preimage)},
        "attestation": attestation,
        "normalized_path": normalized,
        "normalized_sha256": sha256_file(normalized),
        "decoded_sha256": sha256_bytes(pixels),
    }


def view_semantic_digest(pixels: bytes) -> str:
    return hashlib.sha256(pixels).hexdigest()


def inherit_rights_basis_digest(condition_manifest: Mapping[str, Any]) -> str | None:
    """The condition run becomes the downstream local rights basis.

    Downstream stages inherit that immutable approval lineage instead of re-consuming a
    single-use receipt.
    """
    if not condition_manifest.get("approvals"):
        return None
    return canonical_digest(condition_manifest)
