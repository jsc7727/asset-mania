"""Request normalization for the pinned GPT Image profile.

Every field is derived from the approved provider plan, never from a caller's convenience.
The four conditioning attachments bind to `image[]` at indices 0..3, in that order, and the
optional API `mask` part is deliberately absent: the binary mask is a visual reference
image here, not an inpainting mask. Any change to the field name, index, role, media type,
byte size, or digest invalidates the approval.

The prompt is carried as a digest. Its text is read at call time, matched against the
approved digest, and never written to a manifest, a log, or a portable artifact.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from asset_mania_pipeline import sha256_bytes, sha256_file

from .errors import PlanMismatch, RequestRejected
from .transport import OFFICIAL_ENDPOINT, MultipartPart, ProviderRequest

MODEL_SNAPSHOT = "gpt-image-2-2026-04-21"
ATTACHMENT_FIELD = "image[]"
ATTACHMENT_ROLES = ("beauty", "depth_preview", "normal_preview", "mask")
CLOSED_CONTROLS = (
    "n",
    "size",
    "quality",
    "background",
    "output_format",
    "output_compression",
    "moderation",
)
#: Rejected before approval: the provider applies high input fidelity automatically and
#: exposes no control for it, so accepting the field would imply an option that is not real.
FORBIDDEN_CONTROLS = ("input_fidelity", "mask", "style", "response_format")
ALLOWED_OUTPUT_FORMATS = ("png", "jpeg", "webp")
ALLOWED_BACKGROUNDS = ("auto", "opaque")
ALLOWED_QUALITIES = ("low", "medium", "high")
ALLOWED_SIZES = ("1024x1024", "1024x1536", "1536x1024")
ALLOWED_MODERATION = ("auto", "low")


def _reject(detail: str) -> None:
    raise RequestRejected(detail)


def validate_controls(controls: Mapping[str, Any]) -> None:
    """Close the control surface before anything is approved or sent."""
    unknown = sorted(set(controls) - set(CLOSED_CONTROLS))
    if unknown:
        _reject(f"unknown provider controls: {unknown}")
    forbidden = sorted(set(controls) & set(FORBIDDEN_CONTROLS))
    if forbidden:
        _reject(f"controls this profile refuses: {forbidden}")
    missing = sorted(set(CLOSED_CONTROLS) - set(controls))
    if missing:
        _reject(f"missing provider controls: {missing}")

    if controls["n"] != 1:
        _reject("this single-view profile fixes n to 1")
    if controls["size"] == "auto" or controls["size"] not in ALLOWED_SIZES:
        _reject(
            f"size {controls['size']!r} has no approval-bound published cost row; "
            f"allowed sizes are {list(ALLOWED_SIZES)}"
        )
    if controls["quality"] == "auto" or controls["quality"] not in ALLOWED_QUALITIES:
        _reject(f"quality {controls['quality']!r} has no published cost row")
    if controls["background"] == "transparent":
        _reject("a transparent background is rejected for this model")
    if controls["background"] not in ALLOWED_BACKGROUNDS:
        _reject(f"background {controls['background']!r} is outside the profile")
    if controls["output_format"] not in ALLOWED_OUTPUT_FORMATS:
        _reject(f"output_format {controls['output_format']!r} is outside the profile")
    if controls["moderation"] not in ALLOWED_MODERATION:
        _reject(f"moderation {controls['moderation']!r} is outside the profile")

    compression = controls["output_compression"]
    if controls["output_format"] == "png":
        if compression is not None:
            _reject("output_compression is not allowed with PNG")
    else:
        if not isinstance(compression, int) or isinstance(compression, bool):
            _reject("output_compression must be an integer for JPEG or WebP")
        elif not 0 <= compression <= 100:
            _reject("output_compression must fall within 0..100")


def validate_size_matches_conditioning(controls: Mapping[str, Any], resolution) -> None:
    """The requested size must equal the conditioning resolution exactly."""
    width, height = (int(value) for value in resolution)
    if controls["size"] != f"{width}x{height}":
        _reject(
            f"size {controls['size']!r} does not equal the conditioning resolution {width}x{height}"
        )


def read_prompt(path: Path, *, expected_sha256: str) -> str:
    """Read the private prompt and require it to be the approved one.

    The text is returned for transport only. Nothing persists it: a caller that writes it
    anywhere has stepped outside this adapter.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise PlanMismatch("the private prompt file could not be read") from error
    if sha256_bytes(text.encode("utf-8")) != expected_sha256:
        raise PlanMismatch("the prompt does not match the approved plan digest")
    return text


def load_attachments(plan: Mapping[str, Any], *, run_directory: Path) -> list[MultipartPart]:
    """Load the four approved attachments, verifying each digest and size."""
    declared = list(plan["attachments"])
    if [item["role"] for item in declared] != list(ATTACHMENT_ROLES):
        _reject(f"attachments must be exactly {list(ATTACHMENT_ROLES)} in that order")
    if [item["index"] for item in declared] != [0, 1, 2, 3]:
        _reject("attachments must bind to image[] indices 0..3")

    paths = plan["attachment_paths"]
    parts: list[MultipartPart] = []
    for item in declared:
        if item["multipart_field"] != ATTACHMENT_FIELD:
            _reject(f"attachment {item['role']!r} must use the {ATTACHMENT_FIELD} field")
        relative = paths.get(item["role"])
        if relative is None:
            _reject(f"attachment {item['role']!r} has no path")
        path = run_directory / str(relative)
        if not path.is_file():
            _reject(f"attachment {item['role']!r} is missing on disk")
        if sha256_file(path) != item["sha256"]:
            raise PlanMismatch(f"attachment {item['role']!r} is not the approved bytes")
        content = path.read_bytes()
        if len(content) != item["byte_size"]:
            raise PlanMismatch(f"attachment {item['role']!r} is not the approved size")
        parts.append(
            MultipartPart(
                field_name=ATTACHMENT_FIELD,
                # A portable label, never the user's file name.
                filename=f"{item['role']}.png",
                media_type=item["media_type"],
                content=content,
            )
        )
    return parts


def build_request(
    *,
    plan: Mapping[str, Any],
    prompt: str,
    parts: Sequence[MultipartPart],
    timeout_seconds: int,
) -> ProviderRequest:
    """Assemble the one request shape this profile permits."""
    if plan["endpoint"] != OFFICIAL_ENDPOINT:
        _reject(f"this profile binds {OFFICIAL_ENDPOINT}, not {plan['endpoint']!r}")
    if plan["model"] != MODEL_SNAPSHOT:
        _reject(f"this profile binds the {MODEL_SNAPSHOT} snapshot")

    controls = plan["controls"]
    validate_controls(controls)

    fields = {
        "model": MODEL_SNAPSHOT,
        "prompt": prompt,
        "n": str(controls["n"]),
        "size": controls["size"],
        "quality": controls["quality"],
        "background": controls["background"],
        "output_format": controls["output_format"],
        "moderation": controls["moderation"],
    }
    if controls["output_compression"] is not None:
        fields["output_compression"] = str(controls["output_compression"])

    return ProviderRequest(
        method="POST",
        endpoint=OFFICIAL_ENDPOINT,
        fields=fields,
        parts=list(parts),
        timeout_seconds=timeout_seconds,
    )
