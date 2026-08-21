"""The reconstruction input contract.

Decoding and normalization reuse the view rules: 8-bit sRGB, RGB or straight-alpha RGBA, no
implicit resize, rotation, or colour conversion, EXIF orientation absent or 1, metadata
stripped from the normalized copy, and hidden RGB zeroed.

Two things differ from a v0.2 view, and both matter:

* there is no conditioning resolution to match, so an arbitrary size is accepted within the
  decompression limits; and
* a foreground mask is mandatory, because a single-image reconstructor handed a full scene
  reconstructs the scene. An absent mask is a different job, not a permissive default.
"""

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from asset_mania_contracts import (
    DiagnosticCode,
    build_reconstruction_plan,
)
from PIL import Image

from .approvals import require_rights_receipt, require_subject_declaration
from .clearance import require_mask_or_audited_remover, verify_engine_clearance
from .hashing import sha256_bytes, sha256_file
from .views import (
    decode_still,
    normalize_pixels,
    open_bounded,
    reject_metadata_and_colour,
    write_normalized_png,
)


def decode_mask(path: Path) -> "Image.Image":
    """Decode one foreground mask.

    A mask is legitimately single-channel, so it gets its own shape rule while sharing the
    orientation, metadata, and colour checks with every other decoded input.
    """
    image = open_bounded(path)
    try:
        if image.mode not in _MASK_MODES:
            raise ReconstructionRejected(
                DiagnosticCode.UNSUPPORTED_MEDIA_TYPE.value,
                f"a mask in mode {image.mode!r} is outside the profile",
            )
        reject_metadata_and_colour(image)
    except BaseException:
        image.close()
        raise
    return image


NORMALIZED_IMAGE = "reconstruction-input.png"
NORMALIZED_MASK = "reconstruction-mask.png"
MESH_MAX_BYTES = 256 * 1024 * 1024
_MASK_MODES = frozenset({"L", "LA", "RGB", "RGBA"})


class ReconstructionRejected(Exception):
    """The reconstruction input or its declarations are outside the profile."""

    def __init__(self, diagnostic: str, detail: str) -> None:
        super().__init__(f"{diagnostic}: {detail}")
        self.diagnostic = diagnostic


def prepare_input(
    *,
    image_path: Path,
    staging_root: Path,
    mask_path: Path | None = None,
) -> dict[str, Any]:
    """Decode, normalize, and describe the reconstruction input.

    A supplied mask must already match the image exactly. Resizing it would move the
    silhouette, which is the one thing the mask exists to define.
    """
    image = decode_still(image_path)
    try:
        width, height = image.size
        pixels, alpha = normalize_pixels(image)
    finally:
        image.close()

    normalized = write_normalized_png(
        pixels=pixels, width=width, height=height, destination=staging_root / NORMALIZED_IMAGE
    )

    mask_digest: str | None = None
    normalized_mask: Path | None = None
    if mask_path is not None:
        mask = decode_mask(mask_path)
        try:
            if mask.size != (width, height):
                raise ReconstructionRejected(
                    DiagnosticCode.MASK_REQUIRED.value,
                    f"the mask is {mask.size[0]}x{mask.size[1]} and the image is "
                    f"{width}x{height}; this profile never resizes a mask",
                )
            mask_pixels, _ = normalize_pixels(mask.convert("RGBA"))
        finally:
            mask.close()
        normalized_mask = write_normalized_png(
            pixels=mask_pixels,
            width=width,
            height=height,
            destination=staging_root / NORMALIZED_MASK,
        )
        mask_digest = sha256_file(normalized_mask)

    return {
        "image_sha256": sha256_file(image_path),
        "width": width,
        "height": height,
        "alpha": alpha,
        "normalized_image": normalized,
        "normalized_image_sha256": sha256_file(normalized),
        "normalized_mask": normalized_mask,
        "mask_sha256": mask_digest,
        "decoded_sha256": sha256_bytes(pixels),
    }


def plan_reconstruction(
    *,
    image_path: Path,
    staging_root: Path,
    engine: str,
    engine_profile: str,
    clearance: Mapping[str, Any] | None,
    asset_kind: str,
    subject: str,
    now: datetime,
    mask_path: Path | None = None,
    background_removal_clearance: Mapping[str, Any] | None = None,
    rights_receipt: Mapping[str, Any] | None = None,
    plan_sha256_for_receipt: str | None = None,
    expected_output: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a sealed reconstruction plan, or refuse before any engine is considered.

    Order matters and is asserted by tests: the subject declaration is checked first, then
    the engine clearance, then the mask requirement, then the rights receipt. An engine is
    never named to a runner from here -- planning and running are separate steps precisely so
    a plan can be reviewed before anything executes.
    """
    require_subject_declaration(subject)
    clearance_digest = verify_engine_clearance(clearance, engine=engine, now=now)

    prepared = prepare_input(image_path=image_path, staging_root=staging_root, mask_path=mask_path)
    mask_digest, remover_digest = require_mask_or_audited_remover(
        mask_sha256=prepared["mask_sha256"],
        background_removal_clearance=background_removal_clearance,
        engine=engine,
        now=now,
    )

    receipt_digest: str | None = None
    if subject == "real_person":
        if plan_sha256_for_receipt is None:
            raise ReconstructionRejected(
                DiagnosticCode.FACE_RIGHTS_CONFIRMATION_REQUIRED.value,
                "a real_person plan needs the plan digest its receipt is bound to",
            )
        validated = require_rights_receipt(
            subject=subject,
            receipt=rights_receipt,
            plan_sha256=plan_sha256_for_receipt,
            now=now,
        )
        receipt_digest = validated["receipt_sha256"] if validated else None
    elif rights_receipt is not None:
        require_rights_receipt(subject=subject, receipt=rights_receipt, plan_sha256="", now=now)

    plan = build_reconstruction_plan(
        engine=engine,
        engine_profile=engine_profile,
        clearance_sha256=clearance_digest,
        source_image_sha256=prepared["image_sha256"],
        source_width=prepared["width"],
        source_height=prepared["height"],
        alpha=prepared["alpha"],
        mask_sha256=mask_digest,
        background_removal_clearance_sha256=remover_digest,
        asset_kind=asset_kind,
        subject=subject,
        rights_receipt_sha256=receipt_digest,
        expected_output=expected_output
        or {"mesh_format": "glb", "textured": False, "unit_scale_meters": 1.0},
    )
    return {"plan": plan, "input": prepared}


def describe_reconstruction_output(
    *,
    mesh_path: Path,
    plan: Mapping[str, Any],
    triangle_count: int,
    vertex_count: int,
    manifold: str,
) -> dict[str, Any]:
    """Describe a produced mesh on its own terms.

    A reconstruction has no camera correspondence and no authored UVs, so it is validated
    structurally and nothing more. `manifold` is recorded rather than assumed, because
    claiming a watertight mesh that is not one breaks every downstream consumer.
    """
    if not mesh_path.is_file():
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_FAILED.value, "the engine produced no mesh"
        )
    size = mesh_path.stat().st_size
    if size == 0:
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_FAILED.value, "the produced mesh is empty"
        )
    if size > MESH_MAX_BYTES:
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_UNVERIFIED.value,
            f"the produced mesh exceeds {MESH_MAX_BYTES} bytes",
        )
    if triangle_count <= 0 or vertex_count <= 0:
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_UNVERIFIED.value,
            "a mesh with no triangles or no vertices is not a reconstruction",
        )
    if manifold not in ("closed", "open", "unknown"):
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_UNVERIFIED.value,
            f"manifold state {manifold!r} is not a declared state",
        )

    return {
        "role": "reconstructed_mesh",
        "path": mesh_path.name,
        "sha256": sha256_file(mesh_path),
        "byte_size": size,
        "media_type": {
            "glb": "model/gltf-binary",
            "obj": "text/plain",
            "ply": "application/octet-stream",
        }[plan["expected_output"]["mesh_format"]],
        "parents": [{"sha256": plan["source_image_sha256"], "relationship": "generated_from"}],
        "operation": "reconstruct",
        # Generated geometry stays generated. It is never presented as observed.
        "content_origin": "generated",
        "sensitivity": "user-content",
        "upload_eligible": False,
        "validation": {
            "profile": "reconstruction-v1",
            "status": "valid",
            "diagnostics": [],
            "semantic_digest": None,
        },
        "triangle_count": int(triangle_count),
        "vertex_count": int(vertex_count),
        "manifold": manifold,
    }


def refuse_as_bake_input(manifest: Mapping[str, Any]) -> None:
    """A reconstruction manifest may never feed the v0.2 bake path.

    Bake needs authored non-overlapping UVs and a view aligned to a known camera. A
    reconstruction has neither, so the result would look plausible and be wrong -- which is
    worse than failing.
    """
    stage = manifest.get("stage")
    if stage == "reconstruct" or manifest.get("schema_id") == "asset-mania/reconstruction-plan":
        raise ReconstructionRejected(
            DiagnosticCode.RECONSTRUCTION_UNVERIFIED.value,
            "a reconstruction has no camera correspondence or authored UVs, so it cannot "
            "be a bake input",
        )
