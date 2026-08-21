"""The reconstruction input contract: mandatory mask, declarations, and output limits."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from asset_mania_contracts import canonical_digest, load_schema
from asset_mania_pipeline import (
    ReconstructionRejected,
    SubjectDeclarationRequired,
    ViewRejected,
    acknowledgement_text,
    describe_reconstruction_output,
    issue_receipt,
    plan_reconstruction,
    prepare_input,
    refuse_as_bake_input,
    sha256_file,
)
from jsonschema import Draft202012Validator
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"
ENGINE = "triposr-local"
PROFILE = "triposr-local-cpu-v1"
NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


def _clearance(name: str = "engine-clearance-v1") -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "subject.png"
    Image.new("RGB", (320, 240), (30, 90, 160)).save(path)
    return path


@pytest.fixture
def mask(tmp_path: Path) -> Path:
    path = tmp_path / "subject-mask.png"
    canvas = Image.new("L", (320, 240), 0)
    for x in range(80, 240):
        for y in range(40, 200):
            canvas.putpixel((x, y), 255)
    canvas.save(path)
    return path


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    path = tmp_path / "staging"
    path.mkdir()
    return path


def _plan(image, staging, **overrides):
    arguments = {
        "image_path": image,
        "staging_root": staging,
        "engine": ENGINE,
        "engine_profile": PROFILE,
        "clearance": _clearance(),
        "asset_kind": "object",
        "subject": "non_person",
        "now": NOW,
    }
    arguments.update(overrides)
    return plan_reconstruction(**arguments)


# --- Input decoding ---------------------------------------------------------------


def test_an_arbitrary_size_is_accepted(image, staging) -> None:
    """There is no conditioning resolution to match here."""
    prepared = prepare_input(image_path=image, staging_root=staging)
    assert (prepared["width"], prepared["height"]) == (320, 240)


def test_the_original_image_is_not_rewritten(image, staging) -> None:
    before = sha256_file(image)
    prepare_input(image_path=image, staging_root=staging)
    assert sha256_file(image) == before


def test_the_normalized_copy_is_written_into_the_run(image, staging) -> None:
    prepared = prepare_input(image_path=image, staging_root=staging)
    assert prepared["normalized_image"] == staging / "reconstruction-input.png"
    assert prepared["normalized_image"].is_file()


def test_an_unsupported_container_is_refused(tmp_path, staging) -> None:
    path = tmp_path / "subject.bmp"
    Image.new("RGB", (64, 64)).save(path, format="BMP")
    with pytest.raises(ViewRejected, match="UNSUPPORTED_MEDIA_TYPE"):
        prepare_input(image_path=path, staging_root=staging)


def test_a_rotating_exif_orientation_is_refused(tmp_path, staging) -> None:
    path = tmp_path / "subject.jpg"
    canvas = Image.new("RGB", (64, 64), (1, 2, 3))
    exif = canvas.getexif()
    exif[0x0112] = 6
    canvas.save(path, format="JPEG", exif=exif)
    with pytest.raises(ViewRejected, match="VIEW_ALIGNMENT_MISMATCH"):
        prepare_input(image_path=path, staging_root=staging)


# --- The mask -----------------------------------------------------------------------


def test_a_matching_mask_is_normalized_alongside_the_image(image, staging, mask) -> None:
    prepared = prepare_input(image_path=image, staging_root=staging, mask_path=mask)
    assert prepared["mask_sha256"]
    assert prepared["normalized_mask"] == staging / "reconstruction-mask.png"


def test_a_mask_of_another_size_is_refused_rather_than_resized(image, staging, tmp_path) -> None:
    """Resizing a mask moves the silhouette, which is the one thing it defines."""
    wrong = tmp_path / "wrong-mask.png"
    Image.new("L", (160, 120), 255).save(wrong)
    with pytest.raises(ReconstructionRejected, match="never resizes a mask"):
        prepare_input(image_path=image, staging_root=staging, mask_path=wrong)


def test_planning_without_a_mask_or_a_remover_is_refused(image, staging) -> None:
    with pytest.raises(ValueError, match="MASK_REQUIRED"):
        _plan(image, staging)


def test_planning_with_a_mask_succeeds(image, staging, mask) -> None:
    result = _plan(image, staging, mask_path=mask)
    assert result["plan"]["mask_sha256"]
    assert result["plan"]["background_removal_clearance_sha256"] is None


def test_planning_with_an_audited_remover_succeeds(image, staging) -> None:
    result = _plan(image, staging, background_removal_clearance=_clearance())
    assert result["plan"]["mask_sha256"] is None
    assert result["plan"]["background_removal_clearance_sha256"]


def test_an_unpinned_remover_is_refused(image, staging) -> None:
    mutated = copy.deepcopy(_clearance())
    for item in mutated["components"]:
        if item["role"] == "preprocessing_model":
            item["content_sha256"] = ""
    preimage = {k: v for k, v in mutated.items() if k != "clearance_sha256"}
    mutated = {**preimage, "clearance_sha256": canonical_digest(preimage)}
    with pytest.raises(ValueError, match="BACKGROUND_REMOVAL_UNPINNED"):
        _plan(image, staging, background_removal_clearance=mutated)


# --- Clearance and declarations, in order -------------------------------------------


def test_an_uncleared_engine_is_refused_before_the_input_is_read(image, staging) -> None:
    from asset_mania_pipeline import EngineLicenseUncleared

    with pytest.raises(EngineLicenseUncleared):
        _plan(image, staging, mask_path=None, clearance=_clearance("engine-clearance-v1-uncleared"))
    assert not (staging / "reconstruction-input.png").exists()


def test_no_clearance_at_all_is_refused(image, staging, mask) -> None:
    from asset_mania_pipeline import EngineNotCleared

    with pytest.raises(EngineNotCleared):
        _plan(image, staging, mask_path=mask, clearance=None)


def test_an_unknown_subject_is_refused_first_of_all(image, staging, mask) -> None:
    with pytest.raises(SubjectDeclarationRequired):
        _plan(image, staging, mask_path=mask, subject="unknown", clearance=None)


def test_a_real_person_without_a_receipt_is_refused(image, staging, mask) -> None:
    with pytest.raises(ReconstructionRejected, match="FACE_RIGHTS_CONFIRMATION_REQUIRED"):
        _plan(image, staging, mask_path=mask, subject="real_person", asset_kind="face_head")


def test_a_real_person_with_a_plan_bound_receipt_is_accepted(image, staging, mask) -> None:
    plan_digest = "b7" * 32
    receipt = issue_receipt(
        receipt_id="receipt-face-rights-1",
        plan_sha256=plan_digest,
        gate="face_rights",
        acknowledgement=acknowledgement_text("face_rights", plan_digest),
        disclosure="Local face/head reconstruction of a real person.",
        issued_at="2026-08-25T08:00:00Z",
        expires_at="2026-08-25T10:00:00Z",
    )
    result = _plan(
        image,
        staging,
        mask_path=mask,
        subject="real_person",
        asset_kind="face_head",
        rights_receipt=receipt,
        plan_sha256_for_receipt=plan_digest,
    )
    assert result["plan"]["rights_receipt_sha256"] == receipt["receipt_sha256"]


def test_a_receipt_on_a_non_person_plan_is_refused(image, staging, mask) -> None:
    from asset_mania_pipeline import ApprovalRejected

    plan_digest = "b7" * 32
    receipt = issue_receipt(
        receipt_id="receipt-face-rights-1",
        plan_sha256=plan_digest,
        gate="face_rights",
        acknowledgement=acknowledgement_text("face_rights", plan_digest),
        disclosure="x",
        issued_at="2026-08-25T08:00:00Z",
        expires_at="2026-08-25T10:00:00Z",
    )
    with pytest.raises(ApprovalRejected):
        _plan(image, staging, mask_path=mask, rights_receipt=receipt)


# --- The sealed plan ------------------------------------------------------------------


def test_the_plan_validates_against_the_committed_schema(image, staging, mask) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    validator = Draft202012Validator(load_schema("reconstruction-plan", "1.0"))
    assert list(validator.iter_errors(plan)) == []


def test_the_plan_is_bound_to_the_verified_clearance_digest(image, staging, mask) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    assert plan["clearance_sha256"] == _clearance()["clearance_sha256"]


def test_the_plan_carries_no_path(image, staging, mask) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    rendered = json.dumps(plan)
    assert str(image) not in rendered
    assert image.name not in rendered


# --- The output -------------------------------------------------------------------------


@pytest.fixture
def mesh(tmp_path: Path) -> Path:
    path = tmp_path / "mesh.glb"
    path.write_bytes(b"glTF" + bytes(64))
    return path


def _describe(mesh, plan, **overrides):
    arguments = {
        "mesh_path": mesh,
        "plan": plan,
        "triangle_count": 4096,
        "vertex_count": 2048,
        "manifold": "closed",
    }
    arguments.update(overrides)
    return describe_reconstruction_output(**arguments)


def test_a_produced_mesh_is_recorded_as_generated(image, staging, mask, mesh) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    record = _describe(mesh, plan)
    assert record["content_origin"] == "generated"
    assert record["sensitivity"] == "user-content"
    assert record["upload_eligible"] is False
    assert record["parents"][0]["relationship"] == "generated_from"


def test_a_missing_mesh_is_refused(image, staging, mask, tmp_path) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    with pytest.raises(ReconstructionRejected, match="RECONSTRUCTION_FAILED"):
        _describe(tmp_path / "absent.glb", plan)


def test_an_empty_mesh_is_refused(image, staging, mask, tmp_path) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    empty = tmp_path / "empty.glb"
    empty.write_bytes(b"")
    with pytest.raises(ReconstructionRejected, match="RECONSTRUCTION_FAILED"):
        _describe(empty, plan)


@pytest.mark.parametrize(("triangles", "vertices"), [(0, 10), (10, 0), (-1, 10)])
def test_a_mesh_with_no_geometry_is_refused(
    image, staging, mask, mesh, triangles: int, vertices: int
) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    with pytest.raises(ReconstructionRejected, match="RECONSTRUCTION_UNVERIFIED"):
        _describe(mesh, plan, triangle_count=triangles, vertex_count=vertices)


def test_an_oversized_mesh_is_refused(image, staging, mask, mesh, monkeypatch) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    monkeypatch.setattr("asset_mania_pipeline.reconstruction.MESH_MAX_BYTES", 8)
    with pytest.raises(ReconstructionRejected, match="exceeds"):
        _describe(mesh, plan)


@pytest.mark.parametrize("manifold", ["closed", "open", "unknown"])
def test_the_manifold_state_is_recorded_not_assumed(
    image, staging, mask, mesh, manifold: str
) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    assert _describe(mesh, plan, manifold=manifold)["manifold"] == manifold


def test_an_undeclared_manifold_state_is_refused(image, staging, mask, mesh) -> None:
    plan = _plan(image, staging, mask_path=mask)["plan"]
    with pytest.raises(ReconstructionRejected, match="manifold"):
        _describe(mesh, plan, manifold="watertight-ish")


# --- No bake contamination ----------------------------------------------------------------


def test_a_reconstruction_plan_is_refused_as_a_bake_input(image, staging, mask) -> None:
    """Bake needs authored UVs and an aligned view; a reconstruction has neither."""
    plan = _plan(image, staging, mask_path=mask)["plan"]
    with pytest.raises(ReconstructionRejected, match="cannot"):
        refuse_as_bake_input(plan)


def test_a_reconstruct_stage_manifest_is_refused_as_a_bake_input() -> None:
    with pytest.raises(ReconstructionRejected, match="cannot"):
        refuse_as_bake_input({"stage": "reconstruct"})


def test_a_conditioning_manifest_is_still_a_valid_bake_input() -> None:
    condition = json.loads((EXAMPLES / "manifest-v2-condition.json").read_text(encoding="utf-8"))
    refuse_as_bake_input(condition)
