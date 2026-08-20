"""Real-Blender reprojection and bake, driven through the Apache client.

The chain is fixture -> condition -> view ingest -> bake, so this exercises the pieces the
way a real run composes them rather than in isolation. Every expectation was measured
against Blender 5.2.0 locally.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from asset_mania_contracts import FIXTURE_RENDER_PROFILE, canonical_digest
from asset_mania_pipeline import (
    alignment_acknowledgement_text,
    ingest_view,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_blender_e2e.py"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

sys.path.insert(0, str(ROOT / "blender-addon" / "src"))

pytestmark = pytest.mark.skipif(
    not BLENDER.exists(), reason="the pinned Blender 5.2.0 install is unavailable"
)

SALT = bytes(range(32))
TARGET, CAMERA, RIG, ACTION = "Robot_Strip_Body", "Shot_Camera", "Robot_Rig", "Robot_Flex"
ATLAS = [64, 64]
RESOLUTION = [64, 64]
FRAME = 2
MINIMUM_COVERAGE = 0.25
BLENDER_FINGERPRINT = {
    "profile": "blender-5.2.0-cpu-v1-fixture",
    "version": "5.2.0",
    "build_hash": "fbe6228777e7",
    "executable_sha256": "60ba7a9b6743f7acf101274361fa76409e382ae07cd2007ce07dea30f6b129f2",
}
EXPECTED_ARTIFACTS = [
    "local/scene-baked.blend",
    "textures/albedo-linear.exr",
    "textures/albedo.png",
    "textures/coverage.png",
    "textures/padded-coverage.png",
    "textures/preview.png",
]
#: The fixture's three UV islands occupy 3 * 0.30 * 0.46 of UV space.
UV_ISLAND_AREA = 3 * 0.30 * 0.46


def _drive(request: dict, staging: Path, operation: str, timeout: int = 1500) -> dict:
    request_file = staging.parent / f"{operation}-{staging.name}-request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--request",
            str(request_file),
            "--staging",
            str(staging),
            "--operation",
            operation,
            "--timeout",
            str(timeout),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _selection(digest: str) -> dict:
    from asset_mania_blender.selection import selection_digest

    identity = {
        "source_scene_sha256": digest,
        "camera": CAMERA,
        "target": TARGET,
        "target_type": "MESH",
        "armature": RIG,
        "action": ACTION,
    }
    labels = {
        "camera_label": "camera-1",
        "target_label": "mesh-1",
        "armature_label": "armature-1",
        "action_label": "action-1",
    }
    return {
        **labels,
        "selection_digest": selection_digest(salt=SALT, identity=identity, labels=labels),
    }


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    """One full chain, shared by the assertions that only read it."""
    base = tmp_path_factory.mktemp("bake")
    build = base / "build"
    build.mkdir()
    _drive(
        {
            "request_id": "request-fixture-1",
            "operation": "fixture",
            "fixture_name": "fixture.blend",
        },
        build,
        "validate",
    )
    source = base / "fixture.blend"
    shutil.move(build / "fixture.blend", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    common = {
        "source_path": str(source),
        "source_scene_sha256": digest,
        "target_name": TARGET,
        "camera_name": CAMERA,
        "armature_name": RIG,
        "action_name": ACTION,
        "frame": FRAME,
        "selection_salt": SALT.hex(),
        "portable_selection": _selection(digest),
    }

    condition_staging = base / "condition"
    condition_staging.mkdir()
    condition = _drive(
        {
            "request_id": "request-condition-1",
            "operation": "condition",
            "resolution": RESOLUTION,
            "render_profile": FIXTURE_RENDER_PROFILE,
            "blender": BLENDER_FINGERPRINT,
            **common,
        },
        condition_staging,
        "condition",
    )
    assert condition["status"] == "succeeded", condition["diagnostics"]

    # The conditioning beauty preview is a view produced for exactly this framing, which
    # is what a correctly aligned user input looks like.
    supplied = base / "supplied-view.png"
    shutil.copy(condition_staging / "artifacts/conditioning/beauty.png", supplied)
    view_staging = base / "view"
    view_staging.mkdir()
    bundle = json.loads(
        (condition_staging / "artifacts/conditioning/bundle.json").read_text(encoding="utf-8")
    )
    condition_digest = canonical_digest(condition)
    ingested = ingest_view(
        image_path=supplied,
        staging_root=view_staging,
        resolution=tuple(RESOLUTION),
        condition_manifest_sha256=condition_digest,
        conditioning_bundle_sha256=bundle["bundle_sha256"],
        camera_digest=bundle["digests"]["source_scene"],
        origin="observed",
        subject="synthetic_person",
        alignment_acknowledgement=alignment_acknowledgement_text(
            condition_digest, sha256_file(supplied)
        ),
        issued_at="2026-08-19T09:20:00Z",
    )

    bake_staging = base / "bake"
    bake_staging.mkdir()
    response = _drive(
        {
            "request_id": "request-bake-1",
            "operation": "bake",
            "condition_run_directory": str(condition_staging),
            "view_path": str(ingested["normalized_path"]),
            "atlas_size": ATLAS,
            "bake_margin": 2,
            "color_padding": 1,
            "minimum_coverage": MINIMUM_COVERAGE,
            **common,
        },
        bake_staging,
        "bake",
    )
    return {
        "base": base,
        "source": source,
        "digest": digest,
        "common": common,
        "condition": condition,
        "condition_staging": condition_staging,
        "view": ingested,
        "response": response,
        "staging": bake_staging,
    }


# --- The bake succeeded -----------------------------------------------------------


def test_the_bake_succeeds(baked) -> None:
    assert baked["response"]["status"] == "succeeded", baked["response"]["diagnostics"]
    assert baked["response"]["diagnostics"] == []


def test_the_source_is_byte_identical_afterwards(baked) -> None:
    assert hashlib.sha256(baked["source"].read_bytes()).hexdigest() == baked["digest"]


def test_every_declared_artifact_exists_with_its_recorded_digest(baked) -> None:
    outputs = {item["path"]: item for item in baked["response"]["outputs"]}
    assert sorted(outputs) == EXPECTED_ARTIFACTS
    for path, item in outputs.items():
        on_disk = baked["staging"] / path
        assert on_disk.is_file(), path
        assert sha256_file(on_disk) == item["sha256"], path
        assert on_disk.stat().st_size == item["byte_size"], path


def test_the_response_carries_no_private_name(baked) -> None:
    rendered = json.dumps(baked["response"])
    for private in (TARGET, CAMERA, RIG, ACTION, str(baked["source"])):
        assert private not in rendered, private


# --- Coverage -------------------------------------------------------------------


def test_coverage_matches_the_uv_island_area(baked) -> None:
    """The strongest correctness signal available without a golden image.

    Reprojection should reach essentially every texel the UV layout allocates to the
    target. A coverage far below the island area would mean texels were being rejected;
    far above would mean texels outside the islands were being written.
    """
    ratio = baked["response"]["metrics"]["coverage_ratio"]
    assert ratio == pytest.approx(UV_ISLAND_AREA, abs=0.05)


def test_coverage_clears_the_plan_threshold(baked) -> None:
    assert baked["response"]["metrics"]["coverage_ratio"] >= MINIMUM_COVERAGE


def test_observed_and_padded_coverage_are_reported_separately(baked) -> None:
    metrics = baked["response"]["metrics"]
    assert metrics["observed_texel_count"] > 0
    assert metrics["padded_texel_count"] > 0
    total = ATLAS[0] * ATLAS[1]
    assert metrics["observed_texel_count"] + metrics["padded_texel_count"] <= total


def test_padding_never_promotes_observed_coverage(baked) -> None:
    """coverage.png is authoritative; padded-coverage.png is kept separate."""
    coverage = baked["staging"] / "textures/coverage.png"
    padded = baked["staging"] / "textures/padded-coverage.png"
    assert coverage.is_file() and padded.is_file()
    assert sha256_file(coverage) != sha256_file(padded)

    metrics = baked["response"]["metrics"]
    total = ATLAS[0] * ATLAS[1]
    assert metrics["observed_texel_count"] / total == pytest.approx(
        metrics["coverage_ratio"], abs=1e-6
    )


def test_every_texel_is_finite(baked) -> None:
    metrics = baked["response"]["metrics"]
    assert metrics["finite_texel_count"] == ATLAS[0] * ATLAS[1]


# --- Provenance and upload eligibility ---------------------------------------------


def test_the_baked_scene_is_local_and_outside_the_texture_directory(baked) -> None:
    outputs = {item["path"] for item in baked["response"]["outputs"]}
    assert "local/scene-baked.blend" in outputs
    assert not any(path.startswith("textures/") and path.endswith(".blend") for path in outputs)


def test_the_texture_semantic_digest_is_recorded(baked) -> None:
    digest = baked["response"]["metrics"]["texture_semantic_digest"]
    assert len(digest) == 64
    for item in baked["response"]["outputs"]:
        assert item["validation"]["semantic_digest"] == digest


# --- Determinism -------------------------------------------------------------------


@pytest.fixture(scope="module")
def repeated(baked):
    staging = baked["base"] / "bake-again"
    staging.mkdir()
    response = _drive(
        {
            "request_id": "request-bake-1",
            "operation": "bake",
            "condition_run_directory": str(baked["condition_staging"]),
            "view_path": str(baked["view"]["normalized_path"]),
            "atlas_size": ATLAS,
            "bake_margin": 2,
            "color_padding": 1,
            "minimum_coverage": MINIMUM_COVERAGE,
            **baked["common"],
        },
        staging,
        "bake",
    )
    assert response["status"] == "succeeded", response["diagnostics"]
    return {"staging": staging, "response": response}


def test_the_metrics_are_identical_across_runs(baked, repeated) -> None:
    assert baked["response"]["metrics"] == repeated["response"]["metrics"]


@pytest.mark.parametrize(
    "relative",
    [
        "textures/albedo.png",
        "textures/coverage.png",
        "textures/padded-coverage.png",
        "textures/preview.png",
    ],
)
def test_decoded_eight_bit_textures_are_byte_exact_across_runs(
    baked, repeated, relative: str
) -> None:
    assert sha256_file(baked["staging"] / relative) == sha256_file(repeated["staging"] / relative)


# --- Refusals -----------------------------------------------------------------------


def test_a_tampered_selection_refuses_to_bake(baked) -> None:
    staging = baked["base"] / "bake-tampered"
    staging.mkdir()
    request = {
        "request_id": "request-bake-1",
        "operation": "bake",
        "condition_run_directory": str(baked["condition_staging"]),
        "view_path": str(baked["view"]["normalized_path"]),
        "atlas_size": ATLAS,
        "bake_margin": 2,
        "color_padding": 1,
        "minimum_coverage": MINIMUM_COVERAGE,
        **baked["common"],
    }
    request["portable_selection"] = {
        **request["portable_selection"],
        "target_label": "mesh-9",
    }
    response = _drive(request, staging, "bake")
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["PLAN_TAMPERED"]
    assert not (staging / "textures").exists()


def test_a_coverage_threshold_above_the_result_marks_artifacts_incomplete(baked) -> None:
    """Low coverage keeps its artifacts, marked incomplete, and cannot feed an export."""
    staging = baked["base"] / "bake-strict"
    staging.mkdir()
    response = _drive(
        {
            "request_id": "request-bake-1",
            "operation": "bake",
            "condition_run_directory": str(baked["condition_staging"]),
            "view_path": str(baked["view"]["normalized_path"]),
            "atlas_size": ATLAS,
            "bake_margin": 2,
            "color_padding": 1,
            "minimum_coverage": 0.99,
            **baked["common"],
        },
        staging,
        "bake",
    )
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["REPROJECTION_LOW_COVERAGE"]
    assert response["outputs"]
    assert all(item["validation"]["status"] == "incomplete" for item in response["outputs"])


def test_a_view_that_is_not_the_conditioning_resolution_is_refused(baked, tmp_path) -> None:
    from asset_mania_pipeline import ViewRejected
    from PIL import Image

    wrong = tmp_path / "wrong-size.png"
    Image.new("RGBA", (32, 32), (1, 2, 3, 255)).save(wrong)
    condition_digest = canonical_digest(baked["condition"])
    with pytest.raises(ViewRejected, match="never resizes or crops"):
        ingest_view(
            image_path=wrong,
            staging_root=tmp_path / "staging",
            resolution=tuple(RESOLUTION),
            condition_manifest_sha256=condition_digest,
            conditioning_bundle_sha256="c1" * 32,
            camera_digest="c5" * 32,
            origin="observed",
            subject="synthetic_person",
            alignment_acknowledgement=alignment_acknowledgement_text(
                condition_digest, sha256_file(wrong)
            ),
            issued_at="2026-08-19T09:20:00Z",
        )
