"""Real-Blender conditioning, driven through the Apache client.

Every expectation was measured against Blender 5.2.0 locally. The module skips when the
pinned Blender is absent, so the fast suite stays offline and Blender-free.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from asset_mania_contracts import FIXTURE_RENDER_PROFILE, canonical_digest, load_schema
from asset_mania_pipeline import BundleInvalid, validate_conditioning_bundle
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_blender_e2e.py"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

sys.path.insert(0, str(ROOT / "blender-addon" / "src"))

pytestmark = pytest.mark.skipif(
    not BLENDER.exists(), reason="the pinned Blender 5.2.0 install is unavailable"
)

SALT = bytes(range(32))
TARGET, CAMERA, RIG, ACTION = "Robot_Strip_Body", "Shot_Camera", "Robot_Rig", "Robot_Flex"
FRAME = 2
RESOLUTION = [64, 64]
BLENDER_FINGERPRINT = {
    "profile": "blender-5.2.0-cpu-v1-fixture",
    "version": "5.2.0",
    "build_hash": "fbe6228777e7",
    "executable_sha256": "60ba7a9b6743f7acf101274361fa76409e382ae07cd2007ce07dea30f6b129f2",
}
EXPECTED_PASS_ROLES = [
    "beauty_exr",
    "beauty_preview",
    "depth_exr",
    "depth_preview",
    "normal_exr",
    "normal_preview",
    "object_index_exr",
    "mask_png",
]
#: The published limit; the measured value is orders of magnitude smaller.
PROJECTION_LIMIT_PIXELS = 0.25


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(request: dict, staging: Path, operation: str) -> dict:
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
            "1200",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _condition_request(source: Path, digest: str) -> dict:
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
        "request_id": "request-condition-1",
        "operation": "condition",
        "source_path": str(source),
        "source_scene_sha256": digest,
        "target_name": TARGET,
        "camera_name": CAMERA,
        "armature_name": RIG,
        "action_name": ACTION,
        "frame": FRAME,
        "resolution": RESOLUTION,
        "render_profile": FIXTURE_RENDER_PROFILE,
        "blender": BLENDER_FINGERPRINT,
        "selection_salt": SALT.hex(),
        "portable_selection": {
            **labels,
            "selection_digest": selection_digest(salt=SALT, identity=identity, labels=labels),
        },
    }


@pytest.fixture(scope="module")
def conditioned(tmp_path_factory):
    """One conditioning run, shared by the assertions that only read it."""
    base = tmp_path_factory.mktemp("conditioning")
    build_staging = base / "build"
    build_staging.mkdir()
    build = _drive(
        {
            "request_id": "request-fixture-1",
            "operation": "fixture",
            "fixture_name": "fixture.blend",
        },
        build_staging,
        "validate",
    )
    assert build["status"] == "succeeded", build["diagnostics"]

    source = base / "fixture.blend"
    shutil.move(build_staging / "fixture.blend", source)
    digest = _sha256(source)

    run_staging = base / "run"
    run_staging.mkdir()
    response = _drive(_condition_request(source, digest), run_staging, "condition")
    bundle = json.loads(
        (run_staging / "artifacts/conditioning/bundle.json").read_text(encoding="utf-8")
    )
    return {
        "source": source,
        "digest": digest,
        "staging": run_staging,
        "response": response,
        "bundle": bundle,
        "base": base,
    }


# --- The run succeeded ----------------------------------------------------------


def test_conditioning_succeeds(conditioned) -> None:
    response = conditioned["response"]
    assert response["status"] == "succeeded", response["diagnostics"]
    assert response["diagnostics"] == []


def test_the_source_is_byte_identical_afterwards(conditioned) -> None:
    assert _sha256(conditioned["source"]) == conditioned["digest"]


def test_the_portable_labels_carry_no_datablock_name(conditioned) -> None:
    rendered = json.dumps(conditioned["response"])
    for private in (TARGET, CAMERA, RIG, ACTION, str(conditioned["source"])):
        assert private not in rendered, private


# --- The bundle -----------------------------------------------------------------


def test_the_bundle_validates_against_the_committed_schema(conditioned) -> None:
    validator = Draft202012Validator(load_schema("conditioning-bundle", "1.0"))
    errors = sorted(validator.iter_errors(conditioned["bundle"]), key=lambda e: e.json_path)
    assert errors == [], [f"{e.json_path}: {e.message}" for e in errors[:5]]


def test_the_bundle_is_self_sealed(conditioned) -> None:
    bundle = conditioned["bundle"]
    preimage = {key: value for key, value in bundle.items() if key != "bundle_sha256"}
    assert canonical_digest(preimage) == bundle["bundle_sha256"]


def test_the_bundle_passes_the_apache_validator(conditioned) -> None:
    validate_conditioning_bundle(
        conditioned["bundle"],
        metrics=conditioned["response"]["metrics"],
        outputs=conditioned["response"]["outputs"],
        run_directory=conditioned["staging"],
    )


def test_the_pass_inventory_is_complete_and_ordered(conditioned) -> None:
    assert [item["role"] for item in conditioned["bundle"]["passes"]] == EXPECTED_PASS_ROLES


def test_every_declared_pass_exists_with_its_recorded_digest(conditioned) -> None:
    staging = conditioned["staging"]
    for item in conditioned["bundle"]["passes"]:
        path = staging / item["path"]
        assert path.is_file(), item["role"]
        assert _sha256(path) == item["sha256"], item["role"]
        assert path.stat().st_size == item["byte_size"], item["role"]


def test_the_canonical_passes_are_scene_linear_exr_and_previews_are_srgb(conditioned) -> None:
    spaces = {item["role"]: item["color_space"] for item in conditioned["bundle"]["passes"]}
    assert spaces["beauty_exr"] == "scene_linear"
    assert spaces["beauty_preview"] == "srgb"
    assert spaces["depth_exr"] == "data"
    assert spaces["object_index_exr"] == "data"
    assert spaces["mask_png"] == "data"


def test_the_declared_semantics_match_the_binding_profile(conditioned) -> None:
    bundle = conditioned["bundle"]
    assert bundle["pixel_origin"] == "top_left"
    assert bundle["pixel_aspect"] == [1.0, 1.0]
    assert bundle["depth"]["space"] == "camera_euclidean_distance"
    assert bundle["depth"]["background"] == "invalid_by_mask"
    assert bundle["normal"]["space"] == "world"
    assert bundle["mask"] == {
        "target_object_index": 1,
        "foreground": 255,
        "background": 0,
        "pass_alpha_threshold": 0.5,
        "antialiasing": "none",
    }
    assert bundle["render_profile"] == FIXTURE_RENDER_PROFILE
    assert bundle["blender"] == BLENDER_FINGERPRINT


# --- Measured pixels ------------------------------------------------------------


def test_every_mask_foreground_pixel_has_finite_depth(conditioned) -> None:
    metrics = conditioned["response"]["metrics"]
    assert metrics["foreground_pixel_count"] > 0
    assert metrics["finite_foreground_depth_count"] == metrics["foreground_pixel_count"]


def test_the_target_covers_a_usable_share_of_the_frame(conditioned) -> None:
    metrics = conditioned["response"]["metrics"]
    total = RESOLUTION[0] * RESOLUTION[1]
    assert metrics["foreground_pixel_count"] / total > 0.05


def test_the_eroded_interior_carries_unit_normals(conditioned) -> None:
    metrics = conditioned["response"]["metrics"]
    assert metrics["interior_unit_normal_count"] > 0
    assert metrics["interior_unit_normal_count"] <= metrics["foreground_pixel_count"]


def test_the_published_matrices_agree_with_blenders_own_projection(conditioned) -> None:
    """The fiducial cross-check: our matrix chain against Blender's projection path."""
    error = conditioned["response"]["metrics"]["projection_max_error_pixels"]
    assert error is not None
    assert 0.0 <= error <= PROJECTION_LIMIT_PIXELS


# --- Upload eligibility ---------------------------------------------------------


def test_the_derived_scene_is_local_and_outside_the_upload_directory(conditioned) -> None:
    outputs = {item["role"]: item["path"] for item in conditioned["response"]["outputs"]}
    assert outputs["scene_state_blend"] == "artifacts/local/scene-state.blend"
    assert not outputs["scene_state_blend"].startswith("artifacts/conditioning/")
    assert (conditioned["staging"] / outputs["scene_state_blend"]).is_file()


def test_every_upload_eligible_artifact_sits_in_the_conditioning_directory(
    conditioned,
) -> None:
    for item in conditioned["bundle"]["passes"]:
        assert item["upload_eligible"] is True
        assert item["path"].startswith("artifacts/conditioning/")


# --- Determinism ----------------------------------------------------------------


@pytest.fixture(scope="module")
def repeated(conditioned):
    """A second conditioning run of the same request, for the determinism classes."""
    staging = conditioned["base"] / "run-again"
    staging.mkdir()
    response = _drive(
        _condition_request(conditioned["source"], conditioned["digest"]), staging, "condition"
    )
    assert response["status"] == "succeeded", response["diagnostics"]
    bundle = json.loads(
        (staging / "artifacts/conditioning/bundle.json").read_text(encoding="utf-8")
    )
    return {"staging": staging, "response": response, "bundle": bundle}


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/conditioning/mask.png",
        "artifacts/conditioning/beauty.png",
        "artifacts/conditioning/depth-preview.png",
        "artifacts/conditioning/normal-preview.png",
    ],
)
def test_decoded_eight_bit_artifacts_are_byte_exact_across_runs(
    conditioned, repeated, relative: str
) -> None:
    """The byte-exact class: binary masks and 8-bit previews."""
    assert _sha256(conditioned["staging"] / relative) == _sha256(repeated["staging"] / relative)


def test_the_metrics_are_semantically_exact_across_runs(conditioned, repeated) -> None:
    assert conditioned["response"]["metrics"] == repeated["response"]["metrics"]


def test_the_bundle_is_repeat_run_equivalent_after_normalizing_exr_digests(
    conditioned, repeated
) -> None:
    """EXR container bytes are never compared across runs; their content is.

    The design excludes EXR container bytes from every determinism claim, because the
    container embeds run metadata. Normalizing exactly those digests and sizes leaves the
    rest of the bundle, which must match.
    """

    def normalize(bundle: dict) -> dict:
        copy = json.loads(json.dumps(bundle))
        copy.pop("bundle_sha256")
        for item in copy["passes"]:
            if item["media_type"] == "image/x-exr":
                item["sha256"] = "<normalized>"
                item["byte_size"] = 0
        return copy

    assert normalize(conditioned["bundle"]) == normalize(repeated["bundle"])


def test_a_repeated_run_does_not_reuse_the_first_run_directory(conditioned, repeated) -> None:
    assert conditioned["staging"] != repeated["staging"]
    assert (conditioned["staging"] / "artifacts/conditioning/bundle.json").is_file()


# --- Refusals -------------------------------------------------------------------


def test_a_tampered_selection_digest_refuses_to_condition(conditioned) -> None:
    staging = conditioned["base"] / "run-tampered"
    staging.mkdir()
    request = _condition_request(conditioned["source"], conditioned["digest"])
    request["portable_selection"]["target_label"] = "mesh-9"

    response = _drive(request, staging, "condition")
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["PLAN_TAMPERED"]
    assert not (staging / "artifacts").exists()


def test_a_bundle_whose_pass_bytes_changed_fails_validation(conditioned) -> None:
    staging = conditioned["base"] / "run-tampered-bytes"
    staging.mkdir()
    response = _drive(
        _condition_request(conditioned["source"], conditioned["digest"]), staging, "condition"
    )
    assert response["status"] == "succeeded"
    bundle = json.loads(
        (staging / "artifacts/conditioning/bundle.json").read_text(encoding="utf-8")
    )
    (staging / "artifacts/conditioning/mask.png").write_bytes(b"tampered\n")

    with pytest.raises(BundleInvalid, match="recorded digest"):
        validate_conditioning_bundle(
            bundle,
            metrics=response["metrics"],
            outputs=response["outputs"],
            run_directory=staging,
        )
