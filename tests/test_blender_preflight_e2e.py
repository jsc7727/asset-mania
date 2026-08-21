"""Real-Blender preflight, driven through the Apache client.

Every expectation here was measured against Blender 5.2.0 locally, not assumed. The whole
module skips when the pinned Blender is absent, so the fast suite stays offline and
Blender-free; no unit test invokes Blender.
"""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_blender_e2e.py"
WORKER_TESTS = ROOT / "blender-addon" / "tests" / "run_e2e.py"
IMPORT_TEST = ROOT / "blender-addon" / "tests" / "validate_import.py"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")

sys.path.insert(0, str(ROOT / "blender-addon" / "src"))

requires_blender = pytest.mark.skipif(
    not BLENDER.exists(), reason="the pinned Blender 5.2.0 install is unavailable"
)
pytestmark = requires_blender

SALT = bytes(range(32))
TARGET = "Robot_Strip_Body"
CAMERA = "Shot_Camera"
RIG = "Robot_Rig"
ACTION = "Robot_Flex"
BASE_LABELS = ["action-1", "armature-1", "bone-1", "bone-2", "camera-1", "mesh-1"]

#: Measured against Blender 5.2.0. A variant that reports no diagnostic is expected to
#: succeed, which for the malicious variants means sanitization neutralized the surface.
EXPECTED = {
    "valid": [],
    "static-prop": [],
    # Both malicious variants pass preflight because sanitization removed the surface
    # before anything was evaluated. The sentinel assertions below prove nothing was
    # written; a clean preflight here is the intended outcome, not a missed detection.
    "texture-cache": [],
    "compositor-file-output": [],
    "ambiguous-mesh": ["SELECTION_AMBIGUOUS"],
    "ambiguous-camera": ["SELECTION_AMBIGUOUS"],
    "case-colliding-name": ["SELECTION_AMBIGUOUS"],
    "autoexec-driver": ["UNTRUSTED_AUTOEXEC_REQUIRED"],
    "bone-constraint": ["SOURCE_POSE_UNKNOWN"],
    "missing-uv": ["UV_MISSING_OR_INVALID"],
    "overlapping-uv": ["UV_MISSING_OR_INVALID"],
    "uv-outside-unit-range": ["UV_MISSING_OR_INVALID"],
    "negative-determinant": ["POSE_NONFINITE"],
    "singular-scale": ["POSE_NONFINITE"],
    "topology-modifier": ["DEPSGRAPH_TOPOLOGY_CHANGED"],
    "unpacked-image": ["MISSING_LINKED_ASSET"],
    "zero-rig-weights": ["RIG_NOT_FOUND"],
}
#: These variants attempt a write outside staging; the sentinel must never appear.
MALICIOUS = ("autoexec-driver", "compositor-file-output", "texture-cache")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drive(request: dict, staging: Path, operation: str) -> dict:
    request_file = staging.parent / f"{operation}-request.json"
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _selection(source_sha256: str, *, rigged: bool) -> tuple[dict, dict]:
    from asset_mania_blender.selection import selection_digest

    identity = {
        "source_scene_sha256": source_sha256,
        "camera": CAMERA,
        "target": TARGET,
        "target_type": "MESH",
        "armature": RIG if rigged else None,
        "action": ACTION if rigged else None,
    }
    label_map = {
        "camera_label": "camera-1",
        "target_label": "mesh-1",
        "armature_label": "armature-1" if rigged else None,
        "action_label": "action-1" if rigged else None,
    }
    portable = {
        **label_map,
        "selection_digest": selection_digest(salt=SALT, identity=identity, labels=label_map),
    }
    return identity, portable


@pytest.fixture
def workspace(tmp_path: Path):
    def build(variant: str | None = None) -> tuple[Path, Path, Path]:
        build_staging = tmp_path / "build"
        run_staging = tmp_path / "run"
        sentinel = tmp_path / "sentinel" / "must-not-exist.txt"
        for directory in (build_staging, run_staging, sentinel.parent):
            directory.mkdir(parents=True, exist_ok=True)

        request = {
            "request_id": "request-fixture-1",
            "operation": "fixture",
            "fixture_name": "variant.blend",
            "sentinel_path": str(sentinel),
        }
        if variant is not None:
            request["variant"] = variant

        response = _drive(request, build_staging, "validate")
        assert response["status"] == "succeeded", response["diagnostics"]
        source = tmp_path / "variant.blend"
        shutil.move(build_staging / "variant.blend", source)
        return source, run_staging, sentinel

    return build


# --- The base fixture ----------------------------------------------------------


def test_the_fixture_is_generated_into_staging_only(workspace, tmp_path: Path) -> None:
    source, _, _ = workspace()
    assert source.is_file()
    assert source.stat().st_size > 0
    assert not list(tmp_path.parent.glob("*.blend1"))


def test_preflight_accepts_the_fixture_and_reports_portable_labels(workspace) -> None:
    source, staging, _ = workspace()
    digest = _sha256(source)
    _, portable = _selection(digest, rigged=True)

    response = _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": digest,
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": portable,
        },
        staging,
        "preflight",
    )

    assert response["status"] == "succeeded", response["diagnostics"]
    assert response["diagnostics"] == []
    assert response["portable_labels"] == BASE_LABELS
    metrics = response["metrics"]
    assert metrics["target_vertex_count"] == 8
    assert metrics["target_triangle_count"] == 6
    assert metrics["target_uv_layer_count"] == 1
    assert metrics["target_bone_count"] == 2
    assert metrics["external_dependency_count"] == 0
    assert len(metrics["scene_semantic_digest"]) == 64


def test_the_response_carries_no_private_name_or_path(workspace) -> None:
    source, staging, _ = workspace()
    digest = _sha256(source)
    _, portable = _selection(digest, rigged=True)

    response = _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": digest,
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": portable,
        },
        staging,
        "preflight",
    )
    rendered = json.dumps(response)
    for private in (TARGET, CAMERA, RIG, ACTION, str(source), source.name, "Base_Joint"):
        assert private not in rendered, private


def test_the_source_is_byte_identical_after_preflight(workspace) -> None:
    source, staging, _ = workspace()
    before = _sha256(source)
    _, portable = _selection(before, rigged=True)
    _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": before,
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": portable,
        },
        staging,
        "preflight",
    )
    assert _sha256(source) == before


def test_preflight_is_semantically_repeatable(workspace) -> None:
    source, staging, _ = workspace()
    digest = _sha256(source)
    _, portable = _selection(digest, rigged=True)
    request = {
        "request_id": "request-preflight-1",
        "operation": "preflight",
        "source_path": str(source),
        "source_scene_sha256": digest,
        "target_name": TARGET,
        "camera_name": CAMERA,
        "armature_name": RIG,
        "action_name": ACTION,
        "selection_salt": SALT.hex(),
        "portable_selection": portable,
    }
    first = _drive(request, staging, "preflight")
    second_staging = staging.parent / "run-again"
    second_staging.mkdir()
    second = _drive(request, second_staging, "preflight")
    assert first == second


def test_a_tampered_selection_digest_fails(workspace) -> None:
    source, staging, _ = workspace()
    digest = _sha256(source)
    _, portable = _selection(digest, rigged=True)

    response = _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": digest,
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": {**portable, "target_label": "mesh-9"},
        },
        staging,
        "preflight",
    )
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["PLAN_TAMPERED"]


def test_a_different_source_hash_invalidates_the_selection(workspace) -> None:
    source, staging, _ = workspace()
    _, portable = _selection("f" * 64, rigged=True)

    response = _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": _sha256(source),
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": portable,
        },
        staging,
        "preflight",
    )
    assert response["diagnostics"] == ["PLAN_TAMPERED"]


# --- Negative and malicious variants ------------------------------------------


@pytest.mark.parametrize("variant", sorted(EXPECTED))
def test_each_variant_reports_its_expected_diagnostics(workspace, variant: str) -> None:
    source, staging, sentinel = workspace(variant)
    digest = _sha256(source)
    rigged = variant != "static-prop"
    _, portable = _selection(digest, rigged=rigged)

    request = {
        "request_id": "request-preflight-1",
        "operation": "preflight",
        "source_path": str(source),
        "source_scene_sha256": digest,
        "camera_name": CAMERA,
        "target_name": TARGET,
        "selection_salt": SALT.hex(),
        "portable_selection": portable,
    }
    if rigged:
        request["armature_name"] = RIG
        request["action_name"] = ACTION
    if variant == "ambiguous-mesh" or variant == "case-colliding-name":
        del request["target_name"]
    if variant == "ambiguous-camera":
        del request["camera_name"]

    response = _drive(request, staging, "preflight")
    expected = EXPECTED[variant]
    assert response["diagnostics"] == expected, response
    assert response["status"] == ("succeeded" if not expected else "failed")
    assert _sha256(source) == digest
    assert not sentinel.exists()


@pytest.mark.parametrize("variant", MALICIOUS)
def test_a_malicious_variant_never_writes_outside_staging(workspace, variant: str) -> None:
    source, staging, sentinel = workspace(variant)
    digest = _sha256(source)
    _, portable = _selection(digest, rigged=True)

    _drive(
        {
            "request_id": "request-preflight-1",
            "operation": "preflight",
            "source_path": str(source),
            "source_scene_sha256": digest,
            "target_name": TARGET,
            "camera_name": CAMERA,
            "armature_name": RIG,
            "action_name": ACTION,
            "selection_salt": SALT.hex(),
            "portable_selection": portable,
        },
        staging,
        "preflight",
    )

    assert not sentinel.exists()
    assert list(sentinel.parent.iterdir()) == []
    assert source.parent.joinpath("variant.blend").is_file()
    assert not list(source.parent.glob("*.tx"))
    assert _sha256(source) == digest


# --- Worker-side suites --------------------------------------------------------


def test_the_worker_imports_no_apache_package_inside_blender() -> None:
    completed = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--offline-mode",
            "--python",
            str(IMPORT_TEST),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "IMPORT_BOUNDARY ok" in completed.stdout, completed.stdout


def test_the_worker_side_suite_passes_inside_blender() -> None:
    completed = subprocess.run(
        [
            str(BLENDER),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--offline-mode",
            "--python",
            str(WORKER_TESTS),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "WORKER_TESTS ok" in completed.stdout, completed.stdout
