"""Real-Blender export and fresh-process round trip.

The chain runs fixture -> condition -> ingest -> bake -> export, then reimports each
exported file in a *separate* Blender process. Raw archive bytes are never compared; the
design does not claim byte-identical GLB or FBX, so what is compared is a format-aware
semantic fingerprint.
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
    ContainerInvalid,
    alignment_acknowledgement_text,
    ingest_view,
    sha256_file,
    validate_blend,
    validate_fbx,
    validate_glb,
    validate_glb_alpha_profile,
    validate_glb_has_no_absolute_resource,
)

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts" / "run_blender_e2e.py"
BLENDER = Path("/Applications/Blender.app/Contents/MacOS/Blender")
TOOL_INVENTORY = ROOT / "tools" / "gltf-validator.json"

sys.path.insert(0, str(ROOT / "blender-addon" / "src"))

pytestmark = pytest.mark.skipif(
    not BLENDER.exists(), reason="the pinned Blender 5.2.0 install is unavailable"
)

SALT = bytes(range(32))
TARGET, CAMERA, RIG, ACTION = "Robot_Strip_Body", "Shot_Camera", "Robot_Rig", "Robot_Flex"
FRAME = 2
ACTION_RANGE = [1, 2]
RESOLUTION = [64, 64]
ATLAS = [64, 64]
BLENDER_FINGERPRINT = {
    "profile": "blender-5.2.0-cpu-v1-fixture",
    "version": "5.2.0",
    "build_hash": "fbe6228777e7",
    "executable_sha256": "60ba7a9b6743f7acf101274361fa76409e382ae07cd2007ce07dea30f6b129f2",
}
EXPECTED_ARTIFACTS = [
    "exports/asset.blend",
    "exports/asset.fbx",
    "exports/asset.glb",
    "exports/asset.png",
    "exports/fingerprint.json",
]


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


def _selection(digest: str, *, rigged: bool = True) -> dict:
    from asset_mania_blender.selection import selection_digest

    identity = {
        "source_scene_sha256": digest,
        "camera": CAMERA,
        "target": TARGET,
        "target_type": "MESH",
        "armature": RIG if rigged else None,
        "action": ACTION if rigged else None,
    }
    labels = {
        "camera_label": "camera-1",
        "target_label": "mesh-1",
        "armature_label": "armature-1" if rigged else None,
        "action_label": "action-1" if rigged else None,
    }
    return {
        **labels,
        "selection_digest": selection_digest(salt=SALT, identity=identity, labels=labels),
    }


def _run_chain(base: Path, *, variant: str | None = None, rigged: bool = True) -> dict:
    build = base / "build"
    build.mkdir(parents=True)
    request = {
        "request_id": "request-fixture-1",
        "operation": "fixture",
        "fixture_name": "fixture.blend",
        "sentinel_path": str(base / "sentinel.txt"),
    }
    if variant is not None:
        request["variant"] = variant
    _drive(request, build, "validate")

    source = base / "fixture.blend"
    shutil.move(build / "fixture.blend", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    common = {
        "source_path": str(source),
        "source_scene_sha256": digest,
        "target_name": TARGET,
        "camera_name": CAMERA,
        "frame": FRAME,
        "selection_salt": SALT.hex(),
        "portable_selection": _selection(digest, rigged=rigged),
    }
    if rigged:
        common["armature_name"] = RIG
        common["action_name"] = ACTION

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
    baked = _drive(
        {
            "request_id": "request-bake-1",
            "operation": "bake",
            "condition_run_directory": str(condition_staging),
            "view_path": str(ingested["normalized_path"]),
            "atlas_size": ATLAS,
            "bake_margin": 2,
            "color_padding": 1,
            "minimum_coverage": 0.25,
            **common,
        },
        bake_staging,
        "bake",
    )
    assert baked["status"] == "succeeded", baked["diagnostics"]

    export_staging = base / "export"
    export_staging.mkdir()
    export_request = {
        "request_id": "request-export-1",
        "operation": "export",
        "formats": ["blend", "glb", "fbx"],
        "texture_path": str(bake_staging / "textures/albedo.png"),
        **common,
    }
    if rigged:
        export_request["action_range"] = ACTION_RANGE
    exported = _drive(export_request, export_staging, "export")

    return {
        "base": base,
        "source": source,
        "digest": digest,
        "common": common,
        "export_request": export_request,
        "exported": exported,
        "staging": export_staging,
        "bake_staging": bake_staging,
    }


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    return _run_chain(tmp_path_factory.mktemp("export"))


@pytest.fixture(scope="module")
def static_exported(tmp_path_factory):
    """A static prop: no rig, no action, and therefore no exported animation."""
    return _run_chain(tmp_path_factory.mktemp("export-static"), variant="static-prop", rigged=False)


# --- The export succeeded ---------------------------------------------------------


def test_the_export_succeeds(exported) -> None:
    assert exported["exported"]["status"] == "succeeded", exported["exported"]["diagnostics"]
    assert exported["exported"]["diagnostics"] == []


def test_every_declared_format_is_written(exported) -> None:
    outputs = {item["path"]: item for item in exported["exported"]["outputs"]}
    assert sorted(outputs) == EXPECTED_ARTIFACTS
    for path, item in outputs.items():
        on_disk = exported["staging"] / path
        assert on_disk.is_file(), path
        assert sha256_file(on_disk) == item["sha256"], path


def test_the_source_is_byte_identical_afterwards(exported) -> None:
    assert hashlib.sha256(exported["source"].read_bytes()).hexdigest() == exported["digest"]


def test_the_response_carries_no_private_name(exported) -> None:
    rendered = json.dumps(exported["exported"])
    for private in (TARGET, CAMERA, RIG, ACTION, str(exported["source"])):
        assert private not in rendered, private


# --- Containers -------------------------------------------------------------------


def test_the_derived_blend_has_a_valid_uncompressed_header(exported) -> None:
    header = validate_blend(exported["staging"] / "exports/asset.blend")
    assert header.endianness in ("little", "big")
    assert header.version.isdigit()


def test_the_glb_is_a_well_formed_gltf_two_container(exported) -> None:
    container = validate_glb(exported["staging"] / "exports/asset.glb")
    assert container.json_chunk["asset"]["version"] == "2.0"
    assert container.binary_length > 0
    assert container.total_length == (exported["staging"] / "exports/asset.glb").stat().st_size


def test_the_glb_carries_the_selected_mesh_rig_camera_and_material(exported) -> None:
    container = validate_glb(exported["staging"] / "exports/asset.glb")
    assert len(container.json_chunk.get("meshes", [])) == 1
    assert len(container.json_chunk.get("materials", [])) == 1
    assert len(container.json_chunk.get("cameras", [])) == 1
    assert len(container.json_chunk.get("skins", [])) == 1
    assert len(container.json_chunk.get("animations", [])) == 1


def test_unknown_texels_are_exported_as_a_masked_alpha(exported) -> None:
    """The property that makes an uncovered texel read as absent, not black."""
    container = validate_glb(exported["staging"] / "exports/asset.glb")
    validate_glb_alpha_profile(container)
    assert container.json_chunk["materials"][0]["alphaMode"] == "MASK"


def test_the_glb_references_no_resource_outside_the_container(exported) -> None:
    validate_glb_has_no_absolute_resource(validate_glb(exported["staging"] / "exports/asset.glb"))


def test_the_fbx_is_binary_and_within_the_declared_subset(exported) -> None:
    header = validate_fbx(exported["staging"] / "exports/asset.fbx")
    assert 7100 <= header.version <= 8000


def test_the_fbx_texture_travels_as_a_declared_member_of_the_group(exported) -> None:
    """FBX embedding is not round-trip reliable here, so the PNG is a declared sibling."""
    texture = exported["staging"] / "exports/asset.png"
    assert texture.is_file()
    assert sha256_file(texture) == sha256_file(exported["bake_staging"] / "textures/albedo.png")


# --- Fresh-process round trip ------------------------------------------------------


def _exported_fingerprint(exported) -> dict:
    """The fingerprint the export published as a declared artifact."""
    payload = json.loads(
        (exported["staging"] / "exports/fingerprint.json").read_text(encoding="utf-8")
    )
    return payload


def _reimport(exported, kind: str, relative: str) -> dict:
    staging = exported["base"] / f"reimport-{kind}"
    staging.mkdir(exist_ok=True)
    response = _drive(
        {
            "request_id": f"request-reimport-{kind}",
            "operation": "reimport",
            "import_path": str(exported["staging"] / relative),
            "import_kind": kind,
            "sample_frames": _exported_fingerprint(exported)["sample_frames"],
        },
        staging,
        "validate",
        timeout=900,
    )
    reimported = json.loads((staging / "reimport-fingerprint.json").read_text(encoding="utf-8"))
    return {"response": response, "fingerprint": reimported["fingerprint"]}


@pytest.mark.parametrize(
    ("kind", "relative"),
    [
        ("blend", "exports/asset.blend"),
        ("glb", "exports/asset.glb"),
        ("fbx", "exports/asset.fbx"),
    ],
)
def test_each_format_reimports_in_a_fresh_process(exported, kind: str, relative: str) -> None:
    result = _reimport(exported, kind, relative)
    response = result["response"]
    assert response["status"] == "succeeded", response["diagnostics"]
    assert response["metrics"]["profile"] == f"reimport-{kind}-v1"
    assert len(response["metrics"]["semantic_digest"]) == 64
    assert result["fingerprint"]["mesh_count"] >= 1


def test_the_reopened_blend_preserves_the_authoring_scene(exported) -> None:
    """The blend is the editable artifact, so it must come back intact."""
    exported_fingerprint = _exported_fingerprint(exported)["fingerprint"]
    reimported = _reimport(exported, "blend", "exports/asset.blend")["fingerprint"]
    for key in (
        "mesh_count",
        "bone_count",
        "camera_count",
        "material_count",
        "vertex_count",
        "polygon_count",
        "uv_layer_count",
    ):
        assert reimported[key] == exported_fingerprint[key], key


@pytest.mark.parametrize("kind", ["glb", "fbx"])
def test_a_runtime_format_preserves_the_target_topology_and_rig(exported, kind: str) -> None:
    """Compare only what a runtime format preserves.

    GLB and FBX convert axes; glTF stores one vertex per unique position/normal/UV tuple,
    so a seam legitimately raises the vertex count; and glTF stores triangles only, so
    three quads legitimately become six triangles. Triangle count and bone count are the
    invariants. Object counts are not, because an importer may synthesize helper geometry
    that was never in the file.
    """
    relative = f"exports/asset.{kind}"
    exported_fingerprint = _exported_fingerprint(exported)["fingerprint"]
    reimported = _reimport(exported, kind, relative)["fingerprint"]
    assert reimported["triangle_count"] == exported_fingerprint["triangle_count"]
    assert reimported["bone_count"] == exported_fingerprint["bone_count"]
    assert reimported["vertex_count"] >= exported_fingerprint["vertex_count"]
    assert reimported["uv_layer_count"] == exported_fingerprint["uv_layer_count"]


@pytest.mark.parametrize("kind", ["glb", "fbx"])
def test_a_delivered_runtime_artifact_carries_no_private_datablock_name(
    exported, kind: str
) -> None:
    """A GLB writes its node names into the file, so they must be portable labels."""
    payload = (exported["staging"] / f"exports/asset.{kind}").read_bytes()
    for private in (TARGET, CAMERA, RIG, ACTION, b"Base_Joint".decode(), "Tip_Joint"):
        assert private.encode("utf-8") not in payload, private
    assert b"mesh-1" in payload or b"camera-1" in payload


# --- A static target -----------------------------------------------------------------


def test_a_static_target_exports_no_animation(static_exported) -> None:
    assert static_exported["exported"]["status"] == "succeeded", static_exported["exported"][
        "diagnostics"
    ]
    container = validate_glb(static_exported["staging"] / "exports/asset.glb")
    assert container.json_chunk.get("animations", []) == []
    assert container.json_chunk.get("skins", []) == []


def test_a_static_target_still_exports_its_mesh_and_material(static_exported) -> None:
    container = validate_glb(static_exported["staging"] / "exports/asset.glb")
    assert len(container.json_chunk.get("meshes", [])) == 1
    assert len(container.json_chunk.get("materials", [])) == 1


# --- Refusals -------------------------------------------------------------------------


def test_an_existing_output_is_never_overwritten(exported) -> None:
    """A second export into the same staging tree collides rather than replacing."""
    response = _drive(exported["export_request"], exported["staging"], "export")
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["OUTPUT_COLLISION"]


def test_a_tampered_selection_refuses_to_export(exported) -> None:
    staging = exported["base"] / "export-tampered"
    staging.mkdir()
    request = dict(exported["export_request"])
    request["portable_selection"] = {
        **request["portable_selection"],
        "target_label": "mesh-9",
    }
    response = _drive(request, staging, "export")
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["PLAN_TAMPERED"]
    assert not (staging / "exports").exists()


def test_a_missing_texture_refuses_to_export(exported) -> None:
    staging = exported["base"] / "export-no-texture"
    staging.mkdir()
    request = dict(exported["export_request"])
    request["texture_path"] = str(exported["base"] / "absent.png")
    response = _drive(request, staging, "export")
    assert response["status"] == "failed"
    assert response["diagnostics"] == ["EXPORT_OPERATOR_UNAVAILABLE"]


@pytest.mark.parametrize(
    ("kind", "payload"),
    [
        ("blend", b"NOTBLENDER"),
        ("glb", b"glTF" + b"\x00" * 8),
        ("fbx", b"not an fbx file at all" + b"\x00" * 8),
    ],
)
def test_a_malformed_container_is_rejected(tmp_path: Path, kind: str, payload: bytes) -> None:
    path = tmp_path / f"broken.{kind}"
    path.write_bytes(payload)
    validator = {"blend": validate_blend, "glb": validate_glb, "fbx": validate_fbx}[kind]
    with pytest.raises(ContainerInvalid):
        validator(path)


def test_a_compressed_blend_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "compressed.blend"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 32)
    with pytest.raises(ContainerInvalid, match="uncompressed"):
        validate_blend(path)


def test_a_glb_whose_declared_length_is_wrong_is_rejected(tmp_path: Path) -> None:
    import struct

    path = tmp_path / "wrong-length.glb"
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 9999) + b"\x00" * 8)
    with pytest.raises(ContainerInvalid, match="declares"):
        validate_glb(path)


# --- What is not claimed --------------------------------------------------------------


def test_the_gltf_validator_is_recorded_as_unacquired() -> None:
    """Honest gap: no Khronos glTF Validator run backs the GLB claim yet.

    The container structure, the alpha profile, the resource containment, and a
    fresh-process reimport are all verified above. The Khronos validator is a separate,
    stronger check, and its inventory records that no release has been pinned or verified,
    so nothing here claims a validator-clean GLB.
    """
    inventory = json.loads(TOOL_INVENTORY.read_text(encoding="utf-8"))
    assert inventory["tool"] == "gltf-validator"
    assert inventory["acquisition"]["official_archives_status"] == "unverified"
    assert inventory["verified_installs"] == []
    assert inventory["verified_installs_status"] == "not_installed"
