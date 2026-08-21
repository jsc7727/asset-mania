# SPDX-License-Identifier: GPL-3.0-or-later
"""Worker-side tests that must run inside Blender.

These cover the branches an Apache test cannot reach: pure checks over fabricated inputs,
label assignment, sanitization effects on a live scene, and the salted selection digest.

Run with:
    blender --background --factory-startup --disable-autoexec --offline-mode \
      --python blender-addon/tests/run_e2e.py
"""

import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asset_mania_blender import (
    fixture_factory,
    labels,
    protocol,
    scene_inventory,
    selection,
)

_FAILURES: list[str] = []


def check(condition: bool, description: str) -> None:
    if not condition:
        _FAILURES.append(description)


def test_matrix_findings_rejects_a_non_finite_matrix() -> None:
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    check(scene_inventory.matrix_findings(identity, 1.0) == [], "identity is clean")

    for bad in (float("inf"), float("nan"), -float("inf")):
        values = list(identity)
        values[3] = bad
        check(
            scene_inventory.matrix_findings(values, 1.0) == ["nonfinite_matrix"],
            f"non-finite component {bad!r} is rejected",
        )

    check(
        scene_inventory.matrix_findings(identity, float("nan")) == ["nonfinite_determinant"],
        "non-finite determinant is rejected",
    )
    check(
        scene_inventory.matrix_findings(identity, 0.0) == ["singular_transform"],
        "singular transform is rejected",
    )
    check(
        scene_inventory.matrix_findings(identity, -1.0) == ["negative_determinant"],
        "negative determinant is rejected",
    )


def test_labels_are_deterministic_and_private_free() -> None:
    first = labels.assign_labels("mesh", ["Zebra", "Alpha", "Middle"])
    second = labels.assign_labels("mesh", ["Middle", "Zebra", "Alpha"])
    check(first == second, "label assignment ignores input order")
    check(first["Alpha"] == "mesh-1", "labels follow sorted private names")
    check(
        all(not label.startswith(("Zebra", "Alpha")) for label in first.values()),
        "labels reveal no private name",
    )


def test_selection_digest_binds_identity_and_labels() -> None:
    identity = {
        "source_scene_sha256": "1a" * 32,
        "camera": "Shot_Camera",
        "target": "Robot_Strip_Body",
        "target_type": "MESH",
        "armature": "Robot_Rig",
        "action": "Robot_Flex",
    }
    label_map = {
        "camera_label": "camera-1",
        "target_label": "mesh-1",
        "armature_label": "armature-1",
        "action_label": "action-1",
    }
    salt = bytes(range(32))
    baseline = selection.selection_digest(salt=salt, identity=identity, labels=label_map)

    for field, value in identity.items():
        changed = {**identity, field: f"{value}-changed"}
        check(
            selection.selection_digest(salt=salt, identity=changed, labels=label_map) != baseline,
            f"identity field {field} is bound",
        )
    for field in label_map:
        changed = {**label_map, field: "mesh-9"}
        check(
            selection.selection_digest(salt=salt, identity=identity, labels=changed) != baseline,
            f"label field {field} is bound",
        )
    check(
        selection.selection_digest(salt=bytes(32), identity=identity, labels=label_map) != baseline,
        "the salt is bound",
    )

    portable = {**label_map, "selection_digest": baseline}
    check(
        selection.verify_selection(
            salt_hex=salt.hex(), identity=identity, portable_selection=portable
        ),
        "a matching selection verifies",
    )
    check(
        not selection.verify_selection(
            salt_hex="zz", identity=identity, portable_selection=portable
        ),
        "a malformed salt fails closed",
    )


def test_sanitization_disables_every_write_surface() -> None:
    import bpy

    fixture_factory.build_fixture()
    scene = bpy.context.scene

    group = bpy.data.node_groups.new("Probe_Composite", "CompositorNodeTree")
    group.nodes.new("CompositorNodeOutputFile")
    scene.compositing_node_group = group
    scene.use_nodes = True
    scene.render.use_border = True
    scene.render.use_freestyle = True

    with tempfile.TemporaryDirectory() as staging:
        actions = scene_inventory.sanitize_write_surfaces(staging)

        # `use_nodes` is deliberately not asserted: on 5.2 it stays True even when
        # assigned False, so the detached group is the guarantee that matters.
        check(scene.compositing_node_group is None, "the compositor group is detached")
        check(
            "detached_compositor_node_group" in actions,
            "the action list records the detach",
        )
        check(scene.render.use_border is False, "the render border is disabled")
        check(scene.render.use_freestyle is False, "Freestyle is disabled")
        check(
            scene.render.filepath.startswith(staging),
            "render output is redirected below staging",
        )
        check(
            bpy.context.preferences.filepaths.temporary_directory.startswith(staging),
            "the temporary directory is redirected below staging",
        )
        check("redirected_render_output" in actions, "the action list records the redirect")
        check(actions == sorted(actions), "the action list is sorted")

        cycles = getattr(scene, "cycles", None)
        if cycles is not None:
            check(
                getattr(cycles, "shading_system", False) is False,
                "open shading language is disabled",
            )


def test_the_fixture_matches_its_declared_shape() -> None:
    description = fixture_factory.build_fixture()
    check(description["vertex_count"] >= 6, "the fixture has at least six vertices")
    check(description["uv_layer_count"] == 1, "the fixture has one UV layer")
    check(description["bone_names"] == ["Base_Joint", "Tip_Joint"], "the fixture has two bones")
    check(description["external_dependency_count"] == 0, "the fixture has no external file")
    check(description["packed_image_count"] == 1, "the fixture texture is packed")
    check(
        math.isclose(float(description["deformation_degrees"]), 30.0),
        "the deformed frame rotates by 30 degrees",
    )
    check(scene_inventory.external_dependencies() == [], "no external dependency is reported")
    check(scene_inventory.code_execution_surfaces() == [], "no code-execution surface is present")


def test_the_response_seal_ignores_worker_scratch_keys() -> None:
    sealed = protocol.seal_response(
        {"schema_id": protocol.SCHEMA_ID, "_scratch": "local", "response_sha256": ""}
    )
    check("_scratch" not in sealed, "underscore keys are dropped before sealing")
    check(len(sealed["response_sha256"]) == 64, "the response is sealed")


def main() -> int:
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
            except Exception as error:  # noqa: BLE001 - report, never abort the suite
                _FAILURES.append(f"{name} raised {type(error).__name__}: {error}")

    for failure in _FAILURES:
        print(f"WORKER_TEST_FAILED {failure}")
    print(f"WORKER_TESTS {'ok' if not _FAILURES else 'failed'} ({len(_FAILURES)} failures)")
    return 1 if _FAILURES else 0


raise SystemExit(main())
