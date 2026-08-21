"""v0.4: a face_head plan cannot be sealed by declaring the face belongs to nothing.

The rights gate keys off `subject`. `asset_kind` was not consulted, so on the frozen v0.3 code
`build_reconstruction_plan(asset_kind="face_head", subject="non_person",
rights_receipt_sha256=None)` sealed successfully -- a face reconstruction with no rights
receipt, reachable by setting one field on a call that has to be filled in anyway.

The gate is not weak there, it is bypassed, and the bypass matters more than most because
subject is a *declaration* by design. v0.1 decided never to infer a subject from pixels, on the
grounds that a classifier deciding whose face is real would be the worse system. That decision
stands, and it makes the declaration the only thing between an engine and a stranger's face.
"""

from __future__ import annotations

import pytest
from asset_mania_contracts import (
    ASSET_KINDS,
    DECLARABLE_SUBJECTS,
    FACE_CAPABLE_SUBJECTS,
    FIXTURE_RENDER_PROFILE,
    DiagnosticCode,
    build_reconstruction_plan,
    build_workflow_plan,
    load_schema,
)

DIGEST = "0" * 64
RECEIPT = "1" * 64


def _plan(asset_kind: str, subject: str, receipt: str | None = None) -> dict:
    return build_reconstruction_plan(
        engine="triposr-local",
        engine_profile="triposr-local-cpu-v1",
        clearance_sha256=DIGEST,
        source_image_sha256=DIGEST,
        source_width=512,
        source_height=512,
        alpha=True,
        mask_sha256=DIGEST,
        background_removal_clearance_sha256=None,
        asset_kind=asset_kind,
        subject=subject,
        rights_receipt_sha256=receipt,
        expected_output={"mesh_format": "glb", "textured": False, "unit_scale_meters": 1.0},
    )


class TestTheBypassIsClosed:
    def test_face_head_with_non_person_is_refused(self) -> None:
        """The exact call that sealed before this gate existed."""
        with pytest.raises(ValueError, match=DiagnosticCode.SUBJECT_KIND_INCOHERENT.value):
            _plan("face_head", "non_person")

    def test_the_refusal_is_not_the_missing_receipt_diagnostic(self) -> None:
        """Two different problems must not share one code.

        A caller who supplied `non_person` has not forgotten a receipt -- they have supplied a
        declaration that cannot be true. Reporting FACE_RIGHTS_CONFIRMATION_REQUIRED would send
        them looking for a receipt they cannot legitimately obtain for that subject.
        """
        with pytest.raises(ValueError) as raised:
            _plan("face_head", "non_person")
        message = str(raised.value)
        assert DiagnosticCode.SUBJECT_KIND_INCOHERENT.value in message
        assert DiagnosticCode.FACE_RIGHTS_CONFIRMATION_REQUIRED.value not in message

    def test_the_refusal_names_the_subjects_that_would_work(self) -> None:
        with pytest.raises(ValueError) as raised:
            _plan("face_head", "non_person")
        for subject in FACE_CAPABLE_SUBJECTS:
            assert subject in str(raised.value)

    def test_the_same_bypass_is_closed_on_the_workflow_plan(self) -> None:
        """The hole was in every builder taking both fields, not just the reconstruction one.

        Fixing only the reconstruction path would have left the conditioning path open and
        looked like a fix in the diff.
        """
        with pytest.raises(ValueError, match=DiagnosticCode.SUBJECT_KIND_INCOHERENT.value):
            build_workflow_plan(
                source_scene_sha256=DIGEST,
                preflight_manifest_sha256=DIGEST,
                selection={"object": "Subject"},
                asset_kind="face_head",
                subject="non_person",
                frame=1,
                action_range=None,
                resolution=[64, 64],
                render_profile=FIXTURE_RENDER_PROFILE,
            )


class TestTheLegitimatePathsStillWork:
    @pytest.mark.parametrize("asset_kind", [k for k in ASSET_KINDS if k != "face_head"])
    @pytest.mark.parametrize("subject", DECLARABLE_SUBJECTS)
    def test_non_face_kinds_are_unaffected(self, asset_kind: str, subject: str) -> None:
        receipt = RECEIPT if subject == "real_person" else None
        plan = _plan(asset_kind, subject, receipt)
        assert plan["asset_kind"] == asset_kind
        assert plan["subject"] == subject

    def test_synthetic_person_needs_no_receipt(self) -> None:
        """A synthetic face has no subject to obtain rights from.

        Requiring a receipt here would leave a caller with an honest declaration and no way to
        proceed, and the available workaround would be to relabel -- so the gate would be
        training people to lie to it.
        """
        plan = _plan("face_head", "synthetic_person")
        assert plan["rights_receipt_sha256"] is None

    def test_real_person_with_a_receipt_is_allowed(self) -> None:
        plan = _plan("face_head", "real_person", RECEIPT)
        assert plan["rights_receipt_sha256"] == RECEIPT

    def test_real_person_without_a_receipt_is_still_refused(self) -> None:
        """The v0.3 gate has to survive the new one being added in front of it."""
        with pytest.raises(
            ValueError, match=DiagnosticCode.FACE_RIGHTS_CONFIRMATION_REQUIRED.value
        ):
            _plan("face_head", "real_person")

    def test_unknown_is_refused_as_it_always_was(self) -> None:
        with pytest.raises(ValueError, match="SUBJECT_DECLARATION_REQUIRED"):
            _plan("face_head", "unknown")


class TestTheGateIsPublishable:
    def test_the_diagnostic_is_in_the_manifest_enum(self) -> None:
        """A code a run can emit but a manifest cannot carry is a code nobody will see."""
        schema = load_schema("run-manifest", "2.0")
        codes = set()

        def walk(node: object) -> None:
            if isinstance(node, dict):
                if isinstance(node.get("enum"), list):
                    codes.update(v for v in node["enum"] if isinstance(v, str))
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(schema)
        assert DiagnosticCode.SUBJECT_KIND_INCOHERENT.value in codes

    def test_face_capable_subjects_is_a_subset_of_declarable_subjects(self) -> None:
        """Guards against the allowlist drifting into a value no caller could declare."""
        assert set(FACE_CAPABLE_SUBJECTS) <= set(DECLARABLE_SUBJECTS)
        assert "non_person" not in FACE_CAPABLE_SUBJECTS
        assert "unknown" not in FACE_CAPABLE_SUBJECTS
