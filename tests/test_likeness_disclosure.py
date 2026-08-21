"""v0.4: what a face_head mesh carries with it, and what it is forbidden from claiming.

A head mesh on a disk is indistinguishable from any other head mesh, so the disclosure records
where the geometry came from. The harder half is the accuracy section, because the tempting
mistake there is not silence -- it is quoting the one number that exists.

That number, 0.06012, was measured on a subdivided Suzanne against the mesh that rendered its
input image: smooth, symmetric, textureless, controlled light. It is a real measurement and it
says nothing about a human face. These tests exist to keep it labelled as such and to keep
`face_benchmark` null until something has actually been measured on faces.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from asset_mania_contracts import (
    NON_FACE_ACCURACY_REFERENCE,
    PROHIBITED_LIKENESS_CLAIMS,
    DiagnosticCode,
    build_likeness_disclosure,
    canonical_digest,
    load_schema,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "v2" / "likeness-disclosure-v1.json"

PLAN = "a" * 64
IMAGE = "b" * 64
MESH = "c" * 64
RECEIPT = "d" * 64


def _disclosure(subject: str = "real_person", receipt: str | None = RECEIPT, **overrides):
    arguments = {
        "plan_sha256": PLAN,
        "source_image_sha256": IMAGE,
        "mesh_sha256": MESH,
        "subject": subject,
        "rights_receipt_sha256": receipt,
        "engine": "triposr-local",
        "engine_profile": "triposr-local-cpu-v1",
    }
    arguments.update(overrides)
    return build_likeness_disclosure(**arguments)


class TestTheArtifactIsSealedAndValid:
    def test_it_validates_against_its_own_schema(self) -> None:
        jsonschema.validate(_disclosure(), load_schema("likeness-disclosure", "1.0"))

    def test_the_digest_seals_the_whole_artifact(self) -> None:
        disclosure = _disclosure()
        preimage = {k: v for k, v in disclosure.items() if k != "disclosure_sha256"}
        assert disclosure["disclosure_sha256"] == canonical_digest(preimage)

    def test_the_fixture_matches_what_the_builder_produces(self) -> None:
        """A stale normative example is worse than none: it is a wrong answer with authority."""
        stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
        rebuilt = build_likeness_disclosure(
            plan_sha256=stored["plan_sha256"],
            source_image_sha256=stored["source_image_sha256"],
            mesh_sha256=stored["mesh_sha256"],
            subject=stored["subject"],
            rights_receipt_sha256=stored["rights_receipt_sha256"],
            engine=stored["engine"],
            engine_profile=stored["engine_profile"],
            views=stored["likeness_basis"]["views"],
        )
        assert rebuilt == stored

    def test_it_traces_the_mesh_to_one_exact_input(self) -> None:
        disclosure = _disclosure()
        assert disclosure["plan_sha256"] == PLAN
        assert disclosure["source_image_sha256"] == IMAGE
        assert disclosure["mesh_sha256"] == MESH


class TestNoFaceAccuracyIsClaimed:
    def test_face_benchmark_is_null(self) -> None:
        """Null because nothing has been measured on faces, not because the field is unused."""
        assert _disclosure()["measured_accuracy"]["face_benchmark"] is None

    def test_ground_truth_is_reported_absent(self) -> None:
        """There is no reference mesh for a photograph. That is why the engine is being used."""
        assert _disclosure()["measured_accuracy"]["ground_truth_available"] is False

    def test_the_non_face_figure_is_present_and_labelled_with_its_subject(self) -> None:
        """The number is included on purpose, so it cannot later be reached for unlabelled.

        Omitting it would leave the honest figure nowhere, and the next person wanting an
        accuracy number would find 0.06012 in a commit message with no subject attached.
        """
        reference = _disclosure()["measured_accuracy"]["non_face_reference"]
        assert reference == NON_FACE_ACCURACY_REFERENCE
        assert "Suzanne" in reference["subject"]
        assert reference["unit"] == "fraction of the subject's longest axis"

    def test_the_note_says_the_figure_does_not_transfer(self) -> None:
        note = _disclosure()["measured_accuracy"]["note"]
        assert "does not transfer" in note
        assert "No face accuracy has been measured" in note

    def test_the_accuracy_section_cannot_be_supplied_by_the_caller(self) -> None:
        """A disclosure whose accuracy claims came from the publisher would disclose nothing."""
        with pytest.raises(TypeError):
            _disclosure(measured_accuracy={"ground_truth_available": True})

    def test_the_geometry_is_declared_inferred(self) -> None:
        basis = _disclosure()["likeness_basis"]
        assert basis["inferred"] is True
        assert basis["views"] == 1, "a single view underdetermines the geometry behind a face"


class TestProhibitedClaims:
    def test_all_three_claims_are_recorded(self) -> None:
        assert _disclosure()["prohibited_claims"] == PROHIBITED_LIKENESS_CLAIMS

    def test_the_list_covers_identification_biometrics_and_matching(self) -> None:
        claims = set(PROHIBITED_LIKENESS_CLAIMS)
        assert "identification_grade_likeness" in claims
        assert "biometric_record" in claims
        assert "match_to_a_specific_person" in claims

    def test_the_schema_refuses_a_narrowed_list(self) -> None:
        """A caller must not be able to shorten the list by omission."""
        disclosure = _disclosure()
        disclosure["prohibited_claims"] = ["biometric_record"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(disclosure, load_schema("likeness-disclosure", "1.0"))

    def test_the_schema_refuses_an_invented_claim_name(self) -> None:
        disclosure = _disclosure()
        disclosure["prohibited_claims"] = [*PROHIBITED_LIKENESS_CLAIMS, "definitely_accurate"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(disclosure, load_schema("likeness-disclosure", "1.0"))


class TestTheSubjectGatesCarryOver:
    def test_non_person_is_refused(self) -> None:
        with pytest.raises(ValueError, match=DiagnosticCode.SUBJECT_KIND_INCOHERENT.value):
            _disclosure(subject="non_person")

    def test_real_person_without_a_receipt_is_refused(self) -> None:
        with pytest.raises(
            ValueError, match=DiagnosticCode.FACE_RIGHTS_CONFIRMATION_REQUIRED.value
        ):
            _disclosure(subject="real_person", receipt=None)

    def test_synthetic_person_records_no_receipt(self) -> None:
        assert _disclosure(subject="synthetic_person", receipt=None)["rights_receipt_sha256"] is None

    def test_a_receipt_on_a_synthetic_subject_is_refused(self) -> None:
        """A receipt for a subject that cannot grant one is a mislabelled run, not a spare field."""
        with pytest.raises(ValueError, match="does not apply"):
            _disclosure(subject="synthetic_person", receipt=RECEIPT)

    def test_the_schema_pins_asset_kind_to_face_head(self) -> None:
        disclosure = _disclosure()
        assert disclosure["asset_kind"] == "face_head"
        disclosure["asset_kind"] = "object"
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(disclosure, load_schema("likeness-disclosure", "1.0"))
