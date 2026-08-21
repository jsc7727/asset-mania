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
        assert (
            _disclosure(subject="synthetic_person", receipt=None)["rights_receipt_sha256"] is None
        )

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


class TestTheDisclosureTravelsWithTheMeshRecord:
    """The design says the disclosure is produced by the same call that describes the mesh.

    It was not, for a while: the builder existed and nothing called it. A disclosure a caller
    has to remember to produce is one that will eventually travel apart from its mesh, which
    leaves exactly the artifact this version exists to prevent -- a head mesh with nothing
    saying where it came from or what was never measured about it.
    """

    @staticmethod
    def _plan(asset_kind: str, subject: str, receipt: str | None):
        from asset_mania_contracts import build_reconstruction_plan

        return build_reconstruction_plan(
            engine="triposr-local",
            engine_profile="triposr-local-cpu-v1",
            clearance_sha256="a" * 64,
            source_image_sha256="b" * 64,
            source_width=512,
            source_height=512,
            alpha=True,
            mask_sha256="c" * 64,
            background_removal_clearance_sha256=None,
            asset_kind=asset_kind,
            subject=subject,
            rights_receipt_sha256=receipt,
            expected_output={"mesh_format": "glb", "textured": False, "unit_scale_meters": 1.0},
        )

    @staticmethod
    def _record(plan, mesh: Path):
        from asset_mania_pipeline import describe_reconstruction_output

        return describe_reconstruction_output(
            mesh_path=mesh, plan=plan, triangle_count=1000, vertex_count=500, manifold="closed"
        )

    @pytest.fixture
    def mesh(self, tmp_path: Path) -> Path:
        path = tmp_path / "mesh.glb"
        path.write_bytes(b"glTF" + bytes(200))
        return path

    def test_a_face_head_record_carries_a_sealed_disclosure(self, mesh: Path) -> None:
        record = self._record(self._plan("face_head", "real_person", RECEIPT), mesh)
        disclosure = record["disclosure"]
        assert disclosure is not None
        jsonschema.validate(disclosure, load_schema("likeness-disclosure", "1.0"))
        preimage = {k: v for k, v in disclosure.items() if k != "disclosure_sha256"}
        assert disclosure["disclosure_sha256"] == canonical_digest(preimage)

    def test_the_disclosure_digests_the_mesh_that_was_measured(self, mesh: Path) -> None:
        """Not a digest the caller passed in -- the file this record describes."""
        from asset_mania_pipeline import sha256_file

        record = self._record(self._plan("face_head", "real_person", RECEIPT), mesh)
        assert record["disclosure"]["mesh_sha256"] == sha256_file(mesh)

    def test_the_disclosure_cannot_disagree_with_the_plan_that_gated_the_run(
        self, mesh: Path
    ) -> None:
        """Subject and receipt come from the plan, so the two cannot drift into disagreement."""
        plan = self._plan("face_head", "real_person", RECEIPT)
        disclosure = self._record(plan, mesh)["disclosure"]
        assert disclosure["subject"] == plan["subject"]
        assert disclosure["rights_receipt_sha256"] == plan["rights_receipt_sha256"]
        assert disclosure["plan_sha256"] == plan["plan_sha256"]
        assert disclosure["source_image_sha256"] == plan["source_image_sha256"]
        assert disclosure["engine"] == plan["engine"]
        assert disclosure["engine_profile"] == plan["engine_profile"]

    def test_a_synthetic_face_records_no_receipt(self, mesh: Path) -> None:
        plan = self._plan("face_head", "synthetic_person", None)
        assert self._record(plan, mesh)["disclosure"]["rights_receipt_sha256"] is None

    @pytest.mark.parametrize("asset_kind", ["object", "character"])
    def test_other_kinds_get_no_disclosure(self, mesh: Path, asset_kind: str) -> None:
        """`None`, not an empty disclosure: an object mesh has no likeness to disclose."""
        plan = self._plan(asset_kind, "non_person", None)
        assert self._record(plan, mesh)["disclosure"] is None
