"""The v0.2 command surface: parsing, normalization, and exit codes."""

from pathlib import Path

import pytest
from asset_mania.execution_cli import (
    STAGE_COMMANDS,
    UsageError,
    exit_code,
    parse,
    parse_acknowledgement,
    parse_alignment_acknowledgement,
    parse_expires_in,
    parse_frame,
    parse_resolution,
)

PLAN = "b7" * 32
CONDITION = "b3" * 32
VIEW = "1b" * 32


# --- The command surface ------------------------------------------------------------


def test_every_declared_stage_command_parses() -> None:
    invocations = {
        "scene preflight": ["scene", "preflight", "source.blend"],
        "scene plan": [
            "scene",
            "plan",
            "manifest.json",
            "--camera",
            "Cam",
            "--frame",
            "2",
            "--target",
            "Body",
            "--asset-kind",
            "character",
            "--subject",
            "synthetic-person",
        ],
        "scene condition": ["scene", "condition", "source.blend", "--plan", "plan.json"],
        "view ingest": [
            "view",
            "ingest",
            "view.png",
            "--condition-manifest",
            "m.json",
            "--origin",
            "observed",
        ],
        "view provider-plan": [
            "view",
            "provider-plan",
            "m.json",
            "--provider",
            "openai",
            "--model",
            "gpt-image-2-2026-04-21",
            "--prompt-file",
            "p.txt",
            "--evidence",
            "e.json",
        ],
        "view generate": ["view", "generate", "plan.json", "--prompt-file", "p.txt"],
        "texture bake": [
            "texture",
            "bake",
            "--condition-manifest",
            "c.json",
            "--view-manifest",
            "v.json",
        ],
        "export": ["export", "bake.json", "--format", "blend"],
        "approval issue": ["approval", "issue", "plan.json", "--gate", "paid-compute"],
        "provider evidence refresh": [
            "provider",
            "evidence",
            "refresh",
            "openai",
            "--out",
            "evidence",
        ],
        "engine clearance verify": ["engine", "clearance", "verify", "clearance.json"],
        "image reconstruct": [
            "image",
            "reconstruct",
            "subject.png",
            "--engine",
            "triposr-local",
            "--clearance",
            "clearance.json",
            "--asset-kind",
            "object",
            "--subject",
            "non-person",
        ],
    }
    assert sorted(invocations) == sorted(STAGE_COMMANDS)
    for expected, argv in sorted(invocations.items()):
        assert parse(argv).command == expected, expected


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(UsageError):
        parse(["teleport", "asset"])


def test_a_missing_required_option_is_a_usage_error() -> None:
    with pytest.raises(UsageError):
        parse(["scene", "condition", "source.blend"])


def test_there_is_no_global_yes_flag() -> None:
    """An approval needs the exact plan-bound string; a boolean cannot stand in."""
    with pytest.raises(UsageError):
        parse(["approval", "issue", "plan.json", "--gate", "paid-compute", "--yes"])


# --- Kebab normalization ------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("non-person", "non_person"),
        ("synthetic-person", "synthetic_person"),
        ("real-person", "real_person"),
        ("unknown", "unknown"),
    ],
)
def test_subject_spellings_normalize_to_the_json_form(spelling: str, expected: str) -> None:
    request = parse(
        [
            "scene",
            "plan",
            "m.json",
            "--camera",
            "Cam",
            "--frame",
            "2",
            "--target",
            "Body",
            "--asset-kind",
            "object",
            "--subject",
            spelling,
        ]
    )
    assert request.arguments["subject"] == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [("object", "object"), ("character", "character"), ("face-head", "face_head")],
)
def test_asset_kind_spellings_normalize_to_the_json_form(spelling: str, expected: str) -> None:
    request = parse(
        [
            "scene",
            "plan",
            "m.json",
            "--camera",
            "Cam",
            "--frame",
            "2",
            "--target",
            "Body",
            "--asset-kind",
            spelling,
            "--subject",
            "object" if False else "non-person",
        ]
    )
    assert request.arguments["asset_kind"] == expected


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("face-rights", "face_rights"),
        ("external-egress", "external_egress"),
        ("paid-compute", "paid_compute"),
    ],
)
def test_gate_spellings_normalize_to_the_json_form(spelling: str, expected: str) -> None:
    request = parse(["approval", "issue", "plan.json", "--gate", spelling])
    assert request.arguments["gate"] == expected


def test_the_underscore_spelling_is_refused_on_the_command_line() -> None:
    with pytest.raises(ValueError):
        parse(["approval", "issue", "plan.json", "--gate", "paid_compute"])


# --- Value parsing --------------------------------------------------------------------


def test_a_resolution_parses_without_rounding() -> None:
    assert parse_resolution("1024x1024") == (1024, 1024)
    assert parse_resolution("1536X1024") == (1536, 1024)


@pytest.mark.parametrize("value", ["1024", "1024x", "x1024", "0x1024", "-1x8", "axb", "1024*1024"])
def test_a_malformed_resolution_is_a_usage_error(value: str) -> None:
    with pytest.raises(UsageError):
        parse_resolution(value)


def test_expiry_durations_parse() -> None:
    assert parse_expires_in("30m") == 1800
    assert parse_expires_in("45s") == 45
    assert parse_expires_in("2h") == 7200


@pytest.mark.parametrize("value", ["30", "m", "0m", "-5m", "25h", "1d", "abc"])
def test_a_malformed_expiry_is_a_usage_error(value: str) -> None:
    with pytest.raises(UsageError):
        parse_expires_in(value)


def test_a_frame_must_be_a_nonnegative_integer() -> None:
    assert parse_frame("12") == 12
    for value in ("-1", "1.5", "two"):
        with pytest.raises(UsageError):
            parse_frame(value)


def test_formats_are_deduplicated_and_sorted() -> None:
    request = parse(
        ["export", "bake.json", "--format", "glb", "--format", "blend", "--format", "glb"]
    )
    assert request.arguments["formats"] == ["blend", "glb"]


def test_multiple_approvals_are_collected(tmp_path: Path) -> None:
    request = parse(
        [
            "view",
            "generate",
            "plan.json",
            "--prompt-file",
            "p.txt",
            "--approval",
            "a.json",
            "--approval",
            "b.json",
        ]
    )
    assert [path.name for path in request.arguments["approval"]] == ["a.json", "b.json"]


# --- Acknowledgements ------------------------------------------------------------------


def test_the_exact_plan_bound_acknowledgement_is_required() -> None:
    assert (
        parse_acknowledgement(f"paid_compute:{PLAN}", gate="paid_compute", plan_sha256=PLAN)
        == f"paid_compute:{PLAN}"
    )


@pytest.mark.parametrize(
    "value", [None, "yes", "true", "paid_compute", PLAN, "external_egress:" + PLAN]
)
def test_a_wrong_acknowledgement_is_a_usage_error(value) -> None:
    with pytest.raises(UsageError):
        parse_acknowledgement(value, gate="paid_compute", plan_sha256=PLAN)


def test_the_exact_alignment_acknowledgement_is_required() -> None:
    assert parse_alignment_acknowledgement(
        f"{CONDITION}:{VIEW}", condition_sha256=CONDITION, view_sha256=VIEW
    )


@pytest.mark.parametrize("value", [None, "yes", CONDITION, f"{VIEW}:{CONDITION}"])
def test_a_wrong_alignment_acknowledgement_is_a_usage_error(value) -> None:
    with pytest.raises(UsageError):
        parse_alignment_acknowledgement(value, condition_sha256=CONDITION, view_sha256=VIEW)


# --- Exit codes ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("succeeded", 0),
        ("usage", 2),
        ("failed", 3),
        ("unsupported", 3),
        ("contract", 3),
        ("internal", 4),
        ("needs_approval", 5),
        ("canceled", 6),
        ("storage", 73),
    ],
)
def test_the_exit_codes_are_fixed(status: str, code: int) -> None:
    assert exit_code(status) == code


def test_an_unknown_status_has_no_exit_code() -> None:
    with pytest.raises(ValueError):
        exit_code("partly")


# --- v0.3 generic image-to-3D commands -----------------------------------------------


def test_the_engine_clearance_command_parses() -> None:
    request = parse(["engine", "clearance", "verify", "clearance.json"])
    assert request.command == "engine clearance verify"
    assert request.arguments["engine_clearance"].name == "clearance.json"


def test_the_reconstruct_command_parses() -> None:
    request = parse(
        [
            "image",
            "reconstruct",
            "subject.png",
            "--engine",
            "triposr-local",
            "--clearance",
            "clearance.json",
            "--asset-kind",
            "object",
            "--subject",
            "non-person",
            "--mask",
            "subject-mask.png",
        ]
    )
    assert request.command == "image reconstruct"
    assert request.arguments["asset_kind"] == "object"
    assert request.arguments["subject"] == "non_person"
    assert request.arguments["mask"].name == "subject-mask.png"


def test_reconstruction_requires_a_clearance_argument() -> None:
    """There is no way to ask for a reconstruction without naming a clearance."""
    with pytest.raises(UsageError):
        parse(
            [
                "image",
                "reconstruct",
                "subject.png",
                "--engine",
                "triposr-local",
                "--asset-kind",
                "object",
                "--subject",
                "non-person",
            ]
        )


def test_reconstruction_requires_a_subject_declaration() -> None:
    with pytest.raises(UsageError):
        parse(
            [
                "image",
                "reconstruct",
                "subject.png",
                "--engine",
                "triposr-local",
                "--clearance",
                "clearance.json",
                "--asset-kind",
                "object",
            ]
        )


def test_reconstruction_normalizes_the_face_head_spelling() -> None:
    request = parse(
        [
            "image",
            "reconstruct",
            "subject.png",
            "--engine",
            "triposr-local",
            "--clearance",
            "clearance.json",
            "--asset-kind",
            "face-head",
            "--subject",
            "real-person",
            "--rights-receipt",
            "receipt.json",
        ]
    )
    assert request.arguments["asset_kind"] == "face_head"
    assert request.arguments["subject"] == "real_person"


def test_reconstruction_accepts_a_background_remover_instead_of_a_mask() -> None:
    request = parse(
        [
            "image",
            "reconstruct",
            "subject.png",
            "--engine",
            "triposr-local",
            "--clearance",
            "clearance.json",
            "--asset-kind",
            "object",
            "--subject",
            "non-person",
            "--background-removal",
            "remover-clearance.json",
        ]
    )
    assert request.arguments["mask"] is None
    assert request.arguments["background_removal"].name == "remover-clearance.json"


def test_there_is_no_flag_to_skip_clearance() -> None:
    for flag in ("--skip-clearance", "--no-clearance", "--force", "--yes"):
        with pytest.raises(UsageError):
            parse(
                [
                    "image",
                    "reconstruct",
                    "subject.png",
                    "--engine",
                    "triposr-local",
                    "--clearance",
                    "clearance.json",
                    "--asset-kind",
                    "object",
                    "--subject",
                    "non-person",
                    flag,
                ]
            )
