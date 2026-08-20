"""Argument parsing for the v0.2 stage commands.

Parsing is deliberately thin. Every command resolves its arguments, normalizes the kebab
spellings once at this boundary, and hands a plain request to a service that is testable
without a terminal. Nothing here decides policy, opens a source, or launches a worker.

Two rules shape the surface:

* there is no global `--yes`; an approval needs the exact plan-bound acknowledgement, so a
  boolean flag can never stand in for one; and
* CLI values use kebab case and are normalized here exactly once, because portable JSON
  accepts only the underscore forms.
"""

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asset_mania_contracts import (
    EXIT_CANCELED,
    EXIT_CONTRACT,
    EXIT_INTERNAL,
    EXIT_NEEDS_APPROVAL,
    EXIT_STORAGE,
    EXIT_SUCCESS,
    EXIT_USAGE,
    exit_code_for,
)
from asset_mania_pipeline import parse_asset_kind, parse_gate, parse_subject

STAGE_COMMANDS = (
    "scene preflight",
    "scene plan",
    "scene condition",
    "view ingest",
    "view provider-plan",
    "view generate",
    "texture bake",
    "export",
    "approval issue",
    "provider evidence refresh",
)
DEFAULT_EXPIRES_IN = "30m"
DEFAULT_RESOLUTION = "1024x1024"


class UsageError(Exception):
    """The command line itself was wrong, before any run was created."""


@dataclass(frozen=True, slots=True)
class StageRequest:
    """One parsed stage invocation, ready for a service to execute."""

    command: str
    arguments: dict[str, Any]


def parse_resolution(value: str) -> tuple[int, int]:
    """`WIDTHxHEIGHT`, with no silent rounding or aspect correction."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise UsageError("resolution must be given as WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as error:
        raise UsageError("resolution must be two integers") from error
    if width <= 0 or height <= 0:
        raise UsageError("resolution must be positive")
    return width, height


def parse_expires_in(value: str) -> int:
    """A short duration such as `30m` or `2h`, in seconds."""
    text = value.strip().lower()
    units = {"s": 1, "m": 60, "h": 3600}
    if len(text) < 2 or text[-1] not in units:
        raise UsageError("expires-in must look like 30m, 45s, or 2h")
    try:
        amount = int(text[:-1])
    except ValueError as error:
        raise UsageError("expires-in must start with an integer") from error
    if amount <= 0:
        raise UsageError("expires-in must be positive")
    seconds = amount * units[text[-1]]
    if seconds > 24 * 3600:
        raise UsageError("expires-in must not exceed 24 hours")
    return seconds


def parse_frame(value: str) -> int:
    try:
        frame = int(value)
    except ValueError as error:
        raise UsageError("frame must be an integer") from error
    if frame < 0:
        raise UsageError("frame must not be negative")
    return frame


def parse_acknowledgement(value: str | None, *, gate: str, plan_sha256: str) -> str:
    """The exact `GATE:PLAN_SHA256` string. A boolean flag cannot issue a receipt."""
    expected = f"{gate}:{plan_sha256}"
    if value is None:
        raise UsageError(
            "non-interactive use must pass the exact --acknowledgement GATE:PLAN_SHA256"
        )
    if value != expected:
        raise UsageError("the acknowledgement does not match this gate and plan")
    return value


def parse_alignment_acknowledgement(
    value: str | None, *, condition_sha256: str, view_sha256: str
) -> str:
    expected = f"{condition_sha256}:{view_sha256}"
    if value is None:
        raise UsageError(
            "non-interactive use must pass the exact "
            "--alignment-acknowledgement CONDITION_SHA256:VIEW_SHA256"
        )
    if value != expected:
        raise UsageError("the alignment acknowledgement does not match this condition and view")
    return value


def build_parser() -> argparse.ArgumentParser:
    """The full v0.2 command surface. `inspect` keeps its own v0.1 parser."""
    parser = argparse.ArgumentParser(prog="asset-mania", add_help=True)
    commands = parser.add_subparsers(dest="group", required=True)

    scene = commands.add_parser("scene").add_subparsers(dest="stage", required=True)

    preflight = scene.add_parser("preflight")
    preflight.add_argument("source", type=Path)
    preflight.add_argument("--out", type=Path)

    plan = scene.add_parser("plan")
    plan.add_argument("preflight_manifest", type=Path)
    plan.add_argument("--camera", required=True)
    plan.add_argument("--frame", required=True)
    plan.add_argument("--target", required=True)
    plan.add_argument("--asset-kind", required=True)
    plan.add_argument("--subject", required=True)
    plan.add_argument("--armature")
    plan.add_argument("--action")
    plan.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    plan.add_argument("--out", type=Path)

    condition = scene.add_parser("condition")
    condition.add_argument("source", type=Path)
    condition.add_argument("--plan", required=True, type=Path)
    condition.add_argument("--rights-receipt", type=Path)
    condition.add_argument("--blender", type=Path)
    condition.add_argument("--out", type=Path)

    view = commands.add_parser("view").add_subparsers(dest="stage", required=True)

    ingest = view.add_parser("ingest")
    ingest.add_argument("image", type=Path)
    ingest.add_argument("--condition-manifest", required=True, type=Path)
    ingest.add_argument("--origin", required=True)
    ingest.add_argument("--alignment-acknowledgement")
    ingest.add_argument("--out", type=Path)

    provider_plan = view.add_parser("provider-plan")
    provider_plan.add_argument("condition_manifest", type=Path)
    provider_plan.add_argument("--provider", required=True)
    provider_plan.add_argument("--model", required=True)
    provider_plan.add_argument("--prompt-file", required=True, type=Path)
    provider_plan.add_argument("--evidence", required=True, type=Path)
    provider_plan.add_argument("--size", default=DEFAULT_RESOLUTION)
    provider_plan.add_argument("--quality", default="medium")
    provider_plan.add_argument("--background", default="auto")
    provider_plan.add_argument("--output-format", default="png")
    provider_plan.add_argument("--output-compression", type=int)
    provider_plan.add_argument("--moderation", default="auto")
    provider_plan.add_argument("--out", type=Path)

    generate = view.add_parser("generate")
    generate.add_argument("provider_plan", type=Path)
    generate.add_argument("--prompt-file", required=True, type=Path)
    generate.add_argument("--approval", action="append", default=[], type=Path)
    generate.add_argument("--out", type=Path)

    texture = commands.add_parser("texture").add_subparsers(dest="stage", required=True)
    bake = texture.add_parser("bake")
    bake.add_argument("--condition-manifest", required=True, type=Path)
    bake.add_argument("--view-manifest", required=True, type=Path)
    bake.add_argument("--blender", type=Path)
    bake.add_argument("--out", type=Path)

    export = commands.add_parser("export")
    export.add_argument("bake_manifest", type=Path)
    export.add_argument("--format", action="append", default=[], dest="formats")
    export.add_argument("--blender", type=Path)
    export.add_argument("--out", type=Path)

    approval = commands.add_parser("approval").add_subparsers(dest="stage", required=True)
    issue = approval.add_parser("issue")
    issue.add_argument("plan", type=Path)
    issue.add_argument("--gate", required=True)
    issue.add_argument("--expires-in", default=DEFAULT_EXPIRES_IN)
    issue.add_argument("--acknowledgement")

    provider = commands.add_parser("provider").add_subparsers(dest="stage", required=True)
    evidence = provider.add_parser("evidence").add_subparsers(dest="action", required=True)
    refresh = evidence.add_parser("refresh")
    refresh.add_argument("provider_name")
    refresh.add_argument("--out", required=True, type=Path)

    return parser


def normalize(namespace: argparse.Namespace) -> StageRequest:
    """Normalize kebab spellings once, at the parser boundary."""
    group = namespace.group
    stage = getattr(namespace, "stage", None)
    command = f"{group} {stage}".strip() if stage else group
    arguments = {
        key: value
        for key, value in vars(namespace).items()
        if key not in ("group", "stage", "action")
    }

    if "asset_kind" in arguments and arguments["asset_kind"] is not None:
        arguments["asset_kind"] = parse_asset_kind(arguments["asset_kind"])
    if "subject" in arguments and arguments["subject"] is not None:
        arguments["subject"] = parse_subject(arguments["subject"])
    if "gate" in arguments and arguments["gate"] is not None:
        arguments["gate"] = parse_gate(arguments["gate"])
    if "resolution" in arguments and arguments["resolution"] is not None:
        arguments["resolution"] = list(parse_resolution(arguments["resolution"]))
    if "frame" in arguments and arguments["frame"] is not None:
        arguments["frame"] = parse_frame(arguments["frame"])
    if "expires_in" in arguments and arguments["expires_in"] is not None:
        arguments["expires_in_seconds"] = parse_expires_in(arguments.pop("expires_in"))
    if arguments.get("formats"):
        arguments["formats"] = sorted(set(arguments["formats"]))

    if command == "provider evidence":
        command = "provider evidence refresh"

    return StageRequest(command=command, arguments=arguments)


def parse(argv: Sequence[str]) -> StageRequest:
    parser = build_parser()
    try:
        namespace = parser.parse_args(list(argv))
    except SystemExit as error:
        raise UsageError("invalid command usage") from error
    return normalize(namespace)


EXIT_CODES = {
    "succeeded": EXIT_SUCCESS,
    "usage": EXIT_USAGE,
    "contract": EXIT_CONTRACT,
    "internal": EXIT_INTERNAL,
    "needs_approval": EXIT_NEEDS_APPROVAL,
    "canceled": EXIT_CANCELED,
    "storage": EXIT_STORAGE,
}


def exit_code(status: str) -> int:
    """Map a terminal status onto the fixed v2 exit code."""
    if status in ("succeeded", "failed", "unsupported", "needs_approval", "canceled"):
        return exit_code_for(status)
    if status in EXIT_CODES:
        return EXIT_CODES[status]
    raise ValueError(f"{status!r} has no exit code")
