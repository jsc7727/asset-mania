"""Immutable plan hashing and salted selection resolution."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from asset_mania_contracts import canonical_digest, selection_digest

_TAMPERED = "PLAN_TAMPERED"
_LABEL_FIELDS = ("camera_label", "target_label", "armature_label", "action_label")


class PlanTampered(Exception):
    """A plan, or the selection it names, is not the object that was approved."""


def plan_digest(plan: Mapping[str, Any], *, digest_field: str = "plan_sha256") -> str:
    """The canonical digest of a plan over every field except the digest itself."""
    preimage = {key: value for key, value in plan.items() if key != digest_field}
    return canonical_digest(preimage)


def verify_plan(plan: Mapping[str, Any], *, digest_field: str = "plan_sha256") -> None:
    if plan.get(digest_field) != plan_digest(plan, digest_field=digest_field):
        raise PlanTampered(f"{_TAMPERED}: the plan digest does not match its content")


def load_plan(
    path: Path,
    *,
    expected_sha256: str | None = None,
    digest_field: str = "plan_sha256",
) -> dict[str, Any]:
    """Load a plan file, verify its self-digest, and optionally bind it to one digest."""
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlanTampered(f"{_TAMPERED}: the plan is unreadable") from error

    if not isinstance(plan, dict):
        raise PlanTampered(f"{_TAMPERED}: the plan is not an object")

    verify_plan(plan, digest_field=digest_field)
    if expected_sha256 is not None and plan[digest_field] != expected_sha256:
        raise PlanTampered(f"{_TAMPERED}: the plan is not the expected plan")
    return plan


def _selection_preimage(identity: Mapping[str, Any], labels: Mapping[str, Any]) -> dict[str, Any]:
    """The HMAC preimage binds the private identity to the portable labels together.

    Binding both means a relabelled selection over the same datablocks, and a relabelled
    datablock behind the same label, both fail verification.
    """
    return {
        "identity": dict(identity),
        "labels": {field: labels[field] for field in _LABEL_FIELDS},
    }


def build_selection_map(
    *,
    salt: bytes,
    identity: Mapping[str, Any],
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the local-sensitive selection map and its portable label projection."""
    missing = [field for field in _LABEL_FIELDS if field not in labels]
    if missing:
        raise ValueError(f"selection labels are incomplete: {missing}")

    digest = selection_digest(salt=salt, identity=_selection_preimage(identity, labels))
    return {
        "schema_id": "asset-mania/selection-map",
        "schema_version": "1.0",
        "sensitivity": "local-sensitive",
        "upload_eligible": False,
        "salt": salt.hex(),
        "identity": dict(identity),
        "portable_selection": {
            **{field: labels[field] for field in _LABEL_FIELDS},
            "selection_digest": digest,
        },
    }


def verify_selection(
    *,
    selection_map: Mapping[str, Any],
    portable_selection: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> None:
    """Recompute the selection digest after opening the exact source hash."""
    try:
        salt = bytes.fromhex(selection_map["salt"])
    except (KeyError, ValueError) as error:
        raise PlanTampered(f"{_TAMPERED}: the selection map carries no usable salt") from error

    expected = selection_digest(
        salt=salt, identity=_selection_preimage(identity, portable_selection)
    )
    if portable_selection.get("selection_digest") != expected:
        raise PlanTampered(f"{_TAMPERED}: the selection does not match the approved plan")
