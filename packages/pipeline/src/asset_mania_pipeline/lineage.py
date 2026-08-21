"""Parent and consumed-artifact verification.

Every mismatch between an expected digest and the bytes on disk fails here, before
Blender or a provider is invoked.
"""

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from asset_mania_contracts import GATES, PARENT_RELATIONSHIPS, canonical_digest

from .artifacts import contained_path
from .hashing import sha256_file

_MISMATCH = "PARENT_MANIFEST_MISMATCH"


class ParentMismatch(Exception):
    """A parent manifest is absent, unreadable, or not the expected digest."""


class ArtifactMismatch(Exception):
    """A consumed artifact is not the bytes the parent manifest recorded."""


@dataclass(frozen=True, slots=True)
class ParentManifest:
    run_id: str
    manifest_sha256: str
    relationship: str
    document: dict[str, Any]
    directory: Path


def load_parent(
    manifest_path: Path,
    *,
    expected_sha256: str,
    relationship: str,
) -> ParentManifest:
    """Load a parent manifest and bind it to an exact digest and relationship."""
    if relationship not in PARENT_RELATIONSHIPS:
        raise ValueError(f"relationship {relationship!r} is not a declared relationship")

    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ParentMismatch(f"{_MISMATCH}: the parent manifest is unreadable") from error

    if not isinstance(document, dict):
        raise ParentMismatch(f"{_MISMATCH}: the parent manifest is not an object")

    observed = canonical_digest(document)
    if observed != expected_sha256:
        raise ParentMismatch(f"{_MISMATCH}: the parent manifest digest does not match")

    return ParentManifest(
        run_id=document["run_id"],
        manifest_sha256=observed,
        relationship=relationship,
        document=document,
        directory=manifest_path.parent,
    )


def parent_reference(parent: ParentManifest) -> dict[str, str]:
    """The portable three-field parent reference a child manifest records."""
    return {
        "run_id": parent.run_id,
        "manifest_sha256": parent.manifest_sha256,
        "relationship": parent.relationship,
    }


def verify_consumed_artifact(run_directory: Path, artifact: Mapping[str, Any]) -> None:
    """Rehash one artifact inside its own run and require the recorded identity."""
    path = contained_path(run_directory, artifact["path"])
    try:
        observed = sha256_file(path)
        byte_size = path.stat().st_size
    except OSError as error:
        raise ArtifactMismatch(f"{_MISMATCH}: a consumed artifact is unreadable") from error

    if observed != artifact["sha256"] or byte_size != artifact["byte_size"]:
        raise ArtifactMismatch(f"{_MISMATCH}: a consumed artifact changed on disk")


def inherit_rights_basis(manifests: Iterable[Mapping[str, Any]]) -> list[str]:
    """The gates already consumed upstream, in gate order.

    Downstream stages inherit this immutable approval lineage instead of re-consuming a
    single-use receipt.
    """
    gates: set[str] = set()
    for manifest in manifests:
        for consumption in manifest.get("approvals", ()):
            gates.add(consumption["gate"])
    return [gate for gate in GATES if gate in gates]


def consumed_artifact_parents(
    artifacts: Sequence[Mapping[str, Any]],
    *,
    relationship: str = "consumed",
) -> list[dict[str, str]]:
    """Artifact-level parent references for a set of consumed artifacts."""
    return sorted(
        ({"sha256": artifact["sha256"], "relationship": relationship} for artifact in artifacts),
        key=lambda parent: parent["sha256"],
    )
