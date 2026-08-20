# SPDX-License-Identifier: GPL-3.0-or-later
"""Salted selection verification, restated inside the GPL worker.

The Apache side writes a local-sensitive selection map holding a per-plan salt and the
private datablock identity; the portable plan carries only labels and the resulting HMAC.
After opening the exact source hash the worker recomputes that HMAC here and fails with
`PLAN_TAMPERED` on any change to a label, a name, a library identity, the source hash, or
the target type.

The preimage layout is duplicated from `asset_mania_pipeline.plans` rather than imported,
because importing an Apache package across the license boundary is exactly what the split
forbids. `tests/test_worker_protocol_parity.py` keeps the two definitions in step.
"""

import hashlib
import hmac
import json

LABEL_FIELDS = ("camera_label", "target_label", "armature_label", "action_label")
IDENTITY_FIELDS = (
    "source_scene_sha256",
    "camera",
    "target",
    "target_type",
    "armature",
    "action",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def selection_preimage(identity: dict, labels: dict) -> dict:
    """The HMAC preimage binds the private identity to the portable labels together."""
    return {
        "identity": dict(identity),
        "labels": {field: labels[field] for field in LABEL_FIELDS},
    }


def selection_digest(*, salt: bytes, identity: dict, labels: dict) -> str:
    if len(salt) < 16:
        raise ValueError("selection salt must be at least 16 bytes")
    preimage = canonical_json(selection_preimage(identity, labels)).encode("utf-8")
    return hmac.new(salt, preimage, hashlib.sha256).hexdigest()


def verify_selection(*, salt_hex: str, identity: dict, portable_selection: dict) -> bool:
    """Recompute the digest and compare it in constant time."""
    try:
        salt = bytes.fromhex(salt_hex)
    except ValueError:
        return False
    expected = selection_digest(salt=salt, identity=identity, labels=portable_selection)
    return hmac.compare_digest(expected, str(portable_selection.get("selection_digest", "")))
