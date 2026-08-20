# SPDX-License-Identifier: GPL-3.0-or-later
"""The closed request/response protocol, duplicated deliberately.

This module restates the field names the Apache client expects instead of importing
`asset_mania_contracts`. That duplication is the license boundary: the GPL worker must not
link against an Apache package, and the Apache side must not link against GPL code. The
`blender-response-v1` schema in the contracts package is the authority, and
`tests/test_license_boundary.py` plus the client's response validator keep the two in
step.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_ID = "asset-mania/blender-response"
SCHEMA_VERSION = "1.0"
OPERATIONS = ("preflight", "condition", "bake", "export", "validate")
RESPONSE_MAX_BYTES = 1048576


def canonical_json(value: object) -> str:
    """The same canonical encoding the Apache side hashes, restated locally."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"


def canonical_digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seal_response(response: dict) -> dict:
    """Attach `response_sha256` over every other field of the response."""
    preimage = {key: item for key, item in response.items() if key != "response_sha256"}
    return {**preimage, "response_sha256": canonical_digest(preimage)}


def read_request(path: Path) -> dict:
    """Read the private request envelope the client wrote."""
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("the request envelope is not an object")
    return request


def write_response(path: Path, response: dict) -> None:
    """Write the sealed response atomically at mode 0600, refusing an oversized payload."""
    payload = canonical_json(seal_response(response)).encode("utf-8")
    if len(payload) > RESPONSE_MAX_BYTES:
        raise ValueError("the worker response exceeds the closed size limit")

    directory = path.parent
    descriptor, temporary = tempfile.mkstemp(dir=directory, prefix=".response.")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def failure(*, request_id: str, operation: str, diagnostics: list[str]) -> dict:
    """A closed failure response: stable codes only, no traceback and no raw log."""
    return {
        "schema_id": SCHEMA_ID,
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "operation": operation,
        "status": "failed",
        "diagnostics": sorted(set(diagnostics)),
        "portable_labels": [],
        "outputs": [],
        "metrics": None,
        "response_sha256": "",
    }
