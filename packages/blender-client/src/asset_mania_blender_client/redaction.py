"""Redaction for opt-in local debug logs.

Raw Blender stdout and stderr are never forwarded to the user's streams or persisted in a
run, because Blender may print absolute paths and datablock names. When a maintainer
explicitly enables a local debug log, this module is the only sanctioned way to write any
of that text, and the result is never upload-eligible.
"""

import re
from collections.abc import Iterable
from pathlib import Path

MAX_REDACTED_BYTES = 8192
_PLACEHOLDER = "[redacted]"
_ABSOLUTE_PATH = re.compile(r"(?:/[A-Za-z0-9._+\-]+){2,}/?")


def redact(
    text: str,
    *,
    private_paths: Iterable[Path] = (),
    private_names: Iterable[str] = (),
    limit: int = MAX_REDACTED_BYTES,
) -> str:
    """Replace known private paths and names, then every remaining absolute path."""
    redacted = text
    for path in sorted((str(candidate) for candidate in private_paths), key=len, reverse=True):
        redacted = redacted.replace(path, _PLACEHOLDER)
    for name in sorted((str(candidate) for candidate in private_names), key=len, reverse=True):
        if name:
            redacted = redacted.replace(name, _PLACEHOLDER)
    redacted = _ABSOLUTE_PATH.sub(_PLACEHOLDER, redacted)

    encoded = redacted.encode("utf-8", errors="replace")
    if len(encoded) > limit:
        encoded = encoded[:limit]
        return encoded.decode("utf-8", errors="ignore") + "\n[truncated]"
    return encoded.decode("utf-8", errors="ignore")
