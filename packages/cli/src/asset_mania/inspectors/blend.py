"""Syntactic inspection of portable Blender file headers."""

from pathlib import Path

from asset_mania_contracts import DiagnosticCode

_HEADER_SIZE = 12


def inspect_blend(path: Path) -> tuple[dict[str, object], list[DiagnosticCode]]:
    """Parse a Blender header without opening the file in Blender."""
    try:
        with path.open("rb") as source:
            header = source.read(_HEADER_SIZE)
    except OSError:
        return {}, [DiagnosticCode.INPUT_UNREADABLE]

    parsed = _parse_header(header)
    if parsed is None:
        return {"header_valid": False}, [DiagnosticCode.BLEND_HEADER_INVALID]
    return {**parsed, "header_valid": True}, []


def _parse_header(header: bytes) -> dict[str, object] | None:
    if len(header) != _HEADER_SIZE or header[:7] != b"BLENDER":
        return None

    pointer_marker, endian_marker, version_bytes = header[7], header[8], header[9:]
    pointer_size = {ord("-"): 64, ord("_"): 32}.get(pointer_marker)
    endianness = {ord("v"): "little", ord("V"): "big"}.get(endian_marker)
    if pointer_size is None or endianness is None or not version_bytes.isdigit():
        return None

    version = int(version_bytes)
    if not 100 <= version <= 999:
        return None
    return {"version": version, "pointer_size": pointer_size, "endianness": endianness}
