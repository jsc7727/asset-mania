"""Fast container validation for exported assets.

These are cheap structural checks the Apache side can run without Blender and without a
full parser: enough to reject a truncated, mislabelled, or internally inconsistent file
before anything downstream trusts it. They are deliberately *not* a claim of semantic
correctness -- a fresh-process reimport does that, and for GLB the Khronos validator does
the resource-level checks.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BLEND_MAGIC = b"BLENDER"
BLEND_COMPRESSED_MAGICS = (b"\x1f\x8b", b"\x28\xb5\x2f\xfd")
GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_JSON_CHUNK = b"JSON"
GLB_BIN_CHUNK = b"BIN\x00"
#: glTF 2.0's documented default for `alphaCutoff`.
GLTF_DEFAULT_ALPHA_CUTOFF = 0.5
FBX_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"
FBX_MINIMUM_VERSION = 7100
FBX_MAXIMUM_VERSION = 8000

_MALFORMED_BLEND = "BLEND_HEADER_INVALID"
_GLTF_INVALID = "GLTF_VALIDATION_FAILED"
_EXPORT_UNAVAILABLE = "EXPORT_OPERATOR_UNAVAILABLE"
_ROUNDTRIP = "ROUNDTRIP_MISMATCH"


class ContainerInvalid(Exception):
    """An exported container is not a well-formed file of its declared kind."""

    def __init__(self, diagnostic: str, detail: str) -> None:
        super().__init__(f"{diagnostic}: {detail}")
        self.diagnostic = diagnostic


def _require(condition: bool, diagnostic: str, detail: str) -> None:
    if not condition:
        raise ContainerInvalid(diagnostic, detail)


@dataclass(frozen=True, slots=True)
class BlendHeader:
    version: str
    endianness: str


def validate_blend(path: Path) -> BlendHeader:
    """Check a `.blend` header. A compressed file is rejected in this profile.

    Two header layouts exist and both are accepted. The classic one is
    `BLENDER` + pointer-size flag + endianness + a three-digit version; Blender 5.2 writes
    `BLENDER17-01v0502`, with extra fields between the magic and the endianness marker.

    Only what can actually be verified is asserted: the magic, that the file is not
    compressed, the endianness marker, and a numeric version. The meaning of the extra
    5.x fields has not been confirmed, so nothing is claimed about them -- a fresh-process
    reopen is what establishes the file is usable.
    """
    _require(path.is_file(), _MALFORMED_BLEND, "the blend file does not exist")
    header = path.read_bytes()[:24]
    _require(len(header) >= 12, _MALFORMED_BLEND, "the blend header is truncated")

    for magic in BLEND_COMPRESSED_MAGICS:
        _require(
            not header.startswith(magic),
            _MALFORMED_BLEND,
            "this profile writes uncompressed blend files",
        )
    _require(header.startswith(BLEND_MAGIC), _MALFORMED_BLEND, "the blend magic is missing")

    text = header.decode("ascii", errors="replace")
    marker = -1
    for index in range(len(BLEND_MAGIC), len(text)):
        if text[index] in ("v", "V"):
            marker = index
            break
    _require(marker != -1, _MALFORMED_BLEND, "the blend endianness marker is missing")

    digits = ""
    for character in text[marker + 1 :]:
        if not character.isdigit():
            break
        digits += character
    _require(len(digits) >= 3, _MALFORMED_BLEND, "the blend version is not numeric")

    return BlendHeader(
        version=digits,
        endianness="little" if text[marker] == "v" else "big",
    )


@dataclass(frozen=True, slots=True)
class GlbContainer:
    total_length: int
    json_chunk: dict[str, Any]
    binary_length: int


def validate_glb(path: Path) -> GlbContainer:
    """Check GLB magic, version, declared length, and chunk layout, then parse its JSON."""
    _require(path.is_file(), _GLTF_INVALID, "the GLB file does not exist")
    data = path.read_bytes()
    _require(len(data) >= 12, _GLTF_INVALID, "the GLB header is truncated")

    magic, version, declared = struct.unpack_from("<4sII", data, 0)
    _require(magic == GLB_MAGIC, _GLTF_INVALID, "the GLB magic is missing")
    _require(version == GLB_VERSION, _GLTF_INVALID, f"GLB version {version} is not 2")
    _require(
        declared == len(data),
        _GLTF_INVALID,
        f"the GLB declares {declared} bytes but is {len(data)}",
    )

    offset = 12
    json_chunk: dict[str, Any] | None = None
    binary_length = 0
    while offset < len(data):
        _require(offset + 8 <= len(data), _GLTF_INVALID, "a GLB chunk header is truncated")
        length, kind = struct.unpack_from("<I4s", data, offset)
        offset += 8
        _require(offset + length <= len(data), _GLTF_INVALID, "a GLB chunk overruns the file")
        _require(length % 4 == 0, _GLTF_INVALID, "a GLB chunk is not four-byte aligned")
        payload = data[offset : offset + length]
        offset += length

        if kind == GLB_JSON_CHUNK:
            _require(json_chunk is None, _GLTF_INVALID, "the GLB has more than one JSON chunk")
            try:
                parsed = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ContainerInvalid(_GLTF_INVALID, "the GLB JSON chunk is unreadable") from error
            _require(isinstance(parsed, dict), _GLTF_INVALID, "the GLB JSON chunk is not an object")
            json_chunk = parsed
        elif kind == GLB_BIN_CHUNK:
            binary_length += length

    _require(json_chunk is not None, _GLTF_INVALID, "the GLB has no JSON chunk")
    _require(
        json_chunk.get("asset", {}).get("version") == "2.0",
        _GLTF_INVALID,
        "the GLB does not declare glTF 2.0",
    )
    return GlbContainer(total_length=declared, json_chunk=json_chunk, binary_length=binary_length)


def validate_glb_alpha_profile(container: GlbContainer, *, expected_cutoff: float = 0.5) -> None:
    """Unknown texels must be exported as `alphaMode: MASK` with the bound cutoff.

    This is the property that makes an uncovered texel read as absent rather than black in
    a runtime viewer, so it is checked on the exported JSON rather than assumed from the
    exporter's settings.
    """
    materials = container.json_chunk.get("materials") or []
    _require(bool(materials), _GLTF_INVALID, "the GLB exports no material")
    for index, material in enumerate(materials):
        _require(
            material.get("alphaMode") == "MASK",
            _GLTF_INVALID,
            f"material {index} declares alphaMode {material.get('alphaMode')!r}, not MASK",
        )
        # glTF 2.0 defines `alphaCutoff` with a default of 0.5, and Blender's exporter
        # omits the key when the value is exactly that default. Absence therefore *is*
        # 0.5; demanding the key would reject a conformant file.
        cutoff = material.get("alphaCutoff", GLTF_DEFAULT_ALPHA_CUTOFF)
        _require(
            cutoff == expected_cutoff,
            _GLTF_INVALID,
            f"material {index} declares alphaCutoff {cutoff!r}, not {expected_cutoff}",
        )


def validate_glb_has_no_absolute_resource(container: GlbContainer) -> None:
    """No exported resource may point outside the container."""
    for image in container.json_chunk.get("images") or []:
        uri = image.get("uri")
        if uri is None:
            continue
        _require(
            not uri.startswith("/") and "://" not in uri and not uri.startswith("\\\\"),
            _GLTF_INVALID,
            f"an image resource points outside the container: {uri!r}",
        )
    for buffer in container.json_chunk.get("buffers") or []:
        uri = buffer.get("uri")
        if uri is None:
            continue
        _require(
            uri.startswith("data:") or ("://" not in uri and not uri.startswith("/")),
            _GLTF_INVALID,
            f"a buffer points outside the container: {uri!r}",
        )


@dataclass(frozen=True, slots=True)
class FbxHeader:
    version: int


def validate_fbx(path: Path) -> FbxHeader:
    """Check the binary FBX magic and a supported version. ASCII FBX is rejected."""
    _require(path.is_file(), _EXPORT_UNAVAILABLE, "the FBX file does not exist")
    header = path.read_bytes()[:27]
    _require(len(header) >= 27, _EXPORT_UNAVAILABLE, "the FBX header is truncated")
    _require(
        header.startswith(FBX_BINARY_MAGIC),
        _EXPORT_UNAVAILABLE,
        "this profile writes binary FBX only",
    )
    (version,) = struct.unpack_from("<I", header, 23)
    _require(
        FBX_MINIMUM_VERSION <= version <= FBX_MAXIMUM_VERSION,
        _EXPORT_UNAVAILABLE,
        f"FBX version {version} is outside the declared subset",
    )
    return FbxHeader(version=version)


def compare_semantic_fingerprints(
    exported: dict[str, Any], reimported: dict[str, Any], *, tolerance: float = 1e-4
) -> None:
    """Compare a format-aware fingerprint, never raw archive bytes.

    Byte-identical GLB or FBX across runs is explicitly not claimed. What must hold is that
    a fresh process reading the file back sees the same counts and the same sampled bone
    matrices and deformed vertices, within a numeric tolerance.
    """
    for key in sorted(set(exported) | set(reimported)):
        left = exported.get(key)
        right = reimported.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if abs(float(left) - float(right)) > tolerance:
                raise ContainerInvalid(
                    _ROUNDTRIP, f"{key} changed from {left} to {right} across the round trip"
                )
            continue
        if isinstance(left, list) and isinstance(right, list):
            _require(
                len(left) == len(right),
                _ROUNDTRIP,
                f"{key} has {len(left)} entries before and {len(right)} after",
            )
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    if abs(float(a) - float(b)) > tolerance:
                        raise ContainerInvalid(
                            _ROUNDTRIP, f"{key}[{index}] changed from {a} to {b}"
                        )
                elif a != b:
                    raise ContainerInvalid(
                        _ROUNDTRIP, f"{key}[{index}] changed from {a!r} to {b!r}"
                    )
            continue
        if left != right:
            raise ContainerInvalid(_ROUNDTRIP, f"{key} changed from {left!r} to {right!r}")
