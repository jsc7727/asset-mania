"""Local environment inspection without executing discovered programs."""

import os
import platform
import shutil
from pathlib import Path

from asset_mania_contracts import DiagnosticCode

_MACOS_BLENDER_EXECUTABLE = Path("/Applications/Blender.app/Contents/MacOS/Blender")


def inspect_environment(
    configured_blender: Path | None = None,
) -> tuple[dict[str, object], list[DiagnosticCode]]:
    """Report portable host details and Blender availability without invocation."""
    blender_found = _find_blender(configured_blender) is not None
    report = {
        "operating_system": _operating_system(),
        "architecture": _architecture(),
        "python_version": platform.python_version(),
        "blender": {"status": "found" if blender_found else "not_found"},
    }
    diagnostics = [] if blender_found else [DiagnosticCode.BLENDER_NOT_FOUND]
    return report, diagnostics


def _find_blender(configured_blender: Path | None) -> Path | None:
    candidates = [configured_blender, _path_blender(), _MACOS_BLENDER_EXECUTABLE]
    for candidate in candidates:
        if candidate is not None and _is_executable_file(candidate):
            return candidate
    return None


def _is_executable_file(candidate: Path) -> bool:
    try:
        return candidate.is_file() and os.access(candidate, os.X_OK)
    except OSError:
        return False


def _path_blender() -> Path | None:
    discovered = shutil.which("blender")
    return Path(discovered) if discovered is not None else None


def _operating_system() -> str:
    return {"Darwin": "macos", "Linux": "linux"}.get(platform.system(), "unsupported")


def _architecture() -> str:
    return {"aarch64": "arm64", "arm64": "arm64", "AMD64": "x86_64", "x86_64": "x86_64"}.get(
        platform.machine(), "unsupported"
    )
