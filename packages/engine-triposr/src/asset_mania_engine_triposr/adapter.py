"""The clearance-gated engine adapter.

The adapter runs nothing itself. Execution is an injected port, and the default port refuses
every call, which is what makes "no engine ran uncleared" a property a test can check rather
than a promise in a comment.

This module deliberately constructs no subprocess, socket, or model loader. Its own test
suite scans this source to prove it, because the easiest way for a weights download to appear
in a pipeline is for an adapter to grow one quietly.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from asset_mania_contracts import DiagnosticCode
from asset_mania_pipeline import (
    describe_reconstruction_output,
    run_if_cleared,
    sha256_bytes,
)

ENGINE = "triposr-local"
DEFAULT_PROFILE = "triposr-local-cpu-v1"
#: What a caller must supply for the adapter to describe a run at all.
REQUIRED_CLEARANCE_ROLES = ("engine_code", "model_weights", "preprocessing_model")


class EngineUnavailable(Exception):
    """No execution port is installed, or the port refused to start."""


class ReconstructionFailed(Exception):
    """The port ran and produced nothing usable."""


@dataclass(frozen=True, slots=True)
class EngineRequest:
    """One reconstruction request. It carries digests and paths, never model weights."""

    engine: str
    profile: str
    plan_sha256: str
    clearance_sha256: str
    image_path: Path
    mask_path: Path | None
    output_path: Path
    mesh_format: str

    def redacted(self) -> dict[str, Any]:
        """A log-safe view: digests and shapes, no local path."""
        return {
            "engine": self.engine,
            "profile": self.profile,
            "plan_sha256": self.plan_sha256,
            "clearance_sha256": self.clearance_sha256,
            "has_mask": self.mask_path is not None,
            "mesh_format": self.mesh_format,
        }


@dataclass(frozen=True, slots=True)
class EngineResult:
    """What a port reports back. Counts are measured by the port, not guessed here."""

    triangle_count: int
    vertex_count: int
    manifold: str


class ExecutionPort(Protocol):
    """What the adapter needs from an engine, and nothing more."""

    def run(self, request: EngineRequest) -> EngineResult: ...


class RefusingPort:
    """The default port. It refuses, so nothing runs by accident."""

    def __init__(self, reason: str = "no execution port was supplied") -> None:
        self.reason = reason

    def run(self, request: EngineRequest) -> EngineResult:
        raise EngineUnavailable(f"{DiagnosticCode.ENGINE_UNAVAILABLE.value}: {self.reason}")


@dataclass
class FakePort:
    """A port for tests: records requests and writes a declared placeholder mesh.

    It exists so the gate, the ordering, and the output contract can be exercised without an
    engine, a weight, or a download. It never claims to reconstruct anything.
    """

    triangle_count: int = 4096
    vertex_count: int = 2048
    manifold: str = "closed"
    payload: bytes = b"glTF placeholder produced by the fake port"
    write_output: bool = True
    requests: list[dict[str, Any]] = field(default_factory=list)

    def run(self, request: EngineRequest) -> EngineResult:
        self.requests.append(request.redacted())
        if self.write_output:
            request.output_path.parent.mkdir(parents=True, exist_ok=True)
            request.output_path.write_bytes(self.payload)
        return EngineResult(
            triangle_count=self.triangle_count,
            vertex_count=self.vertex_count,
            manifold=self.manifold,
        )


def reconstruct(
    *,
    plan: Mapping[str, Any],
    clearance: Mapping[str, Any] | None,
    image_path: Path,
    staging_root: Path,
    now: datetime,
    mask_path: Path | None = None,
    port: ExecutionPort | None = None,
    profile: str = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Verify clearance, run the port, and describe what came back.

    The clearance check happens inside `run_if_cleared`, so the port is unreachable until it
    passes. A port that returns without writing a mesh is a failure, not an empty success.
    """
    port = port if port is not None else RefusingPort()
    if plan["engine"] != ENGINE:
        raise EngineUnavailable(
            f"{DiagnosticCode.ENGINE_UNAVAILABLE.value}: this adapter serves {ENGINE!r}, "
            f"not {plan['engine']!r}"
        )

    mesh_format = plan["expected_output"]["mesh_format"]
    output = staging_root / f"reconstruction.{mesh_format}"
    if output.exists():
        raise ReconstructionFailed(
            f"{DiagnosticCode.OUTPUT_COLLISION.value}: {output.name} already exists"
        )

    def run(clearance_digest: str) -> EngineResult:
        request = EngineRequest(
            engine=ENGINE,
            profile=profile,
            plan_sha256=plan["plan_sha256"],
            clearance_sha256=clearance_digest,
            image_path=image_path,
            mask_path=mask_path,
            output_path=output,
            mesh_format=mesh_format,
        )
        return port.run(request)

    clearance_digest, result = run_if_cleared(clearance=clearance, engine=ENGINE, now=now, run=run)

    if not output.is_file():
        raise ReconstructionFailed(
            f"{DiagnosticCode.RECONSTRUCTION_FAILED.value}: the port wrote no mesh"
        )

    record = describe_reconstruction_output(
        mesh_path=output,
        plan=plan,
        triangle_count=result.triangle_count,
        vertex_count=result.vertex_count,
        manifold=result.manifold,
    )
    return {
        "clearance_sha256": clearance_digest,
        "mesh": record,
        "engine": ENGINE,
        "profile": profile,
        "semantic_digest": sha256_bytes(output.read_bytes()),
    }


def available_engines(entry_points: Sequence[str] | None = None) -> list[str]:
    """The engines this adapter declares. Discovery is by entry point, never by import."""
    return list(entry_points) if entry_points is not None else [ENGINE]


__all__ = [
    "DEFAULT_PROFILE",
    "ENGINE",
    "REQUIRED_CLEARANCE_ROLES",
    "EngineRequest",
    "EngineResult",
    "EngineUnavailable",
    "ExecutionPort",
    "FakePort",
    "ReconstructionFailed",
    "RefusingPort",
    "available_engines",
    "reconstruct",
]
