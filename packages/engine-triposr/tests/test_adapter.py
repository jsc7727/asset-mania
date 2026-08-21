"""The engine adapter: nothing runs uncleared, and nothing is downloaded."""

import ast
import copy
import json
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from asset_mania_contracts import canonical_digest
from asset_mania_engine_triposr.adapter import (
    ENGINE,
    EngineUnavailable,
    FakePort,
    ReconstructionFailed,
    RefusingPort,
    available_engines,
    reconstruct,
)
from asset_mania_pipeline import (
    EngineLicenseUncleared,
    EngineNotCleared,
    ReconstructionRejected,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "tests" / "fixtures" / "v2"
SOURCE = Path(__file__).resolve().parents[1] / "src" / "asset_mania_engine_triposr"
NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def deny_network_and_processes(monkeypatch):
    """The adapter must reach neither the network nor a subprocess."""

    def refuse(*args, **kwargs):
        raise AssertionError("the engine suite must not open a socket or start a process")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)


def _example(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))


def _reseal(clearance: dict) -> dict:
    preimage = {k: v for k, v in clearance.items() if k != "clearance_sha256"}
    return {**preimage, "clearance_sha256": canonical_digest(preimage)}


@pytest.fixture
def plan() -> dict:
    return _example("reconstruction-plan-v1")


@pytest.fixture
def clearance() -> dict:
    return _example("engine-clearance-v1")


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "subject.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(32))
    return path


@pytest.fixture
def staging(tmp_path: Path) -> Path:
    path = tmp_path / "staging"
    path.mkdir()
    return path


def _run(plan, clearance, image, staging, **overrides):
    arguments = {
        "plan": plan,
        "clearance": clearance,
        "image_path": image,
        "staging_root": staging,
        "now": NOW,
    }
    arguments.update(overrides)
    return reconstruct(**arguments)


# --- The adapter builds no execution machinery of its own -----------------------------


def adapter_layer_sources() -> list[Path]:
    """Every module that must stay free of an engine.

    `ports/` is excluded on purpose. Something has to load the checkpoint eventually, and
    pretending otherwise would only mean the loader hides somewhere less obvious. The split
    is the guarantee: this layer decides *whether* a run may happen, `ports/` performs it,
    and each is checked for the failure it can actually have. A port importing torch is its
    job; a port acquiring its own weights is not, and `test_port_triposr.py` scans for that.
    """
    return [p for p in sorted(SOURCE.rglob("*.py")) if p.parent.name != "ports"]


def test_the_adapter_layer_excludes_only_the_ports_package() -> None:
    """If `ports/` were renamed or the tree reshaped, the scan above would quietly cover
    nothing, so the partition itself is asserted rather than assumed."""
    scanned = {p.relative_to(SOURCE).as_posix() for p in adapter_layer_sources()}
    everything = {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*.py")}
    assert "adapter.py" in scanned
    assert scanned, "the adapter layer scan covers no files"
    assert all(p.startswith("ports/") for p in everything - scanned)


def test_the_adapter_imports_no_process_socket_or_loader() -> None:
    """The easiest way a weights download appears is an adapter growing one quietly."""
    forbidden = {
        "subprocess",
        "socket",
        "http",
        "urllib",
        "requests",
        "httpx",
        "torch",
        "transformers",
        "huggingface_hub",
        "rembg",
        "onnxruntime",
    }
    for path in adapter_layer_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        assert not (roots & forbidden), f"{path.name}: {sorted(roots & forbidden)}"


def test_the_adapter_mentions_no_download_url() -> None:
    for path in adapter_layer_sources():
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text
        assert "https://" not in text


def test_the_engine_is_declared_for_discovery() -> None:
    assert available_engines() == [ENGINE]


def test_the_default_port_refuses(plan, clearance, image, staging) -> None:
    with pytest.raises(EngineUnavailable, match="ENGINE_UNAVAILABLE"):
        _run(plan, clearance, image, staging)


def test_the_refusing_port_names_its_reason() -> None:
    with pytest.raises(EngineUnavailable, match="no execution port"):
        RefusingPort().run(object())  # type: ignore[arg-type]


# --- Nothing runs uncleared -------------------------------------------------------------


def _assert_port_untouched(plan, clearance, image, staging, failure):
    port = FakePort()
    with pytest.raises(failure):
        _run(plan, clearance, image, staging, port=port)
    assert port.requests == []
    assert not list(staging.glob("reconstruction.*"))


def test_no_clearance_never_reaches_the_port(plan, image, staging) -> None:
    _assert_port_untouched(plan, None, image, staging, EngineNotCleared)


def test_an_uncleared_license_never_reaches_the_port(plan, image, staging) -> None:
    _assert_port_untouched(
        plan, _example("engine-clearance-v1-uncleared"), image, staging, EngineLicenseUncleared
    )


def test_a_maintainer_issued_clearance_never_reaches_the_port(
    plan, clearance, image, staging
) -> None:
    _assert_port_untouched(
        plan, _reseal({**clearance, "cleared_by": "maintainer"}), image, staging, EngineNotCleared
    )


def test_an_expired_clearance_never_reaches_the_port(plan, clearance, image, staging) -> None:
    port = FakePort()
    late = datetime(2027, 1, 1, tzinfo=UTC)
    with pytest.raises(EngineNotCleared):
        _run(plan, clearance, image, staging, port=port, now=late)
    assert port.requests == []


def test_an_empty_dependency_list_never_reaches_the_port(plan, clearance, image, staging) -> None:
    mutated = _reseal({**copy.deepcopy(clearance), "runtime_dependencies": []})
    _assert_port_untouched(plan, mutated, image, staging, EngineNotCleared)


def test_a_plan_for_another_engine_never_reaches_the_port(plan, clearance, image, staging) -> None:
    port = FakePort()
    with pytest.raises(EngineUnavailable, match="serves"):
        _run({**plan, "engine": "some-other-engine"}, clearance, image, staging, port=port)
    assert port.requests == []


# --- The cleared path --------------------------------------------------------------------


def test_a_cleared_run_reaches_the_port_exactly_once(plan, clearance, image, staging) -> None:
    port = FakePort()
    result = _run(plan, clearance, image, staging, port=port)
    assert len(port.requests) == 1
    assert result["clearance_sha256"] == clearance["clearance_sha256"]
    assert result["engine"] == ENGINE


def test_the_mesh_is_recorded_as_generated(plan, clearance, image, staging) -> None:
    result = _run(plan, clearance, image, staging, port=FakePort())
    mesh = result["mesh"]
    assert mesh["content_origin"] == "generated"
    assert mesh["sensitivity"] == "user-content"
    assert mesh["upload_eligible"] is False
    assert mesh["operation"] == "reconstruct"


def test_the_mesh_records_its_measured_counts_and_manifold(plan, clearance, image, staging) -> None:
    port = FakePort(triangle_count=777, vertex_count=555, manifold="open")
    mesh = _run(plan, clearance, image, staging, port=port)["mesh"]
    assert mesh["triangle_count"] == 777
    assert mesh["vertex_count"] == 555
    assert mesh["manifold"] == "open"


def test_the_request_record_carries_no_local_path(plan, clearance, image, staging) -> None:
    port = FakePort()
    _run(plan, clearance, image, staging, port=port)
    rendered = json.dumps(port.requests)
    assert str(image) not in rendered
    assert image.name not in rendered
    assert str(staging) not in rendered
    assert set(port.requests[0]) == {
        "engine",
        "profile",
        "plan_sha256",
        "clearance_sha256",
        "has_mask",
        "mesh_format",
    }


def test_a_supplied_mask_is_reported_only_as_a_boolean(
    plan, clearance, image, staging, tmp_path
) -> None:
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"\x89PNG\r\n\x1a\n")
    port = FakePort()
    _run(plan, clearance, image, staging, port=port, mask_path=mask)
    assert port.requests[0]["has_mask"] is True
    assert mask.name not in json.dumps(port.requests)


# --- Output refusals -----------------------------------------------------------------------


def test_a_port_that_writes_nothing_is_a_failure_not_an_empty_success(
    plan, clearance, image, staging
) -> None:
    port = FakePort(write_output=False)
    with pytest.raises(ReconstructionFailed, match="RECONSTRUCTION_FAILED"):
        _run(plan, clearance, image, staging, port=port)
    assert len(port.requests) == 1


def test_an_empty_mesh_is_refused(plan, clearance, image, staging) -> None:
    with pytest.raises(ReconstructionRejected, match="RECONSTRUCTION_FAILED"):
        _run(plan, clearance, image, staging, port=FakePort(payload=b""))


@pytest.mark.parametrize(("triangles", "vertices"), [(0, 10), (10, 0)])
def test_a_mesh_with_no_geometry_is_refused(
    plan, clearance, image, staging, triangles: int, vertices: int
) -> None:
    port = FakePort(triangle_count=triangles, vertex_count=vertices)
    with pytest.raises(ReconstructionRejected, match="RECONSTRUCTION_UNVERIFIED"):
        _run(plan, clearance, image, staging, port=port)


def test_an_undeclared_manifold_state_is_refused(plan, clearance, image, staging) -> None:
    with pytest.raises(ReconstructionRejected, match="manifold"):
        _run(plan, clearance, image, staging, port=FakePort(manifold="probably-closed"))


def test_an_existing_output_is_never_overwritten(plan, clearance, image, staging) -> None:
    existing = staging / "reconstruction.glb"
    existing.write_bytes(b"earlier run")
    with pytest.raises(ReconstructionFailed, match="OUTPUT_COLLISION"):
        _run(plan, clearance, image, staging, port=FakePort())
    assert existing.read_bytes() == b"earlier run"
