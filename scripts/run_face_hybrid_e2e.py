#!/usr/bin/env python3
"""Prepare, anchor, fuse, and verify one private face-hybrid research run."""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from asset_mania_contracts import TURNTABLE_YAWS, canonical_digest, canonical_json
from asset_mania_engine_triposr import (
    CanonicalView,
    FaceHybridSettings,
    canonicalize_views,
    fuse_face_anchor,
)
from asset_mania_engine_triposr.adapter import EngineRequest
from asset_mania_engine_triposr.ports.triposr import TripoSRPort, TripoSRSettings
from asset_mania_pipeline import (
    run_if_cleared,
    sha256_file,
    validate_glb,
    verify_engine_clearance,
)

PROFILE = "face-anchor-visual-hull-v1"
RUN_DIRECTORIES = ("prepare", "anchor", "fusion", "verification")


def _parse_time(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC).replace(microsecond=0)
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromisoformat(value).astimezone(UTC).replace(microsecond=0)


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object")
    return value


def _verify_seal(document: dict, field: str, label: str) -> None:
    preimage = {key: value for key, value in document.items() if key != field}
    if canonical_digest(preimage) != document.get(field):
        raise ValueError(f"{label} digest does not match its content")


def _view_paths(directory: Path) -> list[CanonicalView]:
    return [
        CanonicalView(
            yaw=yaw,
            image_path=directory / f"yaw-{yaw:03d}.png",
            mask_path=directory / f"yaw-{yaw:03d}-mask.png",
        )
        for yaw in TURNTABLE_YAWS
    ]


def _run_directory(output_parent: Path, now: datetime, run_id: str) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    result = output_parent / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{run_id}"
    try:
        result.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {result}") from error
    for relative in RUN_DIRECTORIES:
        (result / relative).mkdir()
    return result


def _run_prepare(arguments: argparse.Namespace, *, now: datetime, run_id: str) -> int:
    source_views = _view_paths(arguments.views.resolve(strict=True))
    run = _run_directory(arguments.output_parent, now, run_id)
    canonical = canonicalize_views(source_views, run / "prepare/canonical")
    preimage = {
        "schema_id": "asset-mania/private-face-hybrid-prepare",
        "schema_version": "0.1",
        "profile": PROFILE,
        "source_views": [
            {
                "label": f"view-{index}",
                "yaw": view.yaw,
                "image_sha256": sha256_file(view.image_path),
                "mask_sha256": sha256_file(view.mask_path),
            }
            for index, view in enumerate(source_views, start=1)
        ],
        "canonical_views": [
            {
                "label": f"view-{index}",
                "yaw": view.yaw,
                "image_sha256": sha256_file(view.image_path),
                "mask_sha256": sha256_file(view.mask_path),
            }
            for index, view in enumerate(canonical, start=1)
        ],
        "prepared_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest = {**preimage, "prepare_sha256": canonical_digest(preimage)}
    (run / "prepare/manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    print(canonical_json({"run_directory": str(run), "prepare_sha256": manifest["prepare_sha256"]}))
    return 0


def _real_anchor_runner(
    *,
    image_path: Path,
    mask_path: Path,
    output_path: Path,
    clearance: dict,
    prepare_sha256: str,
    engine_root: Path,
    weights: Path,
    hub_cache: Path,
    now: datetime,
    **_kwargs,
) -> dict:
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; the face anchor never falls back to CPU")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    port = TripoSRPort(
        TripoSRSettings(
            engine_root=engine_root,
            weights_dir=weights,
            device="cuda",
            mc_resolution=256,
            mc_threshold=25.0,
            hub_cache=hub_cache,
        )
    )

    def execute(clearance_sha256: str):
        return port.run(
            EngineRequest(
                engine="triposr-local",
                profile="triposr-local-cuda-v1",
                plan_sha256=prepare_sha256,
                clearance_sha256=clearance_sha256,
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
                mesh_format="glb",
            )
        )

    started = time.perf_counter()
    _, result = run_if_cleared(
        clearance=clearance,
        engine="triposr-local",
        now=now,
        run=execute,
    )
    torch.cuda.synchronize()
    if not torch.cuda.is_available():
        raise ValueError("CUDA became unavailable during face anchor execution")
    return {
        "device": "cuda",
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "triangle_count": result.triangle_count,
        "vertex_count": result.vertex_count,
        "manifold": result.manifold,
    }


def _run_anchor(
    arguments: argparse.Namespace,
    *,
    now: datetime,
    anchor_runner: Callable | None,
) -> int:
    if arguments.device != "cuda":
        raise ValueError("CUDA is required for the private face anchor")
    run = arguments.run.resolve(strict=True)
    manifest = _load_object(run / "prepare/manifest.json", "prepare manifest")
    _verify_seal(manifest, "prepare_sha256", "prepare manifest")
    clearance = _load_object(arguments.clearance, "engine clearance")
    verify_engine_clearance(clearance, engine="triposr-local", now=now)
    canonical = _view_paths(run / "prepare/canonical")
    source = canonical[0]
    declared = manifest["canonical_views"][0]
    if sha256_file(source.image_path) != declared["image_sha256"]:
        raise ValueError("canonical yaw-0 image differs from the prepare manifest")
    if sha256_file(source.mask_path) != declared["mask_sha256"]:
        raise ValueError("canonical yaw-0 mask differs from the prepare manifest")
    output = run / "anchor/anchor.glb"
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    selected = anchor_runner or _real_anchor_runner
    result = selected(
        image_path=source.image_path,
        mask_path=source.mask_path,
        output_path=output,
        clearance=clearance,
        prepare_sha256=manifest["prepare_sha256"],
        engine_root=arguments.engine_root,
        weights=arguments.weights,
        hub_cache=arguments.hub_cache,
        now=now,
        device="cuda",
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("face anchor runner did not write a GLB")
    preimage = {
        "schema_id": "asset-mania/private-face-anchor",
        "schema_version": "0.1",
        "profile": PROFILE,
        "prepare_sha256": manifest["prepare_sha256"],
        "anchor_sha256": sha256_file(output),
        **dict(result),
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    (run / "anchor/record.json").write_text(canonical_json(record), encoding="utf-8")
    return 0


def _run_fuse(arguments: argparse.Namespace, *, fusion_runner: Callable | None) -> int:
    run = arguments.run.resolve(strict=True)
    manifest = _load_object(run / "prepare/manifest.json", "prepare manifest")
    anchor_record = _load_object(run / "anchor/record.json", "anchor record")
    _verify_seal(manifest, "prepare_sha256", "prepare manifest")
    _verify_seal(anchor_record, "record_sha256", "anchor record")
    anchor = run / "anchor/anchor.glb"
    if sha256_file(anchor) != anchor_record["anchor_sha256"]:
        raise ValueError("face anchor differs from its record")
    canonical = _view_paths(run / "prepare/canonical")
    for view, declared in zip(canonical, manifest["canonical_views"], strict=True):
        if sha256_file(view.image_path) != declared["image_sha256"]:
            raise ValueError(f"canonical image yaw {view.yaw} differs from its manifest")
        if sha256_file(view.mask_path) != declared["mask_sha256"]:
            raise ValueError(f"canonical mask yaw {view.yaw} differs from its manifest")
    output = run / "fusion/face-hybrid.glb"
    selected = fusion_runner or fuse_face_anchor
    result = selected(
        anchor_mesh=anchor,
        views=canonical,
        output_path=output,
        settings=FaceHybridSettings(),
    )
    preimage = {
        "schema_id": "asset-mania/private-face-hybrid-result",
        "schema_version": "0.1",
        "profile": PROFILE,
        "prepare_sha256": manifest["prepare_sha256"],
        "anchor_sha256": anchor_record["anchor_sha256"],
        "hybrid_sha256": sha256_file(output),
        "measurements": {
            field: getattr(result, field)
            for field in (
                "triangle_count",
                "vertex_count",
                "manifold",
                "signed_volume",
                "component_count",
                "minimum_reprojection_iou",
                "mean_reprojection_iou",
                "front_anchor_retention",
                "color_coverage",
            )
        },
        "identity_consistency": "unmeasured",
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    (run / "fusion/record.json").write_text(canonical_json(record), encoding="utf-8")
    return 0


def _find_blender(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(found) if (found := shutil.which("blender")) else None,
        Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise ValueError("Blender 5.2 was not found for face hybrid verification")


def _default_preview_runner(mesh: Path, preview: Path, blender: Path | None) -> None:
    executable = _find_blender(blender)
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/blender_preview.py"),
            "render",
            "--blender",
            str(executable),
            "--mesh",
            str(mesh),
            "--out",
            str(preview),
            "--samples",
            "16",
            "--resolution",
            "500",
            "--views",
            "4",
            "--vertex-colors",
        ],
        cwd=ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Blender preview failed with exit {completed.returncode}")


def _load_single_mesh(path: Path):
    """Load this profile's one-geometry GLB without trimesh's NumPy-2-broken scene dump."""
    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    geometries = list(loaded.geometry.values())
    if len(geometries) != 1:
        raise ValueError("face hybrid GLB must contain exactly one geometry")
    return geometries[0]


def _run_verify(arguments: argparse.Namespace, *, preview_runner: Callable | None) -> int:
    run = arguments.run.resolve(strict=True)
    manifest = _load_object(run / "prepare/manifest.json", "prepare manifest")
    fusion = _load_object(run / "fusion/record.json", "fusion record")
    _verify_seal(manifest, "prepare_sha256", "prepare manifest")
    _verify_seal(fusion, "record_sha256", "fusion record")
    source_views = _view_paths(arguments.views.resolve(strict=True))
    source_unchanged = all(
        sha256_file(path) == declared[field]
        for view, declared in zip(source_views, manifest["source_views"], strict=True)
        for path, field in (
            (view.image_path, "image_sha256"),
            (view.mask_path, "mask_sha256"),
        )
    )
    if not source_unchanged:
        raise ValueError("source views differ from the prepare manifest")
    mesh_path = run / "fusion/face-hybrid.glb"
    if sha256_file(mesh_path) != fusion["hybrid_sha256"]:
        raise ValueError("face hybrid differs from its fusion record")
    container = validate_glb(mesh_path)
    mesh = _load_single_mesh(mesh_path)
    artifact = {
        "glb_version": container.json_chunk["asset"]["version"],
        "byte_size": mesh_path.stat().st_size,
        "sha256": sha256_file(mesh_path),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "signed_volume": float(mesh.volume),
        "triangle_count": len(mesh.faces),
        "vertex_count": len(mesh.vertices),
    }
    if (
        not artifact["watertight"]
        or not artifact["winding_consistent"]
        or artifact["signed_volume"] <= 0
    ):
        raise ValueError("face hybrid artifact verification failed")
    preview = run / "verification/preview.png"
    if preview.exists():
        raise ValueError(f"refusing to overwrite {preview}")
    selected = preview_runner or _default_preview_runner
    selected(mesh_path, preview, arguments.blender)
    report = {
        "schema_id": "asset-mania/private-face-hybrid-verification",
        "schema_version": "0.1",
        "status": "passed",
        "profile": PROFILE,
        "source_unchanged": True,
        "artifact": artifact,
        "visual_quality": "unreviewed",
        "identity_consistency": "unmeasured",
    }
    (run / "verification/report.json").write_text(canonical_json(report), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--views", type=Path, required=True)
    prepare.add_argument("--out", type=Path, dest="output_parent", required=True)

    anchor = commands.add_parser("anchor")
    anchor.add_argument("--run", type=Path, required=True)
    anchor.add_argument("--clearance", type=Path, required=True)
    anchor.add_argument("--engine-root", type=Path, required=True)
    anchor.add_argument("--weights", type=Path, required=True)
    anchor.add_argument("--hub-cache", type=Path, required=True)
    anchor.add_argument("--device", default="cuda")

    fuse = commands.add_parser("fuse")
    fuse.add_argument("--run", type=Path, required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    verify.add_argument("--views", type=Path, required=True)
    verify.add_argument("--blender", type=Path)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    now: str | datetime | None = None,
    id_factory: Callable[[], str] | None = None,
    anchor_runner: Callable | None = None,
    fusion_runner: Callable | None = None,
    preview_runner: Callable | None = None,
) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    timestamp = _parse_time(now)
    if arguments.command == "prepare":
        return _run_prepare(
            arguments,
            now=timestamp,
            run_id=(id_factory or (lambda: secrets.token_hex(4)))(),
        )
    if arguments.command == "anchor":
        return _run_anchor(arguments, now=timestamp, anchor_runner=anchor_runner)
    if arguments.command == "fuse":
        return _run_fuse(arguments, fusion_runner=fusion_runner)
    return _run_verify(arguments, preview_runner=preview_runner)


if __name__ == "__main__":
    raise SystemExit(main())
