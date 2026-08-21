#!/usr/bin/env python3
"""Run one image through the real engine and report what actually came out.

This is not a unit test. It needs a TripoSR checkout, a checkpoint on disk, and a few
gigabytes of resident memory, so it lives here and is invoked deliberately.

It reports measurements, not a verdict dressed up as one. A reconstruction that produces a
mesh is not thereby correct, and the numbers printed here -- triangle count, bounds, whether
the surface closes, which way the normals face -- are the ones worth reading before believing
a result. `--expect-closed` turns the manifold check into an exit code for CI use.

The clearance gate is deliberately not exercised here: issuing an `engine-clearance-v1`
artifact means accepting third-party licence terms, which is the user's decision, so this
script drives the port directly and says so in its output.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "engine-triposr" / "src"))

from asset_mania_engine_triposr.adapter import EngineRequest
from asset_mania_engine_triposr.ports.triposr import (
    DEFAULT_MC_RESOLUTION,
    DEFAULT_MC_THRESHOLD,
    TripoSRPort,
    TripoSRSettings,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="RGBA image, or RGB paired with --mask")
    parser.add_argument("--mask", type=Path, default=None, help="single-channel foreground mask")
    parser.add_argument("--out", type=Path, required=True, help="mesh path to write")
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=REPO_ROOT / "vendor-triposr",
        help="TripoSR checkout",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "vendor-triposr-weights",
        help="directory holding config.yaml and model.ckpt",
    )
    parser.add_argument(
        "--hub-cache",
        type=Path,
        default=REPO_ROOT / ".asset-mania" / "hf-cache",
        help="hub cache holding the pinned architecture config",
    )
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    parser.add_argument("--resolution", type=int, default=DEFAULT_MC_RESOLUTION)
    parser.add_argument("--threshold", type=float, default=DEFAULT_MC_THRESHOLD)
    parser.add_argument(
        "--expect-closed",
        action="store_true",
        help="exit non-zero unless the surface is watertight with outward normals",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists():
        print(f"refusing to overwrite {args.out}", file=sys.stderr)
        return 2

    port = TripoSRPort(
        settings=TripoSRSettings(
            engine_root=args.engine_root,
            weights_dir=args.weights,
            device=args.device,
            mc_resolution=args.resolution,
            mc_threshold=args.threshold,
            hub_cache=args.hub_cache,
        )
    )
    request = EngineRequest(
        engine="triposr-local",
        profile=f"triposr-local-{args.device}-v1",
        plan_sha256="0" * 64,
        clearance_sha256="clearance-not-issued-this-run",
        image_path=args.image,
        mask_path=args.mask,
        output_path=args.out,
        mesh_format=args.out.suffix.lstrip("."),
    )

    started = time.monotonic()
    result = port.run(request)
    elapsed = time.monotonic() - started

    import trimesh

    mesh = trimesh.load(str(args.out), process=False, force="mesh")
    report = {
        "seconds": round(elapsed, 2),
        "device": args.device,
        "mc_resolution": args.resolution,
        "triangles": result.triangle_count,
        "vertices": result.vertex_count,
        "manifold": result.manifold,
        "bytes": args.out.stat().st_size,
        "bounds_min": [round(v, 4) for v in mesh.bounds[0].tolist()],
        "bounds_max": [round(v, 4) for v in mesh.bounds[1].tolist()],
        "extent": [round(v, 4) for v in (mesh.bounds[1] - mesh.bounds[0]).tolist()],
        "surface_area": round(float(mesh.area), 5),
        "signed_volume": round(float(mesh.volume), 5),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "has_vertex_colors": mesh.visual.kind == "vertex",
        "clearance": "not issued -- the port was driven directly",
    }
    print(json.dumps(report, indent=2))

    if args.expect_closed and not (report["watertight"] and report["signed_volume"] > 0):
        print(
            "FAIL: expected a watertight surface with outward normals",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
