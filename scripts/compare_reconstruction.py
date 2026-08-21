#!/usr/bin/env python3
"""Measure a reconstruction against the solid that produced its input image.

    .venv/bin/python scripts/compare_reconstruction.py \\
        --truth /tmp/fixture.png.truth.obj --recon /tmp/out.obj

A picture of a reconstruction tells you whether it looks plausible. This tells you how far it
is from the answer, which is a different question and the one worth asking -- the first
synthetic run here produced a hollow shell that photographed perfectly well.

Two frame problems have to be handled before any comparison means anything:

* TripoSR emits its own canonical orientation, unrelated to the scene the image came from. So
  the reconstruction is tried in all 48 axis permutations and sign flips and the best-scoring
  one is used. A rotation off the axes would need ICP; the measured residual says whether that
  is worth adding.
* Scale is arbitrary in monocular reconstruction. Both meshes are normalised to a unit longest
  axis, so every distance below is a fraction of the subject's own size.

`--max-mean-error` turns the result into an exit code, for use as a regression check.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

#: Surface samples for each distance direction. 20k is enough for the mean to settle to about
#: three decimals on a subject of this size; the numbers are reported to four so drift shows.
SAMPLES = 20_000

#: Samples used while searching the 48 candidate frames. Deliberately smaller -- the search
#: only has to rank frames, and the winner is then measured properly.
ALIGNMENT_SAMPLES = 6_000


def load(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def normalise(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre on the bounding box and scale the longest axis to one.

    Done on the vertex array rather than through `apply_transform`, which raises under trimesh
    4.0.5 with numpy 2.x -- and 4.0.5 is the version TripoSR pins.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    vertices = vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices = vertices / (vertices.max(axis=0) - vertices.min(axis=0)).max()
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces), process=False)


def _permutation_sign(permutation: tuple[int, ...]) -> int:
    """+1 for an even permutation, -1 for an odd one."""
    inversions = sum(
        1
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
        if permutation[i] > permutation[j]
    )
    return -1 if inversions % 2 else 1


def best_frame(
    reconstruction: trimesh.Trimesh, truth: trimesh.Trimesh
) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Pick the axis-aligned *rotation* that puts the reconstruction closest to truth.

    Only proper rotations are allowed. The first version of this searched all 48 permutations
    and sign flips, and picked a reflection -- visible as a negative signed volume in the
    output. A reflection is not a rigid motion, and for a left-right symmetric subject like
    Suzanne it can score better than any real rotation, which flatters the result for free. So
    the search is restricted to determinant +1, and the best reflection is measured anyway and
    reported alongside: if it scores much better, the reconstruction is probably mirrored, and
    that is a finding rather than something to quietly absorb.
    """
    vertices = np.asarray(reconstruction.vertices, dtype=float)
    faces = np.asarray(reconstruction.faces)
    rotations: tuple[float, Any, Any, trimesh.Trimesh] | None = None
    reflections: float | None = None

    for permutation in itertools.permutations(range(3)):
        parity = _permutation_sign(permutation)
        for signs in itertools.product((1, -1), repeat=3):
            determinant = parity * signs[0] * signs[1] * signs[2]
            moved = vertices[:, permutation] * signs
            moved = moved - (moved.min(axis=0) + moved.max(axis=0)) / 2
            candidate = trimesh.Trimesh(vertices=moved, faces=faces, process=False)
            score = float(_surface_distance(truth, candidate, ALIGNMENT_SAMPLES).mean())

            if determinant > 0:
                if rotations is None or score < rotations[0]:
                    rotations = (score, permutation, signs, candidate)
            elif reflections is None or score < reflections:
                reflections = score

    assert rotations is not None
    score, permutation, signs, aligned = rotations
    return aligned, {
        "axis_permutation": list(permutation),
        "axis_signs": list(signs),
        "alignment_search": "proper rotations only (determinant +1)",
        "best_rotation_mean": round(score, 5),
        "best_reflection_mean": None if reflections is None else round(reflections, 5),
        "reflection_fits_better": bool(reflections is not None and reflections < score),
    }


def _surface_distance(source: trimesh.Trimesh, target: trimesh.Trimesh, count: int):
    """Nearest-neighbour distance from samples on `source` to samples on `target`.

    `trimesh.proximity.signed_distance` would be the exact point-to-triangle answer, but it
    raises under trimesh 4.0.5 with numpy 2.x -- as do `apply_transform` and
    `creation.box` -- and 4.0.5 is the version TripoSR pins. A dense point-to-point query is
    an upper bound on the true surface distance, tight to roughly the sample spacing, which at
    this density is well under the errors being reported.
    """
    from scipy.spatial import cKDTree

    a = source.sample(count)
    b = target.sample(max(count * 4, 40_000))
    return cKDTree(b).query(a, k=1)[0]


def _extents(mesh: trimesh.Trimesh) -> list[float]:
    """Bounding-box side lengths, largest first.

    `mesh.extents` is the obvious call and raises: it is implemented as `bounds.ptp(axis=0)`,
    and `ndarray.ptp` was removed in numpy 2.0. Third occurrence of the same trimesh-4.0.5
    incompatibility, so it is computed here rather than worked around at each call site.
    """
    lo, hi = mesh.bounds
    return sorted((hi - lo).tolist(), reverse=True)


def distances(source: trimesh.Trimesh, target: trimesh.Trimesh) -> dict[str, float]:
    values = _surface_distance(source, target, SAMPLES)
    return {
        "mean": round(float(values.mean()), 5),
        "median": round(float(np.median(values)), 5),
        "p95": round(float(np.percentile(values, 95)), 5),
        "max": round(float(values.max()), 5),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--recon", type=Path, required=True)
    parser.add_argument(
        "--max-mean-error",
        type=float,
        default=None,
        help="exit non-zero if the symmetric mean surface distance exceeds this",
    )
    args = parser.parse_args(argv)

    truth = normalise(load(args.truth))
    reconstruction = normalise(load(args.recon))
    aligned, alignment = best_frame(reconstruction, truth)

    forward = distances(truth, aligned)
    backward = distances(aligned, truth)
    symmetric = round((forward["mean"] + backward["mean"]) / 2, 5)

    report = {
        **alignment,
        "extent_truth": [round(v, 4) for v in _extents(truth)],
        "extent_recon": [round(v, 4) for v in _extents(aligned)],
        "volume_truth": round(float(truth.volume), 5),
        "volume_recon": round(float(aligned.volume), 5),
        "volume_ratio": round(float(abs(aligned.volume / truth.volume)), 4),
        "area_ratio": round(float(aligned.area / truth.area), 4),
        "truth_watertight": bool(truth.is_watertight),
        "recon_watertight": bool(aligned.is_watertight),
        "distance_truth_to_recon": forward,
        "distance_recon_to_truth": backward,
        "symmetric_mean_distance": symmetric,
        "unit": "fraction of the subject's longest axis",
    }
    print(json.dumps(report, indent=2))

    if args.max_mean_error is not None and symmetric > args.max_mean_error:
        print(
            f"FAIL: symmetric mean distance {symmetric} exceeds {args.max_mean_error}",
            file=__import__("sys").stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
