"""Reference reprojection oracle over synthetic arrays.

Same standing as `projection`: this is the analytic reference, not the production
implementation. It runs the deterministic texel loop over plain arrays so the sampling
rules, the scan order, the occlusion test, and the coverage semantics can be pinned by
test vectors, and so the GPL worker's output can be compared against a known answer.

Coverage semantics are the part most easily got wrong, so they are explicit here: observed
coverage is only ever set by a texel that actually sampled the view, seam padding extends
RGB without promoting coverage, and an uncovered texel keeps alpha zero. Nothing in this
module invents a colour for a texel it could not see.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .projection import (
    barycentric,
    clip_to_pixel,
    depth_occluded,
    euclidean_distance,
    faces_camera,
    interpolate3,
    owns_texel,
    texel_center,
    transform4,
    within_pixel_bounds,
)

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]


def srgb_to_linear(value: float) -> float:
    """The exact inverse sRGB transfer function."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def linear_to_srgb(value: float) -> float:
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (max(value, 0.0) ** (1.0 / 2.4)) - 0.055


@dataclass(frozen=True, slots=True)
class Sampler:
    """A synthetic image addressed with a top-left origin and pixel centres."""

    width: int
    height: int
    #: Row-major rows of per-pixel tuples. Channel count is whatever the caller stores.
    pixels: Sequence[Sequence[Sequence[float]]]

    def at(self, x: int, y: int) -> Sequence[float]:
        clamped_x = min(max(x, 0), self.width - 1)
        clamped_y = min(max(y, 0), self.height - 1)
        return self.pixels[clamped_y][clamped_x]

    def nearest(self, u: float, v: float) -> Sequence[float]:
        """Nearest sampling, used for the mask and the depth pass."""
        return self.at(math.floor(u + 0.5), math.floor(v + 0.5))

    def bilinear(self, u: float, v: float) -> list[float]:
        """Bilinear sampling, used only for colour."""
        x0 = math.floor(u)
        y0 = math.floor(v)
        fx = u - x0
        fy = v - y0
        top_left = self.at(x0, y0)
        top_right = self.at(x0 + 1, y0)
        bottom_left = self.at(x0, y0 + 1)
        bottom_right = self.at(x0 + 1, y0 + 1)
        return [
            (top_left[channel] * (1 - fx) + top_right[channel] * fx) * (1 - fy)
            + (bottom_left[channel] * (1 - fx) + bottom_right[channel] * fx) * fy
            for channel in range(len(top_left))
        ]


@dataclass
class ReprojectionResult:
    atlas_width: int
    atlas_height: int
    #: Scene-linear RGB per texel; canonical intermediates are linear float.
    colour: list[list[list[float]]]
    #: Authoritative observed coverage: a texel that actually sampled the view.
    observed: list[list[bool]]
    #: Padding is tracked separately and never promotes observed coverage.
    padded: list[list[bool]] = field(default_factory=list)
    #: Stable per-texel record of which view supplied the sample.
    source_view_label: list[list[str | None]] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def observed_coverage(self) -> float:
        total = self.atlas_width * self.atlas_height
        return sum(row.count(True) for row in self.observed) / total if total else 0.0

    @property
    def padded_coverage(self) -> float:
        total = self.atlas_width * self.atlas_height
        if not self.padded or not total:
            return 0.0
        return sum(row.count(True) for row in self.padded) / total

    def alpha(self, x: int, y: int) -> int:
        """Final alpha is written explicitly: observed 255, unknown 0."""
        return 255 if self.observed[y][x] else 0


@dataclass(frozen=True, slots=True)
class Triangle:
    """One mesh triangle with its UV and evaluated world attributes."""

    polygon_index: int
    uv: tuple[Vector2, Vector2, Vector2]
    world: tuple[Vector3, Vector3, Vector3]
    normal: tuple[Vector3, Vector3, Vector3]


def reproject(
    *,
    triangles: Sequence[Triangle],
    atlas_width: int,
    atlas_height: int,
    world_to_clip: Sequence[float],
    camera_origin: Vector3,
    view_width: int,
    view_height: int,
    colour: Sampler,
    mask: Sampler,
    depth: Sampler,
    view_label: str = "view-1",
    first_hit: Callable[[Vector3, Vector3], bool] | None = None,
    decode_srgb: bool = True,
) -> ReprojectionResult:
    """Project every owned atlas texel back into one view, in stable scan order.

    Triangles are visited by polygon index and texels row-major, so the result does not
    depend on iteration accidents. Each rejection reason is counted rather than swallowed,
    which is what makes a low-coverage result diagnosable instead of merely disappointing.
    """
    result = ReprojectionResult(
        atlas_width=atlas_width,
        atlas_height=atlas_height,
        colour=[[[0.0, 0.0, 0.0] for _ in range(atlas_width)] for _ in range(atlas_height)],
        observed=[[False] * atlas_width for _ in range(atlas_height)],
        padded=[[False] * atlas_width for _ in range(atlas_height)],
        source_view_label=[[None] * atlas_width for _ in range(atlas_height)],
        rejected={
            "degenerate_uv": 0,
            "outside_clip": 0,
            "outside_bounds": 0,
            "backfacing": 0,
            "outside_mask": 0,
            "depth_mismatch": 0,
            "occluded": 0,
        },
    )

    for triangle in sorted(triangles, key=lambda item: item.polygon_index):
        for y in range(atlas_height):
            for x in range(atlas_width):
                if result.observed[y][x]:
                    # A texel belongs to exactly one triangle under the fill rule; this
                    # guard makes a caller's overlapping UVs harmless rather than
                    # order-dependent.
                    continue
                centre = texel_center(x, y, atlas_width, atlas_height)
                if not owns_texel(centre, triangle.uv):
                    continue

                weights = barycentric(centre, triangle.uv)
                if weights is None:
                    result.rejected["degenerate_uv"] += 1
                    continue

                world = interpolate3(weights, triangle.world)
                normal = interpolate3(weights, triangle.normal)

                clip = transform4(world_to_clip, (*world, 1.0))
                pixel = clip_to_pixel(clip, view_width, view_height)
                if pixel is None:
                    result.rejected["outside_clip"] += 1
                    continue
                u, v = pixel
                if not within_pixel_bounds(u, v, view_width, view_height):
                    result.rejected["outside_bounds"] += 1
                    continue
                if not faces_camera(normal, world, camera_origin):
                    result.rejected["backfacing"] += 1
                    continue
                if mask.nearest(u, v)[0] <= 0.0:
                    result.rejected["outside_mask"] += 1
                    continue

                expected = euclidean_distance(camera_origin, world)
                neighbourhood = [
                    depth.at(math.floor(u) + dx, math.floor(v) + dy)[0]
                    for dx in (0, 1)
                    for dy in (0, 1)
                ]
                if depth_occluded(expected, neighbourhood):
                    result.rejected["depth_mismatch"] += 1
                    continue
                if first_hit is not None and not first_hit(world, normal):
                    result.rejected["occluded"] += 1
                    continue

                sample = colour.bilinear(u, v)
                result.colour[y][x] = [
                    srgb_to_linear(sample[channel]) if decode_srgb else sample[channel]
                    for channel in range(3)
                ]
                result.observed[y][x] = True
                result.padded[y][x] = True
                result.source_view_label[y][x] = view_label

    return result


def dilate_same_island(result: ReprojectionResult, *, margin: int) -> ReprojectionResult:
    """Bounded, deterministic seam dilation. It never promotes observed coverage.

    Padding exists so texture filtering does not pull background into a seam. It copies an
    already-observed neighbour's colour outward; it does not invent content, and the
    observed mask -- the authority for coverage and for final alpha -- is untouched.
    """
    if margin <= 0:
        return result

    for _ in range(margin):
        additions: list[tuple[int, int, list[float]]] = []
        for y in range(result.atlas_height):
            for x in range(result.atlas_width):
                if result.padded[y][x]:
                    continue
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx, ny = x + dx, y + dy
                    inside = 0 <= nx < result.atlas_width and 0 <= ny < result.atlas_height
                    if inside and result.padded[ny][nx]:
                        additions.append((x, y, list(result.colour[ny][nx])))
                        break
        for x, y, colour in additions:
            result.colour[y][x] = colour
            result.padded[y][x] = True
    return result
