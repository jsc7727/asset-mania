"""Reference camera and raster math for reprojection.

This is a small analytic oracle over synthetic arrays. It is **not** the production
reprojection: the GPL Blender worker owns that, because it has the evaluated mesh, the
render passes, and ray casting without moving private geometry into an Apache process.
What lives here is the conventions written down once, testable without Blender, and used
to check the worker against known vectors.

Every convention below is spelled out rather than inferred, since an off-by-half-pixel or
a flipped V would corrupt every texel while still looking plausible.
"""

import math
from collections.abc import Sequence

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Vector4 = tuple[float, float, float, float]
Matrix4 = Sequence[float]

#: The binding depth agreement: max(absolute, expected * relative).
DEPTH_ABSOLUTE_TOLERANCE_METERS = 0.0001
DEPTH_RELATIVE_TOLERANCE = 0.0002
#: The binding self-hit epsilon: bounding-box diagonal times a scale, clamped.
RAY_EPSILON_SCALE = 1e-7
RAY_EPSILON_MIN_METERS = 1e-7
RAY_EPSILON_MAX_METERS = 1e-3


def texel_center(x: int, y: int, atlas_width: int, atlas_height: int) -> Vector2:
    """The UV coordinate at the centre of atlas texel `(x, y)`.

    V is flipped because the atlas is addressed with a top-left origin while UV space has
    its origin at the bottom left.
    """
    return ((x + 0.5) / atlas_width, 1.0 - (y + 0.5) / atlas_height)


def _edge(a: Vector2, b: Vector2, point: Vector2) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])


def signed_area(triangle: Sequence[Vector2]) -> float:
    a, b, c = triangle
    return _edge(a, b, c) / 2.0


def barycentric(point: Vector2, triangle: Sequence[Vector2]) -> Vector3 | None:
    """Barycentric coordinates of `point` in a 2D triangle, or None if degenerate."""
    a, b, c = triangle
    doubled = _edge(a, b, c)
    if doubled == 0.0 or not math.isfinite(doubled):
        return None
    weight_a = _edge(b, c, point) / doubled
    weight_b = _edge(c, a, point) / doubled
    weight_c = _edge(a, b, point) / doubled
    return (weight_a, weight_b, weight_c)


def _is_top_left_edge(a: Vector2, b: Vector2) -> bool:
    """A top or left edge under a counter-clockwise winding in UV space.

    The fill rule only matters for a sample lying exactly on a shared edge. Assigning such
    a sample to the triangle whose edge is top-or-left makes the ownership total and
    disjoint: no texel is rasterized twice and none is dropped.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    if dy == 0.0:
        return dx < 0.0
    return dy > 0.0


def owns_texel(point: Vector2, triangle: Sequence[Vector2]) -> bool:
    """Whether `triangle` owns `point` under the top-left fill rule."""
    a, b, c = triangle
    area = _edge(a, b, c)
    if area == 0.0 or not math.isfinite(area):
        return False
    if area < 0.0:
        # Normalize to counter-clockwise so the fill rule has one winding to reason about.
        b, c = c, b

    for start, end in ((a, b), (b, c), (c, a)):
        value = _edge(start, end, point)
        if value > 0.0:
            continue
        if value == 0.0 and _is_top_left_edge(start, end):
            continue
        return False
    return True


def interpolate3(weights: Vector3, values: Sequence[Vector3]) -> Vector3:
    """Barycentric interpolation of three 3-vectors."""
    return tuple(
        weights[0] * values[0][index]
        + weights[1] * values[1][index]
        + weights[2] * values[2][index]
        for index in range(3)
    )


def transform4(matrix: Matrix4, point: Vector4) -> Vector4:
    """Row-major 4x4 times a 4-vector."""
    return tuple(
        matrix[row * 4 + 0] * point[0]
        + matrix[row * 4 + 1] * point[1]
        + matrix[row * 4 + 2] * point[2]
        + matrix[row * 4 + 3] * point[3]
        for row in range(4)
    )


def multiply4(left: Matrix4, right: Matrix4) -> list[float]:
    """Row-major 4x4 times 4x4."""
    return [
        sum(left[row * 4 + k] * right[k * 4 + column] for k in range(4))
        for row in range(4)
        for column in range(4)
    ]


def clip_to_pixel(clip: Vector4, width: int, height: int) -> Vector2 | None:
    """Pixel-centre coordinates for a clip-space point, or None if it is not visible.

    `clip.w` must be positive -- a point at or behind the camera plane has no image
    position -- and every NDC component must lie inside the unit cube. The half-pixel
    subtraction converts a normalized coordinate into a pixel-centre coordinate, so integer
    `(u, v)` names the centre of a pixel rather than its corner.
    """
    if clip[3] <= 0.0 or not all(math.isfinite(component) for component in clip):
        return None

    ndc_x = clip[0] / clip[3]
    ndc_y = clip[1] / clip[3]
    ndc_z = clip[2] / clip[3]
    if not all(-1.0 <= component <= 1.0 for component in (ndc_x, ndc_y, ndc_z)):
        return None

    u = (ndc_x * 0.5 + 0.5) * width - 0.5
    v = (1.0 - (ndc_y * 0.5 + 0.5)) * height - 0.5
    return (u, v)


def within_pixel_bounds(u: float, v: float, width: int, height: int) -> bool:
    """The half-open sampling domain of a pixel-centre coordinate."""
    return -0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5


def faces_camera(normal: Vector3, world_point: Vector3, camera_origin: Vector3) -> bool:
    """Whether a surface normal faces the camera at that point."""
    view = tuple(camera_origin[index] - world_point[index] for index in range(3))
    dot = sum(normal[index] * view[index] for index in range(3))
    return dot > 0.0


def depth_tolerance(expected_depth: float) -> float:
    """The binding absolute-plus-relative depth agreement window."""
    return max(
        DEPTH_ABSOLUTE_TOLERANCE_METERS,
        abs(expected_depth) * DEPTH_RELATIVE_TOLERANCE,
    )


def depth_agrees(expected_depth: float, observed_depth: float) -> bool:
    """Whether an interpolated distance matches the Z pass at that texel."""
    if not (math.isfinite(expected_depth) and math.isfinite(observed_depth)):
        return False
    return abs(expected_depth - observed_depth) <= depth_tolerance(expected_depth)


def depth_occluded(expected_depth: float, neighbourhood: Sequence[float]) -> bool:
    """Whether a texel lies behind the surface the depth pass recorded.

    The reference is the nearest finite neighbour, because a sample between pixel centres
    is legitimately nearer than any single one of them, and the neighbourhood's own depth
    span is added to the tolerance, because a surface slanted away from the camera changes
    depth across one pixel by more than the binding tolerance at modest resolutions. A
    genuinely hidden texel sits behind its occluder by far more than one pixel of local
    gradient, so it is still rejected.
    """
    finite = [value for value in neighbourhood if math.isfinite(value)]
    if not finite or not math.isfinite(expected_depth):
        return True
    nearest = min(finite)
    allowance = depth_tolerance(expected_depth) + (max(finite) - nearest)
    return (expected_depth - nearest) > allowance


def ray_epsilon(bounding_box_diagonal_meters: float) -> float:
    """The self-hit epsilon for a first-hit occlusion test."""
    scaled = abs(bounding_box_diagonal_meters) * RAY_EPSILON_SCALE
    return min(max(scaled, RAY_EPSILON_MIN_METERS), RAY_EPSILON_MAX_METERS)


def euclidean_distance(a: Vector3, b: Vector3) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))
