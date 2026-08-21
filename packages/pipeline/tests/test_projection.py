"""Reference camera and raster conventions, pinned by analytic vectors."""

import math

import pytest
from asset_mania_pipeline import (
    DEPTH_ABSOLUTE_TOLERANCE_METERS,
    DEPTH_RELATIVE_TOLERANCE,
    RAY_EPSILON_MAX_METERS,
    RAY_EPSILON_MIN_METERS,
    barycentric,
    clip_to_pixel,
    depth_agrees,
    depth_tolerance,
    euclidean_distance,
    faces_camera,
    interpolate3,
    multiply4,
    owns_texel,
    ray_epsilon,
    signed_area,
    texel_center,
    transform4,
    within_pixel_bounds,
)

IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


# --- Texel centres ---------------------------------------------------------------


def test_a_texel_centre_is_offset_by_half_and_flips_v() -> None:
    assert texel_center(0, 0, 4, 4) == (0.125, 0.875)
    assert texel_center(3, 3, 4, 4) == (0.875, 0.125)


def test_texel_centres_span_the_unit_square_without_touching_its_edges() -> None:
    for x in range(8):
        for y in range(8):
            u, v = texel_center(x, y, 8, 8)
            assert 0.0 < u < 1.0
            assert 0.0 < v < 1.0


def test_the_centre_column_and_row_are_symmetric() -> None:
    assert texel_center(1, 0, 4, 4)[0] == pytest.approx(1.0 - texel_center(2, 0, 4, 4)[0])
    assert texel_center(0, 1, 4, 4)[1] == pytest.approx(1.0 - texel_center(0, 2, 4, 4)[1])


# --- Barycentric and fill rule ----------------------------------------------------

TRIANGLE = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


def test_barycentric_weights_sum_to_one_inside_a_triangle() -> None:
    weights = barycentric((0.25, 0.25), TRIANGLE)
    assert weights is not None
    assert sum(weights) == pytest.approx(1.0)
    assert all(weight >= 0.0 for weight in weights)


def test_barycentric_recovers_each_vertex() -> None:
    for index, corner in enumerate(TRIANGLE):
        weights = barycentric(corner, TRIANGLE)
        assert weights is not None
        assert weights[index] == pytest.approx(1.0)


def test_a_degenerate_triangle_has_no_barycentric_coordinates() -> None:
    assert barycentric((0.5, 0.0), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))) is None
    assert barycentric((0.0, 0.0), ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0))) is None


def test_signed_area_reports_winding() -> None:
    assert signed_area(TRIANGLE) > 0
    assert signed_area((TRIANGLE[0], TRIANGLE[2], TRIANGLE[1])) < 0


def test_a_point_inside_is_owned_regardless_of_winding() -> None:
    reversed_triangle = (TRIANGLE[0], TRIANGLE[2], TRIANGLE[1])
    assert owns_texel((0.25, 0.25), TRIANGLE)
    assert owns_texel((0.25, 0.25), reversed_triangle)


def test_a_point_outside_is_not_owned() -> None:
    assert not owns_texel((0.9, 0.9), TRIANGLE)
    assert not owns_texel((-0.1, 0.5), TRIANGLE)


def test_a_degenerate_triangle_owns_nothing() -> None:
    assert not owns_texel((0.5, 0.0), ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))


def test_a_shared_edge_belongs_to_exactly_one_triangle() -> None:
    """The whole point of the fill rule: total, disjoint ownership."""
    lower = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    upper = ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    shared_points = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
    for point in shared_points:
        owners = [triangle for triangle in (lower, upper) if owns_texel(point, triangle)]
        assert len(owners) == 1, point


def test_every_texel_of_a_split_square_is_owned_exactly_once() -> None:
    lower = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    upper = ((1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    counts = []
    for x in range(16):
        for y in range(16):
            centre = texel_center(x, y, 16, 16)
            counts.append(sum(owns_texel(centre, triangle) for triangle in (lower, upper)))
    assert set(counts) == {1}


# --- Matrices ---------------------------------------------------------------------


def test_transform_by_identity_is_a_no_op() -> None:
    assert transform4(IDENTITY, (1.0, 2.0, 3.0, 1.0)) == (1.0, 2.0, 3.0, 1.0)


def test_matrix_multiply_is_row_major() -> None:
    translate = [1, 0, 0, 5, 0, 1, 0, 6, 0, 0, 1, 7, 0, 0, 0, 1]
    assert multiply4(IDENTITY, translate) == translate
    assert transform4(translate, (0.0, 0.0, 0.0, 1.0)) == (5.0, 6.0, 7.0, 1.0)


def test_interpolation_of_three_vectors_is_barycentric() -> None:
    values = ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 4.0, 0.0))
    assert interpolate3((0.5, 0.5, 0.0), values) == (1.0, 0.0, 0.0)
    assert interpolate3((0.0, 0.0, 1.0), values) == (0.0, 4.0, 0.0)


# --- Clip to pixel ----------------------------------------------------------------


def test_the_clip_centre_maps_to_the_frame_centre() -> None:
    assert clip_to_pixel((0.0, 0.0, 0.0, 1.0), 64, 64) == (31.5, 31.5)


def test_the_ndc_corners_map_to_the_outer_pixel_centres() -> None:
    assert clip_to_pixel((-1.0, 1.0, 0.0, 1.0), 64, 64) == (-0.5, -0.5)
    assert clip_to_pixel((1.0, -1.0, 0.0, 1.0), 64, 64) == (63.5, 63.5)


def test_v_is_flipped_so_positive_ndc_y_is_the_top_row() -> None:
    top = clip_to_pixel((0.0, 0.5, 0.0, 1.0), 64, 64)
    bottom = clip_to_pixel((0.0, -0.5, 0.0, 1.0), 64, 64)
    assert top[1] < bottom[1]


def test_a_point_at_or_behind_the_camera_has_no_image_position() -> None:
    assert clip_to_pixel((0.0, 0.0, 0.0, 0.0), 64, 64) is None
    assert clip_to_pixel((0.0, 0.0, 0.0, -1.0), 64, 64) is None


@pytest.mark.parametrize(
    "clip",
    [
        (1.5, 0.0, 0.0, 1.0),
        (0.0, 1.5, 0.0, 1.0),
        (0.0, 0.0, 1.5, 1.0),
        (-1.5, 0.0, 0.0, 1.0),
    ],
)
def test_a_point_outside_the_unit_cube_is_rejected(clip) -> None:
    assert clip_to_pixel(clip, 64, 64) is None


def test_a_non_finite_clip_point_is_rejected() -> None:
    assert clip_to_pixel((float("nan"), 0.0, 0.0, 1.0), 64, 64) is None
    assert clip_to_pixel((0.0, float("inf"), 0.0, 1.0), 64, 64) is None


def test_the_perspective_divide_is_applied() -> None:
    assert clip_to_pixel((1.0, 0.0, 0.0, 2.0), 64, 64) == clip_to_pixel(
        (0.5, 0.0, 0.0, 1.0), 64, 64
    )


# --- Sampling bounds --------------------------------------------------------------


def test_the_sampling_domain_is_half_open() -> None:
    assert within_pixel_bounds(-0.5, -0.5, 64, 64)
    assert within_pixel_bounds(63.4999, 63.4999, 64, 64)
    assert not within_pixel_bounds(63.5, 0.0, 64, 64)
    assert not within_pixel_bounds(0.0, 63.5, 64, 64)
    assert not within_pixel_bounds(-0.5001, 0.0, 64, 64)


# --- Facing -----------------------------------------------------------------------


def test_a_normal_pointing_at_the_camera_faces_it() -> None:
    assert faces_camera((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 5.0))


def test_a_normal_pointing_away_does_not() -> None:
    assert not faces_camera((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 5.0))


def test_a_normal_exactly_edge_on_does_not_face_the_camera() -> None:
    assert not faces_camera((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 5.0))


# --- Depth agreement ---------------------------------------------------------------


def test_the_tolerance_is_the_larger_of_absolute_and_relative() -> None:
    assert depth_tolerance(0.0) == DEPTH_ABSOLUTE_TOLERANCE_METERS
    assert depth_tolerance(100.0) == pytest.approx(100.0 * DEPTH_RELATIVE_TOLERANCE)
    assert depth_tolerance(0.1) == DEPTH_ABSOLUTE_TOLERANCE_METERS


def test_depth_within_tolerance_agrees() -> None:
    assert depth_agrees(7.0, 7.0)
    assert depth_agrees(7.0, 7.0 + depth_tolerance(7.0) * 0.9)


def test_depth_beyond_tolerance_disagrees() -> None:
    assert not depth_agrees(7.0, 7.0 + depth_tolerance(7.0) * 2.0)
    assert not depth_agrees(7.0, 1e10)


def test_a_non_finite_depth_never_agrees() -> None:
    assert not depth_agrees(7.0, float("inf"))
    assert not depth_agrees(float("nan"), 7.0)


# --- Ray epsilon --------------------------------------------------------------------


def test_the_ray_epsilon_scales_with_the_bounding_box_and_is_clamped() -> None:
    assert ray_epsilon(0.0) == RAY_EPSILON_MIN_METERS
    assert ray_epsilon(1e12) == RAY_EPSILON_MAX_METERS
    middle = ray_epsilon(1e4)
    assert RAY_EPSILON_MIN_METERS <= middle <= RAY_EPSILON_MAX_METERS


def test_euclidean_distance_is_the_norm() -> None:
    assert euclidean_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)
    assert math.isclose(euclidean_distance((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)), 0.0)
