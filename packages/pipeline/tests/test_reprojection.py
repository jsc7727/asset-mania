"""Reference reprojection over synthetic arrays, including coverage semantics."""

import pytest
from asset_mania_pipeline import (
    Sampler,
    Triangle,
    dilate_same_island,
    euclidean_distance,
    linear_to_srgb,
    reproject,
    srgb_to_linear,
)

VIEW = 8
ATLAS = 8
CAMERA_ORIGIN = (0.0, 0.0, 5.0)

#: An orthographic-style clip matrix over x,y in [-1,1] at z = 0, looking down -Z.
#: Row-major; w is fixed to 1 so the mapping is exact and hand-checkable.
ORTHO_CLIP = [
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    -1,
    0,
    0,
    0,
    0,
    1,
]


def _uniform(value, channels: int = 1, size: int = VIEW) -> Sampler:
    return Sampler(
        width=size,
        height=size,
        pixels=[[[value] * channels for _ in range(size)] for _ in range(size)],
    )


def _quad_triangles(depth: float = 0.0) -> list[Triangle]:
    """A unit square filling UV 0..1 and world x,y in [-1,1] at a fixed z."""
    lower = Triangle(
        polygon_index=0,
        uv=((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        world=((-1.0, -1.0, depth), (1.0, -1.0, depth), (-1.0, 1.0, depth)),
        normal=((0.0, 0.0, 1.0),) * 3,
    )
    upper = Triangle(
        polygon_index=1,
        uv=((1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        world=((1.0, -1.0, depth), (1.0, 1.0, depth), (-1.0, 1.0, depth)),
        normal=((0.0, 0.0, 1.0),) * 3,
    )
    return [lower, upper]


def _true_depth(size: int = VIEW, plane_z: float = 0.0) -> Sampler:
    """The Z pass a real render would produce for the planar quad.

    A uniform depth would not do: the pass records the Euclidean distance from the camera,
    which grows toward the frame edges, and the binding tolerance is tight enough that a
    flat value is correctly rejected.
    """
    rows = []
    for v in range(size):
        row = []
        for u in range(size):
            ndc_x = ((u + 0.5) / size - 0.5) * 2.0
            ndc_y = 1.0 - 2.0 * (v + 0.5) / size
            world = (ndc_x, ndc_y, plane_z)
            row.append([euclidean_distance(CAMERA_ORIGIN, world)])
        rows.append(row)
    return Sampler(width=size, height=size, pixels=rows)


def _reproject(**overrides):
    arguments = {
        "triangles": _quad_triangles(),
        "atlas_width": ATLAS,
        "atlas_height": ATLAS,
        "world_to_clip": ORTHO_CLIP,
        "camera_origin": CAMERA_ORIGIN,
        "view_width": VIEW,
        "view_height": VIEW,
        "colour": _uniform(1.0, channels=3),
        "mask": _uniform(1.0),
        "depth": _true_depth(),
    }
    arguments.update(overrides)
    return reproject(**arguments)


# --- Transfer functions ------------------------------------------------------------


def test_the_srgb_transfer_round_trips() -> None:
    for value in (0.0, 0.01, 0.04045, 0.2, 0.5, 1.0):
        assert srgb_to_linear(linear_to_srgb(value)) == pytest.approx(value, abs=1e-9)


def test_srgb_decoding_uses_the_linear_segment_near_black() -> None:
    assert srgb_to_linear(0.04) == pytest.approx(0.04 / 12.92)


def test_srgb_decoding_darkens_midtones() -> None:
    assert srgb_to_linear(0.5) < 0.5


# --- Sampling ----------------------------------------------------------------------


def test_nearest_sampling_snaps_to_the_pixel_centre() -> None:
    pixels = [[[float(x + y * 4)] for x in range(4)] for y in range(4)]
    sampler = Sampler(width=4, height=4, pixels=pixels)
    assert sampler.nearest(0.0, 0.0)[0] == 0.0
    assert sampler.nearest(0.4, 0.0)[0] == 0.0
    assert sampler.nearest(0.6, 0.0)[0] == 1.0
    assert sampler.nearest(3.0, 3.0)[0] == 15.0


def test_bilinear_sampling_interpolates_between_pixel_centres() -> None:
    pixels = [[[0.0], [1.0]], [[0.0], [1.0]]]
    sampler = Sampler(width=2, height=2, pixels=pixels)
    assert sampler.bilinear(0.0, 0.0)[0] == pytest.approx(0.0)
    assert sampler.bilinear(0.5, 0.0)[0] == pytest.approx(0.5)
    assert sampler.bilinear(1.0, 0.0)[0] == pytest.approx(1.0)


def test_sampling_clamps_at_the_border_rather_than_wrapping() -> None:
    pixels = [[[7.0]]]
    sampler = Sampler(width=1, height=1, pixels=pixels)
    assert sampler.nearest(-5.0, -5.0)[0] == 7.0
    assert sampler.bilinear(5.0, 5.0)[0] == pytest.approx(7.0)


# --- A fully visible quad ------------------------------------------------------------


def test_a_fully_visible_quad_covers_every_texel_exactly_once() -> None:
    result = _reproject()
    assert result.observed_coverage == pytest.approx(1.0)
    assert all(all(row) for row in result.observed)
    assert result.rejected["outside_mask"] == 0
    assert result.rejected["depth_mismatch"] == 0


def test_every_covered_texel_records_its_source_view() -> None:
    result = _reproject(view_label="view-7")
    labels = {label for row in result.source_view_label for label in row}
    assert labels == {"view-7"}


def test_colour_is_stored_in_linear_light() -> None:
    result = _reproject(colour=_uniform(0.5, channels=3))
    assert result.colour[0][0][0] == pytest.approx(srgb_to_linear(0.5))


def test_decoding_can_be_disabled_for_already_linear_input() -> None:
    result = _reproject(colour=_uniform(0.5, channels=3), decode_srgb=False)
    assert result.colour[0][0][0] == pytest.approx(0.5)


def test_final_alpha_is_written_explicitly() -> None:
    result = _reproject()
    assert result.alpha(0, 0) == 255
    empty = _reproject(mask=_uniform(0.0))
    assert empty.alpha(0, 0) == 0


# --- Rejections ----------------------------------------------------------------------


def test_a_texel_outside_the_mask_is_not_covered() -> None:
    result = _reproject(mask=_uniform(0.0))
    assert result.observed_coverage == 0.0
    assert result.rejected["outside_mask"] == ATLAS * ATLAS


def test_a_backfacing_triangle_is_rejected() -> None:
    away = [
        Triangle(
            polygon_index=triangle.polygon_index,
            uv=triangle.uv,
            world=triangle.world,
            normal=((0.0, 0.0, -1.0),) * 3,
        )
        for triangle in _quad_triangles()
    ]
    result = _reproject(triangles=away)
    assert result.observed_coverage == 0.0
    assert result.rejected["backfacing"] == ATLAS * ATLAS


def test_a_flat_depth_pass_is_rejected_because_real_depth_varies() -> None:
    result = _reproject(depth=_uniform(1.0))
    assert result.observed_coverage == 0.0
    assert result.rejected["depth_mismatch"] == ATLAS * ATLAS


def test_a_non_finite_depth_pass_rejects_every_texel() -> None:
    result = _reproject(depth=_uniform(float("inf")))
    assert result.observed_coverage == 0.0
    assert result.rejected["depth_mismatch"] == ATLAS * ATLAS


def test_geometry_outside_the_clip_volume_is_rejected() -> None:
    far = _quad_triangles(depth=0.0)
    shifted = [
        Triangle(
            polygon_index=triangle.polygon_index,
            uv=triangle.uv,
            world=tuple((x + 10.0, y, z) for x, y, z in triangle.world),
            normal=triangle.normal,
        )
        for triangle in far
    ]
    result = _reproject(triangles=shifted)
    assert result.observed_coverage == 0.0
    assert result.rejected["outside_clip"] == ATLAS * ATLAS


def test_an_occluded_texel_is_rejected_by_the_first_hit_test() -> None:
    result = _reproject(first_hit=lambda world, normal: False)
    assert result.observed_coverage == 0.0
    assert result.rejected["occluded"] == ATLAS * ATLAS


def test_a_degenerate_uv_triangle_covers_nothing() -> None:
    degenerate = [
        Triangle(
            polygon_index=0,
            uv=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
            world=((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (3.0, -1.0, 0.0)),
            normal=((0.0, 0.0, 1.0),) * 3,
        )
    ]
    result = _reproject(triangles=degenerate)
    assert result.observed_coverage == 0.0


def test_no_uncovered_texel_is_given_a_colour() -> None:
    result = _reproject(mask=_uniform(0.0))
    assert all(all(texel == [0.0, 0.0, 0.0] for texel in row) for row in result.colour)


# --- Determinism ----------------------------------------------------------------------


def test_the_result_does_not_depend_on_triangle_iteration_order() -> None:
    forward = _reproject(triangles=_quad_triangles())
    backward = _reproject(triangles=list(reversed(_quad_triangles())))
    assert forward.colour == backward.colour
    assert forward.observed == backward.observed
    assert forward.source_view_label == backward.source_view_label


def test_repeating_the_reprojection_is_bit_identical() -> None:
    first = _reproject()
    second = _reproject()
    assert first.colour == second.colour
    assert first.observed == second.observed


def test_a_partial_mask_covers_exactly_the_visible_half() -> None:
    half = Sampler(
        width=VIEW,
        height=VIEW,
        pixels=[[[1.0 if x < VIEW // 2 else 0.0] for x in range(VIEW)] for _ in range(VIEW)],
    )
    result = _reproject(mask=half)
    assert result.observed_coverage == pytest.approx(0.5)
    assert result.rejected["outside_mask"] == ATLAS * ATLAS // 2


# --- Seam padding ---------------------------------------------------------------------


def test_padding_extends_rgb_without_promoting_observed_coverage() -> None:
    half = Sampler(
        width=VIEW,
        height=VIEW,
        pixels=[[[1.0 if x < VIEW // 2 else 0.0] for x in range(VIEW)] for _ in range(VIEW)],
    )
    result = _reproject(mask=half, colour=_uniform(1.0, channels=3))
    observed_before = result.observed_coverage

    dilated = dilate_same_island(result, margin=1)
    assert dilated.observed_coverage == observed_before
    assert dilated.padded_coverage > observed_before
    assert dilated.alpha(VIEW - 1, 0) == 0


def test_a_zero_margin_changes_nothing() -> None:
    result = _reproject()
    before = [row[:] for row in result.padded]
    assert dilate_same_island(result, margin=0).padded == before


def test_padding_never_invents_colour_for_a_fully_uncovered_atlas() -> None:
    result = _reproject(mask=_uniform(0.0))
    dilated = dilate_same_island(result, margin=3)
    assert dilated.observed_coverage == 0.0
    assert dilated.padded_coverage == 0.0
    assert all(all(texel == [0.0, 0.0, 0.0] for texel in row) for row in dilated.colour)


def test_padding_is_bounded_by_the_margin() -> None:
    single = Sampler(
        width=VIEW,
        height=VIEW,
        pixels=[[[1.0 if (x, y) == (0, 0) else 0.0] for x in range(VIEW)] for y in range(VIEW)],
    )
    result = _reproject(mask=single)
    covered = result.observed_coverage
    dilated = dilate_same_island(result, margin=1)
    assert dilated.observed_coverage == covered
    # One ring around a handful of texels, not the whole atlas.
    assert dilated.padded_coverage < 0.5
