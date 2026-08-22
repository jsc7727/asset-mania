from pathlib import Path

import numpy as np
import pytest
from asset_mania_engine_dad3dheads.mesh import convert_dad_mesh, inspect_dad_mesh
from PIL import Image


def _tetra_obj(path: Path) -> None:
    path.write_text(
        """
v -1 -1 0
v 1 -1 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
""".lstrip(),
        encoding="utf-8",
    )


def test_inspect_accepts_one_finite_component(tmp_path: Path) -> None:
    obj = tmp_path / "head.obj"
    _tetra_obj(obj)

    result = inspect_dad_mesh(obj)

    assert result.component_count == 1
    assert result.non_manifold_edge_count == 0
    assert result.boundary_loop_count == 0
    assert result.vertex_count == 4
    assert result.triangle_count == 4
    assert result.winding_consistent


def test_inspect_rejects_non_finite_and_fragmented_meshes(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.obj"
    invalid.write_text("v nan 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        inspect_dad_mesh(invalid)

    fragmented = tmp_path / "fragmented.obj"
    fragmented.write_text(
        """
v 0 0 0
v 1 0 0
v 0 1 0
v 10 0 0
v 11 0 0
v 10 1 0
f 1 2 3
f 4 5 6
""".lstrip(),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="one connected component"):
        inspect_dad_mesh(fragmented)


def test_convert_writes_plain_and_front_colored_glbs(tmp_path: Path) -> None:
    obj = tmp_path / "head.obj"
    _tetra_obj(obj)
    projection = tmp_path / "projection.npz"
    np.savez_compressed(
        projection,
        projected_vertices=np.array([[0, 0], [3, 0], [1, 3], [2, 2]], dtype=float),
        image_shape=np.array([4, 4], dtype=np.int64),
    )
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    pixels[0, 3] = (0, 255, 0)
    pixels[3, 1] = (0, 0, 255)
    pixels[2, 2] = (255, 255, 0)
    source = tmp_path / "source.png"
    Image.fromarray(pixels, "RGB").save(source)
    plain = tmp_path / "plain.glb"
    colored = tmp_path / "colored.glb"

    result = convert_dad_mesh(
        obj_path=obj,
        projection_path=projection,
        source_image=source,
        plain_glb=plain,
        colored_glb=colored,
    )

    assert plain.is_file() and plain.stat().st_size > 0
    assert colored.is_file() and colored.stat().st_size > 0
    assert result.component_count == 1
    assert 0.0 < result.observed_color_coverage <= 1.0


def test_convert_is_create_only(tmp_path: Path) -> None:
    obj = tmp_path / "head.obj"
    _tetra_obj(obj)
    projection = tmp_path / "projection.npz"
    np.savez_compressed(
        projection,
        projected_vertices=np.zeros((4, 2)),
        image_shape=np.array([4, 4]),
    )
    source = tmp_path / "source.png"
    Image.new("RGB", (4, 4), "white").save(source)
    plain = tmp_path / "plain.glb"
    plain.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        convert_dad_mesh(
            obj_path=obj,
            projection_path=projection,
            source_image=source,
            plain_glb=plain,
            colored_glb=tmp_path / "colored.glb",
        )
