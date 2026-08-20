"""A source file is read-only and byte-identical across a run."""

from pathlib import Path

import pytest
from asset_mania_pipeline import (
    SourceChanged,
    fingerprint_source,
    sha256_bytes,
    sha256_file,
    verify_source_unchanged,
)


def test_file_and_bytes_digests_agree(source_scene: Path) -> None:
    assert sha256_file(source_scene) == sha256_bytes(source_scene.read_bytes())


def test_digest_streams_a_file_larger_than_one_chunk(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4096
    path = tmp_path / "large.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


def test_unchanged_source_passes_verification(source_scene: Path) -> None:
    before = fingerprint_source(source_scene)
    verify_source_unchanged(source_scene, before)
    assert fingerprint_source(source_scene) == before


def test_content_change_is_detected(source_scene: Path) -> None:
    before = fingerprint_source(source_scene)
    source_scene.write_bytes(b"BLENDER-v502" + bytes(65))
    with pytest.raises(SourceChanged, match="SOURCE_CHANGED_DURING_RUN"):
        verify_source_unchanged(source_scene, before)


def test_replacement_by_a_same_size_file_is_detected(source_scene: Path, tmp_path: Path) -> None:
    before = fingerprint_source(source_scene)
    replacement = tmp_path / "replacement.blend"
    replacement.write_bytes(b"BLENDER-v502" + bytes(64))
    replacement.replace(source_scene)
    with pytest.raises(SourceChanged, match="SOURCE_CHANGED_DURING_RUN"):
        verify_source_unchanged(source_scene, before)


def test_a_removed_source_is_detected(source_scene: Path) -> None:
    before = fingerprint_source(source_scene)
    source_scene.unlink()
    with pytest.raises(SourceChanged, match="SOURCE_CHANGED_DURING_RUN"):
        verify_source_unchanged(source_scene, before)


def test_fingerprint_records_only_portable_and_local_identity_fields(source_scene: Path) -> None:
    fingerprint = fingerprint_source(source_scene)
    assert fingerprint.sha256 == sha256_file(source_scene)
    assert fingerprint.byte_size == source_scene.stat().st_size
    assert str(source_scene) not in repr(fingerprint)
    assert source_scene.name not in repr(fingerprint)
