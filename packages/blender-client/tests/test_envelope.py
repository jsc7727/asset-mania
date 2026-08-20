"""The private envelope is 0700/0600, contained, and always deleted."""

import json
import os
from pathlib import Path

import pytest
from asset_mania_blender_client import PrivateEnvelope
from asset_mania_pipeline import PathEscape

PRIVATE_REQUEST = {
    "request_id": "request-preflight-1",
    "operation": "preflight",
    "source_path": "/Users/example/scenes/private-character.blend",
    "target_name": "Body_LOD0",
    "camera_name": "Camera_Main",
}


def test_the_envelope_directory_is_private(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        assert envelope.directory is not None
        assert os.stat(envelope.directory).st_mode & 0o777 == 0o700
        assert envelope.directory.is_relative_to(staging.resolve())


def test_the_request_file_is_private_and_holds_the_private_inputs(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        path = envelope.write_request(PRIVATE_REQUEST)
        assert os.stat(path).st_mode & 0o777 == 0o600
        assert json.loads(path.read_text()) == PRIVATE_REQUEST


def test_the_envelope_is_deleted_on_success(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        directory = envelope.directory
        envelope.write_request(PRIVATE_REQUEST)
    assert directory is not None
    assert not directory.exists()


def test_the_envelope_is_deleted_even_when_the_body_raises(staging: Path) -> None:
    directory: Path | None = None
    with (
        pytest.raises(RuntimeError, match="worker exploded"),
        PrivateEnvelope(staging) as envelope,
    ):
        directory = envelope.directory
        envelope.write_request(PRIVATE_REQUEST)
        raise RuntimeError("worker exploded")
    assert directory is not None
    assert not directory.exists()


def test_no_private_input_survives_the_envelope(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        envelope.write_request(PRIVATE_REQUEST)

    remaining = "".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in staging.rglob("*")
        if path.is_file()
    )
    for private in ("private-character.blend", "Body_LOD0", "Camera_Main"):
        assert private not in remaining


def test_a_closed_envelope_refuses_to_hand_out_paths(staging: Path) -> None:
    envelope = PrivateEnvelope(staging)
    with pytest.raises(RuntimeError, match="closed"):
        _ = envelope.request_path

    with PrivateEnvelope(staging) as opened:
        pass
    with pytest.raises(RuntimeError, match="closed"):
        _ = opened.response_path


def test_the_request_is_never_silently_overwritten(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        envelope.write_request(PRIVATE_REQUEST)
        with pytest.raises(FileExistsError):
            envelope.write_request(PRIVATE_REQUEST)


def test_an_absent_staging_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(OSError), PrivateEnvelope(tmp_path / "absent"):
        pass


def test_a_staging_root_that_is_a_symlink_out_of_the_tree_is_refused(tmp_path: Path) -> None:
    """A staging root may be a symlink, but only to a real directory it can own."""
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "staging-link"
    link.symlink_to(outside, target_is_directory=True)

    with PrivateEnvelope(link) as envelope:
        assert envelope.directory is not None
        # The envelope resolves through the link and stays under the resolved anchor.
        assert envelope.directory.resolve().is_relative_to(outside.resolve())


def test_the_response_path_sits_beside_the_request(staging: Path) -> None:
    with PrivateEnvelope(staging) as envelope:
        assert envelope.response_path.parent == envelope.request_path.parent
        assert envelope.response_path != envelope.request_path
        assert not envelope.response_path.exists()


def test_two_envelopes_never_share_a_directory(staging: Path) -> None:
    with PrivateEnvelope(staging) as first, PrivateEnvelope(staging) as second:
        assert first.directory != second.directory


def test_path_escape_is_the_declared_containment_failure() -> None:
    assert issubclass(PathEscape, Exception)
