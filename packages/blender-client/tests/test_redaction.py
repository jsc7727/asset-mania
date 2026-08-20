"""Redaction is the only sanctioned path for worker output into a local log."""

from pathlib import Path

from asset_mania_blender_client import MAX_REDACTED_BYTES, redact

SOURCE = Path("/Users/example/scenes/private-character.blend")
STAGING = Path("/Users/example/runs/20260819T091000Z-run-condition-1")


def test_a_known_private_path_is_removed() -> None:
    redacted = redact(f"Read blend: {SOURCE}\n", private_paths=[SOURCE])
    assert str(SOURCE) not in redacted
    assert "[redacted]" in redacted


def test_a_known_datablock_name_is_removed() -> None:
    redacted = redact("Error: Body_LOD0 has no UV map\n", private_names=["Body_LOD0"])
    assert "Body_LOD0" not in redacted
    assert "has no UV map" in redacted


def test_an_unlisted_absolute_path_is_still_removed() -> None:
    """A path the caller never declared private is redacted anyway."""
    unlisted = "/Users/private-person/elsewhere/output.exr"
    redacted = redact(f"saved {unlisted}\n")
    assert unlisted not in redacted
    assert "[redacted]" in redacted


def test_a_home_directory_is_removed_without_being_listed() -> None:
    redacted = redact(f"cwd {STAGING}\n")
    assert "example" not in redacted


def test_relative_text_survives_redaction() -> None:
    redacted = redact("wrote passes/beauty.exr in 3 samples\n")
    assert "passes/beauty.exr" in redacted
    assert "3 samples" in redacted


def test_output_is_truncated_to_the_limit() -> None:
    redacted = redact("a" * (MAX_REDACTED_BYTES * 2))
    assert len(redacted.encode("utf-8")) <= MAX_REDACTED_BYTES + len("\n[truncated]")
    assert redacted.endswith("[truncated]")


def test_a_short_limit_is_honoured() -> None:
    redacted = redact("abcdefghij", limit=4)
    assert redacted.startswith("abcd")
    assert redacted.endswith("[truncated]")


def test_the_longest_private_string_is_replaced_first() -> None:
    """A datablock name that is a prefix of another must not leave a fragment behind."""
    redacted = redact("Body and Body_LOD0\n", private_names=["Body", "Body_LOD0"])
    assert "Body" not in redacted


def test_invalid_utf8_never_raises() -> None:
    assert redact("valid \udcff text") is not None


def test_an_empty_private_name_is_ignored() -> None:
    assert redact("unchanged text\n", private_names=[""]) == "unchanged text\n"
