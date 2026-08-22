import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_face_plugin_e2e import (
    CHECKPOINT_BYTES,
    CHECKPOINT_URL,
    DAD_REVISION,
    SOURCE_URL,
    main,
)


def _planned_run(tmp_path: Path) -> Path:
    output = tmp_path / "runs"
    assert (
        main(
            ["plan", "--out", str(output), "--plugin", "dad3dheads-local"],
            now="2026-08-23T00:00:00+00:00",
            id_factory=lambda: "fixedrun",
        )
        == 0
    )
    return output / "20260823T000000Z-fixedrun"


def test_plan_fixes_model_license_runtime_and_no_egress(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    plan = json.loads((run / "plan/plan.json").read_text(encoding="utf-8"))

    assert plan["plugin_revision"] == DAD_REVISION
    assert plan["source_url"] == SOURCE_URL
    assert plan["checkpoint_url"] == CHECKPOINT_URL
    assert plan["checkpoint_expected_bytes"] == CHECKPOINT_BYTES
    assert plan["license"] == "CC-BY-NC-SA-4.0"
    assert plan["commercial_use"] == "forbidden-for-this-profile"
    assert plan["device"] == "cuda"
    assert plan["torch"] == "2.13.0+cu130"
    assert plan["retry_count"] == 0
    assert plan["face_egress"] == "none"
    assert len(plan["plan_sha256"]) == 64


def test_acquire_requires_exact_approval_reference(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)

    with pytest.raises(ValueError, match="fresh acquisition approval is required"):
        main(["acquire", "--run", str(run), "--approval-reference", "yes"])


def test_acquire_records_exact_revision_and_checkpoint(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    checkpoint_bytes = b"checkpoint"

    def fake_git(url: str, revision: str, destination: Path) -> None:
        assert url == SOURCE_URL
        assert revision == DAD_REVISION
        destination.mkdir()
        (destination / "LICENSE").write_text("CC BY-NC-SA 4.0", encoding="utf-8")

    def fake_download(url: str, destination: Path, expected_bytes: int) -> None:
        assert url == CHECKPOINT_URL
        assert expected_bytes == CHECKPOINT_BYTES
        destination.parent.mkdir(parents=True)
        destination.write_bytes(checkpoint_bytes)

    assert (
        main(
            [
                "acquire",
                "--run",
                str(run),
                "--approval-reference",
                "face-plugin-approval-20260823",
            ],
            git_acquirer=fake_git,
            checkpoint_downloader=fake_download,
            revision_reader=lambda _source: DAD_REVISION,
            expected_checkpoint_bytes=len(checkpoint_bytes),
        )
        == 0
    )
    receipt = json.loads((run / "acquisition/receipt.json").read_text(encoding="utf-8"))
    assert receipt["source_revision"] == DAD_REVISION
    assert receipt["checkpoint_bytes"] == len(checkpoint_bytes)
    assert len(receipt["checkpoint_sha256"]) == 64
    assert receipt["redistribution"] == "uncleared"
    assert "clearance" not in receipt


def test_acquire_is_create_only(tmp_path: Path) -> None:
    run = _planned_run(tmp_path)
    (run / "acquisition/source").mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "acquire",
                "--run",
                str(run),
                "--approval-reference",
                "face-plugin-approval-20260823",
            ]
        )
