"""Atomic, no-replace stage publication."""

import json
import os
from pathlib import Path

import pytest
from asset_mania_pipeline import (
    OutputCollision,
    PathEscape,
    StageState,
    StageStore,
    StorageUnavailable,
)

STAGE = "condition"
RUN_ID = "run-condition-1"
CREATED_AT = "2026-08-19T09:10:00Z"


def _manifest(run_id: str = RUN_ID) -> dict:
    return {"schema_id": "asset-mania/run-manifest", "run_id": run_id, "stage": STAGE}


def _report() -> dict:
    return {"result": {"status": "succeeded"}}


@pytest.fixture
def store(tmp_path: Path) -> StageStore:
    return StageStore(tmp_path / "runs")


def test_publish_creates_a_new_run_directory_with_canonical_records(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    published = store.publish(run, manifest=_manifest(), report=_report())

    assert published.is_dir()
    assert published.name == f"20260819T091000Z-{RUN_ID}"
    assert json.loads((published / "manifest.json").read_text()) == _manifest()
    assert json.loads((published / "report.json").read_text()) == _report()
    assert (published / "logs").is_dir()
    assert (published / "manifest.json").read_bytes().endswith(b"}\n")


def test_records_are_written_with_restrictive_modes(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    published = store.publish(run, manifest=_manifest(), report=_report())

    assert os.stat(published).st_mode & 0o777 == 0o700
    assert os.stat(published / "logs").st_mode & 0o777 == 0o700
    assert os.stat(published / "manifest.json").st_mode & 0o777 == 0o600


def test_staging_disappears_after_publication(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    staging = run.staging
    assert staging.is_dir()
    store.publish(run, manifest=_manifest(), report=_report())
    assert not staging.exists()


def test_publication_never_replaces_an_existing_run(store: StageStore) -> None:
    first = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    published = store.publish(first, manifest=_manifest(), report=_report())
    marker = published / "manifest.json"
    original = marker.read_bytes()

    second = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    with pytest.raises(OutputCollision, match="OUTPUT_COLLISION"):
        store.publish(second, manifest=_manifest(), report=_report())
    assert marker.read_bytes() == original


def test_a_failed_publication_leaves_no_partial_directory(store: StageStore) -> None:
    first = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    store.publish(first, manifest=_manifest(), report=_report())

    second = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    staging = second.staging
    with pytest.raises(OutputCollision):
        store.publish(second, manifest=_manifest(), report=_report())

    assert not staging.exists()
    assert [path.name for path in (store.output_parent).iterdir()] == [f"20260819T091000Z-{RUN_ID}"]


def test_fail_publishes_a_terminal_run_rather_than_discarding_it(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    manifest = {**_manifest(), "result": {"status": "failed"}}
    published = store.fail(run, manifest=manifest, report={"result": {"status": "failed"}})

    assert json.loads((published / "manifest.json").read_text()) == manifest
    assert run.state is StageState.FAILED


def test_a_run_reaches_exactly_one_terminal_state(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    assert run.state is StageState.OPEN
    store.publish(run, manifest=_manifest(), report=_report())
    assert run.state is StageState.PUBLISHED

    with pytest.raises(ValueError, match="terminal"):
        store.publish(run, manifest=_manifest(), report=_report())
    with pytest.raises(ValueError, match="terminal"):
        store.fail(run, manifest=_manifest(), report=_report())


def test_state_never_rolls_back_after_failure(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    store.fail(run, manifest=_manifest(), report=_report())
    with pytest.raises(ValueError, match="terminal"):
        store.publish(run, manifest=_manifest(), report=_report())
    assert run.state is StageState.FAILED


def test_staging_writes_stay_inside_the_run(store: StageStore) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    inside = run.stage_path("passes/beauty.exr")
    assert inside.parent.is_dir()
    assert inside.is_relative_to(run.staging)

    for escape in ("../escape.png", "/tmp/escape.png", "passes/../../escape.png"):
        with pytest.raises(PathEscape):
            run.stage_path(escape)


def test_staging_rejects_a_symlink_that_leaves_the_run(store: StageStore, tmp_path: Path) -> None:
    run = store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run.staging / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathEscape):
        run.stage_path("linked/escape.png")


def test_unwritable_output_parent_is_a_storage_failure(tmp_path: Path) -> None:
    parent = tmp_path / "locked"
    parent.mkdir(mode=0o500)
    try:
        store = StageStore(parent / "runs")
        with pytest.raises(StorageUnavailable, match="OUTPUT_STORAGE_UNAVAILABLE"):
            store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)
    finally:
        parent.chmod(0o700)


def test_output_parent_that_is_a_file_is_a_storage_failure(tmp_path: Path) -> None:
    occupied = tmp_path / "runs"
    occupied.write_text("not a directory\n")
    store = StageStore(occupied)
    with pytest.raises(StorageUnavailable, match="OUTPUT_STORAGE_UNAVAILABLE"):
        store.begin(stage=STAGE, run_id=RUN_ID, created_at=CREATED_AT)


def test_run_identifier_must_be_portable(store: StageStore) -> None:
    for run_id in ("run/condition", "../run", "run condition", "", "r" * 65):
        with pytest.raises(ValueError, match="run id"):
            store.begin(stage=STAGE, run_id=run_id, created_at=CREATED_AT)


def test_created_at_must_be_an_rfc3339_utc_timestamp(store: StageStore) -> None:
    for created_at in ("2026-08-19 09:10:00", "2026-08-19T09:10:00+09:00", "not-a-time"):
        with pytest.raises(ValueError, match="created_at"):
            store.begin(stage=STAGE, run_id=RUN_ID, created_at=created_at)


def test_two_stages_of_the_same_run_identifier_never_share_a_directory(
    store: StageStore,
) -> None:
    first = store.begin(stage="condition", run_id="run-a", created_at=CREATED_AT)
    published = store.publish(first, manifest=_manifest("run-a"), report=_report())
    second = store.begin(stage="bake", run_id="run-b", created_at="2026-08-19T09:45:00Z")
    other = store.publish(second, manifest=_manifest("run-b"), report=_report())
    assert published != other
