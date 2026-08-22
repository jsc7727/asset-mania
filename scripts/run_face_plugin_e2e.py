#!/usr/bin/env python3
"""Plan, acquire, run, convert, and verify one private face-plugin experiment."""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from asset_mania_contracts import canonical_digest, canonical_json
from asset_mania_pipeline import sha256_file

DAD_REVISION = "68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7"
SOURCE_URL = "https://github.com/PinataFarms/DAD-3DHeads.git"
CHECKPOINT_URL = "https://media.pinatafarm.com/public/research/dad-3dheads/dad_3dheads.trcd"
CHECKPOINT_BYTES = 132_711_657
APPROVAL_REFERENCE = "face-plugin-approval-20260823"
RUN_DIRECTORIES = ("plan", "acquisition", "smoke", "inference", "conversion", "verification")


def _parse_time(value: str | datetime | None) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC).replace(microsecond=0)
    if value is None:
        return datetime.now(UTC).replace(microsecond=0)
    return datetime.fromisoformat(value).astimezone(UTC).replace(microsecond=0)


def _load_object(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _verify_seal(document: dict, field: str, label: str) -> None:
    preimage = {key: value for key, value in document.items() if key != field}
    if canonical_digest(preimage) != document.get(field):
        raise ValueError(f"{label} digest does not match its content")


def _run_directory(output_parent: Path, now: datetime, run_id: str) -> Path:
    output_parent.mkdir(parents=True, exist_ok=True)
    run = output_parent / f"{now.strftime('%Y%m%dT%H%M%SZ')}-{run_id}"
    try:
        run.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {run}") from error
    for relative in RUN_DIRECTORIES:
        (run / relative).mkdir()
    return run


def _run_plan(arguments: argparse.Namespace, *, now: datetime, run_id: str) -> int:
    if arguments.plugin != "dad3dheads-local":
        raise ValueError("unsupported face plugin")
    run = _run_directory(arguments.output_parent.resolve(), now, run_id)
    preimage = {
        "schema_id": "asset-mania/private-face-plugin-plan",
        "schema_version": "0.1",
        "plugin": "dad3dheads-local",
        "plugin_revision": DAD_REVISION,
        "source_url": SOURCE_URL,
        "checkpoint_url": CHECKPOINT_URL,
        "checkpoint_expected_bytes": CHECKPOINT_BYTES,
        "license": "CC-BY-NC-SA-4.0",
        "commercial_use": "forbidden-for-this-profile",
        "redistribution": "uncleared",
        "device": "cuda",
        "torch": "2.13.0+cu130",
        "retry_count": 0,
        "face_egress": "none",
        "overwrite_policy": "create_only",
        "planned_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    plan = {**preimage, "plan_sha256": canonical_digest(preimage)}
    (run / "plan/plan.json").write_text(canonical_json(plan), encoding="utf-8")
    print(canonical_json({"run_directory": str(run), "plan_sha256": plan["plan_sha256"]}))
    return 0


def _git_revision(source_root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("acquired DAD source has no readable revision")
    return completed.stdout.strip()


def _acquire_git(url: str, revision: str, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    clone = subprocess.run(
        ["git", "clone", "--no-checkout", url, str(destination)],
        check=False,
        capture_output=True,
    )
    if clone.returncode != 0:
        raise ValueError("DAD source acquisition failed")
    checkout = subprocess.run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        check=False,
        capture_output=True,
    )
    if checkout.returncode != 0:
        raise ValueError("DAD pinned revision checkout failed")


def _download_checkpoint(url: str, destination: Path, expected_bytes: int) -> None:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="GET")
    count = 0
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.geturl() != url:
            raise ValueError("DAD checkpoint redirected outside the approved URL")
        declared = response.headers.get("Content-Length")
        if declared is None or int(declared) != expected_bytes:
            raise ValueError("DAD checkpoint content length differs from the approved plan")
        with destination.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                count += len(chunk)
    if count != expected_bytes:
        raise ValueError("DAD checkpoint byte count differs from the approved plan")


def _run_acquire(
    arguments: argparse.Namespace,
    *,
    git_acquirer: Callable[[str, str, Path], None],
    checkpoint_downloader: Callable[[str, Path, int], None],
    revision_reader: Callable[[Path], str],
    expected_checkpoint_bytes: int,
) -> int:
    if arguments.approval_reference != APPROVAL_REFERENCE:
        raise ValueError("fresh acquisition approval is required")
    run = arguments.run.resolve(strict=True)
    plan = _load_object(run / "plan/plan.json", "face plugin plan")
    _verify_seal(plan, "plan_sha256", "face plugin plan")
    if plan["plugin_revision"] != DAD_REVISION or plan["checkpoint_url"] != CHECKPOINT_URL:
        raise ValueError("face plugin acquisition plan is not the approved DAD plan")
    acquisition = run / "acquisition"
    source = acquisition / "source"
    checkpoint = acquisition / "home/.dad_checkpoints/dad_3dheads.trcd"
    receipt_path = acquisition / "receipt.json"
    if source.exists() or checkpoint.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite face plugin acquisition")
    git_acquirer(SOURCE_URL, DAD_REVISION, source)
    revision = revision_reader(source)
    if revision != DAD_REVISION:
        raise ValueError("acquired DAD source revision mismatch")
    license_path = source / "LICENSE"
    if not license_path.is_file():
        raise ValueError("acquired DAD source has no LICENSE")
    checkpoint_downloader(CHECKPOINT_URL, checkpoint, CHECKPOINT_BYTES)
    size = checkpoint.stat().st_size
    if size != expected_checkpoint_bytes:
        raise ValueError("acquired DAD checkpoint byte count mismatch")
    preimage = {
        "schema_id": "asset-mania/private-face-plugin-acquisition",
        "schema_version": "0.1",
        "plan_sha256": plan["plan_sha256"],
        "approval_reference": APPROVAL_REFERENCE,
        "source_revision": revision,
        "source_license_sha256": sha256_file(license_path),
        "checkpoint_bytes": size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "license": "CC-BY-NC-SA-4.0",
        "commercial_use": "forbidden-for-this-profile",
        "redistribution": "uncleared",
    }
    receipt = {**preimage, "receipt_sha256": canonical_digest(preimage)}
    receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--out", type=Path, dest="output_parent", required=True)
    plan.add_argument("--plugin", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--run", type=Path, required=True)
    acquire.add_argument("--approval-reference", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    now: str | datetime | None = None,
    id_factory: Callable[[], str] | None = None,
    git_acquirer: Callable[[str, str, Path], None] | None = None,
    checkpoint_downloader: Callable[[str, Path, int], None] | None = None,
    revision_reader: Callable[[Path], str] | None = None,
    expected_checkpoint_bytes: int = CHECKPOINT_BYTES,
) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    timestamp = _parse_time(now)
    if arguments.command == "plan":
        return _run_plan(
            arguments,
            now=timestamp,
            run_id=(id_factory or (lambda: secrets.token_hex(4)))(),
        )
    return _run_acquire(
        arguments,
        git_acquirer=git_acquirer or _acquire_git,
        checkpoint_downloader=checkpoint_downloader or _download_checkpoint,
        revision_reader=revision_reader or _git_revision,
        expected_checkpoint_bytes=expected_checkpoint_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
