#!/usr/bin/env python3
"""Plan, acquire, run, convert, and verify one private face-plugin experiment."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from asset_mania_contracts import canonical_digest, canonical_json
from asset_mania_engine_dad3dheads import run_face_plugin
from asset_mania_pipeline import (
    build_face_plugin_request,
    fingerprint_source,
    sha256_file,
    verify_source_unchanged,
    write_face_plugin_request,
)
from PIL import Image, ImageDraw, ImageOps

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


def _runtime_probe(python: Path) -> dict:
    script = (
        "import json,platform,torch;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda_runtime':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'device_type':'cuda' if torch.cuda.is_available() else 'cpu'}))"
    )
    completed = subprocess.run(
        [str(python), "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError("approved CUDA runtime probe failed")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("approved CUDA runtime probe was unreadable") from error
    if not isinstance(result, dict):
        raise TypeError("approved CUDA runtime probe must return an object")
    return result


def _validate_runtime_probe(probe: dict) -> None:
    if (
        probe.get("torch") != "2.13.0+cu130"
        or probe.get("cuda_available") is not True
        or probe.get("device_type") != "cuda"
    ):
        raise ValueError("approved CUDA runtime is unavailable")


def _load_acquisition(run: Path) -> tuple[dict, Path, Path]:
    plan = _load_object(run / "plan/plan.json", "face plugin plan")
    receipt = _load_object(run / "acquisition/receipt.json", "acquisition receipt")
    _verify_seal(plan, "plan_sha256", "face plugin plan")
    _verify_seal(receipt, "receipt_sha256", "acquisition receipt")
    if receipt["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("acquisition receipt does not belong to the plan")
    source_root = run / "acquisition/source"
    isolated_home = run / "acquisition/home"
    checkpoint = isolated_home / ".dad_checkpoints/dad_3dheads.trcd"
    if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from the acquisition receipt")
    return receipt, source_root, isolated_home


def _plugin_environment(source_root: Path, isolated_home: Path) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("SystemRoot", "WINDIR", "COMSPEC", "PATHEXT")
        if key in os.environ
    }
    environment.update(
        {
            "HOME": str(isolated_home),
            "USERPROFILE": str(isolated_home),
            "XDG_CACHE_HOME": str(isolated_home / "xdg-cache"),
            "TORCH_HOME": str(isolated_home / "torch-cache"),
            "HF_HOME": str(isolated_home / "hf-cache"),
            "PYTHONNOUSERSITE": "1",
            "ASSET_MANIA_DAD_SOURCE_ROOT": str(source_root),
            "ASSET_MANIA_DAD_ISOLATED_HOME": str(isolated_home),
        }
    )
    return environment


def _invoke_plugin(
    *,
    run: Path,
    stage: str,
    source_image: Path,
    python: Path,
    plugin_command: Path,
    runtime_probe: Callable[[Path], dict],
    plugin_runner: Callable,
):
    if not python.is_file() or not plugin_command.is_file():
        raise ValueError("explicit face plugin executable is unavailable")
    probe = runtime_probe(python)
    _validate_runtime_probe(probe)
    receipt, source_root, isolated_home = _load_acquisition(run)
    output = run / stage / "plugin-output"
    request_path = run / stage / "request.json"
    result_path = run / stage / "result.json"
    request = build_face_plugin_request(
        plugin="dad3dheads-local",
        plugin_revision=DAD_REVISION,
        source_image=source_image,
        output_directory=output,
        device="cuda",
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )
    write_face_plugin_request(request, request_path)
    result = plugin_runner(
        command=[str(plugin_command)],
        request=request,
        request_path=request_path,
        result_path=result_path,
        timeout_seconds=300,
        environment=_plugin_environment(source_root, isolated_home),
    )
    if result.status != "succeeded":
        raise ValueError(f"face plugin returned {result.status}")
    return probe, receipt, result


def _run_smoke(
    arguments: argparse.Namespace,
    *,
    runtime_probe: Callable[[Path], dict],
    plugin_runner: Callable,
) -> int:
    run = arguments.run.resolve(strict=True)
    source = run / "smoke/source.png"
    if source.exists() or (run / "smoke/record.json").exists():
        raise FileExistsError("refusing to overwrite face plugin smoke")
    image = Image.new("RGB", (256, 256), (238, 238, 238))
    draw = ImageDraw.Draw(image)
    draw.ellipse((48, 24, 208, 232), fill=(196, 160, 132))
    draw.ellipse((86, 98, 106, 112), fill=(30, 30, 30))
    draw.ellipse((150, 98, 170, 112), fill=(30, 30, 30))
    draw.arc((100, 132, 156, 180), 10, 170, fill=(80, 30, 30), width=3)
    image.save(source, format="PNG", compress_level=9)
    probe, receipt, result = _invoke_plugin(
        run=run,
        stage="smoke",
        source_image=source,
        python=arguments.python.resolve(),
        plugin_command=arguments.plugin_command.resolve(),
        runtime_probe=runtime_probe,
        plugin_runner=plugin_runner,
    )
    preimage = {
        "schema_id": "asset-mania/private-face-plugin-smoke",
        "schema_version": "0.1",
        "status": "passed",
        "plugin": "dad3dheads-local",
        "plugin_revision": DAD_REVISION,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "torch": probe["torch"],
        "device": "cuda",
        "vertex_count": result.vertex_count,
        "triangle_count": result.triangle_count,
        "elapsed_seconds": result.elapsed_seconds,
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    (run / "smoke/record.json").write_text(canonical_json(record), encoding="utf-8")
    return 0


def _normalise_source(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    try:
        with Image.open(source) as opened:
            normalized = ImageOps.exif_transpose(opened).convert("RGB")
            normalized.load()
    except OSError as error:
        raise ValueError("private face source is unreadable") from error
    normalized.save(output, format="PNG", compress_level=9)


def _run_private_inference(
    arguments: argparse.Namespace,
    *,
    runtime_probe: Callable[[Path], dict],
    plugin_runner: Callable,
) -> int:
    run = arguments.run.resolve(strict=True)
    smoke = _load_object(run / "smoke/record.json", "smoke record")
    _verify_seal(smoke, "record_sha256", "smoke record")
    receipt, _source_root, _isolated_home = _load_acquisition(run)
    if (
        smoke["plugin_revision"] != DAD_REVISION
        or smoke["checkpoint_sha256"] != receipt["checkpoint_sha256"]
    ):
        raise ValueError("smoke record differs from the acquired plugin")
    source = arguments.source.resolve(strict=True)
    before = fingerprint_source(source)
    normalized = run / "inference/source.png"
    record_path = run / "inference/record.json"
    if record_path.exists():
        raise FileExistsError("refusing to overwrite private face inference")
    _normalise_source(source, normalized)
    _probe, _receipt, result = _invoke_plugin(
        run=run,
        stage="inference",
        source_image=normalized,
        python=arguments.python.resolve(),
        plugin_command=arguments.plugin_command.resolve(),
        runtime_probe=runtime_probe,
        plugin_runner=plugin_runner,
    )
    verify_source_unchanged(source, before)
    preimage = {
        "schema_id": "asset-mania/private-face-plugin-inference",
        "schema_version": "0.1",
        "status": "passed",
        "plugin": "dad3dheads-local",
        "plugin_revision": DAD_REVISION,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_sha256": before.sha256,
        "source_unchanged": True,
        "normalized_sha256": sha256_file(normalized),
        "raw_mesh_sha256": sha256_file(result.raw_mesh),
        "projection_sha256": sha256_file(result.projection_data),
        "vertex_count": result.vertex_count,
        "triangle_count": result.triangle_count,
        "elapsed_seconds": result.elapsed_seconds,
        "identity_consistency": "unmeasured",
    }
    record = {**preimage, "record_sha256": canonical_digest(preimage)}
    record_path.write_text(canonical_json(record), encoding="utf-8")
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
    for name in ("smoke", "run"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--python", type=Path, required=True)
        command.add_argument("--plugin-command", type=Path, required=True)
        if name == "run":
            command.add_argument("--source", type=Path, required=True)
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
    runtime_probe: Callable[[Path], dict] | None = None,
    plugin_runner: Callable | None = None,
) -> int:
    arguments = build_parser().parse_args(list(argv) if argv is not None else None)
    timestamp = _parse_time(now)
    if arguments.command == "plan":
        return _run_plan(
            arguments,
            now=timestamp,
            run_id=(id_factory or (lambda: secrets.token_hex(4)))(),
        )
    if arguments.command == "acquire":
        return _run_acquire(
            arguments,
            git_acquirer=git_acquirer or _acquire_git,
            checkpoint_downloader=checkpoint_downloader or _download_checkpoint,
            revision_reader=revision_reader or _git_revision,
            expected_checkpoint_bytes=expected_checkpoint_bytes,
        )
    selected_probe = runtime_probe or _runtime_probe
    selected_runner = plugin_runner or run_face_plugin
    if arguments.command == "smoke":
        return _run_smoke(
            arguments,
            runtime_probe=selected_probe,
            plugin_runner=selected_runner,
        )
    return _run_private_inference(
        arguments,
        runtime_probe=selected_probe,
        plugin_runner=selected_runner,
    )


if __name__ == "__main__":
    raise SystemExit(main())
