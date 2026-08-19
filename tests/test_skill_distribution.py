import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "skills" / "asset-mania" / "scripts" / "inspect.py"
SKILL_SCHEMA = ROOT / "skills" / "asset-mania" / "references" / "manifest-v1.schema.json"
CONTRACT_SCHEMA = (
    ROOT
    / "packages"
    / "contracts"
    / "src"
    / "asset_mania_contracts"
    / "schema"
    / "manifest-v1.schema.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_launcher_inspects_png_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "private-source-name.png"
    Image.new("RGBA", (8, 6), color=(20, 40, 60, 80)).save(source)
    before = _sha256(source)
    output_parent = tmp_path / "runs"
    environment = {
        "PATH": str(Path(sys.executable).parent),
        "OPENAI_API_KEY": "must-not-be-forwarded",
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(LAUNCHER),
            "inspect",
            str(source),
            "--out",
            str(output_parent),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert _sha256(source) == before
    report = json.loads(completed.stdout)
    assert report["result"] == {
        "status": "succeeded",
        "diagnostics": ["WORKFLOW_NOT_IMPLEMENTED"],
    }
    assert report["parameters"] == {"kind": "object", "workflow": "image-to-3d"}
    assert report["inputs"][0]["label"] == "input-1"
    assert source.name not in completed.stdout
    assert str(source) not in completed.stdout
    run_directories = [path for path in output_parent.iterdir() if path.is_dir()]
    assert len(run_directories) == 1
    manifest = json.loads((run_directories[0] / "manifest.json").read_text())
    assert manifest["result"] == report["result"]


def test_launcher_forwards_arguments_without_shell_or_environment_secrets(tmp_path: Path) -> None:
    executable_directory = tmp_path / "bin"
    executable_directory.mkdir()
    fake_cli = executable_directory / "asset-mania"
    fake_cli.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'environment': dict(os.environ)}))\n"
    )
    fake_cli.chmod(fake_cli.stat().st_mode | stat.S_IXUSR)
    shell_marker = tmp_path / "shell-was-used"
    literal_argument = f"value with spaces; touch {shell_marker}"

    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "inspect", literal_argument],
        cwd=tmp_path,
        env={
            "PATH": str(executable_directory),
            "ASSET_MANIA_TEST_SECRET": "must-not-be-forwarded",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["argv"] == ["inspect", literal_argument]
    assert "ASSET_MANIA_TEST_SECRET" not in payload["environment"]
    assert not shell_marker.exists()


def test_skill_schema_is_byte_identical_to_contract_schema() -> None:
    assert SKILL_SCHEMA.read_bytes() == CONTRACT_SCHEMA.read_bytes()
