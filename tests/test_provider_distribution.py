"""The provider adapter is optional, and the CLI must not depend on it."""

import ast
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI_MANIFEST = ROOT / "packages" / "cli" / "pyproject.toml"
PROVIDER_MANIFEST = ROOT / "packages" / "provider-openai" / "pyproject.toml"


def _manifest(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_the_cli_has_no_runtime_dependency_on_the_provider() -> None:
    dependencies = _manifest(CLI_MANIFEST)["project"]["dependencies"]
    assert not any("provider" in name for name in dependencies), dependencies


def test_the_provider_is_discovered_through_an_entry_point() -> None:
    manifest = _manifest(PROVIDER_MANIFEST)
    entry_points = manifest["project"]["entry-points"]["asset_mania.providers"]
    assert entry_points["openai"].startswith("asset_mania_provider_openai")


def test_the_provider_is_a_separate_distribution() -> None:
    manifest = _manifest(PROVIDER_MANIFEST)
    assert manifest["project"]["name"] == "asset-mania-provider-openai"
    assert manifest["project"]["license"] == "Apache-2.0"


def test_the_adapter_imports_no_blender_module() -> None:
    source_root = ROOT / "packages" / "provider-openai" / "src"
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for module in ("import bpy", "import mathutils"):
            assert module not in text, f"{path}: {module}"


def test_the_adapter_constructs_no_socket_or_url_opener() -> None:
    """Only the explicit live transport may construct the approved HTTPS connection."""
    source_root = ROOT / "packages" / "provider-openai" / "src"
    forbidden = {"socket", "http", "urllib", "requests", "httpx"}
    exceptions: dict[str, set[str]] = {}
    for path in sorted(source_root.rglob("*.py")):
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        imported = roots & forbidden
        if path.name == "live_transport.py":
            exceptions[path.relative_to(source_root).as_posix()] = imported
        else:
            assert not imported, f"{path}: {sorted(imported)}"
    assert exceptions == {"asset_mania_provider_openai/live_transport.py": {"http"}}


@pytest.mark.parametrize("package", ["asset_mania_cli", "asset_mania_contracts"])
def test_no_apache_wheel_bundles_the_provider(tmp_path: Path, package: str) -> None:
    distribution = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", "--all-packages", "-o", str(distribution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"uv build is unavailable here: {completed.stderr[-200:]}")

    wheels = sorted(distribution.glob(f"{package}-*.whl"))
    assert wheels, f"no wheel built for {package}"
    with zipfile.ZipFile(wheels[0]) as archive:
        leaked = [name for name in archive.namelist() if "provider_openai" in name]
    assert leaked == []


def test_the_provider_wheel_carries_only_its_own_module(tmp_path: Path) -> None:
    distribution = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", "--package", "asset-mania-provider-openai", "-o", str(distribution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"uv build is unavailable here: {completed.stderr[-200:]}")

    wheels = sorted(distribution.glob("asset_mania_provider_openai-*.whl"))
    assert wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        modules = {name.split("/")[0] for name in archive.namelist() if name.endswith(".py")}
    assert modules == {"asset_mania_provider_openai"}


ENGINE_MANIFEST = ROOT / "packages" / "engine-triposr" / "pyproject.toml"


def test_the_cli_has_no_runtime_dependency_on_the_engine() -> None:
    dependencies = _manifest(CLI_MANIFEST)["project"]["dependencies"]
    assert not any("engine" in name for name in dependencies), dependencies


def test_the_engine_is_discovered_through_an_entry_point() -> None:
    entry_points = _manifest(ENGINE_MANIFEST)["project"]["entry-points"]["asset_mania.engines"]
    assert entry_points["triposr-local"].startswith("asset_mania_engine_triposr")


def _imported_roots(source: str) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


#: Reaching a network or a hub client. Forbidden everywhere in the engine package: this is
#: how uncleared weights arrive, whichever layer does it.
ACQUISITION_IMPORTS = frozenset(
    {"huggingface_hub", "urllib", "requests", "httpx", "socket", "subprocess"}
)

#: Running a model. Forbidden in the adapter layer, which decides *whether* a run may happen,
#: and expected in `ports/`, which performs it. Banning these outright would not remove the
#: loader -- it would only push it somewhere the check does not look.
EXECUTION_IMPORTS = frozenset({"torch", "rembg", "onnxruntime"})


def _engine_sources() -> tuple[list[Path], list[Path]]:
    """The engine package split into the layer that gates and the layer that executes."""
    source_root = ROOT / "packages" / "engine-triposr" / "src"
    everything = sorted(source_root.rglob("*.py"))
    execution = [
        path
        for path in everything
        if path.parent.name == "ports" or path.name == "multiview.py"
    ]
    adapter = [path for path in everything if path not in execution]
    assert adapter, "no adapter-layer sources found"
    assert execution, "no execution-layer sources found; the split below would check nothing"
    return adapter, execution


def test_the_engine_adapter_bundles_no_weight_or_downloader() -> None:
    """An engine adapter that grows a downloader is how uncleared weights arrive.

    Imports are read from the AST rather than matched as substrings: a field named
    `requests` is not the `requests` library, and a check that cannot tell the difference
    trains people to ignore it.
    """
    adapter, _ = _engine_sources()
    forbidden = ACQUISITION_IMPORTS | EXECUTION_IMPORTS
    for path in adapter:
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        assert not (roots & forbidden), f"{path}: {sorted(roots & forbidden)}"


def test_the_engine_ports_load_but_never_acquire() -> None:
    """A port loads a checkpoint that is already on disk; it does not go and fetch one.

    The distinction is the whole ordering guarantee. If a port could download, acquisition
    would become a side effect of pressing run, and the user would read the licences after
    the bytes had already landed rather than before.
    """
    _, execution = _engine_sources()
    for path in execution:
        roots = _imported_roots(path.read_text(encoding="utf-8"))
        assert not (roots & ACQUISITION_IMPORTS), f"{path}: {sorted(roots & ACQUISITION_IMPORTS)}"


def test_importing_the_engine_does_not_load_optional_runtime_or_network_modules() -> None:
    """Static execution imports remain lazy during the default offline package import."""
    forbidden = (
        "torch",
        "torchmcubes",
        "http.client",
        "socket",
        "urllib.request",
        "requests",
        "httpx",
        "huggingface_hub",
    )
    script = (
        "import sys; import asset_mania_engine_triposr; "
        f"forbidden = {forbidden!r}; "
        "loaded = [name for name in forbidden if name in sys.modules]; "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_the_engine_adapter_names_no_download_url() -> None:
    source_root = ROOT / "packages" / "engine-triposr" / "src"
    for path in sorted(source_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text, path
        assert "https://" not in text, path


@pytest.mark.parametrize(
    ("directory", "module"),
    [
        ("engine-mica", "asset_mania_engine_mica"),
        ("engine-deca", "asset_mania_engine_deca"),
    ],
)
def test_face_geometry_adapter_bundles_no_external_model_or_runtime(
    directory: str, module: str
) -> None:
    package = ROOT / "packages" / directory
    manifest = _manifest(package / "pyproject.toml")
    dependencies = manifest["project"]["dependencies"]
    assert not any(
        forbidden in dependency.lower()
        for dependency in dependencies
        for forbidden in ("torch", "mica", "deca", "flame", "insightface", "onnx")
    )
    forbidden_suffixes = {".tar", ".pkl", ".npy", ".npz", ".onnx", ".pt", ".pth"}
    assert not [path for path in package.rglob("*") if path.suffix.lower() in forbidden_suffixes]
    assert not any(
        "http://" in path.read_text(encoding="utf-8")
        or "https://" in path.read_text(encoding="utf-8")
        for path in (package / "src" / module).rglob("*.py")
    )


def test_notices_and_rules_name_the_local_face_geometry_boundary() -> None:
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    rules = (ROOT / "rules" / "agent" / "behavior-rules.md").read_text(encoding="utf-8")
    assert "packages/engine-mica/" in notices
    assert "packages/engine-deca/" in notices
    assert "transient" in rules.lower()
    assert "persisted_identity_feature_count" in rules


@pytest.mark.parametrize("package", ["asset_mania_cli", "asset_mania_contracts"])
def test_no_apache_wheel_bundles_the_engine(tmp_path: Path, package: str) -> None:
    distribution = tmp_path / "dist"
    completed = subprocess.run(
        ["uv", "build", "--all-packages", "-o", str(distribution)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        pytest.skip(f"uv build is unavailable here: {completed.stderr[-200:]}")

    wheels = sorted(distribution.glob(f"{package}-*.whl"))
    assert wheels, f"no wheel built for {package}"
    with zipfile.ZipFile(wheels[0]) as archive:
        leaked = [name for name in archive.namelist() if "engine_triposr" in name]
    assert leaked == []


def test_the_adapter_is_importable_without_the_cli() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import asset_mania_provider_openai.client as c; "
                "print('asset_mania' in sys.modules, c.PROVIDER)"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "False openai" in completed.stdout


# --- The planned-capability gate actually fires ------------------------------------


def test_the_planned_capability_gate_catches_an_unbacked_claim(tmp_path: Path) -> None:
    """A `Planned` row must not be able to become `Available` without evidence."""
    clone = tmp_path / "clone"
    (clone / "scripts").mkdir(parents=True)
    (clone / "tools").mkdir()
    (clone / "scripts" / "check_publication.py").write_text(
        (ROOT / "scripts" / "check_publication.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (clone / "THIRD_PARTY_NOTICES.md").write_text("# Notices\n", encoding="utf-8")
    (clone / "skills").mkdir()
    (clone / "skills" / "asset-mania").mkdir()
    (clone / "skills" / "asset-mania" / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    def run() -> subprocess.CompletedProcess[str]:
        subprocess.run(["git", "add", "-A"], cwd=clone, check=True, capture_output=True)
        return subprocess.run(
            [sys.executable, str(clone / "scripts" / "check_publication.py")],
            cwd=clone,
            capture_output=True,
            text=True,
            check=False,
        )

    subprocess.run(["git", "init", "-q"], cwd=clone, check=True)

    # Planned: accepted.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Planned | contracts only |\n", encoding="utf-8"
    )
    assert "UNBACKED_CAPABILITY_CLAIM" not in run().stdout

    # Available with no evidence phrase: refused.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Available | it works now |\n", encoding="utf-8"
    )
    assert "UNBACKED_CAPABILITY_CLAIM" in run().stdout

    # Available with the evidence phrase present: accepted.
    (clone / "README.md").write_text(
        "| Generic image to 3D | Available | see below |\n"
        "clearance is user-issued and unissued here\n",
        encoding="utf-8",
    )
    assert "UNBACKED_CAPABILITY_CLAIM" not in run().stdout


def test_the_readme_qualifies_generic_image_to_3d_with_what_is_still_missing() -> None:
    """The row left `Planned` when a reconstruction ran. It has to keep saying what did not.

    A working engine on a developer's machine is not a cleared engine on a user's, and the
    wheels still bundle nothing. Both halves stay in the row: the previous assertion here --
    that the row said `Planned` -- would now be enforcing a claim weaker than the truth,
    which is its own kind of wrong.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    row = next(line for line in readme.splitlines() if line.startswith("| Generic image to 3D |"))
    assert "Planned" not in row, "a reconstruction has run; the row is no longer a plan"
    assert "clearance is user-issued and unissued here" in row
    assert "no wheel ships an engine or a weight" in row


def test_the_readme_documents_the_limit_of_the_hole_repair() -> None:
    """The surface closes now, which makes the *bound* on the repair the thing to publish.

    This assertion used to require the README to say "open, not watertight", which was right
    while that was the measured state and became wrong once a repair pass landed. What has to
    stay documented is the part a reader could otherwise not know: filling holes is also how a
    failed reconstruction gets dressed up as a solid object, so the README has to say where the
    line is and that a wider opening leaves the mesh open.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Capping a *large* opening is not repair" in readme
    assert "leaves the\nmesh `open`" in readme or "leaves the mesh `open`" in readme
    assert "82%" in readme, "the fabrication case must be quantified, not just described"


def test_the_skill_refuses_generic_image_to_3d() -> None:
    """The skill must refuse the request without misstating why.

    It previously asserted nothing had ever run, which stopped being true. The refusal now
    rests on the fact that survives: no clearance exists in the user's installation, and the
    skill is forbidden from writing one. Overstating the blocker is as much a defect as
    overstating the capability -- it just fails in the flattering direction.
    """
    skill = (ROOT / "skills" / "asset-mania" / "SKILL.md").read_text(encoding="utf-8")
    assert "do not generate 3D geometry" in skill
    assert "no cleared engine" in skill.lower()
    assert "make this person 3D" in skill
    assert "Never issue that clearance" in skill
    assert "`open` when the model failed to reconstruct a region" in skill
