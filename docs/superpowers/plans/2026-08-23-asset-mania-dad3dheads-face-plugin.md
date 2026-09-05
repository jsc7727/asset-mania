# DAD-3DHeads Face Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the failed private TripoSR face experiment with an explicit, non-commercial
DAD-3DHeads plugin and produce a structurally verified Blender comparison from the authorized
portrait.

**Architecture:** Add a closed local subprocess protocol in the pipeline package, then implement
an Apache-licensed adapter that imports an untracked pinned DAD checkout only inside the plugin
process. Keep acquisition, compatibility dependencies, checkpoint, face input, OBJ/GLB outputs,
patches, and renders under ignored `.asset-mania` storage; public tests use a fake executable and
synthetic meshes only.

**Tech Stack:** CPython 3.11, PyTorch 2.13.0+cu130, CUDA 13, JSON subprocess protocol, NumPy,
Pillow, trimesh 4.0.5, DAD-3DHeads pinned at
`68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7`, Blender 5.2, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-asset-mania-dad3dheads-face-plugin-design.md`

## Global Constraints

- The user approved the DAD-3DHeads CC BY-NC-SA 4.0 non-commercial research condition, the exact
  pinned source revision, the 132,711,657-byte official checkpoint download, and local PyTorch
  `2.13.0+cu130` execution on 2026-08-23.
- Do not classify DAD, its checkpoint, FLAME assets, or its dependency closure as commercially
  cleared, Apache-licensed, or redistributable.
- Do not track or publish upstream source, weights, compatibility dependencies, patches, face
  inputs, generated views, masks, meshes, previews, reports, local paths, or private basenames.
- The observed portrait stays read-only and local. No face or derived face data leaves the host.
- Acquisition gets exactly one attempt per artifact. Inference performs no network request.
- No model, revision, PyTorch, device, input resolution, topology, or workflow fallback is allowed.
- Existing TripoSR code and results remain intact as historical experiments.
- All outputs are create-only. A collision stops the stage before partial replacement.
- Public fixtures are generated synthetic non-human geometry and images only.
- `identity_consistency` remains `unmeasured` even if manual visual review passes.
- Work inline on the existing scoped branch; the user previously declined extra worktree ceremony.

---

### Task 1: Closed local face-plugin protocol

**Files:**
- Create: `packages/pipeline/src/asset_mania_pipeline/face_plugins.py`
- Modify: `packages/pipeline/src/asset_mania_pipeline/__init__.py`
- Create: `packages/pipeline/tests/test_face_plugins.py`

**Interfaces:**
- Consumes: an explicit executable command, one private source path, one new output directory,
  pinned plugin revision, checkpoint digest, device, and timeout.
- Produces: `FacePluginRequest`, `FacePluginResult`, `build_face_plugin_request`,
  `write_face_plugin_request`, and `load_face_plugin_result`. The pipeline remains pure and starts
  no subprocess; Task 2 owns `run_face_plugin` in the optional engine package.

- [ ] **Step 1: Write failing request-validation tests**

```python
def test_request_is_closed_and_requires_cuda(tmp_path: Path) -> None:
    request = build_face_plugin_request(
        plugin="dad3dheads-local",
        plugin_revision=DAD_REVISION,
        source_image=tmp_path / "source.png",
        output_directory=tmp_path / "output",
        device="cuda",
        checkpoint_sha256="a" * 64,
    )
    assert request.schema == "asset-mania.face-plugin-request.v0"
    assert request.network == "denied-during-inference"
    with pytest.raises(ValueError, match="device must be cuda"):
        replace(request, device="cpu")
```

Add separate tests for an unknown plugin, non-absolute source/output paths, malformed revision or
SHA-256, an existing output directory, source/output containment, and extra JSON fields.

- [ ] **Step 2: Run the request tests and confirm RED**

Run:

```powershell
uv run pytest packages/pipeline/tests/test_face_plugins.py -q
```

Expected: import failure because `asset_mania_pipeline.face_plugins` does not exist.

- [ ] **Step 3: Implement immutable request and result types**

Implement these exact public types:

```python
@dataclass(frozen=True, slots=True)
class FacePluginRequest:
    schema: str
    plugin: str
    plugin_revision: str
    source_image: Path
    output_directory: Path
    device: Literal["cuda"]
    checkpoint_sha256: str
    network: Literal["denied-during-inference"]


@dataclass(frozen=True, slots=True)
class FacePluginResult:
    schema: str
    plugin: str
    status: Literal["succeeded", "incompatible_runtime", "invalid_output", "execution_failed"]
    raw_mesh: Path | None
    projection_data: Path | None
    vertex_count: int
    triangle_count: int
    elapsed_seconds: float
    device: Literal["cuda"]
    checkpoint_sha256: str
```

`build_face_plugin_request` accepts only `dad3dheads-local` in v0. Absolute paths are required in
the private request, but no public serialization helper may accept that document.

- [ ] **Step 4: Write failing closed-result tests**

Write result JSON directly and assert mismatched plugin, digest, device, missing output, extra
fields, unexpected files, and failure-shaped output paths are rejected.

- [ ] **Step 5: Implement create-only JSON I/O and closed result validation**

Use `canonical_json` for request files. Accept only `head.obj` and `projection.npz` in the plugin
output directory for a successful v0 result. Do not import `subprocess` or `socket` in pipeline.

- [ ] **Step 6: Run focused tests and commit**

```powershell
uv run pytest packages/pipeline/tests/test_face_plugins.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_plugins.py packages/pipeline/tests/test_face_plugins.py
git add packages/pipeline/src/asset_mania_pipeline/face_plugins.py packages/pipeline/src/asset_mania_pipeline/__init__.py packages/pipeline/tests/test_face_plugins.py
git commit -m "feat: add local face plugin protocol"
```

Expected: all face-plugin protocol tests pass.

---

### Task 2: Out-of-tree DAD plugin adapter

**Files:**
- Create: `packages/engine-dad3dheads/LICENSE`
- Create: `packages/engine-dad3dheads/pyproject.toml`
- Create: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/__init__.py`
- Create: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/plugin.py`
- Create: `packages/engine-dad3dheads/tests/test_plugin.py`

**Interfaces:**
- Consumes: `--request`, `--result`, `ASSET_MANIA_DAD_SOURCE_ROOT`, and
  `ASSET_MANIA_DAD_ISOLATED_HOME`.
- Produces: `run_face_plugin`, a v0 result, and exactly `head.obj` plus `projection.npz` inside the
  reserved output.

- [ ] **Step 1: Write failing adapter-boundary tests**

```python
def test_adapter_requires_pinned_source_and_preplaced_checkpoint(tmp_path: Path) -> None:
    settings = DADPluginSettings(
        source_root=tmp_path / "source",
        isolated_home=tmp_path / "home",
        revision=DAD_REVISION,
        checkpoint_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="pinned source revision is unavailable"):
        validate_dad_runtime(settings)
```

Add cases for the wrong Git HEAD, absent or wrong-size checkpoint, digest mismatch, missing
`predictor.py`, missing FLAME static assets, output collision, CPU-only torch, and attempted
network use.

- [ ] **Step 2: Run adapter tests and confirm RED**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_plugin.py -q
```

Expected: package import failure.

- [ ] **Step 3: Create the optional Apache adapter package**

Use the same workspace package pattern as `packages/engine-triposr`, but declare dependencies only
on `asset-mania-pipeline`. Do not declare torch, DAD, FLAME, OpenCV, or checkpoint packages in the
public wheel. Expose this console entry point:

```toml
[project.scripts]
asset-mania-dad3dheads-plugin = "asset_mania_engine_dad3dheads.plugin:main"
```

- [ ] **Step 4: Implement guarded upstream loading**

Implement:

```python
@dataclass(frozen=True, slots=True)
class DADPluginSettings:
    source_root: Path
    isolated_home: Path
    revision: str
    checkpoint_sha256: str


def validate_dad_runtime(settings: DADPluginSettings) -> Path: ...
def execute_dad_request(
    request_path: Path, result_path: Path, settings: DADPluginSettings
) -> int: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

Before importing upstream code, verify the exact Git HEAD and checkpoint digest. Set `HOME`,
`USERPROFILE`, `XDG_CACHE_HOME`, `TORCH_HOME`, and `HF_HOME` to descendants of the isolated home.
Patch `requests.sessions.Session.request` to raise `RuntimeError("network denied during DAD inference")`.
Import `FaceMeshPredictor` only after those guards are active.

Read the RGB image locally, call the pinned predictor once, and write the upstream vertices/faces
as OBJ plus `projected_vertices` as compressed NPZ. Write a success result only after CUDA
synchronization, finite-value checks, and output readback.

- [ ] **Step 5: Test with fake upstream modules**

Inject fake `torch`, `predictor`, and mesh arrays through private callables. Assert one model load,
one inference, exact checkpoint digest echo, 1-based OBJ faces, original-image projection
coordinates, and no request attempt. Failure paths must return one closed failure status without a
success-shaped mesh path.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_plugin.py packages/pipeline/tests/test_face_plugins.py -q
uv run ruff check packages/engine-dad3dheads packages/pipeline/src/asset_mania_pipeline/face_plugins.py
git add packages/engine-dad3dheads
git commit -m "feat: add guarded DAD face plugin adapter"
```

Expected: all adapter and protocol tests pass without importing or downloading DAD.

---

### Task 3: DAD mesh validation and GLB conversion

**Files:**
- Create: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/mesh.py`
- Modify: `packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/__init__.py`
- Create: `packages/engine-dad3dheads/tests/test_mesh.py`

**Interfaces:**
- Consumes: raw OBJ, `projection.npz`, observed normalized PNG, and two new GLB paths.
- Produces: `DADMeshMeasurements` and `convert_dad_mesh(...)`.

- [ ] **Step 1: Write failing structural validation tests**

Generate an asymmetric closed synthetic head plus a single neck boundary. Assert:

```python
measurements = inspect_dad_mesh(obj_path)
    assert measurements.component_count in (1, 3)
assert measurements.non_manifold_edge_count == 0
assert measurements.boundary_loop_count == 1
assert measurements.vertex_count == 12
assert measurements.triangle_count == 20
```

Add rejection cases for empty, NaN/Inf, zero-area majority, out-of-range faces, two components,
non-manifold edges, and an existing destination.

- [ ] **Step 2: Run mesh tests and confirm RED**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_mesh.py -q
```

Expected: `inspect_dad_mesh` is missing.

- [ ] **Step 3: Implement measured normalization and axis conversion**

Define:

```python
@dataclass(frozen=True, slots=True)
class DADMeshMeasurements:
    vertex_count: int
    triangle_count: int
    component_count: int
    boundary_edge_count: int
    boundary_loop_count: int
    non_manifold_edge_count: int
    winding_consistent: bool
    signed_volume: float | None
    observed_color_coverage: float


def inspect_dad_mesh(path: Path) -> DADMeshMeasurements: ...


def convert_dad_mesh(
    *,
    obj_path: Path,
    projection_path: Path,
    source_image: Path,
    plain_glb: Path,
    colored_glb: Path,
) -> DADMeshMeasurements: ...
```

Centre on bounds and scale the longest extent to `1.0`. Determine the DAD-to-Blender transform
from a synthetic landmark fixture and freeze it in one constant matrix. Never mirror by visual
guessing during the private run.

- [ ] **Step 4: Write failing projected-color tests**

Use a four-quadrant synthetic image and known projected vertices. Assert visible/in-bounds
vertices receive exact sampled RGB, rear or out-of-bounds vertices receive `(160, 145, 140, 255)`,
and observed coverage is measured. Assert the GLB reloads with one geometry and vertex colors.

- [ ] **Step 5: Implement front-only vertex colors and GLB readback**

Use predicted projected coordinates only for front-facing vertices. Do not infer rear UVs. Export
plain and colored GLBs create-only, reload both with trimesh, and recompute counts/components.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest packages/engine-dad3dheads/tests/test_mesh.py -q
uv run ruff check packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads/mesh.py packages/engine-dad3dheads/tests/test_mesh.py
git add packages/engine-dad3dheads/src/asset_mania_engine_dad3dheads packages/engine-dad3dheads/tests/test_mesh.py
git commit -m "feat: convert DAD head meshes to GLB"
```

Expected: synthetic conversion and GLB round-trip tests pass.

---

### Task 4: Sealed plan and one-attempt acquisition

**Files:**
- Create: `scripts/run_face_plugin_e2e.py`
- Create: `tests/test_face_plugin_e2e.py`

**Interfaces:**
- Consumes: `plan` and `acquire` CLI arguments plus injected Git/downloader functions in tests.
- Produces: private `plan.json`, pinned checkout, checkpoint, license copy, and `receipt.json`.

- [ ] **Step 1: Write failing plan-stage test**

```python
assert (
    main(
        [
            "plan",
            "--out",
            str(tmp_path / "runs"),
            "--plugin",
            "dad3dheads-local",
        ],
        now="2026-08-23T00:00:00+00:00",
        id_factory=lambda: "fixedrun",
    )
    == 0
)
```

Assert the plan fixes the revision, official source/checkpoint URLs, expected byte count, CC
BY-NC-SA restriction, `cuda`, `2.13.0+cu130`, create-only policy, no retry, no egress, and a
canonical plan digest. Assert no face path is accepted by `plan`.

- [ ] **Step 2: Run plan test and confirm RED**

```powershell
uv run pytest tests/test_face_plugin_e2e.py -k plan -q
```

Expected: runner import failure.

- [ ] **Step 3: Implement create-only plan stage**

Create these run directories: `plan`, `acquisition`, `smoke`, `inference`, `conversion`, and
`verification`. Write only stable public facts into `plan/plan.json`; print the new private run
path to the terminal but never copy it into public documentation.

- [ ] **Step 4: Write failing acquisition tests**

Inject a fake Git function and a fake streaming downloader. Assert `acquire`:

- verifies the sealed plan before network work;
- refuses without `--approval-reference face-plugin-approval-20260823`;
- checks out only the exact revision;
- performs one checkpoint GET and rejects wrong length;
- hashes source receipt, LICENSE, and checkpoint;
- rejects existing destination, redirect to a different host, partial download, or Git HEAD drift;
- writes no engine-clearance claim.

- [ ] **Step 5: Implement acquisition with no retry**

Use argument arrays for Git, `urllib.request` with an allowlist of the exact HTTPS URL, a temporary
file inside the new acquisition directory, byte-count verification, SHA-256, and atomic rename.
Copy the upstream LICENSE into private acquisition evidence and label the checkpoint
`redistribution=uncleared`, `commercial_use=forbidden_for_this_profile`.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest tests/test_face_plugin_e2e.py -k "plan or acquire" -q
uv run ruff check scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git add scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git commit -m "feat: plan and acquire face plugins safely"
```

Expected: deterministic plan/acquisition tests pass without network.

---

### Task 5: Synthetic smoke and private inference stages

**Files:**
- Modify: `scripts/run_face_plugin_e2e.py`
- Modify: `tests/test_face_plugin_e2e.py`

**Interfaces:**
- Consumes: verified acquisition receipt, explicit CUDA Python, explicit plugin command, and either
  a generated synthetic smoke image or the private source image.
- Produces: sealed smoke/inference records and raw plugin artifacts.

- [ ] **Step 1: Write failing smoke-stage tests**

Inject a fake plugin command that writes a valid synthetic OBJ/result. Assert `smoke` generates a
programmatic 256-square face-like drawing under private output, invokes exactly
`dad3dheads-local`, validates the result digest/counts, and records no input path.

Add failures for CPU, mismatched torch version, CUDA unavailable, acquisition digest mismatch,
plugin timeout, unexpected files, and any fallback attempt.

- [ ] **Step 2: Run smoke tests and confirm RED**

```powershell
uv run pytest tests/test_face_plugin_e2e.py -k smoke -q
```

Expected: `smoke` subcommand is absent.

- [ ] **Step 3: Implement runtime probe and smoke**

Probe the explicit Python with a fixed `-c` script returning JSON for Python, torch, CUDA runtime,
CUDA availability, and device type. Require exact torch `2.13.0+cu130`, `cuda_available=true`, and
`device_type=cuda`. Invoke the plugin through `run_face_plugin`; do not import DAD in the
orchestrator process.

- [ ] **Step 4: Write failing private-run tests**

Use a synthetic source fixture and injected plugin. Assert `run` fingerprints the source, creates
one deterministic normalized working copy, invokes the plugin once, verifies source unchanged,
and seals raw OBJ/projection hashes without serializing the source basename or path.

- [ ] **Step 5: Implement private run stage**

Require `--source`, `--python`, and `--plugin-command`. Normalize with the existing image pipeline,
reserve `inference/plugin-output`, and pass an environment containing only the DAD source root,
isolated home, cache paths, CUDA visibility, and required system variables. Refuse to run unless a
successful smoke record references the same plugin revision and checkpoint digest.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest tests/test_face_plugin_e2e.py packages/pipeline/tests/test_face_plugins.py packages/engine-dad3dheads/tests/test_plugin.py -q
git add scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py
git commit -m "feat: run guarded face plugin inference"
```

Expected: all fake-plugin smoke/run tests pass with no network, CUDA, model, or face fixture.

---

### Task 6: Conversion, Blender verification, and comparison

**Files:**
- Modify: `scripts/run_face_plugin_e2e.py`
- Modify: `tests/test_face_plugin_e2e.py`
- Modify if needed for deterministic labels only: `scripts/blender_preview.py`

**Interfaces:**
- Consumes: sealed inference record, raw OBJ/projection, source fingerprint, Blender executable,
  and two prior private comparison GLBs.
- Produces: plain/colored GLBs, independent measurements, four-view preview, comparison PNG, and
  private report.

- [ ] **Step 1: Write failing conversion-stage E2E test**

Inject a synthetic raw plugin result and call `convert`. Assert both GLBs are create-only, hashes
and `DADMeshMeasurements` are sealed, source remains unchanged, and
`identity_consistency="unmeasured"`.

- [ ] **Step 2: Run conversion test and confirm RED**

```powershell
uv run pytest tests/test_face_plugin_e2e.py -k convert -q
```

Expected: `convert` subcommand is absent.

- [ ] **Step 3: Implement conversion stage**

Call `convert_dad_mesh`, reload both GLBs, verify either the synthetic one-component topology or
the fixed DAD head plus two equal eye-shell components, require zero non-manifold edges, and
preserve the raw OBJ. Do not require a closed neck or positive watertight volume.

- [ ] **Step 4: Write failing verification-stage tests**

Inject a preview runner that writes three labeled four-view strips. Assert `verify` uses identical
samples, resolution, camera distances, background, lighting, and vertex-color mode for DAD,
TripoSR anchor, and face-hybrid inputs. Assert collision, missing prior mesh, altered source, or
failed Blender return code aborts before a report.

- [ ] **Step 5: Implement Blender comparison**

Run Blender 5.2 separately for all three meshes with `samples=16`, `resolution=500`, and
`views=4`. Combine the resulting strips with Pillow into rows labeled `DAD-3DHeads`,
`TripoSR front anchor`, and `TripoSR face hybrid`. Set `visual_quality="unreviewed"` and require a
manual update to `passed` or `failed`; never infer the verdict from metrics.

- [ ] **Step 6: Run tests and commit**

```powershell
uv run pytest tests/test_face_plugin_e2e.py packages/engine-dad3dheads/tests/test_mesh.py -q
uv run ruff check scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py packages/engine-dad3dheads
git add scripts/run_face_plugin_e2e.py tests/test_face_plugin_e2e.py scripts/blender_preview.py
git commit -m "feat: verify face plugins in Blender"
```

Expected: deterministic fake-plugin E2E through comparison passes.

---

### Task 7: License, publication, and research boundaries

**Files:**
- Modify: `scripts/check_release.py`
- Modify: `scripts/check_publication.py`
- Modify: `tests/test_check_release.py`
- Modify: `tests/test_license_boundary.py`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `docs/research.md`
- Modify: `docs/getting-started.md`
- Modify: `README.md` only if needed to keep the capability table honest

**Interfaces:**
- Consumes: public adapter files and deterministic fake-plugin evidence.
- Produces: enforced rejection of private/model artifacts and research-only documentation.

- [ ] **Step 1: Write failing release/publication tests**

Add fixtures proving tracked `.trcd`, `dad_checkpoints`, DAD upstream module copies, FLAME static
assets, private patch files, real-person images, OBJ/GLB outputs, and comparison renders fail.
Assert the original Apache adapter and synthetic text fixtures remain allowed.

- [ ] **Step 2: Run the boundary tests and confirm RED**

```powershell
uv run pytest tests/test_check_release.py tests/test_license_boundary.py -q
```

Expected: at least the `.trcd` and DAD-vendoring cases pass through incorrectly.

- [ ] **Step 3: Implement exact deny rules**

Add `.trcd` to model suffix rejection. Reject tracked paths containing
`.dad_checkpoints`, `model_training/model/static/flame`, or
`.asset-mania/dad3dheads`. Do not reject the adapter package name, design/plan text, source URL,
or `THIRD_PARTY_NOTICES.md` entry.

- [ ] **Step 4: Document deterministic status only**

Record DAD as an optional external CC BY-NC-SA 4.0 non-commercial research dependency with pinned
revision, official URL, no redistribution, and uncleared checkpoint terms. Before the live run,
state `designed; fake-plugin E2E only; live quality unverified` and keep the TripoSR failure visible.

- [ ] **Step 5: Run canonical gates and commit**

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest packages/pipeline/tests/test_face_plugins.py packages/engine-dad3dheads/tests tests/test_face_plugin_e2e.py tests/test_check_release.py tests/test_license_boundary.py -q
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_publication.py
uv run python scripts/check_release.py
git diff --check
git add scripts/check_release.py scripts/check_publication.py tests/test_check_release.py tests/test_license_boundary.py THIRD_PARTY_NOTICES.md docs/research.md docs/getting-started.md README.md
git commit -m "docs: bound DAD face plugin research use"
```

Expected: every listed command exits 0.

---

### Task 8: Approved acquisition and actual private face E2E

**Files:**
- Create only under ignored `.asset-mania/dad3dheads/` and
  `.asset-mania/private-face-run/dad3dheads-runs/`.
- Modify `docs/research.md` only after measured live evidence.
- Do not track any source checkout, checkpoint, environment, patch, face, mesh, render, or report.

**Interfaces:**
- Consumes: the approved source/checkpoint plan, existing verified CUDA Python, private authorized
  portrait, prior comparison GLBs, Blender 5.2, and completed Tasks 1-7.
- Produces: a private acquisition receipt, runtime receipt, smoke result, DAD OBJ/GLBs, Blender
  comparison, manual verdict, and an evidence-limited research update.

- [ ] **Step 1: Create the private plan and acquire exact artifacts**

```powershell
uv run python scripts/run_face_plugin_e2e.py plan --out .asset-mania/private-face-run/dad3dheads-runs --plugin dad3dheads-local
uv run python scripts/run_face_plugin_e2e.py acquire --run $env:ASSET_MANIA_DAD_RUN --approval-reference face-plugin-approval-20260823
```

Expected: Git HEAD equals `68cc9b51974e2628f7a8f8ed2dadc5f73b3f8aa7`, checkpoint length equals
`132711657`, and receipt contains freshly computed hashes. No retry is permitted.

- [ ] **Step 2: Build an isolated compatibility environment without mutating TripoSR**

Create `.asset-mania/dad3dheads/venv` from the same base Python, then add a `.pth` file pointing
read-only to `.asset-mania/triposr-cuda-venv/Lib/site-packages` so torch 2.13.0+cu130 is reused.
Install only these inference dependencies into the new environment after recording package URLs,
versions, hashes, and licenses:

```text
numpy==1.26.4
opencv-python-headless==4.10.0.84
albumentations==1.0.0
smplx==0.1.26
pytorch-toolbelt==0.5.0
PyYAML==6.0.2
requests==2.32.5
```

Run a probe that asserts exact torch version, CUDA availability, RTX device type, imports, and no
modification to either TripoSR environment.

- [ ] **Step 3: Run the synthetic CUDA smoke**

```powershell
uv run python scripts/run_face_plugin_e2e.py smoke --run $env:ASSET_MANIA_DAD_RUN --python .asset-mania/dad3dheads/venv/Scripts/python.exe --plugin-command $env:ASSET_MANIA_DAD_PLUGIN
```

Expected: the pinned TorchScript checkpoint loads once on CUDA and produces finite one-component
synthetic OBJ geometry. Record elapsed time and peak allocated VRAM.

- [ ] **Step 4: Handle compatibility failure without substitution**

If smoke fails, invoke `superpowers:systematic-debugging`. Preserve the exact traceback privately,
identify the first incompatible API, and make only the smallest private checkout patch. Save
`patches/compatibility.patch`, its SHA-256, and before/after synthetic test evidence. Do not change
weights, topology, input size, PyTorch, CUDA, or device. If the same blocker survives three focused
attempts, stop and report `incompatible_runtime`.

- [ ] **Step 5: Fingerprint and run the private source once**

Set `$env:ASSET_MANIA_FACE_SOURCE` privately without printing it. Then run:

```powershell
uv run python scripts/run_face_plugin_e2e.py run --run $env:ASSET_MANIA_DAD_RUN --source $env:ASSET_MANIA_FACE_SOURCE --python .asset-mania/dad3dheads/venv/Scripts/python.exe --plugin-command $env:ASSET_MANIA_DAD_PLUGIN
uv run python scripts/run_face_plugin_e2e.py convert --run $env:ASSET_MANIA_DAD_RUN
```

Expected: source fingerprint matches before/after, inference makes no network request, and raw OBJ
plus plain/colored GLBs pass structural readback.

- [ ] **Step 6: Render and inspect the Blender comparison**

```powershell
uv run python scripts/run_face_plugin_e2e.py verify --run $env:ASSET_MANIA_DAD_RUN --blender "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --triposr-anchor $env:ASSET_MANIA_TRIPOSR_ANCHOR --triposr-hybrid $env:ASSET_MANIA_TRIPOSR_HYBRID
```

Open the private comparison image and inspect front, right, rear, and left views. Mark
`visual_quality=passed` only if the coherent face relief criteria in the spec beat both TripoSR
outputs. Otherwise mark `failed` with the observed reason. Keep
`identity_consistency=unmeasured`.

- [ ] **Step 7: Run full repository verification**

Use the exact Makefile bodies because GNU Make is unavailable on this Windows host:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
uv run python scripts/validate_skill.py skills/asset-mania
uv run python scripts/check_release.py
uv run python scripts/check_license_boundary.py
uv run python scripts/check_schema_distribution.py
uv run python scripts/check_publication.py
git diff --check
git status --short
git ls-files .asset-mania
```

Report focused pass counts and full-suite Windows/POSIX incompatibilities separately. Require
`git ls-files .asset-mania` to print nothing.

- [ ] **Step 8: Commit only measured public documentation**

Update `docs/research.md` with exact runtime, geometry measurements, and manual visual verdict.
Do not include private paths, filenames, hashes tied to the face, images, or identity claims.

```powershell
git add docs/research.md
git diff --cached --check
git commit -m "docs: record DAD face plugin E2E evidence"
```

Expected: the final commit contains documentation only; every private artifact remains ignored.
