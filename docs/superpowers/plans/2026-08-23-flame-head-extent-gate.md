# FLAME Full-Head Extent Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept valid full FLAME heads up to 0.32 metres while continuing to reject unit errors.

**Architecture:** Change the same inclusive metric gate in the public loader and both sealed
single-file workers. Bind the two exact bounds into new private geometry plans; do not transform
geometry.

**Tech Stack:** Python 3.9-compatible workers, Python 3.11-3.13 workspace, NumPy, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-23-flame-head-extent-gate-design.md`

## Global Constraints

- The accepted interval is exactly `0.15 <= extent <= 0.32` metres.
- No clamp, scale, repair, fallback, provider change, model execution, or face access in tests.
- Both workers remain directly executable under CPython 3.9.
- New geometry plans bind `minimum_head_extent_metres=0.15` and
  `maximum_head_extent_metres=0.32`.

---

### Task 1: Revise and bind the full-head metric gate

**Files:**
- Modify: `packages/pipeline/src/asset_mania_pipeline/face_geometry.py`
- Modify: `packages/pipeline/tests/test_face_geometry.py`
- Modify: `packages/engine-mica/src/asset_mania_engine_mica/plugin.py`
- Modify: `packages/engine-mica/tests/test_mica_plugin.py`
- Modify: `packages/engine-deca/src/asset_mania_engine_deca/plugin.py`
- Modify: `packages/engine-deca/tests/test_deca_plugin.py`
- Modify: `scripts/run_face_geometry_e2e.py`
- Modify: `tests/test_face_geometry_e2e.py`

**Interfaces:**
- Consumes: validated metric FLAME geometry and private plan preimages.
- Produces: the exact inclusive `0.15..0.32` gate at all three validation boundaries and plan-bound
  `minimum_head_extent_metres` / `maximum_head_extent_metres` fields.

- [ ] **Step 1: Write failing boundary and plan-binding tests**

Add literal behavior cases proving `0.309499189` and `0.32` pass, while `0.320001` and `0.149999`
fail, for the public loader and both adapter prediction validators. Assert a new geometry plan
contains the two exact bound fields and its canonical digest covers them.

- [ ] **Step 2: Run RED**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry.py packages/engine-mica/tests/test_mica_plugin.py packages/engine-deca/tests/test_deca_plugin.py tests/test_face_geometry_e2e.py -q
```

Expected: failures because `0.309499189` is rejected and plan fields are absent.

- [ ] **Step 3: Implement the exact inclusive gate**

Use named constants in the public loader and duplicate the same literals inside each self-contained
worker. Update diagnostics to say `0.15 and 0.32 metres`. Add the two bounds to `_run_plan`'s
`gates` object. Do not modify vertex values.

- [ ] **Step 4: Run GREEN and static checks**

```powershell
uv run pytest packages/pipeline/tests/test_face_geometry.py packages/engine-mica/tests/test_mica_plugin.py packages/engine-deca/tests/test_deca_plugin.py tests/test_face_geometry_e2e.py -q
uv run ruff check packages/pipeline/src/asset_mania_pipeline/face_geometry.py packages/engine-mica packages/engine-deca scripts/run_face_geometry_e2e.py
uv run ruff format --check packages/pipeline/src/asset_mania_pipeline/face_geometry.py packages/engine-mica packages/engine-deca scripts/run_face_geometry_e2e.py
git diff --check
```

- [ ] **Step 5: Commit**

```powershell
git add packages/pipeline/src/asset_mania_pipeline/face_geometry.py packages/pipeline/tests/test_face_geometry.py packages/engine-mica packages/engine-deca scripts/run_face_geometry_e2e.py tests/test_face_geometry_e2e.py
git commit -m "fix(face-geometry): accept full FLAME head extent"
```

