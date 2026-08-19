# Asset Mania v0.1 Monorepo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Publish a polished, guide-first open-source monorepo containing a deterministic offline Asset Mania inspection CLI, its versioned contract package, and a reusable Agent Skill.

**Architecture:** A uv workspace separates portable contracts from the CLI while keeping one lockfile and one test toolchain. The root AGENTS.md stays short and routes agents into focused rules documents; human documentation lives under docs. The Agent Skill is a thin orchestration layer that invokes the tested CLI and refuses every unimplemented upload, generation, model-download, or paid action.

**Tech Stack:** CPython 3.11-3.13, uv workspace, uv_build, Pillow, argparse, JSON Schema draft 2020-12, pytest, Ruff, GitHub Actions, Markdown, Codex Agent Skills.

**Spec:** docs/superpowers/specs/2026-08-19-asset-mania-open-source-skill-design.md

## Global Constraints

- Support CPython 3.11 through 3.13 on macOS 14+ and Ubuntu 22.04/24.04.
- v0.1 performs no network request, upload, model download, Blender invocation, rendering, or GPU job.
- Inspection may create a new run directory but never mutates the source input.
- Portable output contains fixed labels, hashes, allowlisted metadata, and no absolute path, basename, credential, raw EXIF/IPTC/XMP value, or face embedding.
- The root CLI/contracts/skill/docs license is Apache-2.0; future bpy code is a separately packaged GPL-3.0-or-later component.
- Do not commit the supplied face ZIP, real-person imagery, generated face frames, weights, caches, datasets, secrets, or opaque binaries.
- All behavior code follows red-green-refactor. Human prose and static configuration are validated by link, structure, or consumer behavior rather than text-matching tests.
- Use Conventional Commits and repository-local jsc7727 author identity. Do not change global Git/GitLab configuration.

---

## Execution Setup

Complete this preflight before Task 1:

1. From the primary checkout, verify main is clean and record the implementation base:

       git switch main
       git status --porcelain
       git rev-parse HEAD

   Expected: empty status output. Record the printed SHA as IMPLEMENTATION_BASE in the execution ledger.

2. The planning commit already contains the root .gitignore entry .worktrees/. Verify it, then create the branch/worktree:

       git check-ignore -q --no-index .worktrees/initial-monorepo
       git worktree add .worktrees/initial-monorepo -b codex/initial-monorepo

   Expected: the branch is created from IMPLEMENTATION_BASE. If it already exists, verify its merge-base equals IMPLEMENTATION_BASE instead of recreating it.

3. Configure and read back repository-local identity from the implementation worktree:

       git -C .worktrees/initial-monorepo config --local user.name jsc7727
       git -C .worktrees/initial-monorepo config --local user.email 20225380+jsc7727@users.noreply.github.com
       git -C .worktrees/initial-monorepo config --local --get user.name
       git -C .worktrees/initial-monorepo config --local --get user.email
       git config --global --get user.name
       git config --global --get user.email

   Expected: local values are jsc7727 and its GitHub noreply address; global values remain gm2302035 and scjang@gabia.com.

4. Run every Task 1-6 command from .worktrees/initial-monorepo. Before each commit, verify git branch --show-current prints codex/initial-monorepo.

---

## File Map

### Workspace and public surface

- Create: pyproject.toml — uv workspace membership and shared development dependencies.
- Create: .python-version — pins local development to Python 3.12.
- Create: uv.lock — shared locked dependency graph.
- Create: Makefile — stable setup, check, test, skill validation, and release-check commands.
- Modify: .gitignore — preserve the pre-committed worktree exclusion and add environments, run outputs, caches, weights, and logs.
- Create: README.md — public pre-alpha positioning, working quickstart, capability table, architecture, privacy, and roadmap.
- Create: LICENSE — canonical Apache-2.0 text for root components.
- Create: THIRD_PARTY_NOTICES.md — initial dependency and fixture notice policy.
- Create: CONTRIBUTING.md — setup, TDD, commit, docs, and review workflow.
- Create: CODE_OF_CONDUCT.md — Contributor Covenant text or link with enforcement route.
- Create: SECURITY.md — private vulnerability reporting instructions and supported version.

### Agent and documentation system

- Create: AGENTS.md — short repository entrypoint with valid commands and rule links.
- Create: rules/README.md — rules-system purpose and navigation.
- Create: rules/index.md — task-to-rule routing table.
- Create: rules/project/tech-stack.md — exact workspace/runtime/dependency boundaries.
- Create: rules/project/project-structure.md — ownership map for packages, skill, docs, and tests.
- Create: rules/development/feature-development.md — plan, TDD, docs, and validation sequence.
- Create: rules/development/coding-conventions.md — Python contracts, deterministic JSON, privacy, and error conventions.
- Create: rules/development/git-conventions.md — branch, commit, author, and push rules.
- Create: rules/testing/README.md — red-green evidence and verification matrix.
- Create: rules/agent/behavior-rules.md — approval gates and no-silent-fallback rules.
- Create: docs/README.md — human documentation index.
- Create: docs/getting-started.md — installation and first inspection.
- Create: docs/architecture.md — package and future provider boundaries.
- Create: docs/development.md — contributor environment and commands.
- Create: docs/concepts/run-manifest.md — schema, streams, exit codes, provenance, and privacy.
- Create: docs/security-and-privacy.md — local-only v0.1 guarantees and future external gates.
- Create: docs/research.md — curated primary papers, official projects, and product landscape.
- Create: docs/roadmap.md — v0.1 through cloud sequence.

### Contracts package

- Create: packages/contracts/pyproject.toml — asset-mania-contracts package metadata.
- Create: packages/contracts/src/asset_mania_contracts/__init__.py — public exports.
- Create: packages/contracts/src/asset_mania_contracts/diagnostics.py — stable codes and result statuses.
- Create: packages/contracts/src/asset_mania_contracts/models.py — JSON-compatible typed contracts and canonical serializer.
- Create: packages/contracts/src/asset_mania_contracts/schema/manifest-v1.schema.json — persisted manifest schema.
- Create: packages/contracts/tests/test_contracts.py — serialization, schema, status, and redaction contract.

### CLI package

- Create: packages/cli/pyproject.toml — asset-mania-cli package and console entrypoint.
- Create: packages/cli/src/asset_mania/__init__.py — version.
- Create: packages/cli/src/asset_mania/__main__.py — python -m entrypoint.
- Create: packages/cli/src/asset_mania/cli.py — argparse, stream, and exit behavior.
- Create: packages/cli/src/asset_mania/environment.py — OS, Python, and Blender executable discovery without invocation.
- Create: packages/cli/src/asset_mania/inspectors/__init__.py — media inspector routing.
- Create: packages/cli/src/asset_mania/inspectors/image.py — allowlisted Pillow metadata.
- Create: packages/cli/src/asset_mania/inspectors/blend.py — 12-byte Blender header parser.
- Create: packages/cli/src/asset_mania/run.py — atomic run directory, manifest/report lifecycle, and hashing.
- Create: packages/cli/src/asset_mania/service.py — inspect command orchestration independent from stdio.
- Create: packages/cli/tests/conftest.py — temporary fixture factories.
- Create: packages/cli/tests/test_image_inspector.py — image allowlist and sensitive metadata behavior.
- Create: packages/cli/tests/test_blend_inspector.py — valid and invalid header behavior.
- Create: packages/cli/tests/test_service.py — manifest lifecycle, defaults, and source integrity.
- Create: packages/cli/tests/test_cli.py — stdout/stderr and exit-code integration.
- Create: tests/fixtures/manifest-v1-success.json — committed compatibility fixture containing no paths or private data.

### Agent Skill

- Create: skills/asset-mania/SKILL.md — concise trigger, v0.1 workflow, boundaries, and approval gates.
- Create: skills/asset-mania/agents/openai.yaml — display metadata and default prompt.
- Create: skills/asset-mania/scripts/inspect.py — deterministic CLI launcher with missing-install guidance.
- Create: skills/asset-mania/references/cli-contract.md — command, stream, and exit reference.
- Create: skills/asset-mania/references/manifest-v1.schema.json — byte-identical schema copy produced from the contracts package.
- Create: skills/asset-mania/references/safety-and-licenses.md — external-action and license boundaries.
- Create: skills/asset-mania/references/evals.md — five fixed forward-evaluation requests and pass criteria.
- Create: tests/test_skill_distribution.py — schema parity and launcher behavior.
- Create: scripts/validate_skill.py — repository-owned structural validator used by local checks and CI.
- Create: tests/test_validate_skill.py — validator behavior for valid and malformed skill packages.

### Release automation

- Create: scripts/check_release.py — denylist, secret-pattern, link, inventory, and schema-copy checks.
- Create: tests/test_check_release.py — release checker behavior using temporary Git-like trees.
- Create: tests/fixtures/PROVENANCE.md — states that binary fixtures are generated at test runtime and none are tracked.
- Create: .github/workflows/ci.yml — supported Python/OS matrix and skill/release gates.
- Create: .github/ISSUE_TEMPLATE/bug_report.yml — privacy-safe bug template.
- Create: .github/ISSUE_TEMPLATE/feature_request.yml — scoped feature template.
- Create: .github/pull_request_template.md — tests, docs, privacy, license, and provenance checklist.

---

### Task 1: Guide-First Monorepo and Agent Rules

**Files:**
- Create all files in “Workspace and public surface” and “Agent and documentation system”.

**Interfaces:**
- Consumes: approved design spec.
- Produces: uv workspace skeleton; canonical commands make setup, make check, make test, make skill-check, and make release-check; repository rules and architecture guides used by every later task. Executable quickstart output is intentionally deferred until Task 6.

- [ ] **Step 1: Preview the rules scaffold without writing**

Run:

    python3 ~/.codex/skills/rules-setup/scripts/setup_rules.py . --project-name asset-mania --dry-run

Expected: paths are printed; no tracked file changes.

- [ ] **Step 2: Create the uv workspace and repository metadata**

Create a non-package root project with workspace members:

    [project]
    name = "asset-mania-workspace"
    version = "0.1.0"
    requires-python = ">=3.11,<3.14"
    dependencies = []

    [dependency-groups]
    dev = [
      "jsonschema>=4.25,<5",
      "pytest>=8.4,<9",
      "pytest-cov>=6,<8",
      "ruff>=0.12,<1",
    ]

    [tool.uv]
    package = false

    [tool.uv.workspace]
    members = ["packages/*"]

    [tool.pytest.ini_options]
    addopts = "-ra"
    testpaths = ["packages", "tests"]

    [tool.ruff]
    line-length = 100
    target-version = "py311"

The Makefile commands are:

    setup:
        uv sync --locked --all-packages --dev

    check:
        uv run ruff check .
        uv run ruff format --check .

    test:
        uv run pytest

    skill-check:
        uv run python scripts/validate_skill.py skills/asset-mania

    release-check:
        uv run python scripts/check_release.py

- [ ] **Step 3: Write AGENTS.md and focused rules**

Keep AGENTS.md under 100 lines. It links rules/README.md, tells agents to search rules before implementation, lists the five valid Make targets, requires source-read-only inspection, and forbids silent provider/model fallback or external upload without explicit approval. Do not copy detailed tutorials into AGENTS.md.

- [ ] **Step 4: Write the human guide skeleton and public README**

README must visibly label the repository as pre-alpha and implementation-in-progress. Do not show an installation or execution example until the command exists. Use this target capabilities table:

    | Capability | v0.1 target |
    | --- | --- |
    | Offline image and .blend inspection | In development |
    | Versioned manifest and report | In development |
    | Image to 3D generation | Planned |
    | Blender scene to GPT Image | Planned |
    | Face/head reconstruction | Research |
    | Asset Mania Cloud | Later |

Docs must cross-link through docs/README.md, explain the intended command contract, and contain no claim that an executable quickstart currently works. Task 6 replaces the in-development status with verified output from the real CLI.

- [ ] **Step 5: Lock and validate the workspace**

Run:

    uv lock
    uv sync --locked --all-packages --dev
    uv run python -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
    git diff --check

Expected: all commands exit 0; uv.lock is created; no package code exists yet.

- [ ] **Step 6: Commit the foundation**

Run:

    git add AGENTS.md README.md LICENSE THIRD_PARTY_NOTICES.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md Makefile pyproject.toml .python-version .gitignore uv.lock docs rules
    git commit -m "docs: establish guide-first monorepo"

### Task 2: Versioned Contracts Package

**Files:**
- Create all files under packages/contracts.

**Interfaces:**
- Produces: DiagnosticCode string enum; ResultStatus string enum; canonical_json(value: object) -> str; build_manifest(...) -> dict[str, object]; load_manifest_schema() -> dict[str, object].
- Consumed by: CLI service, release checker, schema copy in the Agent Skill.

- [ ] **Step 1: Create package metadata, then write failing contract tests**

Create packages/contracts/pyproject.toml before running tests:

    [project]
    name = "asset-mania-contracts"
    version = "0.1.0"
    description = "Portable contracts for Asset Mania inspection runs"
    requires-python = ">=3.11,<3.14"
    dependencies = []

    [build-system]
    requires = ["uv_build>=0.12.1,<0.13"]
    build-backend = "uv_build"

Create an empty packages/contracts/src/asset_mania_contracts/__init__.py so the workspace member can be installed without implementing behavior. Then run:

    uv lock
    uv sync --locked --all-packages --dev

This captures and installs the workspace member before the red test. The test must still fail because build_manifest and the contract types do not exist.

The first test defines canonical JSON and required fields:

    def test_build_manifest_uses_portable_labels_and_canonical_json():
        manifest = build_manifest(
            run_id="run-1",
            created_at="2026-08-19T00:00:00Z",
            tool_version="0.1.0",
            input_sha256="a" * 64,
            byte_size=12,
            media_type="image/png",
            parameters={"workflow": "image-to-3d", "kind": "object"},
            result_status=ResultStatus.SUCCEEDED,
            diagnostics=[DiagnosticCode.WORKFLOW_NOT_IMPLEMENTED],
        )
        assert manifest["inputs"] == [{
            "label": "input-1",
            "sha256": "a" * 64,
            "byte_size": 12,
            "media_type": "image/png",
        }]
        assert "Users/" not in canonical_json(manifest)

Add tests that validate a literal successful and failed manifest against manifest-v1.schema.json and reject an absolute-path field.

- [ ] **Step 2: Run tests and verify RED**

Run:

    uv run pytest packages/contracts/tests/test_contracts.py -v

Expected: FAIL because asset_mania_contracts does not exist.

- [ ] **Step 3: Implement the minimum contract package**

Use string enums:

    class ResultStatus(StrEnum):
        SUCCEEDED = "succeeded"
        FAILED = "failed"
        UNSUPPORTED = "unsupported"
        NEEDS_APPROVAL = "needs_approval"

    class DiagnosticCode(StrEnum):
        INPUT_NOT_FOUND = "INPUT_NOT_FOUND"
        UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
        INPUT_UNREADABLE = "INPUT_UNREADABLE"
        EXIF_SENSITIVE_METADATA_PRESENT = "EXIF_SENSITIVE_METADATA_PRESENT"
        BLEND_HEADER_INVALID = "BLEND_HEADER_INVALID"
        BLENDER_NOT_FOUND = "BLENDER_NOT_FOUND"
        WORKFLOW_NOT_IMPLEMENTED = "WORKFLOW_NOT_IMPLEMENTED"

canonical_json uses json.dumps with sort_keys=True, separators=(",", ":"), ensure_ascii=False, and one trailing newline.

- [ ] **Step 4: Run contract tests and full checks**

Run:

    uv run pytest packages/contracts/tests/test_contracts.py -v
    uv run ruff check packages/contracts
    uv run ruff format --check packages/contracts

Expected: all exit 0.

- [ ] **Step 5: Commit contracts**

Run:

    git add packages/contracts uv.lock
    git commit -m "feat: define portable inspection contracts"

### Task 3: Image and Blender Header Inspectors

**Files:**
- Create inspector and environment files plus their tests under packages/cli.

**Interfaces:**
- Consumes: DiagnosticCode from asset-mania-contracts.
- Produces: inspect_image(path: Path) -> tuple[dict[str, object], list[DiagnosticCode]]; inspect_blend(path: Path) -> tuple[dict[str, object], list[DiagnosticCode]]; inspect_environment(configured_blender: Path | None = None) -> tuple[dict[str, object], list[DiagnosticCode]].
- Invariant: returned data contains no source path, basename, or non-allowlisted metadata value.

- [ ] **Step 1: Create CLI package metadata, then write failing image tests**

Create packages/cli/pyproject.toml:

    [project]
    name = "asset-mania-cli"
    version = "0.1.0"
    description = "Offline Asset Mania inspection CLI"
    requires-python = ">=3.11,<3.14"
    dependencies = [
      "asset-mania-contracts",
      "Pillow>=11.3,<13",
    ]

    [project.scripts]
    asset-mania = "asset_mania.cli:entrypoint"

    [tool.uv.sources]
    asset-mania-contracts = { workspace = true }

    [build-system]
    requires = ["uv_build>=0.12.1,<0.13"]
    build-backend = "uv_build"

Create empty packages/cli/src/asset_mania/__init__.py and packages/cli/src/asset_mania/inspectors/__init__.py so the package installs without implementing behavior. Then run:

    uv lock
    uv sync --locked --all-packages --dev

This captures Pillow and the workspace dependency before the red tests. The tests must still fail because the inspector functions do not exist.

Use Pillow to create temporary PNG, JPEG, and WebP files. Include a JPEG with numeric orientation and sensitive Artist, Model, DateTime, and GPS-like metadata. Assert only these values are emitted:

    assert report == {
        "format": "JPEG",
        "width": 8,
        "height": 6,
        "mode": "RGB",
        "bit_depth": 8,
        "channels": 3,
        "has_alpha": False,
        "orientation": 6,
        "metadata_blocks": {
            "exif": True,
            "gps": True,
            "icc": False,
            "iptc": False,
            "xmp": False,
        },
    }
    assert "camera-owner@example.com" not in json.dumps(report)
    assert diagnostics == [DiagnosticCode.EXIF_SENSITIVE_METADATA_PRESENT]

- [ ] **Step 2: Verify image tests fail**

Run:

    uv run pytest packages/cli/tests/test_image_inspector.py -v

Expected: FAIL because inspect_image is missing.

- [ ] **Step 3: Implement allowlisted image inspection**

Open with Pillow, call verify before reading metadata, and emit only width, height, mode, derived bit depth/channel count, alpha presence, numeric orientation, and presence flags. Never emit image.info values or arbitrary EXIF tag values.

- [ ] **Step 4: Write failing Blender-header and environment tests**

Use literal 12-byte headers:

    @pytest.mark.parametrize(
        ("header", "version", "pointer_size", "endianness"),
        [
            (b"BLENDER-v280", 280, 64, "little"),
            (b"BLENDER_V400", 400, 32, "big"),
        ],
    )

Add invalid magic, pointer marker, endian marker, non-digit version, and short-file cases. Test Blender discovery with a temporary executable path and a controlled PATH; do not invoke the executable.

Add a truncated/corrupt PNG test that returns INPUT_UNREADABLE without leaking the source name. The service and CLI integration tests in Task 4 will verify the failure manifest and exit code.

- [ ] **Step 5: Verify Blender tests fail**

Run:

    uv run pytest packages/cli/tests/test_blend_inspector.py -v

Expected: FAIL because inspect_blend and inspect_environment are missing.

- [ ] **Step 6: Implement Blender header and executable discovery**

Parse exactly 12 bytes. Accept pointer markers - for 64-bit and _ for 32-bit, endian markers v for little and V for big, and version digits 100-999. Search an explicit configured path, PATH, and the standard macOS Blender app executable path; report found/not_found only.

- [ ] **Step 7: Run inspector tests and commit**

Run:

    uv run pytest packages/cli/tests/test_image_inspector.py packages/cli/tests/test_blend_inspector.py -v
    uv run ruff check packages/cli
    uv run ruff format --check packages/cli
    git add packages/cli uv.lock
    git commit -m "feat: inspect image and Blender inputs"

Expected: tests and checks exit 0 before commit.

### Task 4: Run Lifecycle and CLI Contract

**Files:**
- Create packages/cli/src/asset_mania/run.py.
- Create packages/cli/src/asset_mania/service.py.
- Create packages/cli/src/asset_mania/cli.py.
- Create packages/cli/src/asset_mania/__main__.py.
- Create packages/cli/tests/test_service.py.
- Create packages/cli/tests/test_cli.py.

**Interfaces:**
- Consumes: inspectors and contracts.
- Produces: execute_inspect(request: InspectRequest, *, clock: Clock, id_factory: IdFactory) -> CommandResult; main(argv: Sequence[str] | None = None) -> int.
- CommandResult fields: exit_code, report, primary_diagnostic, run_dir.

- [ ] **Step 1: Write failing service tests**

Cover:

- image defaults to workflow image-to-3d and kind object;
- .blend defaults to workflow scene-to-image and rejects kind;
- omitted --out uses .asset-mania/runs beneath the current directory;
- environment output records OS, host architecture, Python version, Blender found/not_found, and unavailable future provider capabilities;
- declared face-head inspection succeeds locally, records a future rights advisory, and does not return a needs-approval status;
- .blend inspection succeeds with exit 0 when Blender is absent and includes BLENDER_NOT_FOUND as a capability warning;
- source SHA-256 is unchanged before/after;
- run directory contains manifest.json and report.json;
- every successful bootstrap creates an empty logs/ directory atomically with the report files;
- portable JSON contains input-1 and no basename or absolute path;
- every success and failure manifest validates against manifest-v1.schema.json;
- two reports for the same input are byte-identical after removing run_id and created_at;
- tokens, absolute home paths, sensitive EXIF values, and temporary basenames never appear in manifest, report, stdout, stderr, or logs;
- tests/fixtures/manifest-v1-success.json remains readable and schema-valid;
- existing run path is not overwritten;
- missing input creates a failed run and exits 3;
- a corrupt image, invalid .blend header, unreadable file, and unsupported .txt input each exit 3, emit the stable diagnostic, and write a schema-valid failure manifest when storage remains writable;
- unwritable parent returns exit 73 without claiming a manifest.

Use injected fixed clock and ID factory:

    result = execute_inspect(
        InspectRequest(input_path=source, output_parent=tmp_path / "runs"),
        clock=lambda: datetime(2026, 8, 19, tzinfo=UTC),
        id_factory=lambda: "abc123",
    )
    assert result.run_dir.name == "20260819T000000Z-abc123"

- [ ] **Step 2: Verify service tests fail**

Run:

    uv run pytest packages/cli/tests/test_service.py -v

Expected: FAIL because service and run lifecycle are missing.

- [ ] **Step 3: Implement atomic run lifecycle and service**

Create a temporary sibling directory containing manifest.json, report.json, and an empty logs/ directory, write canonical JSON with mode 0o600, then atomically rename it to the final run directory. On failures after creation, persist failed status when writable. Use fixed labels and hashes only. Debug output, when enabled in the future, may add files inside the existing logs/ directory.

- [ ] **Step 4: Write failing CLI stream tests**

Invoke the installed console command in subprocesses and assert:

- omitted --format emits canonical JSON on stdout;
- --format text changes only stdout while persisted manifest.json and report.json remain canonical JSON;
- exit 0: JSON or text stdout, empty stderr;
- exit 2: empty stdout and argparse stderr;
- invalid --format, image with scene-to-image, .blend with image-to-3d, and --kind with scene-to-image each exit 2 before creating a run directory;
- exit 3/4: completed report on stdout and primary diagnostic on stderr;
- exit 73: empty stdout and sanitized storage diagnostic on stderr;
- corrupt image, invalid .blend, unreadable file, and unsupported media each use exit 3 with the matching stable diagnostic and a schema-valid failure report;
- no output contains the temporary source basename or absolute path.

- [ ] **Step 5: Verify CLI tests fail**

Run:

    uv run pytest packages/cli/tests/test_cli.py -v

Expected: FAIL because the console entrypoint behavior is absent.

- [ ] **Step 6: Implement argparse and stream mapping**

Keep the console script registered in packages/cli/pyproject.toml:

    [project.scripts]
    asset-mania = "asset_mania.cli:entrypoint"

entrypoint calls main, writes canonical JSON or deterministic text, and returns the documented code. It never prints tracebacks without an explicit debug flag.

- [ ] **Step 7: Run CLI package verification and commit**

Run:

    uv run pytest packages/cli/tests -v
    uv run ruff check packages/cli packages/contracts
    uv run ruff format --check packages/cli packages/contracts
    uv run asset-mania --help
    git add packages/cli pyproject.toml uv.lock
    git commit -m "feat: add deterministic inspect CLI"

Expected: all commands exit 0 before commit.

### Task 5: Reusable Asset Mania Agent Skill

**Files:**
- Create every file under “Agent Skill”.
- Create tests/test_skill_distribution.py.
- Create scripts/validate_skill.py.
- Create tests/test_validate_skill.py.

**Interfaces:**
- Consumes: installed asset-mania CLI and canonical manifest schema.
- Produces: automatically discoverable asset-mania skill that can run local inspection and truthfully refuse generation or external work in v0.1.

- [ ] **Step 1: Read UI metadata guidance and initialize the skill**

Run:

    cat ~/.codex/skills/.system/skill-creator/references/openai_yaml.md
    python3 ~/.codex/skills/.system/skill-creator/scripts/init_skill.py asset-mania --path skills --resources scripts,references --interface display_name="Asset Mania" --interface short_description="Inspect image-to-3D and Blender-to-image workflows safely"

Expected: skills/asset-mania contains SKILL.md and agents/openai.yaml with no examples/placeholders retained after editing.

- [ ] **Step 2: Write failing distribution and validator tests**

The tests execute skills/asset-mania/scripts/inspect.py against a temporary PNG and assert the resulting manifest. They also assert the schema copied into references is byte-identical to the contracts package resource.

Validator tests build temporary skill folders and assert stable findings for missing YAML frontmatter, wrong folder/name pairing, missing name or description, unfinished scaffold markers, missing agents/openai.yaml, and undiscoverable references. A valid skill returns no findings.

- [ ] **Step 3: Verify distribution tests fail**

Run:

    uv run pytest tests/test_skill_distribution.py tests/test_validate_skill.py -v

Expected: FAIL because the launcher, final skill content, and repository-owned validator are absent.

- [ ] **Step 4: Implement the thin skill and launcher**

SKILL.md must:

- trigger on Asset Mania preflight/inspection and planned generation requests;
- run inspect locally when an input is available;
- state that v0.1 cannot generate images or 3D;
- stop without approval or network activity when generation, upload, model download, or paid compute is requested;
- route to cli-contract.md for command details and safety-and-licenses.md for future external-action questions.

The launcher locates asset-mania on PATH, otherwise uses uv run --package asset-mania-cli asset-mania when executed inside the repository, and otherwise exits with installation guidance. It forwards no environment secrets and uses argument-list subprocess execution without a shell.

scripts/validate_skill.py uses only the standard library, checks the structural rules named in Step 2, prints sorted findings, and exits 0 only for a valid skill. It does not replace behavioral forward evaluation.

- [ ] **Step 5: Validate and forward-test the skill**

Run:

    uv run pytest tests/test_skill_distribution.py tests/test_validate_skill.py -v
    uv run python scripts/validate_skill.py skills/asset-mania
    python3 ~/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/asset-mania

Then run independent agent evaluations for all five requests in references/evals.md:

    Use $asset-mania to inspect sample.png for generic image-to-3D readiness.
    Use $asset-mania to preflight sample.png as a face-head asset.
    Use $asset-mania to inspect sample.blend for scene-to-image readiness.
    Use $asset-mania to inspect sample.blend when Blender is not installed.
    Use $asset-mania to generate a paid cloud 3D model from sample.png.

Expected: the first four truthfully report local inspection and capability diagnostics without source mutation. The paid-cloud request may inspect sample.png locally but clearly reports generation unavailable; every evaluation performs no network, upload, download, approval prompt, or paid action.

- [ ] **Step 6: Commit the skill**

Run:

    git add skills scripts/validate_skill.py tests/test_skill_distribution.py tests/test_validate_skill.py
    git commit -m "feat: add asset mania agent skill"

### Task 6: Release Gates, CI, and Documentation Proof

**Files:**
- Create every file under “Release automation”.
- Modify README.md and docs/getting-started.md with output captured from the real CLI.

**Interfaces:**
- Consumes: complete workspace, schema, CLI, skill.
- Produces: check_release(root: Path) -> list[Finding]; CI gates; truthful executable README examples.

- [ ] **Step 1: Write failing release-check tests**

Use temporary trees to prove findings are returned for:

- forbidden .env, token, cookie, weight, cache, and absolute-home strings;
- tracked binary fixture missing from PROVENANCE.md;
- third-party file missing from THIRD_PARTY_NOTICES.md;
- broken relative Markdown link;
- skill schema differing from the contracts schema.

Also prove a minimal clean tree returns an empty list.

- [ ] **Step 2: Verify release-check tests fail**

Run:

    uv run pytest tests/test_check_release.py -v

Expected: FAIL because scripts/check_release.py is absent.

- [ ] **Step 3: Implement the release checker**

Use only the Python standard library. Findings contain stable code, relative path, and message. The command prints sorted findings, returns 1 when any exist, and returns 0 for a clean tree. It never opens ignored run outputs or follows symlinks outside the root.

- [ ] **Step 4: Add CI and GitHub community files**

CI jobs:

- Ubuntu 22.04 with Python 3.11, 3.12, 3.13;
- Ubuntu 24.04 with Python 3.12;
- macOS 14 with Python 3.12.

Each job first runs uv lock --check, then uv sync --locked --all-packages --dev, make check, make test, make skill-check, and make release-check. Configure uv dependency caching by uv.lock hash.

- [ ] **Step 5: Capture a real quickstart example**

Create a temporary 8x6 PNG outside the repository, run:

    uv run asset-mania inspect /tmp/asset-mania-example.png --out /tmp/asset-mania-runs --format text

Copy only the privacy-safe text report into README.md and docs/getting-started.md. Do not copy the image or run directory into Git.

- [ ] **Step 6: Run full verification**

Run:

    make check
    make test
    make skill-check
    make release-check
    asset_build_dir=$(mktemp -d /tmp/asset-mania-dist.XXXXXX)
    uv build --all-packages --out-dir "$asset_build_dir"
    git diff --check

Expected: every command exits 0 with no warning or failure.

- [ ] **Step 7: Commit release automation**

Run:

    git add .github scripts tests README.md docs Makefile pyproject.toml uv.lock
    git commit -m "ci: verify the pre-alpha release"

### Task 7: Independent Review, Merge, and GitHub Publication

**Files:**
- Review all files changed since IMPLEMENTATION_BASE recorded during setup.
- No source change unless review findings require a fix.

**Interfaces:**
- Consumes: all task commits and the approved design/plan.
- Produces: reviewed main branch and public jsc7727/asset-mania repository.

- [ ] **Step 1: Run fresh completion evidence**

Run:

    make check
    make test
    make skill-check
    make release-check
    git status --short

Expected: all gates exit 0 and status shows no unexpected files.

- [ ] **Step 2: Scan Git history and tracked files**

Run the release checker, then check whether gitleaks is already available:

    make release-check
    command -v gitleaks
    gitleaks git --redact --no-banner

If gitleaks is absent, stop and ask for authorization to install a pinned scanner or use an approved ephemeral scanner. Do not infer installation permission and do not publish without a successful history scan.

- [ ] **Step 3: Request whole-branch code review**

Use superpowers:requesting-code-review with:

- Description: guide-first uv monorepo, contracts, inspection CLI, Agent Skill, release gates.
- Requirements: this implementation plan and the approved design spec.
- Base SHA: git merge-base main HEAD, which must equal IMPLEMENTATION_BASE.
- Head SHA: current HEAD.

Fix all Critical and Important findings through reviewed fix rounds, then rerun Step 1.

- [ ] **Step 4: Fast-forward reviewed work to main**

From the primary checkout:

    git switch main
    git merge --ff-only codex/initial-monorepo

Expected: main advances without a merge commit.

- [ ] **Step 5: Create and push the public repository**

Immediately before creation, run:

    gh auth status
    gh api user --jq .login
    gh repo view jsc7727/asset-mania --json nameWithOwner,visibility,url
    git remote -v

Expected: active account jsc7727, the target repository does not yet exist, and no conflicting remote exists. Present the exact owner, name, public visibility, description, and branch to the user and require an affirmative final publication approval. A prior approval remains valid only when all five values are unchanged.

Run:

    gh repo create jsc7727/asset-mania --public --source . --remote origin --push --description "Open-source Agent Skill and reproducible CLI for image-to-3D and Blender-guided image workflows"

Then set topics:

    gh repo edit jsc7727/asset-mania --add-topic agent-skills --add-topic blender --add-topic image-to-3d --add-topic 3d-generation --add-topic python --add-topic reproducible-pipelines

- [ ] **Step 6: Verify the public state**

Run:

    gh repo view jsc7727/asset-mania --json nameWithOwner,visibility,defaultBranchRef,url,description
    git remote -v
    git status --short --branch

Expected: public jsc7727/asset-mania, default branch main, origin points to that repository, and local main tracks origin/main cleanly.
